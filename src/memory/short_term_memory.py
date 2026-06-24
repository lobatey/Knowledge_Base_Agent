from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any
from uuid import uuid4

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    AnyMessage,
    HumanMessage,
    RemoveMessage,
    SystemMessage,
    ToolMessage,
)

# 消息时间只精确到分钟
MESSAGE_TIME_FORMAT = "%Y-%m-%d %H:%M"
MESSAGE_TIME_KEY = "created_at"


@dataclass
class ConversationTurn:
    """
    一个完整问答轮次。

    messages 包含：
    HumanMessage
    中间可能存在的 AI tool call
    ToolMessage
    最终 AIMessage
    """

    messages: list[AnyMessage]
    human_message: HumanMessage
    ai_message: AIMessage
    created_at: datetime


def current_minute_string() -> str:
    """返回分钟级时间，例如 2026-06-19 14:35。"""
    return datetime.now().strftime(MESSAGE_TIME_FORMAT)


def create_timed_human_message(
    content: str,
    original_content: str | None = None,
) -> HumanMessage:
    additional_kwargs = {
        MESSAGE_TIME_KEY: current_minute_string(),
    }

    if original_content is not None:
        additional_kwargs["original_content"] = original_content

    return HumanMessage(
        id=str(uuid4()),
        content=content,
        additional_kwargs=additional_kwargs,
    )


def ensure_ai_message_timestamp(message: AIMessage) -> AIMessage:
    """
    确保 AIMessage 包含时间戳。

    大模型通常会自动生成 message id，但为了保证后续可删除，
    如果不存在 id，这里也主动补上。
    """
    additional_kwargs = dict(message.additional_kwargs or {})

    if MESSAGE_TIME_KEY not in additional_kwargs:
        additional_kwargs[MESSAGE_TIME_KEY] = current_minute_string()

    return message.model_copy(
        update={
            "id": message.id or str(uuid4()),
            "additional_kwargs": additional_kwargs,
        }
    )


def get_message_time(message: AnyMessage) -> datetime | None:
    """
    从消息 additional_kwargs 中读取时间。

    兼容没有时间戳的旧 checkpoint：
    没有时间戳时返回 None，不立即将其判定为过期。
    """
    value = (message.additional_kwargs or {}).get(MESSAGE_TIME_KEY)

    if not value or not isinstance(value, str):
        return None

    try:
        return datetime.strptime(value, MESSAGE_TIME_FORMAT)
    except ValueError:
        return None


def get_message_text(message: AnyMessage) -> str:
    """将 LangChain 消息内容转换为适合摘要的文本。"""
    content = message.content

    if isinstance(content, str):
        return content.strip()

    return str(content).strip()


def is_final_ai_message(message: AnyMessage) -> bool:
    """
    判断是不是一轮问答中的最终 AI 回答。

    带 tool_calls 的 AIMessage 只是请求调用工具，
    不能视为本轮最终回答。
    """
    return isinstance(message, AIMessage) and not message.tool_calls


def split_completed_turns(
    messages: list[AnyMessage],
) -> tuple[list[ConversationTurn], list[AnyMessage]]:
    """
    将 messages 拆成完整问答轮次和未完成消息。

    一个完整轮次从 HumanMessage 开始，到不带 tool_calls 的
    最终 AIMessage 结束。

    当前正在处理但尚未产生最终 AI 回答的消息会进入 incomplete_messages，
    不会被错误删除。
    """
    completed_turns: list[ConversationTurn] = []
    incomplete_messages: list[AnyMessage] = []

    current_turn_messages: list[AnyMessage] = []
    current_human: HumanMessage | None = None

    for message in messages:
        if isinstance(message, HumanMessage):
            # 遇到新的 HumanMessage 时，之前若有未闭合内容，
            # 将其保留为未完成消息。
            if current_turn_messages:
                incomplete_messages.extend(current_turn_messages)

            current_turn_messages = [message]
            current_human = message
            continue

        if current_human is None:
            # 不属于明确问答轮次的历史消息，保留。
            incomplete_messages.append(message)
            continue

        current_turn_messages.append(message)

        if is_final_ai_message(message):
            human_time = get_message_time(current_human)
            ai_time = get_message_time(message)

            # 优先使用用户提问时间，其次使用回答时间。
            # 对没有时间戳的旧数据，暂时使用当前时间，避免升级后立即过期。
            created_at = human_time or ai_time or datetime.now()

            completed_turns.append(
                ConversationTurn(
                    messages=list(current_turn_messages),
                    human_message=current_human,
                    ai_message=message,
                    created_at=created_at,
                )
            )

            current_turn_messages = []
            current_human = None

    if current_turn_messages:
        incomplete_messages.extend(current_turn_messages)

    return completed_turns, incomplete_messages


def select_turns_to_archive(
    completed_turns: list[ConversationTurn],
    max_turns: int,
    expire_days: int,
) -> list[ConversationTurn]:
    """
    选出需要归档到摘要的轮次。

    归档条件：
    1. 问答超过 expire_days；
    2. 删除过期问答后，剩余问答仍超过 max_turns，
       则归档最旧的超额问答。
    """
    if not completed_turns:
        return []

    expire_before = datetime.now() - timedelta(days=expire_days)

    expired_turns = [
        turn
        for turn in completed_turns
        if turn.created_at < expire_before
    ]

    expired_ids = {id(turn) for turn in expired_turns}

    active_turns = [
        turn
        for turn in completed_turns
        if id(turn) not in expired_ids
    ]

    excess_count = max(0, len(active_turns) - max_turns)
    overflow_turns = active_turns[:excess_count]

    archive_turns = expired_turns + overflow_turns
    archive_turns.sort(key=lambda turn: turn.created_at)

    return archive_turns


def format_turns_for_summary(turns: list[ConversationTurn]) -> str:
    """把待归档问答整理成摘要模型输入。"""
    blocks: list[str] = []

    for index, turn in enumerate(turns, start=1):
        time_text = turn.created_at.strftime(MESSAGE_TIME_FORMAT)
        human_text = (
                turn.human_message.additional_kwargs.get("original_content")
                or get_message_text(turn.human_message)
        )

        ai_text = get_message_text(turn.ai_message)

        blocks.append(
            "\n".join(
                [
                    f"轮次 {index}",
                    f"时间：{time_text}",
                    f"用户：{human_text}",
                    f"助手：{ai_text}",
                ]
            )
        )

    return "\n\n".join(blocks)


async def merge_turns_into_summary(
    model: BaseChatModel,
    old_summary: str,
    turns: list[ConversationTurn],
) -> str:
    """
    将旧摘要和本次待归档问答合并成新的滚动摘要。
    """
    if not turns:
        return old_summary

    turns_text = format_turns_for_summary(turns)

    prompt = [
        SystemMessage(
            content=(
                "你是对话短期记忆压缩器。"
                "请把旧摘要和待归档问答合并为一份结构清晰、简洁且可继续使用的会话摘要。\n"
                "必须保留：\n"
                "1. 用户明确提出的目标、约束和偏好；\n"
                "2. 已经确认的重要事实与结论；\n"
                "3. 尚未解决的问题和待办事项；\n"
                "4. 对后续追问有帮助的实体、术语和指代关系；\n"
                "5. 重要时间信息。\n\n"
                "不要保留寒暄、重复表达、工具调用过程和无关细节。"
                "不要编造原对话中不存在的信息。"
                "使用中文输出摘要。"
            )
        ),
        HumanMessage(
            content=(
                f"已有会话摘要：\n"
                f"{old_summary.strip() or '暂无'}\n\n"
                f"本次需要归档的问答：\n"
                f"{turns_text}\n\n"
                "请输出合并后的完整摘要。"
            )
        ),
    ]

    response = await model.ainvoke(prompt)
    summary = get_message_text(response)

    # 极端情况下模型返回空内容时，不覆盖已有摘要。
    return summary or old_summary


def create_remove_messages(
    turns: list[ConversationTurn],
) -> list[RemoveMessage]:
    """
    为待归档轮次中的全部消息生成 RemoveMessage。

    除用户和最终回答外，也会删除该轮对应的工具调用消息，
    避免遗留孤立 ToolMessage。
    """
    removals: list[RemoveMessage] = []

    for turn in turns:
        for message in turn.messages:
            if message.id:
                removals.append(RemoveMessage(id=message.id))

    return removals


def format_short_term_summary_for_prompt(summary: str) -> str:
    """将短期摘要格式化为模型上下文。"""
    if not summary.strip():
        return ""

    return (
        "以下是当前会话中较早问答经过压缩后形成的短期记忆摘要。\n"
        "该摘要仅用于理解上下文、历史目标和指代关系；"
        "涉及知识库事实时仍必须重新检索。\n\n"
        f"会话摘要：\n{summary.strip()}"
    )