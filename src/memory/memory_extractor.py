import json
import logging
import re
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from src.core.llm import get_model
from src.core.settings import settings
from src.memory.long_term_memory import create_memory


logger = logging.getLogger(__name__)


MEMORY_EXTRACT_SYSTEM_PROMPT = """
你是一个长期记忆抽取器。你的任务是判断用户输入中是否包含值得长期保存的信息。

长期记忆只保存稳定、可复用、对未来对话有帮助的信息。

可以保存的信息包括：
1. 用户身份：例如学生、研究生、医生、开发者、求职者。
2. 用户背景：例如研究方向、专业、项目、工作领域。
3. 用户偏好：例如希望回答更简洁、希望结合面试话术、希望用中文回答。
4. 长期目标：例如准备面试、准备论文、学习某个方向。
5. 当前长期项目背景：例如正在做知识库智能问答 Agent 平台。

不要保存的信息包括：
1. 临时问题本身。
2. 一次性任务。
3. 文档内容事实，例如半月板摘要里的实验指标。
4. 模型推测出来但用户没有明确表达的信息。
5. 敏感隐私信息，例如身份证号、手机号、住址、银行卡、密码、密钥。
6. 医疗诊断、政治倾向、宗教信仰等敏感属性。
7. 用户假设性表达，例如“假设我是研究生”。

你必须只输出 JSON，不要输出任何解释。

JSON 格式如下：
{
  "should_save": true,
  "memories": [
    {
      "memory_type": "profile | preference | project | goal | note",
      "content": "用户是一个研究生",
      "confidence": 0.95
    }
  ]
}

如果没有值得保存的信息，输出：
{
  "should_save": false,
  "memories": []
}
"""


def _extract_json(text: str) -> dict[str, Any]:
    """
    Extract JSON object from LLM output.
    """

    if not text:
        return {"should_save": False, "memories": []}

    text = text.strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", text, re.S)
    if not match:
        return {"should_save": False, "memories": []}

    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return {"should_save": False, "memories": []}


def _clean_memory_content(content: str) -> str:
    """
    Clean extracted memory content.
    """

    content = content.strip()
    content = content.strip("`")
    content = content.replace("\n", " ")
    content = re.sub(r"\s+", " ", content)
    return content


def _is_memory_safe(content: str) -> bool:
    """
    Basic safety filter for long-term memory.
    """

    if not content:
        return False

    if len(content) < 4:
        return False

    if len(content) > 120:
        return False

    blocked_keywords = [
        "密码",
        "密钥",
        "token",
        "api key",
        "apikey",
        "身份证",
        "银行卡",
        "手机号",
        "住址",
        "家庭住址",
        "政治倾向",
        "宗教信仰",
        "诊断为",
    ]

    lowered = content.lower()

    if any(keyword in lowered for keyword in blocked_keywords):
        return False

    return True


async def auto_extract_and_store_memories(
    user_message: str,
    user_id: str,
    model: Any | None = None,
    min_confidence: float = 0.7,
) -> list[dict[str, Any]]:
    """
    Use LLM to automatically extract long-term memories from user input,
    then store valid memories into SQLite.

    This function is intentionally fail-safe:
    memory extraction failure must not break normal chat.
    """

    if not user_message or not user_message.strip():
        return []

    user_message = user_message.strip()
    user_id = user_id.strip() or "default"

    try:
        model_name = model or settings.DEFAULT_MODEL

        if model_name is None:
            return []

        llm = get_model(model_name)

        response = await llm.ainvoke(
            [
                SystemMessage(content=MEMORY_EXTRACT_SYSTEM_PROMPT),
                HumanMessage(
                    content=(
                        "请判断下面用户输入中是否包含需要保存的长期记忆。\n\n"
                        f"用户输入：{user_message}"
                    )
                ),
            ]
        )

        raw_content = getattr(response, "content", "")
        if isinstance(raw_content, list):
            raw_content = "\n".join(str(item) for item in raw_content)

        parsed = _extract_json(str(raw_content))

        if not parsed.get("should_save"):
            return []

        memories = parsed.get("memories", [])
        if not isinstance(memories, list):
            return []

        saved_memories = []

        for memory in memories:
            if not isinstance(memory, dict):
                continue

            memory_type = str(memory.get("memory_type", "note")).strip()
            content = _clean_memory_content(str(memory.get("content", "")))

            try:
                confidence = float(memory.get("confidence", 0))
            except Exception:
                confidence = 0

            if confidence < min_confidence:
                continue

            if memory_type not in {"profile", "preference", "project", "goal", "note"}:
                memory_type = "note"

            if not _is_memory_safe(content):
                continue

            saved = create_memory(
                user_id=user_id,
                memory_type=memory_type,
                content=content,
                source_text=user_message,
            )

            if saved:
                saved_memories.append(saved)

        return saved_memories

    except Exception as exc:
        logger.warning("Auto memory extraction failed: %s", exc)
        return []
