from datetime import datetime
from typing import Literal

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, RemoveMessage, SystemMessage
from langchain_core.runnables import (
    RunnableConfig,
    RunnableLambda,
    RunnableSerializable,
)
from src.memory.short_term_memory import (
    create_remove_messages,
    ensure_ai_message_timestamp,
    format_short_term_summary_for_prompt,
    merge_turns_into_summary,
    select_turns_to_archive,
    split_completed_turns,
)
from langgraph.graph import END, MessagesState, StateGraph
from langgraph.managed import RemainingSteps
from langgraph.prebuilt import ToolNode

from src.agents.safeguard import Safeguard, SafeguardOutput, SafetyAssessment
from src.agents.tools import database_search
from src.core import get_model, settings


class AgentState(MessagesState, total=False):
    safety: SafeguardOutput
    remaining_steps: RemainingSteps

    # 较早问答压缩后的滚动摘要。
    # 该字段会和 messages 一样存入 LangGraph checkpoint。
    conversation_summary: str


tools = [database_search]


current_date = datetime.now().strftime("%B %d, %Y")
instructions = f"""
你是一个知识库智能问答 Agent，负责基于知识库内容回答用户问题。
今天日期是{current_date}。

多轮追问规则：
1. 如果用户的问题包含“它”“这个”“该方法”“上述”“刚才”“继续”“详细说”等指代或省略表达，你必须结合最近对话理解用户真实问题。
2. 最近对话只能用于理解问题含义，不能直接作为事实依据。
3. 即使是多轮追问，也必须重新调用 Database_Search 工具检索知识库。
4. 如果上一轮限定了 source，当前追问默认继承该 source，除非用户明确切换文档或要求全库检索。
5. 如果当前问题是“继续说”“展开讲讲”“详细说说”，你应该围绕上一轮主题重新检索并补充说明，而不是凭记忆续写。
6. 如果根据历史对话仍然无法判断用户指代对象，应先说明需要用户明确问题，而不是编造答案。

长期记忆规则：
1. 如果用户输入中包含“用户长期记忆”，你可以使用它理解用户偏好、个人背景、项目背景和回答风格要求。
2. 长期记忆不能替代知识库证据；涉及知识库事实、文档内容、指标、制度、技术细节时，仍然必须调用 Database_Search 工具检索。
3. 如果长期记忆与知识库检索结果冲突，以知识库检索结果为准。
4. 不要在回答中泄露“系统注入了长期记忆”这类内部实现细节。
5. 长期记忆主要用于个性化表达和上下文理解，不应用来编造知识库中没有的事实。

重要规则：
1. 当用户询问制度、产品文档、技术文档、操作流程、内部资料等内容时，必须优先调用 Database_Search 工具检索知识库。
1.1 如果用户明确要求只基于某个文档、某个 source、某个文件名回答，调用 Database_Search 工具时必须传入 source 参数。
1.2 如果用户问题中包含“只看某个文档”“限定某个文档”“source=xxx”“文件名为 xxx”等约束，不能全库检索，必须按 source 过滤。
1.3 如果指定文档中没有找到依据，应回答“该文档中没有找到足够依据”，不要用其他文档补充。

防幻觉规则：
1. 你是知识库问答 Agent，不是自由发挥型聊天助手。
2. 对于知识库相关问题，必须先调用 Database_Search 工具获取依据。
3. 如果 Database_Search 返回“检索状态：NO_EVIDENCE”，必须拒绝回答具体内容，只能回答：
   “知识库中没有找到足够依据，无法回答该问题。”
4. 如果用户指定了某个文档，而 Database_Search 返回“检索状态：NO_EVIDENCE”，必须回答：
   “该文档中没有找到足够依据，无法回答该问题。”
5. 如果 Database_Search 返回“检索状态：HAS_EVIDENCE”，最终回答只能基于工具返回的内容，不允许加入工具结果之外的事实、数字、结论或建议。
6. 如果工具结果中没有出现某个事实、数字、日期、名称、结论，你不能把它写进最终回答。
7. 不允许使用“根据常识”“一般来说”“我认为”等方式补充知识库外的信息。
8. 不允许为了让答案看起来完整而编造引用来源。

引用规则：
1. 回答知识库问题时，必须在关键结论后标注引用编号，例如：[1]、[2]。
2. 回答末尾必须包含“引用来源”部分，列出用到的文档来源。
3. 引用来源格式必须为：
   [编号] 文档名 ｜ page: 页码 ｜ chunk_id: Chunk ID
4. 只能引用 Database_Search 工具返回的引用编号，不能自己编造引用。
5. 如果知识库检索结果不足以回答问题，应明确说明“知识库中没有找到足够依据”，不要强行给出引用。

回答格式：
1. 如果有知识库依据：
   先直接回答用户问题，并在关键结论后添加引用编号。
   末尾添加“引用来源”。
2. 如果没有知识库依据：
   只回答“知识库中没有找到足够依据，无法回答该问题。”
"""


def build_model_messages(state: AgentState):
    """
    模型实际读取的上下文：

    系统提示词
    + 较早问答摘要
    + 最近最多 20 轮消息
    + 当前问题
    """
    model_messages = [SystemMessage(content=instructions)]

    summary = state.get("conversation_summary", "")
    summary_context = format_short_term_summary_for_prompt(summary)

    if summary_context:
        model_messages.append(SystemMessage(content=summary_context))

    model_messages.extend(state["messages"])
    return model_messages


def wrap_model(model: BaseChatModel) -> RunnableSerializable[AgentState, AIMessage]:
    bound_model = model.bind_tools(tools)

    preprocessor = RunnableLambda(
        build_model_messages,
        name="StateModifier",
    )

    return preprocessor | bound_model  # type: ignore[return-value]


def format_safety_message(safety: SafeguardOutput) -> AIMessage:
    content = (
        f"This conversation was flagged for unsafe content: {', '.join(safety.unsafe_categories)}"
    )
    return AIMessage(content=content)


async def acall_model(state: AgentState, config: RunnableConfig) -> AgentState:
    m = get_model(config["configurable"].get("model", settings.DEFAULT_MODEL))
    model_runnable = wrap_model(m)
    response = await model_runnable.ainvoke(state, config)

    if state["remaining_steps"] < 2 and response.tool_calls:
        return {
            "messages": [
                AIMessage(
                    id=response.id,
                    content="Sorry, need more steps to process this request.",
                )
            ]
        }
    # We return a list, because this will get added to the existing list
    response = ensure_ai_message_timestamp(response)
    return {"messages": [response]}

async def compact_short_term_memory(
    state: AgentState,
    config: RunnableConfig,
) -> AgentState:
    """
    整理当前 thread_id 对应的短期记忆。

    规则：
    1. 超过 30 天的完整问答合并进摘要并删除；
    2. 删除过期问答后，如果仍超过 20 轮，
       将最旧的超额问答合并进摘要并删除；
    3. 保留最近最多 20 轮有效问答；
    4. 当前未完成的一轮不会被删除。
    """
    messages = list(state.get("messages", []))

    completed_turns, _ = split_completed_turns(messages)

    turns_to_archive = select_turns_to_archive(
        completed_turns=completed_turns,
        max_turns=settings.SHORT_TERM_MEMORY_MAX_TURNS,
        expire_days=settings.SHORT_TERM_MEMORY_EXPIRE_DAYS,
    )

    if not turns_to_archive:
        return {}

    model = get_model(
        config["configurable"].get("model", settings.DEFAULT_MODEL)
    )

    old_summary = state.get("conversation_summary", "")

    new_summary = await merge_turns_into_summary(
        model=model,
        old_summary=old_summary,
        turns=turns_to_archive,
    )

    removals: list[RemoveMessage] = create_remove_messages(
        turns_to_archive
    )

    return {
        "conversation_summary": new_summary,
        "messages": removals,
    }

async def safeguard_input(state: AgentState, config: RunnableConfig) -> AgentState:
    safeguard = Safeguard()
    safety_output = await safeguard.ainvoke(state["messages"])
    return {"safety": safety_output, "messages": []}


async def block_unsafe_content(state: AgentState, config: RunnableConfig) -> AgentState:
    safety: SafeguardOutput = state["safety"]
    return {"messages": [format_safety_message(safety)]}


# Define the graph
agent = StateGraph(AgentState)
agent.add_node("model", acall_model)
agent.add_node("tools", ToolNode(tools))
agent.add_node("guard_input", safeguard_input)
agent.add_node("block_unsafe_content", block_unsafe_content)
agent.add_node("compact_memory", compact_short_term_memory)
agent.set_entry_point("guard_input")


# Check for unsafe input and block further processing if found
def check_safety(state: AgentState) -> Literal["unsafe", "safe"]:
    safety: SafeguardOutput = state["safety"]
    match safety.safety_assessment:
        case SafetyAssessment.UNSAFE:
            return "unsafe"
        case _:
            return "safe"


agent.add_conditional_edges(
    "guard_input", check_safety, {"unsafe": "block_unsafe_content", "safe": "model"}
)

# Always END after blocking unsafe content
agent.add_edge("block_unsafe_content", "compact_memory")

# Always run "model" after "tools"
agent.add_edge("tools", "model")


# After "model", if there are tool calls, run "tools". Otherwise END.
def pending_tool_calls(state: AgentState) -> Literal["tools", "done"]:
    last_message = state["messages"][-1]
    if not isinstance(last_message, AIMessage):
        raise TypeError(f"Expected AIMessage, got {type(last_message)}")
    if last_message.tool_calls:
        return "tools"
    return "done"


agent.add_conditional_edges(
    "model",
    pending_tool_calls,
    {
        "tools": "tools",
        "done": "compact_memory",
    },
)

agent.add_edge("compact_memory", END)

rag_assistant = agent.compile()
