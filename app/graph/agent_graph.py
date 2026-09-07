from typing import Annotated, Any, Literal, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode


class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]  # reducer：新消息自动 append


def build_agent_graph(llm: Any, tools: list[Any]) -> Any:
    """组装 LangGraph Agent。llm / tools 复用 Day 5 的 build_llm() + to_langchain_tools()。"""
    llm_with_tools = llm.bind_tools(tools)          # 把工具 schema 绑到 LLM（Day 5 已讲过 bind_tools）
    tool_node = ToolNode(tools)                     # 框架替你执行工具 + 回填 ToolMessage

    def call_model(state: AgentState) -> dict[str, list[BaseMessage]]:
        # 只返回"新增的一条消息"，add_messages reducer 负责追加，不用手动 append
        return {"messages": [llm_with_tools.invoke(state["messages"])]}

    def should_continue(state: AgentState) -> Literal["tools", "end"]:
        last = state["messages"][-1]
        # 有 tool_calls → 回 tools 节点执行；没有 → 收尾。正是 Day 4 的终止条件
        return "tools" if getattr(last, "tool_calls", None) else "end"

    graph = StateGraph(AgentState)
    graph.add_node("agent", call_model)
    graph.add_node("tools", tool_node)
    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", should_continue, {"tools": "tools", "end": END})
    graph.add_edge("tools", "agent")               # 工具执行完 → 回 agent 继续推理（循环）
    return graph.compile()