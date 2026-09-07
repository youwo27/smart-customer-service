"""LangGraph 版 Agentic RAG — 把 Day 3 的 if/else 顺序调用改成显式状态机。

三个决策点（app/rag/agentic.py 里的 ①②③）→ 图里的条件边：
  ① need_retrieval → classify 节点 → 条件边：闲聊 → generate / 需检索 → rewrite
  ② is_sufficient  → grade 节点   → 条件边：充分 → generate / 不够 → refine
  ③ refine（换角度）→ refine 节点 → refined_rounds 轮次上限（对应 AgentLoop 的 max_turns）

与 Day 3 agentic_rag() 唯一差异：业务逻辑全复用（_need_retrieval / _is_sufficient /
rewrite_query / generate_answer 原样导入），流转规则从 if/else 变成图；chunks 不再
是函数里手动合并的局部变量，而是共享状态字段，靠 reducer 跨轮累积（等价 AgentState
的 add_messages 之于消息）。
"""

from typing import Annotated, Any, Literal, TypedDict

from langgraph.graph import END, START, StateGraph

from app.llm.client import LLMClient
from app.rag.agentic import _is_sufficient, _merge_dedup, _need_retrieval, generate_answer
from app.rag.query import rewrite_query
from app.tools.builtin._shared import RagService


class AgenticRagState(TypedDict, total=False):
    question: str
    need_retrieval: bool
    search_query: str                # rewrite 后的检索词
    chunks: Annotated[list[dict[str, Any]], _merge_dedup]  # 检索结果，跨轮累积（状态流转的核心）
    refined_rounds: int              # 换角度改写次数（防死循环，= AgentLoop 的 max_turns）
    sufficient: bool                 # grade 节点的产物，仅供条件边判断，不进最终答案
    answer: str


def _chunks(state: AgenticRagState) -> list[dict[str, Any]]:
    """累计到当前的检索结果（可能一次检索都还没发生过 → 空）。"""
    return state.get("chunks", [])


def _rounds(state: AgenticRagState) -> int:
    return state.get("refined_rounds", 0)


def build_agentic_rag_graph(
    llm: LLMClient,
    rag_service: RagService,
    *,
    max_refined_rounds: int = 1,
    search_top_k: int = 5,
    checkpointer: Any | None = None,
) -> Any:
    """组装 LangGraph 版 Agentic RAG 状态机，返回 compiled app。

    llm / rag_service 复用 Day 3 的能力：决策 + 生成走 app/rag/agentic.py，
    检索走 RagService.search()（混合检索 + Rerank）。
    传 checkpointer=MemorySaver() 即可用 thread_id 做断点续跑 / get_state 读快照。
    """

    async def classify(state: AgenticRagState) -> dict[str, Any]:
        return {"need_retrieval": await _need_retrieval(llm, state["question"])}

    async def rewrite(state: AgenticRagState) -> dict[str, Any]:
        return {"search_query": await rewrite_query(llm, state["question"])}

    async def search(state: AgenticRagState) -> dict[str, Any]:
        # 只返回"本轮新检出的 chunks"，reducer 负责与已有结果合并去重
        return {"chunks": await rag_service.search(state["search_query"], top_k=search_top_k)}

    async def grade(state: AgenticRagState) -> dict[str, Any]:
        return {"sufficient": await _is_sufficient(llm, state["question"], _chunks(state))}

    async def refine(state: AgenticRagState) -> dict[str, Any]:
        # 换角度改写 + 轮次 +1，回到 search 用新检索词再来一轮
        rounds = _rounds(state) + 1
        refined = await rewrite_query(
            llm, f"{state['question']}\n（初检无相关内容，请换角度改写检索词）"
        )
        return {"refined_rounds": rounds, "search_query": refined}

    async def generate(state: AgenticRagState) -> dict[str, Any]:
        if not state.get("need_retrieval"):
            # 决策① 判定为闲聊 → 直接答（对应 Day 3 agentic_rag 的 no-retrieval 分支）
            resp = await llm.chat([{"role": "user", "content": state["question"]}])
            return {"answer": resp.content}
        answer = await generate_answer(llm, state["question"], _chunks(state))
        return {"answer": answer}

    def route_classify(state: AgenticRagState) -> Literal["rewrite", "generate"]:
        return "rewrite" if state.get("need_retrieval") else "generate"

    def route_grade(state: AgenticRagState) -> Literal["refine", "generate"]:
        # 检索够用，或换角度已达上限 → 收尾；否则回去 refine 再来一轮
        if state.get("sufficient") or _rounds(state) >= max_refined_rounds:
            return "generate"
        return "refine"

    graph = StateGraph(AgenticRagState)
    graph.add_node("classify", classify)
    graph.add_node("rewrite", rewrite)
    graph.add_node("search", search)
    graph.add_node("grade", grade)
    graph.add_node("refine", refine)
    graph.add_node("generate", generate)
    graph.add_edge(START, "classify")
    graph.add_conditional_edges(
        "classify", route_classify, {"rewrite": "rewrite", "generate": "generate"}
    )
    graph.add_edge("rewrite", "search")
    graph.add_edge("search", "grade")
    graph.add_conditional_edges(
        "grade", route_grade, {"refine": "refine", "generate": "generate"}
    )
    graph.add_edge("refine", "search")
    graph.add_edge("generate", END)
    return graph.compile(checkpointer=checkpointer)
