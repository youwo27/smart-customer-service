"""LangGraph 版客服分流链 — Day 6 最简单的多 Agent 骨架（Supervisor Pattern）。

一个"主管"节点 triage 把用户问题分成 闲聊 / 可检索 / 需人工 三类，再路由到三个
"子处理"节点：

    START → triage ─ chat → generate（直接答）
                    ├ knowledge → knowledge（走 RagService.search + generate_answer）
                    └ escalation → escalation（转人工话术 + escalated 标记）→ END

客服要"可控、可审计、可回溯"，所以天然选 Supervisor（中心主管决定下一步），而不是
Swarm（Agent 自由交接）——今天的 escalate 只打标记，Day 12 才升级为真实 HITL 审批。
与 Part 4 约定一致：业务能力（分类 / 检索 / 生成）全复用 app/rag/ 现成函数，这里只画流转。
"""

from typing import Any, Literal, TypedDict

from langgraph.graph import END, START, StateGraph

from app.llm.client import LLMClient
from app.rag.agentic import generate_answer
from app.rag.query import rewrite_query
from app.tools.builtin._shared import RagService

# 决策：主管把问题分到哪一类（只回一个词）
TRIAGE_PROMPT = """你是客服分流主管，把用户问题分到三类之一：
- KNOWLEDGE：退换货 / 退款 / 物流 / 优惠券等政策规则问题，可查知识库回答
- CHAT：寒暄、闲聊、夸赞、情绪表达，直接回应即可
- ESCALATION：高额退款 / 投诉纠纷 / 知识库解决不了 / 用户明确要求人工
只输出一个词：CHAT / KNOWLEDGE / ESCALATION。

用户问题：{question}"""

# 三个叶子节点共同的身份：不做 Agent，只是"把某类问题处理完"
Category = Literal["chat", "knowledge", "escalation"]


class TriageState(TypedDict, total=False):
    question: str
    category: Category            # triage 节点的产物，供条件边路由
    chunks: list[dict[str, Any]]  # knowledge 叶子检到的片段（留给审计 / 后续自检）
    answer: str
    escalated: bool               # escalation 叶子置 True（Day 12 从这里接 HITL）


_CATEGORY_MAP: dict[str, Category] = {
    "CHAT": "chat",
    "KNOWLEDGE": "knowledge",
    "ESCALATION": "escalation",
}


def _normalize_category(content: str) -> Category:
    """LLM 返回可能带解释，宽松匹配三类词；无法识别时宁转人工（更安全）。"""
    token = content.strip().upper()
    for key, value in _CATEGORY_MAP.items():
        if key in token:
            return value
    return "escalation"


def build_triage_graph(
    llm: LLMClient,
    rag_service: RagService,
    *,
    search_top_k: int = 5,
    checkpointer: Any | None = None,
) -> Any:
    """组装客服分流链（Supervisor 骨架），返回 compiled app。

    llm / rag_service 与 Part 4 同源：分类走这里的主管提示词，可检索类走
    RagService.search()（混合检索 + Rerank）再 generate_answer() 回答。
    """

    async def triage(state: TriageState) -> dict[str, Any]:
        resp = await llm.chat(
            [{"role": "user", "content": TRIAGE_PROMPT.format(question=state["question"])}]
        )
        return {"category": _normalize_category(resp.content)}

    async def generate(state: TriageState) -> dict[str, Any]:
        # CHAT：闲聊直接答，不检索（同 Part 4 classify 判"无需检索"后的分支）
        resp = await llm.chat([{"role": "user", "content": state["question"]}])
        return {"answer": resp.content}

    async def knowledge(state: TriageState) -> dict[str, Any]:
        # KNOWLEDGE：改写 → 检索 → 生成，复用一个最小 RAG 子处理链
        query = await rewrite_query(llm, state["question"])
        chunks = await rag_service.search(query, top_k=search_top_k)
        answer = await generate_answer(llm, state["question"], chunks)
        return {"chunks": chunks, "answer": answer}

    async def escalation(state: TriageState) -> dict[str, Any]:
        # ESCALATION：只生成转人工话术并打标记；真实人工审批/交接留到 Day 12
        answer = (
            f"收到，你的问题「{state['question']}」需要人工客服进一步处理，"
            "已为你转接人工，请稍候，客服会尽快联系你。"
        )
        return {"escalated": True, "answer": answer}

    def route(state: TriageState) -> Category:
        return state.get("category", "escalation")

    graph = StateGraph(TriageState)
    graph.add_node("triage", triage)
    graph.add_node("generate", generate)
    graph.add_node("knowledge", knowledge)
    graph.add_node("escalation", escalation)
    graph.add_edge(START, "triage")
    graph.add_conditional_edges(
        "triage",
        route,
        {"chat": "generate", "knowledge": "knowledge", "escalation": "escalation"},
    )
    graph.add_edge("generate", END)
    graph.add_edge("knowledge", END)
    graph.add_edge("escalation", END)
    return graph.compile(checkpointer=checkpointer)
