"""LangGraph 版 Agentic RAG 单元测试 — mock LLM / mock 检索，不碰真实 DeepSeek。

覆盖 Day 6 Part 4 的三条路由 + Part 5 的 checkpointer：
  1. 闲聊（classify 判"不需要检索"）→ 直接答，一次检索都不发生
  2. 首轮检索充分（grade=ENOUGH）→ 单轮直接生成
  3. 首轮不够（NOT_ENOUGH）→ refine 换角度再检 → chunks 去重合并 → 生成
  4. 换角度达到轮次上限仍不够 → 强制收尾（防死循环，对应 AgentLoop 的 max_turns）
  5. MemorySaver 挂载后 get_state 能读任意步的快照（断点续跑的前提）
"""

from typing import Any

from langgraph.checkpoint.memory import MemorySaver

from app.graph.agentic_rag_graph import build_agentic_rag_graph
from app.llm.client import LLMResponse

NEED_RETRIEVAL_MARK = "判断用户问题是否需要检索"
IS_SUFFICIENT_MARK = "判断检索结果是否足以回答"
REWRITE_MARK = "搜索查询优化器"
REFINE_HINT = "初检无相关内容"


def _resp(content: str) -> LLMResponse:
    return LLMResponse(content=content, model="mock", usage={"input_tokens": 0, "output_tokens": 0})


def _chunk(doc_id: str, text: str, rrf: float = 1.0) -> dict[str, Any]:
    return {"id": doc_id, "text": text, "rrf_score": rrf}


class FakeLLM:
    """按决策阶段返回预设结果的假 LLM（靠提示词特征识别当前阶段）。"""

    def __init__(self) -> None:
        self.calls: list[list[dict[str, Any]]] = []
        self.need_retrieval = True
        self.sufficient_answers: list[bool] = []  # 按 is_sufficient 调用顺序消费
        self.rewrite_hint_calls = 0  # 统计 rewrite 收到"换角度"提示的次数

    async def chat(self, messages: list[dict[str, Any]]) -> LLMResponse:
        self.calls.append(list(messages))
        first = messages[0]["content"]
        if NEED_RETRIEVAL_MARK in first:
            return _resp("YES" if self.need_retrieval else "NO")
        if IS_SUFFICIENT_MARK in first:
            return _resp("ENOUGH" if self.sufficient_answers.pop(0) else "NOT_ENOUGH")
        if REWRITE_MARK in first:
            if REFINE_HINT in first:
                self.rewrite_hint_calls += 1
            return _resp("改写后的检索词")
        if len(messages) == 2 and "参考内容：" in messages[0]["content"]:
            return _resp("已根据知识库片段回答")
        # 兜底：无需检索的闲聊，直接答
        return _resp("我是客服，帮你处理退换货、订单、物流、退款等问题")


class FakeSearch:
    """按调用次序返回脚本结果的假检索服务（混合检索 + Rerank 的替身）。"""

    def __init__(self, results: list[list[dict[str, Any]]]) -> None:
        self._results = results
        self.calls = 0
        self.queries: list[str] = []

    async def search(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        self.calls += 1
        self.queries.append(query)
        if self.calls > len(self._results):
            return []
        return list(self._results[self.calls - 1])


class TestAgenticRagGraph:
    async def test_chitchat_skips_retrieval(self) -> None:
        """闲聊：classify 判不需要检索 → 直接答，一次检索都不发生。"""
        llm = FakeLLM()
        llm.need_retrieval = False
        search = FakeSearch([[_chunk("a-0", "退货政策")]])
        app = build_agentic_rag_graph(llm, search)  # type: ignore[arg-type]

        out = await app.ainvoke({"question": "你好"})

        assert out["need_retrieval"] is False
        assert "客服" in out["answer"]
        assert search.calls == 0
        assert out.get("chunks", []) == []

    async def test_single_round_sufficient_answer(self) -> None:
        """检索一次就够：grade=ENOUGH → 单轮直接生成答案。"""
        llm = FakeLLM()
        llm.sufficient_answers = [True]
        chunks = [_chunk("a-0", "退货政策内容")]
        search = FakeSearch([chunks])
        app = build_agentic_rag_graph(llm, search)  # type: ignore[arg-type]

        out = await app.ainvoke({"question": "退货政策是什么？"})

        assert search.calls == 1
        assert out["chunks"] == chunks
        assert out.get("refined_rounds", 0) == 0
        assert "知识库" in out["answer"]

    async def test_insufficient_triggers_refine_and_dedups(self) -> None:
        """首轮不够 → refine 换角度再检 → 两轮 chunks 去重合并后生成。"""
        llm = FakeLLM()
        llm.sufficient_answers = [False, True]
        first = [_chunk("a-0", "初检片段", rrf=2.0)]
        second = [_chunk("a-0", "初检片段", rrf=2.0), _chunk("b-0", "补充片段", rrf=1.0)]
        search = FakeSearch([first, second])
        app = build_agentic_rag_graph(llm, search)  # type: ignore[arg-type]

        out = await app.ainvoke({"question": "退款多久到账？"})

        assert search.calls == 2
        assert out["refined_rounds"] == 1
        assert llm.rewrite_hint_calls == 1  # refine 的改写确实带了"换角度"提示
        assert [c["id"] for c in out["chunks"]] == ["a-0", "b-0"]  # a-0 只出现一次

    async def test_refine_capped_at_max_rounds(self) -> None:
        """轮次上限：refine 已到 max_refined_rounds 仍不够 → 强制生成（防死循环）。"""
        llm = FakeLLM()
        llm.sufficient_answers = [False, False]
        search = FakeSearch([
            [_chunk("a-0", "第一次结果")],
            [_chunk("b-0", "第二次结果")],
        ])
        app = build_agentic_rag_graph(llm, search, max_refined_rounds=1)  # type: ignore[arg-type]

        out = await app.ainvoke({"question": "优惠券能叠加吗？"})

        assert out["refined_rounds"] == 1  # 触顶后不再循环
        assert search.calls == 2
        assert "知识库" in out["answer"]

    async def test_checkpointer_snapshots_state(self) -> None:
        """挂 MemorySaver 后 get_state 能读完整中间态（chunks/answer），Part 5 断点续跑的地基。"""
        llm = FakeLLM()
        llm.sufficient_answers = [True]
        search = FakeSearch([[_chunk("a-0", "物流规则")]])
        app = build_agentic_rag_graph(llm, search, checkpointer=MemorySaver())  # type: ignore[arg-type]
        config = {"configurable": {"thread_id": "session-001"}}

        await app.ainvoke({"question": "物流一般几天到？"}, config)
        snap = await app.aget_state(config)

        assert snap.values.get("chunks")  # 检索片段进了快照
        assert snap.values.get("answer")
