"""客服分流链（triage）单元测试 — mock LLM / mock 检索，不碰真实 API。

覆盖 Day 6 Part 6 主管分流的三条叶子路由：
  1. CHAT → generate：直接答，一次检索都不发生
  2. KNOWLEDGE → knowledge：改写 + 检索 + 知识库生成
  3. ESCALATION → escalation：转人工话术 + escalated 标记
  4. LLM 返回无法识别的词 → 兜底转人工（宁安全，不错放）
"""

from typing import Any

from app.graph.triage_graph import build_triage_graph
from app.llm.client import LLMResponse

TRIAGE_MARK = "客服分流主管"
REWRITE_MARK = "搜索查询优化器"


def _resp(content: str) -> LLMResponse:
    return LLMResponse(content=content, model="mock", usage={"input_tokens": 0, "output_tokens": 0})


def _chunk(doc_id: str, text: str) -> dict[str, Any]:
    return {"id": doc_id, "text": text, "rrf_score": 1.0}


class FakeLLM:
    """按调用阶段返回预设结果的假 LLM（triage 返回配置好的分类，其余按提示词特征）。"""

    def __init__(self) -> None:
        self.calls: list[list[dict[str, Any]]] = []
        self.category = "knowledge"

    async def chat(self, messages: list[dict[str, Any]]) -> LLMResponse:
        self.calls.append(list(messages))
        first = messages[0]["content"]
        if TRIAGE_MARK in first:
            return _resp(self.category.upper())
        if REWRITE_MARK in first:
            return _resp("改写后的检索词")
        if len(messages) == 2 and "参考内容" in messages[0]["content"]:
            return _resp("已根据知识库片段回答")
        # 兜底：闲聊直接答
        return _resp("我是客服，帮你处理退换货、订单、物流、退款等问题")


class FakeSearch:
    """假检索服务：记录调用，返回脚本结果。"""

    def __init__(self, results: list[list[dict[str, Any]]]) -> None:
        self._results = results
        self.calls = 0

    async def search(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        self.calls += 1
        if self.calls > len(self._results):
            return []
        return list(self._results[self.calls - 1])


class TestTriageGraph:
    async def test_chat_route_direct_answer(self) -> None:
        """CHAT：triage 归闲聊 → generate 直接答，不检索、不标记人工。"""
        llm = FakeLLM()
        llm.category = "chat"
        search = FakeSearch([[_chunk("a-0", "退货政策")]])
        app = build_triage_graph(llm, search)  # type: ignore[arg-type]

        out = await app.ainvoke({"question": "你好"})

        assert out["category"] == "chat"
        assert "客服" in out["answer"]
        assert search.calls == 0
        assert out.get("escalated") is None

    async def test_knowledge_route_searches_and_answers(self) -> None:
        """KNOWLEDGE：triage 归可检索 → knowledge 检索 + 知识库回答。"""
        llm = FakeLLM()
        llm.category = "knowledge"
        chunks = [_chunk("a-0", "退款规则内容")]
        search = FakeSearch([chunks])
        app = build_triage_graph(llm, search)  # type: ignore[arg-type]

        out = await app.ainvoke({"question": "退款多久到账？"})

        assert out["category"] == "knowledge"
        assert search.calls == 1
        assert out["chunks"] == chunks
        assert "知识库" in out["answer"]
        assert out.get("escalated") is None

    async def test_escalation_route_marks_handoff(self) -> None:
        """ESCALATION：生成转人工话术并打 escalated 标记，不检索。"""
        llm = FakeLLM()
        llm.category = "escalation"
        search = FakeSearch([[_chunk("a-0", "无关")]])
        app = build_triage_graph(llm, search)  # type: ignore[arg-type]

        out = await app.ainvoke({"question": "我要投诉，叫你们经理来"})

        assert out["category"] == "escalation"
        assert out["escalated"] is True
        assert "转接人工" in out["answer"]
        assert search.calls == 0

    async def test_unrecognized_category_falls_back_to_escalation(self) -> None:
        """LLM 返回垃圾词：宁转人工，不让问题溜进错误分支。"""
        llm = FakeLLM()
        llm.category = "banana"
        search = FakeSearch([])
        app = build_triage_graph(llm, search)  # type: ignore[arg-type]

        out = await app.ainvoke({"question": "退款多久到账？"})

        assert out["category"] == "escalation"
        assert out["escalated"] is True
