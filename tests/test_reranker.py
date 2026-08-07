"""Reranker 的单元测试。

覆盖两条路径：
  1. 未配置 provider（""）→ 原序返回兜底，不联网
  2. 配置 siliconflow → mock httpx 返回 relevance_score，验证排序 + 打标

用 httpx.MockTransport 拦截请求，测试不联网。
"""

from typing import Any

import httpx

from app.rag.reranker import Reranker


def _candidates() -> list[dict[str, Any]]:
    """造 3 个候选，按"原序"即 A、B、C。"""
    return [
        {"id": "doc-a", "text": "无理由退货：退货运费由买家承担"},
        {"id": "doc-b", "text": "质量问题退货：退货运费由商家承担"},
        {"id": "doc-c", "text": "促销活动规则：满 300 减 50"},
    ]


class TestRerankerFallback:
    """未配置 provider：原序返回，不联网。"""

    async def test_no_provider_returns_in_order(self) -> None:
        """无 provider 时 Top-K 按原序返回。"""
        r = Reranker(provider="")
        result = await r.rerank("退货运费谁出", _candidates(), top_k=2)
        assert [c["id"] for c in result] == ["doc-a", "doc-b"]
        assert "rerank_score" not in result[0]  # 兜底不打分

    async def test_no_provider_respects_top_k(self) -> None:
        r = Reranker(provider="")
        result = await r.rerank("退货运费谁出", _candidates(), top_k=5)
        assert len(result) == 3  # 候选就 3 个，不超过


class TestRerankerAPI:
    """配置 siliconflow：mock API，验证排序和打分。"""

    def _reranker_with_mock(self, results: list[dict[str, Any]]) -> Reranker:
        def handler(request: httpx.Request) -> httpx.Response:
            assert "/rerank" in str(request.url)  # 打到硅基流动 rerank 端点
            return httpx.Response(200, json={"results": results})

        transport = httpx.MockTransport(handler)
        return Reranker(
            provider="siliconflow",
            api_key="test-key",
            client=httpx.AsyncClient(transport=transport),
        )

    async def test_rerank_sorts_by_score(self) -> None:
        """API 返回的顺序不重要，按 relevance_score 重排。"""
        r = self._reranker_with_mock(
            [
                {"index": 1, "relevance_score": 0.95},  # doc-b 最高分
                {"index": 0, "relevance_score": 0.50},  # doc-a
                {"index": 2, "relevance_score": 0.10},  # doc-c 最低
            ]
        )
        result = await r.rerank("退货运费谁出", _candidates(), top_k=3)
        assert [c["id"] for c in result] == ["doc-b", "doc-a", "doc-c"]
        assert result[0]["rerank_score"] == 0.95  # 打上了分数

    async def test_rerank_top_k_limits(self) -> None:
        """top_k 限制返回数量，但按分数取最高的那几个。"""
        r = self._reranker_with_mock(
            [
                {"index": 2, "relevance_score": 0.99},  # doc-c 最高
                {"index": 0, "relevance_score": 0.60},
                {"index": 1, "relevance_score": 0.30},
            ]
        )
        result = await r.rerank("退货运费谁出", _candidates(), top_k=1)
        assert [c["id"] for c in result] == ["doc-c"]
        assert result[0]["rerank_score"] == 0.99
