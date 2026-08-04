"""RAG：检索器（占位实现，Day 2-3 实现）。

职责：检索相关文档片段。
策略：Dense（向量相似度）+ Sparse（BM25）+ RRF 融合 + Rerank。
"""

from typing import Any


class Retriever:
    """文档检索器。

    TODO (Day 2): Dense 检索
    TODO (Day 3): 混合检索（Dense + Sparse + RRF）+ Rerank
    """

    async def search(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        """检索 Top-K 相关文档。Day 2 实现。"""
        raise NotImplementedError("Day 2 实现")

    async def hybrid_search(self, query: str, top_k: int = 10) -> list[dict[str, Any]]:
        """混合检索（Dense + Sparse + RRF）。Day 3 实现。"""
        raise NotImplementedError("Day 3 实现")
