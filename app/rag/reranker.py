"""RAG：重排序器（占位实现，Day 3 实现）。

职责：对检索结果用 Cross-Encoder 重新打分排序。
"""

from typing import Any


class Reranker:
    """检索结果重排序器。

    TODO (Day 3):
    - rerank(query, candidates, top_k) -> list[dict]
    - 用 Cross-Encoder 模型（如 bge-reranker-base）
    - 模型名通过配置注入
    """

    async def rerank(
        self, query: str, candidates: list[dict[str, Any]], top_k: int = 5
    ) -> list[dict[str, Any]]:
        """对候选结果重排序。Day 3 实现。"""
        raise NotImplementedError("Day 3 实现")
