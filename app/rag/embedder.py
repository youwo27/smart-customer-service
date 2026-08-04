"""RAG：向量化（占位实现，Day 2 实现）。

职责：将文本转为向量，支持批量 + 单条。
"""

from typing import Any


class Embedder:
    """文本向量化器。

    TODO (Day 2):
    - 单条向量化 embed(text) -> list[float]
    - 批量向量化 embed_batch(texts) -> list[list[float]]
    - 模型通过配置注入（禁止硬编码模型名）
    """

    async def embed(self, text: str) -> list[float]:
        """单条向量化。Day 2 实现。"""
        raise NotImplementedError("Day 2 实现")

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """批量向量化。Day 2 实现。"""
        raise NotImplementedError("Day 2 实现")
