"""RAG：向量存储封装（占位实现，Day 2 实现）。

职责：封装向量数据库（ChromaDB/Qdrant），提供 CRUD + 相似度检索。
"""

from typing import Any


class VectorStore:
    """向量数据库封装。

    TODO (Day 2):
    - add(ids, embeddings, documents, metadatas)
    - query(query_embedding, top_k) -> list[dict]
    - delete(ids)
    - 持久化目录通过配置注入
    """

    async def add(self, ids: list[str], embeddings: list[list[float]], documents: list[str]) -> None:
        """批量写入。Day 2 实现。"""
        raise NotImplementedError("Day 2 实现")

    async def query(self, query_embedding: list[float], top_k: int = 5) -> list[dict[str, Any]]:
        """相似度检索。Day 2 实现。"""
        raise NotImplementedError("Day 2 实现")
