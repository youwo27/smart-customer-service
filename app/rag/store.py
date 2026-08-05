"""RAG：向量存储封装 — 基于 ChromaDB。

提供：写入、相似度检索、按 id 删除。
collection 参数：不同知识库可建不同 collection。
持久化目录从配置注入（零硬编码，见 AGENTS.md 3.1）。
"""

import chromadb
from typing import Any

from app.config import settings


class VectorStore:
    """ChromaDB 向量库封装。"""

    def __init__(
        self,
        collection_name: str = "support_kb",
        persist_dir: str | None = None,
    ) -> None:
        # 持久化到本地目录（配置注入，零硬编码）
        persist_dir = persist_dir or settings.chroma_persist_dir
        self._client = chromadb.PersistentClient(path=persist_dir)
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},  # 用余弦相似度
        )

    async def add(
        self,
        ids: list[str],
        documents: list[str],
        embeddings: list[list[float]] | None = None,
        metadatas: list[dict[str, Any]] | None = None,
    ) -> None:
        """批量写入。不传 embeddings 时 ChromaDB 自动算。"""
        self._collection.add(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
        )

    async def query(
        self,
        query_text: str | None = None,
        query_embedding: list[float] | None = None,
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        """相似度检索。传入文本或向量二选一。"""
        kwargs = {"n_results": top_k}
        if query_text:
            kwargs["query_texts"] = [query_text]
        if query_embedding:
            kwargs["query_embeddings"] = [query_embedding]

        result = self._collection.query(**kwargs)

        # 整理返回结构
        items: list[dict[str, Any]] = []
        for i, doc_id in enumerate(result["ids"][0]):
            items.append({
                "id": doc_id,
                "text": result["documents"][0][i],
                "distance": result["distances"][0][i] if result["distances"] else None,
            })
        return items

    async def delete(self, ids: list[str]) -> None:
        """按 id 删除。"""
        self._collection.delete(ids=ids)
