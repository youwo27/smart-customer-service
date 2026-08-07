"""RAG：向量存储封装 — 基于 ChromaDB。

提供：写入、相似度检索、按 id 删除。
collection 参数：不同知识库可建不同 collection。
持久化目录从配置注入（零硬编码，见 AGENTS.md 3.1）。
"""

from typing import Any

import chromadb

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

    async def get_all(self) -> list[dict[str, Any]]:
        """取回库内全部条目（建 BM25 索引 / 重索引时用）。"""
        data = self._collection.get(include=["documents", "metadatas"])
        ids = data.get("ids") or []
        texts = data.get("documents") or []
        metas = data.get("metadatas") or []
        return [
            {"id": i, "text": t, "metadata": m or {}}
            for i, t, m in zip(ids, texts, metas)
        ]

    async def clear(self) -> None:
        """清空整个 collection（换 Embedding 后重建索引前必须用）。"""
        ids = self._collection.get()["ids"]
        if ids:
            self._collection.delete(ids=ids)

    async def drop(self) -> None:
        """删除整个 collection（跨维度重建时用）。

        注意：ChromaDB 的 collection 维度一经创建即固定，`clear()` 只能删数据、
        改不了维度。换 provider 导致维度变化（如 384 → 1024）时，必须删掉
        集合让下次 `__init__` 重新创建。
        """
        self._client.delete_collection(name=self._collection.name)
