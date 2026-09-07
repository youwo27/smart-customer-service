"""RAG 工具共享依赖 — 惰性构建/复用 Retriever + 文档缓存。

设计决策：
- RagService 首次使用时构建（避免 import 时触发向量库加载），此后复用同一实例
- 文档清单构建一次后缓存（list_documents / read_document 共用）
- 检索结果里 id 是 chunk id（如 "01-return-policy-chunk-0"），
  read_document 需要的是一整篇，所以要能从 chunk id 反推文档 id（前缀）
"""

import asyncio
from typing import Any

from app.rag.embedder import Embedder
from app.rag.reranker import Reranker
from app.rag.retriever import Retriever
from app.rag.sparse import SparseIndex
from app.rag.store import VectorStore


class RagService:
    """把检索能力打包成一个可复用的服务（Retriever + Reranker）。"""

    def __init__(self, retriever: Retriever, reranker: Reranker) -> None:
        self.retriever = retriever
        self.reranker = reranker

    async def search(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        """混合检索 + 重排，返回 Top-K chunk。"""
        candidates = await self.retriever.hybrid_search(query, top_k=top_k * 2)
        return await self.reranker.rerank(query, candidates, top_k=top_k)

    async def get_all_chunks(self) -> list[dict[str, Any]]:
        """全部 chunk（文档清单用）。"""
        return await self.retriever.store.get_all()


_service: RagService | None = None
_service_lock = asyncio.Lock()
_doc_meta_cache: list[dict[str, Any]] | None = None


async def get_service() -> RagService:
    """惰性构建并缓存 RagService（第一次调用才真正连向量库）。"""
    global _service
    if _service is None:
        async with _service_lock:
            if _service is None:  # 双重检查，避免并发重复构建
                embedder = Embedder()
                store = VectorStore(collection_name="support_kb")
                sparse = SparseIndex(await store.get_all())
                _service = RagService(
                    retriever=Retriever(embedder=embedder, store=store, sparse=sparse),
                    reranker=Reranker(),  # 无 key 时原序兜底
                )
    return _service


async def get_doc_meta() -> list[dict[str, Any]]:
    """文档清单（id + chunk 数），缓存复用。"""
    global _doc_meta_cache
    if _doc_meta_cache is None:
        service = await get_service()
        all_chunks = await service.get_all_chunks()
        docs: dict[str, int] = {}
        for c in all_chunks:
            did = doc_id_from_chunk(c["id"])
            docs[did] = docs.get(did, 0) + 1
        _doc_meta_cache = [{"id": did, "chunk_count": n} for did, n in sorted(docs.items())]
    return _doc_meta_cache


def doc_id_from_chunk(chunk_id: str) -> str:
    """从 chunk id 反推文档 id（"01-return-policy-chunk-0" → "01-return-policy"）。"""
    return chunk_id.rsplit("-chunk-", 1)[0] if "-chunk-" in chunk_id else chunk_id
