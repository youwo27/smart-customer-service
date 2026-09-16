"""RAG：检索器 — 混合检索（Dense + Sparse + RRF 融合）。

Day 2：单路 Dense（ChromaDB 默认 embedding）。
Day 3：
  - Dense：真实 Embedding（Part 1 的 Embedder）+ VectorStore.query
  - Sparse：BM25 关键词检索（SparseIndex）
  - 融合：Reciprocal Rank Fusion（RRF），k=60
"""

from typing import Any

from app.observability.tracing import short_text, span
from app.rag.embedder import Embedder
from app.rag.sparse import SparseIndex
from app.rag.store import VectorStore


class Retriever:
    """混合检索器。"""

    def __init__(
        self,
        embedder: Embedder,
        store: VectorStore,
        sparse: SparseIndex | None = None,
    ) -> None:
        self.embedder = embedder
        self.store = store
        self.sparse = sparse

    async def search(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        """纯 Dense 检索（保留，用于和混合检索对比）。"""
        vec = await self.embedder.embed(query)
        return await self.store.query(query_embedding=vec, top_k=top_k)

    async def hybrid_search(
        self,
        query: str,
        top_k: int = 10,
        dense_k: int = 20,
        sparse_k: int = 20,
        k: int = 60,
    ) -> list[dict[str, Any]]:
        """混合检索：Dense + Sparse → RRF 融合 → Top-K。

        Day 9：两路召回各开一个子 span，挂在 `rag.hybrid_search` 之下 ——
        "检索慢"是个太笼统的结论，得能一眼看出是 dense（embedding API + 向量库）
        慢还是 sparse（BM25 本地算）慢。query 进 attribute 前先脱敏 + 截断（Jaeger
        也是一条出口线，用户原话不能裸着进去）。
        """
        with span("rag.hybrid_search", query=short_text(query), top_k=top_k, dense_k=dense_k,
                  sparse_k=sparse_k) as sp:
            # 1. 两路召回
            with span("rag.dense_search", k=dense_k, enabled=True) as dense_sp:
                dense_results = await self.search(query, top_k=dense_k)
                dense_sp.set_attribute("result_count", len(dense_results))

            with span("rag.sparse_search", k=sparse_k, enabled=self.sparse is not None) as sparse_sp:
                sparse_results = self.sparse.search(query, top_k=sparse_k) if self.sparse else []
                sparse_sp.set_attribute("result_count", len(sparse_results))

            # 2. RRF 融合（只按名次打分）
            scores: dict[str, float] = {}
            merged: dict[str, dict[str, Any]] = {}
            for rank, r in enumerate(dense_results):
                rid = r["id"]
                scores[rid] = scores.get(rid, 0.0) + 1.0 / (k + rank + 1)
                merged[rid] = r
            for r in sparse_results:
                rid = r["id"]
                scores[rid] = scores.get(rid, 0.0) + 1.0 / (k + r["rank"] + 1)
                merged[rid] = r

            # 3. 按融合分排序取 Top-K
            ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
            results = [{**merged[rid], "rrf_score": score} for rid, score in ranked[:top_k]]
            # 融合前后各多少条，是判断"哪一路在贡献"的关键数字（比如 sparse 全空 → 关键词那条链废了）
            sp.set_attribute("candidate_count", len(scores))
            sp.set_attribute("result_count", len(results))
            return results
