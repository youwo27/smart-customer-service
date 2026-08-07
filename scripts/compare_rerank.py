"""Rerank 前后对比 — 混合检索 Top-10 → Rerank 取 Top-5。

用法：
    python scripts/compare_rerank.py                       # 默认 5 个 query
    python scripts/compare_rerank.py 退货运费谁出 发票怎么开  # 指定 query

没配置 RERANK_API_KEY 时走"原序返回"兜底（重排后 = 重排前 Top-5），
配置 key 后即可对比真正的 Cross-Encoder 重排效果。
"""

import asyncio
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from app.rag.embedder import Embedder
from app.rag.reranker import Reranker
from app.rag.retriever import Retriever
from app.rag.sparse import SparseIndex
from app.rag.store import VectorStore

DEFAULT_QUERIES = [
    "怎么退货",
    "退货运费谁出",
    "满 300 减 50 怎么用",
    "发票怎么开",
    "沙发配送",
]

SNIPPET_LEN = 45


def _snippet(text: str) -> str:
    one_line = text.replace("\n", " ")
    return one_line[:SNIPPET_LEN] + "..." if len(one_line) > SNIPPET_LEN else one_line


async def main(queries: list[str]) -> None:
    embedder = Embedder()
    store = VectorStore(collection_name="support_kb")
    sparse = SparseIndex(await store.get_all())
    retriever = Retriever(embedder=embedder, store=store, sparse=sparse)
    reranker = Reranker()
    print(f"Rerank provider: {reranker._provider or '(空 → 原序兜底)'}")

    for q in queries:
        print(f"\n{'=' * 70}\n[query] {q}\n{'=' * 70}")
        candidates = await retriever.hybrid_search(q, top_k=10)
        before = candidates[:5]
        after = await reranker.rerank(q, candidates, top_k=5)

        for label, results in [("重排前 Top-5", before), ("重排后 Top-5", after)]:
            print(f"\n--- {label} ---")
            for c in results:
                score = c.get("rerank_score")
                suffix = f"  score={score:.3f}" if score is not None else ""
                print(f"  {c['id']}{suffix}")
                print(f"    {_snippet(c['text'])}")


if __name__ == "__main__":
    queries = sys.argv[1:] or DEFAULT_QUERIES
    asyncio.run(main(queries))
