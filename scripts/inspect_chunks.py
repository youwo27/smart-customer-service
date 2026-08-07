"""检索结果详情查看 — 打印每个 chunk 的实际文字，用于人工验证检索是否合格。

用法：
    python scripts/inspect_chunks.py                       # 默认 4 个 query
    python scripts/inspect_chunks.py 退货运费谁出 发票怎么开  # 指定 query

对每个 query，分别打印 Dense（纯向量）和 Hybrid（混合检索）的 Top-3：
id + 文本（前 80 字）。验证标准：这段文字是否真的能回答问题。
"""

import asyncio
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from app.rag.embedder import Embedder
from app.rag.retriever import Retriever
from app.rag.sparse import SparseIndex
from app.rag.store import VectorStore

DEFAULT_QUERIES = ["怎么退货", "退货运费谁出", "满 300 减 50 怎么用", "发票怎么开"]


async def main(queries: list[str]) -> None:
    embedder = Embedder()
    store = VectorStore(collection_name="support_kb")
    sparse = SparseIndex(await store.get_all())
    retriever = Retriever(embedder=embedder, store=store, sparse=sparse)

    for q in queries:
        print(f"\n{'=' * 70}\n[query] {q}\n{'=' * 70}")
        dense = await retriever.search(q, top_k=3)
        hybrid = await retriever.hybrid_search(q, top_k=3)
        for label, results in [("Dense ", dense), ("Hybrid", hybrid)]:
            print(f"\n--- {label} Top-3 ---")
            for r in results:
                text = r["text"].replace("\n", " ")
                snippet = text[:80] + "..." if len(text) > 80 else text
                print(f"  {r['id']}")
                print(f"    {snippet}")


if __name__ == "__main__":
    queries = sys.argv[1:] or DEFAULT_QUERIES
    asyncio.run(main(queries))
