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


async def main() -> None:
    embedder = Embedder()
    store = VectorStore(collection_name="support_kb")
    sparse = SparseIndex(await store.get_all())
    retriever = Retriever(embedder=embedder, store=store, sparse=sparse)

    for q in ["怎么退货", "退货运费谁出", "满 300 减 50 怎么用", "发票怎么开"]:
        dense = await retriever.search(q, top_k=3)
        hybrid = await retriever.hybrid_search(q, top_k=3)
        print(f"\n[query] {q}")
        print("  Dense :", [d["id"] for d in dense])
        print("  Hybrid:", [h["id"] for h in hybrid])


asyncio.run(main())