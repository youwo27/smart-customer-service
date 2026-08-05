import asyncio
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.rag.loader import DocumentLoader
from app.rag.chunker import Chunker
from app.rag.store import VectorStore


async def main(strategy: str = "recursive") -> None:
    print(f"=== 入库开始（切割策略: {strategy}）===")

    # 1. 加载
    docs = DocumentLoader("data/raw_documents").load_all()
    print(f"① 加载 {len(docs)} 篇文档")

    # 2. 切割
    chunker = Chunker(chunk_size=200, overlap=50)
    all_chunks = []
    for doc in docs:
        all_chunks.extend(chunker.chunk(doc, strategy=strategy))
    print(f"② 切割成 {len(all_chunks)} 个 chunk")

    # 3. 写入向量库（让 ChromaDB 自动 embedding）
    store = VectorStore(collection_name="support_kb")
    await store.add(
        ids=[c["id"] for c in all_chunks],
        documents=[c["text"] for c in all_chunks],
        metadatas=[{"source": c["source"], "index": c["index"]} for c in all_chunks],
    )
    print(f"③ 写入 {len(all_chunks)} 个 chunk 到向量库")

    # 4. 检索验证
    test_queries = ["怎么退货", "退货运费谁出", "优惠券怎么用"]
    print("\n=== 检索验证 ===")
    for q in test_queries:
        results = await store.query(query_text=q, top_k=3)
        print(f"\n[query] {q}")
        for r in results[:2]:
            print(f"  - [{r['distance']:.3f}] {r['text'][:40]}...")


if __name__ == "__main__":
    strategy = sys.argv[1] if len(sys.argv) > 1 else "recursive"
    asyncio.run(main(strategy))