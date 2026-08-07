"""入库脚本 — 把切割后的 chunk 写入向量库（用真实 Embedder 显式算向量）。

Day 2：让 ChromaDB 自动 embedding（内置 384 维 onnx-mini-lm）。
Day 3：改用 app.rag.embedder.Embedder（zhipu 1024 维 / chroma 384 维），
      显式传入向量，保证与查询侧维度一致。

用法：
    python scripts/index_documents.py [strategy] [--rebuild]

  strategy  切割策略（recursive | fixed），默认 recursive
  --rebuild 先删除旧集合再重建（跨维度升级时必用：如 384 → 1024）

注意：
  - ChromaDB 集合维度一经创建即固定，维度变了必须 --rebuild。
  - 不传 --rebuild 时，如果集合维度与当前 Embedder 不符会直接报错，不会静默覆盖。
"""

import asyncio
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.rag.chunker import Chunker
from app.rag.embedder import Embedder
from app.rag.loader import DocumentLoader
from app.rag.store import VectorStore


async def main(strategy: str = "recursive", rebuild: bool = False) -> None:
    print(f"=== 入库开始（切割策略: {strategy}, rebuild={rebuild}）===")

    # 1. 加载
    docs = DocumentLoader("data/raw_documents").load_all()
    print(f"① 加载 {len(docs)} 篇文档")

    # 2. 切割
    chunker = Chunker(chunk_size=200, overlap=50)
    all_chunks = []
    for doc in docs:
        all_chunks.extend(chunker.chunk(doc, strategy=strategy))
    print(f"② 切割成 {len(all_chunks)} 个 chunk")

    # 3. 写入向量库：先算向量再显式传入（维度与查询侧一致）
    store = VectorStore(collection_name="support_kb")
    if rebuild:
        print("   --rebuild：删除旧集合（维度升级时必须）")
        await store.drop()
        store = VectorStore(collection_name="support_kb")

    embedder = Embedder()
    print(f"   使用 Embedder: provider={embedder.provider}, dimension={embedder.dimension}")

    ids = [c["id"] for c in all_chunks]
    documents = [c["text"] for c in all_chunks]
    metadatas = [{"source": c["source"], "index": c["index"]} for c in all_chunks]
    embeddings = await embedder.embed_batch(documents)
    await store.add(
        ids=ids,
        documents=documents,
        embeddings=embeddings,
        metadatas=metadatas,
    )
    print(f"③ 写入 {len(all_chunks)} 个 chunk 到向量库（维度 {embedder.dimension}）")

    # 4. 检索验证
    test_queries = ["怎么退货", "退货运费谁出", "优惠券怎么用"]
    print("\n=== 检索验证 ===")
    for q in test_queries:
        vec = await embedder.embed(q)
        results = await store.query(query_embedding=vec, top_k=3)
        print(f"\n[query] {q}")
        for r in results[:2]:
            print(f"  - [{r['distance']:.3f}] {r['text'][:40]}...")


if __name__ == "__main__":
    args = sys.argv[1:]
    strategy = args[0] if args and args[0] not in ("--rebuild",) else "recursive"
    rebuild = "--rebuild" in args
    asyncio.run(main(strategy, rebuild))
