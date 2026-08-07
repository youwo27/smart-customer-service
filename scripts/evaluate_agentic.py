import asyncio, os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from app.llm.client import LLMClientFactory, LLMConfigFactory
from app.rag.agentic import agentic_rag
from app.rag.embedder import Embedder
from app.rag.reranker import Reranker
from app.rag.retriever import Retriever
from app.rag.sparse import SparseIndex
from app.rag.store import VectorStore

# (query, 期望命中文档, 问题类型)
CASES = [
    ("怎么退货",                "01-return-policy",   "需检索"),
    ("退货运费谁出",            "01-return-policy",   "需检索"),
    ("退款要多久到账",          "05-refund-rules",    "需检索"),
    ("优惠券怎么用",            "17-coupon-rules",    "需检索"),
    ("我的快递到哪了",          "11-logistics-tracking", "需检索"),
    ("支持价格保护吗",          "16-price-protection", "需检索"),
    ("发票怎么开",              "15-invoice",         "需检索"),
    ("你好",                    None,                 "闲聊"),
    ("谢谢",                    None,                 "闲聊"),
    ("羽绒服起球了怎么办",      None,                 "检索不充分(知识库无此内容)"),
]


async def main() -> None:
    llm = LLMClientFactory.create(LLMConfigFactory.from_settings())
    embedder = Embedder()
    store = VectorStore(collection_name="support_kb")
    retriever = Retriever(embedder=embedder, store=store,
                          sparse=SparseIndex(await store.get_all()))
    reranker = Reranker()

    print(f"{'查询':<14}  {'需检索?':>5}  {'检索到期望文档':>10}  {'回答(前40字)'}")
    print("-" * 70)
    for q, expected, _ in CASES:
        result = await agentic_rag(q, llm, retriever, reranker)
        hit = "YES" if expected and any(c["id"].startswith(expected) for c in result["chunks"]) else (
            "NO" if expected else "-")
        print(f"{q:<14}  {str(result['need_retrieval']):>5}  {hit:>10}  {result['answer'][:40]}")


asyncio.run(main())