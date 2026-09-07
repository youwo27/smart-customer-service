"""LCEL RAG 链 — retriever | prompt | llm | StrOutputParser（Day 5 交付物）。

与自研 RAG 流程对标：
    自研：retriever.hybrid_search → assemble_context → llm.chat → 输出
    LCEL： rag_service.search | format_docs | prompt | llm | StrOutputParser

LCEL 的 `|` pipe 把每步串成一条 Runnable 链，整条链暴露统一接口：
    .invoke()  同步
    .ainvoke() 异步
    .stream()  流式（Day 7 SSE 会用到）
    .bind()    绑定参数（如 temperature）

运行方式：
    python scripts/lcel_rag_chain.py "退货政策是什么？"
"""

import asyncio
import os
import sys
from typing import Any

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnableLambda, RunnablePassthrough
from langchain_openai import ChatOpenAI

from app.config import settings
from app.tools.builtin._shared import get_service

# RAG 专用提示词：只要求"根据上下文回答"，不要求调工具
RAG_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        (
            "你是电商客服助手。只根据下面提供的知识库片段回答用户问题。"
            "如果片段里找不到答案，就如实说'知识库里没有相关信息'，不要编造。\n\n"
            "知识库片段：\n{context}"
        ),
    ),
    ("human", "{question}"),
])


def format_docs(docs: list[dict[str, Any]]) -> str:
    """把检索结果拼成上下文（自研里 assemble_context 干的事）。"""
    return "\n\n".join(d.get("text", "") for d in docs)


def build_rag_chain() -> Runnable[Any, str]:
    """组装 LCEL RAG 链。

    - rag_service.search：复用自研混合检索 + Rerank（RagService）
    - format_docs：把 chunks 拼成上下文
    - RunnablePassthrough：把原始 question 原样透传给 prompt
    - StrOutputParser：LLM 返回的 AIMessage → 字符串
    """
    llm = ChatOpenAI(  # type: ignore[call-arg]  # openai_api_key/base 是 pydantic 字段，0.3.x stub 未暴露
        model=settings.llm_model,
        openai_api_key=settings.llm_api_key,
        openai_api_base=settings.llm_base_url,
        temperature=0.2,
    )

    async def _search(query: str) -> str:
        """检索 + 拼上下文一步完成（返回给 prompt 的 context 字符串）。"""
        service = await get_service()
        docs = await service.search(query, top_k=5)
        return format_docs(docs)

    # 普通函数不能直接 |，要用 RunnableLambda 包成 Runnable 才能进 LCEL 链。
    # LCEL 链的真实类型是 RunnableSerializable，用较宽的 Runnable[Any, str] 注解：
    # 输入是"问题字符串"，RunnablePassthrough 把它原样透传成 {question}。
    search_step: Runnable[Any, str] = RunnableLambda(_search)
    rag_chain: Runnable[Any, str] = (
        {"context": search_step, "question": RunnablePassthrough()}
        | RAG_PROMPT
        | llm
        | StrOutputParser()
    )
    return rag_chain


async def main() -> None:
    chain = build_rag_chain()
    query = sys.argv[1] if len(sys.argv) > 1 else "退货政策是什么？"
    print(f"\n用户：{query}\n{'=' * 70}")
    answer = await chain.ainvoke(query)
    print(answer)


if __name__ == "__main__":
    asyncio.run(main())
