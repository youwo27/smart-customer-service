"""RAG：Agentic 检索决策 — 让 Agent 决定"要不要检索 / 够不够 / 再检不检"。

与 Advanced RAG 的区别：这里把流程的控制权从代码交给 LLM。
代价：每个决策都是一次 LLM 调用（延迟 + 成本），换来的是不用检索时不检索、
      检索不够时主动补救——客服场景值得。

三个决策点：
  ① need_retrieval：闲聊/情绪 → 直接答，不浪费检索
  ② is_sufficient：检出的内容能不能回答问题
  ③ refine：不够 → 换角度改写 → 再检一轮 → 合并去重
"""

from typing import Any

from app.llm.client import LLMClient
from app.rag.query import rewrite_query


# 决策①：需要检索吗？（只回 YES / NO）
NEED_RETRIEVAL_PROMPT = """判断用户问题是否需要检索客服知识库（政策/规则/流程文档）。
需要检索：怎么退货、退款多久到账、优惠券规则、改地址、发票、价保…
不需要检索：你好、谢谢、你是谁、闲聊、夸赞、情绪发泄、拉家常。
只回答 YES 或 NO。

问题：{question}"""

# 决策②：检索够吗？（只回 ENOUGH / NOT_ENOUGH）
IS_SUFFICIENT_PROMPT = """判断检索结果是否足以回答用户问题。
如果检索结果中没有任何相关内容能回答问题，回答 NOT_ENOUGH，否则回答 ENOUGH。
只回答一个词。

问题：{question}

检索结果：
{chunks}"""

# 决策③：生成。检索到的片段拼进系统提示词（RAG 的标准拼法）
GENERATE_PROMPT = """你是电商客服助手。只根据"参考内容"回答用户问题。
如果参考内容不包含答案，明确说"我无法确认，已为你转接人工客服"，不要编造。

参考内容：
{context}

用户问题：{question}"""


async def _need_retrieval(llm: LLMClient, question: str) -> bool:
    resp = await llm.chat(
        [{"role": "user", "content": NEED_RETRIEVAL_PROMPT.format(question=question)}]
    )
    return resp.content.strip().upper() == "YES"


async def _is_sufficient(llm: LLMClient, question: str, chunks: list[dict[str, Any]]) -> bool:
    snippets = "\n".join(f"[{i+1}] {c['text'][:200]}" for i, c in enumerate(chunks))
    resp = await llm.chat(
        [{"role": "user", "content": IS_SUFFICIENT_PROMPT.format(
            question=question, chunks=snippets)}]
    )
    return resp.content.strip().upper() == "ENOUGH"


def _merge_dedup(primary: list[dict[str, Any]], extra: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """合并两轮检索结果并按 rrf_score 去重排序。"""
    seen: dict[str, dict[str, Any]] = {}
    for c in [*primary, *extra]:
        seen.setdefault(c["id"], c)
    return sorted(seen.values(), key=lambda c: c.get("rrf_score", 0.0), reverse=True)


async def generate_answer(llm: LLMClient, question: str, chunks: list[dict[str, Any]]) -> str:
    context = "\n".join(c["text"] for c in chunks)
    resp = await llm.chat(
        [
            {"role": "system", "content": GENERATE_PROMPT.format(context=context, question=question)},
            {"role": "user", "content": question},
        ]
    )
    return resp.content


async def agentic_rag(
    question: str,
    llm: LLMClient,
    retriever: Any,   # Retriever
    reranker: Any,    # Reranker
) -> dict[str, Any]:
    """Agentic RAG 完整循环，返回结构化结果（便于评估）。"""
    log = {"question": question, "rounds": 0, "need_retrieval": None}

    # 决策①：不检索就直接答
    log["need_retrieval"] = await _need_retrieval(llm, question)
    if not log["need_retrieval"]:
        return {**log, "answer": (await llm.chat(
            [{"role": "user", "content": question}])).content, "chunks": []}

    # 检索：改写 → 混合检索 → 重排
    q = await rewrite_query(llm, question)
    candidates = await retriever.hybrid_search(q, top_k=10)
    chunks = await reranker.rerank(question, candidates, top_k=5)
    log["rounds"] += 1

    # 决策②：不够 → 换角度再检一轮
    if not await _is_sufficient(llm, question, chunks):
        refined = await rewrite_query(
            llm, f"{question}\n（初检无相关内容，请换角度改写检索词）"
        )
        extra = await retriever.hybrid_search(refined, top_k=5)
        chunks = _merge_dedup(chunks, extra)
        log["rounds"] += 1

    answer = await generate_answer(llm, question, chunks)
    return {**log, "answer": answer, "chunks": chunks}
