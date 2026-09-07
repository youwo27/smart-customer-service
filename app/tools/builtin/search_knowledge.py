"""工具：search_knowledge_base — 混合检索知识库，返回最相关的片段。

description 关键（AGENTS.md 4.1）：
- 写清"什么时候用"：用户问政策/规则/流程（退货、退款、优惠券、发票…）
- 写清"什么时候不用"：闲聊、情绪、数学计算 → 避免 Agent 对每个问题都乱调知识库
"""

from typing import Any

from app.tools.base import BaseToolImpl
from app.tools.builtin._shared import get_service


class SearchKnowledgeBaseTool(BaseToolImpl):
    """检索知识库。用户问退货/退款/优惠券/发票/价保等政策时用；闲聊、数学计算不适用。"""

    name = "search_knowledge_base"
    description = (
        "检索电商客服知识库，返回与问题最相关的文档片段。"
        "当用户询问退货、退款、运费、优惠券、发票、价保、物流等政策规则时使用。"
        "不适用于：闲聊、情绪发泄、数学计算。参数 query 是检索关键词。"
    )
    parameters = {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "检索关键词，如 '退货政策'"},
            "top_k": {"type": "integer", "description": "返回片段数，默认 5", "default": 5},
        },
        "required": ["query"],
    }

    async def execute(self, **kwargs: Any) -> dict[str, Any]:
        query = kwargs.get("query", "")
        if not query.strip():
            return {"error": "query 不能为空"}
        top_k = int(kwargs.get("top_k", 5))
        service = await get_service()
        chunks = await service.search(query, top_k=top_k)
        if not chunks:
            return {"chunks": [], "message": "知识库中未找到相关内容"}
        return {
            "chunks": [
                {
                    "id": c["id"],
                    "text": c["text"][:500],
                    "source": c.get("source", ""),
                    "score": c.get("rrf_score", c.get("rerank_score")),
                }
                for c in chunks
            ]
        }
