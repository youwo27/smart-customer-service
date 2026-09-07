"""工具：read_document — 读取知识库单篇文档的完整内容。

场景：search_knowledge_base 只返回片段，用户需要"整篇规则"时用这个补全。
"""

from typing import Any

from app.rag.loader import DocumentLoader
from app.tools.base import BaseToolImpl


class ReadDocumentTool(BaseToolImpl):
    """读取知识库单篇文档完整内容。当检索片段不够、需要完整规则时使用。"""

    name = "read_document"
    description = (
        "读取知识库单篇文档的完整内容。"
        "当 search_knowledge_base 返回的片段不够完整、需要看整篇规则时使用。"
        "参数 doc_id 是文档编号，如 '01-return-policy'。"
    )
    parameters = {
        "type": "object",
        "properties": {
            "doc_id": {"type": "string", "description": "文档编号，如 '01-return-policy'"},
        },
        "required": ["doc_id"],
    }

    async def execute(self, **kwargs: Any) -> dict[str, Any]:
        doc_id = kwargs.get("doc_id", "")
        if not doc_id.strip():
            return {"error": "doc_id 不能为空"}
        try:
            doc = DocumentLoader().load_one(doc_id)
        except FileNotFoundError:
            return {"error": f"文档不存在: {doc_id}"}
        return {"id": doc.id, "text": doc.text}
