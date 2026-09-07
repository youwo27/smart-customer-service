"""工具：list_documents — 列出知识库文档清单。

场景：Agent 想了解"知识库里都有什么"时用（检索前探路）。
"""

from typing import Any

from app.tools.base import BaseToolImpl
from app.tools.builtin._shared import get_doc_meta


class ListDocumentsTool(BaseToolImpl):
    """列出知识库全部文档清单（编号 + 片段数）。了解知识库里有什么时使用。"""

    name = "list_documents"
    description = (
        "列出电商知识库的全部文档清单（文档编号 + 片段数）。"
        "当需要了解知识库里有哪些文档、想浏览文档目录时使用。无参数。"
    )
    parameters = {
        "type": "object",
        "properties": {},
        "required": [],
    }

    async def execute(self, **kwargs: Any) -> dict[str, Any]:
        docs = await get_doc_meta()
        return {"total": len(docs), "documents": docs}
