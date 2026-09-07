"""内置工具集合 — 6 个工具：3 RAG + 3 客服业务。

提供 register_defaults() 一键注册全部内置工具（demo / 测试 / 生产入口共用）。
"""

from app.tools.builtin.customer_service import (
    ApplyRefundTool,
    QueryLogisticsTool,
    QueryOrderTool,
)
from app.tools.builtin.list_documents import ListDocumentsTool
from app.tools.builtin.read_document import ReadDocumentTool
from app.tools.builtin.search_knowledge import SearchKnowledgeBaseTool
from app.tools.registry import ToolRegistry

ALL_TOOLS = [
    SearchKnowledgeBaseTool,
    ReadDocumentTool,
    ListDocumentsTool,
    QueryOrderTool,
    QueryLogisticsTool,
    ApplyRefundTool,
]


def register_defaults(registry: ToolRegistry) -> ToolRegistry:
    """把全部内置工具注册进 registry，返回同一个 registry。"""
    for tool_cls in ALL_TOOLS:
        registry.register(tool_cls())
    return registry
