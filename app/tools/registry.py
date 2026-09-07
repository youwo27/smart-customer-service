"""工具注册中心 — 注册、发现、按名称获取工具。

职责：
- register()：注册工具，name 唯一（重复注册报错）
- get()：按名获取，找不到 → KeyError
- list()：转成 LLM 需要的 API 格式（chat_with_tools / bind_tools 都认）
- execute()：统一执行入口，按名分发给对应工具

关键设计决策（见 AGENTS.md 4.1）：
- 每个工具必须有 name / description / parameters(JSON Schema) / async execute()
- description 写清"什么时候用、什么时候不用、参数怎么写"（决定工具调用准确率）
- execute() 统一返回 dict；参数校验失败 / 工具异常也返回结构化错误，让 Agent 自我纠正
"""

from typing import Any, Protocol

from app.logging_config import get_logger

logger = get_logger(__name__)


class BaseTool(Protocol):
    """工具接口协议。"""

    name: str
    description: str
    parameters: dict[str, Any]

    async def execute(self, **kwargs: Any) -> dict[str, Any]: ...


class ToolRegistry:
    """工具注册中心。"""

    def __init__(self) -> None:
        self._tools: dict[str, BaseTool] = {}

    def register(self, tool: BaseTool) -> None:
        """注册工具。name 重复 → 报错。"""
        if not tool.name:
            raise ValueError("工具 name 不能为空")
        if tool.name in self._tools:
            raise ValueError(f"工具重复注册: {tool.name}")
        self._tools[tool.name] = tool
        logger.info("tool_registered", name=tool.name)

    def get(self, name: str) -> BaseTool:
        """按名获取工具，找不到 → KeyError。"""
        if name not in self._tools:
            raise KeyError(f"未注册的工具: {name}")
        return self._tools[name]

    def list(self) -> list[dict[str, Any]]:
        """转成 LLM 需要的 API 格式（tool schema 列表）。"""
        return [
            {
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description,
                    "parameters": t.parameters,
                },
            }
            for t in self._tools.values()
        ]

    async def execute(self, name: str, **kwargs: Any) -> dict[str, Any]:
        """统一执行入口。按名分发给对应工具。

        未知工具 → 返回结构化错误（让 Agent 自我纠正），不抛异常。
        """
        tool = self._tools.get(name)
        if tool is None:
            logger.warning("tool_not_found", name=name)
            return {"error": f"未注册的工具: {name}"}
        logger.info("tool_executing", name=name, kwargs=kwargs)
        return await tool.execute(**kwargs)
