"""工具注册中心（占位实现，Day 4 实现）。

职责：注册、发现、按名称获取工具。
规则（见 AGENTS.md 4.1）：
- 每个工具必须有 name / description / parameters / execute
- description 写清：什么时候用、什么时候不用、参数怎么写
"""

from typing import Any, Protocol


class BaseTool(Protocol):
    """工具接口协议。"""

    name: str
    description: str
    parameters: dict[str, Any]

    async def execute(self, **kwargs: Any) -> dict[str, Any]: ...


class ToolRegistry:
    """工具注册中心。

    TODO (Day 4):
    - register() / get() / list()
    - 权限校验（required_permissions）
    """

    def __init__(self) -> None:
        self._tools: dict[str, BaseTool] = {}

    async def execute(self, name: str, **kwargs: Any) -> dict[str, Any]:
        """执行指定工具。Day 4 实现。"""
        raise NotImplementedError("Day 4 实现")
