"""工具基类 — 定义工具接口与权限声明。

与 app/tools/registry.py 的 BaseTool(Protocol) 不同，这里提供**可选继承**的实体基类：
- 提供统一的 constructor 写法
- 让"危险操作"能声明 required_permissions（Day 8 权限校验的钩子）

Day 4 约定：权限字段先声明 + 留钩子，不实现完整校验（见 Day 4 常见坑 #5）。
"""

from typing import Any, Protocol


class BaseTool(Protocol):
    """工具接口协议（registry 认的）——定义在 registry.py，这里引用保持一致。"""

    name: str
    description: str
    parameters: dict[str, Any]

    async def execute(self, **kwargs: Any) -> dict[str, Any]: ...


class BaseToolImpl(BaseTool):
    """实体工具基类（可选继承）。危险操作可覆写 `required_permissions` 声明所需权限。"""

    name: str = ""
    description: str = ""
    parameters: dict[str, Any] = {}
    required_permissions: list[str] = []  # 危险操作（退款/改地址）声明权限，Day 8 校验

    async def execute(self, **kwargs: Any) -> dict[str, Any]:
        raise NotImplementedError
