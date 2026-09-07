"""自研 6 工具 → LangChain StructuredTool 转换器（Day 5 交付物）。

设计决策（对比文档里"框架替你做了什么"的证据）：
- 自研工具是"类 + 实例方法 execute"（app/tools/builtin/），LangChain 认的是
  "可调用函数 + 类型注解 + Pydantic model"。转换层把 execute 包成 async 函数，
  传给 StructuredTool.from_function(coroutine=...)。
- 自研工具的参数 schema 是**手写 JSON Schema**（parameters 字段，AGENTS.md 4.1）。
  LangChain 的 StructuredTool 可以自动从类型注解生成，但为了两端 schema 完全一致，
  这里把自研 schema 转成 Pydantic model 传给 args_schema——参数名、必填、description
  都以 app/tools 里那份手写版为准（单一事实源），不重写第二份。
- 这里只做"翻译"，不改任何业务逻辑：execute 的返回 dict 原样透传。

用法：
    from scripts._lc_tools import to_langchain_tools
    tools = to_langchain_tools(registry)
    tools[0].name / tools[0].description / await tools[0].ainvoke({...})
"""

from __future__ import annotations

from typing import Any

from langchain_core.tools import StructuredTool
from pydantic import Field, create_model

from app.tools.registry import BaseTool, ToolRegistry


def _args_model(schema: dict[str, Any]) -> Any:
    """把自研 JSON Schema 转成 Pydantic model（供 args_schema 使用）。

    自研 schema 形如 {"type": "object", "properties": {...}, "required": [...]}。
    这里按 properties 的 name → (type 注解, 默认值/必填标记 + description) 建字段：
    - required 里的参数 → 必填（无默认值）
    - 其余参数 → 可选 + 从 properties 读 default（如 search_knowledge_base 的 top_k=5）
    - type 映射：string→str / integer→int / number→float
    """
    required = set(schema.get("required") or [])
    props = schema.get("properties") or {}

    fields: dict[str, tuple[Any, Any]] = {}
    for name, prop in props.items():
        py_type = _to_python_type(prop.get("type"))
        description = prop.get("description")
        if name in required:
            fields[name] = (py_type, Field(..., description=description))
        else:
            default = prop.get("default")
            fields[name] = (
                py_type,
                Field(default=default, description=description),
            )

    # 模型名用工具无关的通用名即可；field_definition 顺序保持 schema 原序
    return create_model("ToolArgs", **fields)  # type: ignore[call-overload]


def _to_python_type(json_type: str | None) -> type:
    """JSON Schema type → Python 类型。未知类型一律按 str 兜底。"""
    return {
        "string": str,
        "integer": int,
        "number": float,
        "boolean": bool,
        "array": list,
        "object": dict,
    }.get(json_type or "", str)


def make_lc_tool(tool: BaseTool) -> StructuredTool:
    """把单个自研工具转换成 LangChain StructuredTool（async 版本）。

    - name / description 直接复用（工具调用准确率取决于 description，见 AGENTS.md 4.1）
    - args_schema = 从自研手写 schema 转换出的 Pydantic model，保证两端参数定义一致
    - 执行逻辑 = 原样调用自研 execute(**kwargs)
    """
    async def _run(**kwargs: Any) -> dict[str, Any]:
        # 用该工具的exxecute()方法
        return await tool.execute(**kwargs)
    # 创建 StructuredTool,组装
    return StructuredTool.from_function(
        coroutine=_run,
        name=tool.name,
        description=tool.description,
        args_schema=_args_model(tool.parameters),
    )


def to_langchain_tools(registry: ToolRegistry) -> list[StructuredTool]:
    """把 registry 里的全部自研工具批量转成 StructuredTool（顺序与注册一致）。"""
    # registry.list() 给的是 LLM API 格式（含 type/function 包装），要拿到原始对象
    # 调 execute() 得直接遍历注册表；这里用一个公开小接口封装，避免依赖私有字段。
    return [make_lc_tool(t) for t in registry._tools.values()]
