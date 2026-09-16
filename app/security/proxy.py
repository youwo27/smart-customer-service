"""受控工具注册器 — 权限校验 + 审计的统一关口（Day 8 Part 4）。

思路（零侵入，别改 agent.py）：AgentLoop 每轮通过 `self.tool_registry.execute(...)` 调工具，
且它已把任何异常兜成 tool_result 让 Agent 自我纠正。所以给 Agent 塞一个"受控注册器"代理，
它转发前先查权限、转发后记审计——**Agent 完全无感**，改的只是 main 里构造 Agent 时传谁的引用。

会话来源：AgentLoop 调 execute 时不传 session_id，故经 contextvar（logging_config.session_id_ctx）
取——orchestrator.handle() 入口已 set_session_id。这样代理无需改 AgentLoop 签名。
"""

from typing import Any, cast

from app.logging_config import get_logger, get_session_id
from app.observability.tracing import short_text, span
from app.security.audit import AuditEntry
from app.security.permissions import PermissionGuard
from app.security.pii import redact

logger = get_logger(__name__)


def _summary(obj: Any, limit: int = 120) -> str:
    """把工具参数/结果压成一行摘要（脱敏 + 截断），供审计 target/detail 用。"""
    return redact(str(obj))[:limit]


class SecuredToolRegistry:
    """受控注册器：与 ToolRegistry 同表面（list / get / execute），委托真注册器。"""

    def __init__(self, registry: Any, guard: PermissionGuard, audit: Any) -> None:
        self._registry = registry
        self._guard = guard
        self._audit = audit

    def list(self) -> list[dict[str, Any]]:
        return cast("list[dict[str, Any]]", self._registry.list())

    def get(self, name: str) -> Any:
        return self._registry.get(name)

    async def execute(self, name: str, **kwargs: Any) -> dict[str, Any]:
        """执行前查权限、执行后记审计。拒绝 → 结构化错误（Agent 会当 tool_result 自然收口）。

        Day 9：这里同时是**工具 span 的埋点处** —— 受控注册器是所有工具的唯一必经关口，
        埋一处，六个工具自动全都有 span；埋在各自 execute 里得写六遍、加新工具还会忘。
        而且这里天然拿得到 Day 8 的判定结果，正好当 span attribute（观测和安全共用一处）。
        """
        session_id = get_session_id()
        tool = self._try_get(name)
        permissions = list(getattr(tool, "required_permissions", []) or []) if tool is not None else []

        with span("tool.execute", tool_name=name, session_id=session_id) as sp:
            for permission in permissions:
                reason = await self._guard.authorize(session_id, permission)
                if reason is not None:
                    sp.set_attribute("status", "denied")
                    sp.set_attribute("permission", permission)
                    sp.set_attribute("reason", short_text(reason))
                    await self._audit.record(
                        AuditEntry(
                            kind="tool_denied",
                            session_id=session_id,
                            actor="agent",
                            action=name,
                            target=_summary(kwargs),
                            detail=redact(reason),
                        )
                    )
                    logger.warning("tool_denied", name=name, permission=permission)
                    # 结构化错误 → Agent 拿它当 tool_result，向用户回"需人工确认"类话术，绝不真执行
                    return {"error": f"操作被拒绝：{reason}"}

            result: dict[str, Any] = await self._registry.execute(name, **kwargs)
            sp.set_attribute("status", "ok")
            sp.set_attribute("permission", ",".join(permissions))
            sp.set_attribute("result_bytes", len(str(result)))
            await self._audit.record(
                AuditEntry(
                    kind="tool_call",
                    session_id=session_id,
                    actor="agent",
                    action=name,
                    target=_summary(kwargs),
                    detail=_summary(result),
                )
            )
            return result

    def _try_get(self, name: str) -> Any | None:
        try:
            return self._registry.get(name)
        except (KeyError, AttributeError):
            return None  # 未注册工具交由下游 registry.execute 返回结构化错误
