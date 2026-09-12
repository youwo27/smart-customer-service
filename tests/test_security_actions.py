"""权限 + 审计 + 受控注册器测试（Day 8 Part 4）。

覆盖：危险操作默认拒绝 / grant 放行 / 按会话隔离 / 审计留痕；受控注册器
执行前查权限、执行后记审计，拒绝返回客服话术（Agent 当 tool_result 收口）。
"""

from app.logging_config import set_session_id
from app.security.audit import AuditEntry, MemoryAuditLog, SqliteAuditLog, create_audit_log
from app.security.permissions import PermissionGuard
from app.security.proxy import SecuredToolRegistry
from app.tools.registry import ToolRegistry


class TestPermissionGuard:
    async def test_dangerous_denied_by_default(self) -> None:
        guard = PermissionGuard()
        reason = await guard.authorize("s1", "order.refund")
        assert reason is not None  # 拒绝 + 给 Agent 的原因

    async def test_grant_allows(self) -> None:
        guard = PermissionGuard()
        guard.grant("s1", "order.refund")
        assert await guard.authorize("s1", "order.refund") is None

    async def test_grant_is_per_session(self) -> None:
        guard = PermissionGuard()
        guard.grant("s1", "order.refund")
        assert await guard.authorize("s2", "order.refund") is not None  # 别的会话不共享

    async def test_revoke(self) -> None:
        guard = PermissionGuard()
        guard.grant("s1", "order.refund")
        guard.revoke("s1", "order.refund")
        assert await guard.authorize("s1", "order.refund") is not None

    async def test_non_dangerous_permission_allowed(self) -> None:
        guard = PermissionGuard()
        assert await guard.authorize("s1", "order.read") is None  # 非危险权限放行


class TestAuditLog:
    async def test_memory_records(self) -> None:
        log = MemoryAuditLog()
        await log.record(
            AuditEntry(kind="tool_call", session_id="s1", actor="agent", action="query_order")
        )
        assert len(log.entries) == 1
        assert log.entries[0].action == "query_order"

    async def test_sqlite_records_and_queries(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        log = SqliteAuditLog(str(tmp_path / "audit.db"))
        await log.record(
            AuditEntry(kind="tool_denied", session_id="s1", actor="agent", action="apply_refund")
        )
        await log.record(
            AuditEntry(kind="tool_call", session_id="s2", actor="agent", action="query_order")
        )
        s1 = await log.query("s1")
        assert len(s1) == 1
        assert s1[0].kind == "tool_denied"

    def test_factory_rejects_unknown_backend(self) -> None:
        try:
            create_audit_log(backend="nope")
        except ValueError as exc:
            assert "audit_backend" in str(exc)
        else:
            raise AssertionError("未知后端应报错")


# ============ 受控注册器（权限 → 执行 → 审计） ============


class _RefundTool:
    name = "apply_refund"
    description = "退款"
    parameters: dict = {}
    required_permissions = ["order.refund"]

    async def execute(self, **kwargs: object) -> dict:
        return {"status": "refunded"}


class _OrderTool:
    name = "query_order"
    description = "查订单"
    parameters: dict = {}

    async def execute(self, **kwargs: object) -> dict:
        return {"status": "已发货"}


def _secured(tool: object, guard: PermissionGuard, audit: MemoryAuditLog) -> SecuredToolRegistry:
    reg = ToolRegistry()
    reg.register(tool)  # type: ignore[arg-type]
    return SecuredToolRegistry(reg, guard, audit)


class TestSecuredToolRegistry:
    async def test_dangerous_tool_denied_and_audited(self) -> None:
        audit = MemoryAuditLog()
        secured = _secured(_RefundTool(), PermissionGuard(), audit)
        set_session_id("s1")

        result = await secured.execute("apply_refund", order_id="12345")

        assert "error" in result and "拒绝" in result["error"]  # 客服话术，非裸异常
        assert any(e.kind == "tool_denied" for e in audit.entries)

    async def test_granted_tool_executes_and_audits(self) -> None:
        audit = MemoryAuditLog()
        guard = PermissionGuard()
        guard.grant("s1", "order.refund")
        secured = _secured(_RefundTool(), guard, audit)
        set_session_id("s1")

        result = await secured.execute("apply_refund", order_id="12345")

        assert result == {"status": "refunded"}  # 真执行了
        assert any(e.kind == "tool_call" for e in audit.entries)
        assert not any(e.kind == "tool_denied" for e in audit.entries)

    async def test_benign_tool_passes(self) -> None:
        secured = _secured(_OrderTool(), PermissionGuard(), MemoryAuditLog())
        set_session_id("s1")
        result = await secured.execute("query_order", order_id="12345")
        assert result["status"] == "已发货"

    async def test_list_delegates_to_registry(self) -> None:
        secured = _secured(_OrderTool(), PermissionGuard(), MemoryAuditLog())
        assert secured.list()[0]["function"]["name"] == "query_order"
