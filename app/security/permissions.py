"""动作权限 — required_permissions 声明 → 这里运行时校验（Day 8 Part 4）。

纵深防御第 3 层：文本守卫挡的是"嘴"，真正危险的放行口是**工具**。
把"越权骗 Agent 退全款"这条路物理堵死的那道闸。

关键设计决策（面试点）：
- 校验放在"注册器代理"（proxy.py）而不是 agent.py 循环里，也不是每个工具 execute() 开头：
  放 agent.py = 改核心循环有回归风险（Day 4 单测锁死）；放各工具 = 每个工具各写一遍、忘写一个漏一个。
  注册器是"所有工具必经的单一关口"，代理包一层 = 一处校验、全局生效、可整体替换。
- 危险操作**默认拒绝**（fail-closed）：DANGEROUS_DEFAULT 里的权限没显式 grant 就拒。
- 放行来源（人工审批）留给 Day 12 HITL；今天只实现"拦截 + 一次性 grant 开关"。
"""

# 没有显式 grant 就拒绝的权限（危险操作白名单之外的一律拒绝）
DANGEROUS_DEFAULT: dict[str, str] = {
    "order.refund": "退款属于高危操作，需人工确认后执行",
}


class PermissionGuard:
    """动作权限：默认拒绝危险工具；grant() 放行指定 session 的一次性权限。"""

    def __init__(self, dangerous: dict[str, str] | None = None) -> None:
        self._dangerous = dict(dangerous) if dangerous is not None else dict(DANGEROUS_DEFAULT)
        self._grants: dict[str, set[str]] = {}  # session_id -> 已放行权限

    def grant(self, session_id: str, permission: str) -> None:
        """为某会话放行一个权限（一次性；生产由人工审批调用，Day 12 HITL）。"""
        self._grants.setdefault(session_id, set()).add(permission)

    def revoke(self, session_id: str, permission: str) -> None:
        """收回某会话的一个权限。"""
        if session_id in self._grants:
            self._grants[session_id].discard(permission)

    def is_granted(self, session_id: str, permission: str) -> bool:
        """该会话是否已放行此权限。"""
        return permission in self._grants.get(session_id, set())

    async def authorize(self, session_id: str, permission: str) -> str | None:
        """放行 → None；拒绝 → 拒绝原因（给 Agent 看的客服话术）。

        规则：显式 grant 过 → 放行；属于危险权限且未 grant → 拒绝；
        非危险权限 → 放行（不在 DANGEROUS_DEFAULT 里的没被标为高危）。
        """
        if self.is_granted(session_id, permission):
            return None
        reason = self._dangerous.get(permission)
        if reason is not None:
            return reason
        return None
