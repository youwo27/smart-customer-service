"""编排层 — 业务调度中枢（占位实现，Day 7 实现）。

职责：
- 会话管理（创建/恢复/过期）
- 上下文组装（ContextAssembler）+ 上下文压缩
- 调用 AgentLoop → 响应格式化
"""


from app.core.agent import AgentLoop, AgentRunResult
from app.logging_config import get_logger

logger = get_logger(__name__)


class Orchestrator:
    """编排用户请求的全过程。

    TODO (Day 7):
    - 组装上下文 → AgentLoop → 格式化
    - 通过 AgentEvent 数据结构串联各模块
    """

    def __init__(self, agent: AgentLoop | None = None) -> None:
        self.agent = agent

    async def handle(self, message: str, session_id: str | None) -> AgentRunResult:
        """处理一条用户消息。Day 7 实现。"""
        raise NotImplementedError("Day 7 实现")
