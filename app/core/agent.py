"""Agent 核心循环（占位实现，Day 4 从零实现）。

职责：ReAct 循环 = 推理 → 工具调用 → 观察 → 推理。
规则（见 AGENTS.md 4.1）：所有外部调用必须有重试 + 超时，工具失败转 tool_result。
"""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class AgentLoopConfig:
    """Agent 循环配置（全部来自 settings，禁止硬编码）。"""

    max_turns: int = 15
    tool_timeout_seconds: int = 30
    llm_retry_max: int = 3
    llm_retry_base_delay: float = 1.0


@dataclass
class AgentRunResult:
    """一次 Agent 循环的运行结果。"""

    messages: list[dict[str, Any]] = field(default_factory=list)
    tool_calls_count: int = 0
    tokens_used: int = 0
    model: str = ""
    finish_reason: str = ""


class AgentLoop:
    """ReAct Agent 主循环。

    TODO (Day 4):
    - 实现 run()：while 循环，终止条件 end_turn / max_turns
    - 解析 tool_use → 执行工具 → 构造 tool_result
    - 支持并行工具调用
    - LLM 调用重试（指数退避）
    """

    def __init__(
        self,
        client: Any,  # LLMClient，Day 4 注入
        tool_registry: Any,  # ToolRegistry，Day 4 注入
        config: AgentLoopConfig | None = None,
    ) -> None:
        self.client = client
        self.tool_registry = tool_registry
        self.config = config or AgentLoopConfig()

    async def run(self, messages: list[dict[str, Any]], session_id: str) -> AgentRunResult:
        """执行一轮 Agent 循环。Day 4 实现。"""
        raise NotImplementedError("Day 4 实现")
