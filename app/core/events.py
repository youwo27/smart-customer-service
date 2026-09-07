"""Agent 事件 — 模块间统一"信封"，供流式输出 + 审计 + 追踪共用（Day 7）。

为什么不用裸 dict：每个模块都 return dict，字段名各写各的，下游解析靠猜。
事件让"一条消息经历了什么"成为一等公民：SSE 按事件发，审计按事件记。
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass
class AgentEvent:
    """一次编排步骤的产物（user / assistant / tool_call / tool_result / end）。"""

    kind: str                    # user | assistant | tool_call | tool_result | end
    session_id: str
    content: str = ""
    data: dict[str, Any] = field(default_factory=dict)  # 工具名/参数/耗时等结构化字段
    created_at: datetime = field(default_factory=datetime.now)
    trace_id: str = ""
