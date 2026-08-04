"""上下文组装器（占位实现，Day 7 实现）。

职责：组装系统提示词 + 历史消息 + 动态上下文。
规则（见 AGENTS.md 4.2）：
- 系统提示词分层：System → Task → Dynamic
- 上下文预算：System 10K + 工具 20K + 历史 150K + 预留 20K
"""

from typing import Any


class ContextAssembler:
    """上下文组装器。

    TODO (Day 7):
    - build(system_prompt, history, dynamic) -> messages
    - 触发压缩阈值时调用 ContextCompressor
    """

    async def assemble(self, message: str, history: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """组装上下文。Day 7 实现。"""
        raise NotImplementedError("Day 7 实现")
