"""上下文组装器（Day 7 实现）。

职责：组装 系统提示词 + 历史消息 + 当前问题 → 一次 AgentLoop 要发的 messages。
规则（见 AGENTS.md 4.2 + Day 7 文档）：
- 分层：System（固定人设/纪律）→ 历史（可压缩区）→ 当前 user 消息
- 预算：历史区超阈值 → 先交给 ContextCompressor 压一遍再拼
- 当前问题永远最新、不参与压缩（压缩只发生在历史区）
"""

from typing import Any

from app.config import settings
from app.context.compressor import ContextCompressor

SYSTEM_PROMPT = """你是 MiniSupport 电商客服助手。回答要简洁、礼貌、准确。
涉及退换货 / 退款 / 物流 / 优惠券等政策规则时，先调用工具查知识库，不要凭记忆编造；
查到订单 / 物流走业务工具；查不到或超出能力时，明确告诉用户并建议转人工。"""


def estimate_tokens(messages: list[dict[str, Any]]) -> int:
    """粗估一组消息的 token 数（中英文混排按字符数折算；纯函数，便于单测）。"""
    total = 0.0
    for m in messages:
        content = m.get("content", "")
        total += len(str(content)) / 1.5
        # 工具调用参数、角色标记等附加开销
        total += len(str(m.get("tool_calls", ""))) / 1.5
        total += 4
    return int(total)


class ContextAssembler:
    """把系统提示词 + 历史 + 当前问题组装成一次 Agent 循环的输入。"""

    def __init__(
        self,
        compressor: ContextCompressor | None = None,
        system_prompt: str = SYSTEM_PROMPT,
        threshold: int = settings.compression_threshold_tokens,
    ) -> None:
        self.compressor = compressor
        self.system_prompt = system_prompt
        self.threshold = threshold

    async def assemble(
        self, message: str, history: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """组装一次请求的消息。返回的列表可安全被 agent.run() 原地追加（可写副本）。"""
        # 历史区超阈值 → 先压缩（压缩器只压更早轮次，最近 keep_recent 条保原文）
        if self.compressor is not None and estimate_tokens(history) > self.threshold:
            history = await self.compressor.compress(history)

        return [
            {"role": "system", "content": self.system_prompt},
            *history,
            {"role": "user", "content": message},
        ]
