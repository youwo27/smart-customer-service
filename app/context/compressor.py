"""上下文压缩器 — 历史超预算时，把"更早的轮次"压成一条摘要（Day 7 实现）。

设计（对齐 assembler 的分层）：
- 只压缩"更早"的部分，最近 keep_recent 条保留原文 —— 用户刚说过的不能并掉
- 摘要塞在压缩后历史的最前面（role="assistant" 的回顾），由 assembler 在头部拼 System
- 压缩是 LLM 调用，本身也吃 token：由 assembler 的 estimate_tokens 决定何时触发

谁调用：ContextAssembler.assemble() 在 estimate_tokens(history) > threshold 时调用。
"""

from typing import Any

from app.config import settings
from app.llm.client import LLMClient

COMPRESS_PROMPT = """把下面的客服对话历史压缩成一段简明摘要。要求：
- 保留：用户诉求、已答复的结论、已查到的订单号/物流号/金额、已承诺的事项
- 丢弃：寒暄、客套、重复、错误的中间推理
- 输出"用户要点 + 客服已答复要点"两部分，各一两句

对话历史：
{history}"""


class ContextCompressor:
    """把较旧的历史轮次压成摘要，返回可继续拼接的消息列表。"""

    def __init__(
        self,
        llm: LLMClient,
        keep_recent: int = settings.compression_keep_recent,
    ) -> None:
        self.llm = llm
        self.keep_recent = keep_recent

    async def compress(self, history: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """压缩 history。消息太少或全是最近内容时不压缩，原样返回。"""
        if len(history) <= self.keep_recent:
            return history

        older = history[:-self.keep_recent]
        recent = history[-self.keep_recent:]
        # tool 结果若离开了它的发起者（被压进 older），就没上下文了，直接丢掉
        recent = [m for m in recent if m.get("role") != "tool"]

        transcript = "\n".join(
            f"{m.get('role')}: {str(m.get('content', ''))[:300]}" for m in older
        )
        resp = await self.llm.chat(
            [{"role": "user", "content": COMPRESS_PROMPT.format(history=transcript)}]
        )
        summary = resp.content.strip()
        if not summary:
            return recent

        recap: dict[str, Any] = {
            "role": "assistant",
            "content": f"[历史摘要] {summary}",
        }
        return [recap, *recent]
