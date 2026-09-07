"""编排层 — 业务调度中枢（Day 7 实现）。

职责：
- 会话管理（恢复历史 / 压缩 / 落库）
- 上下文组装（ContextAssembler）+ 调用 AgentLoop
- 把 Agent 的一轮输出 diff 出来，翻译成 AgentEvent（供 SSE / 审计 / 追踪）

关键设计决策：
- AgentLoop.run() 是"原地改 messages"的（agent.py），已经能用且有测试锁死；
  这里不改它，而是在调用前记录 messages 长度、跑完 diff 新增的部分，
  逐条翻译成 AgentEvent —— 零侵入拿事件流。
- 三个依赖（agent / assembler / session_store）全部构造注入，测试塞 fake。
"""

from collections.abc import Callable, Coroutine
from typing import Any

from app.context.assembler import ContextAssembler
from app.core.agent import AgentLoop, AgentRunResult
from app.core.events import AgentEvent
from app.logging_config import get_logger

logger = get_logger(__name__)

# 事件回调签名：收到一个 AgentEvent（可以同步处理，也可以 await async 回调）
EventSink = Callable[[AgentEvent], Coroutine[Any, Any, None] | None]

# 自研消息（agent.py 的 dict 格式）→ AgentEvent.kind
_MESSAGE_KIND = {
    "assistant": "assistant",
    "tool": "tool_result",
    "user": "user",
}


class Orchestrator:
    """编排用户请求的全过程：组装上下文 → Agent 循环 → 事件化 → 落库。"""

    def __init__(
        self,
        agent: AgentLoop,
        assembler: ContextAssembler,
        session_store: Any,  # SessionStore（Protocol，见 app/storage/session_store.py）
        emit: EventSink | None = None,
    ) -> None:
        self.agent = agent
        self.assembler = assembler
        self.session_store = session_store
        # 默认不 emit：纯 handle() 调用方不需要事件流
        self._emit_fn = emit

    # ============ 事件翻译（零侵入：只看 agent.run 新增的消息） ============
    @staticmethod
    def to_event(session_id: str, message: dict[str, Any]) -> AgentEvent:
        """把一条自研消息（agent.py 格式）翻译成 AgentEvent。

        四种消息各就各位：
          user       → kind="user"
          assistant  → 有 tool_calls → kind="tool_call"（动作预告）
                       没有 tool_calls → kind="assistant"（普通回复）
          tool       → kind="tool_result"，挂 tool_call_id / 结果
        """
        role = message.get("role")
        base_kind = _MESSAGE_KIND.get(role or "", "assistant")
        if role == "assistant":
            tool_calls = message.get("tool_calls") or []
            if tool_calls:
                # 一个 assistant 消息可带多个并行 tool_call，摊平成多个事件
                return AgentEvent(
                    kind="tool_call",
                    session_id=session_id,
                    content="",
                    data={
                        "tool_calls": [
                            {
                                "name": tc.get("function", {}).get("name", ""),
                                "arguments": tc.get("function", {}).get("arguments", ""),
                                "id": tc.get("id", ""),
                            }
                            for tc in tool_calls
                        ],
                    },
                )
            return AgentEvent(kind="assistant", session_id=session_id, content=message.get("content", ""))
        return AgentEvent(
            kind=base_kind,
            session_id=session_id,
            content=str(message.get("content", "")),
            data={
                "tool_call_id": message.get("tool_call_id", ""),
            },
        )

    async def _emit(self, event: AgentEvent, sink: EventSink | None = None) -> None:
        sink = sink if sink is not None else self._emit_fn
        if sink is None:
            return
        result = sink(event)
        if result is not None:
            await result

    # ============ 主入口 ============
    async def handle(
        self,
        message: str,
        session_id: str,
        *,
        emit: EventSink | None = None,
    ) -> AgentRunResult:
        """编排一次用户请求：组装上下文 → Agent 循环 → 事件化 → 落库。

        emit：本次调用的事件回调，覆盖构造时注入的 _emit_fn（SSE 每请求绑一个
        队列，把事件实时推给响应生成器）。缺省沿用 _emit_fn——纯 handle()
        调用方不需要事件流。
        """
        sink = emit if emit is not None else self._emit_fn

        # ① 从 SessionStore 恢复历史 → 组装本次要发的 messages
        history = await self.session_store.load(session_id)
        messages = await self.assembler.assemble(message, history)

        # ② agent.run() 原地 append；组装层拼的是可写副本，直接传入即可
        before = len(messages)
        result = await self.agent.run(messages, session_id=session_id)

        # ③ diff 出新增消息，逐条翻译成事件（assistant/tool 交替）
        for m in messages[before:]:
            await self._emit(self.to_event(session_id, m), sink=sink)
        await self._emit(
            AgentEvent(
                kind="end",
                session_id=session_id,
                content=result.finish_reason,
                data={"tokens_used": result.tokens_used},
            ),
            sink=sink,
        )

        # ④ 落库：完整 messages（不含 system —— system 每次由 assembler 重建）
        saved_history = [m for m in messages if m.get("role") != "system"]
        await self.session_store.save(
            session_id,
            messages=saved_history,
            tokens_used=result.tokens_used,
            finish_reason=result.finish_reason,
        )
        return result
