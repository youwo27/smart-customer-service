"""API 路由定义（接入层）。

职责边界（见 AGENTS.md 2）：
- 只做请求校验与响应返回，不写业务逻辑
- 业务逻辑在编排层（core/orchestrator.py）
"""

import asyncio
import json
from collections.abc import AsyncIterator
from typing import Any

from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse

from app.core.events import AgentEvent
from app.logging_config import get_logger, set_trace_id
from app.models.schemas import ChatRequest, ChatResponse, ErrorResponse
from app.security.pii import redact

logger = get_logger(__name__)

router = APIRouter(prefix="/api/v1", tags=["chat"])


def _extract_answer(messages: list[dict[str, Any]]) -> str:
    """取最后一条"纯文本 assistant 回复"作为对外 answer（跳过工具动作）。"""
    for m in reversed(messages):
        if m.get("role") == "assistant" and not m.get("tool_calls"):
            return str(m.get("content", ""))
    return "抱歉，我暂时没能完成这个请求，请稍后再试或转人工客服。"


@router.post(
    "/chat",
    response_model=ChatResponse,
    responses={400: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
    summary="发送聊天消息",
    description="向客服 Agent 发送一条消息，返回回复。支持多轮对话（通过 session_id）。",
)
async def chat(body: ChatRequest, request: Request) -> ChatResponse:
    """Day 7：经 Orchestrator 走真实 Agent 管道（组装 → AgentLoop → 落库）。"""
    trace_id = set_trace_id()
    logger.info(
        "chat_request",
        message=redact(body.message)[:100],
        session_id=body.session_id,
        trace_id=trace_id,
    )

    orchestrator = request.app.state.orchestrator
    session_id = body.session_id or f"session-{trace_id}"
    result = await orchestrator.handle(body.message, session_id)
    return ChatResponse(
        session_id=session_id,
        answer=_extract_answer(result.messages),
        tool_calls_count=result.tool_calls_count,
        tokens_used=result.tokens_used,
        model=result.model,
    )


def _sse_frame(event: AgentEvent) -> str:
    """AgentEvent → SSE 帧（`event: <kind>` + `data: <JSON>` 两行 + 空行）。

    data 用 ensure_ascii=False 的 JSON 序列化：换行/emoji 都被转义成单行，
    SSE 的 data 字段不会因多行被打断。
    """
    payload = {
        "kind": event.kind,
        "session_id": event.session_id,
        "content": event.content,
        "data": event.data,
    }
    return f"event: {event.kind}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"


@router.post(
    "/chat/stream",
    summary="SSE 流式聊天",
    description="POST + SSE：请求经 Orchestrator，按 AgentEvent 逐事件下发（assistant / tool_call / tool_result / end）。",
)
async def chat_stream(body: ChatRequest, request: Request) -> StreamingResponse:
    """把 Orchestrator 的事件流接到 SSE：每次回复按 AgentEvent 逐帧下发。

    客户端可逐帧渲染 Agent 全过程（tool_call → tool_result → assistant → end），
    而不是一次性收到整段回复。每请求建一个 asyncio.Queue，emit 回调往里 put
    AgentEvent；后台任务跑完 handle()（含落库）后放 None 哨兵收尾；响应生成器
    从队列逐帧下发。失败不硬断连接，转一条 kind="error" 事件再收尾。
    """
    trace_id = set_trace_id()
    logger.info(
        "chat_stream_request",
        message=redact(body.message)[:100],
        session_id=body.session_id,
        trace_id=trace_id,
    )

    orchestrator = request.app.state.orchestrator
    session_id = body.session_id or f"session-{trace_id}"
    queue: asyncio.Queue[AgentEvent | None] = asyncio.Queue()

    async def emit(event: AgentEvent) -> None:
        event.trace_id = event.trace_id or trace_id
        await queue.put(event)

    async def produce() -> None:
        try:
            await orchestrator.handle(body.message, session_id, emit=emit)
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 — 流式中途失败不硬断，转 error 事件
            logger.error("chat_stream_failed", error=str(exc), trace_id=trace_id)
            await queue.put(
                AgentEvent(kind="error", session_id=session_id, content=str(exc))
            )
        finally:
            await queue.put(None)

    task = asyncio.create_task(produce())

    async def stream() -> AsyncIterator[str]:
        try:
            while True:
                event = await queue.get()
                if event is None:
                    break
                yield _sse_frame(event)
        finally:
            if not task.done():
                task.cancel()

    return StreamingResponse(
        stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
    )
