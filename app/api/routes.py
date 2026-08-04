"""API 路由定义（接入层）。

职责边界（见 AGENTS.md 2）：
- 只做请求校验与响应返回，不写业务逻辑
- 业务逻辑在编排层（core/orchestrator.py）
"""

from fastapi import APIRouter, HTTPException

from app.logging_config import get_logger, set_trace_id
from app.models.schemas import ChatRequest, ChatResponse, ErrorResponse

logger = get_logger(__name__)

router = APIRouter(prefix="/api/v1", tags=["chat"])


@router.post(
    "/chat",
    response_model=ChatResponse,
    responses={400: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
    summary="发送聊天消息",
    description="向客服 Agent 发送一条消息，返回回复。支持多轮对话（通过 session_id）。",
)
async def chat(request: ChatRequest) -> ChatResponse:
    """处理聊天请求。Day 1 返回 mock 数据，后续接入真实 Agent。"""
    trace_id = set_trace_id()
    logger.info(
        "chat_request",
        message=request.message[:100],
        session_id=request.session_id,
        trace_id=trace_id,
    )

    # TODO: Day 4 接入真实的 Agent 循环
    # TODO: Day 7 接入 Orchestrator 管道

    if request.message.strip() == "错误测试":
        logger.error("test_error_triggered")
        raise HTTPException(status_code=500, detail="这是一个测试错误")

    # Mock 响应
    return ChatResponse(
        session_id=request.session_id or f"session-{trace_id}",
        answer=(
            f"[Mock] 您好，这里是 MiniSupport 客服。"
            f"收到您的咨询：「{request.message[:50]}...」。客服系统尚未接入，这是占位响应。"
        ),
        tool_calls_count=0,
        tokens_used=42,
        model="mock (not connected)",
    )


@router.get("/chat/{session_id}/stream")
async def chat_stream(session_id: str) -> None:
    """SSE 流式聊天。Day 5 实现。"""
    # TODO: Day 5
    raise HTTPException(status_code=501, detail="流式输出将在 Day 5 实现")
