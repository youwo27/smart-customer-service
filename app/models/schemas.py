"""API 请求/响应数据模型（Pydantic schema）。"""

from datetime import datetime

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    """聊天请求。"""

    message: str = Field(..., min_length=1, max_length=10000, description="用户输入消息")
    session_id: str | None = Field(None, description="会话 ID，不传则创建新会话")


class ChatResponse(BaseModel):
    """聊天响应。"""

    session_id: str
    answer: str
    tool_calls_count: int = 0
    tokens_used: int = 0
    model: str = Field("", description="实际使用的模型")
    created_at: datetime = Field(default_factory=datetime.now)


class ErrorResponse(BaseModel):
    """错误响应。"""

    error: str
    detail: str | None = None
    trace_id: str | None = None
