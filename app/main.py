"""MiniSupport Agent — 电商客服应用入口。

启动方式:
    uvicorn app.main:app --reload
"""

from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api.routes import router
from app.config import settings
from app.logging_config import get_logger, get_trace_id, setup_logging

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理。"""
    setup_logging(settings.log_level)
    logger.info(
        "app_starting",
        host=settings.host,
        port=settings.port,
        model=settings.llm_model,
    )
    # TODO: Day 4 初始化 ToolRegistry
    # TODO: Day 4 初始化 VectorStore
    yield
    logger.info("app_shutting_down")
    # TODO: Day 4 清理连接池


app = FastAPI(
    title="MiniSupport Agent",
    description="电商客服 Agent —— 集成 RAG + Agentic RAG + 多 Agent 分流 + 人在回路",
    version="0.1.0",
    lifespan=lifespan,
)

# CORS（开发环境允许所有来源）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """全局异常处理：记录错误日志，返回统一格式。"""
    logger.error(
        "unhandled_exception",
        error_type=type(exc).__name__,
        error=str(exc),
        path=request.url.path,
    )
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "detail": str(exc) if settings.log_level == "DEBUG" else None,
            "trace_id": get_trace_id(),
        },
    )


app.include_router(router)


@app.get("/health", tags=["health"])
async def health_check() -> dict[str, str]:
    """健康检查 endpoint。"""
    return {"status": "ok", "version": "0.1.0"}


if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        reload=True,
        log_level=settings.log_level.lower(),
    )
