"""MiniSupport Agent — 电商客服应用入口。

启动方式:
    uvicorn app.main:app --reload
"""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api.routes import router
from app.config import settings
from app.core.orchestrator import Orchestrator
from app.logging_config import get_logger, get_trace_id, setup_logging

logger = get_logger(__name__)


def _build_orchestrator() -> Orchestrator:
    """Day 7/8：组装真管道（AgentLoop + 安全层 + 上下文 + 会话存储）的单例。

    工具注册是纯内存操作、LLM 客户端是惰性对象，这里只"组装"不真正发请求，
    真正的网络调用发生在每次请求的工具/LLM 执行时。
    """
    from app.context.assembler import ContextAssembler
    from app.context.compressor import ContextCompressor
    from app.core.agent import AgentLoop, AgentLoopConfig
    from app.llm.client import LLMClientFactory
    from app.security.audit import create_audit_log
    from app.security.classifier import build_semantic_classifier
    from app.security.guardrails import InputGuard
    from app.security.permissions import PermissionGuard
    from app.security.proxy import SecuredToolRegistry
    from app.storage.session_store import create_session_store
    from app.tools.builtin import register_defaults
    from app.tools.registry import ToolRegistry

    registry = register_defaults(ToolRegistry())
    client = LLMClientFactory.create()

    # Day 8：安全层——受控注册器（权限校验 + 审计）包住真注册器，Agent 完全无感
    audit = create_audit_log()
    secured = SecuredToolRegistry(registry, PermissionGuard(), audit)

    agent = AgentLoop(
        client=client,
        tool_registry=secured,
        config=AgentLoopConfig(
            max_turns=settings.max_turns,
            tool_timeout_seconds=settings.tool_timeout_seconds,
            llm_retry_max=settings.llm_retry_max,
            llm_retry_base_delay=settings.llm_retry_base_delay,
        ),
    )
    assembler = ContextAssembler(compressor=ContextCompressor(llm=client))
    store = create_session_store()

    # 输入守卫：开关开着才建 L2 语义分类器（fast model，惰性对象，不发请求）
    input_guard: InputGuard | None = None
    if settings.input_guard_enabled:
        input_guard = InputGuard(build_semantic_classifier())

    return Orchestrator(
        agent=agent,
        assembler=assembler,
        session_store=store,
        input_guard=input_guard,
        audit=audit,
    )


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """应用生命周期管理。"""
    setup_logging(settings.log_level)
    logger.info(
        "app_starting",
        host=settings.host,
        port=settings.port,
        model=settings.llm_model,
    )
    app.state.orchestrator = _build_orchestrator()  # Day 7：路由经它处理真实请求
    yield
    logger.info("app_shutting_down")


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
