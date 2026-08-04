"""结构化日志配置 — 使用 structlog，告别 print 调试。

规则（见 AGENTS.md 3.2）：
- 每条日志自动包含 trace_id / session_id / module
- 禁止输出敏感信息（API Key、手机号、地址全文）
"""

import logging
import uuid
from contextvars import ContextVar

import structlog

# === 上下文变量（协程安全） ===
trace_id_ctx: ContextVar[str] = ContextVar("trace_id", default="")
session_id_ctx: ContextVar[str] = ContextVar("session_id", default="")


def set_trace_id(trace_id: str | None = None) -> str:
    """设置当前协程的 trace_id。如果不传则自动生成。"""
    tid = trace_id or uuid.uuid4().hex[:16]
    trace_id_ctx.set(tid)
    return tid


def get_trace_id() -> str:
    """获取当前协程的 trace_id。"""
    return trace_id_ctx.get()


def set_session_id(session_id: str) -> None:
    """设置当前协程的 session_id。"""
    session_id_ctx.set(session_id)


def get_session_id() -> str:
    """获取当前协程的 session_id。"""
    return session_id_ctx.get()


def setup_logging(log_level: str = "INFO") -> None:
    """初始化结构化日志系统。应用启动时调用一次。"""
    structlog.reset_defaults()

    shared_processors: list = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso", utc=False),
        structlog.dev.ConsoleRenderer(colors=True),
    ]

    structlog.configure(
        processors=shared_processors,
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(logging, log_level.upper(), logging.INFO)
        ),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )

    logging.basicConfig(
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        level=getattr(logging, log_level.upper(), logging.INFO),
    )


def get_logger(name: str | None = None) -> structlog.BoundLogger:
    """获取带模块名的 logger。

    用法:
        logger = get_logger(__name__)
        logger.info("agent_started", session_id="abc", model="sonnet")
    """
    return structlog.get_logger(name).bind(
        trace_id=get_trace_id(),
        session_id=get_session_id(),
    )
