"""OpenTelemetry 全链路追踪（Day 9 Part 6）。

给 Agent 的每一步装上"有起止时间的树"：一次用户请求 = 一条 Trace，LLM 调用 / 工具执行 /
检索 = 挂在树上的 Span。日志告诉你"发生了什么"，追踪告诉你"这一步花了多久、挂在谁下面、
花了多少 token" —— 出问题时能在 Jaeger 里 30 秒定位到"是哪次检索慢、哪个工具挂了"。

四类埋点（都落在真实代码的唯一关口上，见调用方）：
    agent.request          orchestrator.handle   一次用户请求（含 guard 决策）
    └── context.assemble   ContextAssembler.assemble
    └── agent.run          AgentLoop.run         turns / tokens_used / finish_reason
        └── llm.chat_with_tools   LLMClient      model / token_in / token_out（每次尝试一个）
        └── tool.execute          SecuredToolRegistry  tool_name / status(ok|denied)
        └── rag.hybrid_search     Retriever       query / result_count
            ├── rag.dense_search
            └── rag.sparse_search

设计要点：
1. **埋点埋在"唯一关口"**：工具埋在 Day 8 的受控注册器（所有工具的唯一必经之路），
   而不是每个工具的 execute 里 —— 一处埋点，六个工具自动都有 span。
2. **属性只放标量**（model / token / count / status / 截断后的短文本），整段 messages
   绝不进 attribute —— trace 会爆（Day 9 常见坑 #5）。
3. **失败不影响业务**：装配失败（缺包、导不出）只记一条 warning，绝不阻断应用启动；
   span 里抛出的异常先 `record_exception` 再原样抛出，不吞异常改变业务语义。
4. **跨 await 自动串**：OTel 把"当前 span"存在 contextvars 里，`with span(...)` 里
   `await` 出去的子 span 会自动挂对父级，不需要手动传 trace 上下文。
"""

import socket
import time
from collections.abc import Generator
from contextlib import contextmanager
from typing import Any
from urllib.parse import urlparse

from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.trace import Span, Status, StatusCode, Tracer

from app.logging_config import get_logger
from app.security.pii import redact_all

logger = get_logger(__name__)

# span 里的文本属性上限 —— attribute 是给人扫的，不是日志全文（坑 #5）
ATTR_TEXT_LIMIT = 200

# 全局只装配一次：TracerProvider 是进程级的，重复 set 会打
# "Overriding of current TracerProvider is not allowed" 且旧 provider 的 processor 泄漏线程。
_provider: TracerProvider | None = None


def get_tracer() -> Tracer:
    """取本模块的 tracer。

    ★ 每次现取（而不是模块级缓存一个 tracer 对象）：全局 TracerProvider 是**延迟**生效的，
    在 setup_tracing() 之前拿到的 tracer 只是个代理。现取能保证 span 一定落在
    当前真正生效的 provider 上 —— 测试里临时换成内存 exporter 时尤其重要。
    """
    return trace.get_tracer(__name__)


def short_text(text: str, limit: int = ATTR_TEXT_LIMIT) -> str:
    """给 attribute 用的短文本：先脱敏（PII + 密钥）再截断。

    Jaeger 也是一条出口线（trace 会被存下来、被人看、被导出到别的系统），
    所以用户原话进 attribute 前同样要过一道脱敏 —— 和日志出口一个道理。
    """
    return redact_all(text)[:limit]


def _set_attrs(target: Span, attrs: dict[str, Any]) -> None:
    """把属性挂到 span 上。

    - None **跳过**：OTel 不允许 None，硬塞会打 warning 且属性丢失；
      "没拿到值"和"值就是空"在 trace 里没必要区分。
    - 非标量（dict/list）转成字符串再截断：trace 后端只认标量，塞大对象等于把
      trace 变成一个没人看得懂的 JSON blob（坑 #5）。
    """
    for key, value in attrs.items():
        if value is None:
            continue
        if isinstance(value, (bool, int, float, str)):
            target.set_attribute(key, value)
        else:
            target.set_attribute(key, str(value)[:ATTR_TEXT_LIMIT])


def set_attrs(**attrs: Any) -> None:
    """给**当前** span 挂属性（没有活跃 span 时是 no-op）。

    给"埋点入口拿不到 span 对象、但知道该记什么"的地方用：orchestrator 的守卫分支
    要记 guard 决策，而 span 是在 handle() 里开的 —— 让它去拿当前 span，不必把
    span 对象当参数一路往下传（传参会把每个中间层都污染成"得知道追踪存在"）。
    """
    _set_attrs(trace.get_current_span(), attrs)


@contextmanager
def span(name: str, **attrs: Any) -> Generator[Span, None, None]:
    """开一个 span 的上下文管理器：`with span("llm.call", model=...) as sp:`。

    行为：
    - 进入时把 attrs 挂上（None 自动跳过）；
    - 正常退出 → 自动补 `duration_ms`（span 自身的起止时间 Jaeger 也会画，但补成
      attribute 才**可查可聚合**：能按 duration_ms 排序找慢调用）；
    - 抛出异常 → record_exception + status=ERROR，**然后原样抛出**。
      观测层永远不能替业务做"吞掉异常"的决定 —— 那是 orchestrator / agent 的职责。

    ★ 只支持 `with`（同步）用法，但它兼容 await：OTel 把当前 span 存在 contextvars，
      `with span(...)` 块里 await 出去，子 span 会自动挂到它下面，不需要手动传上下文（坑 #4）。
      注意别把 span 开在 asyncio.create_task 外面又指望任务里挂对 —— 任务会复制一份
      上下文快照，任务里新开的 span 挂的是快照那一刻的父级。
    """
    started = time.perf_counter()
    with get_tracer().start_as_current_span(name) as sp:
        _set_attrs(sp, attrs)
        try:
            yield sp
        except Exception as exc:  # 兜住只是为了记进 span，记完必须原样抛出（观测层不改变控制流）
            sp.record_exception(exc)
            sp.set_status(Status(StatusCode.ERROR, str(exc)[:ATTR_TEXT_LIMIT]))
            raise
        finally:
            sp.set_attribute("duration_ms", round((time.perf_counter() - started) * 1000, 2))


def _build_sampler(rate: float) -> Any:
    """按采样率选 sampler（开发全采，生产按比例 —— 坑 #3）。

    - rate >= 1.0 → AlwaysOnSampler：本地调试要"每条都看得见"；
    - 否则 → ParentBased(TraceIdRatioBased(rate))：比例采样，但**已采样 trace 的子 span
      一律跟着采**。父决定子这点很关键 —— 否则一次请求可能只有一半的 span 被采到，
      树断在半路，比不采还难查。
    """
    from opentelemetry.sdk.trace.sampling import ALWAYS_ON, ParentBased, TraceIdRatioBased

    if rate >= 1.0:
        return ALWAYS_ON
    return ParentBased(root=TraceIdRatioBased(max(rate, 0.0)))


def _endpoint_reachable(endpoint: str, timeout: float = 0.3) -> bool:
    """探测 OTLP 端点是否连得上（**纯提示用**，不改变任何行为）。

    ★ 为什么值得探一次：Jaeger 没起时，SDK 会按批重试导出并刷
      "Transient error StatusCode.UNAVAILABLE ... retrying" + "Failed to export"。
      那是准确但难懂的日志 —— 一个"看不到 trace"的初学者对着它多半会怀疑自己埋点写错了。
      启动时先说清"是 Jaeger 没起"，比事后让他猜有用。

    探测失败也照样把 provider 装起来：Jaeger 后起（docker compose up jaeger）时
    导出会自然恢复，不用重启应用。
    """
    parsed = urlparse(endpoint)
    host = parsed.hostname or "localhost"
    port = parsed.port or 4317
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def setup_tracing(
    service_name: str = "minisupport-agent",
    endpoint: str = "http://localhost:4317",
    sample_rate: float = 1.0,
    enabled: bool = True,
) -> TracerProvider | None:
    """装配 TracerProvider + OTLP exporter（应用启动时调用一次）。

    返回生效的 provider（未启用/装配失败时返回 None，此时 `span()` 退化成 no-op，
    业务代码一行都不用改）。

    ★ 必须在 FastAPI 应用**启动之前**调用：instrument_fastapi 会往中间件栈里插东西，
      Starlette 在应用启动后再 add_middleware 会直接 RuntimeError —— 所以别把这段
      放进 lifespan（那是启动中）。
    ★ 导出失败不抛异常：Jaeger 没起、4317 不通时只是"看不到 trace"，
      绝不能让可观测性把应用本身拖挂（观测层是旁路，不是主链路）。
    """
    global _provider
    if not enabled:
        logger.info("tracing_disabled")  # 明确记一条：免得"看不到 trace"被误判成 bug
        return None
    if _provider is not None:
        return _provider

    try:
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
    except ImportError as exc:  # 依赖没装齐时只降级，不阻断启动
        logger.warning("tracing_unavailable", error=str(exc), hint="pip install opentelemetry-exporter-otlp")
        return None

    if not _endpoint_reachable(endpoint):
        logger.warning(
            "tracing_endpoint_unreachable",
            endpoint=endpoint,
            hint="Jaeger 没起？先 `docker compose up jaeger`；不打算看 trace 就设 OTEL_ENABLED=false，"
            "免得 SDK 反复重试导出刷 UNAVAILABLE 日志",
        )

    provider = TracerProvider(
        resource=Resource.create({"service.name": service_name}),
        sampler=_build_sampler(sample_rate),
    )
    # insecure=True：本地 Jaeger 的 4317 是明文 gRPC，不加 TLS
    provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(endpoint=endpoint, insecure=True)))
    trace.set_tracer_provider(provider)
    _provider = provider
    logger.info("tracing_enabled", service=service_name, endpoint=endpoint, sample_rate=sample_rate)
    return provider


def instrument_fastapi(app: Any, excluded_urls: str = "/health") -> None:
    """给 FastAPI 打自动埋点：每个 HTTP 请求一个 server span（trace 的根）。

    有了它，路由里不用写一行埋点代码，所有 span 自动挂在这条请求之下 —— 这就是
    "Context Propagation"：trace 上下文从 HTTP 层一路传到 handle → agent → client。

    excluded_urls：健康检查这类高频无信息量的端点排除掉，免得把 trace 列表刷满。
    exclude_spans=["receive", "send"]：ASGI 埋点默认还会为每次 http.send/receive
    各开一个 INTERNAL span（一次请求多出好几条 "GET /x http send" 这种）。它们不携带
    业务信息，却会把"LLM → 工具 → 检索"那棵树淹掉一半 —— 埋点的目的是看清主链路，
    不是为了塞满界面。
    """
    try:
        from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
    except ImportError as exc:
        logger.warning("fastapi_instrumentation_unavailable", error=str(exc))
        return
    FastAPIInstrumentor.instrument_app(
        app,
        excluded_urls=excluded_urls,
        exclude_spans=["receive", "send"],
    )
    logger.info("fastapi_instrumented", excluded_urls=excluded_urls)


def shutdown_tracing() -> None:
    """刷出缓冲区里的 span 并关掉后台线程（应用关闭时调用）。

    ★ 不能省：BatchSpanProcessor 是**批量**导出的，进程直接退出时最后一批
      （很可能就是出错前那几条最关键的）会随缓冲区一起丢掉。
    """
    global _provider
    if _provider is None:
        return
    _provider.shutdown()
    _provider = None
    logger.info("tracing_shutdown")
