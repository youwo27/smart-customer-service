"""Day 9 Part 1：结构化日志上下文串接测试。

覆盖两个真实 bug 的回归：
1. 值存进原生 ContextVar 后，structlog 的 `merge_contextvars` 扫不到（名字缺
   `structlog_` 前缀）→ 本方案改用我们自己的 `inject_request_context` processor，
   在打日志那一刻直接读 ContextVar，绕开前缀匹配。
2. `get_logger` 在模块顶层 `.bind()` 空值，请求来了也不更新 → 本方案不再静态 bind。

断言一律针对 event_dict（结构化数据），不断言渲染后的文本 —— 时间戳、颜色码、
绝对路径都不稳定，断言文本的测试一跑就 flaky。
"""

from typing import Any

import pytest
import structlog
from fastapi.testclient import TestClient
from structlog.testing import capture_logs

from app.core.agent import AgentRunResult
from app.logging_config import (
    get_logger,
    get_trace_id,
    inject_request_context,
    make_sample_processor,
    redact_processor,
    session_id_ctx,
    set_session_id,
    set_trace_id,
    setup_logging,
    trace_id_ctx,
)

# 模块级 logger —— 模拟 routes.py:21 / main.py:20 那种 import 时就建好的 logger。
# bug#2 的回归全靠它：老实现在这里 bind 了空串，之后请求来了也改不掉。
MODULE_LOGGER = get_logger(__name__)


@pytest.fixture(autouse=True)
def _clean_context() -> Any:
    """每个用例前后清空 trace/session 上下文，避免用例之间串味。"""
    t = trace_id_ctx.set("")
    s = session_id_ctx.set("")
    yield
    trace_id_ctx.reset(t)
    session_id_ctx.reset(s)


# === 单元：processor 本身 ===


def test_processor_injects_trace_and_session() -> None:
    """processor 在打日志那一刻读取当前上下文并注入。"""
    with capture_logs(processors=[inject_request_context]) as logs:
        set_trace_id("abc123")
        set_session_id("s-1")
        MODULE_LOGGER.info("agent_step")

    assert logs[0]["trace_id"] == "abc123"
    assert logs[0]["session_id"] == "s-1"


def test_module_level_logger_sees_later_context() -> None:
    """bug#2 回归：import 时就建好的 logger，请求中打的日志也要带当时的新值。

    老实现 `get_logger` 在 import 那刻 `.bind(trace_id=get_trace_id())` 绑的是空串，
    且 `merge_contextvars` 用 setdefault 不会覆盖 —— 这条断言在旧代码上必红。
    """
    with capture_logs(processors=[inject_request_context]) as logs:
        set_trace_id("t-later")
        MODULE_LOGGER.info("tool_executing", tool_name="read_document")

    assert logs[0]["trace_id"] == "t-later"
    assert logs[0]["tool_name"] == "read_document"


def test_explicit_value_wins_over_context() -> None:
    """调用方显式传了**非空** trace_id（routes.py 就是这么写的）优先，processor 不覆盖。"""
    with capture_logs(processors=[inject_request_context]) as logs:
        set_trace_id("from-context")
        get_logger("x").info("chat_request", trace_id="explicit")

    assert logs[0]["trace_id"] == "explicit"


def test_explicit_empty_value_falls_back_to_context() -> None:
    """显式传空串时用上下文兜底 —— 空值不携带信息，不能让它把好值挤掉。

    旧代码的指纹正是"显式传了个空 trace_id"（Day 8 前 get_logger 静态 bind 空串），
    setdefault 式的实现会把这个空串保住，本字段的"非空"约束就守不住了。
    """
    with capture_logs(processors=[inject_request_context]) as logs:
        set_trace_id("from-context")
        get_logger("x").info("chat_request", trace_id="")

    assert logs[0]["trace_id"] == "from-context"


def test_empty_context_still_emits_keys() -> None:
    """没有请求上下文时（如启动日志）字段仍在（空串），日志 schema 保持一致。"""
    with capture_logs(processors=[inject_request_context]) as logs:
        get_logger("x").info("app_starting")

    assert logs[0]["trace_id"] == ""
    assert logs[0]["session_id"] == ""


def test_set_trace_id_generates_when_absent() -> None:
    """不传参时自动生成 16 位 hex，且能被 get_trace_id 读回。"""
    tid = set_trace_id()

    assert len(tid) == 16
    assert get_trace_id() == tid


# === 单元：setup_logging 的接线 ===


def test_setup_logging_registers_inject_processor() -> None:
    """setup_logging 真的把 inject_request_context 挂进了 processor 链。

    只测 processor 本身不够 —— 它没被注册的话，真实日志照样不带 trace_id。
    """
    setup_logging("INFO")

    assert inject_request_context in structlog.get_config()["processors"]


def test_setup_logging_no_longer_relies_on_merge_contextvars() -> None:
    """确认已不再依赖 merge_contextvars（那条路对我们的 ContextVar 无效）。"""
    setup_logging("INFO")

    names = [getattr(p, "__name__", "") for p in structlog.get_config()["processors"]]
    assert "merge_contextvars" not in names


def test_get_logger_does_not_bind_empty_context() -> None:
    """bug#2 根因回归：get_logger 不能把空值焊到 logger 对象上。

    旧实现在这里 bind，于是 logger 自己带着 `trace_id=""`，processor 补不上。
    """
    logger = get_logger("bound-check")

    assert "trace_id" not in logger._context


# === 集成：真 HTTP 层走一遍，对齐验收红线① ===


class _LoggingOrchestrator:
    """假编排层：不发外部请求，只在被调用时打一条**不带 trace_id 参数**的日志。

    模拟真实 orchestrator/security 里的那些日志点 —— 它们不像 routes.py 那样显式传
    trace_id，只能靠上下文串下去。这正是本测试要验证的传导路径。
    """

    async def handle(self, message: str, session_id: str, emit: Any = None) -> AgentRunResult:
        _LOGGED_ORCHESTRATOR_LOGGER.info("orchestrator_turn", session_id=session_id, message=message)
        return AgentRunResult(
            messages=[{"role": "assistant", "content": "（假编排）ok"}],
            tool_calls_count=0,
            tokens_used=1,
            model="fake",
            finish_reason="stop",
        )


_LOGGED_ORCHESTRATOR_LOGGER = get_logger("app.core.orchestrator")


def test_http_request_logs_carry_nonempty_trace_id() -> None:
    """验收红线①：发一条消息，日志里 trace_id / session_id 都非空。

    用 TestClient 起真 HTTP 层 + 假编排层，不连 DeepSeek、不检索。
    """
    from app.main import app

    app.state.orchestrator = _LoggingOrchestrator()
    client = TestClient(app)

    with capture_logs(processors=[inject_request_context]) as logs:
        resp = client.post("/api/v1/chat", json={"message": "怎么退货？"})

    assert resp.status_code == 200

    by_event = {entry["event"]: entry for entry in logs}
    assert "chat_request" in by_event, f"没抓到 chat_request，实际抓到：{list(by_event)}"

    # 路由自己显式传了 trace_id
    assert by_event["chat_request"]["trace_id"]
    # 编排层那条没传 —— 全靠上下文传导，这才是本测试的重点
    assert by_event["orchestrator_turn"]["trace_id"]
    assert by_event["orchestrator_turn"]["session_id"]
    assert by_event["chat_request"]["trace_id"] == by_event["orchestrator_turn"]["trace_id"]


# === 日志出口脱敏（Part 3） ===
#
# 定位是"最后一道兜底"：Day 8 的源头脱敏依赖"每个写日志的地方都记得调 redact"，
# 人总会忘（Part 4 就在 query.py 里挖出过裸 print）。这里断言的是"出口拦得住"。


def _redact(event: str, **kw: Any) -> dict[str, Any]:
    """把 event_dict 喂给脱敏 processor，返回它吐出来的结果。"""
    return dict(redact_processor(None, "info", {"event": event, **kw}))


def test_redact_processor_masks_pii_in_string_fields() -> None:
    out = _redact("user_message", message="我的手机号是13812345678", note="订单12345")

    assert "13812345678" not in out["message"]
    assert out["note"] == "订单12345"  # 正常文本零误伤

def test_redact_processor_masks_secrets() -> None:
    out = _redact("llm_client_created", api_key="sk-abcdefghijklmnop1234")

    assert "sk-abcdefghijklmnop1234" not in out["api_key"]


def test_redact_processor_masks_sensitive_keys_wholesale() -> None:
    """字段名就是凭证时整个值打码 —— `api_key="hunter2secret"` 这种**无前缀**口令，
    光扫值的形态是扫不出来的，只有靠字段名。"""
    out = _redact("config_loaded", api_key="hunter2secret", password="Swordfish99", token="opaque-token")

    assert out["api_key"] == "***"
    assert out["password"] == "***"
    assert out["token"] == "***"


def test_redact_processor_does_not_mangle_token_counters() -> None:
    """★ 反向防线：`token_in` / `tokens_used` 是 token **计数**，不是凭证。
    把它们打码等于毁掉今天刚接好的可观测性。"""
    out = _redact("llm_usage", token_in=3500, token_out=80, tokens_used=330)

    assert out["token_in"] == 3500
    assert out["token_out"] == 80
    assert out["tokens_used"] == 330


def test_redact_processor_keeps_trace_id_intact() -> None:
    """trace_id 走豁免通道，哪怕它长得**正好**像银行卡。

    这不是假想：trace_id 是 uuid4().hex[:16]，全数字的概率约 0.04%，此时银行卡规则
    （16-19 位连续数字）会命中，把它打成 "***" —— 这条日志就再也串不回它的 trace 了，
    而且是静默的。session_id 不豁免（它可能原样带回客户端输入，真像手机号就该被打码）。
    """
    out = _redact("chat_request", trace_id="0123456789012345", session_id="13812345678")

    assert out["trace_id"] == "0123456789012345"
    assert out["session_id"] == "***"


def test_redact_processor_recurses_into_containers() -> None:
    """嵌套结构里的裸 PII 也要拦：只扫顶层的话，它躺在 payload 里照样落地。"""
    out = _redact(
        "tool_result",
        payload={"customer": {"phone": "13812345678"}, "tags": ["13812345678", 3]},
    )

    assert out["payload"]["customer"]["phone"] == "***"
    assert out["payload"]["tags"][0] == "***"
    assert out["payload"]["tags"][1] == 3  # 非字符串原样返回


def test_redact_processor_leaves_non_string_values_alone() -> None:
    out = _redact("counting", duration_ms=12.5, ok=True, count=3, nothing=None)

    assert (out["duration_ms"], out["ok"], out["count"], out["nothing"]) == (12.5, True, 3, None)


def test_setup_logging_registers_redact_processor() -> None:
    """脱敏得真的接进 processor 链 —— 光有函数不算数。"""
    setup_logging("INFO")

    processors = structlog.get_config()["processors"]
    assert any(getattr(p, "__name__", "") == "redact_processor" for p in processors)


def test_redaction_runs_after_sampling() -> None:
    """顺序：采样在前、脱敏在后 —— 被丢掉的日志根本不会落地，没必要为它跑正则。

    断言顺序而不是"更快"：顺序反了功能也对，只是白烧 CPU，这种退化不该无声发生。
    """
    setup_logging("INFO", sample_rate=0.1)

    names = [getattr(p, "__name__", "") for p in structlog.get_config()["processors"]]
    chain = [n for n in names if n in ("sample_processor", "redact_processor")]
    assert chain == ["sample_processor", "redact_processor"]


def test_real_chain_masks_before_landing() -> None:
    """在 setup_logging 装配出来的真实链路上跑一遍：日志出口真的把 PII 拦下来了。"""
    setup_logging("INFO", sample_rate=1.0)
    chain = [
        p
        for p in structlog.get_config()["processors"]
        if not isinstance(p, structlog.dev.ConsoleRenderer)
    ]
    set_trace_id("abc123")

    with capture_logs(processors=chain) as logs:
        get_logger("chain-probe").info("tool_result", message="客户电话13812345678")

    assert "13812345678" not in logs[0]["message"]
    assert logs[0]["trace_id"] == "abc123"  # 脱敏没把串联字段误伤掉


# === 采样（Part 2） ===


def _kept(processor: Any, method_name: str, event: str, **kw: Any) -> bool:
    """直接把 event_dict 喂给 processor，返回这条日志是否被保留。"""
    try:
        processor(None, method_name, {"event": event, **kw})
    except structlog.DropEvent:
        return False
    return True


def test_rate_1_keeps_everything() -> None:
    """rate=1.0 是本地调试档：一条都不丢。"""
    processor = make_sample_processor(1.0)

    for i in range(200):
        assert _kept(processor, "info", "chat_request", trace_id=f"{i:016x}")


def test_rate_0_drops_samplable_info() -> None:
    processor = make_sample_processor(0.0)

    assert not _kept(processor, "info", "chat_request", trace_id="abc")


def test_sampling_decides_once_per_request_not_per_line() -> None:
    """同一个 trace_id 的所有日志共享一个决定 —— 逐行随机会把一次请求打成碎片。

    这是采样最要紧的性质：留下 chat_request 却丢掉紧跟其后的 tool_executing，
    trace 就断了，比不采样还难查。
    """
    processor = make_sample_processor(0.5)
    events = ("chat_request", "tool_executing", "llm_call", "response_formatting")

    for i in range(50):
        tid = f"trace-{i:04x}"
        decisions = {_kept(processor, "info", ev, trace_id=tid) for ev in events}
        assert len(decisions) == 1, f"trace {tid} 被逐行拆散了：{decisions}"


def test_sampling_rate_is_approximately_honored() -> None:
    """10% 的桶要真的落在 10% 附近，不能是"几乎全留"或"几乎全丢"。"""
    processor = make_sample_processor(0.1)
    n = 5000
    kept = sum(
        _kept(processor, "info", "chat_request", trace_id=f"{i:016x}") for i in range(n)
    )

    assert 0.08 < kept / n < 0.12, f"实际保留率 {kept / n}"


def test_warning_and_error_are_never_sampled() -> None:
    """信号不能被采样掉：量小、每条都可能独立成因。"""
    processor = make_sample_processor(0.0)

    for method in ("warning", "error", "critical"):
        assert _kept(processor, method, "tool_failed", trace_id="abc"), method


def test_audit_is_never_sampled() -> None:
    """审计留痕一条都不能少（kind=guard_block|tool_denied|pii_redacted|...）。"""
    processor = make_sample_processor(0.0)

    assert _kept(processor, "info", "audit", trace_id="abc", kind="guard_block")


def test_guard_events_are_never_sampled() -> None:
    """orchestrator/guardrails 里的拦截事件同理。"""
    processor = make_sample_processor(0.0)

    for event in ("guard_blocked", "guard_l1_block", "guard_l2_block", "guard_l2_flag"):
        assert _kept(processor, "info", event, trace_id="abc"), event


def test_sample_force_escapes_sampling() -> None:
    """逃生舱：将来新增的事件不想被采样，打 sample_force=True 即可。"""
    processor = make_sample_processor(0.0)

    assert _kept(processor, "info", "whatever", trace_id="abc", sample_force=True)


def test_logs_without_trace_id_are_never_sampled() -> None:
    """启动/关闭日志没有请求上下文，量小且关键 → 全留。"""
    processor = make_sample_processor(0.0)

    assert _kept(processor, "info", "app_starting")
    assert _kept(processor, "info", "app_shutting_down", trace_id="")


def test_non_hex_trace_id_does_not_crash() -> None:
    """trace_id 未必是 uuid 生成的 hex，调用方可以塞任意字符串。

    所以定桶用哈希，而不是 int(trace_id[:8], 16) —— 后者遇到 "sec-block-1" 会 ValueError。
    """
    processor = make_sample_processor(0.5)

    for tid in ("sec-block-1", "会话-42", "x"):
        assert _kept(processor, "info", "chat_request", trace_id=tid) in (True, False)


def test_setup_logging_registers_sampler() -> None:
    """采样得真的接进 processor 链 —— 光有个 make_sample_processor 不算数。"""
    setup_logging("INFO", sample_rate=0.0)

    processors = structlog.get_config()["processors"]
    assert any(getattr(p, "__name__", "") == "sample_processor" for p in processors)


def test_real_chain_drops_info_but_keeps_audit_and_warning() -> None:
    """在 setup_logging 装配出来的真实链路上跑一遍，确认采样与豁免的实际效果。

    capture_logs 会替换整条链，所以这里把 ConsoleRenderer 摘掉，其余 processor 原样用。
    """
    setup_logging("INFO", sample_rate=0.0)
    chain = [
        p
        for p in structlog.get_config()["processors"]
        if not isinstance(p, structlog.dev.ConsoleRenderer)
    ]
    set_trace_id("abc123")

    with capture_logs(processors=chain) as logs:
        get_logger("chain-probe").info("chat_request", message="怎么退货")
        get_logger("chain-probe").info("audit", kind="guard_block")
        get_logger("chain-probe").warning("tool_failed", tool_name="query_order")

    events = [entry["event"] for entry in logs]
    assert "chat_request" not in events, "普通 INFO 在 rate=0 时应被丢弃"
    assert "audit" in events, "审计事件不能被采样掉"
    assert "tool_failed" in events, "WARNING 不能被采样掉"
