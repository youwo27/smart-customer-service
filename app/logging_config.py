"""结构化日志配置 — 使用 structlog，告别 print 调试。

规则（见 AGENTS.md 3.2）：
- 每条日志自动包含 trace_id / session_id / module
- 禁止输出敏感信息（API Key、手机号、地址全文）—— 由 `redact_processor` 在出口强制兜底，
  不依赖"每个日志点都记得脱敏"（Day 9 Part 3）

上下文（trace_id / session_id）存在**本模块自己的 ContextVar** 里，由
`setup_logging` 注册的 `inject_request_context` processor 在**每条日志生成的那一刻**
读取并注入 —— 而不是在 `get_logger()` 里静态 bind。

为什么这么做（Day 9 Part 1 踩过的两个真坑）：
1. 不能指望 `structlog.contextvars.merge_contextvars` 自动读到我们的值：它遍历当前
   context 里所有 ContextVar，只挑名字以 `structlog_` 开头的（STRUCTLOG_KEY_PREFIX）。
   我们建的 `ContextVar("trace_id")` 名字没这个前缀，会被直接跳过 —— 值明明 set 进去了，
   日志里永远是空。
2. 不能把值 bind 在 `get_logger()` 里：模块级 `logger = get_logger(__name__)` 在
   **import 时**就执行，那时还没进任何请求，bind 进去的是空值；之后请求来了，logger
   对象上那个空值已经焊死（且 merge 用的是 setdefault，正确的值也盖不上）。

正确姿势：值用 ContextVar 存，注入交给 processor 在"打日志那一刻"做 —— 时序才对。
"""

import hashlib
import logging
import uuid
from contextvars import ContextVar
from typing import Any

import structlog
from structlog.typing import EventDict, Processor, WrappedLogger

from app.security.pii import redact_all, redact_secrets

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


def inject_request_context(
    logger: WrappedLogger, method_name: str, event_dict: EventDict
) -> EventDict:
    """在每条日志生成的那一刻，把当前 trace_id / session_id 注入 event_dict。

    ★ 必须是 processor，不能在 get_logger() 里 bind：processor 在"打日志那一刻"执行，
    能读到当时的上下文；get_logger 在 import 时执行，只会焊死一个空值。

    取值规则："**非空**的显式值优先，其余一律用当前上下文兜底"：
    - 显式传了非空值 → 尊重调用方（routes.py 就是显式传的）；
    - 没传 or 传了空串 → 用上下文里的值。

    这里刻意不用 `setdefault`：若调用方显式传了 `trace_id=""`（Day 8 之前的写法就是
    这样把空值焊死的），setdefault 会把这个空串保住 —— 而本字段的约束是"非空"。
    空值不携带信息，被上下文值覆盖只会更好。
    """
    if not event_dict.get("trace_id"):
        event_dict["trace_id"] = get_trace_id()
    if not event_dict.get("session_id"):
        event_dict["session_id"] = get_session_id()
    return event_dict


# === 日志采样（Day 9 Part 2） ===

# 这些事件 100% 保留 —— 采样是为了降噪，不是为了丢掉事故现场。
# 名单按仓库里**真实存在**的 INFO 级事件定（见 `grep -rn "logger.info(" app/`）：
#   audit    → security/audit.py 的审计留痕，携带 kind=guard_block|guard_flag|tool_call
#              |tool_denied|pii_redacted|leak_blocked —— 一个都不能少
#   guard_*  → core/orchestrator.py 的 guard_blocked、security/guardrails.py 的
#              guard_l1_block / guard_l2_block / guard_l2_flag
# 用前缀而非枚举，是为了新加 guard_l3_* 这类事件时不会再漏。
ALWAYS_KEEP_EVENTS: frozenset[str] = frozenset({"audit"})
ALWAYS_KEEP_PREFIXES: tuple[str, ...] = ("guard_",)

_SAMPLE_BUCKETS = 10_000


def _sample_bucket(trace_id: str) -> int:
    """把 trace_id 确定性地映射到 [0, 10000) —— 同一个 trace 永远落同一个桶。

    用哈希而不是 `int(trace_id[:8], 16)`：trace_id 未必是 uuid 生成的 hex，
    调用方可以塞任意字符串（测试里就有 "sec-block-1" 这种），int(...,16) 会炸。
    """
    digest = hashlib.blake2b(trace_id.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big") % _SAMPLE_BUCKETS


def _is_sampling_exempt(event_dict: EventDict) -> bool:
    """安全/审计事件，以及显式打了 sample_force 的，一律不采样。"""
    if event_dict.get("sample_force"):
        return True
    event = str(event_dict.get("event", ""))
    return event in ALWAYS_KEEP_EVENTS or event.startswith(ALWAYS_KEEP_PREFIXES)


def make_sample_processor(rate: float) -> Processor:
    """造一个采样 processor：正常 INFO 按 rate 保留，其余一律不碰。

    ★ 采样单位是**一次请求**（按 trace_id 定桶），不是"一行日志"。
      逐行 `random.random() > 0.1` 会把一次请求打成碎片 —— chat_request 留下了、
      紧接着的 tool_executing 却丢了，trace 断掉反而更难查。按 trace_id 定桶则整条
      链一起留、一起丢，且是**确定性**的：同一个 trace 在哪儿算结果都一样，既不需要
      缓存也不会泄漏内存（逐行随机还得为一堆 trace 记状态）。

    ★ 只可能丢 INFO，其余全部原样放行：
      - WARNING / ERROR / CRITICAL 是信号（量小、每条都可能独立成因）→ 一条不丢；
      - audit / guard_* 事故现场 → 一条不丢（名单见上）；
      - 带 `sample_force=True` 的 → 一条不丢（逃生舱，给将来的新事件用）；
      - 没有 trace_id 的（启动/关闭/脚本日志）→ 全留，量小且关键。

    rate=1.0 全留（本地调试用），rate=0.0 丢掉全部可采样的 INFO。
    """

    threshold = int(rate * _SAMPLE_BUCKETS)

    def sample_processor(
        logger: WrappedLogger, method_name: str, event_dict: EventDict
    ) -> EventDict:
        if method_name != "info":  # WARNING/ERROR 是信号，DEBUG 是你主动开的不采样
            return event_dict
        if _is_sampling_exempt(event_dict):
            return event_dict
        trace_id = str(event_dict.get("trace_id") or "")
        if not trace_id:
            return event_dict
        if _sample_bucket(trace_id) >= threshold:
            raise structlog.DropEvent
        return event_dict

    return sample_processor


# === 日志出口脱敏（Day 9 Part 3） ===
#
# 定位：**最后一道兜底**，不是主防线。Day 8 的脱敏在源头（写库 / 出站 / 审计 detail），
# 依赖"每个写日志的地方都记得调 redact"—— 人总会忘（Part 4 就在 rag/query.py 里挖出
# 过裸 print）。这里挂在日志出口：无论谁写的、写的什么，落地前统一过滤一遍。
# 两层是纵深防御，不是重复劳动。

# 这些 key 的值整体打码 —— 光靠正则扫"值"扫不到 `api_key=hunter2` 这种无前缀口令，
# 得靠字段名判断。★ 用**精确匹配**而不是前缀/包含匹配：字段名里带 token 的
# `token_in` / `token_out` / `tokens_used` 是 token **计数**，不是凭证，误伤它们等于
# 毁掉今天刚接好的可观测性。
SENSITIVE_KEYS: frozenset[str] = frozenset(
    {
        "api_key",
        "apikey",
        "access_token",
        "auth_token",
        "refresh_token",
        "client_secret",
        "secret",
        "password",
        "passwd",
        "pwd",
        "token",
        "authorization",
    }
)

_REDACTED = "***"
_MAX_DEPTH = 6

# 这些 key 的值跳过 PII 扫描（仍然过密钥规则）。
#
# ★ 为什么单保 trace_id：它是**服务端生成**的 16 位 hex，唯一用途就是串联一次请求的
#   所有日志。而 PII 规则里的银行卡是"16-19 位连续数字"—— uuid4().hex[:16] 恰好全是
#   数字的概率约 (10/16)^16 ≈ 0.04%，一旦命中，这条日志的 trace_id 会被打码成 "***"，
#   整个 trace 的串联能力当场断掉，而且是**静默**的（没人会发现少了一条链）。
#   代价是"若调用方故意把手机号当 trace_id 传"会漏过 —— 但 trace_id 是服务端 set 的，
#   不来自用户输入；用户输入走的是 session_id（**不保护**，客户端的怪异输入照样被拦）。
PII_EXEMPT_KEYS: frozenset[str] = frozenset({"trace_id"})


def _is_sensitive_key(key: Any) -> bool:
    """字段名是否表示"值是凭证"。`api-key` / `API_KEY` / `apiKey` 归一化后都命中。"""
    if not isinstance(key, str):
        return False
    return key.strip().lower().replace("-", "_") in SENSITIVE_KEYS


def _normalize_key(key: Any) -> str:
    return key.strip().lower().replace("-", "_") if isinstance(key, str) else ""


def _redact_value(value: Any, depth: int = 0, *, pii_exempt: bool = False) -> Any:
    """递归脱敏：字符串过 PII + 密钥规则，容器逐项递归，其余类型原样返回。

    为什么递归容器：`logger.info("tool_result", payload={"user": "13812345678"})` 里的
    裸手机号躺在嵌套 dict 里，只扫顶层的话它会大摇大摆地落地。
    为什么限深：日志字段理论上可以是任意对象图，加个上限避免在环形/超深结构上转圈。
    pii_exempt：跳过 PII 规则、只过密钥规则（trace_id 这类"长得像 PII 的自身标识"）。
    """
    if isinstance(value, str):
        if not value:
            return value
        return redact_secrets(value) if pii_exempt else redact_all(value)
    if depth >= _MAX_DEPTH:
        return value
    if isinstance(value, dict):
        return {
            k: _REDACTED
            if _is_sensitive_key(k)
            else _redact_value(v, depth + 1, pii_exempt=pii_exempt or _normalize_key(k) in PII_EXEMPT_KEYS)
            for k, v in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_redact_value(v, depth + 1, pii_exempt=pii_exempt) for v in value]
    return value


def redact_processor(
    logger: WrappedLogger, method_name: str, event_dict: EventDict
) -> EventDict:
    """日志落地前，把 event_dict 里的字符串统一过一遍 PII + 密钥脱敏。

    - `message` 和任意字符串字段 → `redact`（手机号/身份证/邮箱/银行卡）+ `redact_secrets`
    - 字段名命中 SENSITIVE_KEYS → 整个值打码（不依赖值的形态，`api_key="hunter2"` 这种
      无前缀口令只有靠字段名才拦得住）
    - 嵌套 dict / list 递归处理（深度上限见 _MAX_DEPTH）

    ★ 只重建**确实命中**的字符串：`redact` / `redact_secrets` 无命中时原样返回输入，
      所以绝大多数日志（含 trace_id 这种 hex）不会多一次字符串分配。
    ★ trace_id 走"只过密钥规则"的豁免通道（见 PII_EXEMPT_KEYS），保住日志串联能力；
      session_id 不豁免 —— 它可能原样带回客户端输入，真像手机号就该被打码。
    """
    for key, value in event_dict.items():
        if _is_sensitive_key(key):
            event_dict[key] = _REDACTED
        else:
            event_dict[key] = _redact_value(
                value, pii_exempt=_normalize_key(key) in PII_EXEMPT_KEYS
            )
    return event_dict


def setup_logging(log_level: str = "INFO", sample_rate: float = 0.1) -> None:
    """初始化结构化日志系统。应用启动时调用一次。

    sample_rate：正常 INFO 日志的保留比例（1.0 全留，本地调试用；生产 0.1）。
    错误与安全审计事件不受它影响，始终 100%（见 make_sample_processor）。
    """
    structlog.reset_defaults()

    shared_processors: list[Processor] = [
        inject_request_context,  # 1. 先注入 trace_id / session_id
        make_sample_processor(sample_rate),  # 2. 再采样（依赖上一步的 trace_id）
        # 3. 脱敏放在采样**之后**：被采样丢掉的日志根本不会落地，没必要为它跑一整套正则。
        #    顺序反了功能没错，只是白烧 CPU —— 高频 INFO 有 90% 是要丢的。
        redact_processor,
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


def get_logger(name: str | None = None) -> Any:
    """获取带模块名的 logger。

    ★ 这里**不再** `.bind(trace_id=..., session_id=...)`：模块级 logger 在 import 时
    执行，那时没有请求上下文，bind 进去的是空值且此后不会更新。这两个字段由
    `inject_request_context` processor 在打日志时注入。

    用法:
        logger = get_logger(__name__)
        logger.info("agent_started", session_id="abc", model="sonnet")
    """
    return structlog.get_logger(name)
