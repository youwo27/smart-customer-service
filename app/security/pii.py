r"""PII 检测与脱敏 — 手机号 / 身份证 / 邮箱 / 银行卡（Day 8 实现）+ 密钥（Day 9 Part 3）。

设计：返回 (脱敏文本, 命中列表)，命中列表供审计用。
规则必须带"边界"防止误伤：订单号"12345"不是 PII，11 位手机号才是。

接入点（"数据进边界就脱敏，别等写日志时再想"，见 AGENTS 3.2）：
  1. 审计/日志前：guard reason、工具参数摘要、审计 detail 都不带裸手机号
  2. Agent 回复前（输出审核，output_guard.py）：新增 assistant 文本过一遍
  3. 保存历史前：存进 SessionStore 的 messages 也要是脱敏后的
  4. 日志出口兜底（logging_config.redact_processor，Day 9 Part 3）：
     `redact` + `redact_secrets` 一起挂在 processor 链上，落地前统一过滤

为什么源头脱敏之外还要有日志出口兜底：源头脱敏依赖"每个写日志的地方都记得调 redact"，
人总会忘（Day 9 Part 4 就在 app/rag/query.py 里挖出过裸 print）。出口 processor 是
"无论谁写的、写的什么，落地前统一过滤"的不依赖自觉的网。两层是纵深防御，不冲突。
"""

import re
from dataclasses import dataclass

# 规则必须带"边界"（(?<!\d)/(?!\d)）防止误伤：订单号"12345"不是 PII，11 位手机号才是。
PHONE = re.compile(r"(?<!\d)1[3-9]\d{9}(?!\d)")  # 大陆手机号
EMAIL = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
ID_CARD = re.compile(r"(?<!\d)\d{17}[\dXx](?!\d)")  # 18 位身份证（含校验位）
BANK_CARD = re.compile(r"(?<!\d)\d{16,19}(?!\d)")  # 银行卡

# 顺序即优先级：先具体后宽泛（身份证 18 位先于银行卡 16-19 位），重叠时保留先命中的。
_RULES: list[tuple[str, re.Pattern[str]]] = [
    ("phone", PHONE),
    ("email", EMAIL),
    ("id_card", ID_CARD),
    ("bank_card", BANK_CARD),
]


@dataclass
class PIIHit:
    """一次 PII 命中。start/end 是原文下标；masked 是脱敏预览（供审计，不含裸值）。"""

    type: str
    start: int
    end: int
    masked: str


def _partial_mask(ptype: str, value: str) -> str:
    """生成"局部打码"预览：可辨认是哪条数据，但不含完整裸值。"""
    if ptype == "phone":
        return f"{value[:3]}****{value[-2:]}"
    if ptype == "id_card":
        return f"{value[:4]}**********{value[-2:]}"
    if ptype == "bank_card":
        return f"{value[:4]}****{value[-4:]}"
    if ptype == "email":
        local, _, domain = value.partition("@")
        return f"{local[:1]}***@{domain}"
    return "***"


def scan(text: str) -> list[PIIHit]:
    """纯函数：找出所有 PII 命中（不重叠，按出现顺序）。

    重叠消解：身份证 / 银行卡都是长数字，可能落在同一段；先命中者保留（规则表中
    身份证在前），后面的同区段命中跳过，避免重复打码。
    """
    hits: list[PIIHit] = []
    occupied: list[tuple[int, int]] = []
    for ptype, regex in _RULES:
        for m in regex.finditer(text):
            start, end = m.span()
            if any(start < oe and os_ < end for os_, oe in occupied):
                continue  # 与已命中区段重叠 → 跳过（先到先得）
            occupied.append((start, end))
            hits.append(
                PIIHit(type=ptype, start=start, end=end, masked=_partial_mask(ptype, m.group()))
            )
    hits.sort(key=lambda h: h.start)
    return hits


def redact(text: str, mask: str = "***") -> str:
    """把命中的 PII 替换为 mask（默认 ***）。纯函数：先 scan 再按区间拼接。"""
    hits = scan(text)
    if not hits:
        return text
    out: list[str] = []
    last = 0
    for h in hits:
        out.append(text[last : h.start])
        out.append(mask)
        last = h.end
    out.append(text[last:])
    return "".join(out)


# === 密钥脱敏（Day 9 Part 3） ===
#
# 与 PII 正交的一块：手机号是"用户的隐私"，API Key 是"你的凭证"。
# 日志里出现 sk-xxx 等于把账号交出去，而 PII 规则（手机/身份证/邮箱/银行卡）一条都盖不住它。
#
# 只做"替换"不做"扫描命中"：密钥的形态不像手机号那样需要局部打码供人工辨认
# （没人需要从日志里认出是哪把 key），全量 *** 就够，所以不引入 Scan 那套结构。

# 1) 前缀式密钥 —— 有固定前缀的几乎零误伤，可以直接整段替换。
#    sk- 覆盖 OpenAI / DeepSeek / Anthropic（sk-ant-...）；
#    其余几个是常见云厂商的 key 格式，顺手一起拦。
SECRET_PREFIX = re.compile(
    r"\bsk-[A-Za-z0-9_\-]{8,}"  # OpenAI / DeepSeek / Anthropic(sk-ant-api03-...)
    r"|\b(?:ghp|gho|ghu|ghs|ghr)_[A-Za-z0-9]{20,}"  # GitHub token
    r"|\bAKIA[0-9A-Z]{16}\b"  # AWS Access Key ID
    r"|\bAIza[0-9A-Za-z_\-]{35}\b"  # Google API Key
)

# 2) 认证头里的 Bearer token —— 形态无固定前缀，靠"Bearer "这个上下文词兜住。
#    token 部分限 [A-Za-z0-9._\-]：真的凭证不会带空格/引号，收窄字符集能少误伤散文。
BEARER = re.compile(r"(?i)(\bbearer\s+)([A-Za-z0-9._\-]{8,})")

# 3) key=value / key: value 形式 —— 覆盖 .env、URL query、报错回显里的凭证。
#    ★ 正则刻意不宽：\b 卡在字段名左侧，`max_tokens` / `input_tokens` / `token_in`
#      这类字段名里的 token 不会被当成凭证字段（`_` 是词字符，中间没有词边界）；
#      值要求 ≥6 字符（真凭证没有 3 位以下的短口令，能挡掉 token=ok 这种状态值）。
#    ★ 这里**不含** session_id / trace_id —— 它们不是凭证，被误伤会让日志失去串联能力
#      （那正是 Day 9 Part 1 刚修好的东西）。
KV_SECRET = re.compile(
    r"(?i)\b(api[_-]?key|apikey|access[_-]?token|auth[_-]?token|refresh[_-]?token"
    r"|client[_-]?secret|secret|password|passwd|pwd|token)"
    r"(\s*[=:]\s*)([\"']?)([^\s\"',;}]{6,})"
)


def redact_secrets(text: str, mask: str = "***") -> str:
    """把文本里的 API Key / 凭证替换为 mask。纯函数，与 `redact`（PII）正交。

    顺序有讲究：先替换有固定前缀的（sk-xxx 形态最明确、误伤最低），再处理 Bearer，
    最后处理 kv 形式。反过来也行，但先干掉 sk-xxx 能让后两条规则面对更干净的文本
    （例如 `token=sk-xxx` 会先变成 `token=***`，kv 规则的值长度要求就不会把它再啃一遍）。

    返回替换后的文本；不含密钥时原样返回（无命中不重建字符串）。
    """
    if not text:
        return text
    out = SECRET_PREFIX.sub(mask, text)
    out = BEARER.sub(lambda m: m.group(1) + mask, out)
    out = KV_SECRET.sub(lambda m: f"{m.group(1)}{m.group(2)}{m.group(3)}{mask}", out)
    return out


def redact_all(text: str, mask: str = "***") -> str:
    """PII + 密钥一把过 —— "最后一道网"（日志出口 / trace attribute）统一走这个入口。

    顺序定死（先 PII 后密钥）只是为了让调用方不必纠结"该先调哪个"；两边规则本来不重叠
    （手机号不是 `sk-` 开头，key=value 里也不会出现手机号）。
    """
    return redact_secrets(redact(text, mask), mask)


