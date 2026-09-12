r"""PII 检测与脱敏 — 手机号 / 身份证 / 邮箱 / 银行卡（Day 8 实现）。

设计：返回 (脱敏文本, 命中列表)，命中列表供审计用。
规则必须带"边界"防止误伤：订单号"12345"不是 PII，11 位手机号才是。

接入点（"数据进边界就脱敏，别等写日志时再想"，见 AGENTS 3.2）：
  1. 审计/日志前：guard reason、工具参数摘要、审计 detail 都不带裸手机号
  2. Agent 回复前（输出审核，output_guard.py）：新增 assistant 文本过一遍
  3. 保存历史前：存进 SessionStore 的 messages 也要是脱敏后的

为什么在"写入前"而不是 structlog 出口过滤：结构化日志是"出口"，出口过滤等于每条 log
都要背着整个 PII 引擎跑、且总有新出口漏过去（文件/DB/审计/SSE……）。在源头一次脱敏，
所有下游自然安全。
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
