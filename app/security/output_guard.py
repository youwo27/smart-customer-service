"""输出审核 — Agent 回复流出前的最后一道闸（Day 8 实现）。

两件事（orchestrator 在 agent.run() 之后、_emit/落库之前调用）：
  1. PII 脱敏：别让别人的手机号顺着回答流出去（复用 pii.redact）。
  2. 系统提示词泄漏扫描：回答里若出现 system prompt 的特征片段，替换掉。

为什么放在"源头"（orchestrator 的出站口）而不是每条出口各写一遍：
  SSE / 落库 / 日志 / 审计……出口会越来越多，在生成处一次审核，所有下游自然安全。
"""

from dataclasses import dataclass, field

from app.security.pii import PIIHit, redact, scan

# system prompt 的可识别特征片段（被注入套出时，回答里会出现它们）。
# 取 SYSTEM_PROMPT 里最有辨识度的整句片段，避免误伤正常客服措辞。
SYSTEM_LEAK_MARKERS: list[str] = [
    "你是 MiniSupport 电商客服助手",
    "不要凭记忆编造",
    "先调用工具查知识库",
]

_LEAK_REPLACEMENT = "［已隐藏］"


@dataclass
class OutputReview:
    """一次输出审核的结果。"""

    text: str  # 审核/脱敏后的文本
    pii_hits: list[PIIHit] = field(default_factory=list)  # 命中的 PII（供审计）
    leaked_markers: list[str] = field(default_factory=list)  # 命中的泄漏特征词


def review_output(text: str, mask: str = "***") -> OutputReview:
    """输出审核：先扫 PII（记录命中），再脱敏 PII、替换 system prompt 泄漏片段。"""
    hits = scan(text)
    leaked = [m for m in SYSTEM_LEAK_MARKERS if m in text]

    cleaned = redact(text, mask=mask)
    for marker in leaked:
        cleaned = cleaned.replace(marker, _LEAK_REPLACEMENT)

    return OutputReview(text=cleaned, pii_hits=hits, leaked_markers=leaked)
