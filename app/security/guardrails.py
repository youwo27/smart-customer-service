"""InputGuard — 输入防御管道：L1 规则先跑、L2 语义兜底。

组装（`check(message) -> GuardResult`）：
  ① L1（screen_by_rules，同步、零成本）先把大头滤掉——命中即 block，**不调 LLM**
     （省 token，也避免攻击文本被送进模型）。
  ② 全过再 L2（SemanticClassifier，异步 LLM 三分类）：
       attack      → block
       suspicious  → flag（不放行也不拦死，透传"有风险"提醒 / 走人工）
       normal      → allow
  ③ L2 调用失败/超时 → **fail-open 降级成 flag 而不是 block**——语义层是"加强"，
     丢了不该让客服整体瘫痪（但 L1 规则层仍在兜底）。

哪层 fail-closed、哪层 fail-open 的取舍，是今天第一个安全面试题：
    L1 规则 fail-closed（确定性，可用性受损无所谓、安全优先）；
    L2 语义 fail-open（可误判/可能挂，不该因一个分类器让客服不可用）。
"""

import asyncio
from typing import Any

from app.logging_config import get_logger
from app.security.input_guard import GuardResult, screen_by_rules

logger = get_logger(__name__)

# 对外客服话术：block 时对用户讲这个，**绝不披露命中哪条规则/具体 payload**。
# （讲太细等于教攻击者改 payload，细节只进审计/日志。）
BLOCK_REPLY = "我无法处理这个请求，请转人工客服。"


class InputGuard:
    """L1 规则 + L2 语义的守卫。classifier 可空 = 纯 L1 模式（测试/降级用）。"""

    def __init__(self, classifier: Any | None = None) -> None:
        """classifier：SemanticClassifier 或 None。None 时仅跑 L1（不调 LLM）。"""
        self._classifier = classifier

    async def check(self, message: str) -> GuardResult:
        """先 L1（同步），全过再 L2（异步）。返回 allow | block | flag。"""
        # ① L1：同步、零成本，命中即 block，不调 LLM
        rule = screen_by_rules(message)
        if rule.decision == "block":
            logger.info("guard_l1_block", matched=rule.matched_rule)
            return rule

        # ② 无 L2 → 只用 L1（软攻击会放过，属纯 L1 模式）
        if self._classifier is None:
            return rule

        # ③ L2：异步语义分类，独立超时由 SemanticClassifier 内部控制
        try:
            verdict = await self._classifier.classify(message)
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 — L2 失败降级 flag，不让客服瘫痪
            logger.warning("guard_l2_unavailable", error=str(exc))
            return GuardResult(decision="flag", reason=f"L2 分类器不可用，降级为 flag: {exc}")

        if verdict == "attack":
            logger.info("guard_l2_block", verdict=verdict)
            return GuardResult(decision="block", reason="L2 语义判定为攻击")
        if verdict == "suspicious":
            logger.info("guard_l2_flag", verdict=verdict)
            return GuardResult(decision="flag", reason="L2 语义判定为可疑")
        return GuardResult(decision="allow")
