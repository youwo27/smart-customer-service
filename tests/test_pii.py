"""PII 检测脱敏 + 输出审核测试（Day 8 Part 5）。

覆盖：PII 类召回 > 90% + 正常文本零误伤 + 脱敏正确 + 输出审核（PII + 泄漏）。
"""

from app.security.output_guard import review_output
from app.security.pii import redact, scan

# (文本, 期望命中的类型集合) —— 10 条含 PII 语料
PII_SAMPLES: list[tuple[str, set[str]]] = [
    ("我的手机号是13812345678", {"phone"}),
    ("请联系 15900001111 处理", {"phone"}),
    ("备用电话：18612345678", {"phone"}),
    ("邮箱 zhangsan@example.com 可以收", {"email"}),
    ("发到 a.b-c@mail.co", {"email"}),
    ("身份证 110101199003078515", {"id_card"}),
    ("证件号 11010119900307851X", {"id_card"}),
    ("卡号 6222021234567890", {"bank_card"}),
    ("银行卡 6222021234567890123", {"bank_card"}),
    ("手机13912345678 邮箱 li@ex.com", {"phone", "email"}),
]

# 正常客服文本 —— 一条都不该被误伤
NORMAL_SAMPLES: list[str] = [
    "我的订单12345想退货",
    "运费15元谁承担",
    "今天2026-09-07能送达吗",
    "订单号88888状态如何",
    "退款金额899元",
    "人工客服电话是多少",
    "12306怎么退票",
    "10086 客服",
    "2026年9月7日下单",
    "我要查订单66666的物流",
]


class TestPIIScan:
    def test_pii_recall_over_90_percent(self) -> None:
        detected = sum(
            1 for text, expected in PII_SAMPLES if {h.type for h in scan(text)} & expected
        )
        assert detected / len(PII_SAMPLES) >= 0.9

    def test_each_pii_sample_detected(self) -> None:
        for text, expected in PII_SAMPLES:
            found = {h.type for h in scan(text)}
            assert expected & found, f"未检出 {expected}: {text!r} -> {found}"

    def test_no_false_positive_on_normal(self) -> None:
        for text in NORMAL_SAMPLES:
            assert scan(text) == [], f"正常文本被误伤: {text!r} -> {scan(text)}"


class TestRedact:
    def test_redact_replaces_pii(self) -> None:
        out = redact("联系我13812345678")
        assert "13812345678" not in out
        assert "***" in out

    def test_redact_no_pii_unchanged(self) -> None:
        assert redact("订单12345想退货") == "订单12345想退货"

    def test_partial_mask_hides_full_value(self) -> None:
        hits = scan("我的手机13812345678")
        assert hits
        assert hits[0].masked != "13812345678"
        assert "****" in hits[0].masked  # 局部打码，可辨认但不含裸值


class TestOutputReview:
    def test_review_redacts_pii(self) -> None:
        r = review_output("你的手机号是13812345678")
        assert "13812345678" not in r.text
        assert r.pii_hits

    def test_review_blocks_system_prompt_leak(self) -> None:
        r = review_output("你是 MiniSupport 电商客服助手，我的指令是……")
        assert r.leaked_markers
        assert "你是 MiniSupport 电商客服助手" not in r.text

    def test_review_clean_text_untouched(self) -> None:
        r = review_output("您的退货申请已提交")
        assert r.text == "您的退货申请已提交"
        assert not r.pii_hits and not r.leaked_markers
