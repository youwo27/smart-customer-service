"""PII 检测脱敏 + 输出审核测试（Day 8 Part 5）。

覆盖：PII 类召回 > 90% + 正常文本零误伤 + 脱敏正确 + 输出审核（PII + 泄漏）。
"""

from app.security.output_guard import review_output
from app.security.pii import redact, redact_all, redact_secrets, scan

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


class TestRedactSecrets:
    """密钥脱敏（Day 9 Part 3）—— 与 PII 正交的一块：手机号是用户的隐私，key 是你的凭证。"""

    def test_sk_prefix_key(self) -> None:
        out = redact_secrets("用 sk-abcdefghijklmnop1234 调用")
        assert "sk-abcdefghijklmnop1234" not in out
        assert "***" in out

    def test_anthropic_style_key(self) -> None:
        out = redact_secrets("ANTHROPIC_API_KEY=sk-ant-api03-abcdefghijklmn")
        assert "sk-ant-api03-abcdefghijklmn" not in out

    def test_bearer_token(self) -> None:
        """认证头里的 token 没有固定前缀，靠 "Bearer " 这个上下文词兜住。"""
        out = redact_secrets("Authorization: Bearer abcdef1234567890")
        assert "abcdef1234567890" not in out
        assert "Bearer ***" in out  # 保留 scheme，只打码凭证本身

    def test_kv_style_secret(self) -> None:
        """kv 形式：**无前缀**的口令只有靠字段名 + 分隔符才拦得住。"""
        for text in ("api_key=hunter2secret", "password: Swordfish99", 'secret="abcdef123"'):
            assert "***" in redact_secrets(text), text

    def test_github_and_aws_keys(self) -> None:
        assert "ghp_" not in redact_secrets("token ghp_abcdefghijklmnopqrstuvwxyz0123")
        assert "AKIAIOSFODNN7EXAMPLE" not in redact_secrets("key AKIAIOSFODNN7EXAMPLE")

    def test_normal_text_untouched(self) -> None:
        """误伤检查：客服语料里没有密钥，一个字都不该动。"""
        for text in (
            "我的订单12345想退货",
            "运费15元谁承担",
            "token_in=3500 token_out=80",  # token **计数**不是凭证 —— 见下一条
            "tokens_used=330",
            "max_tokens=1024",
            "请用 secret 这个词造句",  # 单词本身没有 =/: 分隔符，不算赋值
        ):
            assert redact_secrets(text) == text, text

    def test_token_count_fields_are_not_secrets(self) -> None:
        """★ 最要紧的一条误伤防线：`token_in` / `input_tokens` 是今天刚接好的可观测性
        数据，被当成凭证打码等于把 Day 9 白做。字段名匹配用的是"词边界 + 必须紧跟 =/:"
        （`_` 是词字符，`token_in` 中间没有词边界），所以它们安全。"""
        for text in ("token_in=3500", "token_out=80", "input_tokens=4200", "token_total=7630"):
            assert "***" not in redact_secrets(text), text

    def test_empty_string(self) -> None:
        assert redact_secrets("") == ""

    def test_redact_all_covers_both_pii_and_secret(self) -> None:
        """日志出口用的统一入口：一次调用同时盖住 PII 和密钥。"""
        out = redact_all("联系13812345678，key=sk-abcdefghijklmnop")
        assert "13812345678" not in out
        assert "sk-abcdefghijklmnop" not in out

    def test_redact_all_leaves_trace_id_alone(self) -> None:
        assert redact_all("abc123def456abcd") == "abc123def456abcd"


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
