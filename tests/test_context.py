"""上下文组装 + 压缩单元测试 — mock LLM，不碰真实 API。

Day 7 Part 2 验收点：
  1. estimate_tokens：纯函数，长度单调
  2. assemble：System → 历史 → 当前 user 的分层；超阈值才触发压缩
  3. ContextCompressor：只压更早轮次、保最近原文；tool 结果若失去上下文则丢弃
"""

from typing import Any

from app.context.assembler import ContextAssembler, estimate_tokens
from app.context.compressor import COMPRESS_PROMPT, ContextCompressor
from app.llm.client import LLMResponse


def _resp(content: str) -> LLMResponse:
    return LLMResponse(content=content, model="mock", usage={"input_tokens": 0, "output_tokens": 0})


class FakeLLM:
    def __init__(self) -> None:
        self.calls: list[list[dict[str, Any]]] = []

    async def chat(self, messages: list[dict[str, Any]]) -> LLMResponse:
        self.calls.append(messages)
        return _resp("用户要退货；客服已告知需在 7 天内申请。")


class FakeCompressor:
    """记录是否被调用的假压缩器。"""

    def __init__(self) -> None:
        self.calls = 0

    async def compress(self, history: list[dict[str, Any]]) -> list[dict[str, Any]]:
        self.calls += 1
        return history


def _msg(role: str, content: str) -> dict[str, str]:
    return {"role": role, "content": content}


class TestEstimateTokens:
    def test_monotonic_with_length(self) -> None:
        short = estimate_tokens([_msg("user", "你好")])
        long = estimate_tokens([_msg("user", "你好" * 200)])
        assert long > short
        assert short > 0


class TestAssembler:
    async def test_layers_system_history_user(self) -> None:
        history = [_msg("assistant", "可以退货"), _msg("user", "那运费呢")]
        assembler = ContextAssembler(compressor=None)  # 不压缩

        out = await assembler.assemble("谁出运费", history)

        assert [m["role"] for m in out] == ["system", "assistant", "user", "user"]
        assert out[0]["content"] == assembler.system_prompt
        assert out[-1]["content"] == "谁出运费"

    async def test_compresses_only_when_over_threshold(self) -> None:
        compressor = FakeCompressor()
        assembler = ContextAssembler(compressor=compressor, threshold=1_000_000)  # 阈值极高

        await assembler.assemble("x", [_msg("user", "y")])
        assert compressor.calls == 0

        small = ContextAssembler(compressor=compressor, threshold=0)  # 阈值极低 → 必触发
        await small.assemble("x", [_msg("user", "y" * 100)])
        assert compressor.calls == 1


class TestCompressor:
    async def test_compresses_older_keeps_recent(self) -> None:
        llm = FakeLLM()
        comp = ContextCompressor(llm=llm, keep_recent=4)
        history = [_msg("user", f"问题{i}") if i % 2 == 0 else _msg("assistant", f"答复{i}") for i in range(10)]

        out = await comp.compress(history)

        # 1 条摘要 + 最近 4 条原文（< 原 10 条）
        assert len(out) == 5
        assert out[0]["role"] == "assistant"
        assert "历史摘要" in out[0]["content"]
        assert len(llm.calls) == 1
        assert COMPRESS_PROMPT.splitlines()[0] in llm.calls[0][0]["content"]

    async def test_short_history_no_compress(self) -> None:
        llm = FakeLLM()
        comp = ContextCompressor(llm=llm, keep_recent=5)
        history = [_msg("user", "a"), _msg("assistant", "b")]

        out = await comp.compress(history)

        assert out is history  # 原样返回，不发 LLM
        assert len(llm.calls) == 0

    async def test_drops_orphan_tool_results(self) -> None:
        llm = FakeLLM()
        comp = ContextCompressor(llm=llm, keep_recent=4)
        recent_with_tool_head = [
            _msg("tool", '{"ok": true}'),        # 它的发起者已被压进 older → 丢弃
            _msg("assistant", "查到了"),
            _msg("user", "多少钱"),
            _msg("assistant", "50 元"),
        ]
        history = [_msg("user", f"早{i}") for i in range(6)] + recent_with_tool_head

        out = await comp.compress(history)

        roles = [m["role"] for m in out]
        assert "tool" not in roles
        assert len(out) == 1 + 3  # 摘要 + 去掉 tool 后的最近 3 条
