"""AgentLoop 单元测试 — 用 MockLLM 验证循环逻辑（不碰真实 LLM）。

覆盖 Day 4 文档 Part 6 要求的三个场景：
  1. LLM 返回 tool_calls → 正确执行工具 → 回填 tool 消息 → 继续 → 最终回答
  2. 工具抛异常 → 循环不死，错误进 tool_result
  3. 到 max_turns 强制终止 → finish_reason 记 "max_turns"
"""

import asyncio

from app.core.agent import AgentLoop, AgentLoopConfig
from app.llm.client import LLMResponse, ToolCall
from app.tools.registry import ToolRegistry


class MockLLM:
    """按剧本返回响应的假 LLM。"""

    def __init__(self, script: list[LLMResponse]) -> None:
        self.script = list(script)
        self.calls: list[list[dict]] = []

    async def chat_with_tools(self, messages: list[dict], tools: list[dict]) -> LLMResponse:
        # 存快照（AgentLoop 会就地改 messages，不能存引用）
        self.calls.append(list(messages))
        if not self.script:
            raise AssertionError("MockLLM 剧本用完了")
        return self.script.pop(0)


class QueryOrderTool:
    """和 customer_service 同款的最小 mock 工具（避免依赖真实 mock 数据）。"""

    name = "query_order"
    description = "查询订单状态"
    parameters = {
        "type": "object",
        "properties": {"order_id": {"type": "string"}},
        "required": ["order_id"],
    }

    async def execute(self, **kwargs) -> dict:
        return {"order_id": kwargs.get("order_id"), "status": "已发货"}


class ExplodingTool:
    """故意抛异常的工具，测试兜底。"""

    name = "explode"
    description = "故意抛异常"
    parameters = {"type": "object", "properties": {}}

    async def execute(self, **kwargs) -> dict:
        raise RuntimeError("数据库挂了")


def _tool_response(name: str = "query_order", args: dict | None = None) -> LLMResponse:
    """构造一个返回 tool_calls 的响应。"""
    return LLMResponse(
        content="",
        model="mock",
        usage={"input_tokens": 10, "output_tokens": 5},
        finish_reason="tool_use",
        tool_calls=[ToolCall(id="call_1", name=name, arguments=args or {"order_id": "12345"})],
    )


def _final_response(content: str = "订单已发货") -> LLMResponse:
    """构造一个最终回答的响应。"""
    return LLMResponse(
        content=content,
        model="mock",
        usage={"input_tokens": 10, "output_tokens": 5},
        finish_reason="end_turn",
        tool_calls=[],
    )


class TestAgentLoopRun:
    """正常路径。"""

    async def test_tool_call_then_final_answer(self) -> None:
        """tool_calls → 执行 → 回填 → 再调 LLM → 最终回答。"""
        llm = MockLLM([_tool_response(), _final_response()])
        reg = ToolRegistry()
        reg.register(QueryOrderTool())
        agent = AgentLoop(client=llm, tool_registry=reg, config=AgentLoopConfig(max_turns=5))

        messages: list[dict] = [{"role": "user", "content": "查订单"}]
        result = await agent.run(messages)

        assert result.tool_calls_count == 1
        assert result.finish_reason == "end_turn"
        assert result.messages[-1]["role"] == "assistant"
        assert result.messages[-1]["content"] == "订单已发货"
        # 中间应有一条 tool 结果
        assert any(m["role"] == "tool" and "已发货" in m["content"] for m in result.messages)
        # 统计：两轮调用，input+output 各 15
        assert result.tokens_used == 30

    async def test_no_tool_call_direct_answer(self) -> None:
        """模型不调工具 → 一轮结束。"""
        llm = MockLLM([_final_response("你好")])
        agent = AgentLoop(client=llm, tool_registry=ToolRegistry(), config=AgentLoopConfig())
        result = await agent.run([{"role": "user", "content": "你好"}])
        assert result.tool_calls_count == 0
        assert result.finish_reason == "end_turn"
        assert result.messages[-1]["content"] == "你好"

    async def test_assistant_tool_calls_format(self) -> None:
        """assistant 消息的 tool_calls 字段格式必须和 API 返回一致。"""
        llm = MockLLM([_tool_response(), _final_response()])
        reg = ToolRegistry()
        reg.register(QueryOrderTool())
        agent = AgentLoop(client=llm, tool_registry=reg, config=AgentLoopConfig(max_turns=5))
        await agent.run([{"role": "user", "content": "查订单"}])

        # 第二次调 LLM 时的消息结构（快照）：user → assistant(tool_calls) → tool
        assistant_msg = llm.calls[1][-2]
        assert assistant_msg["role"] == "assistant"
        tc = assistant_msg["tool_calls"][0]
        assert tc["type"] == "function"
        assert tc["function"]["name"] == "query_order"
        # 且 assistant 后面跟着一条 tool 消息，tool_call_id 对得上
        tool_msg = llm.calls[1][-1]
        assert tool_msg["role"] == "tool"
        assert tool_msg["tool_call_id"] == "call_1"


class TestAgentLoopFailure:
    """失败路径。"""

    async def test_tool_exception_does_not_kill_loop(self) -> None:
        """工具抛异常 → 循环不死，错误进 tool_result，agent 还能继续。"""
        llm = MockLLM([_tool_response(name="explode"), _final_response("抱歉，出错了")])
        reg = ToolRegistry()
        reg.register(ExplodingTool())
        agent = AgentLoop(client=llm, tool_registry=reg, config=AgentLoopConfig(max_turns=5))

        result = await agent.run([{"role": "user", "content": "查订单"}])

        assert result.finish_reason == "end_turn"  # 循环正常走完
        assert result.messages[-1]["content"] == "抱歉，出错了"
        # 工具失败的错误信息进了 tool_result
        tool_msg = [m for m in result.messages if m["role"] == "tool"]
        assert tool_msg and "数据库挂了" in tool_msg[0]["content"]

    async def test_max_turns_forced_stop(self) -> None:
        """模型永远要调工具 → max_turns 截断，finish_reason 记 "max_turns"。"""
        # 剧本只有 1 个工具调用响应，但 max_turns=1 → 第二轮调用会触发 AssertionError。
        # 因此这里用 max_turns=2 + 2 个 tool 响应，验证跑满后被截断。
        llm = MockLLM([_tool_response(), _tool_response()])
        reg = ToolRegistry()
        reg.register(QueryOrderTool())
        agent = AgentLoop(client=llm, tool_registry=reg, config=AgentLoopConfig(max_turns=2))

        result = await agent.run([{"role": "user", "content": "查订单"}])

        assert result.finish_reason == "max_turns"
        assert result.tool_calls_count == 2
        # 被截断时没有最终 assistant 回答
        assert result.messages[-1]["role"] == "tool"


class TestExecuteToolSafely:
    """_execute_tool_safely 兜底（工具超时/异常）。"""

    async def test_timeout(self) -> None:
        """工具超时 → 返回超时错误，不抛。"""
        class SlowTool:
            name = "slow"
            description = "慢"
            parameters = {"type": "object", "properties": {}}

            async def execute(self, **kwargs) -> dict:
                await asyncio.sleep(10)
                return {}

        reg = ToolRegistry()
        reg.register(SlowTool())
        agent = AgentLoop(
            client=None, tool_registry=reg, config=AgentLoopConfig(tool_timeout_seconds=1)
        )
        tc = ToolCall(id="c1", name="slow", arguments={})
        result = await agent._execute_tool_safely(tc)
        assert "超时" in result
