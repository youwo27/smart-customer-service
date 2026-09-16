"""Agent 核心循环 — ReAct 主循环（推理 → 行动 → 观察 → 推理）。

Day 4 从占位实现变成真代码：
- run()：while 循环，终止条件 end_turn / max_turns
- 工具执行保护：异常转 tool_result，让 Agent 自我纠正（AGENTS.md 3.3）
- 并行工具调用：一个 assistant 消息带多个 tool_calls，逐条回填 tool 结果
- 指数退避重试 + 超时（asyncio.wait_for）
- token 统计：每次调用累加 usage.input_tokens / output_tokens

关键设计决策：
- finish_reason 区分正常收尾（LLM 返回的 end_turn/stop）与 max_turns 截断，
  上层 / 日志可据此区分"正常"与"被上限截断"
- 消息组装格式与 demo_react.py 完全一致（id / type / function.name / function.arguments），
  保证 API 下一轮能读懂（见 Day 4 常见坑 #1）
"""

import asyncio
import json
from dataclasses import dataclass, field
from typing import Any

from tenacity import retry, stop_after_attempt, wait_exponential

from app.logging_config import get_logger
from app.observability.tracing import span

logger = get_logger(__name__)


@dataclass
class AgentLoopConfig:
    """Agent 循环配置（全部来自 settings，禁止硬编码）。"""

    max_turns: int = 15
    tool_timeout_seconds: int = 30
    llm_retry_max: int = 3
    llm_retry_base_delay: float = 1.0


@dataclass
class AgentRunResult:
    """一次 Agent 循环的运行结果。"""

    messages: list[dict[str, Any]] = field(default_factory=list)
    tool_calls_count: int = 0
    tokens_used: int = 0
    model: str = ""
    finish_reason: str = ""


class AgentLoop:
    """ReAct Agent 主循环。"""

    def __init__(
        self,
        client: Any,  # LLMClient
        tool_registry: Any,  # ToolRegistry
        config: AgentLoopConfig | None = None,
    ) -> None:
        self.client = client
        self.tool_registry = tool_registry
        self.config = config or AgentLoopConfig()

    # ============ ① 指数退避重试 + 超时（与 LLMClient 同款） ============
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        reraise=True,
    )
    async def _call_with_retry(self, messages: list[dict[str, Any]]) -> Any:
        """调用 LLM，带重试 + 超时。"""
        return await asyncio.wait_for(
            self.client.chat_with_tools(messages, self.tool_registry.list()),
            timeout=self.config.tool_timeout_seconds,
        )

    # ============ ② 消息组装（格式与 demo_react.py 一致） ============
    def _assistant_msg(self, resp: Any) -> dict[str, Any]:
        """把 LLM 响应组装成 assistant 消息。

        有 tool_calls → content + tool_calls 数组；没有 → 仅 content。
        """
        if resp.tool_calls:
            return {
                "role": "assistant",
                "content": resp.content,
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.name,
                            "arguments": json.dumps(tc.arguments, ensure_ascii=False),
                        },
                    }
                    for tc in resp.tool_calls
                ],
            }
        return {"role": "assistant", "content": resp.content}

    def _tool_msg(self, tool_call_id: str, result: str) -> dict[str, Any]:
        """把工具执行结果组装成 tool 消息（用 tool_call_id 关联回那次调用）。"""
        return {"role": "tool", "tool_call_id": tool_call_id, "content": result}

    # ============ ③ 工具执行保护（异常转 tool_result，不抛上层） ============
    async def _execute_tool_safely(self, tc: Any) -> str:
        """执行单个工具，异常转成 {"error": ...} 字符串返回（AGENTS.md 3.3）。"""
        try:
            result = await asyncio.wait_for(
                self.tool_registry.execute(tc.name, **tc.arguments),
                timeout=self.config.tool_timeout_seconds,
            )
            return json.dumps(result, ensure_ascii=False)
        except asyncio.TimeoutError:
            return json.dumps({"error": f"工具 {tc.name} 执行超时"}, ensure_ascii=False)
        except Exception as e:  # noqa: BLE001 — 工具失败必须兜住，不让循环崩溃
            logger.warning("tool_failed", name=tc.name, error=str(e))
            return json.dumps({"error": f"工具 {tc.name} 执行失败: {e}"}, ensure_ascii=False)

    # ============ ④ 主循环 ============
    async def run(self, messages: list[dict[str, Any]], session_id: str = "") -> AgentRunResult:
        """ReAct 主循环：推理 → 工具调用 → 观察 → 推理，直到 end_turn 或 max_turns。

        Day 9：整轮循环包一个 `agent.run` span。明细（每次 LLM 调用的 token、每次工具
        执行的耗时）都在各自的子 span 里，这里只记"一次请求的总账"：turns / tokens_used /
        finish_reason —— 打开 Jaeger 先看这个，要深挖再往子 span 里钻。

        turns 用"本次**新增**的 assistant 消息条数"算：每轮循环恰好 append 一条 assistant
        消息，但传进来的 messages 里还带着历史轮次的 assistant 消息（orchestrator 拼的），
        直接数总数会把前几轮的算进来。
        """
        before = len(messages)
        with span("agent.run", session_id=session_id, max_turns=self.config.max_turns) as sp:
            result = await self._run_loop(messages)
            sp.set_attribute(
                "turns", sum(1 for m in messages[before:] if m.get("role") == "assistant")
            )
            sp.set_attribute("tool_calls_count", result.tool_calls_count)
            sp.set_attribute("tokens_used", result.tokens_used)
            sp.set_attribute("finish_reason", result.finish_reason)
            sp.set_attribute("model", result.model)
        return result

    async def _run_loop(self, messages: list[dict[str, Any]]) -> AgentRunResult:
        """真正的主循环。

        刻意和 run() 分开：span 只管"包住 + 计时"，循环逻辑本身不掺任何观测代码 ——
        埋点要是把业务代码改得不好读了，下次就没人愿意再埋。
        """
        total_tokens = 0
        tool_calls_count = 0
        model = ""
        finish_reason = "max_turns"  # 默认：循环被上限截断

        for turn in range(self.config.max_turns):
            # ① 调 LLM，拿这一轮的响应
            resp = await self._call_with_retry(messages)
            model = resp.model
            total_tokens += resp.usage.get("input_tokens", 0) + resp.usage.get("output_tokens", 0)

            # ② 把 assistant 消息（含 tool_calls）append 进 messages
            messages.append(self._assistant_msg(resp))

            # ③ 没有 tool_calls → 最终回答，正常收尾
            if not resp.tool_calls:
                finish_reason = resp.finish_reason or "end_turn"
                break

            # ④ 有 tool_calls → 逐条执行 + 逐条回填 tool 结果，继续循环
            for tc in resp.tool_calls:
                result = await self._execute_tool_safely(tc)
                messages.append(self._tool_msg(tc.id, result))
                tool_calls_count += 1

        return AgentRunResult(
            messages=messages,
            tool_calls_count=tool_calls_count,
            tokens_used=total_tokens,
            model=model,
            finish_reason=finish_reason,
        )
