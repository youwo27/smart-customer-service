"""路由验证 — 用全部 6 个工具，验证 Agent 能否按用户意图路由到正确工具。

运行方式（对应 Day 4 交付物核心）:
    python scripts/demo_agentloop.py "退货政策是什么？"    → search_knowledge_base
    python scripts/demo_agentloop.py "订单 12345 到哪了？" → query_order
    python scripts/demo_agentloop.py "我要退款"            → apply_refund（触发二次确认钩子）
"""

import asyncio
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Windows 终端 GBK 编码不支持部分 Unicode（emoji 等），改用 UTF-8 输出，避免崩溃
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from app.core.agent import AgentLoop
from app.llm.client import LLMClientFactory, LLMConfigFactory
from app.tools.builtin import register_defaults
from app.tools.registry import ToolRegistry

SYSTEM_PROMPT = (
    "你是家居电商客服助手。回答用户问题时，如果涉及政策/规则/流程，"
    "先检索知识库再回答；涉及订单状态用 query_order，物流用 query_logistics，"
    "退款必须用户明确要求才用 apply_refund。用简短友好的中文回答。"
)


async def run_one(agent: AgentLoop, user_message: str) -> None:
    messages: list[dict] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_message},
    ]
    print(f"\n{'=' * 70}\n用户：{user_message}\n{'=' * 70}")

    result = await agent.run(messages)

    print("---- 循环过程 ----")
    for msg in result.messages:
        role = msg["role"]
        if role == "tool":
            # 截断显示，避免刷屏
            print(f"  [tool] {msg['content'][:160]}")
        elif role == "assistant" and msg.get("tool_calls"):
            for tc in msg["tool_calls"]:
                print(f"  [tool_calls] {tc['function']['name']}({tc['function']['arguments']})")
        elif role == "assistant" and msg["content"]:
            print(f"  [assistant] {msg['content'][:160]}")

    print("\n---- 最终回答 ----")
    print(result.messages[-1]["content"])
    print(f"\n工具调用 {result.tool_calls_count} 次 | tokens {result.tokens_used} | "
          f"模型 {result.model} | 结束 {result.finish_reason}")


async def main() -> None:
    llm = LLMClientFactory.create(LLMConfigFactory.from_settings())
    registry = ToolRegistry()
    register_defaults(registry)
    agent = AgentLoop(client=llm, tool_registry=registry)

    queries = sys.argv[1:] or [
        "退货政策是什么？",
        "订单 12345 到哪了？",
        "我要退款",
    ]
    for q in queries:
        await run_one(agent, q)


if __name__ == "__main__":
    asyncio.run(main())
