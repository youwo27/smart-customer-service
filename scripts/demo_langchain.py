"""LangChain 版 Agent 路由演示 — 与自研 demo_agentloop.py 对标（Day 5 交付物）。

同一份能力，两条实现路线：
- 自研（demo_agentloop.py）：自己管消息列表、自己解析 tool_calls、自己循环
- 本文件（LangChain）：AgentExecutor + create_tool_calling_agent 内建循环

运行方式（对应 Day 4 的 3 条路由 query）:
    python scripts/demo_langchain.py "退货政策是什么？"    → search_knowledge_base
    python scripts/demo_langchain.py "订单 12345 到哪了？" → query_order
    python scripts/demo_langchain.py "我要退款"            → apply_refund（危险操作守门）

对比点（写进 docs/Day5_framework_compare.md）：
- AgentExecutor 内建循环 → 自研的 while + max_turns 不用写了
- ChatPromptTemplate 的 placeholder 占位 → 消息列表自动维护
- 但 token 统计要自己挂 callback（LangChain 不默认给你 usage）
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Windows 终端 GBK 编码不支持部分 Unicode，改用 UTF-8 输出，避免崩溃
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]

from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from app.config import settings
from app.tools.builtin import register_defaults
from app.tools.registry import ToolRegistry
from scripts._lc_tools import to_langchain_tools

SYSTEM_PROMPT = (
    "你是家居电商客服助手。回答用户问题时，如果涉及政策/规则/流程，"
    "先检索知识库再回答；涉及订单状态用 query_order，物流用 query_logistics，"
    "退款必须用户明确要求才用 apply_refund。用简短友好的中文回答。"
)


def build_llm() -> ChatOpenAI:
    """DeepSeek 接入 LangChain：openai_api_base 指向 DeepSeek 兼容端点（零硬编码）。"""
    return ChatOpenAI(  # type: ignore[call-arg]  # openai_api_key/base 是 pydantic 字段，0.3.x stub 未暴露
        model=settings.llm_model,
        openai_api_key=settings.llm_api_key,
        openai_api_base=settings.llm_base_url,
        temperature=0.2,
    )


def build_executor() -> AgentExecutor:
    """组装 LangChain Agent（工具定义 + Prompt + AgentExecutor）。"""
    registry = ToolRegistry()
    register_defaults(registry)
    tools = to_langchain_tools(registry)

    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        # 多轮历史占位：AgentExecutor 自动维护；今天只演示单轮，传空列表即可
        ("placeholder", "{chat_history}"),
        ("human", "{input}"),
        # 工具调用历史占位：必须有，否则 tool calling 循环跑不起来
        ("placeholder", "{agent_scratchpad}"),
    ])

    agent = create_tool_calling_agent(build_llm(), tools, prompt)
    return AgentExecutor(
        agent=agent,
        tools=tools,
        handle_parsing_errors=True,  # 框架替我们兜解析错误（自研的 @retry + 异常回填）
        max_iterations=10,  # 对应自研的 max_turns（防死循环）
        return_intermediate_steps=True,  # 拿回工具调用过程（verbose=False 时也能展示）
        verbose=False,
    )


async def run_one(executor: AgentExecutor, user_message: str) -> None:
    """跑一次用户问题，打印 Agent 走了哪些工具 + 最终回答。"""
    print(f"\n{'=' * 70}\n用户：{user_message}\n{'=' * 70}")

    result = await executor.ainvoke(
        {"input": user_message, "chat_history": []}
    )

    print("---- Agent 调用过程（intermediate_steps） ----")
    for action, observation in result.get("intermediate_steps", []):
        tool_name = getattr(action, "tool", "?")
        tool_input = getattr(action, "tool_input", {})
        obs = str(observation)
        print(f"  [tool] {tool_name}({tool_input}) → {obs[:160]}")

    print("\n---- 最终回答 ----")
    print(result["output"])


async def main() -> None:
    executor = build_executor()
    queries = sys.argv[1:] or [
        "退货政策是什么？",
        "订单 12345 到哪了？",
        "我要退款",
    ]
    for q in queries:
        await run_one(executor, q)


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
