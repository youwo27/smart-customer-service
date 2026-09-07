"""LangChain 适配层单元测试 — 全部用 mock，不真调 DeepSeek / 不联网（Day 5 交付物）。

覆盖 4 块（对应 Day 5 Part 4）：
  1. 工具转换：to_langchain_tools 生成的 StructuredTool 与自研 6 工具的 name/description 一一对应
  2. 工具执行：await tool.ainvoke(...) 调到了自研 execute（query_order 命中 mock DB）
  3. LCEL 链：假 retriever + 假 llm，验证 rag_chain.invoke 产出
  4. Agent 组装：create_tool_calling_agent 能正常创建（不真跑循环）
"""

from typing import Any

import pytest
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough

from app.rag.reranker import Reranker
from app.tools.builtin import ALL_TOOLS, register_defaults
from app.tools.registry import ToolRegistry
from scripts._lc_tools import make_lc_tool, to_langchain_tools

# ============ 1. 工具转换：一一对应 ============


class TestToolConversion:
    """to_langchain_tools 必须忠实翻译自研 6 工具，不丢字段、不改 schema。"""

    def _registry(self) -> ToolRegistry:
        reg = ToolRegistry()
        register_defaults(reg)
        return reg

    def test_all_6_tools_converted(self) -> None:
        tools = to_langchain_tools(self._registry())
        assert len(tools) == len(ALL_TOOLS) == 6

    def test_name_description_match(self) -> None:
        """StructuredTool 的 name/description 与自研工具完全一致（决定调用准确率）。"""
        reg = self._registry()
        for lc in to_langchain_tools(reg):
            ours = reg.get(lc.name)
            assert lc.name == ours.name
            assert lc.description == ours.description

    def test_schema_equals_handwritten(self) -> None:
        """LangChain 自动 schema 与自研手写 schema 参数一致（单一事实源）。"""
        reg = self._registry()
        for lc in to_langchain_tools(reg):
            ours = reg.get(lc.name)
            lc_schema = lc.args_schema.model_json_schema()  # type: ignore[union-attr]
            ours_schema = ours.parameters
            # 参数名集合一致
            assert set(lc_schema.get("properties", {})) == set(
                ours_schema.get("properties", {})
            )
            # 必填集合一致
            assert lc_schema.get("required", []) == ours_schema.get("required", [])

    def test_query_order_has_order_id(self) -> None:
        reg = self._registry()
        lc = next(t for t in to_langchain_tools(reg) if t.name == "query_order")
        schema = lc.args_schema.model_json_schema()  # type: ignore[union-attr]
        assert schema["required"] == ["order_id"]


# ============ 2. 工具执行：ainvoke 调到了自研 execute ============


class TestToolExecution:
    """直接 await tool.ainvoke(...) 验证链路通到自研 execute。"""

    async def test_query_order_executes(self) -> None:
        reg = ToolRegistry()
        register_defaults(reg)
        tools = to_langchain_tools(reg)
        lc = next(t for t in tools if t.name == "query_order")

        result = await lc.ainvoke({"order_id": "12345"})
        assert result["status"] == "已发货"  # mock DB 命中

    async def test_query_order_not_found(self) -> None:
        reg = ToolRegistry()
        register_defaults(reg)
        lc = next(t for t in to_langchain_tools(reg) if t.name == "query_order")

        result = await lc.ainvoke({"order_id": "99999"})
        assert "error" in result

    async def test_apply_refund_validation(self) -> None:
        """LangChain 的 args_schema 会先做 pydantic 必填校验 → 缺参直接抛 ValidationError。

        对比自研：execute(**kwargs) 拿到缺参后返回结构化错误 dict，让 Agent 自我纠正。
        这个差异正是框架"替你校验"的代价——错误类型从"业务错误"变成"框架异常"。
        """
        from pydantic import ValidationError

        reg = ToolRegistry()
        register_defaults(reg)
        lc = next(t for t in to_langchain_tools(reg) if t.name == "apply_refund")

        with pytest.raises(ValidationError):
            await lc.ainvoke({"order_id": "12345", "amount": 899})  # 缺 reason

    async def test_apply_refund_full_params_executes(self) -> None:
        """参数齐全时链路通到自研 execute，返回结构化结果。"""
        reg = ToolRegistry()
        register_defaults(reg)
        lc = next(t for t in to_langchain_tools(reg) if t.name == "apply_refund")

        result = await lc.ainvoke({"order_id": "88888", "amount": 1299, "reason": "拍错了"})
        assert result["status"] == "退款成功"


# ============ 3. LCEL 链：假 retriever + 假 llm ============


class TestLCELRagChain:
    """用假检索 + 假 LLM 验证 LCEL 链的管道能跑通并产出。"""

    def _build(self, fake_llm: FakeListChatModel) -> Any:
        """构造一条 rag_chain：假 search（返回固定 chunks）| format_docs | prompt | fake llm。"""

        async def fake_search(query: str) -> str:
            assert query == "退货政策是什么？"
            return "支持 7 天无理由退货"

        prompt = ChatPromptTemplate.from_messages([
            ("system", "根据上下文回答：\n{context}"),
            ("human", "{question}"),
        ])
        # 0.3.x 的 RunnableLambda stub 不接受纯 async 函数 → type: ignore
        return (
            {
                "context": RunnableLambda(fake_search),  # type: ignore[arg-type]
                "question": RunnablePassthrough(),
            }
            | prompt
            | fake_llm
            | StrOutputParser()
        )

    async def test_chain_invokes_and_outputs(self) -> None:
        fake_llm = FakeListChatModel(responses=["支持 7 天无理由退货。"])
        chain = self._build(fake_llm)
        out = await chain.ainvoke("退货政策是什么？")
        assert out == "支持 7 天无理由退货。"

    async def test_chain_passes_context_to_prompt(self) -> None:
        """验证检索结果确实拼进了 prompt 的 {context}。"""
        seen: list[str] = []

        async def fake_search(query: str) -> str:
            return "退货运费由买家承担"

        prompt = ChatPromptTemplate.from_messages([
            ("system", "上下文：\n{context}"),
            ("human", "{question}"),
        ])
        chain: Any = (
            {
                "context": RunnableLambda(fake_search),  # type: ignore[arg-type]
                "question": RunnablePassthrough(),
            }
            | prompt
            | RunnableLambda(lambda msg: seen.append(str(msg)))
        )
        await chain.ainvoke("运费谁出？")
        assert any("退货运费由买家承担" in s for s in seen)


# ============ 4. Agent 组装：不真跑循环 ============


class TestAgentAssembly:
    """create_tool_calling_agent + AgentExecutor 能正常创建（不真调 LLM）。"""

    async def test_executor_builds_with_all_tools(self) -> None:
        reg = ToolRegistry()
        register_defaults(reg)
        tools = to_langchain_tools(reg)

        # 组装不真跑循环，所以用真实 ChatOpenAI 实例即可（build 阶段不发请求）；
        # FakeListChatModel 不实现 bind_tools，不能用于 create_tool_calling_agent。
        from langchain_openai import ChatOpenAI

        llm = ChatOpenAI(  # type: ignore[call-arg]  # pydantic 字段，0.3.x stub 未暴露
            model="deepseek-chat",
            openai_api_key="test-key",
            openai_api_base="https://api.deepseek.com",
            temperature=0.2,
        )
        prompt = ChatPromptTemplate.from_messages([
            ("system", "你是客服助手。"),
            ("placeholder", "{chat_history}"),
            ("human", "{input}"),
            ("placeholder", "{agent_scratchpad}"),
        ])

        from langchain.agents import AgentExecutor, create_tool_calling_agent

        agent = create_tool_calling_agent(llm, tools, prompt)
        executor = AgentExecutor(agent=agent, tools=tools, max_iterations=5)

        # 不真跑循环，只验证结构
        assert len(executor.tools) == 6
        assert executor.max_iterations == 5

    def test_make_lc_tool_single(self) -> None:
        """单工具转换入口也能用（make_lc_tool）。"""
        from app.tools.builtin.customer_service import QueryOrderTool

        lc = make_lc_tool(QueryOrderTool())
        assert lc.name == "query_order"


# ============ 附带：Reranker 离线兜底回归（防真实 API） ============


class TestRerankerOfflineGuard:
    """provider='' 必须禁用重排，不回落到 .env 配置（防单测打真实 API）。"""

    async def test_empty_provider_disables(self) -> None:
        r = Reranker(provider="")
        candidates = [
            {"id": "a", "text": "无理由退货运费由买家承担"},
            {"id": "b", "text": "质量问题退货运费由商家承担"},
        ]
        result = await r.rerank("退货运费谁出", candidates, top_k=2)
        assert [c["id"] for c in result] == ["a", "b"]
        assert "rerank_score" not in result[0]  # 不打分，不联网
