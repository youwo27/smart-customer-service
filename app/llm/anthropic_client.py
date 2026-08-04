"""Anthropic 客户端（可选，切回 Claude 时使用）。

仅当 LLM_PROVIDER=anthropic 时被加载。当前项目默认 DeepSeek。
"""

from typing import Any

from app.llm.client import LLMClient, LLMResponse


class AnthropicClient(LLMClient):
    """Anthropic Claude 客户端。TODO: 需要时实现。"""

    async def chat(self, messages: list[dict[str, Any]]) -> LLMResponse:
        raise NotImplementedError("切回 Anthropic 时实现")
