"""LLM 客户端工厂 — 支持多 Provider 切换（默认 DeepSeek）。

设计：
- 通过 `llm_provider` 配置切换，避免业务代码感知底层 API 差异
- DeepSeek / OpenAI 走 OpenAI 兼容协议（openai 包 + base_url）
- Anthropic 走 anthropic 包
- 所有 LLM 调用统一封装：超时 + 重试（指数退避）+ token 统计

用法:
    client = LLMClientFactory.create(settings)          # 根据配置创建
    resp = await client.chat(messages)                  # 对话
    tools = client.to_api_tools([...])                  # 工具定义转换
    resp = await client.chat_with_tools(messages, tools)  # 带工具对话
"""

from dataclasses import dataclass, field
from typing import Any

from tenacity import retry, stop_after_attempt, wait_exponential, wait_fixed

from app.logging_config import get_logger
from app.observability.tracing import span

logger = get_logger(__name__)


@dataclass
class LLMConfig:
    """LLM 客户端配置（从 Settings 映射而来）。"""

    provider: str = "deepseek"  # deepseek | anthropic | openai
    api_key: str = ""
    base_url: str = "https://api.deepseek.com"
    model: str = "deepseek-chat"
    fast_model: str = "deepseek-chat"
    timeout_seconds: float = 60.0
    max_retries: int = 3
    retry_base_delay: float = 1.0


@dataclass
class ToolCall:
    """一次工具调用请求（模型发起的）。"""

    id: str
    name: str
    arguments: dict[str, Any]  # 已解析为 dict 的参数


@dataclass
class LLMResponse:
    """统一 LLM 调用返回。"""

    content: str
    model: str
    usage: dict[str, int] = field(default_factory=dict)  # {input_tokens, output_tokens}
    finish_reason: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)  # 模型想调用的工具


def _record_llm_span(sp: Any, parsed: LLMResponse, message_count: int = 0) -> None:
    """把一次 LLM 调用的关键信息挂到 span 上。

    落点选在这里而不是调用处：token 数只有 `_parse` 之后才知道（usage 在响应里），
    所以在"拿到 LLMResponse 的那一刻"统一记录，chat / chat_with_tools 两条路径都覆盖。

    只放标量与计数：model、token_in/out、finish_reason、tool_calls 个数。
    整段 messages / content 进 attribute 会让 trace 爆掉（Day 9 坑 #5）——
    Jaeger 是用来"量"的，不是用来当日志全文库的。
    """
    usage = parsed.usage or {}
    sp.set_attribute("model", parsed.model or "")
    sp.set_attribute("token_in", usage.get("input_tokens", 0))
    sp.set_attribute("token_out", usage.get("output_tokens", 0))
    sp.set_attribute("token_total", usage.get("input_tokens", 0) + usage.get("output_tokens", 0))
    sp.set_attribute("finish_reason", parsed.finish_reason or "")
    sp.set_attribute("tool_calls_count", len(parsed.tool_calls))
    sp.set_attribute("message_count", message_count)


class LLMClient:
    """LLM 客户端基类（接口统一，实现分 provider）。"""

    def __init__(self, config: LLMConfig) -> None:
        self.config = config

    async def chat(self, messages: list[dict[str, Any]]) -> LLMResponse:
        """纯对话。子类实现。"""
        raise NotImplementedError

    async def chat_with_tools(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> LLMResponse:
        """带工具调用的对话。子类实现。"""
        raise NotImplementedError


class DeepSeekClient(LLMClient):
    """DeepSeek 客户端（OpenAI 兼容协议）。

    DeepSeek 与 OpenAI 的 tool_calls 格式一致，因此复用 OpenAI 兼容逻辑。
    """

    def __init__(self, config: LLMConfig) -> None:
        super().__init__(config)
        # 延迟导入，避免未安装依赖时应用无法启动
        from openai import AsyncOpenAI

        self._client = AsyncOpenAI(api_key=config.api_key, base_url=config.base_url)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        reraise=True,
    )
    async def _request(self, **kwargs: Any) -> Any:
        return await self._client.chat.completions.create(**kwargs)

    async def chat(self, messages: list[dict[str, Any]]) -> LLMResponse:
        with span("llm.chat", model=self.config.model, provider=self.config.provider) as sp:
            resp = await self._request(model=self.config.model, messages=messages)
            parsed = self._parse(resp)
            _record_llm_span(sp, parsed, message_count=len(messages))
            return parsed


    # fixed timeout 15s + retry 3x (1s interval)
    LLM_REQUEST_TIMEOUT = 15
    LLM_MAX_RETRY = 3
    LLM_RETRY_SLEEP = 1.0

    @retry(
        stop=stop_after_attempt(LLM_MAX_RETRY),
        wait=wait_fixed(LLM_RETRY_SLEEP),
        reraise=True,
    )
    async def chat_with_tools(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> LLMResponse:
        # span 开在函数体内 = 重试时**每次尝试各一个** span，而不是整段重试一个。
        # 这是想要的：能看到"第 1 次 15s 超时、第 2 次 800ms 成功"这种真实形态，
        # 只画一个 span 的话，超时的时间和重试的间隔全糊在一起，看不出问题在哪。
        with span(
            "llm.chat_with_tools",
            model=self.config.model,
            provider=self.config.provider,
            tool_count=len(tools),
        ) as sp:
            resp = await self._request(
                model=self.config.model,
                messages=messages,
                tools=tools,
                timeout=self.LLM_REQUEST_TIMEOUT,
            )
            parsed = self._parse(resp)
            _record_llm_span(sp, parsed, message_count=len(messages))
            return parsed

    def _parse(self, resp: Any) -> LLMResponse:
        choice = resp.choices[0]
        content = choice.message.content or ""
        usage = {
            "input_tokens": getattr(resp.usage, "prompt_tokens", 0),
            "output_tokens": getattr(resp.usage, "completion_tokens", 0),
        }
        # 解析模型发起的工具调用（OpenAI/DeepSeek 格式）
        tool_calls: list[ToolCall] = []
        raw_calls = getattr(choice.message, "tool_calls", None) or []
        for tc in raw_calls:
            args: dict[str, Any] = {}
            try:
                import json

                args = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {"_raw": tc.function.arguments}
            tool_calls.append(
                ToolCall(id=tc.id, name=tc.function.name, arguments=args)
            )
        return LLMResponse(
            content=content,
            model=resp.model,
            usage=usage,
            finish_reason=choice.finish_reason or "",
            tool_calls=tool_calls,
        )


class LLMClientFactory:
    """根据 provider 配置创建对应客户端。"""

    @staticmethod
    def create(config: LLMConfig | None = None) -> LLMClient:
        """创建 LLM 客户端。缺省配置时从 Settings 读取。"""
        if config is None:
            config = LLMConfigFactory.from_settings()
        logger.info(
            "llm_client_created",
            provider=config.provider,
            model=config.model,
            base_url=config.base_url,
        )
        provider = config.provider.lower()
        if provider in ("deepseek", "openai"):
            return DeepSeekClient(config)  # 两者都走 OpenAI 兼容协议
        if provider == "anthropic":
            from app.llm.anthropic_client import AnthropicClient

            return AnthropicClient(config)
        raise ValueError(f"不支持的 LLM provider: {provider}")

    @staticmethod
    def to_api_tools(tools: list[Any]) -> list[dict[str, Any]]:
        """把 BaseTool 列表转换为 API 工具定义（OpenAI/DeepSeek 格式）。

        Anthropic 格式不同，由 AnthropicClient 内部自行转换。
        """
        return [
            {
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description,
                    "parameters": t.parameters,
                },
            }
            for t in tools
        ]


class LLMConfigFactory:
    """从项目 Settings 构建 LLMConfig。"""

    @staticmethod
    def from_settings() -> LLMConfig:
        from app.config import settings

        api_key = (
            settings.llm_api_key
            if settings.llm_provider.lower() != "anthropic"
            else settings.anthropic_api_key
        )
        return LLMConfig(
            provider=settings.llm_provider,
            api_key=api_key,
            base_url=settings.llm_base_url,
            model=settings.llm_model,
            fast_model=settings.llm_fast_model,
            timeout_seconds=settings.tool_timeout_seconds * 2,
            max_retries=settings.llm_retry_max,
            retry_base_delay=settings.llm_retry_base_delay,
        )
