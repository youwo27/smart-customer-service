"""RAG：文本向量化器（Day 3 真实实现）。

Day 2：占位实现（零向量），只求管道跑通。
Day 3：接入真实 Embedding，两种 provider 二选一（配置注入，零硬编码）：
  - zhipu：智谱 embedding-3（OpenAI 兼容 /embeddings，中文效果好，默认 1024 维）
  - chroma：ChromaDB 内置 DefaultEmbeddingFunction（离线、零配置、384 维，兜底方案）

设计决策：
- 接口（embed / embed_batch / dimension / provider）保持稳定，实现可无痛替换。
- 未配置 API key 时自动降级到 chroma 离线模式（与 Day 3 Part 1 的"没有 key 也别卡住"一致）。
- 智谱 API 调用与 LLMClient 同款：超时 30s + 指数退避重试（tenacity）。
"""

from typing import Any

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

from app.config import settings
from app.logging_config import get_logger

logger = get_logger(__name__)


class Embedder:
    """文本向量化器（真实实现）。

    接口稳定：embed / embed_batch / dimension / provider。
    provider 从配置注入（测试时可用显式参数覆盖）。
    """

    def __init__(
        self,
        model: str = "default",
        *,
        provider: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        embed_model: str | None = None,
    ) -> None:
        # 显式参数优先，否则读配置（测试时可注入 mock）
        self.model = model
        self.provider = (provider or settings.embedding_provider).lower()
        self._api_key = api_key or settings.embedding_api_key
        self._base_url = base_url or settings.embedding_base_url
        self._embed_model = embed_model or settings.embedding_model
        self._fn: Any = None  # chroma 模式的 EmbeddingFunction；zhipu 模式下为 None

        # 选型：需要联网的 provider 但没有 key → 自动降级离线（避免 401 卡住）
        if self.provider != "chroma" and not self._api_key:
            logger.warning(
                "embedder_downgraded_offline",
                requested=self.provider,
                reason="missing_api_key",
            )
            self.provider = "chroma"

        if self.provider == "chroma":
            from chromadb.utils.embedding_functions import DefaultEmbeddingFunction

            self._fn = DefaultEmbeddingFunction()
            self._dimension = 384  # onnx-mini-lm
        else:
            self._fn = None
            self._dimension = 1024  # 智谱 embedding-3

    @property
    def dimension(self) -> int:
        """向量维度，建库/重索引时用。"""
        return self._dimension

    async def embed(self, text: str) -> list[float]:
        """单条向量化。"""
        if self.provider == "chroma":
            return [float(x) for x in self._fn([text])[0]]
        return await self._embed_api(text)

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """批量向量化。"""
        return [await self.embed(t) for t in texts]

    # ---- 智谱 / OpenAI 兼容 API（与 LLMClient 同款：超时 + 指数退避重试） ----
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        reraise=True,
    )
    async def _embed_api(self, text: str) -> list[float]:
        url = f"{self._base_url.rstrip('/')}/embeddings"
        payload = {
            "model": self._embed_model,
            "input": text,
            # 显式指定维度：保证返回维度与 dimension 属性一致（避免 2048/1024 漂移）
            "dimensions": self._dimension,
        }
        headers = {"Authorization": f"Bearer {self._api_key}"}
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.post(url, headers=headers, json=payload)
            resp.raise_for_status()
            data = resp.json()["data"][0]["embedding"]
            return [float(x) for x in data]
