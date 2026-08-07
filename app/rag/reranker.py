"""RAG：重排序器 — Cross-Encoder 对候选重新打分（精排）。

为什么需要：混合检索给的是"大概相关的 Top-20"，噪声不少。
重排让 query 和每个候选"互相看见"再打分，把真正相关的排到最前。

provider：
  - siliconflow：bge-reranker-v2-m3 API（国内直连，推荐）
  - ""（空）：原序返回（兜底，无 key 也能跑）
"""

from typing import Any

import httpx

from app.config import settings
from app.logging_config import get_logger

logger = get_logger(__name__)


class Reranker:
    """检索结果重排序器。"""

    def __init__(
        self,
        *,
        provider: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self._provider = (provider or settings.rerank_provider).lower()
        self._api_key = api_key or settings.rerank_api_key
        self._base_url = base_url or settings.rerank_base_url
        self._model = model or settings.rerank_model
        self._client = client  # 测试时可注入 mock client

    async def rerank(
        self,
        query: str,
        candidates: list[dict[str, Any]],
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        """对候选重排，返回 Top-K。未配置 provider 时原序返回。"""
        if not self._provider:
            return candidates[:top_k]

        url = f"{self._base_url.rstrip('/')}/rerank"
        payload = {
            "model": self._model,
            "query": query,
            "documents": [c["text"] for c in candidates],
            "top_n": top_k,
        }
        headers = {"Authorization": f"Bearer {self._api_key}"}
        client = self._client or httpx.AsyncClient(timeout=30)
        try:
            resp = await client.post(url, headers=headers, json=payload)
            resp.raise_for_status()
            data = resp.json()
        finally:
            # 自己创建的 client 用完关闭；注入的不归这里管
            if self._client is None:
                await client.aclose()

        # 结果按相关性从高到低返回，index 指向 candidates 里的原位置
        by_score = sorted(data["results"], key=lambda r: r["relevance_score"], reverse=True)
        reranked = []
        for r in by_score:
            cand = dict(candidates[r["index"]])
            cand["rerank_score"] = r["relevance_score"]
            reranked.append(cand)
        return reranked[:top_k]
