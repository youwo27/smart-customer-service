"""RAG：稀疏检索器（BM25）。混合检索的 Sparse 路。

为什么用字符级分词：中文没有空格，先按"中文单字 + 英文数字词"切，
比整段喂给 BM25 效果好。想更好可换 jieba（pyproject 未含，属可选增强）。
"""

import re
from typing import Any

from rank_bm25 import BM25Okapi


class SparseIndex:
    """对 chunk 语料建 BM25 索引，按关键词精确匹配检索。"""

    def __init__(self, chunks: list[dict[str, Any]]) -> None:
        self._chunks = chunks
        self._bm25 = BM25Okapi([self._tokenize(c["text"]) for c in chunks])

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        """中文按单字切，英文数字按词切。"""
        return re.findall(r"[一-鿿]|[a-zA-Z0-9]+", text)

    def search(self, query: str, top_k: int = 10) -> list[dict[str, Any]]:
        """BM25 检索，返回带 score 和 rank 的结果。"""
        scores = self._bm25.get_scores(self._tokenize(query))
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        results: list[dict[str, Any]] = []
        for i in ranked:
            if len(results) >= top_k or scores[i] <= 0:
                break
            results.append({**self._chunks[i], "score": scores[i], "rank": len(results)})
        return results
