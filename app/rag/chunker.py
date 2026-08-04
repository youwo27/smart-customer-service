"""RAG：文档切割器（占位实现，Day 2 实现）。

职责：把原始文档切成适合检索的 chunk。
策略：固定大小 / 递归字符 / 语义切割。
规则（见 AGENTS.md）：切割策略与 chunk_size 必须可配置，禁止硬编码。
"""

from typing import Any


class Chunker:
    """文档切割器。

    TODO (Day 2):
    - 固定大小切割（chunk_size + overlap）
    - 递归字符切割
    - 语义切割
    """

    def __init__(self, chunk_size: int = 512, overlap: int = 128) -> None:
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(self, text: str) -> list[dict[str, Any]]:
        """切割文本为 chunks。Day 2 实现。"""
        raise NotImplementedError("Day 2 实现")
