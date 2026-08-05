"""RAG：文档切割器 — 把长文档切成适合检索的 chunk。

三种策略：
- fixed:    固定大小切割（token 或字符）+ overlap
- recursive:递归字符切割（按层级分隔符逐步细分，保留语义）
- semantic: 语义切割（用 embedding 检测语义边界）→ Day 3 进阶，今天先做前两种
"""

from typing import Any

from app.rag.loader import Document


class Chunker:
    """文档切割器，支持多种策略。"""

    def __init__(self, chunk_size: int = 512, overlap: int = 128) -> None:
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(self, doc: Document, strategy: str = "fixed") -> list[dict[str, Any]]:
        """按策略切割一篇文档，返回 chunk 列表。

        每个 chunk: {"id": 唯一id, "text": 文本, "source": 来源, "index": 顺序}
        """
        text = doc.text
        if strategy == "fixed":
            chunks = self._chunk_fixed(text)
        elif strategy == "recursive":
            chunks = self._chunk_recursive(text)
        else:
            raise ValueError(f"未知切割策略: {strategy}")

        return [
            {
                "id": f"{doc.id}-chunk-{i}",
                "text": c,
                "source": doc.source,
                "index": i,
            }
            for i, c in enumerate(chunks)
        ]

    # ---- 策略 1：固定大小切割（字符级） ----
    def _chunk_fixed(self, text: str) -> list[str]:
        """按固定 chunk_size 切割，相邻 chunk 有 overlap 重叠。

        关键理解：为什么用 overlap？
        - 如果一段话恰好被从中间切开，语义就断了
        - overlap 让上下文延续到下一个 chunk，减少语义断裂
        """
        chunks: list[str] = []
        step = self.chunk_size - self.overlap
        for i in range(0, len(text), step):
            chunk = text[i : i + self.chunk_size]
            chunks.append(chunk)
        return chunks

    # ---- 策略 2：递归字符切割（按层级分隔符） ----
    def _chunk_recursive(self, text: str) -> list[str]:
        """按分隔符层级递归切割，尽可能让每个 chunk 语义完整。

        分隔符优先级：段落(\n\n) → 行(\n) → 句子(。！？) → 字符
        先按最粗的分隔符切，若 chunk 还太大，再按下一级切。
        """
        separators = ["\n\n", "\n", "。", "！", "？", " "]
        return self._recursive_split(text, separators, 0)

    def _recursive_split(self, text: str, seps: list[str], depth: int) -> list[str]:
        if len(text) <= self.chunk_size or depth >= len(seps):
            return [text] if text else []

        sep = seps[depth]
        parts = text.split(sep)
        # 用分隔符重新拼接，保留分隔符（比如"。！？"不该丢）
        pieces: list[str] = []
        current = ""
        for p in parts:
            piece = p + sep if sep not in (" ",) else p + " "
            if len(current) + len(piece) > self.chunk_size and current:
                pieces.append(current.strip())
                current = piece
            else:
                current += piece
        if current.strip():
            pieces.append(current.strip())

        # 每个 piece 如果还太大，递归下一级分隔符
        result: list[str] = []
        for p in pieces:
            if len(p) > self.chunk_size:
                result.extend(self._recursive_split(p, seps, depth + 1))
            else:
                result.append(p)
        return result
