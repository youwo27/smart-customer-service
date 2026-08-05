"""loader 和 chunker 的单元测试（真正的 pytest 测试）。"""

import pytest

from app.rag.chunker import Chunker
from app.rag.loader import Document, DocumentLoader


class TestDocumentLoader:
    """DocumentLoader 测试。"""

    def test_load_one_success(self) -> None:
        """load_one 能加载单篇文档。"""
        loader = DocumentLoader("data/raw_documents")
        doc = loader.load_one("01-return-policy")
        assert doc.id == "01-return-policy"
        assert len(doc.text) > 100  # 有内容

    def test_load_one_not_found(self) -> None:
        """load_one 找不到文档时报错。"""
        loader = DocumentLoader("data/raw_documents")
        with pytest.raises(FileNotFoundError):
            loader.load_one("not-exist")

    def test_load_all_count(self) -> None:
        """load_all 加载全部 20 篇文档。"""
        loader = DocumentLoader("data/raw_documents")
        docs = loader.load_all()
        assert len(docs) == 20


class TestChunker:
    """Chunker 测试。"""

    def _make_doc(self, text: str) -> Document:
        return Document(id="test", text=text, source="test.md")

    def test_fixed_chunk_count(self) -> None:
        """固定切割：400 字 chunk_size=200 overlap=50 → 应为 3 个 chunk。"""
        doc = self._make_doc("字" * 400)
        chunker = Chunker(chunk_size=200, overlap=50)
        chunks = chunker.chunk(doc, strategy="fixed")
        # step=150: 切在第 0,150,300 处 → 3 个 chunk
        assert len(chunks) == 3
        # 每个 chunk 有 id/source/index 字段
        assert chunks[0]["id"] == "test-chunk-0"
        assert chunks[0]["source"] == "test.md"
        assert chunks[0]["index"] == 0

    def test_fixed_overlap_preserved(self) -> None:
        """固定切割：相邻 chunk 应有 overlap 字符重叠。"""
        text = "A" * 100 + "B" * 100 + "C" * 100 + "D" * 100  # 400 字
        doc = self._make_doc(text)
        chunker = Chunker(chunk_size=100, overlap=30)
        chunks = chunker.chunk(doc, strategy="fixed")
        # step=70: chunk0=[0,100), chunk1=[70,170) → 重叠 [70,100)
        assert chunks[0]["text"][-30:] == chunks[1]["text"][:30]

    def test_recursive_keeps_sentence(self) -> None:
        """递归切割：短文本不应被切断。"""
        text = "第一句。第二句。第三句。"
        doc = self._make_doc(text)
        chunker = Chunker(chunk_size=200, overlap=50)  # 大于全文
        chunks = chunker.chunk(doc, strategy="recursive")
        # 全文小于 chunk_size → 只有一个完整 chunk
        assert len(chunks) == 1
        assert chunks[0]["text"] == text

    def test_unknown_strategy_raises(self) -> None:
        """未知策略报错。"""
        doc = self._make_doc("hello")
        chunker = Chunker()
        with pytest.raises(ValueError):
            chunker.chunk(doc, strategy="semantic")
