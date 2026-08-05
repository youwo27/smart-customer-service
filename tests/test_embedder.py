"""Embedder 的单元测试。

Day 2 阶段 Embedder 是占位实现，测试硬编码 384。
Day 3 换成真实模型后，维度随 provider 变化（chroma=384 / zhipu=1024），
因此不再硬编码，而是断言「返回值维度 == Embedder().dimension」的一致性。
"""

from app.rag.embedder import Embedder


class TestEmbedder:
    """Embedder 测试：验证接口稳定，维度与 provider 一致。"""

    async def test_dimension_matches_provider(self) -> None:
        """维度与 provider 一致（不再硬编码 384）。"""
        e = Embedder()
        assert e.dimension == (384 if e.provider == "chroma" else 1024)

    async def test_embed_returns_float_vector(self) -> None:
        """单条向量化：返回 float 列表，维度与实例一致。"""
        e = Embedder()
        vec = await e.embed("怎么退货")
        assert isinstance(vec, list)
        assert len(vec) == e.dimension
        assert all(isinstance(x, float) for x in vec)

    async def test_embed_batch_matches_input_length(self) -> None:
        """批量向量化：输入 n 条 → 输出 n 个向量。"""
        e = Embedder()
        texts = ["怎么退货", "退货运费谁出", "优惠券怎么用"]
        vecs = await e.embed_batch(texts)
        assert len(vecs) == len(texts)
        assert all(len(v) == e.dimension for v in vecs)

    async def test_chroma_mode_offline(self) -> None:
        """chroma 模式：不联网也能出向量。"""
        e = Embedder(provider="chroma")
        vec = await e.embed("怎么退货")
        assert len(vec) == 384
