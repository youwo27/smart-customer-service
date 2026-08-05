"""Embedder 的单元测试。

Day 2 阶段 Embedder 是占位实现，所以这里只验证"接口形状"稳定
（维度、返回值类型、批量数量），不断言具体向量值——
Day 3 换成真实模型后，这些测试仍应通过。
"""

from app.rag.embedder import Embedder


class TestEmbedder:
    """Embedder 测试：验证接口稳定，便于 Day 3 无缝替换实现。"""

    async def test_dimension(self) -> None:
        """维度 = onnx-mini-lm 的 384。"""
        assert Embedder().dimension == 384

    async def test_embed_returns_float_vector(self) -> None:
        """单条向量化：返回 384 维 float 列表。"""
        vec = await Embedder().embed("怎么退货")
        assert isinstance(vec, list)
        assert len(vec) == 384
        assert all(isinstance(x, float) for x in vec)

    async def test_embed_batch_matches_input_length(self) -> None:
        """批量向量化：输入 n 条 → 输出 n 个向量。"""
        texts = ["怎么退货", "退货运费谁出", "优惠券怎么用"]
        vecs = await Embedder().embed_batch(texts)
        assert len(vecs) == len(texts)
        assert all(len(v) == 384 for v in vecs)

    async def test_model_param_reserved(self) -> None:
        """model 参数预留：Day 3 用于切换真实模型。"""
        assert Embedder(model="default").model == "default"
