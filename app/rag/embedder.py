"""RAG：文本向量化器。

Day 2 方案：占位实现，返回固定维度的零向量，让 RAG 管道先跑通。
Day 3 进阶：替换为真实 Embedding 模型（bge-large-zh 等），与混合检索一起做。

接口（embed / embed_batch / dimension）保持稳定，方便后续无缝替换实现。
"""


class Embedder:
    """文本向量化器。"""

    def __init__(self, model: str = "default") -> None:
        # model 参数预留：Day 3 换成真实模型时，这里切换实现
        self.model = model
        self._dimension = 384  # onnx-mini-lm（ChromaDB 内置默认模型）的维度

    @property
    def dimension(self) -> int:
        """向量维度，建库时用。"""
        return self._dimension

    async def embed(self, text: str) -> list[float]:
        """单条向量化。Day 2 返回占位零向量，让管道先跑通。"""
        # TODO (Day 3): 换真实模型，调用 Embedding API/模型
        return [0.0] * self._dimension

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """批量向量化。"""
        return [await self.embed(t) for t in texts]
