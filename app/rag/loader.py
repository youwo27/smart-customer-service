"""文档加载器 — 读取 data/raw_documents 下的原始文档。

支持格式：.txt / .md / .pdf
返回：list[Document]（id + 文本 + 元信息）
"""

from dataclasses import dataclass
from pathlib import Path



@dataclass
class Document:
    """一篇原始文档。"""

    id: str
    text: str
    source: str  # 文件路径
    metadata: dict = None  # 自定义元信息（主题/日期等）


class DocumentLoader:
    """从目录加载文档，支持多种格式。"""

    SUPPORTED = {".txt", ".md", ".pdf"}

    def __init__(self, data_dir: str | Path = "data/raw_documents") -> None:
        self.data_dir = Path(data_dir)

    def load_all(self) -> list[Document]:
        """加载目录下所有受支持格式的文档。"""
        docs: list[Document] = []
        for path in sorted(self.data_dir.iterdir()):
            if path.suffix.lower() not in self.SUPPORTED:
                continue
            text = self._read(path)
            docs.append(
                Document(
                    id=path.stem,
                    text=text,
                    source=str(path),
                )
            )
        return docs

    def load_one(self, filename: str) -> Document:
        """加载目录下的单篇文档（按文件名，不含扩展名或含扩展名均可）。"""
        path = self.data_dir / filename
        if not path.exists():
            # 兼容不带扩展名的调用，如 load_one("01-return-policy")
            path = self.data_dir / f"{filename}.md"
        if not path.exists():
            raise FileNotFoundError(f"文档不存在: {path}")
        if path.suffix.lower() not in self.SUPPORTED:
            raise ValueError(f"不支持的格式: {path.suffix}")
        return Document(
            id=path.stem,
            text=self._read(path),
            source=str(path),
        )

    def _read(self, path: Path) -> str:
        """读取单个文件。.pdf 用 PyPDF2 解析。"""
        if path.suffix.lower() == ".pdf":
            return self._read_pdf(path)
        return path.read_text(encoding="utf-8")

    def _read_pdf(self, path: Path) -> str:
        from PyPDF2 import PdfReader

        reader = PdfReader(str(path))
        return "\n".join(page.extract_text() or "" for page in reader.pages)