"""切割验证脚本 — 挑一篇文章，对比两种切割策略的效果。

运行:
    python scripts/chunk_compare.py [文件名] [chunk_size] [overlap]
    # 例：python scripts/chunk_compare.py 01-return-policy 500 50
    #     python scripts/chunk_compare.py 05-refund-rules 300 30
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Windows 终端 GBK 编码不支持部分 Unicode，改用 UTF-8 输出
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from app.rag.chunker import Chunker
from app.rag.loader import DocumentLoader


def main(filename: str = "01-return-policy", chunk_size: int = 500, overlap: int = 50) -> None:
    loader = DocumentLoader("data/raw_documents")
    doc = loader.load_one(filename)  # ★ 用 load_one 挑单篇
    print(f"===== 文档: {doc.id}（{len(doc.text)} 字）=====\n")

    chunker = Chunker(chunk_size=chunk_size, overlap=overlap)

    for strategy in ("fixed", "recursive"):
        chunks = chunker.chunk(doc, strategy=strategy)
        print(f"[{strategy}] 一篇文档 → {len(chunks)} 个 chunk（chunk_size={chunk_size}, overlap={overlap}）")
        for i, c in enumerate(chunks):
            # 标出 chunk 首尾，观察切割边界
            first = c["text"][:25].replace("\n", "⏎")
            last = c["text"][-15:].replace("\n", "⏎")
            print(f"  chunk {i}: {first} … {last}")
        print()


if __name__ == "__main__":
    # 支持命令行传参：文件名 [chunk_size] [overlap]
    args = sys.argv[1:]
    name = args[0] if len(args) > 0 else "01-return-policy"
    size = int(args[1]) if len(args) > 1 else 500
    ovl = int(args[2]) if len(args) > 2 else 50
    main(name, size, ovl)
