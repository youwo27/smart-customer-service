"""知识库级切割验证 — 检查一个 chunk_size 是否让所有文档都"完整成 1 块"。

运行:
    python scripts/validate_chunking.py [chunk_size] [overlap]
    # 例：python scripts/validate_chunking.py 500 50
    #     python scripts/validate_chunking.py 600 60
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from app.rag.chunker import Chunker
from app.rag.loader import DocumentLoader


def main(chunk_size: int = 500, overlap: int = 50) -> None:
    loader = DocumentLoader("data/raw_documents")
    docs = loader.load_all()
    chunker = Chunker(chunk_size=chunk_size, overlap=overlap)

    print(f"=== 知识库级切割验证（chunk_size={chunk_size}, overlap={overlap}, recursive）===\n")
    print(f"{'文档':<32}{'字数':>6}  {'chunk数':>7}  状态")
    print("-" * 65)

    longest = max(docs, key=lambda d: len(d.text))
    problems: list[str] = []
    total_chunks = 0

    for doc in docs:
        chunks = chunker.chunk(doc, strategy="recursive")
        total_chunks += len(chunks)
        ok = "✅ 完整 1 块" if len(chunks) == 1 else f"⚠️  切成 {len(chunks)} 块"
        if len(chunks) > 1:
            problems.append(f"{doc.id} ({len(chunks)} 块)")
        print(f"{doc.id:<32}{len(doc.text):>6}  {len(chunks):>7}  {ok}")

    print("-" * 65)
    print(f"最长文档: {longest.id}（{len(longest.text)} 字）")
    print(f"总 chunk 数: {total_chunks}（20 篇文档）")
    print()

    if not problems:
        print(f"✅ 结论: chunk_size={chunk_size} 让全部 {len(docs)} 篇文档完整成 1 块，参数合理。")
    else:
        print(f"⚠️  结论: 以下文档被切成多块，chunk_size={chunk_size} 不够：")
        for p in problems:
            print(f"  - {p}")
        print(f"  建议: 把 chunk_size 调到 ≥ {len(longest.text)}（最长文档字数）")


if __name__ == "__main__":
    size = int(sys.argv[1]) if len(sys.argv) > 1 else 500
    ovl = int(sys.argv[2]) if len(sys.argv) > 2 else 50
    main(size, ovl)
