"""B-163 验证：用真实 Chroma 语料统计 ParagraphLocator 的段落号恢复率。

只读验证脚本，不修改任何数据。
运行: agent/venv/Scripts/python tests/verify_paragraph_locator_recovery.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import chromadb  # noqa: E402

from agent.server.rag.config import rag_config  # noqa: E402
from agent.server.rag.paragraph_locator import get_paragraph_locator  # noqa: E402


def main() -> None:
    client = chromadb.PersistentClient(path=str(rag_config.chroma_persist_path))
    col = client.get_or_create_collection("literature_chunks")
    got = col.get(include=["documents", "metadatas"])
    docs = got.get("documents") or []
    metas = got.get("metadatas") or []
    print(f"向量库 chunk 总数: {len(docs)}")

    locator = get_paragraph_locator()
    chs = locator.chapters()
    print(f"定位器章节: {sorted(chs.keys())}, 段落数: { {k: len(v) for k, v in chs.items()} }")

    total = 0
    invalid_before = 0      # 原 paragraph_index < 0
    located_ok = 0          # locate 成功
    located_changed = 0     # locate 结果 != 原值
    located_fail = 0        # locate 失败（含无效原值）
    fail_samples = []
    for doc, meta in zip(docs, metas):
        if not meta:
            continue
        # 只统计正文章节（研究文献没有章节段落坐标系，跳过不修）
        ch = str(meta.get("chapter_number") or "")
        if ch not in chs:
            continue
        total += 1
        orig = meta.get("paragraph_index", -1)
        if not isinstance(orig, int) or orig < 0:
            invalid_before += 1
        located = locator.locate(ch, doc or "")
        if located is None:
            located_fail += 1
            if len(fail_samples) < 5:
                fail_samples.append((ch, orig, (doc or "")[:40]))
        else:
            located_ok += 1
            if located != orig:
                located_changed += 1

    print(f"\n正文章节 chunk: {total}")
    print(f"原 paragraph_index 无效(<0): {invalid_before} ({invalid_before * 100 // max(total, 1)}%)")
    print(f"locate 成功: {located_ok} ({located_ok * 100 // max(total, 1)}%)")
    print(f"locate 失败: {located_fail}")
    print(f"locate 结果与原值不同: {located_changed}")
    if fail_samples:
        print("\n失败样本(前5):")
        for ch, orig, head in fail_samples:
            print(f"  ch={ch} orig={orig} head={head!r}")


if __name__ == "__main__":
    main()
