"""B-163 E2E前置验证：直连 Chroma 模拟 find_appearances('许仙') 的段落号修正结果。

运行: agent/venv/Scripts/python tests/verify_find_appearances_fix.py
"""
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "agent"))

from server.rag.paragraph_locator import get_paragraph_locator  # noqa: E402


def fake_find_appearances(docs, entity_name: str) -> list[dict]:
    """复刻 retriever.find_appearances 的修正逻辑（仅统计部分）。"""
    locator = get_paragraph_locator()
    out = []
    for d in docs:
        content = d.get("content", "") or ""
        if entity_name not in content:
            continue
        meta = d.get("metadata", {}) or {}
        located = locator.locate(meta.get("chapter_number", ""), content)
        orig = meta.get("paragraph_index", 0)
        out.append({
            "chapter": meta.get("chapter_number", ""),
            "orig": orig,
            "fixed": located if located is not None else orig,
            "located_ok": located is not None,
        })
    return out


def main() -> None:
    import chromadb
    from server.rag.config import rag_config

    client = chromadb.PersistentClient(path=str(rag_config.chroma_persist_path))
    col = client.get_or_create_collection("literature_chunks")
    got = col.get(include=["documents", "metadatas"])
    docs = [
        {"content": doc, "metadata": meta}
        for doc, meta in zip(got.get("documents") or [], got.get("metadatas") or [])
    ]
    rows = fake_find_appearances(docs, "许仙")
    ok = sum(1 for r in rows if r["located_ok"])
    invalid_orig = sum(1 for r in rows if isinstance(r["orig"], int) and r["orig"] < 0)
    print(f"含'许仙'的 chunk: {len(rows)}")
    print(f"原 paragraph_index 无效: {invalid_orig}")
    print(f"修正后有效: {ok} ({ok * 100 // max(len(rows), 1)}%)")
    print("\n样例（前10）:")
    for r in rows[:10]:
        print(f"  ch={r['chapter']} orig={r['orig']:>3} -> fixed={r['fixed']:>3} ok={r['located_ok']}")


if __name__ == "__main__":
    main()
