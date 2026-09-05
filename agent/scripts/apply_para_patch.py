# -*- coding: utf-8 -*-
"""C3: 落库 paragraph_index 补算结果
1. 备份 corpus.jsonl
2. ChromaDB merge patch（365 块主文献 paragraph_index）
3. corpus.jsonl 同步重写
4. 复查 Chroma 分布
"""
import json
import shutil
import sys
from collections import Counter

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, r"e:\Flow\white-snake-museum-public\agent")

from server.rag.tag_store import CORPUS_PATH, load_corpus  # noqa: E402
from server.rag.config import rag_config  # noqa: E402
import chromadb  # noqa: E402
from chromadb.config import Settings as ChromaSettings  # noqa: E402

patch = json.load(open(CORPUS_PATH.parent / "_para_patch.json", encoding="utf-8"))
patch = {cid: r for cid, r in patch.items() if r["new"] >= 0}
print(f"待 patch: {len(patch)} 块")

# 1. 备份
bak = CORPUS_PATH.with_suffix(".jsonl.bak_para")
shutil.copy(CORPUS_PATH, bak)
print(f"备份: {bak}")

# 2. ChromaDB merge patch
client = chromadb.PersistentClient(
    path=str(rag_config.chroma_persist_path),
    settings=ChromaSettings(anonymized_telemetry=False, allow_reset=False),
)
col = client.get_collection("literature_chunks")
ids = sorted(patch.keys())
existing = col.get(ids=ids, include=["metadatas"])
meta_map = {cid: (m or {}) for cid, m in zip(existing["ids"], existing["metadatas"])}
updated = 0
for cid in ids:
    m = meta_map.get(cid)
    if m is None:
        print(f"  ⚠️ Chroma 中不存在: {cid}")
        continue
    m = dict(m)
    m["paragraph_index"] = int(patch[cid]["new"])
    col.update(ids=[cid], metadatas=[m])
    updated += 1
print(f"Chroma 已更新: {updated} 块")

# 3. corpus.jsonl 同步重写（保持行序与原字段，只改 paragraph_index）
corpus_rows = []
with open(CORPUS_PATH, encoding="utf-8") as f:
    for line in f:
        if line.strip():
            corpus_rows.append(json.loads(line))
changed = 0
for row in corpus_rows:
    did = row.get("doc_id") or row.get("_id")
    if did in patch:
        row.setdefault("metadata", {})["paragraph_index"] = int(patch[did]["new"])
        changed += 1
with open(CORPUS_PATH, "w", encoding="utf-8") as f:
    for row in corpus_rows:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
print(f"corpus.jsonl 已重写: {changed} 块变更")

# 4. 复查 Chroma 分布
got = col.get(include=["metadatas"])
combos = Counter()
for meta in got["metadatas"]:
    if meta.get("source_type") != "primary_literature":
        continue
    combos[(meta.get("dynasty"), str(meta.get("chapter_number")), str(meta.get("paragraph_index")))] += 1
print(f"\n复查 Chroma 主文献 (dynasty, chapter, paragraph) 组合数: {len(combos)} (原 8)")
