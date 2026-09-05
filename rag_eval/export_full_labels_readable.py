"""导出 chunk_labels_full.json 为可读文本（含原文 + 标签），供人工审阅。"""
import json
from pathlib import Path

RESULTS = Path("rag_eval/results")
GT = Path("rag_eval/ground_truth")

data = json.load(open(RESULTS / "chunk_labels_full.json", encoding="utf-8"))
labels = data["chunk_labels"]

# 读原文
corpus = {}
with open(GT / "corpus.jsonl", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        d = json.loads(line)
        corpus[d.get("doc_id") or d.get("_id")] = d.get("text", "")

lines = []
for cid in sorted(labels.keys()):
    v = labels[cid]
    lab = v.get("label") or {}
    lines.append("=" * 70)
    lines.append(f"[{cid}]  genre={v['genre']}")
    lines.append(f"原文: {corpus.get(cid, '(未找到)')}")
    if "_error" in lab:
        lines.append(f"  [ERROR] {lab['_error']}")
        lines.append("")
        continue
    lines.append("标签:")
    if v["genre"] == "research_literature":
        lines.append(f"  涉及版本: {'、'.join(lab.get('涉及版本', []) or []) or '(空)'}")
        lines.append(f"  分析主题: {'、'.join(lab.get('分析主题', []) or []) or '(空)'}")
        ent = lab.get("引用实体", {}) or {}
        lines.append(f"  引用实体.persons: {'、'.join(ent.get('persons', []) or []) or '(空)'}")
        lines.append(f"  引用实体.locations: {'、'.join(ent.get('locations', []) or []) or '(空)'}")
        lines.append(f"  引用情节: {'、'.join(lab.get('引用情节', []) or []) or '(空)'}")
    else:
        lines.append(f"  persons: {'、'.join(lab.get('persons', []) or []) or '(空)'}")
        lines.append(f"  locations: {'、'.join(lab.get('locations', []) or []) or '(空)'}")
        lines.append(f"  plot_unit: {'、'.join(lab.get('plot_unit', []) or []) or '(空)'}")
        lines.append(f"  plot_detail: {lab.get('plot_detail', '') or '(空)'}")
    lines.append(f"  summary: {lab.get('summary', '') or '(空)'}")
    lines.append("")

out = RESULTS / "chunk_labels_full_readable.txt"
out.write_text("\n".join(lines), encoding="utf-8")
print(f"[OUT] {out}（{len(labels)} chunks，含原文）")
