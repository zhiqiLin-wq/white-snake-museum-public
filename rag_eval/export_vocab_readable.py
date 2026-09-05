"""导出 merged_vocab2.json 为可读文本，供人工审阅。"""
import json
from pathlib import Path

RESULTS = Path("rag_eval/results")
vocab = json.load(open(RESULTS / "merged_vocab2.json", encoding="utf-8"))

lines = []
for kind in ["persons", "locations", "plot_units", "topics"]:
    entries = vocab.get(kind, [])
    lines.append(f"=== {kind}（{len(entries)} 词条）===")
    for i, e in enumerate(entries, 1):
        canonical = e.get("canonical", "")
        raw = e.get("raw", [])
        if canonical == "待删除":
            lines.append(f"  {i}. [待删除] raw({len(raw)}): {'、'.join(raw)}")
        else:
            lines.append(f"  {i}. {canonical} <- {'、'.join(raw)}")
    lines.append("")

out = RESULTS / "merged_vocab2_readable.txt"
out.write_text("\n".join(lines), encoding="utf-8")
print(f"[OUT] {out}")
