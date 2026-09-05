"""导出 query_labels.json 为可读文本，供人工审阅。"""
import json
from pathlib import Path

RESULTS = Path("rag_eval/results")
data = json.load(open(RESULTS / "query_labels.json", encoding="utf-8"))

lines = []
for r in data:
    lab = r.get("query_label") or {}
    lines.append("=" * 70)
    lines.append(f"[{r['dataset']}/{r['query_id']}]  category={r.get('category','')}")
    lines.append(f"Q: {r.get('query_text','')}")
    if "_error" in lab:
        lines.append(f"  [ERROR] {lab['_error']}")
        lines.append("")
        continue
    lines.append(f"query_type: {lab.get('query_type','')}")
    lines.append(f"need_examples: {lab.get('need_examples')}")
    lines.append(f"need_analysis: {lab.get('need_analysis')}")
    for field in ["persons", "locations", "plot_unit", "分析主题"]:
        vals = lab.get(field) or []
        if not vals:
            lines.append(f"{field}: (空)")
        else:
            items = "、".join(f"{v.get('canonical','')}(raw={v.get('raw','')})" for v in vals if isinstance(v, dict))
            lines.append(f"{field}: {items}")
    lines.append(f"plot_detail: {lab.get('plot_detail','')}")
    lines.append(f"对比维度: {json.dumps(lab.get('对比维度', {}), ensure_ascii=False)}")
    lines.append(f"summary: {lab.get('summary','')}")
    lines.append(f"genre: {lab.get('genre','')}")
    lines.append("")

out = RESULTS / "query_labels_readable.txt"
out.write_text("\n".join(lines), encoding="utf-8")
print(f"[OUT] {out}")
