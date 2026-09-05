"""代码归一 + 匹配分析：把 relabel_v2 的原始标签经词表映射成标准名，再按 query 类型匹配。

不调 LLM，纯 dict 查表归一。验证重构后标签体系（双轨 + 分类型 + 受控词表）的匹配度。

匹配逻辑（按 query_type 分层）:
- 情节题: 文学文本 chunk 的 plot_unit / plot_detail 是否匹配 query
- 研究分析题: 研究文献 chunk 的分析主题是否匹配 query
- comparison题: 文学文本的 raw 称谓 + 研究文献的分析主题/涉及版本
"""
import json
from collections import Counter
from pathlib import Path

RESULTS_DIR = Path(__file__).resolve().parent / "results"


def build_normalizer(vocab: list) -> dict:
    m = {}
    for entry in vocab:
        std = entry["标准名"]
        m[std] = std
        for a in entry["别名"]:
            m[a] = std
    return m


def norm(v, normalizer):
    return normalizer.get(v, v) if v else ""


def main():
    merged = json.load(open(RESULTS_DIR / "merged_vocab.json", encoding="utf-8"))
    plot_norm = build_normalizer(merged["plot_units"])
    topic_norm = build_normalizer(merged["topics"])
    loc_norm = build_normalizer(merged["locations"])

    data = json.load(open(RESULTS_DIR / "relabel_v2.json", encoding="utf-8"))

    lines = []
    n_chunk = 0
    n_match = 0
    stat = Counter()

    for r in data:
        ql = r["query_label"]
        if not isinstance(ql, dict) or "_error" in ql:
            continue
        qtype = ql.get("query_type", "")
        q_plot = norm(ql.get("plot_unit", ""), plot_norm)
        q_topic = norm(ql.get("分析主题", ""), topic_norm)
        q_persons = {p.get("canonical", "") for p in (ql.get("persons") or []) if isinstance(p, dict)}
        q_persons_raw = {p.get("raw", "") for p in (ql.get("persons") or []) if isinstance(p, dict)}

        lines.append("=" * 70)
        lines.append(f"[{r['dataset']}/{r['query_id']}] type={qtype} 分析主题={q_topic!r} plot={q_plot!r}")

        for c in r["core_chunks"]:
            lab = c["label"]
            if not isinstance(lab, dict) or "_error" in lab:
                continue
            n_chunk += 1

            if c["genre"] == "research_literature":
                c_topic = norm(lab.get("分析主题", ""), topic_norm)
                c_vers = set(lab.get("涉及版本") or [])
                # 研究文献匹配: 分析主题相同
                matched = (q_topic != "" and q_topic == c_topic)
                key = f"研究[{c_topic}] vs query[{q_topic}]"
            else:
                c_plot = norm(lab.get("plot_unit", ""), plot_norm)
                c_persons = {p.get("canonical", "") for p in (lab.get("persons") or []) if isinstance(p, dict)}
                c_persons_raw = {p.get("raw", "") for p in (lab.get("persons") or []) if isinstance(p, dict)}
                # 文学文本匹配: plot_unit 相同 或 persons canonical 交集 或 raw 称谓交集
                plot_m = (q_plot != "" and q_plot == c_plot)
                person_m = bool(q_persons & c_persons)
                raw_m = bool(q_persons_raw & c_persons_raw)
                matched = plot_m or person_m or raw_m
                key = f"文学[plot:{q_plot}vs{c_plot} 人:{bool(q_persons&c_persons)} 称谓:{bool(q_persons_raw&c_persons_raw)}]"

            if matched:
                n_match += 1
                stat[(qtype, "match")] += 1
                mark = "MATCH"
            else:
                stat[(qtype, "miss")] += 1
                mark = "MISS "
            lines.append(f"  [{c['chunk_id'][-5:]}/{c['genre']}] {mark}  {key}")

        lines.append("")

    lines.append("=" * 70)
    lines.append(f"汇总（核心 chunk 总数 {n_chunk}）:")
    lines.append(f"  归一后匹配: {n_match}/{n_chunk} = {n_match/n_chunk:.1%}")
    lines.append("")
    lines.append("  按 query_type 分:")
    for qt in sorted({k[0] for k in stat.keys()}):
        m = stat.get((qt, "match"), 0)
        miss = stat.get((qt, "miss"), 0)
        lines.append(f"    {qt}: 匹配 {m}/{m+miss} = {m/(m+miss):.1%}")

    out = RESULTS_DIR / "normalize_match.txt"
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"done -> {out}")


if __name__ == "__main__":
    main()
