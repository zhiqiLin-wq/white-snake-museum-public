"""匹配度分析：query 标准名标签 vs 核心 chunk 标签（经规范表归一）。

不重跑检索、不调 LLM，纯算标签匹配，先回答核心问题:
    unretrievable 的核心 chunk，结构化标签和 query 标签能不能匹配上？
    能匹配上，boost 才有意义；匹配不上，方案要重新想。

匹配逻辑:
    用 label_vocab.json 把 chunk 标签里的别名归一到标准名，
    再和 query 的标准名标签比较。
"""
import json
from pathlib import Path

RESULTS_DIR = Path(__file__).resolve().parent / "results"


def build_reverse(vocab: dict) -> dict:
    rev = {}
    for std, aliases in vocab.items():
        rev[std] = std
        for a in aliases:
            rev[a] = std
    return rev


def norm(v, rev: dict) -> str:
    return rev.get(v, v) if v else ""


def main():
    vocab = json.load(open(RESULTS_DIR / "label_vocab.json", encoding="utf-8"))
    query_data = json.load(open(RESULTS_DIR / "relabel_query_vocab.json", encoding="utf-8"))
    chunk_data = json.load(open(RESULTS_DIR / "label_pilot_10q.json", encoding="utf-8"))

    person_rev = build_reverse(vocab["person_aliases"])
    loc_rev = build_reverse(vocab["location_aliases"])
    plot_rev = build_reverse(vocab["plot_units"])

    # 建 chunk 标签索引: (dataset, query_id, chunk_id) -> label
    chunk_label = {}
    for r in chunk_data:
        for c in r["core_chunks"]:
            chunk_label[(r["dataset"], r["query_id"], c["chunk_id"])] = c["label"]

    lines = []
    n_chunk_total = 0
    n_plot_match = 0
    n_person_overlap = 0
    n_loc_overlap = 0
    n_any_match = 0

    for r in query_data:
        ds, qid = r["dataset"], r["query_id"]
        ql = r["new_label"]
        if not isinstance(ql, dict) or "_error" in ql:
            lines.append(f"[{ds}/{qid}] query 解析失败，跳过")
            lines.append("")
            continue

        q_persons = {norm(p, person_rev) for p in (ql.get("persons") or [])}
        q_locs = {norm(l, loc_rev) for l in (ql.get("locations") or [])}
        q_plot = norm(ql.get("plot_unit", ""), plot_rev)

        lines.append("=" * 70)
        lines.append(f"[{ds}/{qid}]")
        lines.append(f"query 标准名: persons={sorted(q_persons)} locations={sorted(q_locs)} plot_unit={q_plot!r}")
        lines.append("")

        # 该 query 的核心 chunk
        core_rows = [c for c in (r for x in chunk_data if x["query_id"] == qid and x["dataset"] == ds for c in x["core_chunks"])]
        core_rows = []
        for x in chunk_data:
            if x["query_id"] == qid and x["dataset"] == ds:
                core_rows = x["core_chunks"]
                break

        for c in core_rows:
            lab = c["label"]
            if not isinstance(lab, dict) or "_error" in lab:
                continue
            n_chunk_total += 1
            c_persons = {norm(p, person_rev) for p in (lab.get("persons") or [])}
            c_locs = {norm(l, loc_rev) for l in (lab.get("locations") or [])}
            c_plot = norm(lab.get("plot_unit", ""), plot_rev)

            p_over = q_persons & c_persons
            l_over = q_locs & c_locs
            plot_match = (q_plot != "" and q_plot == c_plot)

            if plot_match:
                n_plot_match += 1
            if p_over:
                n_person_overlap += 1
            if l_over:
                n_loc_overlap += 1
            if plot_match or p_over or l_over:
                n_any_match += 1

            flag = "MATCH" if (plot_match or p_over or l_over) else "NOMATCH"
            lines.append(
                f"  [{c['chunk_id'][-5:]}/{c['genre']}] {flag}  "
                f"plot({q_plot!r} vs {c_plot!r})  "
                f"person交集={sorted(p_over)}  loc交集={sorted(l_over)}"
            )
        lines.append("")

    lines.append("=" * 70)
    lines.append("汇总（核心 chunk 总数 {}）:".format(n_chunk_total))
    lines.append(f"  plot_unit 匹配: {n_plot_match}/{n_chunk_total}")
    lines.append(f"  persons 有交集: {n_person_overlap}/{n_chunk_total}")
    lines.append(f"  locations 有交集: {n_loc_overlap}/{n_chunk_total}")
    lines.append(f"  任一维度匹配: {n_any_match}/{n_chunk_total}")

    out = RESULTS_DIR / "match_analysis.txt"
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"done -> {out}")


if __name__ == "__main__":
    main()
