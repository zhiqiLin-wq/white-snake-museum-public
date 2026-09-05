"""抽 unretrievable query（核心 rel=2 在 dense/sparse top-50 都排不上）。

复用 recall_rank_attribution.json 的 detail（category == "unretrievable"），
用 src_only qrels 重新筛选（chunk_id 是 src_only rel=2）。
"""
import json
from collections import defaultdict
from pathlib import Path

RESULTS = Path("rag_eval/results")
GT = Path("rag_eval/ground_truth")
DATASETS = ["academic_deepseek", "academic_qwen", "daily_deepseek", "daily_qwen"]

N_PICK = 40


def load_qrels(path):
    qrels = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) < 3:
                continue
            qid, did = parts[0], parts[1]
            rel = int(parts[2])
            if rel == 0:
                continue
            qrels.setdefault(qid, {})[did] = rel
    return qrels


def load_queries(path):
    qs = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            q = json.loads(line)
            qs[q["query_id"]] = q
    return qs


def main():
    rank = json.load(open(RESULTS / "recall_rank_attribution.json", encoding="utf-8"))
    detail_map = {}
    for x in rank["detail"]:
        detail_map[(x["dataset"], x["query_id"], x["chunk_id"])] = x["category"]

    candidates = []
    for ds in DATASETS:
        qrels = load_qrels(GT / f"qrels_{ds}_src_only.tsv")
        qs = load_queries(GT / f"queries_{ds}.jsonl")
        for qid, rels in qrels.items():
            core = {c for c, r in rels.items() if r == 2}
            if not core:
                continue
            unretrievable = [c for c in core if detail_map.get((ds, qid, c)) == "unretrievable"]
            if unretrievable:
                qcat = qs.get(qid, {}).get("category", "")
                candidates.append((ds, qid, qcat, unretrievable))

    print(f"有 src_only unretrievable 核心的 query 总数: {len(candidates)}")
    by_cat = defaultdict(int)
    for c in candidates:
        by_cat[c[2]] += 1
    print("按 category:", dict(by_cat))

    candidates.sort(key=lambda c: -len(c[3]))
    picked = []
    per_ds = defaultdict(int)
    for c in candidates:
        ds = c[0]
        if per_ds[ds] >= 10:
            continue
        picked.append(c)
        per_ds[ds] += 1
        if len(picked) >= N_PICK:
            break

    print(f"抽中 {len(picked)} 个 unretrievable query")
    out = [
        {"dataset": ds, "query_id": qid, "category": qcat, "unretrievable_chunks": uc}
        for ds, qid, qcat, uc in picked
    ]
    out_path = RESULTS / "picked_unretrievable_queries.json"
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[OUT] {out_path}")


if __name__ == "__main__":
    main()
