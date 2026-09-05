"""融合策略消融 — 在 core recall 目标下对比 4 种融合。

背景:
    当前融合 _score_normalized_fusion 用 min-max 归一化 + 线性加权，
    且「不在某一路候选里」的 chunk 缺省记 0.0，会系统性压低单路强命中的核心。
    本脚本手动复现生产管线（dense top-100 + sparse top-100 -> 融合 -> research 降权
    -> 版本 boost -> top-10），只替换融合函数，对比:

      minmax_zero    : 当前生产（min-max 归一化，缺省 0.0，dense=0.5）
      minmax_median  : 缺省改成 0.5（min-max 中点），修单路命中被压低
      rrf            : rank 融合（对单路命中天然友好）
      consensus_bonus: 缺省 0.5 + 两路共识加成，同时修缺省并信任共识

主指标 core recall@10（macro），护栏 raw precision@10 + raw recall@10（rel>=1, macro）。

Usage:
    python rag_eval/fusion_ablation_core.py --limit 10        # smoke test
    python rag_eval/fusion_ablation_core.py                   # 全量
"""
import argparse
import json
import sys
import time
from pathlib import Path
from typing import Callable

PROJECT_ROOT = Path(__file__).resolve().parent.parent
AGENT_DIR = PROJECT_ROOT / "agent"
for p in [str(PROJECT_ROOT), str(AGENT_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from rag_eval.detail_log import build_components, preprocess_query
from rag_eval.loader import load_corpus, load_queries, load_qrels

DATASETS = ["academic_deepseek", "academic_qwen", "daily_deepseek", "daily_qwen"]
DENSE_K = 100   # 对齐生产 dense_candidate_k
SPARSE_K = 100  # 对齐生产 sparse_candidate_k
TOP_K = 10
DENSE_WEIGHT = 0.5


# ==============================================================================
# 融合函数（输入 [(chunk_id, score)]，输出 [(chunk_id, fused_score)] 降序）
# ==============================================================================
def _minmax(d: dict[str, float]) -> dict[str, float]:
    if not d:
        return {}
    lo, hi = min(d.values()), max(d.values())
    if hi == lo:
        return {k: 0.5 for k in d}
    return {k: (v - lo) / (hi - lo) for k, v in d.items()}


def fusion_minmax_zero(dense, sparse, w=DENSE_WEIGHT):
    dn = _minmax({c: s for c, s in dense})
    sn = _minmax({c: s for c, s in sparse})
    fused = {}
    for cid in set(dn) | set(sn):
        fused[cid] = w * dn.get(cid, 0.0) + (1 - w) * sn.get(cid, 0.0)
    return sorted(fused.items(), key=lambda x: x[1], reverse=True)


def fusion_minmax_median(dense, sparse, w=DENSE_WEIGHT):
    dn = _minmax({c: s for c, s in dense})
    sn = _minmax({c: s for c, s in sparse})
    fused = {}
    for cid in set(dn) | set(sn):
        fused[cid] = w * dn.get(cid, 0.5) + (1 - w) * sn.get(cid, 0.5)
    return sorted(fused.items(), key=lambda x: x[1], reverse=True)


def fusion_rrf(dense, sparse, w=DENSE_WEIGHT, k=60):
    dr = {c: i + 1 for i, (c, _) in enumerate(dense)}
    sr = {c: i + 1 for i, (c, _) in enumerate(sparse)}
    fused = {}
    for cid in set(dr) | set(sr):
        fused[cid] = w / (k + dr.get(cid, 10_000)) + (1 - w) / (k + sr.get(cid, 10_000))
    return sorted(fused.items(), key=lambda x: x[1], reverse=True)


def fusion_consensus_bonus(dense, sparse, w=DENSE_WEIGHT, bonus=0.1):
    dd = {c: s for c, s in dense}
    ss = {c: s for c, s in sparse}
    dn = _minmax(dd)
    sn = _minmax(ss)
    fused = {}
    for cid in set(dd) | set(ss):
        score = w * dn.get(cid, 0.5) + (1 - w) * sn.get(cid, 0.5)
        if cid in dd and cid in ss:
            score += bonus
        fused[cid] = score
    return sorted(fused.items(), key=lambda x: x[1], reverse=True)


FUSIONS: dict[str, Callable] = {
    "minmax_zero": fusion_minmax_zero,
    "minmax_median": fusion_minmax_median,
    "rrf": fusion_rrf,
    "consensus_bonus": fusion_consensus_bonus,
}


# ==============================================================================
# 后处理（对齐生产 retriever.py 的 research_penalty / version_boost）
# ==============================================================================
def _meta_matches(meta: dict, cond: dict) -> bool:
    for k, v in cond.items():
        mv = meta.get(k)
        if isinstance(v, list):
            if mv not in v:
                return False
        elif mv != v:
            return False
    return True


def apply_post(fused, meta_lookup, research_penalty, version_boost, boost_factor):
    out = []
    for cid, score in fused:
        meta = meta_lookup.get(cid, {})
        if research_penalty < 1.0 and meta.get("source_type") == "research_literature":
            score *= research_penalty
        if version_boost and boost_factor > 1.0 and _meta_matches(meta, version_boost):
            score *= boost_factor
        out.append((cid, score))
    out.sort(key=lambda x: x[1], reverse=True)
    return out


def macro_mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    from server.rag.config import rag_config

    print("[INIT] 加载组件 ...", flush=True)
    embedder, vector_store, bm25, expander = build_components()
    gt = PROJECT_ROOT / "rag_eval" / "ground_truth"
    corpus = load_corpus(gt / "corpus.jsonl")
    meta_lookup = {cid: doc.get("metadata", {}) for cid, doc in corpus.items()}

    # 每个策略的逐 query 指标
    acc = {name: {"core": [], "prec": [], "recall": []} for name in FUSIONS}

    t0 = time.monotonic()
    n_processed = 0
    for ds in DATASETS:
        queries = load_queries(gt / f"queries_{ds}.jsonl")
        qrels = load_qrels(gt / f"qrels_{ds}_augmented.tsv")
        print(f"\n[{ds}] {len(queries)} queries", flush=True)
        for qi, q in enumerate(queries):
            if args.limit and n_processed >= args.limit:
                break
            qid = q.get("query_id", "")
            rels = qrels.get(qid, {})
            core_set = {c for c, r in rels.items() if r == 2}
            rel1_set = {c for c, r in rels.items() if r == 1}
            all_set = core_set | rel1_set
            if not all_set:
                continue
            n_processed += 1

            pp = preprocess_query(q.get("text", ""), expander)

            # dense top-100 / sparse top-100（原始分数）
            qvec = embedder.embed_query(pp["dense_query"])
            dense_raw = vector_store.query(qvec, top_k=DENSE_K)
            sparse_raw = bm25.search(
                pp["sparse_query"], top_k=SPARSE_K, query_weights=pp["sparse_weights"]
            )
            dense_list = [(r["chunk_id"], r["score"]) for r in dense_raw if r.get("chunk_id")]
            sparse_list = [(r["chunk_id"], r["bm25_score"]) for r in sparse_raw if r.get("chunk_id")]

            for name, fn in FUSIONS.items():
                fused = fn(dense_list, sparse_list)
                fused = apply_post(
                    fused, meta_lookup, pp["research_penalty"],
                    pp["version_boost"], rag_config.version_boost_factor,
                )
                top10 = [cid for cid, _ in fused[:TOP_K]]

                hit_core = sum(1 for c in top10 if c in core_set)
                hit_all = sum(1 for c in top10 if c in all_set)
                if core_set:
                    acc[name]["core"].append(hit_core / len(core_set))
                if all_set:
                    acc[name]["prec"].append(hit_all / TOP_K)
                    acc[name]["recall"].append(hit_all / len(all_set))

            if (qi + 1) % 50 == 0:
                print(f"  [{ds}] {qi + 1}/{len(queries)} ({time.monotonic() - t0:.0f}s)", flush=True)
        if args.limit and n_processed >= args.limit:
            break

    # 汇总
    print("\n" + "=" * 74)
    print(f"融合消融结果 — core recall 主指标（共 {n_processed} 条有标注 query）")
    print("=" * 74)
    print(f"{'策略':<18}{'core recall@10':>15}{'raw prec@10':>13}{'raw recall@10':>15}")
    print("-" * 74)
    rows = []
    for name in FUSIONS:
        c = macro_mean(acc[name]["core"])
        p = macro_mean(acc[name]["prec"])
        r = macro_mean(acc[name]["recall"])
        rows.append((name, c, p, r))
        print(f"{name:<18}{c:>15.4f}{p:>13.4f}{r:>15.4f}")
    print("=" * 74)

    base = rows[0]
    print("\n相对基线(minmax_zero)的 core recall 变化:")
    for name, c, p, r in rows[1:]:
        print(f"  {name:<18} {c - base[1]:+.4f}")

    out = PROJECT_ROOT / "rag_eval" / "results" / "fusion_ablation_core.json"
    out.write_text(json.dumps({
        "n_processed": n_processed,
        "results": {name: {
            "core_recall": macro_mean(acc[name]["core"]),
            "raw_precision": macro_mean(acc[name]["prec"]),
            "raw_recall": macro_mean(acc[name]["recall"]),
        } for name in FUSIONS},
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n结果已写: {out}")


if __name__ == "__main__":
    main()
