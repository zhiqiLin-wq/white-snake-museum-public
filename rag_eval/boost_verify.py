"""结构化标签 + boost 验证脚本：验证「标签 boost 能否把 sortable 核心捞回 top-10」。

分三步跑（每步落盘，便于先看结果再烧 token）:
    python rag_eval/boost_verify.py --phase retrieve   # 抽 sortable query + 检索候选集
    python rag_eval/boost_verify.py --phase label      # 对候选集 chunk 打标（耗 LLM）
    python rag_eval/boost_verify.py --phase eval       # 归一 + boost 重排序 + 三个硬指标

设计文档: results/结构化标签boost方案设计.md（第五/六节）。

关键口径:
- 核心 = src_only qrels 里 rel=2（排除 B 类污染核心）。
- sortable 核心 = boost 前 best_rank(dense/sparse) 在 top-11~50。
- boost 前基线 = minmax 融合 + research_penalty + version_boost（对齐生产，复用
  fusion_ablation_core 的 fusion_minmax_zero + apply_post）。
- boost 后 = fused_base + 标签 boost，再乘同样的 research_penalty + version_boost，
  唯一差异就是「加没加标签 boost」，干净衡量增量。
"""
import argparse
import asyncio
import json
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
AGENT_DIR = PROJECT_ROOT / "agent"
for p in [str(PROJECT_ROOT), str(AGENT_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from rag_eval.detail_log import build_components, preprocess_query
from rag_eval.loader import load_corpus, load_queries, load_qrels
from rag_eval.relabel_v2 import (
    LIT_CHUNK_SYSTEM, RESEARCH_CHUNK_SYSTEM, QUERY_SYSTEM,
    GENRE_META, _parse_json, extract, genre_info,
)
from rag_eval.scripts.generate_eval_dataset import PROVIDER_CONFIGS
from server.llm.openai_compatible_provider import GenericOpenAIProvider
from server.rag.retriever import _meta_matches

RESULTS_DIR = PROJECT_ROOT / "rag_eval" / "results"
GT = PROJECT_ROOT / "rag_eval" / "ground_truth"
DATASETS = ["academic_deepseek", "academic_qwen", "daily_deepseek", "daily_qwen"]

D_TOP_K = 50
S_TOP_K = 50
TOP_K = 10
DENSE_WEIGHT = 0.5
CHUNK_TRUNC = 900
N_PICK = 14          # 抽样 query 总数
N_COMPARISON = 6     # comparison 题优先抽几条（测版本元信息 boost）
N_COMPARISON_PER_DS = 2
N_OTHER_PER_DS = 2

CANDIDATES_FILE = RESULTS_DIR / "boost_candidates.json"
LABELS_FILE = RESULTS_DIR / "boost_labels.json"

# 对比维度名 -> GENRE_META 字段名
DIM_FIELD = {"朝代": "dynasty", "文本": "text", "体裁": "name", "作者": "author"}

# 版本归属/源流强信号词（comparison 题里命中这些词的优先抽，测 GENRE_META 版本元信息 boost）
VERSION_KW = ("朝代", "承袭", "源流", "改编", "清代", "明代", "宋代", "唐代", "哪个版本")


# ==============================================================================
# 词表归一（复用 normalize_and_match 的 build_normalizer）
# ==============================================================================
def build_normalizer(vocab: list) -> dict:
    m = {}
    for entry in vocab:
        std = entry["标准名"]
        m[std] = std
        for a in entry["别名"]:
            m[a] = std
    return m


def load_normalizers() -> tuple[dict, dict, dict]:
    merged = json.load(open(RESULTS_DIR / "merged_vocab.json", encoding="utf-8"))
    plot_norm = build_normalizer(merged["plot_units"])
    topic_norm = build_normalizer(merged["topics"])
    loc_norm = build_normalizer(merged["locations"])
    return plot_norm, topic_norm, loc_norm


def norm(v, normalizer):
    return normalizer.get(v, v) if v else ""


# ==============================================================================
# 融合函数（复用 fusion_ablation_core 的 minmax_zero）
# ==============================================================================
def _minmax(d: dict[str, float]) -> dict[str, float]:
    if not d:
        return {}
    lo, hi = min(d.values()), max(d.values())
    if hi == lo:
        return {k: 0.5 for k in d}
    return {k: (v - lo) / (hi - lo) for k, v in d.items()}


def fusion_minmax_zero(dense, sparse, w=DENSE_WEIGHT):
    dn = _minmax(dict(dense))
    sn = _minmax(dict(sparse))
    fused = {}
    for cid in set(dn) | set(sn):
        fused[cid] = w * dn.get(cid, 0.0) + (1 - w) * sn.get(cid, 0.0)
    return fused


def apply_post(fused: dict, meta_lookup, research_penalty, version_boost, boost_factor):
    """research_penalty + version_boost 后处理（对齐生产），返回 {cid: score}。"""
    out = {}
    for cid, score in fused.items():
        meta = meta_lookup.get(cid, {})
        if research_penalty < 1.0 and meta.get("source_type") == "research_literature":
            score *= research_penalty
        if version_boost and boost_factor > 1.0 and _meta_matches(meta, version_boost):
            score *= boost_factor
        out[cid] = score
    return out


# ==============================================================================
# 对比维度解析（兼容 dict 新格式 / list 旧格式）
# ==============================================================================
def cmp_dim_names(cmp_dim) -> set:
    if isinstance(cmp_dim, dict):
        return set(cmp_dim.keys())
    if isinstance(cmp_dim, list):
        return set(cmp_dim)
    return set()


def cmp_has_version_dim(cmp_dim) -> bool:
    return bool(cmp_dim_names(cmp_dim) & {"朝代", "文本", "体裁", "作者"})


def cmp_has_appellation(cmp_dim) -> bool:
    return "称谓" in cmp_dim_names(cmp_dim)


def version_matches(genre: str, cmp_dim) -> bool:
    """GENRE_META[genre] 的字段是否命中对比维度的具体值。"""
    if genre not in GENRE_META or not isinstance(cmp_dim, dict):
        return False
    meta = GENRE_META[genre]
    for dim, val in cmp_dim.items():
        if dim not in DIM_FIELD or not val:
            continue
        fv = meta.get(DIM_FIELD[dim])
        if not fv:
            continue
        if fv == val:
            return True
        # 宽松匹配：LLM 打的具体值可能带后缀（"清代" vs "清"）
        if fv in val or val in fv:
            return True
    return False


# ==============================================================================
# plot_detail overlap（char bigram Jaccard，纯代码）
# ==============================================================================
def _bigrams(s: str) -> set:
    s = re.sub(r"\s+", "", s)
    return {s[i:i + 2] for i in range(len(s) - 1)}


def plot_detail_overlap(a: str, b: str, thresh: float = 0.15) -> bool:
    if not a or not b:
        return False
    ba, bb = _bigrams(a), _bigrams(b)
    if not ba or not bb:
        return False
    inter = len(ba & bb)
    if inter == 0:
        return False
    jac = inter / len(ba | bb)
    return jac >= thresh


# ==============================================================================
# 标签 boost 计算（完整版，按 query_type / chunk 双轨分流）
# ==============================================================================
def _persons_set(persons) -> set:
    return {p.get("canonical", "") for p in (persons or []) if isinstance(p, dict)}


def _raw_set(persons) -> set:
    return {p.get("raw", "") for p in (persons or []) if isinstance(p, dict)}


def _locs_set(locs) -> set:
    return {l.get("canonical", "") for l in (locs or []) if isinstance(l, dict)}


def compute_boost(q, c_label, c_genre, summary_sim, plot_norm, topic_norm, loc_norm):
    """按设计文档第五节的完整 boost 公式算标签 boost。q/c_label 均为 dict。"""
    boost = 0.0
    if not isinstance(q, dict) or "_error" in q:
        return boost
    if not isinstance(c_label, dict) or "_error" in c_label:
        return boost

    qtype = q.get("query_type", "")
    q_plot_unit = norm(q.get("plot_unit", ""), plot_norm)
    q_plot_detail = q.get("plot_detail", "") or ""
    q_topic = norm(q.get("分析主题", ""), topic_norm)
    q_persons = _persons_set(q.get("persons"))
    q_persons_raw = _raw_set(q.get("persons"))
    q_locs = {norm(x, loc_norm) for x in _locs_set(q.get("locations")) if x}
    q_cmp = q.get("对比维度")

    if c_genre == "research_literature":
        # ---- 研究文献 chunk ----
        c_topic = norm(c_label.get("分析主题", ""), topic_norm)
        c_vers = set(c_label.get("涉及版本") or [])
        ent = c_label.get("引用实体") or {}
        c_ent_persons = {p for p in (ent.get("persons") or []) if isinstance(p, str)}
        c_ent_locs = {norm(l, loc_norm) for l in (ent.get("locations") or []) if l}
        c_plot = {norm(p, plot_norm) for p in (c_label.get("引用情节") or []) if p}

        if q_topic and c_topic == q_topic:
            boost += 0.05
        if cmp_has_version_dim(q_cmp) and any(version_matches(g, q_cmp) for g in c_vers):
            boost += 0.05
        if q_plot_unit and q_plot_unit in c_plot:
            boost += 0.05
        boost += 0.03 * len(q_persons & c_ent_persons)
        boost += 0.02 * len(q_locs & c_ent_locs)
    else:
        # ---- 文学文本 chunk ----
        c_plot_unit = norm(c_label.get("plot_unit", ""), plot_norm)
        c_plot_detail = c_label.get("plot_detail", "") or ""
        c_persons = _persons_set(c_label.get("persons"))
        c_persons_raw = _raw_set(c_label.get("persons"))
        c_locs = {norm(x, loc_norm) for x in _locs_set(c_label.get("locations")) if x}

        if q_plot_unit and c_plot_unit == q_plot_unit:
            boost += 0.10
        if plot_detail_overlap(q_plot_detail, c_plot_detail):
            boost += 0.08
        boost += 0.03 * len(q_persons & c_persons)
        boost += 0.02 * len(q_locs & c_locs)
        if cmp_has_appellation(q_cmp):
            boost += 0.05 * len(q_persons_raw & c_persons_raw)

    # ---- 版本元信息（comparison 题，文学+研究都适用）----
    if qtype == "comparison题" and cmp_has_version_dim(q_cmp):
        if c_genre in GENRE_META and version_matches(c_genre, q_cmp):
            boost += 0.05

    # ---- 摘要第二路语义（所有 chunk）----
    boost += 0.05 * summary_sim

    return boost


# ==============================================================================
# Phase 1: 抽 sortable query + 检索候选集
# ==============================================================================
def rank_map(results: list[tuple], top_k: int) -> dict[str, int]:
    rmap = {}
    for i, (cid, _s) in enumerate(results[:top_k], start=1):
        if cid:
            rmap[cid] = i
    return rmap


def select_sortable_queries() -> list[tuple[str, str]]:
    rank = json.load(open(RESULTS_DIR / "recall_rank_attribution.json", encoding="utf-8"))
    by_query = defaultdict(list)
    for x in rank["detail"]:
        if x["category"] == "sortable":
            by_query[(x["dataset"], x["query_id"])].append(x)

    qcat, qtext = {}, {}
    for ds in DATASETS:
        for q in load_queries(GT / f"queries_{ds}.jsonl"):
            qcat[(ds, q["query_id"])] = q.get("category", "")
            qtext[(ds, q["query_id"])] = q.get("text", "")

    comparison = [(k, v) for k, v in by_query.items() if qcat.get(k) == "comparison"]
    others = [(k, v) for k, v in by_query.items() if qcat.get(k) != "comparison"]

    picked = []
    per_ds = defaultdict(int)
    # comparison 题优先：版本归属关键词命中的排前，再按 sortable 核心数降序，跨 dataset 均衡
    for (ds, qid), chunks in sorted(
        comparison,
        key=lambda kv: (-sum(1 for kw in VERSION_KW if kw in qtext.get(kv[0], "")), -len(kv[1])),
    ):
        if per_ds[ds] >= N_COMPARISON_PER_DS:
            continue
        picked.append((ds, qid))
        per_ds[ds] += 1
        if len(picked) >= N_COMPARISON:
            break

    # 其余类（plot/entity）补足到 N_PICK
    per_ds2 = defaultdict(int)
    for (ds, qid), chunks in sorted(others, key=lambda kv: -len(kv[1])):
        if per_ds2[ds] >= N_OTHER_PER_DS:
            continue
        picked.append((ds, qid))
        per_ds2[ds] += 1
        if len(picked) >= N_PICK:
            break

    return picked


def phase_retrieve():
    from server.rag.config import rag_config

    print("[INIT] 加载 embedder / vector_store / BM25 ...", flush=True)
    embedder, vector_store, bm25, expander = build_components()

    corpus = load_corpus(GT / "corpus.jsonl")
    meta_lookup = {cid: doc.get("metadata", {}) for cid, doc in corpus.items()}

    picked = select_sortable_queries()
    print(f"[SELECT] 抽中 {len(picked)} 条 sortable query: {picked}", flush=True)

    # 预加载 query 文本 / category / src_only qrels（按 dataset 分，避免 query_id 跨数据集覆盖）
    qtext_by_ds, qcat_by_ds, qrels_by_ds = {}, {}, {}
    for ds in DATASETS:
        qs = load_queries(GT / f"queries_{ds}.jsonl")
        qtext_by_ds[ds] = {q["query_id"]: q["text"] for q in qs}
        qcat_by_ds[ds] = {q["query_id"]: q.get("category", "") for q in qs}
        qrels_by_ds[ds] = load_qrels(GT / f"qrels_{ds}_src_only.tsv")

    out = []
    t0 = time.monotonic()
    for ds, qid in picked:
        qtext = qtext_by_ds[ds].get(qid, "")
        rels = qrels_by_ds[ds].get(qid, {})
        core = {c for c, r in rels.items() if r == 2}
        rel1 = {c for c, r in rels.items() if r == 1}
        if not core:
            print(f"  [skip] {ds}/{qid} src_only 无 rel=2 核心", flush=True)
            continue

        pp = preprocess_query(qtext, expander)

        # D / S 原始分数（不走 HybridRetriever，避免 post 污染）
        qvec = embedder.embed_query(pp["dense_query"])
        dense_raw = vector_store.query(qvec, top_k=D_TOP_K)
        sparse_raw = bm25.search(pp["sparse_query"], top_k=S_TOP_K, query_weights=pp["sparse_weights"])
        dense_list = [(r["chunk_id"], r["score"]) for r in dense_raw if r.get("chunk_id")]
        sparse_list = [(r["chunk_id"], r["bm25_score"]) for r in sparse_raw if r.get("chunk_id")]

        d_rank = rank_map(dense_list, D_TOP_K)
        s_rank = rank_map(sparse_list, S_TOP_K)

        # 判定 src_only 核心的 best_rank / category
        core_best = {}
        n_sortable = 0
        for cid in core:
            dr = d_rank.get(cid, 999)
            sr = s_rank.get(cid, 999)
            best = min(dr, sr)
            core_best[cid] = best if best <= 50 else None
            if 11 <= best <= 50:
                n_sortable += 1
        if n_sortable == 0:
            print(f"  [skip] {ds}/{qid} src_only 口径下无 sortable 核心", flush=True)
            continue

        # boost 前基线：minmax 融合 + post
        fused = fusion_minmax_zero(dense_list, sparse_list)
        fused = apply_post(fused, meta_lookup, pp["research_penalty"], pp["version_boost"],
                           rag_config.version_boost_factor)
        f_top10_before = [cid for cid, _ in sorted(fused.items(), key=lambda x: x[1], reverse=True)[:TOP_K]]

        # 候选集 = D∪S top-50（去重）
        candidates = {}
        for cid, score in dense_list:
            candidates[cid] = {"chunk_id": cid, "dense_score": score, "bm25_score": None,
                               "dense_rank": d_rank.get(cid), "sparse_rank": s_rank.get(cid)}
        for cid, score in sparse_list:
            if cid in candidates:
                candidates[cid]["bm25_score"] = score
                candidates[cid]["sparse_rank"] = s_rank.get(cid)
            else:
                candidates[cid] = {"chunk_id": cid, "dense_score": None, "bm25_score": score,
                                   "dense_rank": d_rank.get(cid), "sparse_rank": s_rank.get(cid)}

        cand_list = []
        for cid, info in candidates.items():
            meta = meta_lookup.get(cid, {})
            cand_list.append({
                "chunk_id": cid,
                "genre": meta.get("genre", ""),
                "source_type": meta.get("source_type", ""),
                "dense_score": info["dense_score"],
                "bm25_score": info["bm25_score"],
                "dense_rank": info["dense_rank"],
                "sparse_rank": info["sparse_rank"],
                "text": corpus.get(cid, {}).get("text", "")[:CHUNK_TRUNC],
                "is_core": cid in core,
            })

        out.append({
            "dataset": ds,
            "query_id": qid,
            "query_text": qtext,
            "category": qcat_by_ds[ds].get(qid, ""),
            "research_penalty": pp["research_penalty"],
            "version_boost": pp["version_boost"],
            "core": sorted(core),
            "core_best_rank": core_best,
            "rel1": sorted(rel1),
            "candidates": cand_list,
            "f_top10_before": f_top10_before,
        })
        print(f"  [done] {ds}/{qid} core={len(core)} sortable={n_sortable} "
              f"候选={len(cand_list)} ({time.monotonic() - t0:.0f}s)", flush=True)

    CANDIDATES_FILE.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[OUT] 候选集已写: {CANDIDATES_FILE}（共 {len(out)} 条有效 query）")


# ==============================================================================
# Phase 2: 打标（query + 候选集 chunk）
# ==============================================================================
async def phase_label():
    data = json.load(open(CANDIDATES_FILE, encoding="utf-8"))
    provider = GenericOpenAIProvider(
        api_key=PROVIDER_CONFIGS["deepseek"]["api_key"],
        base_url=PROVIDER_CONFIGS["deepseek"]["base_url"],
        default_model="deepseek-v4-flash",
    )
    sem = asyncio.Semaphore(6)

    def chunk_prompt(c):
        g = c["genre"]
        if g == "research_literature":
            return RESEARCH_CHUNK_SYSTEM
        return LIT_CHUNK_SYSTEM.format(genre_info=genre_info(g))

    out = []
    for r in data:
        ds, qid = r["dataset"], r["query_id"]
        qlabel = await extract(provider, QUERY_SYSTEM.format(category=r["category"]), r["query_text"], sem)

        # 并发打标候选 chunk（sem 限制并发，避免串行慢）
        tasks = [extract(provider, chunk_prompt(c), c["text"], sem) for c in r["candidates"]]
        labels_list = await asyncio.gather(*tasks)
        chunk_labels = {c["chunk_id"]: lab for c, lab in zip(r["candidates"], labels_list)}

        out.append({
            "dataset": ds,
            "query_id": qid,
            "query_label": qlabel,
            "chunk_labels": chunk_labels,
        })
        print(f"  [done] {ds}/{qid} 打标 {len(chunk_labels)} chunks", flush=True)

    await provider.close()
    LABELS_FILE.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[OUT] 标签已写: {LABELS_FILE}")


# ==============================================================================
# Phase 3: 归一 + boost 重排序 + 三个硬指标
# ==============================================================================
def _cosine(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-9
    return float(a @ b / denom)


def build_embedder():
    from server.rag.config import rag_config
    from server.rag.embedder import Embedder

    print("[INIT] 加载 embedder（summary 语义路）...", flush=True)
    emb = Embedder(rag_config.embedding_model, local_path=rag_config.embedding_local_path or None)
    emb.start_loading()
    waited = 0
    while not emb.is_ready and waited < 300:
        time.sleep(1)
        waited += 1
    if not emb.is_ready:
        raise RuntimeError("embedder 未就绪")
    return emb


def phase_eval():
    from server.rag.config import rag_config

    candidates = json.load(open(CANDIDATES_FILE, encoding="utf-8"))
    labels = json.load(open(LABELS_FILE, encoding="utf-8"))
    labels_by_key = {(x["dataset"], x["query_id"]): x for x in labels}
    plot_norm, topic_norm, loc_norm = load_normalizers()

    corpus = load_corpus(GT / "corpus.jsonl")
    meta_lookup = {cid: doc.get("metadata", {}) for cid, doc in corpus.items()}

    embedder = build_embedder()

    lines = []
    lines.append("=" * 80)
    lines.append("结构化标签 boost 验证结果")
    lines.append(f"boost_factor={rag_config.version_boost_factor}  dense_weight={DENSE_WEIGHT}")
    lines.append("=" * 80)

    # 汇总指标
    sum_salvaged = 0
    sum_sortable = 0
    core_recall_before, core_recall_after = [], []
    side_effect_lost, side_effect_total = 0, 0

    for r in candidates:
        ds, qid = r["dataset"], r["query_id"]
        key = (ds, qid)
        if key not in labels_by_key:
            continue
        qlabel = labels_by_key[key]["query_label"]
        chunk_labels = labels_by_key[key]["chunk_labels"]

        core = set(r["core"])
        rel1 = set(r["rel1"])
        rel_all = core | rel1

        # sortable 核心 = boost 前 best_rank 在 11~50
        sortable_core = {cid for cid in core if r["core_best_rank"].get(cid) and 11 <= r["core_best_rank"][cid] <= 50}

        # 计算每个候选 chunk 的 fused_base + boost + post -> final
        # 先归一化 summary（批量 embed）
        q_summary = (qlabel.get("summary", "") or "") if isinstance(qlabel, dict) else ""
        cid_to_summary = {}
        for c in r["candidates"]:
            lab = chunk_labels.get(c["chunk_id"])
            if isinstance(lab, dict) and "_error" not in lab and lab.get("summary"):
                cid_to_summary[c["chunk_id"]] = lab["summary"]
        summary_sims = {}
        if q_summary and cid_to_summary:
            texts = [q_summary] + list(cid_to_summary.values())
            embs = embedder.embed(texts)
            q_emb = embs[0]
            for cid, emb in zip(cid_to_summary.keys(), embs[1:]):
                summary_sims[cid] = _cosine(q_emb, emb)

        # fused_base：对候选集 D/S 两路整体 minmax 归一（缺省 0，对齐 minmax_zero）
        dense_scores = {c["chunk_id"]: c["dense_score"] for c in r["candidates"] if c["dense_score"] is not None}
        sparse_scores = {c["chunk_id"]: c["bm25_score"] for c in r["candidates"] if c["bm25_score"] is not None}
        dn = _minmax(dense_scores)
        sn = _minmax(sparse_scores)

        final = {}
        for c in r["candidates"]:
            cid = c["chunk_id"]
            fused_base = DENSE_WEIGHT * dn.get(cid, 0.0) + (1 - DENSE_WEIGHT) * sn.get(cid, 0.0)
            boost = compute_boost(
                qlabel, chunk_labels.get(cid), c["genre"], summary_sims.get(cid, 0.0),
                plot_norm, topic_norm, loc_norm,
            )
            final[cid] = fused_base + boost

        # post：research_penalty + version_boost（对齐生产，与 boost 前基线一致）
        final = apply_post(final, meta_lookup, r["research_penalty"], r["version_boost"],
                           rag_config.version_boost_factor)
        top10_after = [cid for cid, _ in sorted(final.items(), key=lambda x: x[1], reverse=True)[:TOP_K]]

        # ---- 指标 1：核心捞回率 ----
        salvaged = sortable_core & set(top10_after)
        sum_salvaged += len(salvaged)
        sum_sortable += len(sortable_core)

        # ---- 指标 2：core recall@10 变化 ----
        if core:
            core_recall_before.append(len(set(r["f_top10_before"]) & core) / len(core))
            core_recall_after.append(len(set(top10_after) & core) / len(core))

        # ---- 指标 3：副作用（挤掉的真相关 chunk）----
        rel_in_before = set(r["f_top10_before"]) & rel_all
        rel_in_after = set(top10_after) & rel_all
        side_effect_total += len(rel_in_before)
        side_effect_lost += len(rel_in_before - rel_in_after)

        lines.append("")
        lines.append("-" * 80)
        lines.append(f"[{ds}/{qid}] core={len(core)} sortable={len(sortable_core)} "
                     f"候选={len(r['candidates'])}")
        lines.append(f"  boost 前 top-10: {r['f_top10_before']}")
        lines.append(f"  boost 后 top-10: {top10_after}")
        lines.append(f"  捞回 sortable 核心: {sorted(salvaged) if salvaged else '无'}")
        lines.append(f"  sortable 核心 best_rank: { {c: r['core_best_rank'].get(c) for c in sortable_core} }")
        lines.append(f"  core recall@10: {core_recall_before[-1] if core else 0:.3f} -> "
                     f"{core_recall_after[-1] if core else 0:.3f}")

    # ---- 汇总 ----
    lines.append("")
    lines.append("=" * 80)
    lines.append("汇总")
    lines.append("=" * 80)
    salvaged_rate = sum_salvaged / sum_sortable if sum_sortable else 0.0
    lines.append(f"指标1 核心捞回率: {sum_salvaged}/{sum_sortable} = {salvaged_rate:.3f} "
                 f"（sortable 核心从 top-11~50 进 top-10 的比例）")
    if core_recall_before:
        lines.append(f"指标2 core recall@10: {np.mean(core_recall_before):.4f} -> "
                     f"{np.mean(core_recall_after):.4f} "
                     f"（delta {np.mean(core_recall_after) - np.mean(core_recall_before):+.4f}）")
    else:
        lines.append("指标2 core recall@10: 无有效核心")
    side_rate = side_effect_lost / side_effect_total if side_effect_total else 0.0
    lines.append(f"指标3 副作用: 挤出 {side_effect_lost}/{side_effect_total} = {side_rate:.3f} "
                 f"（boost 前 top-10 里的真相关 chunk 被挤出的比例）")

    out_txt = RESULTS_DIR / "boost_eval.txt"
    out_txt.write_text("\n".join(lines), encoding="utf-8")
    json.dump({
        "salvaged": {"hit": sum_salvaged, "total": sum_sortable, "rate": salvaged_rate},
        "core_recall": {
            "before": float(np.mean(core_recall_before)) if core_recall_before else None,
            "after": float(np.mean(core_recall_after)) if core_recall_after else None,
        },
        "side_effect": {"lost": side_effect_lost, "total": side_effect_total, "rate": side_rate},
    }, open(RESULTS_DIR / "boost_eval.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2)

    print("\n".join(lines))
    print(f"\n[OUT] 结果已写: {out_txt} / boost_eval.json")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", required=True, choices=["retrieve", "label", "eval"])
    args = parser.parse_args()

    if args.phase == "retrieve":
        phase_retrieve()
    elif args.phase == "label":
        asyncio.run(phase_label())
    else:
        phase_eval()


if __name__ == "__main__":
    main()
