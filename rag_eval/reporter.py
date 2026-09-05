"""RAG 检索评估 — 报告生成器 (Phase 4)。

生成文本和 JSON 格式的评估报告，包含:
- 数据摘要 (Data Summary)
- 测试方法说明 (Methodology)
- 总览表 (Overall Results)
- 按类别分层 (Category Breakdown)
- 按难度分层 (Difficulty Breakdown)
- 统计检验报告 (Statistical Tests)
- 诊断指标报告 (Diagnostics — 含 Reranker 状态)
- 指标参考 (Metrics Reference — 公式 + 说明)
- CI 质量门禁
"""
import json
from collections import Counter
from datetime import datetime
from pathlib import Path


# ==============================================================================
# T04.01: generate_report 入口
# ==============================================================================
def generate_report(
    results: dict[str, dict],
    test_config: dict,
    output_dir: Path,
    timestamp: str | None = None,
    corpus: dict[str, dict] | None = None,
    queries: list[dict] | None = None,
    qrels: dict[str, dict[str, int]] | None = None,
    all_qrels: dict[str, dict[str, int]] | None = None,
) -> None:
    """生成全部报告文件。

    Args:
        results: {"D": {...}, "S": {...}, "F": {...}, "R": {...}}
        test_config: 测试配置字典
        output_dir: 输出目录
        timestamp: UTC 时间戳，None 时自动生成
        corpus: 语料库字典 (可选, 用于数据摘要)
        queries: 查询列表 (可选, 用于数据摘要)
        qrels: 正标注 qrels (可选, 用于数据摘要)
        all_qrels: 全量 qrels 含 hard negatives (可选, 用于数据摘要)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%dT%H%M%S")

    # 写入带时间戳的独立子目录，绝不复用
    run_dir = output_dir / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)

    # 构建评估元信息
    eval_meta = _build_eval_meta(results, corpus, queries, qrels, all_qrels)

    # 文本报告
    txt_content = _build_text_report(results, test_config, timestamp, eval_meta)
    txt_path = run_dir / "report.txt"
    txt_path.write_text(txt_content, encoding="utf-8")

    # JSON 报告
    json_data = _build_json_report(results, test_config, timestamp, eval_meta)
    json_path = run_dir / "report.json"
    json_path.write_text(
        json.dumps(json_data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


# ==============================================================================
# 评估元信息构建
# ==============================================================================
def _build_eval_meta(
    results: dict[str, dict],
    corpus: dict[str, dict] | None,
    queries: list[dict] | None,
    qrels: dict[str, dict[str, int]] | None,
    all_qrels: dict[str, dict[str, int]] | None,
) -> dict:
    """从原始数据构建评估摘要信息。"""
    meta: dict = {}

    # ---- Corpus 统计 ----
    if corpus is not None:
        meta["corpus_total_docs"] = len(corpus)
        genres = Counter(doc.get("metadata", {}).get("genre", "unknown")
                         for doc in corpus.values())
        dynasties = Counter(doc.get("metadata", {}).get("dynasty", "unknown")
                            for doc in corpus.values())
        meta["corpus_genres"] = dict(genres)
        meta["corpus_dynasties"] = dict(dynasties)
    else:
        meta["corpus_total_docs"] = "N/A"
        meta["corpus_genres"] = {}
        meta["corpus_dynasties"] = {}

    # ---- 查询统计 ----
    if queries is not None:
        meta["queries_total"] = len(queries)
        cats = Counter(q.get("category", "unknown") for q in queries)
        diffs = Counter(q.get("difficulty", "unknown") for q in queries)
        meta["queries_categories"] = dict(cats)
        meta["queries_difficulties"] = dict(diffs)

        # 从 per_query 统计标注覆盖情况
        n_annotated = 0
        n_skipped = 0
        n_hard_neg = 0
        for result in results.values():
            for pq in result.get("per_query", []):
                if pq.get("eval_skipped"):
                    n_skipped += 1
                    continue
                relevant_ids = pq.get("relevant_ids", [])
                hn_ids = pq.get("hard_negative_ids", [])
                if relevant_ids:
                    n_annotated += 1
                if hn_ids:
                    n_hard_neg += 1
            break  # 只看第一个配置的 per_query

        meta["queries_with_annotations"] = n_annotated
        meta["queries_with_hard_negatives"] = n_hard_neg
        meta["queries_skipped"] = n_skipped
        meta["annotation_coverage_pct"] = (
            round(n_annotated / max(len(queries), 1) * 100, 1)
            if queries else "N/A"
        )
    else:
        meta["queries_total"] = "N/A"
        meta["queries_categories"] = {}
        meta["queries_difficulties"] = {}
        meta["queries_with_annotations"] = "N/A"
        meta["queries_skipped"] = "N/A"
        meta["annotation_coverage_pct"] = "N/A"

    # ---- qrels 统计 ----
    if qrels is not None:
        meta["qrels_positive_groups"] = len(qrels)
        meta["qrels_positive_pairs"] = sum(len(docs) for docs in qrels.values())
    if all_qrels is not None:
        hn_count = sum(1 for docs in all_qrels.values()
                       for rel in docs.values() if rel == 0)
        meta["qrels_hard_negative_pairs"] = hn_count

    # ---- 标注来源 ----
    meta["annotation_source"] = (
        "AI 自动生成 (LLM multi-doc query generation + Reranker scoring + LLM reasoning)。"
        "qrels 中 relevance=2 为核心答案块，relevance=1 为辅助上下文块，relevance=0 为 BM25 hard negatives。"
        "⚠️ 此数据未经人工审核，指标值仅供参考 — 建议按 ANNOTATION_LOG.md 抽检复核。"
    )

    # ---- Reranker 状态 (已从评估管线移除) ----
    meta["reranker_active"] = None

    return meta


# ==============================================================================
# 报告章节构建器
# ==============================================================================

def _build_header(timestamp: str) -> str:
    return f"""\
{'='*80}
                   RAG 检索评估报告
{'='*80}
生成时间: {timestamp}
"""


def _build_data_summary(meta: dict) -> str:
    """一、数据摘要"""
    lines = [
        "",
        "一、数据摘要 (Data Summary)",
        "-" * 50,
        "",
        "【语料库】",
    ]
    if meta.get("corpus_total_docs") != "N/A":
        lines.append(f"  文档总数: {meta['corpus_total_docs']}")
        if meta.get("corpus_genres"):
            lines.append(f"  体裁分布: {json.dumps(meta['corpus_genres'], ensure_ascii=False)}")
        if meta.get("corpus_dynasties"):
            lines.append(f"  朝代分布: {json.dumps(meta['corpus_dynasties'], ensure_ascii=False)}")
    else:
        lines.append("  (未提供语料库数据)")

    lines.extend([
        "",
        "【查询集】",
    ])
    if meta.get("queries_total") != "N/A":
        lines.append(f"  查询总数: {meta['queries_total']}")
        if meta.get("queries_categories"):
            cats_str = ", ".join(f"{k}={v}" for k, v in sorted(meta['queries_categories'].items()))
            lines.append(f"  类别分布: {cats_str}")
        if meta.get("queries_difficulties"):
            diffs_str = ", ".join(f"{k}={v}" for k, v in sorted(meta['queries_difficulties'].items()))
            lines.append(f"  难度分布: {diffs_str}")
    else:
        lines.append("  (未提供查询数据)")

    lines.extend([
        "",
        "【标注覆盖 (qrels)】",
    ])
    if meta.get("annotation_coverage_pct") != "N/A":
        lines.append(f"  有正标注 (rel≥1) 查询数: {meta.get('queries_with_annotations', '?')}")
        lines.append(f"  有 hard negatives (rel=0) 查询数: {meta.get('queries_with_hard_negatives', '?')}")
        lines.append(f"  无任何正标注 (被跳过) 查询数: {meta.get('queries_skipped', '?')}")
        lines.append(f"  标注覆盖率: {meta.get('annotation_coverage_pct')}%")
        coverage = float(meta.get('annotation_coverage_pct', 0))
        if coverage < 50:
            lines.append("  ⚠️ 标注覆盖率过低，所有指标仅对标注子集计算，不代表全量查询表现。")
        elif coverage < 80:
            lines.append("  ⚡ 标注覆盖率一般，建议增加标注以提高指标代表性。")
        else:
            lines.append("  ✅ 标注覆盖率良好。")
    else:
        lines.append("  (未提供标注数据)")

    lines.extend([
        "",
        "【标注来源与有效性】",
        f"  {meta.get('annotation_source', 'N/A')}",
        "",
    ])
    return "\n".join(lines)


def _build_methodology() -> str:
    """二、测试方法说明"""
    return """
二、测试方法说明 (Methodology)
--------------------------------------------------

【测试配置】
  D (Dense Only)   — 仅语义向量检索 (Embedder → ChromaDB dense search)
  S (Sparse Only)  — 仅关键词检索 (BM25 Okapi)
  F (Fusion)       — 双路融合 (Dense + Sparse → RRF 融合, 无 Reranker)
  R (Full Pipeline) — 完整管道 (Dense + Sparse → RRF 融合 → Cross-Encoder Reranker 精排)

  设计目的: 通过 D→S→F→R 四路逐级对比，量化每个组件的增量贡献。
  - S vs D: 关键词检索 vs 语义检索 在中文古典文献上的优劣
  - F vs max(D,S): 融合是否产生 1+1>2 的效果
  - R vs F: Reranker 精排是否改善排序质量

【评估流程】
  1. 加载 corpus / queries / qrels 数据集
  2. 对每条 query，分别用 D/S/F/R 四种配置检索 top-10 结果
  3. 逐查询计算各指标 (recall, mrr, ndcg, ...)
  4. Bootstrap 重采样 (1000 次) 估计均值的 95% 置信区间
  5. 配对 Bootstrap 检验比较配置间差异的显著性
  6. 按类别/难度分层报告

【统计方法】
  Bootstrap 95% CI: 有放回重采样 per-query 指标得分，线性插值百分位法估计置信区间。
  Paired Bootstrap Test: 对配对查询随机交换 A/B 标签，构建零假设分布，双尾检验。
  效应量 (Cohen's d): d = (mean_a - mean_b) / sqrt((var_a + var_b) / 2)

"""


def _build_overall_table(results: dict[str, dict]) -> str:
    """三、总览结果对比表"""
    metrics = [
        "recall_at_10", "recall_at_5", "recall_at_3",
        "normalized_recall_at_10", "recall_at_10_core",
        "mrr", "ndcg_at_10", "binary_ndcg_at_10",
        "hit_at_1", "precision_at_10",
        "average_precision", "r_precision", "f1_at_10",
        "coverage_at_10", "false_positive_rate_at_10",
        "filter_precision_at_10",
    ]
    configs_order = ["D", "S", "F", "R"]

    lines = [
        "三、总览结果 (Overall Results)",
        "=" * 70,
    ]

    # 表头
    header = f"{'Metric':<22}"
    for cid in configs_order:
        if cid in results:
            header += f" {cid:<18}"
    lines.append(header)
    lines.append("-" * 70)

    for metric in metrics:
        row = f"{metric:<22}"
        for cid in configs_order:
            if cid not in results:
                row += f" {'N/A':<18}"
                continue
            agg = results[cid].get("aggregates", {}).get(metric, {})
            mean = agg.get("mean", 0.0)
            n = agg.get("n", "?")
            row += f" {mean:<7.4f} (n={n})"
        lines.append(row)

    # 指标列说明
    lines.append("")
    lines.append("  每格格式: mean (n=有效查询数)。CI 范围见 JSON 报告。")
    lines.append("")
    return "\n".join(lines)


def _build_statistical_tests(results: dict[str, dict]) -> str:
    """四、统计检验报告"""
    from . import metrics as M

    lines = [
        "四、统计检验 (Statistical Tests)",
        "=" * 70,
    ]

    if "F" in results and "D" in results:
        f_recall = [
            pq.get("recall_at_10", 0.0)
            for pq in results["F"].get("per_query", [])
            if pq.get("error") is None and pq.get("recall_at_10") is not None
        ]
        d_recall = [
            pq.get("recall_at_10", 0.0)
            for pq in results["D"].get("per_query", [])
            if pq.get("error") is None and pq.get("recall_at_10") is not None
        ]
        # 对齐: 只取两者都有的查询
        common_ids = [
            pq["query_id"] for pq in results["F"]["per_query"]
            if pq.get("query_id") and pq.get("error") is None and pq.get("recall_at_10") is not None
        ]
        d_by_id = {
            pq["query_id"]: pq.get("recall_at_10", 0.0)
            for pq in results["D"]["per_query"]
            if pq.get("query_id") in common_ids and pq.get("error") is None
        }
        f_aligned = [
            pq.get("recall_at_10", 0.0)
            for pq in results["F"]["per_query"]
            if pq.get("query_id") in d_by_id
        ]
        d_aligned = [d_by_id[qid] for qid in
                     [pq["query_id"] for pq in results["F"]["per_query"]
                      if pq.get("query_id") in d_by_id]]

        if len(f_aligned) == len(d_aligned) and len(f_aligned) >= 2:
            test = M.paired_bootstrap_test(f_aligned, d_aligned)
            sig = _sig_stars(test["p_value"])
            lines.append(
                f"\n  Fusion vs Dense (配对检验, n={len(f_aligned)}):\n"
                f"    ΔRecall@10 = {test['mean_diff']:+.4f}\n"
                f"    p = {test['p_value']:.4f}{sig}\n"
                f"    Cohen's d = {test['cohens_d']:.3f}\n"
                f"    {'✅ 显著差异' if test['significant'] else '❌ 无显著差异'}"
            )

    if "R" in results and "F" in results:
        r_recall = [
            pq.get("recall_at_10", 0.0)
            for pq in results["R"].get("per_query", [])
            if pq.get("error") is None and pq.get("recall_at_10") is not None
        ]
        f_recall = [
            pq.get("recall_at_10", 0.0)
            for pq in results["F"].get("per_query", [])
            if pq.get("error") is None and pq.get("recall_at_10") is not None
        ]

        # 对齐
        common_ids = [
            pq["query_id"] for pq in results["R"]["per_query"]
            if pq.get("query_id") and pq.get("error") is None and pq.get("recall_at_10") is not None
        ]
        f_by_id = {
            pq["query_id"]: pq.get("recall_at_10", 0.0)
            for pq in results["F"]["per_query"]
            if pq.get("query_id") in common_ids and pq.get("error") is None
        }
        r_aligned = [
            pq.get("recall_at_10", 0.0)
            for pq in results["R"]["per_query"]
            if pq.get("query_id") in f_by_id
        ]
        f_aligned = [f_by_id[qid] for qid in
                     [pq["query_id"] for pq in results["R"]["per_query"]
                      if pq.get("query_id") in f_by_id]]

        if len(r_aligned) == len(f_aligned) and len(r_aligned) >= 2:
            test = M.paired_bootstrap_test(r_aligned, f_aligned)
            sig = _sig_stars(test["p_value"])
            lines.append(
                f"\n  Reranker vs Fusion (配对检验, n={len(r_aligned)}):\n"
                f"    ΔRecall@10 = {test['mean_diff']:+.4f}\n"
                f"    p = {test['p_value']:.4f}{sig}\n"
                f"    Cohen's d = {test['cohens_d']:.3f}\n"
                f"    {'✅ 显著差异' if test['significant'] else '❌ 无显著差异'}"
            )

    if len(lines) == 2:
        lines.append("\n  (需要 F/D 或 R/F 两组配置都运行)")

    lines.append("")
    return "\n".join(lines)


def _build_reranker_diagnostics(results: dict[str, dict]) -> str:
    """五、Reranker 诊断"""
    lines = [
        "五、Reranker 诊断 (Reranker Diagnostics)",
        "=" * 70,
    ]

    if "R" not in results:
        lines.append("  (R 配置未运行, 无法诊断)")
        lines.append("")
        return "\n".join(lines)

    r_active = results["R"].get("reranker_active", False)
    if r_active:
        lines.append("  ✅ Reranker 状态: ACTIVE — Cross-Encoder 精排已生效")
        lines.append("")

        # F vs R 关键指标对比
        if "F" in results:
            lines.append("  F (Fusion) vs R (Full Pipeline) 排序指标对比:")
            lines.append(f"  {'Metric':<22} {'F':<12} {'R':<12} {'Δ':<12}")
            lines.append("  " + "-" * 58)
            for metric, mname in [
                ("mrr", "MRR"),
                ("ndcg_at_10", "NDCG@10 (Graded)"),
                ("binary_ndcg_at_10", "NDCG@10 (Binary)"),
                ("hit_at_1", "Hit@1"),
                ("average_precision", "Avg Precision"),
            ]:
                f_val = results["F"]["aggregates"].get(metric, {}).get("mean", 0)
                r_val = results["R"]["aggregates"].get(metric, {}).get("mean", 0)
                delta = r_val - f_val
                sign = "+" if delta >= 0 else ""
                lines.append(f"  {mname:<22} {f_val:<12.4f} {r_val:<12.4f} {sign}{delta:<11.4f}")
            lines.append("")
            lines.append("  说明: Recall@10 不受 Reranker 影响(集合不变)，")
            lines.append("        MRR/NDCG/Hit@1 受排序影响 — 正值表示 Reranker 改善了排序质量。")

        # 排序变化统计
        if "F" in results:
            f_pq = {pq["query_id"]: pq for pq in results["F"]["per_query"]
                    if pq.get("error") is None}
            r_pq = {pq["query_id"]: pq for pq in results["R"]["per_query"]
                    if pq.get("error") is None}
            common = set(f_pq) & set(r_pq)
            improved = 0
            degraded = 0
            unchanged = 0
            for qid in common:
                f_mrr = f_pq[qid].get("mrr", 0) or 0
                r_mrr = r_pq[qid].get("mrr", 0) or 0
                if r_mrr > f_mrr:
                    improved += 1
                elif r_mrr < f_mrr:
                    degraded += 1
                else:
                    unchanged += 1
            lines.append(f"\n  按 MRR 变化统计 ({len(common)} 共同查询):")
            lines.append(f"    改善: {improved} ({improved/max(len(common),1)*100:.1f}%)")
            lines.append(f"    恶化: {degraded} ({degraded/max(len(common),1)*100:.1f}%)")
            lines.append(f"    不变: {unchanged} ({unchanged/max(len(common),1)*100:.1f}%)")
    else:
        lines.append("  ❌ Reranker 状态: INACTIVE — Cross-Encoder 精排未生效!")
        lines.append("     可能原因: 模型加载失败/超时、配置中 reranker_enabled=False")
        lines.append("     ⚠️ 当前 R 配置结果等同于 F 配置，排序指标无意义。")

    lines.append("")
    return "\n".join(lines)


def _build_diagnostics(results: dict[str, dict]) -> str:
    """六、融合诊断指标: 融合增量 / 双路独占比 / 排名逆转"""
    from . import metrics as M

    lines = [
        "六、融合与两路诊断 (Fusion & Retrieval Diagnostics)",
        "=" * 70,
    ]

    required = ["D", "S", "F"]
    if any(c not in results for c in required):
        lines.append("  (需要 D/S/F 三组配置全部运行)")
        lines.append("")
        return "\n".join(lines)

    def _index_by_qid(config_id: str) -> dict[str, dict]:
        return {
            pq["query_id"]: pq
            for pq in results[config_id].get("per_query", [])
            if pq.get("error") is None and not pq.get("eval_skipped")
        }

    d_index = _index_by_qid("D")
    s_index = _index_by_qid("S")
    f_index = _index_by_qid("F")

    common_qids = set(d_index) & set(s_index) & set(f_index)
    if not common_qids:
        lines.append("  (无三组配置均成功的共同查询)")
        lines.append("")
        return "\n".join(lines)

    fusion_deltas: list[float] = []
    dense_uniques: list[float] = []
    sparse_uniques: list[float] = []

    for qid in common_qids:
        d_pq = d_index[qid]
        s_pq = s_index[qid]
        f_pq = f_index[qid]

        relevant = set(d_pq.get("relevant_ids", []))

        d_r10 = d_pq.get("recall_at_10") or 0.0
        s_r10 = s_pq.get("recall_at_10") or 0.0
        f_r10 = f_pq.get("recall_at_10") or 0.0
        fusion_deltas.append(M.fusion_delta(f_r10, max(d_r10, s_r10)))

        d_hits = set(d_pq.get("retrieved_ids", [])) & relevant
        s_hits = set(s_pq.get("retrieved_ids", [])) & relevant
        if relevant:
            dense_uniques.append(M.unique_from_dense(d_hits, s_hits, relevant))
            sparse_uniques.append(M.unique_from_sparse(d_hits, s_hits, relevant))

    n = len(common_qids)

    mean_fd = sum(fusion_deltas) / n if fusion_deltas else 0.0
    mean_du = sum(dense_uniques) / len(dense_uniques) if dense_uniques else 0.0
    mean_su = sum(sparse_uniques) / len(sparse_uniques) if sparse_uniques else 0.0

    lines.append(f"  共同有标注查询数: {n}")
    lines.append(f"  ΔRecall@10 (Fusion - best_single): {mean_fd:+.4f}")
    lines.append(f"    > 0 表示融合优于两路中较好者 (1+1>2)")
    lines.append(f"    = 0 表示融合未带来额外收益")
    lines.append(f"    < 0 表示融合反而降低了召回 (应注意融合权重)")
    lines.append(f"")
    lines.append(f"  Dense 路独有相关文档比例: {mean_du:.4f}")
    lines.append(f"    (仅在 Dense 路命中、Sparse 路未命中的相关文档占比)")
    lines.append(f"  Sparse 路独有相关文档比例: {mean_su:.4f}")
    lines.append(f"    (仅在 Sparse 路命中、Dense 路未命中的相关文档占比)")
    lines.append(f"  双路互补性: Dense独有 + Sparse独有 = {mean_du + mean_su:.4f}")
    lines.append(f"    (值越高说明两路检索互补性越强，融合价值越大)")

    # Hard negative 分析
    fpr_per_config: dict[str, float] = {}
    for cid in ["D", "S", "F", "R"]:
        if cid not in results:
            continue
        fpr_vals = [
            pq.get("false_positive_rate_at_10")
            for pq in results[cid].get("per_query", [])
            if pq.get("error") is None
            and pq.get("false_positive_rate_at_10") is not None
        ]
        if fpr_vals:
            fpr_per_config[cid] = sum(fpr_vals) / len(fpr_vals)

    if fpr_per_config:
        lines.append("")
        lines.append("  Hard Negative 分析 (FPR@10 = 前10结果中硬负例占比, 越低越好):")
        for cid in ["D", "S", "F", "R"]:
            if cid in fpr_per_config:
                lines.append(f"    {cid}: FPR@10 = {fpr_per_config[cid]:.4f}")

    lines.append("")
    return "\n".join(lines)


def _sig_stars(p: float) -> str:
    if p < 0.001:
        return " ***"
    elif p < 0.01:
        return " **"
    elif p < 0.05:
        return " *"
    return ""


# ==============================================================================
# 按类别 / 难度分层
# ==============================================================================
def _build_category_breakdown(results: dict[str, dict]) -> str:
    """按类别分层表 (含 bootstrap 95% CI 半宽)。"""
    from . import metrics as M

    lines = [
        "七、按类别分层 (Category Breakdown)",
        "=" * 100,
    ]

    all_categories: set[str] = set()
    for cid, result in results.items():
        for pq in result.get("per_query", []):
            cat = pq.get("category", "unknown")
            all_categories.add(cat)

    categories = sorted(all_categories)

    cols = ["D R@10", "S R@10", "F R@10", "R R@10", "R MAP", "R NDCG@10", "#Q"]
    lines.append(f"{'Category':<16}" + "".join(f" {c:<14}" for c in cols))
    lines.append("-" * 118)

    for cat in categories:
        row = f"{cat:<16}"
        for cid in ["D", "S", "F", "R"]:
            if cid not in results:
                row += f" {'N/A':<14}"
                continue
            cat_pqs = [
                pq for pq in results[cid].get("per_query", [])
                if pq.get("category") == cat
                and pq.get("error") is None
                and pq.get("recall_at_10") is not None
            ]
            if not cat_pqs:
                row += f" {'N/A':<14}"
            else:
                scores = [pq.get("recall_at_10", 0.0) for pq in cat_pqs]
                ci = M.bootstrap_confidence_interval(scores)
                half_w = (ci["ci_upper"] - ci["ci_lower"]) / 2.0
                row += f" {ci['mean']:.3f}+/-{half_w:.3f}"

        # MAP 和 NDCG (仅 R 配置)
        if "R" in results:
            r_cat_pqs = [
                pq for pq in results["R"].get("per_query", [])
                if pq.get("category") == cat
                and pq.get("error") is None
                and pq.get("average_precision") is not None
            ]
            if r_cat_pqs:
                map_scores = [pq.get("average_precision", 0.0) for pq in r_cat_pqs]
                map_ci = M.bootstrap_confidence_interval(map_scores)
                ndcg_scores = [pq.get("ndcg_at_10", 0.0) for pq in r_cat_pqs]
                ndcg_ci = M.bootstrap_confidence_interval(ndcg_scores)
                row += f" {map_ci['mean']:.3f} {ndcg_ci['mean']:.3f}"
            else:
                row += f" {'N/A':<14} {'N/A':<14}"
            row += f" {len(r_cat_pqs):<3}"
        else:
            row += f" {'N/A':<14} {'N/A':<14}"

        lines.append(row)

    lines.append("")
    lines.append("  +/- 后的值为 Bootstrap 95% CI 半宽。值大 = 该类别查询间差异大。")
    lines.append("")
    return "\n".join(lines)


def _build_difficulty_breakdown(results: dict[str, dict]) -> str:
    """按难度分层 (含 bootstrap 95% CI)。"""
    from . import metrics as M

    lines = [
        "八、按难度分层 (Difficulty Breakdown)",
        "=" * 60,
    ]
    for diff in ["easy", "medium", "hard"]:
        if "R" not in results:
            break
        pqs = [
            pq for pq in results["R"].get("per_query", [])
            if pq.get("difficulty") == diff
            and pq.get("error") is None
            and pq.get("recall_at_10") is not None
        ]
        if not pqs:
            continue
        recall_scores = [pq.get("recall_at_10", 0.0) for pq in pqs]
        map_scores = [pq.get("average_precision") for pq in pqs
                      if pq.get("average_precision") is not None]
        ndcg_scores = [pq.get("ndcg_at_10") for pq in pqs
                       if pq.get("ndcg_at_10") is not None]
        r_ci = M.bootstrap_confidence_interval(recall_scores)
        m_ci = M.bootstrap_confidence_interval(map_scores) if map_scores else {"mean": "N/A"}
        n_ci = M.bootstrap_confidence_interval(ndcg_scores) if ndcg_scores else {"mean": "N/A"}
        lines.append(
            f"  {diff:<8}: R@10={r_ci['mean']:.4f} [{r_ci['ci_lower']:.4f}, {r_ci['ci_upper']:.4f}]  "
            f"MAP={m_ci['mean'] if isinstance(m_ci['mean'], str) else format(m_ci['mean'], '.4f')}  "
            f"NDCG={n_ci['mean'] if isinstance(n_ci['mean'], str) else format(n_ci['mean'], '.4f')}  "
            f"n={len(pqs)}"
        )
    lines.append("")
    return "\n".join(lines)


# ==============================================================================
# 指标参考
# ==============================================================================
def _build_metrics_reference() -> str:
    """九、指标参考 — 公式 + 说明"""
    from .metrics import METRICS_REFERENCE

    lines = [
        "九、指标参考 (Metrics Reference)",
        "=" * 80,
        "",
        "以下为报告中使用的所有评估指标的定义和解读:",
        "",
    ]

    # 主指标
    lines.append("【主要指标 (Primary Metrics)】")
    lines.append("")
    for key, info in METRICS_REFERENCE.items():
        if info.get("category") != "primary":
            continue
        lines.append(f"  ■ {info['name_zh']} ({key})")
        lines.append(f"    公式: {info['formula']}")
        lines.append(f"    含义: {info['meaning']}")
        lines.append(f"    范围: {info['range']}")
        if info.get("note"):
            lines.append(f"    注: {info['note']}")
        lines.append("")

    lines.append("【诊断指标 (Diagnostic Metrics)】")
    lines.append("")
    for key, info in METRICS_REFERENCE.items():
        if info.get("category") != "diagnostic":
            continue
        lines.append(f"  ■ {info['name_zh']} ({key})")
        lines.append(f"    公式: {info['formula']}")
        lines.append(f"    含义: {info['meaning']}")
        lines.append(f"    范围: {info['range']}")
        if info.get("note"):
            lines.append(f"    注: {info['note']}")
        lines.append("")

    lines.append("【指标解读指南】")
    lines.append("")
    lines.append("  核心流程: D → S → F → R 四路逐级对比")
    lines.append("  - S vs D: 看 Recall@10 判断关键词 vs 语义谁更适合当前语料")
    lines.append("  - F vs max(D,S): 看 ΔRecall@10 和双路独占比，判断融合价值")
    lines.append("  - R vs F: 看 MRR/NDCG/Hit@1 的 Δ，判断 Reranker 是否改善排序")
    lines.append("  - 最终 R 配置的 Recall@10 是最重要的综合指标")
    lines.append("")
    lines.append("  Recall@10 高 + MRR 低 = 找得到但排序差 (结果相关但顺序不对)")
    lines.append("  Recall@10 低 + Coverage 高 = 多数查询能找到至少一个，但覆盖不全")
    lines.append("  Binary NDCG ≈ Graded NDCG = 标注分级可靠 (rel=1 vs rel=2 区分有意义)")
    lines.append("  Binary NDCG ≫ Graded NDCG = 标注分级可能有问题 (核心/辅助区分不可靠)")
    lines.append("")
    lines.append("=" * 80)
    return "\n".join(lines)


# ==============================================================================
# CI 质量门禁
# ==============================================================================
def _build_gate_section(results: dict[str, dict], test_config: dict) -> str:
    """CI 质量门禁。"""
    thresholds = test_config.get("thresholds", {})
    r_result = results.get("R", {})
    r_agg = r_result.get("aggregates", {})

    recall10 = r_agg.get("recall_at_10", {}).get("mean", 0.0)
    mrr_val = r_agg.get("mrr", {}).get("mean", 0.0)
    ndcg10 = r_agg.get("ndcg_at_10", {}).get("mean", 0.0)

    recall_fail = thresholds.get("recall_at_10_fail", 0.65)
    recall_warn = thresholds.get("recall_at_10_warn", 0.75)
    mrr_fail = thresholds.get("mrr_fail", 0.40)
    mrr_warn = thresholds.get("mrr_warn", 0.55)
    ndcg_fail = thresholds.get("ndcg_at_10_fail", 0.50)
    ndcg_warn = thresholds.get("ndcg_at_10_warn", 0.65)

    lines = [
        "",
        "十、CI 质量门禁 (Quality Gate)",
        "=" * 50,
        "",
        f"  {'Indicator':<22} {'Value':<10} {'Fail <':<10} {'Warn <':<10} {'Status'}",
        "  " + "-" * 60,
    ]

    for name, val, fail_t, warn_t in [
        ("recall_at_10", recall10, recall_fail, recall_warn),
        ("mrr", mrr_val, mrr_fail, mrr_warn),
        ("ndcg_at_10", ndcg10, ndcg_fail, ndcg_warn),
    ]:
        if val == 0.0:
            status = "N/A (无数据)"
        elif val < fail_t:
            status = "🔴 FAIL"
        elif val < warn_t:
            status = "🟡 WARN"
        else:
            status = "🟢 PASS"
        lines.append(f"  {name:<22} {val:<10.4f} {fail_t:<10.2f} {warn_t:<10.2f} {status}")

    # 组合判断
    has_fail = (
        (recall10 > 0 and recall10 < recall_fail)
        or (mrr_val > 0 and mrr_val < mrr_fail)
        or (ndcg10 > 0 and ndcg10 < ndcg_fail)
    )
    has_warn = not has_fail and (
        (recall10 > 0 and recall10 < recall_warn)
        or (mrr_val > 0 and mrr_val < mrr_warn)
        or (ndcg10 > 0 and ndcg10 < ndcg_warn)
    )
    has_data = recall10 > 0 or mrr_val > 0 or ndcg10 > 0

    lines.append("")
    if not has_data:
        lines.append("  GATE: N/A — 无有效评估数据 (所有查询均无正标注?)")
    elif has_fail:
        lines.append("  GATE: 🔴 FAIL — 至少一项指标低于 fail 阈值")
    elif has_warn:
        lines.append("  GATE: 🟡 WARN — 至少一项指标低于 warn 阈值")
    else:
        lines.append("  GATE: 🟢 PASS — 所有指标 >= warn 阈值")
    lines.append("")
    return "\n".join(lines)


# ==============================================================================
# 全量文本报告构建
# ==============================================================================
def _build_text_report(
    results: dict[str, dict],
    test_config: dict,
    timestamp: str,
    eval_meta: dict,
) -> str:
    """构建完整文本报告 (10 个章节)。"""
    sections = [
        _build_header(timestamp),
        _build_data_summary(eval_meta),
        _build_methodology(),
        _build_overall_table(results),
        _build_statistical_tests(results),
        _build_reranker_diagnostics(results),
        _build_diagnostics(results),
        _build_category_breakdown(results),
        _build_difficulty_breakdown(results),
        _build_metrics_reference(),
        _build_gate_section(results, test_config),
        "",
        "报告结束",
        "=" * 80,
    ]

    return "\n".join(sections)


# ==============================================================================
# JSON 报告
# ==============================================================================
def _compute_category_breakdown_data(results: dict[str, dict]) -> dict[str, dict]:
    """计算按类别分层的结构化数据 (供 JSON 报告使用)。"""
    from . import metrics as M

    all_categories: set[str] = set()
    for result in results.values():
        for pq in result.get("per_query", []):
            all_categories.add(pq.get("category", "unknown"))

    breakdown: dict[str, dict] = {}
    for cat in sorted(all_categories):
        breakdown[cat] = {}
        for cid, result in results.items():
            cat_pqs = [
                pq for pq in result.get("per_query", [])
                if pq.get("category") == cat
                and pq.get("error") is None
                and pq.get("recall_at_10") is not None
            ]
            if not cat_pqs:
                breakdown[cat][cid] = {"n": 0}
                continue

            cat_data: dict[str, dict] = {"n": len(cat_pqs)}
            for metric in ["recall_at_10", "recall_at_5", "recall_at_3",
                            "mrr", "ndcg_at_10", "binary_ndcg_at_10",
                            "hit_at_1", "precision_at_10",
                            "average_precision", "r_precision", "f1_at_10"]:
                scores = [pq.get(metric) for pq in cat_pqs if pq.get(metric) is not None]
                if scores:
                    cat_data[metric] = M.bootstrap_confidence_interval(scores)
            breakdown[cat][cid] = cat_data

    return breakdown


def _compute_difficulty_breakdown_data(results: dict[str, dict]) -> dict[str, dict]:
    """计算按难度分层的结构化数据 (供 JSON 报告使用)。"""
    from . import metrics as M

    breakdown: dict[str, dict] = {}
    for diff in ["easy", "medium", "hard"]:
        breakdown[diff] = {}
        for cid, result in results.items():
            pqs = [
                pq for pq in result.get("per_query", [])
                if pq.get("difficulty") == diff
                and pq.get("error") is None
                and pq.get("recall_at_10") is not None
            ]
            if not pqs:
                breakdown[diff][cid] = {"n": 0}
                continue

            diff_data: dict[str, dict] = {"n": len(pqs)}
            for metric in ["recall_at_10", "mrr", "ndcg_at_10", "average_precision"]:
                scores = [pq.get(metric) for pq in pqs if pq.get(metric) is not None]
                if scores:
                    diff_data[metric] = M.bootstrap_confidence_interval(scores)
            breakdown[diff][cid] = diff_data

    return breakdown


def _build_json_report(
    results: dict[str, dict],
    test_config: dict,
    timestamp: str,
    eval_meta: dict,
) -> dict:
    """构建完整 JSON 报告结构。"""
    from .metrics import METRICS_REFERENCE

    report = {
        "metadata": {
            "timestamp": timestamp,
            "config_file": str(test_config.get("_config_path", "")),
        },
        "data_summary": eval_meta,
        "aggregates": {},
        "category_breakdown": _compute_category_breakdown_data(results),
        "difficulty_breakdown": _compute_difficulty_breakdown_data(results),
        "statistical_tests": {},  # 统计检验结果由 reporter consumer 填充
        "reranker": {
            "active": eval_meta.get("reranker_active", None),
        },
        "per_query_count": {},
        "metrics_reference": METRICS_REFERENCE,
        "thresholds": test_config.get("thresholds", {}),
    }

    for cid, result in results.items():
        report["aggregates"][cid] = result.get("aggregates", {})
        report["per_query_count"][cid] = len(result.get("per_query", []))

    return report


# ==============================================================================
# 遗漏归因分析 (保留原有功能)
# ==============================================================================
def _build_miss_attribution_report(
    diagnostics: dict[str, dict[str, list]],
    qrels: dict[str, dict[str, int]],
) -> str:
    """分析被遗漏的相关文档原因。

    Args:
        diagnostics: {query_id: {phase1_dense: [...], phase1_sparse: [...],
                      phase2_fusion: [...], phase3_reranker: [...]}}
        qrels: {query_id: {doc_id: relevance}}

    Returns:
        Markdown 格式的遗漏归因报告
    """
    lines = ["MISS ATTRIBUTION ANALYSIS", "=" * 60]

    total_missed = 0
    missed_in_pool = 0
    missed_after_rerank = 0
    missed_by_truncation = 0

    for query_id, diag in diagnostics.items():
        relevant = set(qrels.get(query_id, {}).keys())
        if not relevant:
            continue

        dense_ids = set()
        sparse_ids = set()
        fusion_ids = set()
        reranker_ids = set()

        for phase_data in diag.get("phase1_dense", []):
            for chunk_id, _score in phase_data.get("results", []):
                dense_ids.add(chunk_id)

        for phase_data in diag.get("phase1_sparse", []):
            for chunk_id, _score in phase_data.get("results", []):
                sparse_ids.add(chunk_id)

        for phase_data in diag.get("phase2_fusion", []):
            for chunk_id, _score in phase_data.get("results", []):
                fusion_ids.add(chunk_id)

        for phase_data in diag.get("phase3_reranker", []):
            for item in phase_data.get("results", []):
                if len(item) >= 1:
                    reranker_ids.add(item[0])

        for doc_id in relevant:
            total_missed += 1
            in_dense = doc_id in dense_ids
            in_sparse = doc_id in sparse_ids
            in_fusion = doc_id in fusion_ids
            in_reranker = doc_id in reranker_ids

            if not in_dense and not in_sparse:
                missed_in_pool += 1
            elif in_fusion and not in_reranker:
                missed_after_rerank += 1
            elif not in_fusion and (in_dense or in_sparse):
                missed_by_truncation += 1

    if total_missed > 0:
        lines.append(f"Total missed: {total_missed}")
        lines.append(f"  Not in candidate pool: {missed_in_pool} ({missed_in_pool/total_missed*100:.1f}%)")
        lines.append(f"  In pool but removed by Reranker: {missed_after_rerank} ({missed_after_rerank/total_missed*100:.1f}%)")
        lines.append(f"  In pool but truncated: {missed_by_truncation} ({missed_by_truncation/total_missed*100:.1f}%)")
    else:
        lines.append("No missed documents found in diagnostics.")

    lines.append("")
    return "\n".join(lines)
