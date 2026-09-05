"""RAG 检索评估 — 指标计算引擎。

Phase 2: 一级指标 + 诊断指标 + 统计检验 + 聚合入口。
"""
import math
import random
from typing import Optional


# ==============================================================================
# T02.01: Recall@k
# ==============================================================================
def recall_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
    """计算 Recall@k。

    Recall@k = |retrieved[:k] ∩ relevant| / |relevant|

    Args:
        retrieved_ids: 检索结果 chunk_id 列表 (按排名升序)
        relevant_ids: 相关文档 chunk_id 集合
        k: 截断值

    Returns:
        Recall@k 分数 [0, 1]

    Boundary conditions:
        - relevant_ids 为空 → 0.0
        - k <= 0 → 0.0
        - retrieved_ids 为空 → 0.0
        - k > len(retrieved_ids) → 使用全部 retrieved_ids
    """
    if not relevant_ids:
        return 0.0
    if k <= 0:
        return 0.0
    if not retrieved_ids:
        return 0.0

    top_k = retrieved_ids[:k]
    hits = sum(1 for doc_id in top_k if doc_id in relevant_ids)
    return hits / len(relevant_ids)


# ==============================================================================
# 天花板归一化召回率 (Ceiling-Normalized Recall@k)
# ==============================================================================
def normalized_recall_at_k(
    retrieved_ids: list[str], relevant_ids: set[str], k: int = 10,
) -> float:
    """天花板归一化召回率 = recall@k / min(1, k/|relevant|)。

    Recall@k 受 top-k 截断限制，理论上限为 min(1, k/|relevant|)：
    - |relevant| <= k 时上限 = 1.0（可全部召回）
    - |relevant| >  k 时上限 = k/|relevant|（最多只能塞进 k 个）

    归一化后等价于 |retrieved[:k] ∩ relevant| / min(|relevant|, k)，把「检索质量」
    与「相关文档数量」解耦——多版本/多相关场景下可跨查询横向对比，不因某条
    query 相关 chunk 特别多而被结构性压低。

    Returns:
        [0, 1]。1.0 = 填满全部可召回槽位（或找全所有相关 chunk）。
    """
    if not relevant_ids or k <= 0 or not retrieved_ids:
        return 0.0

    hits = sum(1 for doc_id in retrieved_ids[:k] if doc_id in relevant_ids)
    denom = min(len(relevant_ids), k)
    if denom <= 0:
        return 0.0
    return hits / denom


# ==============================================================================
# T02.02: MRR (Mean Reciprocal Rank)
# ==============================================================================
def mrr(retrieved_ids: list[str], relevant_ids: set[str]) -> float:
    """计算 Mean Reciprocal Rank。

    MRR = 1 / rank_of_first_relevant

    Args:
        retrieved_ids: 检索结果 chunk_id 列表 (按排名升序)
        relevant_ids: 相关文档 chunk_id 集合

    Returns:
        MRR 分数 [0, 1]; 无相关文档 → 0.0
    """
    if not relevant_ids or not retrieved_ids:
        return 0.0

    for rank, doc_id in enumerate(retrieved_ids, start=1):
        if doc_id in relevant_ids:
            return 1.0 / rank

    return 0.0


# ==============================================================================
# T02.03: NDCG@k
# ==============================================================================
def ndcg_at_k(
    retrieved_ids: list[str], relevance_map: dict[str, int], k: int
) -> float:
    """计算 NDCG@k (Normalized Discounted Cumulative Gain)。

    DCG@k = sum_{i=1}^{k} (2^{rel_i} - 1) / log2(i + 1)
    NDCG@k = DCG@k / IDCG@k

    Args:
        retrieved_ids: 检索结果 chunk_id 列表 (按排名升序)
        relevance_map: {chunk_id: relevance_score} (relevance ∈ {1, 2})
        k: 截断值

    Returns:
        NDCG@k 分数 [0, 1]

    Boundary conditions:
        - k <= 0 → 0.0
        - retrieved_ids 为空 → 0.0
        - IDCG == 0 → 0.0
    """
    if k <= 0:
        return 0.0
    if not retrieved_ids:
        return 0.0

    def _dcg(ids: list[str]) -> float:
        dcg = 0.0
        for i, doc_id in enumerate(ids[:k], start=1):
            rel = relevance_map.get(doc_id, 0)
            if rel > 0:
                dcg += (2 ** rel - 1) / math.log2(i + 1)
        return dcg

    dcg = _dcg(retrieved_ids)

    # IDCG: 理想排序 (按 relevance 降序)
    ideal = sorted(relevance_map.values(), reverse=True)
    ideal_dcg = 0.0
    for i, rel in enumerate(ideal[:k], start=1):
        ideal_dcg += (2 ** rel - 1) / math.log2(i + 1)

    if ideal_dcg == 0.0:
        return 0.0

    return dcg / ideal_dcg


def ndcg_at_k_binary(
    retrieved_ids: list[str], relevant_ids: set[str], k: int = 10,
) -> float:
    """计算 Binary NDCG@k（所有相关文档权重相等）。

    与 ndcg_at_k（graded）不同，此函数将所有相关文档视为等权重 (rel=1)。
    适用于自动标注数据中 relevance 分级（1 vs 2）可能不可靠的场景，
    作为 graded NDCG 的对照指标。两者差值越大，说明分级标注的噪声越大。

    Binary gain: rel ∈ {0, 1}，即 gain = 1 if doc in relevant_ids else 0.
    IDCG = sum_{i=1}^{min(k, |relevant|)} 1 / log2(i + 1)

    Args:
        retrieved_ids: 检索结果 chunk_id 列表 (按排名升序)
        relevant_ids: 相关文档 chunk_id 集合
        k: 截断值

    Returns:
        Binary NDCG@k 分数 [0, 1]
    """
    # 复用 ndcg_at_k 的 DCG 逻辑: 构建 binary relevance_map (relevant → 1, else 0)
    binary_map = {doc_id: 1 for doc_id in relevant_ids}
    return ndcg_at_k(retrieved_ids, binary_map, k)


# ==============================================================================
# T02.04: Hit@k 和 Precision@k
# ==============================================================================
def hit_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int = 1) -> float:
    """计算 Hit@k (前 k 结果中是否至少有一个相关文档)。

    Args:
        retrieved_ids: 检索结果 chunk_id 列表
        relevant_ids: 相关文档 chunk_id 集合
        k: 截断值

    Returns:
        1.0 如果有相关文档，否则 0.0
    """
    if not relevant_ids or not retrieved_ids or k <= 0:
        return 0.0
    return 1.0 if any(doc_id in relevant_ids for doc_id in retrieved_ids[:k]) else 0.0


def precision_at_k(
    retrieved_ids: list[str], relevant_ids: set[str], k: int = 10
) -> float:
    """计算 Precision@k。

    Precision@k = |retrieved[:k] ∩ relevant| / k

    使用标准 TREC 定义: 分母始终为 k，而非 min(k, len(retrieved))。
    当返回结果不足 k 条时，缺失位置计为不相关，这会正确惩罚召回不足的系统。

    Args:
        retrieved_ids: 检索结果 chunk_id 列表
        relevant_ids: 相关文档 chunk_id 集合
        k: 截断值

    Returns:
        Precision@k 分数 [0, 1]
    """
    if k <= 0:
        return 0.0
    if not retrieved_ids:
        return 0.0

    top_k = retrieved_ids[:k]
    hits = sum(1 for doc_id in top_k if doc_id in relevant_ids)
    return hits / k


def average_precision(
    retrieved_ids: list[str], relevant_ids: set[str],
) -> float:
    """计算 Average Precision (AP)。

    AP = sum_{k=1}^{n} P@k * rel(k) / |relevant|

    其中 P@k 是前 k 个结果的精度，rel(k) 是第 k 个结果是否相关 (0/1)。
    AP 综合评估所有相关文档的排序质量，是 MAP (Mean Average Precision)
    的逐查询基础。

    Args:
        retrieved_ids: 检索结果 chunk_id 列表 (按排名升序)
        relevant_ids: 相关文档 chunk_id 集合

    Returns:
        AP 分数 [0, 1]; 无相关文档 → 0.0

    Reference:
        TREC Eval / MS MARCO 的标准定义
    """
    if not relevant_ids or not retrieved_ids:
        return 0.0

    hits = 0
    sum_prec = 0.0
    for k, doc_id in enumerate(retrieved_ids, start=1):
        if doc_id in relevant_ids:
            hits += 1
            sum_prec += hits / k

    return sum_prec / len(relevant_ids)


def r_precision(
    retrieved_ids: list[str], relevant_ids: set[str],
) -> float:
    """计算 R-Precision。

    R-Precision = Precision@R, 其中 R = |relevant_ids|。
    不需要手动选择 k，自动适配每个查询的相关文档数量。

    Args:
        retrieved_ids: 检索结果 chunk_id 列表 (按排名升序)
        relevant_ids: 相关文档 chunk_id 集合

    Returns:
        R-Precision 分数 [0, 1]; 无相关文档 → 0.0
    """
    if not relevant_ids or not retrieved_ids:
        return 0.0

    r = len(relevant_ids)
    top_r = retrieved_ids[:r]
    hits = sum(1 for doc_id in top_r if doc_id in relevant_ids)
    return hits / r


# ==============================================================================
# 百分位计算辅助函数
# ==============================================================================
def _percentile(sorted_data: list[float], p: float) -> float:
    """线性插值百分位数计算。

    相比直接 int() 截断，在样本量较小时（如 CI 模式的 30 条查询）
    给出更精确的百分位估计。

    Args:
        sorted_data: 已排序的数值列表
        p: 百分位 (0.0 ~ 1.0)，如 0.025 表示第 2.5 百分位

    Returns:
        插值后的百分位值
    """
    n = len(sorted_data)
    if n == 0:
        return 0.0
    if n == 1:
        return sorted_data[0]

    rank = p * (n - 1)
    lo = int(rank)
    hi = min(lo + 1, n - 1)
    frac = rank - lo
    return sorted_data[lo] * (1.0 - frac) + sorted_data[hi] * frac


# ==============================================================================
# T02.05: Bootstrap 置信区间
# ==============================================================================
def bootstrap_confidence_interval(
    per_query_scores: list[float],
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
    seed: int = 42,
) -> dict:
    """Bootstrap 重采样估计均值的置信区间。

    Args:
        per_query_scores: 逐查询的指标得分列表
        n_bootstrap: bootstrap 迭代次数
        confidence: 置信水平 (如 0.95)
        seed: 随机种子 (确保可复现)

    Returns:
        {"mean": float, "ci_lower": float, "ci_upper": float,
         "std": float, "n": int}

    Boundary: per_query_scores 为空 → 返回全 0。
    """
    n = len(per_query_scores)
    if n == 0:
        return {"mean": 0.0, "ci_lower": 0.0, "ci_upper": 0.0, "std": 0.0, "n": 0}

    mean = sum(per_query_scores) / n

    # 计算样本标准差 (使用 n-1 作为 Bessels correction)
    if n > 1:
        variance = sum((x - mean) ** 2 for x in per_query_scores) / (n - 1)
        std = math.sqrt(variance)
    else:
        std = 0.0

    rng = random.Random(seed)
    boot_means: list[float] = []
    for _ in range(n_bootstrap):
        sample = [rng.choice(per_query_scores) for _ in range(n)]
        boot_means.append(sum(sample) / n)

    boot_means.sort()
    alpha = (1.0 - confidence) / 2.0
    ci_lower = _percentile(boot_means, alpha)
    ci_upper = _percentile(boot_means, 1.0 - alpha)

    return {
        "mean": mean,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "std": std,
        "n": n,
    }


# ==============================================================================
# T02.07: Cohen's d 效应量
# ==============================================================================
def cohens_d(scores_a: list[float], scores_b: list[float]) -> float:
    """计算 Cohen's d 效应量。

    d = (mean_a - mean_b) / sqrt((var_a + var_b) / 2)

    Args:
        scores_a: 方法 A 的逐查询得分
        scores_b: 方法 B 的逐查询得分

    Returns:
        Cohen's d 值 (有符号)
    """
    n_a = len(scores_a)
    n_b = len(scores_b)
    if n_a < 2 or n_b < 2:
        return 0.0

    mean_a = sum(scores_a) / n_a
    mean_b = sum(scores_b) / n_b

    var_a = sum((x - mean_a) ** 2 for x in scores_a) / (n_a - 1)
    var_b = sum((x - mean_b) ** 2 for x in scores_b) / (n_b - 1)

    pooled_var = (var_a + var_b) / 2.0
    if pooled_var == 0.0:
        return 0.0

    return (mean_a - mean_b) / math.sqrt(pooled_var)


# ==============================================================================
# T02.06: Paired Bootstrap Test
# ==============================================================================
def paired_bootstrap_test(
    scores_a: list[float],
    scores_b: list[float],
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> dict:
    """配对 Bootstrap 显著性检验。

    零假设: 两种方法没有差异 (可交换标签)。
    对每条查询随机交换 A/B 标签，构建差异的 bootstrap 分布。

    Args:
        scores_a: 方法 A 的逐查询得分
        scores_b: 方法 B 的逐查询得分
        n_bootstrap: bootstrap 迭代次数
        seed: 随机种子

    Returns:
        {"p_value": float, "mean_diff": float, "ci_95_diff": [lo, hi],
         "cohens_d": float, "significant": bool}

    Boundary:
        - 两组样本数不一致 → ValueError
        - 样本数 < 2 → p=1.0, significant=False
    """
    if len(scores_a) != len(scores_b):
        raise ValueError(
            f"样本数不一致: len(A)={len(scores_a)}, len(B)={len(scores_b)}"
        )

    n = len(scores_a)
    if n < 2:
        return {
            "p_value": 1.0,
            "mean_diff": 0.0,
            "ci_95_diff": [0.0, 0.0],
            "cohens_d": 0.0,
            "significant": False,
        }

    # 观察到的差值
    observed_diffs = [a - b for a, b in zip(scores_a, scores_b)]
    observed_mean_diff = sum(observed_diffs) / n

    # Bootstrap: 随机翻转标签
    rng = random.Random(seed)
    boot_diffs: list[float] = []
    for _ in range(n_bootstrap):
        sample_diff = 0.0
        for a, b in zip(scores_a, scores_b):
            if rng.random() < 0.5:
                sample_diff += a - b
            else:
                sample_diff += b - a
        boot_diffs.append(sample_diff / n)

    boot_diffs.sort()

    # 双尾 p-value (连续性校正: 避免 p=0 的不合理结果)
    n_extreme = sum(1 for d in boot_diffs if abs(d) >= abs(observed_mean_diff))
    p_value = (n_extreme + 1) / (n_bootstrap + 1)

    # 95% CI of the difference (使用线性插值百分位)
    ci_lo = _percentile(boot_diffs, 0.025)
    ci_hi = _percentile(boot_diffs, 0.975)

    d = cohens_d(scores_a, scores_b)

    return {
        "p_value": p_value,
        "mean_diff": observed_mean_diff,
        "ci_95_diff": [ci_lo, ci_hi],
        "cohens_d": d,
        "significant": p_value < 0.05,
    }


# ==============================================================================
# T02.08: 诊断指标
# ==============================================================================
def coverage_rate(per_query_hit: list[float]) -> float:
    """覆盖率: 至少有一个相关文档被检索到的查询比例。

    Args:
        per_query_hit: 逐查询 hit@k 值列表 (0.0 或 1.0)

    Returns:
        覆盖率 [0, 1]
    """
    if not per_query_hit:
        return 0.0
    return sum(per_query_hit) / len(per_query_hit)


def coverage_at_k(
    retrieved_ids: list[str], relevant_ids: set[str], k: int = 10,
) -> float:
    """Coverage@k: 前 k 个结果中至少命中一个相关文档则为 1，否则为 0。

    与 coverage_rate 不同，此函数是逐查询的原子计算（而非聚合），
    且限定在 top-k 窗口内判断。

    Args:
        retrieved_ids: 检索结果 chunk_id 列表
        relevant_ids: 相关文档 chunk_id 集合
        k: 截断值

    Returns:
        1.0 或 0.0; 无相关文档 → 0.0
    """
    if not relevant_ids or not retrieved_ids or k <= 0:
        return 0.0
    return 1.0 if any(doc_id in relevant_ids for doc_id in retrieved_ids[:k]) else 0.0


def false_positive_rate_at_k(
    retrieved_ids: list[str], hard_negative_ids: set[str], k: int = 10,
) -> float:
    """FPR@k (False Positive Rate): 前 k 结果中 hard negative 的比例。

    利用 qrels 中的 relevance=0 标注（hard negatives）衡量检索系统
    将"已知不相关"文档排到顶部的倾向。值越低越好。

    Args:
        retrieved_ids: 检索结果 chunk_id 列表
        hard_negative_ids: hard negative 文档 chunk_id 集合
        k: 截断值

    Returns:
        FPR@k [0, 1]; 无 hard negatives → 0.0; 无检索结果 → 0.0
    """
    if k <= 0 or not retrieved_ids or not hard_negative_ids:
        return 0.0
    top_k = retrieved_ids[:k]
    fp = sum(1 for doc_id in top_k if doc_id in hard_negative_ids)
    return fp / k


def f1_at_k(
    retrieved_ids: list[str], relevant_ids: set[str], k: int = 10,
) -> float:
    """F1@k: Precision@k 与 Recall@k 的调和平均。

    F1 = 2 * P@k * R@k / (P@k + R@k)

    综合衡量检索结果的准确率和召回率平衡。

    Args:
        retrieved_ids: 检索结果 chunk_id 列表
        relevant_ids: 相关文档 chunk_id 集合
        k: 截断值

    Returns:
        F1@k [0, 1]; 无相关文档 或 P+R=0 → 0.0
    """
    if k <= 0 or not retrieved_ids or not relevant_ids:
        return 0.0

    top_k = retrieved_ids[:k]
    tp = sum(1 for doc_id in top_k if doc_id in relevant_ids)
    precision = tp / k
    recall = tp / len(relevant_ids)

    if precision + recall == 0.0:
        return 0.0
    return 2.0 * precision * recall / (precision + recall)


def filter_precision_at_k(
    retrieved_ids: list[str],
    metadata_filters: dict | None,
    corpus_meta_lookup: dict[str, dict],
    k: int = 10,
) -> float | None:
    """Filter Precision@k: 前 k 结果中满足 metadata_filters 的文档比例。

    仅对设置了 metadata_filters 的查询有意义。无 filter 的查询返回 None
    （聚合时跳过），避免虚高均值。

    Args:
        retrieved_ids: 检索结果 chunk_id 列表
        metadata_filters: 查询的元数据过滤条件，如 {"dynasty": "清", "genre": "chapter_vernacular"}
        corpus_meta_lookup: {doc_id: metadata_dict} 语料元数据查找表
        k: 截断值

    Returns:
        Filter Precision [0, 1]; 无 filter → None; 无检索结果 → 0.0
    """
    if not metadata_filters:
        return None
    if k <= 0 or not retrieved_ids:
        return 0.0

    top_k = retrieved_ids[:k]
    if not top_k:
        return 0.0

    matching = 0
    for doc_id in top_k:
        doc_meta = corpus_meta_lookup.get(doc_id, {})
        if all(doc_meta.get(key) == val for key, val in metadata_filters.items()):
            matching += 1

    return matching / len(top_k)


def fusion_delta(fusion_recall10: float, best_single_recall10: float) -> float:
    """融合的增量优势: ΔRecall@10 = fusion - max(dense, sparse)。

    Args:
        fusion_recall10: 融合后 Recall@10
        best_single_recall10: 最佳单路 Recall@10

    Returns:
        ΔRecall@10 (可正可负)
    """
    return fusion_recall10 - best_single_recall10


def unique_from_dense(
    dense_hits: set[str], sparse_hits: set[str], all_relevant: set[str],
) -> float:
    """Dense 路独有的相关文档比例。

    Args:
        dense_hits: Dense 路命中的相关文档集合
        sparse_hits: Sparse 路命中的相关文档集合
        all_relevant: 全部相关文档集合

    Returns:
        Dense 独占比 [0, 1]
    """
    if not all_relevant:
        return 0.0
    unique = (dense_hits - sparse_hits) & all_relevant
    return len(unique) / len(all_relevant)


def unique_from_sparse(
    dense_hits: set[str], sparse_hits: set[str], all_relevant: set[str],
) -> float:
    """Sparse 路独有的相关文档比例。

    Args:
        dense_hits: Dense 路命中的相关文档集合
        sparse_hits: Sparse 路命中的相关文档集合
        all_relevant: 全部相关文档集合

    Returns:
        Sparse 独占比 [0, 1]
    """
    if not all_relevant:
        return 0.0
    unique = (sparse_hits - dense_hits) & all_relevant
    return len(unique) / len(all_relevant)


def rank_reversal_count(
    fusion_top_ranks: dict[str, int],
    reranker_ranks: dict[str, int],
    relevant_ids: set[str],
    threshold: int = 5,
) -> int:
    """统计排名逆转数: 融合阶段排名 <= threshold 但 Reranker 后排名 > 10 的相关文档。

    Args:
        fusion_top_ranks: {chunk_id: fusion_rank} (1-based)
        reranker_ranks: {chunk_id: reranker_rank} (1-based)
        relevant_ids: 相关文档集合
        threshold: 融合排名阈值 (默认 5)

    Returns:
        逆转计数
    """
    count = 0
    for chunk_id in relevant_ids:
        f_rank = fusion_top_ranks.get(chunk_id, 999)
        r_rank = reranker_ranks.get(chunk_id, 999)
        if f_rank <= threshold and r_rank > 10:
            count += 1
    return count


# ==============================================================================
# T07.03: Cohen's Kappa (标注一致性)
# ==============================================================================
def cohens_kappa(annotator_a: list[int], annotator_b: list[int]) -> float:
    """计算 Cohen's Kappa 标注一致性系数。

    κ = (po - pe) / (1 - pe)
    其中 po = 观察到的一致率, pe = 随机一致率。

    Args:
        annotator_a: 标注者 A 的 relevance 评分列表
        annotator_b: 标注者 B 的 relevance 评分列表

    Returns:
        Kappa 值 [-1, 1]; 完全一致 → 1.0; 随机 → ~0.0
    """
    if len(annotator_a) != len(annotator_b):
        raise ValueError("两位标注者评分数量不一致")
    n = len(annotator_a)
    if n == 0:
        return 0.0

    # 计算观察一致率
    agree = sum(1 for a, b in zip(annotator_a, annotator_b) if a == b)
    po = agree / n

    # 计算期望一致率
    # 收集所有类别
    categories = sorted(set(annotator_a) | set(annotator_b))
    cat_to_idx = {c: i for i, c in enumerate(categories)}

    n_cats = len(categories)
    count_a = [0] * n_cats
    count_b = [0] * n_cats
    for a, b in zip(annotator_a, annotator_b):
        count_a[cat_to_idx[a]] += 1
        count_b[cat_to_idx[b]] += 1

    pe = 0.0
    for i in range(n_cats):
        pe += (count_a[i] / n) * (count_b[i] / n)

    if pe == 1.0:
        return 1.0 if po == 1.0 else 0.0

    return (po - pe) / (1.0 - pe)


# ==============================================================================
# T02.09: 聚合入口 —— compute_all_metrics
# ==============================================================================
def compute_all_metrics(
    per_query_results: list[dict],
    metric_names: list[str] | None = None,
) -> dict[str, dict]:
    """对所有一级 + 二级指标计算 bootstrap 均值和 95% CI。

    从 per_query 中读取预计算指标值（由 _run_single_config 写入），
    对所有指标统一处理：跳过 error 和 None 值（None = 该查询不适用此指标，
    如无正标注/无 hard negatives/无 metadata_filters）。

    Args:
        per_query_results: [{query_id, retrieved_ids, relevant_ids, recall_at_10,
                             mrr, ndcg_at_10, ...}, ...]
        metric_names: 指定指标列表，默认全部一级 + 二级

    Returns:
        {"recall_at_10": {"mean": 0.82, "ci_lower": 0.76, "ci_upper": 0.87,
                          "std": 0.12, "n": 200}, ...}
        其中 n 为实际参与聚合的有效查询数（排除 error/None 后）
    """
    if metric_names is None:
        metric_names = [
            # Primary
            "recall_at_10", "recall_at_5", "recall_at_3",
            "normalized_recall_at_10", "recall_at_10_core",
            "mrr", "ndcg_at_10", "binary_ndcg_at_10",
            "hit_at_1", "precision_at_10",
            "average_precision", "r_precision", "f1_at_10",
            # Diagnostic
            "coverage_at_10",
            "false_positive_rate_at_10",
            "filter_precision_at_10",
        ]

    results: dict[str, dict] = {}

    for name in metric_names:
        per_query_scores: list[float] = []

        for pq in per_query_results:
            # 跳过出错/超时的查询
            if pq.get("error") is not None:
                continue
            val = pq.get(name)
            # 跳过 None 值: 该查询不适用此指标
            # (无正标注 / 无 hard negatives / 无 metadata_filters)
            if val is None:
                continue
            per_query_scores.append(float(val))

        if not per_query_scores:
            results[name] = {"mean": 0.0, "ci_lower": 0.0, "ci_upper": 0.0,
                             "std": 0.0, "n": 0}
            continue

        results[name] = bootstrap_confidence_interval(per_query_scores)

    return results


# ==============================================================================
# 指标参考字典 — 供报告生成器使用
# ==============================================================================
METRICS_REFERENCE: dict[str, dict] = {
    "recall_at_10": {
        "name_zh": "召回率@10",
        "formula": "|前10命中 ∩ 相关| / |相关|",
        "meaning": "所有相关文档中，有多少比例出现在前10个检索结果里。衡量系统'找全'的能力。",
        "range": "[0, 1]，越高越好",
        "category": "primary",
        "note": "最核心的检索评估指标。不受排序影响，仅衡量集合覆盖。",
    },
    "recall_at_5": {
        "name_zh": "召回率@5",
        "formula": "|前5命中 ∩ 相关| / |相关|",
        "meaning": "前5个结果中命中的相关文档比例。更严格的召回率，关注顶部结果覆盖。",
        "range": "[0, 1]，越高越好",
        "category": "primary",
    },
    "recall_at_3": {
        "name_zh": "召回率@3",
        "formula": "|前3命中 ∩ 相关| / |相关|",
        "meaning": "前3个结果中命中的相关文档比例。最严格的召回率，反映精准定位能力。",
        "range": "[0, 1]，越高越好",
        "category": "primary",
    },
    "normalized_recall_at_10": {
        "name_zh": "天花板归一化召回率@10",
        "formula": "|前10命中| / min(|相关|, 10)",
        "meaning": "Recall@10 除以其理论上限 min(1, 10/|相关|)。解耦检索质量与相关文档数量，多版本/多相关场景下可跨查询横向对比。",
        "range": "[0, 1]，越高越好",
        "category": "primary",
        "note": "|相关|<=10 时等于 Recall@10；|相关|>10 时等于 Precision@10。1.0 = 填满全部可召回槽位。",
    },
    "recall_at_10_core": {
        "name_zh": "核心答案召回率@10",
        "formula": "|前10命中 ∩ 核心(rel=2)| / |核心(rel=2)|",
        "meaning": "只统计 relevance=2 的核心答案块，剔除 rel=1 辅助上下文的灌水。衡量「找全答案」的能力。",
        "range": "[0, 1]，越高越好",
        "category": "primary",
        "note": "仅对有 rel=2 标注的查询计算。每条 query 平均仅 2.5~3.7 个核心块。",
    },
    "mrr": {
        "name_zh": "平均倒数排名 (MRR)",
        "formula": "1 / (第一个相关文档的排名)",
        "meaning": "衡量系统多快能找到第一个相关文档。排名第1得1.0分，排名第10得0.1分。",
        "range": "[0, 1]，越高越好",
        "category": "primary",
        "note": "排序敏感指标。仅关注第一个相关文档的排名，后续相关文档不影响得分。",
    },
    "ndcg_at_10": {
        "name_zh": "归一化折损累积增益@10 (Graded NDCG)",
        "formula": "DCG/IDCG，其中 DCG = Σ(2^rel - 1) / log₂(rank+1)",
        "meaning": "带权重的排序质量指标。核心答案(relevance=2)权重是辅助上下文(relevance=1)的3倍。越相关越靠前得分越高。",
        "range": "[0, 1]，越高越好",
        "category": "primary",
        "note": "排序敏感 + 相关性分级。依赖 qrels 的 rel=1 vs rel=2 区分准确性，受标注噪声影响。",
    },
    "binary_ndcg_at_10": {
        "name_zh": "二值归一化折损累积增益@10 (Binary NDCG)",
        "formula": "DCG/IDCG，其中 DCG = Σrel_binary / log₂(rank+1), rel∈{0,1}",
        "meaning": "不区分相关程度的排序质量指标。所有相关文档等权重(gain=1)。与 graded NDCG 对比可评估分级标注的可靠性。",
        "range": "[0, 1]，越高越好",
        "category": "primary",
        "note": "排序敏感但不对相关性分级。Binary 与 Graded NDCG 差异大 → 标注分级噪声大。",
    },
    "hit_at_1": {
        "name_zh": "首位命中率 (Hit@1)",
        "formula": "第一个结果是否相关 (0/1 二值)",
        "meaning": "排名第一的结果是否相关。衡量系统在最佳位置的精准度。",
        "range": "{0, 1}，均值越高越好",
        "category": "primary",
        "note": "极端严格：只有第一位命中才算成功。适合衡量'最佳答案'类场景。",
    },
    "precision_at_10": {
        "name_zh": "准确率@10",
        "formula": "|前10命中| / 10 (TREC 定义，分母恒为 k)",
        "meaning": "前10个结果中相关文档的占比。衡量'找得准'的能力。",
        "range": "[0, 1]，越高越好",
        "category": "primary",
        "note": "分母恒为10（TREC标准），返回不足10条时缺失位计为不相关。",
    },
    "average_precision": {
        "name_zh": "平均精度 (Average Precision / AP)",
        "formula": "Σ(P@k × rel(k)) / |相关|",
        "meaning": "在每个相关文档命中位置计算精度并取平均。综合评估排序和覆盖的质量。",
        "range": "[0, 1]，越高越好",
        "category": "primary",
        "note": "MAP 的逐查询基础，对排序和召回率均敏感。",
    },
    "r_precision": {
        "name_zh": "R-精度 (R-Precision)",
        "formula": "Precision@R，其中 R = |相关文档数|",
        "meaning": "截断值自动适配每个查询的相关文档数量。无需手动选择 k，适合相关文档数差异大的场景。",
        "range": "[0, 1]，越高越好",
        "category": "primary",
    },
    "f1_at_10": {
        "name_zh": "F1 分数@10",
        "formula": "2 × P@10 × R@10 / (P@10 + R@10)",
        "meaning": "准确率与召回率的调和平均，综合衡量'找得准'和'找得全'的平衡。",
        "range": "[0, 1]，越高越好",
        "category": "primary",
    },
    "coverage_at_10": {
        "name_zh": "覆盖率@10",
        "formula": "前10结果是否至少命中1个相关文档 (0/1)",
        "meaning": "有多大比例的查询能在前10个结果中找到至少一个相关文档。衡量系统的'兜底'能力。",
        "range": "[0, 1]，越高越好",
        "category": "diagnostic",
        "note": "仅对有正标注的查询计算。值高 = 大部分查询至少有一个相关结果；值低 = 存在信息盲区。",
    },
    "false_positive_rate_at_10": {
        "name_zh": "假阳性率@10 (FPR@10)",
        "formula": "|前10中的hard_negative| / 10",
        "meaning": "前10结果中已知不相关文档(hard negatives)的占比。衡量系统过滤无关内容的能力。",
        "range": "[0, 1]，越低越好",
        "category": "diagnostic",
        "note": "仅对有 hard negatives 标注的查询计算。Hard negatives 来自 qrels 中 rel=0 的标注。",
    },
    "filter_precision_at_10": {
        "name_zh": "过滤精度@10",
        "formula": "|前10中满足 metadata_filters 的结果| / 10",
        "meaning": "前10结果中满足元数据过滤条件(如朝代、体裁)的文档比例。衡量过滤功能的准确度。",
        "range": "[0, 1]，越高越好",
        "category": "diagnostic",
        "note": "仅对有 metadata_filters 的查询计算。无 filter 的查询跳过，避免虚高均值。",
    },
}

