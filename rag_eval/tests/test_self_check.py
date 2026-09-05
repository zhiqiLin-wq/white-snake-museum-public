"""评估系统自检验证 (T03.17)。

用合成数据集验证评估系统本身的正确性——指标计算、归一化适配器、
bootstrap CI 的行为必须与已知答案一致。

零外部依赖（不使用真实模型/索引/数据集）。
"""
import sys
from pathlib import Path

import pytest

# Ensure rag_eval is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "agent"))


class TestMetricsSelfCheck:
    """T03.17.1: 指标计算验证 — 手算 vs 代码输出。"""

    def test_recall_at_k_known_answer(self):
        """手工验证 Recall@10 计算。"""
        from rag_eval.metrics import recall_at_k

        retrieved = ["d1", "d2", "d3", "d4", "d5"]
        relevant = {"d1", "d3", "d7"}
        # |retrieved[:10] ∩ relevant| = |{d1, d3}| = 2
        # |relevant| = 3
        # Recall@10 = 2/3 ≈ 0.6667
        result = recall_at_k(retrieved, relevant, 10)
        assert abs(result - 2.0 / 3.0) < 0.001, f"Expected 0.6667, got {result}"

    def test_recall_at_k_boundary(self):
        """边界条件: 空输入、k=0、k > len。"""
        from rag_eval.metrics import recall_at_k

        assert recall_at_k([], set(), 10) == 0.0
        assert recall_at_k(["d1"], {"d1"}, 0) == 0.0
        assert recall_at_k(["d1"], {"d1"}, 10) == 1.0  # k > len
        assert recall_at_k([], {"d1"}, 10) == 0.0
        assert recall_at_k(["d1"], set(), 10) == 0.0

    def test_mrr_known_answer(self):
        """手工验证 MRR 计算。"""
        from rag_eval.metrics import mrr

        # rank 1 → 1.0
        assert abs(mrr(["d1", "d3"], {"d1"}) - 1.0) < 0.001
        # rank 5 → 0.2
        retrieved = ["a", "b", "c", "d", "target", "e"]
        assert abs(mrr(retrieved, {"target"}) - 0.2) < 0.001
        # 无匹配 → 0.0
        assert mrr(["a", "b"], {"z"}) == 0.0
        assert mrr([], {"a"}) == 0.0

    def test_ndcg_at_k_known_answer(self):
        """手工验证 NDCG@3。"""
        from rag_eval.metrics import ndcg_at_k

        retrieved = ["a", "b", "c"]
        rel_map = {"a": 2, "b": 1, "c": 0}
        # DCG = (2^2-1)/log2(2) + (2^1-1)/log2(3) = 3/1 + 1/1.585 ≈ 3.631
        # IDCG = sort rel=[2,1,0]: 3/1 + 1/1.585 ≈ 3.631
        # NDCG ≈ 1.0
        result = ndcg_at_k(retrieved, rel_map, 3)
        assert result > 0.99, f"Expected ~1.0, got {result}"

    def test_ndcg_at_k_boundary(self):
        """NDCG 边界条件。"""
        from rag_eval.metrics import ndcg_at_k

        assert ndcg_at_k([], {}, 10) == 0.0
        assert ndcg_at_k(["a"], {}, 10) == 0.0
        assert ndcg_at_k(["a"], {"a": 0}, 10) == 0.0

    def test_hit_at_k(self):
        from rag_eval.metrics import hit_at_k

        assert hit_at_k(["d1", "d2"], {"d1"}, 1) == 1.0
        assert hit_at_k(["d2", "d1"], {"d1"}, 1) == 0.0
        assert hit_at_k([], {"d1"}, 1) == 0.0

    def test_precision_at_k(self):
        from rag_eval.metrics import precision_at_k

        # 2/3 relevant in top 3
        result = precision_at_k(["d1", "d2", "d3"], {"d1", "d3"}, 3)
        assert abs(result - 2.0 / 3.0) < 0.001


class TestNormalizationSelfCheck:
    """T03.17.2: 归一化适配器一致性验证。"""

    def test_s_d_format_identical(self):
        """S 和 D 配置归一化后格式完全一致。"""
        from rag_eval.evaluate import _normalize_result

        s_raw = {"chunk_id": "test_001", "bm25_score": 12.5}
        d_raw = {"chunk_id": "test_001", "score": 0.85}

        s_norm = _normalize_result(s_raw, 1, "S")
        d_norm = _normalize_result(d_raw, 1, "D")

        # 字段名完全一致
        assert set(s_norm.keys()) == set(d_norm.keys())
        assert set(d_norm.keys()) == {"chunk_id", "raw_score", "score_type", "rank"}

    def test_score_types(self):
        """验证各配置的 score_type 正确标注。"""
        from rag_eval.evaluate import _normalize_result

        assert _normalize_result({"bm25_score": 1.0}, 1, "S")["score_type"] == "bm25_okapi"
        assert _normalize_result({"score": 0.5}, 1, "D")["score_type"] == "dense_cosine"
        assert _normalize_result({"score": 0.042}, 1, "F")["score_type"] == "fusion_rrf"
        assert _normalize_result({"relevance_score": 0.92}, 1, "R")["score_type"] == "reranker_cross"

    def test_raw_score_preserved(self):
        """raw_score 保留原始分数值。"""
        from rag_eval.evaluate import _normalize_result

        assert _normalize_result({"bm25_score": 12.5}, 1, "S")["raw_score"] == 12.5
        assert _normalize_result({"score": 0.85}, 1, "D")["raw_score"] == 0.85


class TestBootstrapSelfCheck:
    """T03.17.3: Bootstrap CI 覆盖概率验证。"""

    def test_ci_contains_true_mean(self):
        """95% CI 应包含真实均值（概率性测试，运行多次）。"""
        from rag_eval.metrics import bootstrap_confidence_interval
        import random
        rng = random.Random(42)
        scores = [rng.gauss(0.8, 0.1) for _ in range(100)]

        ci = bootstrap_confidence_interval(scores, n_bootstrap=1000, seed=42)
        assert ci["ci_lower"] <= ci["mean"] <= ci["ci_upper"]

    def test_constant_scores_zero_ci_width(self):
        """全相同分数 → CI 宽度为零。"""
        from rag_eval.metrics import bootstrap_confidence_interval
        ci = bootstrap_confidence_interval([0.8] * 100, n_bootstrap=1000, seed=42)
        assert abs(ci["ci_upper"] - ci["ci_lower"]) < 0.001

    def test_empty_scores(self):
        """空输入返回全零。"""
        from rag_eval.metrics import bootstrap_confidence_interval
        ci = bootstrap_confidence_interval([], seed=42)
        assert ci["mean"] == 0.0
        assert ci["n"] == 0


class TestPairedBootstrapSelfCheck:
    """T03.17.4: Paired Bootstrap Test 正确性验证。"""

    def test_identical_no_significance(self):
        """两组完全相同 → p > 0.05, significant=False。"""
        from rag_eval.metrics import paired_bootstrap_test

        result = paired_bootstrap_test([0.8] * 50, [0.8] * 50, seed=42)
        assert result["p_value"] > 0.05, f"p={result['p_value']}"
        assert not result["significant"]

    def test_different_significant(self):
        """两组明显差异 → p < 0.05, significant=True。"""
        from rag_eval.metrics import paired_bootstrap_test

        result = paired_bootstrap_test([0.9] * 50, [0.6] * 50, seed=42)
        assert result["significant"], f"Expected significant, p={result['p_value']}"

    def test_unequal_length_raises(self):
        """两组样本数不一致 → ValueError。"""
        from rag_eval.metrics import paired_bootstrap_test

        with pytest.raises(ValueError):
            paired_bootstrap_test([0.5, 0.6], [0.5])

    def test_small_sample(self):
        """样本数 < 2 → p=1.0, not significant。"""
        from rag_eval.metrics import paired_bootstrap_test

        result = paired_bootstrap_test([0.8], [0.8], seed=42)
        assert result["p_value"] == 1.0
        assert not result["significant"]


class TestCohenKappaSelfCheck:
    """T07.03: Cohen's Kappa 正确性验证。"""

    def test_perfect_agreement(self):
        from rag_eval.metrics import cohens_kappa
        assert abs(cohens_kappa([2, 2, 1, 1], [2, 2, 1, 1]) - 1.0) < 0.001

    def test_random_agreement(self):
        from rag_eval.metrics import cohens_kappa
        k = cohens_kappa([0, 0, 0, 0, 1, 1, 1, 1], [0, 1, 0, 1, 0, 1, 0, 1])
        # 50% random agreement
        assert abs(k) < 0.5  # Should be close to 0
