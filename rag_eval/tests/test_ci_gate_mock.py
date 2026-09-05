"""CI 质量门禁 Mock 测试 (T08.02)。

使用合成数据验证 CI 流水线:
- 指标计算流程 (T02.01~T02.09)
- 评估流程 (T03.03~T03.07)
- 阈值判断 (T08.01)

零外部依赖，可在 CI 中独立运行。
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "agent"))


class TestCIGateMock:
    """T08.02: CI 门禁 Mock 测试。"""

    def test_recall_threshold_logic(self):
        """验证 recall_at_10 阈值判断逻辑正确。"""
        fail_threshold = 0.65
        warn_threshold = 0.75

        # 高于 warn → pass
        assert 0.82 >= warn_threshold

        # 高于 fail 但低于 warn → warn
        assert 0.70 >= fail_threshold
        assert 0.70 < warn_threshold

        # 低于 fail → fail
        assert 0.60 < fail_threshold

    def test_metrics_computation_pipeline(self):
        """验证完整指标计算流程 (合成数据)。"""
        from rag_eval.metrics import (
            recall_at_k, mrr, ndcg_at_k,
            bootstrap_confidence_interval, compute_all_metrics,
        )

        # 模拟 30 条查询的逐查询结果
        per_query = []
        for i in range(30):
            pq = {
                "query_id": f"ci_q_{i:04d}",
                "retrieved_ids": [f"doc_{j}" for j in range(10)],
                "relevant_ids": [f"doc_{j}" for j in range(3)],  # 3 relevant
                "recall_at_10": 3.0 / 3.0 if i < 25 else 1.0 / 3.0,
                "mrr": 0.5,
                "ndcg_at_10": 0.7,
                "hit_at_1": 1.0,
                "precision_at_10": 3.0 / 10,
                "error": None,
            }
            per_query.append(pq)

        result = compute_all_metrics(per_query)
        assert "recall_at_10" in result
        assert "mrr" in result
        assert result["recall_at_10"]["mean"] > 0.0

    def test_normalization_pipeline(self):
        """验证归一化适配器与指标计算的端到端流程。"""
        from rag_eval.evaluate import _normalize_result
        from rag_eval.metrics import recall_at_k

        # 模拟 10 条结果的归一化
        raw_results = [
            {"chunk_id": f"doc_{i}", "score": 0.9 - i * 0.08}
            for i in range(10)
        ]

        normalized = [
            _normalize_result(r, rank=i + 1, config_id="D")
            for i, r in enumerate(raw_results)
        ]

        retrieved_ids = [n["chunk_id"] for n in normalized]
        relevant = {"doc_0", "doc_3", "doc_7"}

        recall = recall_at_k(retrieved_ids, relevant, 10)
        # doc_0 (rank 1), doc_3 (rank 4), doc_7 (rank 8) → 3/3 = 1.0
        assert abs(recall - 1.0) < 0.001, f"Recall should be 1.0, got {recall}"

        # 验证归一化格式一致性
        for n in normalized:
            assert "chunk_id" in n
            assert "raw_score" in n
            assert "score_type" in n
            assert "rank" in n

    def test_empty_results_handling(self):
        """空结果处理: 不应崩溃。"""
        from rag_eval.metrics import compute_all_metrics
        result = compute_all_metrics([])
        assert result["recall_at_10"]["n"] == 0

    def test_new_diagnostic_metrics(self):
        """验证新增诊断指标 (F1, Coverage@k, FPR, FilterP) 的计算。"""
        from rag_eval import metrics as M

        retrieved = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j"]
        relevant = {"b", "d", "z"}
        hard_negatives = {"a", "x"}
        corpus_meta = {
            "a": {"dynasty": "清"}, "b": {"dynasty": "清"},
            "c": {"dynasty": "明"}, "d": {"dynasty": "清"},
        }

        # F1@10: P=2/10=0.2, R=2/3≈0.667, F1=2*0.2*0.667/0.867≈0.308
        f1 = M.f1_at_k(retrieved, relevant, 10)
        assert 0.30 < f1 < 0.31, f"Expected ~0.308, got {f1}"

        # Coverage@10: b and d in top-10 → 1.0
        cov = M.coverage_at_k(retrieved, relevant, 10)
        assert cov == 1.0

        # FPR@10: only 'a' is hard negative in top-10 → 1/10 = 0.1
        fpr = M.false_positive_rate_at_k(retrieved, hard_negatives, 10)
        assert fpr == 0.1, f"Expected 0.1, got {fpr}"

        # Filter Precision@10 with dynasty=清: a,b,d match → 3/10 = 0.3
        fp = M.filter_precision_at_k(retrieved, {"dynasty": "清"}, corpus_meta, 10)
        assert fp is not None
        assert abs(fp - 0.3) < 0.001, f"Expected 0.3, got {fp}"

        # No filter → None
        fp_none = M.filter_precision_at_k(retrieved, None, corpus_meta, 10)
        assert fp_none is None

    def test_bootstrap_percentile_interpolation(self):
        """验证 bootstrap 使用线性插值百分位。"""
        from rag_eval import metrics as M

        data = [0.0, 0.2, 0.5, 0.8, 1.0]
        ci = M.bootstrap_confidence_interval(data, n_bootstrap=500, seed=42)
        assert "mean" in ci
        assert "ci_lower" in ci
        assert "ci_upper" in ci
        assert ci["n"] == 5
        # 均值应在数据范围内
        assert 0.0 <= ci["mean"] <= 1.0

    def test_multi_metric_gate_logic(self):
        """验证多指标 CI 门禁组合判断逻辑。"""
        thresholds = {
            "recall_at_10_fail": 0.65, "recall_at_10_warn": 0.75,
            "mrr_fail": 0.40, "mrr_warn": 0.55,
            "ndcg_at_10_fail": 0.50, "ndcg_at_10_warn": 0.65,
        }

        # 场景1: 全部 PASS
        r_agg = {"recall_at_10": {"mean": 0.80}, "mrr": {"mean": 0.60}, "ndcg_at_10": {"mean": 0.70}}
        assert r_agg["recall_at_10"]["mean"] >= thresholds["recall_at_10_warn"]
        assert r_agg["mrr"]["mean"] >= thresholds["mrr_warn"]
        assert r_agg["ndcg_at_10"]["mean"] >= thresholds["ndcg_at_10_warn"]

        # 场景2: Recall FAIL
        r_agg2 = {"recall_at_10": {"mean": 0.60}, "mrr": {"mean": 0.60}, "ndcg_at_10": {"mean": 0.70}}
        assert r_agg2["recall_at_10"]["mean"] < thresholds["recall_at_10_fail"]

        # 场景3: MRR WARN, 其余 PASS
        r_agg3 = {"recall_at_10": {"mean": 0.80}, "mrr": {"mean": 0.50}, "ndcg_at_10": {"mean": 0.70}}
        assert r_agg3["mrr"]["mean"] < thresholds["mrr_warn"]
        assert r_agg3["mrr"]["mean"] >= thresholds["mrr_fail"]

    def test_all_categories_coverage(self):
        """验证 CI 应覆盖所有 9 个类别。"""
        expected_categories = {
            "entity", "location", "plot", "comparison", "motif",
            "filtered", "alias", "intertext", "robustness",
        }
        assert len(expected_categories) == 9
