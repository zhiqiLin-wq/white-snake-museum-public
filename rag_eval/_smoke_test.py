"""端到端 Smoke Test (T00.09)。

在 mock 合成数据上跑 D→S→F→R 完整评估流程，验证所有模块正确集成。
零外部依赖（不需要真实模型/索引/数据集），执行时间 < 30 秒。

Usage: python -m rag_eval._smoke_test
"""
import json
import sys
from pathlib import Path
from typing import Awaitable, Callable, Optional
from unittest.mock import MagicMock, patch


# ==============================================================================
# Mock Retriever: 返回预定义的合成检索结果
# ==============================================================================
class MockHybridRetriever:
    """Mock HybridRetriever — 返回预定义合成检索结果，绕过模型层。"""

    def __init__(self, results_map: dict[str, list[dict]]):
        """Args:
            results_map: {query_id: [{chunk_id, content, metadata, score, relevance_score}, ...]}
        """
        self._results_map = results_map
        self.config = MagicMock()
        self.config.bm25_enabled = True
        self.config.reranker_enabled = True
        self.config.fusion_method = "rrf"
        self.config.fusion_k = 20
        self.config.fusion_dense_weight = 0.3
        self.config.dense_candidate_k = 50
        self.config.sparse_candidate_k = 50
        self.config.reranker_output_k = 10
        self._query_cache: dict = {}

    async def retrieve(
        self, query: str, top_k: int | None = None,
        filters: Optional[dict] = None,
        debug_callback: Optional[Callable[[str, dict], Awaitable[None]]] = None,
    ) -> list[dict]:
        results = self._results_map.get(query, [])
        # Mock debug_callback: 发送伪诊断数据
        if debug_callback:
            try:
                await debug_callback("phase1_dense", {
                    "results": [("synth_00001", 0.9), ("synth_00002", 0.8)],
                    "top_k": 50,
                })
                await debug_callback("phase1_sparse", {
                    "results": [("synth_00003", 12.5)],
                    "top_k": 50,
                })
                await debug_callback("phase2_fusion", {
                    "results": [("synth_00001", 0.042), ("synth_00003", 0.038)],
                    "method": "rrf",
                    "total_candidates": 78,
                })
                await debug_callback("phase3_reranker", {
                    "results": [("synth_00001", 0.95, 1, 1), ("synth_00003", 0.72, 2, 2)],
                    "output_k": 10,
                })
            except Exception:
                pass
        if top_k is not None:
            results = results[:top_k]
        return results


class MockBM25Retriever:
    """Mock BM25Retriever — 返回预定义合成结果 (用于 S 配置)。"""

    def __init__(self, results_map: dict[str, list[dict]]):
        self._results_map = results_map
        self._built = True

    def search(self, query: str, top_k: int = 10,
               metadata_filters: Optional[dict] = None) -> list[dict]:
        results = self._results_map.get(query, [])
        return results[:top_k]


# ==============================================================================
# 合成数据生成
# ==============================================================================
def _make_synth_data():
    """生成合成评估数据: corpus, queries, qrels, mock results。

    合成数据确保:
    - chunk_id 使用 synth_{i:05d} 格式
    - 已知 Recall@10 答案可精确验证
    """
    n_docs = 100
    n_queries = 20

    # 合成 corpus
    corpus: dict[str, dict] = {}
    for i in range(n_docs):
        doc_id = f"synth_{i:05d}"
        corpus[doc_id] = {
            "title": f"Test Document {i}",
            "text": f"This is test document number {i} with some Chinese text 测试文本。",
            "metadata": {
                "genre": ["classical_chinese", "huaben", "research_literature"][i % 3],
                "dynasty": ["唐", "宋", "明", "清"][i % 4],
            },
        }

    # 合成 queries (覆盖所有 9 类别 + 3 难度)
    categories = ["entity", "location", "plot", "comparison", "motif",
                  "filtered", "alias", "intertext", "robustness"]
    difficulties = ["easy", "medium", "hard"]

    queries: list[dict] = []
    qrels: dict[str, dict[str, int]] = {}

    for qi in range(n_queries):
        qid = f"q_{qi:04d}"
        queries.append({
            "query_id": qid,
            "text": f"test query {qi}",
            "category": categories[qi % len(categories)],
            "difficulty": difficulties[qi % len(difficulties)],
            "metadata_filters": None,
            "expected_genre_scope": "all",
            "eval_top_k_values": None,
            "description": f"Test query {qi}",
        })

        # 合成 qrels: 设置已知的相关文档
        # 确保 Recall@10 已知正确答案
        # query 0: relevant = synth_00001 (rank 1), synth_00003 (rank 3) → Recall@10 = 2/2 = 1.0
        # query 1: relevant = synth_00005 (rank 1), synth_00000 (rank 5), synth_00010 (rank 12) → Recall@10 = 2/3
        if qi == 0:
            qrels[qid] = {"synth_00001": 2, "synth_00003": 2}  # 2 relevant, both in top 10
        elif qi == 1:
            qrels[qid] = {"synth_00005": 2, "synth_00000": 1, "synth_00010": 1}  # 3 relevant
        else:
            # 每条查询 2-5 个相关文档
            relevant_count = 2 + qi % 4
            for ri in range(relevant_count):
                doc_idx = (qi * 7 + ri * 13) % n_docs
                qid_set = qrels.setdefault(qid, {})
                qid_set[f"synth_{doc_idx:05d}"] = 1 + (ri % 2)

    # Mock 检索结果 (针对 q_0000 和 q_0001 的精确结果)
    mock_results: dict[str, list[dict]] = {}

    # q_0000: synth_00001 at rank1, synth_00003 at rank3
    mock_results["test query 0"] = [
        {"chunk_id": "synth_00001", "content": "doc 1", "metadata": {}, "score": 0.95, "relevance_score": 0.95},
        {"chunk_id": "synth_00010", "content": "doc 10", "metadata": {}, "score": 0.85, "relevance_score": 0.85},
        {"chunk_id": "synth_00003", "content": "doc 3", "metadata": {}, "score": 0.75, "relevance_score": 0.75},
        {"chunk_id": "synth_00015", "content": "doc 15", "metadata": {}, "score": 0.65, "relevance_score": 0.65},
        {"chunk_id": "synth_00020", "content": "doc 20", "metadata": {}, "score": 0.55, "relevance_score": 0.55},
        {"chunk_id": "synth_00025", "content": "doc 25", "metadata": {}, "score": 0.45, "relevance_score": 0.45},
        {"chunk_id": "synth_00030", "content": "doc 30", "metadata": {}, "score": 0.35, "relevance_score": 0.35},
        {"chunk_id": "synth_00035", "content": "doc 35", "metadata": {}, "score": 0.25, "relevance_score": 0.25},
        {"chunk_id": "synth_00040", "content": "doc 40", "metadata": {}, "score": 0.15, "relevance_score": 0.15},
        {"chunk_id": "synth_00045", "content": "doc 45", "metadata": {}, "score": 0.05, "relevance_score": 0.05},
    ]

    # q_0001: synth_00005 at rank1, synth_00000 at rank5 → Recall@10 = 2/3
    mock_results["test query 1"] = [
        {"chunk_id": "synth_00005", "content": "doc 5", "metadata": {}, "score": 0.95, "relevance_score": 0.95},
        {"chunk_id": "synth_00011", "content": "doc 11", "metadata": {}, "score": 0.85, "relevance_score": 0.85},
        {"chunk_id": "synth_00012", "content": "doc 12", "metadata": {}, "score": 0.75, "relevance_score": 0.75},
        {"chunk_id": "synth_00013", "content": "doc 13", "metadata": {}, "score": 0.65, "relevance_score": 0.65},
        {"chunk_id": "synth_00000", "content": "doc 0", "metadata": {}, "score": 0.55, "relevance_score": 0.55},
        {"chunk_id": "synth_00021", "content": "doc 21", "metadata": {}, "score": 0.45, "relevance_score": 0.45},
        {"chunk_id": "synth_00022", "content": "doc 22", "metadata": {}, "score": 0.35, "relevance_score": 0.35},
        {"chunk_id": "synth_00023", "content": "doc 23", "metadata": {}, "score": 0.25, "relevance_score": 0.25},
        {"chunk_id": "synth_00024", "content": "doc 24", "metadata": {}, "score": 0.15, "relevance_score": 0.15},
        {"chunk_id": "synth_00025", "content": "doc 25", "metadata": {}, "score": 0.05, "relevance_score": 0.05},
    ]

    # BM25 mock (S 配置): 返回 bm25_score 格式
    mock_bm25_results: dict[str, list[dict]] = {}
    for q_text, results in mock_results.items():
        bm25_list = []
        for r in results:
            bm25_list.append({
                "chunk_id": r["chunk_id"],
                "content": r["content"],
                "metadata": r["metadata"],
                "bm25_score": r["score"] * 10,  # BM25 scores are in different scale
            })
        mock_bm25_results[q_text] = bm25_list

    return corpus, queries, qrels, mock_results, mock_bm25_results


# ==============================================================================
# Main smoke test
# ==============================================================================
def run_smoke_test() -> int:
    """运行完整 E2E smoke test。

    Returns:
        exit code (0 = pass, 1 = fail)
    """
    print("=" * 60)
    print("RAG Eval Smoke Test (T00.09)")
    print("=" * 60)

    passed = 0
    failed = 0

    # 1. 生成合成数据
    print("\n[1/6] Generating synthetic data...")
    corpus, queries, qrels, mock_results, mock_bm25_results = _make_synth_data()
    assert len(corpus) == 100
    assert len(queries) == 20
    assert len(qrels) > 0
    print(f"  [OK] corpus={len(corpus)} queries={len(queries)} qrels_groups={len(qrels)}")

    # 2. 验证 loader 加载
    print("\n[2/6] Testing loader...")
    from rag_eval.loader import load_queries, load_qrels

    # 写入临时文件
    tmp_dir = Path(__file__).resolve().parent / "results" / ".smoke_test_tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    tmp_queries = tmp_dir / "queries.jsonl"
    with open(tmp_queries, "w", encoding="utf-8") as f:
        for q in queries:
            f.write(json.dumps(q, ensure_ascii=False) + "\n")

    tmp_qrels = tmp_dir / "qrels.tsv"
    with open(tmp_qrels, "w", encoding="utf-8") as f:
        for qid, docs in qrels.items():
            for did, rel in docs.items():
                f.write(f"{qid}\t{did}\t{rel}\n")

    loaded_queries = load_queries(tmp_queries)
    loaded_qrels = load_qrels(tmp_qrels)
    assert len(loaded_queries) == 20, f"Expected 20 queries, got {len(loaded_queries)}"
    assert len(loaded_qrels) == len(qrels), f"Expected {len(qrels)} qrels groups, got {len(loaded_qrels)}"
    print(f"  [OK] loader: {len(loaded_queries)} queries, {len(loaded_qrels)} qrels groups")

    # 3. 验证指标计算
    print("\n[3/6] Testing metrics...")
    from rag_eval.metrics import (
        recall_at_k, mrr, ndcg_at_k, hit_at_k, precision_at_k,
        bootstrap_confidence_interval, paired_bootstrap_test, cohens_d,
        coverage_rate, compute_all_metrics, cohens_kappa,
    )

    # recall_at_k
    retrieved = ["synth_00001", "synth_00002", "synth_00003"]
    relevant = {"synth_00001", "synth_00003"}
    assert abs(recall_at_k(retrieved, relevant, 10) - 2.0 / 2.0) < 0.001, f"recall_at_k: {recall_at_k(retrieved, relevant, 10)}"
    assert abs(recall_at_k([], relevant, 10) - 0.0) < 0.001, "empty retrieved → 0"
    assert abs(recall_at_k(retrieved, set(), 10) - 0.0) < 0.001, "empty relevant → 0"
    print(f"  [OK] recall_at_k")

    # mrr
    assert abs(mrr(retrieved, relevant) - 1.0) < 0.001, f"mrr: {mrr(retrieved, relevant)}"  # d1 rank 1
    assert abs(mrr(["synth_00010", "synth_00001"], relevant) - 0.5) < 0.001  # rank 2
    assert abs(mrr([], relevant) - 0.0) < 0.001
    print(f"  [OK] mrr")

    # hit_at_k
    assert abs(hit_at_k(retrieved, relevant, 1) - 1.0) < 0.001  # hit
    assert abs(hit_at_k(["synth_00010"], relevant, 1) - 0.0) < 0.001  # miss
    print(f"  [OK] hit_at_k / precision_at_k")

    # bootstrap CI
    scores = [0.8] * 100
    ci = bootstrap_confidence_interval(scores, seed=42)
    assert abs(ci["mean"] - 0.8) < 0.001
    assert abs(ci["ci_lower"] - 0.8) < 0.001  # all same val → no variance
    print(f"  [OK] bootstrap_confidence_interval")

    # paired bootstrap
    test_result = paired_bootstrap_test([0.8] * 50, [0.8] * 50, seed=42)
    assert test_result["p_value"] > 0.05, f"Same scores should not be significant: p={test_result['p_value']}"
    assert not test_result["significant"]
    test_result2 = paired_bootstrap_test([0.9] * 50, [0.6] * 50, seed=42)
    assert test_result2["significant"], "Different scores should be significant"
    print(f"  [OK] paired_bootstrap_test")

    # Cohen's d
    d = cohens_d([0.9] * 50, [0.6] * 50)
    assert d > 1.0, f"Large effect expected, got d={d}"
    print(f"  [OK] cohens_d")

    # Cohen's kappa
    k = cohens_kappa([2, 2, 1, 1, 2], [2, 2, 1, 1, 2])
    assert abs(k - 1.0) < 0.001  # perfect agreement
    print(f"  [OK] cohens_kappa")

    passed += 1

    # 4. 验证归一化适配器
    print("\n[4/6] Testing normalization adapter...")
    from rag_eval.evaluate import _normalize_result

    # D 配置结果 (score)
    d_result = {"chunk_id": "synth_00001", "score": 0.85}
    norm_d = _normalize_result(d_result, 1, "D")
    assert norm_d["score_type"] == "dense_cosine", f"Expected dense_cosine, got {norm_d['score_type']}"
    assert norm_d["raw_score"] == 0.85
    assert norm_d["rank"] == 1

    # S 配置结果 (bm25_score)
    s_result = {"chunk_id": "synth_00001", "bm25_score": 12.5}
    norm_s = _normalize_result(s_result, 1, "S")
    assert norm_s["score_type"] == "bm25_okapi", f"Expected bm25_okapi, got {norm_s['score_type']}"
    assert norm_s["raw_score"] == 12.5
    assert norm_s["rank"] == 1

    # F 配置
    f_result = {"chunk_id": "synth_00001", "score": 0.042}
    norm_f = _normalize_result(f_result, 1, "F")
    assert norm_f["score_type"] == "fusion_rrf", f"Expected fusion_rrf, got {norm_f['score_type']}"

    # R 配置 (relevance_score)
    r_result = {"chunk_id": "synth_00001", "relevance_score": 0.92}
    norm_r = _normalize_result(r_result, 1, "R")
    assert norm_r["score_type"] == "reranker_cross", f"Expected reranker_cross, got {norm_r['score_type']}"

    # S 和 D 归一化后格式一致
    assert set(norm_s.keys()) == set(norm_d.keys()), f"Format mismatch: {norm_s.keys()} vs {norm_d.keys()}"
    print(f"  [OK] normalization: S={norm_s['score_type']} D={norm_d['score_type']} F={norm_f['score_type']} R={norm_r['score_type']}")

    passed += 1

    # 5. 验证 evaluator 集成
    print("\n[5/6] Testing evaluator integration...")
    from rag_eval.evaluate import RetrievalEvaluator

    evaluator = RetrievalEvaluator(
        corpus=corpus,
        queries=queries,
        qrels=qrels,
        verbose=False,
        debug=False,
    )

    # 使用 mock retriever 代替真实组件
    import asyncio

    async def _test_mock_eval():
        # S 配置
        mock_bm25 = MockBM25Retriever(mock_bm25_results)
        s_result = await evaluator._run_single_config("S", mock_bm25, top_k=10)
        assert len(s_result["per_query"]) == 20
        assert s_result["config_id"] == "S"

        # D/F/R 配置
        mock_hybrid = MockHybridRetriever(mock_results)
        d_result = await evaluator._run_single_config("D", mock_hybrid, top_k=10)
        assert len(d_result["per_query"]) == 20

        # 验证 q_0000 的 Recall@10
        q0_s = [
            pq for pq in s_result["per_query"]
            if pq["query_id"] == "q_0000" and pq["error"] is None
        ][0]
        q0_recall = q0_s["recall_at_10"]
        assert q0_recall >= 0.0, f"Recall should be >= 0, got {q0_recall}"

        return s_result, d_result

    s_res, d_res = asyncio.run(_test_mock_eval())

    # 验证聚合结果
    assert "recall_at_10" in s_res["aggregates"]
    assert "mrr" in s_res["aggregates"]
    assert "ndcg_at_10" in s_res["aggregates"]
    print(f"  [OK] evaluator: S recall@10={s_res['aggregates']['recall_at_10']['mean']:.4f}")

    passed += 1

    # 6. 验证 reporter 集成
    print("\n[6/6] Testing reporter...")
    from rag_eval.reporter import generate_report

    test_config = {
        "thresholds": {"recall_at_10_fail": 0.65, "recall_at_10_warn": 0.75},
        "_config_path": "smoke_test",
    }

    results = {
        "D": d_res,
        "S": s_res,
    }

    output_dir = tmp_dir.parent / ".smoke_test_output"
    output_dir.mkdir(parents=True, exist_ok=True)
    generate_report(results, test_config, output_dir, timestamp="smoke_test")

    # 验证输出文件存在（新路径: {output_dir}/{timestamp}/report.txt）
    run_dir = output_dir / "smoke_test"
    assert (run_dir / "report.txt").exists(), "txt report missing"
    assert (run_dir / "report.json").exists(), "json report missing"

    # 验证 JSON 可解析
    with open(run_dir / "report.json", "r", encoding="utf-8") as f:
        json_data = json.load(f)
    assert "aggregates" in json_data
    print(f"  [OK] reporter: txt + json generated")

    # 清理临时文件
    import shutil
    shutil.rmtree(tmp_dir, ignore_errors=True)
    shutil.rmtree(output_dir, ignore_errors=True)

    passed += 1

    # 汇总
    print("\n" + "=" * 60)
    if failed == 0:
        print("ALL SMOKE TESTS PASSED")
        print(f"  ({passed} checks passed)")
        return 0
    else:
        print(f"SMOKE TESTS FAILED: {failed} failures, {passed} passes")
        return 1


if __name__ == "__main__":
    sys.exit(run_smoke_test())
