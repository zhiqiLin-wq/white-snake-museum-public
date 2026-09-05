"""RAG 检索评估 — 主评估脚本 (Phase 3)。

T03.01~T03.19: RetrievalEvaluator + CLI + 消融实验 + 诊断。
"""
import asyncio
import copy
import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Awaitable, Callable, Optional

import yaml

# 动态添加 agent 目录
AGENT_DIR = Path(__file__).resolve().parent.parent / "agent"
if str(AGENT_DIR) not in sys.path:
    sys.path.insert(0, str(AGENT_DIR))

logger = logging.getLogger("rag_eval")


# ==============================================================================
# 结果归一化适配器 (T03.03)
# ==============================================================================
def _normalize_result(r: dict, rank: int, config_id: str) -> dict:
    """将所有检索结果归一化为统一格式，消除 S 与 D/F/R 的格式差异。

    返回四元组: (chunk_id, raw_score, score_type, rank)
    """
    if "score" in r:
        raw_score = r["score"]
        if config_id == "F":
            score_type = "fusion_rrf"
        else:
            score_type = "dense_cosine"
    elif "bm25_score" in r:
        raw_score = r["bm25_score"]
        score_type = "bm25_okapi"
    elif "relevance_score" in r:
        raw_score = r["relevance_score"]
        score_type = "reranker_cross"
    else:
        raw_score = 0.0
        score_type = "unknown"

    return {
        "chunk_id": r.get("chunk_id", ""),
        "raw_score": raw_score,
        "score_type": score_type,
        "rank": rank,
    }


# ==============================================================================
# RetrievalEvaluator (T03.01)
# ==============================================================================
class RetrievalEvaluator:
    """RAG 检索评估器。

    负责:
    1. 加载 ground truth 数据集
    2. 构建检索器 (D/S/F/R 四组配置)
    3. 逐查询执行检索 + 指标计算
    4. 统计聚合 + 诊断收集
    """

    def __init__(
        self,
        corpus: dict[str, dict],
        queries: list[dict],
        qrels: dict[str, dict[str, int]],
        verbose: bool = False,
        debug: bool = False,
        all_qrels: dict[str, dict[str, int]] | None = None,
    ):
        self.corpus = corpus
        self.queries = queries
        self.qrels = qrels
        self.verbose = verbose
        self.debug = debug

        # Hard negatives: 从全量 qrels 中提取 rel=0 的文档
        self._hard_negatives: dict[str, set[str]] = {}
        if all_qrels:
            for qid, doc_rel in all_qrels.items():
                self._hard_negatives[qid] = {
                    doc_id for doc_id, rel in doc_rel.items() if rel == 0
                }

        # 语料元数据查找表 (用于 filter_precision 计算)
        self._corpus_meta: dict[str, dict] = {
            doc_id: doc.get("metadata", {})
            for doc_id, doc in self.corpus.items()
        }

        # 诊断存储
        self._per_query_diagnostics: dict[str, dict[str, list]] = {}
        self._per_query_raw: list[dict] = []  # 逐查询原始结果

        # 共享组件 (延迟初始化)
        self._shared: dict | None = None

    # ========================================================================
    # T03.02: 共享组件初始化 (避免 D→F→R 重复加载模型 → OOM)
    # ========================================================================
    def _build_shared_components(self) -> dict:
        """一次性加载 Embedder/VectorStore/BM25/Reranker。

        Returns:
            {"embedder": ..., "vector_store": ..., "bm25": ..., "reranker": ...}
        """
        if self._shared is not None:
            return self._shared

        from server.rag.config import rag_config
        from server.rag.embedder import Embedder
        from server.rag.vector_store import VectorStore
        from server.rag.bm25_retriever import BM25Retriever
        if self.verbose:
            print("[INIT] 加载共享组件...")

        # Embedder
        embedder = Embedder(
            rag_config.embedding_model,
            local_path=rag_config.embedding_local_path or None,
        )
        embedder.start_loading()
        # 等待就绪
        waited = 0
        while not embedder.is_ready and waited < 300:
            time.sleep(1)
            waited += 1
            if self.verbose and waited % 30 == 0:
                print(f"  [WAIT] Embedder 加载中... ({waited}s)")
        if not embedder.is_ready:
            raise RuntimeError(f"Embedder 加载超时 ({waited}s). error: {embedder.load_error}")

        # VectorStore — 连接到已有 ChromaDB，若为空则从 corpus 构建
        vector_store = VectorStore(rag_config.chroma_persist_path, embedder)
        if not vector_store.is_healthy() or vector_store.collection_count() == 0:
            if self.verbose:
                print("  [INIT] VectorStore 为空，从 corpus 构建 ChromaDB...")
            # 将 corpus 字典转为类 Document 对象列表用于 build_from_chunks
            class _CorpusDoc:
                def __init__(self, doc_id, text, metadata):
                    self.content = text
                    self.metadata = metadata
            chunk_docs = []
            for doc_id, doc in self.corpus.items():
                raw_meta = doc.get("metadata", {})
                # ChromaDB 不允许 metadata 值为 None，过滤并转为字符串
                clean_meta = {}
                for k, v in raw_meta.items():
                    if v is None:
                        clean_meta[k] = ""
                    elif isinstance(v, (str, int, float, bool)):
                        clean_meta[k] = v
                    else:
                        clean_meta[k] = str(v)
                clean_meta["chunk_id"] = doc_id
                chunk_docs.append(_CorpusDoc(doc_id, doc.get("text", ""), clean_meta))
            vector_store.build_from_chunks(chunk_docs)
            if self.verbose:
                print(f"  [INIT] ChromaDB 构建完成: {vector_store.collection_count()} 文档")

        # BM25 — 从 corpus 构建索引
        bm25 = BM25Retriever()
        bm25_docs = []
        for doc_id, doc in self.corpus.items():
            bm25_docs.append({
                "chunk_id": doc_id,
                "content": doc.get("text", ""),
                "metadata": doc.get("metadata", {}),
            })
        bm25.index(bm25_docs)
        if self.verbose:
            print(f"  [BM25] index built: {bm25._total_docs} documents")

        # Reranker — 已从评估管线移除 (古籍场景无增量且导致 Windows segfault)
        reranker = None
        if self.verbose:
            print("  [INIT] Reranker 已跳过 (评估管线已移除 R 配置)")

        # 查询预处理 (P0: HyDE 段落 + P1: 别名扩展)
        # 双路分离: dense_query -> Dense Embedding, sparse_query -> BM25
        from server.rag.query_expander import QueryExpander
        from server.rag.query_intent import research_penalty_for, detect_version_filters

        expander = QueryExpander()

        async def query_preprocessor(query: str) -> dict:
            """查询预处理管道：别名扩展 + 词权重 + 研究降权 + 版本软 boost。

            Returns:
                {"dense_query": str, "sparse_query": str, "sparse_weights": dict,
                 "research_penalty": float, "version_boost": dict | None}
            """
            research_penalty = research_penalty_for(query)
            version_boost = detect_version_filters(query)
            expand_result = expander.expand_weighted(query)
            return {
                "dense_query": query,
                "sparse_query": expand_result["expanded_query"],
                "sparse_weights": expand_result["token_weights"],
                "research_penalty": research_penalty,
                "version_boost": version_boost,
            }

        if self.verbose:
            print("[INIT] 共享组件加载完成")

        self._shared = {
            "embedder": embedder,
            "vector_store": vector_store,
            "bm25": bm25,
            "reranker": reranker,
            "query_preprocessor": query_preprocessor,
        }
        return self._shared

    def _build_retriever(self, config_id: str, cfg: dict):
        """为指定配置构建检索器实例。

        Args:
            config_id: "D" | "S" | "F" | "R"
            cfg: 配置参数字典 (与 HybridRetrieverConfig 字段对应)

        Returns:
            HybridRetriever 实例 (D/F/R) 或 BM25Retriever 实例 (S)
        """
        shared = self._build_shared_components()

        if config_id == "S":
            # S 配置: 直接返回 BM25Retriever (不包装 HybridRetriever)
            return shared["bm25"]

        from server.rag.retriever import HybridRetriever, HybridRetrieverConfig
        from server.rag.config import rag_config

        hc = HybridRetrieverConfig(
            bm25_enabled=cfg.get("bm25_enabled", False),
            reranker_enabled=cfg.get("reranker_enabled", False),
            fusion_method=cfg.get("fusion_method", "rrf"),
            fusion_k=cfg.get("fusion_k", 20),
            fusion_dense_weight=cfg.get("fusion_dense_weight", 0.3),
            dense_candidate_k=cfg.get("dense_candidate_k", 50),
            sparse_candidate_k=cfg.get("sparse_candidate_k", 50),
            reranker_output_k=cfg.get("reranker_output_k", 10),
            version_boost_factor=rag_config.version_boost_factor,
            context_window=cfg.get("context_window", 0),
            context_decay=cfg.get("context_decay", 0.9),
        )

        return HybridRetriever(
            shared["vector_store"],
            bm25_retriever=shared["bm25"] if hc.bm25_enabled else None,
            reranker=shared["reranker"] if hc.reranker_enabled else None,
            config=hc,
            # 查询预处理已在 _run_single_config 的 per-query 循环中统一应用，
            # 此处不注入 query_preprocessor，避免双重预处理
        )

    # ========================================================================
    # T03.04: debug_callback 中间结果收集
    # ========================================================================
    def _build_debug_callback(self, query_id: str):
        """构建诊断回调闭包，捕获指定查询的所有阶段数据。"""
        if query_id not in self._per_query_diagnostics:
            self._per_query_diagnostics[query_id] = {}

        async def callback(phase: str, data: dict) -> None:
            if phase not in self._per_query_diagnostics[query_id]:
                self._per_query_diagnostics[query_id][phase] = []
            self._per_query_diagnostics[query_id][phase].append(data)

        return callback

    # ========================================================================
    # T03.03: _run_single_config + 归一化适配器
    # ========================================================================
    async def _run_single_config(
        self,
        config_id: str,
        retriever,  # HybridRetriever | BM25Retriever
        top_k: int = 10,
    ) -> dict:
        """对指定配置运行全部查询。

        Args:
            config_id: "D" | "S" | "F" | "R"
            retriever: 检索器实例 (D/F/R 为 HybridRetriever, S 为 BM25Retriever)
            top_k: 检索结果数

        Returns:
            {config_id, per_query: [...], aggregates: {...}}
        """
        from . import metrics as M

        per_query: list[dict] = []
        total_start = time.monotonic()

        for q_idx, q in enumerate(self.queries):
            query_id = q.get("query_id", f"q_{q_idx:04d}")
            query_text = q.get("text", "")
            query_filters = q.get("metadata_filters")  # dict | None

            # Phase 0: 查询预处理 (P0: HyDE + P1: 别名扩展)
            # 双路分离: dense_query -> Dense Embedding, sparse_query -> BM25
            preprocessor = self._shared.get("query_preprocessor") if self._shared else None
            eq_dense = query_text
            eq_sparse = query_text
            eq_weights = None
            eq_research_penalty = 1.0
            eq_version_boost = None
            if preprocessor:
                try:
                    preprocessed = await preprocessor(query_text)
                    if isinstance(preprocessed, dict):
                        eq_dense = preprocessed.get("dense_query", query_text)
                        eq_sparse = preprocessed.get("sparse_query", query_text)
                        eq_weights = preprocessed.get("sparse_weights")
                        eq_research_penalty = preprocessed.get("research_penalty", 1.0)
                        eq_version_boost = preprocessed.get("version_boost")
                    elif isinstance(preprocessed, str) and preprocessed.strip():
                        eq_dense = eq_sparse = preprocessed
                except Exception as e:
                    logger.warning(f"评估查询预处理失败 (query={query_id}): {e}")

            query_relevant = self.qrels.get(query_id, {})

            pq: dict = {
                "query_id": query_id,
                "config_id": config_id,
                "text": query_text[:100],
                "category": q.get("category", ""),
                "difficulty": q.get("difficulty", ""),
                "retrieved_ids": [],
                "retrieved_ranks": {},
                "relevant_ids": list(query_relevant.keys()),
                "hard_negative_ids": list(
                    self._hard_negatives.get(query_id, set())
                ),
                "error": None,
                "timing_ms": 0,
            }

            try:
                # T03.12: 超时保护 (30s)
                q_start = time.monotonic()

                # T03.04: debug_callback only for D/F/R (HybridRetriever)
                debug_cb = None
                if config_id != "S" and self.debug:
                    debug_cb = self._build_debug_callback(query_id)

                if config_id == "S":
                    # S 配置: 直接调用 BM25Retriever.search()，使用关键词+别名+权重
                    results = await asyncio.to_thread(
                        retriever.search,
                        eq_sparse,
                        top_k=top_k,
                        metadata_filters=query_filters,
                        query_weights=eq_weights,
                    )
                else:
                    # D/F: 通过 HybridRetriever.retrieve()，传入双路查询 + 权重
                    eq_dict = {
                        "dense_query": eq_dense,
                        "sparse_query": eq_sparse,
                        "sparse_weights": eq_weights,
                        "research_penalty": eq_research_penalty,
                        "version_boost": eq_version_boost,
                    }
                    results = await asyncio.wait_for(
                        retriever.retrieve(
                            eq_dict,
                            top_k=top_k,
                            filters=query_filters,
                            debug_callback=debug_cb,
                        ),
                        timeout=30.0,  # T03.12
                    )

                q_elapsed = (time.monotonic() - q_start) * 1000
                pq["timing_ms"] = round(q_elapsed, 1)

                # 归一化结果
                retrieved_ids: list[str] = []
                for rank, r in enumerate(results, start=1):
                    norm = _normalize_result(r, rank, config_id)
                    retrieved_ids.append(norm["chunk_id"])
                    pq["retrieved_ranks"][norm["chunk_id"]] = rank

                pq["retrieved_ids"] = retrieved_ids

                # 计算一级指标
                relevant_set = set(query_relevant.keys())

                if relevant_set:
                    pq["recall_at_10"] = M.recall_at_k(retrieved_ids, relevant_set, 10)
                    pq["recall_at_5"] = M.recall_at_k(retrieved_ids, relevant_set, 5)
                    pq["recall_at_3"] = M.recall_at_k(retrieved_ids, relevant_set, 3)
                    pq["normalized_recall_at_10"] = M.normalized_recall_at_k(retrieved_ids, relevant_set, 10)
                    core_set = {k for k, v in query_relevant.items() if v >= 2}
                    pq["recall_at_10_core"] = (
                        M.recall_at_k(retrieved_ids, core_set, 10) if core_set else None
                    )
                    pq["mrr"] = M.mrr(retrieved_ids, relevant_set)
                    pq["ndcg_at_10"] = M.ndcg_at_k(retrieved_ids, query_relevant, 10)
                    pq["binary_ndcg_at_10"] = M.ndcg_at_k_binary(retrieved_ids, relevant_set, 10)
                    pq["hit_at_1"] = M.hit_at_k(retrieved_ids, relevant_set, 1)
                    pq["precision_at_10"] = M.precision_at_k(retrieved_ids, relevant_set, 10)
                    pq["average_precision"] = M.average_precision(retrieved_ids, relevant_set)
                    pq["r_precision"] = M.r_precision(retrieved_ids, relevant_set)
                    pq["f1_at_10"] = M.f1_at_k(retrieved_ids, relevant_set, 10)
                    pq["coverage_at_10"] = M.coverage_at_k(retrieved_ids, relevant_set, 10)
                    pq["eval_skipped"] = False
                else:
                    # 无正标注 (relevant_set 为空): 所有指标设为 None，
                    # 聚合时跳过，避免 0.0 值人为压低均值
                    pq["eval_skipped"] = True
                    for key in ("recall_at_10", "recall_at_5", "recall_at_3",
                                 "normalized_recall_at_10", "recall_at_10_core",
                                 "mrr", "ndcg_at_10", "binary_ndcg_at_10",
                                 "hit_at_1", "precision_at_10",
                                 "average_precision", "r_precision", "f1_at_10",
                                 "coverage_at_10"):
                        pq[key] = None

                # FPR: 仅对至少有 hard negatives 的查询有意义；
                # 无 hard negatives → None，聚合时跳过
                hn_ids = pq.get("hard_negative_ids", [])
                pq["false_positive_rate_at_10"] = (
                    M.false_positive_rate_at_k(retrieved_ids, set(hn_ids), 10)
                    if hn_ids else None
                )

                # Filter precision: 仅对设置了 metadata_filters 的查询计算
                pq["filter_precision_at_10"] = M.filter_precision_at_k(
                    retrieved_ids, query_filters, self._corpus_meta, 10,
                )

            except asyncio.TimeoutError:
                pq["error"] = "timeout"
                pq["timing_ms"] = 30000
                if self.verbose:
                    print(f"  [TIMEOUT] {query_id}: {query_text[:50]}")

            except Exception as e:
                pq["error"] = str(e)[:200]
                if self.verbose:
                    print(f"  [ERROR] {query_id}: {e}")

            per_query.append(pq)

            if self.verbose and (q_idx + 1) % 20 == 0:
                elapsed = time.monotonic() - total_start
                print(f"  [{config_id}] {q_idx + 1}/{len(self.queries)} ({elapsed:.0f}s)")

        # 聚合指标
        aggregates = self._compute_aggregates(per_query)

        total_elapsed = time.monotonic() - total_start
        if self.verbose:
            n_errors = sum(1 for pq in per_query if pq.get("error"))
            n_skipped = sum(1 for pq in per_query if pq.get("eval_skipped"))
            print(f"  [{config_id}] Done: {len(per_query)} queries in {total_elapsed:.0f}s "
                  f"({n_errors} errors/timeouts, {n_skipped} skipped/no-annotations)")

        return {
            "config_id": config_id,
            "per_query": per_query,
            "aggregates": aggregates,
            "total_time_s": round(total_elapsed, 1),
        }

    # ========================================================================
    # T03.05: _compute_aggregates
    # ========================================================================
    def _compute_aggregates(self, per_query: list[dict]) -> dict:
        """逐配置指标聚合 (bootstrap CI)。委托给统一的 compute_all_metrics。"""
        from . import metrics as M

        return M.compute_all_metrics(per_query)

    # ========================================================================
    # T03.06: run_all_configs
    # ========================================================================
    def run_all_configs(self, configs: dict, top_k: int = 10) -> dict[str, dict]:
        """运行 D/S/F/R 四组配置评估。

        Args:
            configs: {"D": {...}, "S": {...}, "F": {...}, "R": {...}}
            top_k: 检索结果数

        Returns:
            {"D": {...}, "S": {...}, "F": {...}, "R": {...}}
        """
        results: dict[str, dict] = {}
        config_order = ["D", "S", "F"]  # v2: 移除 R (Reranker在古籍场景无增量)

        for config_id in config_order:
            if config_id not in configs:
                continue

            cfg = configs[config_id]
            if self.verbose:
                print(f"\n{'='*60}")
                print(f"[CONFIG] {config_id}: {cfg.get('note', '')}")

            retriever = self._build_retriever(config_id, cfg)

            # T03.13: 清除缓存 (防止前一个配置的缓存污染)
            if config_id != "S" and hasattr(retriever, '_query_cache'):
                retriever._query_cache.clear()
                if self.verbose:
                    print(f"  [CACHE] Cleared query cache before config {config_id}")

            result = asyncio.run(
                self._run_single_config(config_id, retriever, top_k=top_k)
            )
            results[config_id] = result

        return results

    # ========================================================================
    # T03.08: run_chunking_ablation (简化版 — 复杂索引重建待 Phase 5)
    # ========================================================================
    def run_chunking_ablation(
        self, ablation_config: dict, quick: bool = False, resume: bool = True,
    ) -> dict:
        """Chunking 消融实验。

        注意: 此占位实现用于接口验证，完整实现在 T03.08 索引重建中。
        """
        variants = ablation_config.get("ablation", {}).get("variants", {})
        if quick:
            variants = {
                k: v for k, v in variants.items()
                if k in ablation_config.get("quick_mode_variants", ["C0", "C1", "C2"])
            }

        results: dict[str, dict] = {}
        for variant_id, variant_cfg in variants.items():
            if self.verbose:
                print(f"[ABLATION] {variant_id}: {variant_cfg.get('desc', '')}")
                print(f"  changes: {variant_cfg.get('changes', {})}")
            results[variant_id] = {
                "variant": variant_id,
                "status": "not_implemented",
                "note": "完整索引重建实现在 T03.08",
            }

        return results

    # ========================================================================
    # T03.09: 降级路径评估
    # ========================================================================
    async def run_degradation_test(self) -> dict:
        """测试全部降级路径。

        Returns:
            {scenario: {results, status, ...}}
        """
        results: dict = {}
        shared = self._build_shared_components()
        from server.rag.retriever import HybridRetriever, HybridRetrieverConfig

        # Sample one query for degradation testing
        test_q = self.queries[0] if self.queries else {"text": "test", "query_id": "test"}
        query_text = test_q.get("text", "test")
        filters = test_q.get("metadata_filters")

        scenarios = {
            "reranker_degraded": {
                "desc": "Reranker 不可用",
                "config": HybridRetrieverConfig(
                    bm25_enabled=True, reranker_enabled=True,
                    dense_candidate_k=50, sparse_candidate_k=50, reranker_output_k=10,
                ),
            },
            "bm25_degraded": {
                "desc": "BM25 不可用",
                "config": HybridRetrieverConfig(
                    bm25_enabled=True, reranker_enabled=False,
                    dense_candidate_k=50, sparse_candidate_k=50,
                ),
            },
        }

        for scenario_id, s in scenarios.items():
            try:
                # 创建带有 fake Reranker/Bm25 状态的检索器
                retriever = HybridRetriever(
                    shared["vector_store"],
                    bm25_retriever=shared["bm25"],
                    reranker=shared["reranker"],
                    config=s["config"],
                )
                # 模拟降级状态
                if scenario_id == "reranker_degraded":
                    retriever.reranker._is_ready = False
                elif scenario_id == "bm25_degraded":
                    if retriever.bm25:
                        retriever.bm25._built = False

                res = await retriever.retrieve(query_text, top_k=10, filters=filters)
                results[scenario_id] = {
                    "status": "ok",
                    "results_count": len(res),
                    "error": None,
                }
            except Exception as e:
                results[scenario_id] = {
                    "status": "error",
                    "results_count": 0,
                    "error": str(e)[:200],
                }

        return results

    # ========================================================================
    # T03.10: 融合策略消融
    # ========================================================================
    def run_fusion_ablation(self, base_config: dict | None = None) -> dict:
        """对比三种融合策略: RRF vs ScoreNormalized vs LinearCombination。"""
        # 从 base_config 读取非扫描参数
        fw = base_config.get("fusion_dense_weight", 0.5) if base_config else 0.5
        fk = base_config.get("fusion_k", 20) if base_config else 20
        dk = base_config.get("dense_candidate_k", 100) if base_config else 100
        sk = base_config.get("sparse_candidate_k", 100) if base_config else 100

        results: dict = {}
        for method in ["rrf", "score_normalized", "linear_combination"]:
            cfg = {
                "bm25_enabled": True,
                "reranker_enabled": False,
                "fusion_method": method,
                "fusion_k": fk,
                "fusion_dense_weight": fw,
                "dense_candidate_k": dk,
                "sparse_candidate_k": sk,
                "reranker_output_k": 10,
            }
            retriever = self._build_retriever("F", cfg)
            if hasattr(retriever, '_query_cache'):
                retriever._query_cache.clear()
            result = asyncio.run(self._run_single_config("F", retriever, top_k=10))
            result["fusion_method"] = method
            results[method] = result

        return results

    # ========================================================================
    # T03.11: 融合权重扫描
    # ========================================================================
    def run_weight_sweep(self, base_config: dict | None = None) -> list[dict]:
        """扫描 fusion_dense_weight 从 0.0 到 1.0，步长 0.1。

        直接打印结果表格到 stdout，返回结构化数据供 caller 使用。

        Args:
            base_config: 从 YAML 加载的 F 配置，用作非扫描参数的默认值。
                         若为 None，使用内置默认值。

        Returns:
            [{weight, recall_at_10, mrr, ndcg_at_10, coverage_at_10, n_queries}, ...]
        """
        # 从 base_config 提取固定参数，扫描参数使用默认值兜底
        fusion_method = base_config.get("fusion_method", "score_normalized") if base_config else "score_normalized"
        fusion_k = base_config.get("fusion_k", 20) if base_config else 20
        dense_candidate_k = base_config.get("dense_candidate_k", 100) if base_config else 100
        sparse_candidate_k = base_config.get("sparse_candidate_k", 100) if base_config else 100

        table_rows: list[dict] = []
        total_start = time.monotonic()

        print(f"\n{'='*65}")
        print(f"  Weight Sweep: fusion_dense_weight 0.0 → 1.0 (step=0.1)")
        print(f"  Fusion method: {fusion_method}  |  candidate_k: {dense_candidate_k}/{sparse_candidate_k}")
        print(f"{'='*65}")
        print(f"  {'Weight':<8} {'Recall@10':<12} {'MRR':<12} {'NDCG@10':<12} {'Coverage':<12} {'Time':<8}")
        print(f"  {'-'*60}")

        for w_int in range(0, 11):
            w = w_int / 10.0
            cfg = {
                "bm25_enabled": True,
                "reranker_enabled": False,
                "fusion_method": fusion_method,
                "fusion_k": fusion_k,
                "fusion_dense_weight": w,
                "dense_candidate_k": dense_candidate_k,
                "sparse_candidate_k": sparse_candidate_k,
                "reranker_output_k": 10,
            }
            retriever = self._build_retriever("F", cfg)
            if hasattr(retriever, '_query_cache'):
                retriever._query_cache.clear()

            w_start = time.monotonic()
            result = asyncio.run(self._run_single_config("F", retriever, top_k=10))
            w_elapsed = time.monotonic() - w_start

            recall_val = result["aggregates"].get("recall_at_10", {}).get("mean", 0.0)
            mrr_val = result["aggregates"].get("mrr", {}).get("mean", 0.0)
            ndcg_val = result["aggregates"].get("ndcg_at_10", {}).get("mean", 0.0)
            coverage_val = result["aggregates"].get("coverage_at_10", {}).get("mean", 0.0)
            n_q = result["aggregates"].get("recall_at_10", {}).get("n", 0)

            row = {
                "weight": w,
                "recall_at_10": round(recall_val, 4),
                "mrr": round(mrr_val, 4),
                "ndcg_at_10": round(ndcg_val, 4),
                "coverage_at_10": round(coverage_val, 4),
                "n_queries": n_q,
            }
            table_rows.append(row)

            # 实时打印每行
            marker = " <-- BEST" if w == max(
                [r["recall_at_10"] for r in table_rows] or [0]
            ) else ""
            print(
                f"  {w:<8.1f} {recall_val:<12.4f} {mrr_val:<12.4f} "
                f"{ndcg_val:<12.4f} {coverage_val:<12.4f} {w_elapsed:.0f}s{marker}"
            )

        total_elapsed = time.monotonic() - total_start

        # 找出最优权重
        best_by_recall = max(table_rows, key=lambda r: r["recall_at_10"])
        best_by_mrr = max(table_rows, key=lambda r: r["mrr"])

        print(f"  {'-'*60}")
        print(f"  Best Recall@10: weight={best_by_recall['weight']:.1f} → "
              f"{best_by_recall['recall_at_10']:.4f}")
        print(f"  Best MRR:       weight={best_by_mrr['weight']:.1f} → "
              f"{best_by_mrr['mrr']:.4f}")
        print(f"  Total sweep time: {total_elapsed:.0f}s ({total_elapsed/11:.0f}s/weight)")
        print(f"{'='*65}\n")

        return table_rows

    # ========================================================================
    # T03.14: top_k 方差评估
    # ========================================================================
    def run_topk_variance(self, k_values: list[int] | None = None) -> dict:
        """多 top_k 值 Recall 曲线。

        Args:
            k_values: [3, 5, 10, 20, 50] 默认

        Returns:
            {top_k_value: {"recall_at_k": float, "mrr": float, ...}}
        """
        if k_values is None:
            k_values = [3, 5, 10, 20, 50]

        results: dict = {}
        # 使用 R 配置
        cfg = {
            "bm25_enabled": True, "reranker_enabled": True,
            "fusion_method": "rrf", "fusion_k": 20,
            "fusion_dense_weight": 0.3, "dense_candidate_k": 50,
            "sparse_candidate_k": 50, "reranker_output_k": 10,
        }

        for k in k_values:
            retriever = self._build_retriever("R", cfg)
            if hasattr(retriever, '_query_cache'):
                retriever._query_cache.clear()
            result = asyncio.run(self._run_single_config("R", retriever, top_k=k))
            results[f"k={k}"] = result["aggregates"]

        return results


# ==============================================================================
# T03.07: CLI
# ==============================================================================
def main():
    """评估脚本 CLI 入口。"""
    import argparse

    parser = argparse.ArgumentParser(
        description="RAG 检索评估系统",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--config", required=True, help="YAML 配置文件路径")
    parser.add_argument("--ci", action="store_true", help="仅运行 CI 子集 (30 条)")
    parser.add_argument("--query", type=str, help="仅测试单条查询 (调试用)")
    parser.add_argument("--verbose", action="store_true", help="Level 2 详细日志")
    parser.add_argument("--debug", action="store_true", help="Level 3 调试日志 (含诊断)")
    parser.add_argument("--ablation", choices=["chunking", "fusion", "weight"], help="消融实验类型")
    parser.add_argument("--quick", action="store_true", help="Quick 模式 (仅限消融实验)")
    parser.add_argument("--output-dir", default="rag_eval/results", help="报告输出目录")
    parser.add_argument("--top-k", type=int, default=10, help="检索结果数")
    parser.add_argument("--restore-index", action="store_true", help="仅恢复索引 (不跑消融)")

    args = parser.parse_args()

    # 配置日志
    log_level = logging.DEBUG if args.debug else (logging.INFO if args.verbose else logging.WARNING)
    logging.basicConfig(level=log_level, format="%(asctime)s [%(levelname)s] %(message)s")

    # 加载 YAML 配置
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"[ERROR] 配置文件不存在: {config_path}")
        sys.exit(1)

    with open(config_path, "r", encoding="utf-8") as f:
        test_config = yaml.safe_load(f)

    # 加载 ground truth
    from .loader import load_corpus, load_queries, load_qrels, load_all_qrels

    gt_dir = Path(test_config.get("evaluation", {}).get("dataset", {}).get("path", "rag_eval/ground_truth"))
    corpus_file = gt_dir / test_config.get("evaluation", {}).get("dataset", {}).get("corpus_file", "corpus.jsonl")
    queries_file = gt_dir / test_config.get("evaluation", {}).get("dataset", {}).get("queries_file", "queries.jsonl")
    qrels_file = gt_dir / test_config.get("evaluation", {}).get("dataset", {}).get("qrels_file", "qrels.tsv")

    print(f"[LOAD] 加载数据集: {gt_dir}")
    corpus = load_corpus(corpus_file)
    queries = load_queries(queries_file)
    qrels = load_qrels(qrels_file)
    all_qrels = load_all_qrels(qrels_file)
    hard_neg_count = sum(
        len([r for r in rels.values() if r == 0])
        for rels in all_qrels.values()
    )
    print(f"  corpus: {len(corpus)} docs")
    print(f"  queries: {len(queries)} queries")
    print(f"  qrels: {len(qrels)} query-groups (positive only)")
    print(f"  hard negatives: {hard_neg_count} total (rel=0)")

    # CI 模式: 采样
    if args.ci:
        ci_file = gt_dir / test_config.get("evaluation", {}).get("dataset", {}).get("ci_queries_file", "queries_ci.jsonl")
        if ci_file.exists():
            queries = load_queries(ci_file)
            print(f"  [CI] Using CI subset: {len(queries)} queries")
        else:
            from .loader import sample_ci_subset
            queries = sample_ci_subset(queries)
            print(f"  [CI] Sampled subset: {len(queries)} queries")

    # 单条模式
    if args.query:
        queries = [q for q in queries if q.get("query_id") == args.query]
        if not queries:
            print(f"[ERROR] 未找到查询: {args.query}")
            sys.exit(1)
        print(f"  [DEBUG] Single query: {args.query}")

    # T03.18: 检测无 qrels 标注的查询
    unannotated_ids = [
        q.get("query_id", f"q_{i:04d}")
        for i, q in enumerate(queries)
        if q.get("query_id", "") not in qrels
    ]
    if unannotated_ids:
        print(
            f"  [WARN] {len(unannotated_ids)}/{len(queries)} 条查询缺少 qrels 标注 "
            f"({len(unannotated_ids)/len(queries)*100:.1f}%) — "
            f"这些查询的所有指标将被计为 0.0"
        )
        if args.verbose and len(unannotated_ids) <= 20:
            print(f"    缺失标注的查询ID: {', '.join(unannotated_ids)}")
        elif args.verbose:
            print(f"    缺失标注的查询ID (前20): {', '.join(unannotated_ids[:20])}")

    # 构建评估器
    evaluator = RetrievalEvaluator(
        corpus=corpus,
        queries=queries,
        qrels=qrels,
        verbose=args.verbose or args.debug,
        debug=args.debug,
        all_qrels=all_qrels,
    )

    # 确定 top_k
    top_k = args.top_k

    # 运行评估
    configs = test_config.get("configs", {})
    if args.ablation == "chunking":
        ablation_config_path = Path(__file__).resolve().parent / "configs" / "chunking_ablation.yaml"
        with open(ablation_config_path, "r", encoding="utf-8") as f:
            ablation_cfg = yaml.safe_load(f)
        results = evaluator.run_chunking_ablation(ablation_cfg, quick=args.quick)
    elif args.ablation == "fusion":
        f_cfg = configs.get("F", {}) if configs else {}
        results = evaluator.run_fusion_ablation(base_config=f_cfg)
    elif args.ablation == "weight":
        # 权重扫描：传入 base_config 中的 F 配置作为非扫描参数
        f_cfg = configs.get("F", {}) if configs else {}
        evaluator.run_weight_sweep(base_config=f_cfg)
        # 权重扫描直接打印到 stdout，不生成报告文件
        print("[DONE] Weight sweep complete (no report file generated).")
        return
    else:
        results = evaluator.run_all_configs(configs, top_k=top_k)

    # 保存配置路径供报告使用
    test_config["_config_path"] = str(config_path)

    # 生成报告 (使用 reporter.generate_report)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%dT%H%M%S")

    from .reporter import generate_report
    generate_report(
        results,
        test_config,
        output_dir,
        timestamp=timestamp,
        corpus=corpus,
        queries=queries,
        qrels=qrels,
        all_qrels=all_qrels,
    )

    report_dir = output_dir / timestamp
    print(f"\n[DONE] Reports saved to: {report_dir}")
    print(f"  {report_dir / 'report.txt'}")
    print(f"  {report_dir / 'report.json'}")

    # CI 门禁判断 (v2: 使用 F 配置替代已移除的 R 配置)
    thresholds = test_config.get("thresholds", {})
    # 优先取 F 配置, 若不存在则回退到第一个可用配置
    gate_config_id = "F" if "F" in results else next(iter(results), "F")
    r_agg = results.get(gate_config_id, {}).get("aggregates", {})

    recall10 = r_agg.get("recall_at_10", {}).get("mean", 0.0)
    mrr_val = r_agg.get("mrr", {}).get("mean", 0.0)
    ndcg10 = r_agg.get("ndcg_at_10", {}).get("mean", 0.0)

    # 各指标阈值
    recall_fail = thresholds.get("recall_at_10_fail", 0.65)
    recall_warn = thresholds.get("recall_at_10_warn", 0.75)
    mrr_fail = thresholds.get("mrr_fail", 0.40)
    mrr_warn = thresholds.get("mrr_warn", 0.55)
    ndcg_fail = thresholds.get("ndcg_at_10_fail", 0.50)
    ndcg_warn = thresholds.get("ndcg_at_10_warn", 0.65)

    # 逐指标判断
    gate_results = []
    for name, val, fail_t, warn_t in [
        ("recall_at_10", recall10, recall_fail, recall_warn),
        ("mrr", mrr_val, mrr_fail, mrr_warn),
        ("ndcg_at_10", ndcg10, ndcg_fail, ndcg_warn),
    ]:
        if val < fail_t:
            gate_results.append(("FAIL", name, val, fail_t))
        elif val < warn_t:
            gate_results.append(("WARN", name, val, warn_t))
        else:
            gate_results.append(("PASS", name, val, warn_t))

    # 输出逐指标状态
    print(f"\n[GATE] CI 门禁 ({gate_config_id} 配置, 多指标):")
    for status, name, val, threshold in gate_results:
        marker = "[FAIL]" if status == "FAIL" else ("[WARN]" if status == "WARN" else "[PASS]")
        print(f"  {marker} {name}: {val:.4f} (threshold: {threshold:.2f})")

    # 组合判断: 任一 FAIL → exit(1); 任一 WARN → stderr
    has_fail = any(s == "FAIL" for s, _, _, _ in gate_results)
    has_warn = any(s == "WARN" for s, _, _, _ in gate_results)

    if has_fail:
        failed_names = [n for s, n, _, _ in gate_results if s == "FAIL"]
        print(f"[GATE] FAIL: {', '.join(failed_names)} 低于 fail 阈值", file=sys.stderr)
        sys.exit(1)
    elif has_warn:
        warned_names = [n for s, n, _, _ in gate_results if s == "WARN"]
        print(f"[GATE] WARN: {', '.join(warned_names)} 低于 warn 阈值", file=sys.stderr)
    else:
        print(f"[GATE] PASS: 所有指标 >= warn 阈值")


if __name__ == "__main__":
    main()
