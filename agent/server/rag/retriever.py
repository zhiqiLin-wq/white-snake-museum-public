"""两阶段混合检索：双路粗排 → RRF融合。

CU-32~CU-38: HybridRetriever 两阶段管道。
Phase 1: Dense + Sparse 双路召回
Phase 2: RRF 加权融合 (默认 dense=0.3 / sparse=0.7)，按融合分排序返回
Cross-Encoder Reranker 保留为可选替代路径（历史证明在此语料上无增量）。
"""
import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from typing import Awaitable, Callable, Optional

# B-163: 内容级段落重定位（出场轨迹等工具的 paragraph_index 修正）
from .paragraph_locator import get_paragraph_locator

logger = logging.getLogger(__name__)

# chunk metadata 的 chapter_number 存的是中文数字（'一'~'七'），研究文献为 '研究'。
# 前端跳转需要 int 章节号（Number('七') = NaN 会导致 TextReader 空白），
# 输出给前端前必须经此转换；非正文章节归为 ''（isResearch）。
_CH_CHINESE_TO_INT = {"一": 1, "二": 2, "三": 3, "四": 4, "五": 5, "六": 6, "七": 7}


def chapter_number_to_int(raw) -> int | str:
    """chunk metadata 的章节号 → 前端可用的 int；正文以外的值归为 ''。"""
    if isinstance(raw, int):
        return raw if 1 <= raw <= 99 else ""
    s = str(raw).strip()
    if s in _CH_CHINESE_TO_INT:
        return _CH_CHINESE_TO_INT[s]
    if s.isdigit():
        return int(s)
    return ""


@dataclass
class RetrievedDoc:
    content: str
    metadata: dict = field(default_factory=dict)
    score: float = 0.0
    relevance_score: float = 0.0

    @property
    def page_content(self) -> str:
        return self.content


MAX_TOP_K = 50


# ============================================================================
# CU-32: HybridRetrieverConfig
# ============================================================================
@dataclass
class HybridRetrieverConfig:
    bm25_enabled: bool = True
    reranker_enabled: bool = True
    fusion_method: str = "score_normalized"
    fusion_k: int = 20
    fusion_dense_weight: float = 0.5
    dense_candidate_k: int = 100
    sparse_candidate_k: int = 100
    reranker_output_k: int = 10
    reranker_top_k_multiplier: int = 3
    bm25_top_k_multiplier: int = 5
    # 版本软 boost 系数（>1 升匹配版本，不屏蔽别版本；1.0 = 关闭）
    version_boost_factor: float = 1.5
    # 上下文扩展：把同章相邻 chunk（seq ± window）补进融合候选，分数乘 decay^delta。
    # 0 = 关闭。针对「找不全」根因：相关 chunk 60%+ 是同章相邻段（连续 run）。
    context_window: int = 0
    context_decay: float = 0.9


# ============================================================================
# CU-33: RRF 融合算法
# ============================================================================
def _reciprocal_rank_fusion(
    dense_results: list[tuple[str, float]],
    sparse_results: list[tuple[str, float]],
    k: int = 20,
    dense_weight: float = 0.3,
    sparse_weight: float = 0.7,
) -> list[tuple[str, float]]:
    """加权 Reciprocal Rank Fusion。

    Args:
        dense_results: [(doc_id, score), ...]  语义检索结果
        sparse_results: [(doc_id, score), ...]  BM25 结果
        k: RRF 常数
        dense_weight: 语义排名权重 (0~1)
        sparse_weight: 关键词排名权重 (0~1)

    Returns:
        [(doc_id, fused_score), ...] 按融合分数降序
    """
    scores: dict[str, float] = {}
    for rank, (doc_id, _score) in enumerate(dense_results, start=1):
        scores[doc_id] = scores.get(doc_id, 0.0) + dense_weight / (k + rank)
    for rank, (doc_id, _score) in enumerate(sparse_results, start=1):
        scores[doc_id] = scores.get(doc_id, 0.0) + sparse_weight / (k + rank)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


# ============================================================================
# CU-34: Score Normalized 融合算法
# ============================================================================
def _score_normalized_fusion(
    dense_results: list[tuple[str, float]],
    sparse_results: list[tuple[str, float]],
    dense_weight: float = 0.3,
) -> list[tuple[str, float]]:
    """Min-max 归一化 + 线性加权融合。

    Args:
        dense_results: [(doc_id, score), ...]
        sparse_results: [(doc_id, score), ...]
        dense_weight: 语义部分权重 (0~1)

    Returns:
        [(doc_id, fused_score), ...] 按融合分数降序
    """
    # 提取分数列表
    dense_dict: dict[str, float] = {}
    for doc_id, score in dense_results:
        # 同一个doc_id取最高分
        dense_dict[doc_id] = max(dense_dict.get(doc_id, 0.0), score)

    sparse_dict: dict[str, float] = {}
    for doc_id, score in sparse_results:
        sparse_dict[doc_id] = max(sparse_dict.get(doc_id, 0.0), score)

    # Min-max 归一化
    def _minmax_normalize(d: dict[str, float]) -> dict[str, float]:
        if not d:
            return d
        values = list(d.values())
        v_min = min(values)
        v_max = max(values)
        if v_max == v_min:
            return {k: 0.5 for k in d}
        return {k: (v - v_min) / (v_max - v_min) for k, v in d.items()}

    dense_norm = _minmax_normalize(dense_dict)
    sparse_norm = _minmax_normalize(sparse_dict)

    # 线性加权
    all_ids = set(dense_norm.keys()) | set(sparse_norm.keys())
    fused: dict[str, float] = {}
    for doc_id in all_ids:
        d_score = dense_norm.get(doc_id, 0.0)
        s_score = sparse_norm.get(doc_id, 0.0)
        fused[doc_id] = dense_weight * d_score + (1.0 - dense_weight) * s_score

    return sorted(fused.items(), key=lambda x: x[1], reverse=True)


def _to_chroma_where(filters: Optional[dict]) -> Optional[dict]:
    """把硬过滤 filters 透传给 ChromaDB where（单值匹配）。

    业务硬过滤（evolution_workbench / MCP tools）只用单值 filters，无 list 值。
    版本同名书的 OR 匹配走 _meta_matches（软 boost），不经此处。
    """
    if not filters:
        return None
    return dict(filters)


def _meta_matches(meta: dict, cond: dict) -> bool:
    """判断 metadata 是否满足约束 cond。

    cond 语义同 filters：value 为 list 表示 in 匹配（OR），否则等于匹配。
    所有字段都命中才算匹配（AND）。
    """
    for k, v in cond.items():
        mv = meta.get(k)
        if isinstance(v, list):
            if mv not in v:
                return False
        elif mv != v:
            return False
    return True


# ============================================================================
# CU-32~CU-38: HybridRetriever 三阶段管道
# ============================================================================
class HybridRetriever:
    """两阶段混合检索器。

    Phase 1: Dense + Sparse 双路粗排
    Phase 2: RRF / Score Normalized 融合，按融合分排序返回
    Phase 3 (可选): Cross-Encoder Reranker 精排
    """

    # 默认权重：语义检索 0.3 / 关键词检索 0.7。
    # 古典文献语料上 BM25 召回是语义检索的 2x+，因此以关键词为主导。
    # 这些常量作为类级参考；实际融合权重由 config.fusion_dense_weight 控制。
    SEMANTIC_WEIGHT = 0.3
    KEYWORD_WEIGHT = 0.7

    def __init__(self, vector_store,
                 bm25_retriever=None,
                 reranker=None,
                 config: Optional[HybridRetrieverConfig] = None,
                 query_preprocessor=None):
        self.vs = vector_store
        self.bm25 = bm25_retriever
        self.reranker = reranker
        self._query_preprocessor = query_preprocessor

        if config is None:
            from .config import rag_config
            config = HybridRetrieverConfig(
                bm25_enabled=rag_config.bm25_enabled,
                reranker_enabled=rag_config.reranker_enabled,
                fusion_method=rag_config.fusion_method_validated,
                fusion_k=rag_config.fusion_k,
                fusion_dense_weight=rag_config.fusion_dense_weight,
                dense_candidate_k=rag_config.dense_candidate_k,
                sparse_candidate_k=rag_config.sparse_candidate_k,
                reranker_output_k=rag_config.reranker_output_k,
                reranker_top_k_multiplier=rag_config.reranker_top_k_multiplier,
                bm25_top_k_multiplier=rag_config.bm25_top_k_multiplier,
                version_boost_factor=rag_config.version_boost_factor,
                context_window=rag_config.context_window,
                context_decay=rag_config.context_decay,
            )
        self.config = config

        self._query_cache: dict[str, tuple[float, list[dict]]] = {}
        self._cache_ttl = 300

    # ========================================================================
    # CU-36: BM25 结果到 doc_id 的映射 — DEPRECATED
    # ========================================================================
    # DEPRECATED: chunk_id 直传已替代 content 匹配。保留供未来 ID 迁移场景参考。
    # 当前代码路径中未被调用——BM25 结果直接通过 chunk_id 字段做 ID 关联。
    # T03.18: 全仓库搜索确认零调用方后标记为 deprecated。
    def _map_bm25_to_doc_ids(
        self, bm25_results: list[dict],
    ) -> list[tuple[str, float]]:
        """通过 content 精确匹配，将 BM25 结果映射到 VectorStore 的 doc_id。

        VectorStore 使用 "chunk_N" 作为 doc_id。
        无法匹配的结果丢弃，记录 warning。
        """
        if not bm25_results:
            return []

        mapped: list[tuple[str, float]] = []
        for r in bm25_results:
            content = r.get("content", "")
            score = r.get("bm25_score", 0.0)
            # 通过 content 生成与 VectorStore 一致的 chunk_id
            # VectorStore 按顺序使用 "chunk_{i}" 作为 ID
            # 策略：通过 content hash 匹配
            # 实际上我们需要更可靠的映射方式
            # 这里使用 content 本身作为临时 key
            mapped.append((content, score))

        if not mapped:
            logger.warning("BM25 结果全部无法映射到 doc_id")
        return mapped

    # ========================================================================
    # CU-35: 三阶段检索主流程
    # ========================================================================
    async def retrieve(self, query: str, top_k: int | None = None,
                       filters: Optional[dict] = None,
                       version_boost: Optional[dict] = None,
                       debug_callback: Optional[Callable[[str, dict], Awaitable[None]]] = None) -> list[dict]:
        if top_k is None:
            from .config import rag_config
            top_k = rag_config.default_top_k
        """三阶段混合检索主入口。

        Args:
            query: 搜索查询 (str)，或预处理后的双路查询 (dict {dense_query, sparse_query})
            top_k: 返回结果数
            filters: metadata 硬过滤条件，如 {"dynasty": "明"}（只检索匹配 chunk）
            version_boost: 版本软 boost 约束，如 {"genre": "opera_libretto"}
                （匹配版本融合分上浮，别版本不屏蔽；与 filters 独立）
            debug_callback: 可选诊断回调，接收 (phase_name, phase_data) 并返回 Awaitable。
        """
        top_k = min(top_k, MAX_TOP_K)

        # ---- Phase 0: 查询预处理 (HyDE + Expansion) ----
        # 双路分离: dense_query (自然语言) -> Dense Embedding
        #           sparse_query (关键词)  -> BM25
        #
        # 支持两种输入模式:
        #   str: 原始查询，由内部 preprocessor 处理（main.py 生产路径）
        #   dict: 外部已预处理 {dense_query, sparse_query}（evaluate.py 评估路径）
        #         此时内部 preprocessor 为 None，避免双重预处理
        original_query = query
        dense_query = query
        sparse_query = query

        # 处理外部传入的 dict（评估管线预处理器已在外部调用）
        sparse_weights: dict[str, float] | None = None
        research_penalty: float = 1.0
        if isinstance(query, dict):
            ext_dense = query.get("dense_query", "")
            ext_sparse = query.get("sparse_query", "")
            sparse_weights = query.get("sparse_weights")  # optional
            research_penalty = query.get("research_penalty", 1.0)  # optional
            version_boost = query.get("version_boost") or version_boost  # optional
            # 选择原始查询字符串作为 fallback 标识
            original_query = ext_dense or ext_sparse
            if ext_dense and ext_dense.strip():
                dense_query = ext_dense
            if ext_sparse and ext_sparse.strip():
                sparse_query = ext_sparse

        # 内部预处理器（main.py 生产路径：注入 Expander）
        if self._query_preprocessor is not None:
            # 使用原始字符串查询调用预处理器
            raw_query = original_query
            if isinstance(raw_query, dict):
                raw_query = raw_query.get("dense_query", "") or raw_query.get("sparse_query", "")
            try:
                preprocessed = await self._query_preprocessor(raw_query)
                if isinstance(preprocessed, dict):
                    dq = preprocessed.get("dense_query", "")
                    sq = preprocessed.get("sparse_query", "")
                    # 内部预处理器返回的权重优先于外部传入的
                    sw = preprocessed.get("sparse_weights")
                    if sw is not None:
                        sparse_weights = sw
                    rp = preprocessed.get("research_penalty")
                    if rp is not None:
                        research_penalty = rp
                    # 版本软 boost：内部预处理器解析出的 version_boost 优先于调用方参数
                    vb = preprocessed.get("version_boost")
                    if vb:
                        version_boost = vb
                    if dq and dq.strip():
                        dense_query = dq
                    if sq and sq.strip():
                        sparse_query = sq
                elif isinstance(preprocessed, str) and preprocessed.strip():
                    dense_query = sparse_query = preprocessed
            except Exception as e:
                logger.warning(f"查询预处理失败，使用原始查询: {e}")

        # 安全归一化：确保 dense_query / sparse_query 是有效字符串
        def _to_str(v, fallback: str = "") -> str:
            if isinstance(v, str) and v.strip():
                return v
            if isinstance(v, dict):
                return v.get("dense_query", "") or v.get("sparse_query", "") or fallback
            return fallback

        dense_query = _to_str(dense_query, original_query if isinstance(original_query, str) else "")
        sparse_query = _to_str(sparse_query, dense_query)

        # 检查缓存
        import hashlib
        cache_raw = f"{dense_query}|{sparse_query}|{top_k}|{research_penalty}|{json.dumps(filters or {}, sort_keys=True)}|{json.dumps(version_boost or {}, sort_keys=True)}|{self.config.version_boost_factor}"
        cache_key = hashlib.md5(cache_raw.encode()).hexdigest()
        if cache_key in self._query_cache:
            ts, result = self._query_cache[cache_key]
            if time.monotonic() - ts < self._cache_ttl:
                return result

        try:
            result = await self._three_stage_search(
                query, top_k, filters=filters,
                version_boost=version_boost,
                debug_callback=debug_callback,
                dense_query=dense_query,
                sparse_query=sparse_query,
                sparse_weights=sparse_weights,
                research_penalty=research_penalty,
            )
            self._query_cache[cache_key] = (time.monotonic(), result)
            return result
        except (RuntimeError, ValueError, ConnectionError) as e:
            logger.warning(f"三阶段检索失败: {e}")
            return await self._degraded_search(
                query, top_k, filters=filters,
                dense_query=dense_query,
            )

    async def _three_stage_search(self, query: str, top_k: int = 5,
                                   filters: Optional[dict] = None,
                                   version_boost: Optional[dict] = None,
                                   debug_callback: Optional[Callable[[str, dict], Awaitable[None]]] = None,
                                   dense_query: str = "",
                                   sparse_query: str = "",
                                   sparse_weights: dict[str, float] | None = None,
                                   research_penalty: float = 1.0) -> list[dict]:
        """两阶段检索实现：双路粗排 → RRF融合 + 可选Reranker精排 (CU-35)。

        Args:
            query: 原始查询 (fallback 用)
            dense_query: Dense 路使用的查询文本 (HyDE 段落或原句)
            sparse_query: Sparse 路使用的查询文本 (关键词+别名)
            sparse_weights: BM25 token 权重 {token: weight}，原词=1.0 别名=0.5
        """
        from .embedder import Embedder

        embedder = self.vs.embedder

        # 回退：如果没有指定，使用原始 query
        _dense_q = dense_query if dense_query else query
        _sparse_q = sparse_query if sparse_query else query

        # ================================================================
        # Phase 1: 双路粗排
        # ================================================================
        # Dense 路 — 使用自然语言段落 (HyDE passage 或原句)
        dense_candidate_k = self.config.dense_candidate_k
        chunk_id_to_meta: dict[str, dict] = {}  # U05: chunk_id -> metadata lookup
        if not embedder.is_ready:
            # CU-37: Embedder 不可用 → 跳过语义检索
            logger.warning("Embedder 未就绪，跳过语义检索")
            dense_results: list[tuple[str, float]] = []
        else:
            try:
                query_vec = embedder.embed_query(_dense_q)
                semantic_raw = self.vs.query(
                    query_vec,
                    top_k=dense_candidate_k,
                    where=_to_chroma_where(filters),
                )
            except Exception as e:
                # 向量库异常（集合为空/锁冲突等）不再让整路检索崩溃——
                # 降级为仅 BM25/空结果，工具仍可返回可用信息
                logger.warning(f"Dense 语义检索失败，降级处理: {e}")
                semantic_raw = []
            dense_results = []
            for r in semantic_raw:
                content = r.get("content", "")
                score = r.get("score", 0.0)
                meta = r.get("metadata", {})
                chunk_id = r.get("chunk_id", "")
                if not chunk_id:
                    chunk_id = f"dense_{hash(content) & 0xFFFFFFFF:08x}"
                    logger.warning(f"Dense 结果缺少 chunk_id，使用临时 id: {chunk_id[:20]}")
                dense_results.append((chunk_id, score))
                if chunk_id and chunk_id not in chunk_id_to_meta:
                    meta_copy = dict(meta)
                    meta_copy["_content"] = content
                    chunk_id_to_meta[chunk_id] = meta_copy

        # T00.07: Phase 1 Dense 诊断回调
        if debug_callback:
            try:
                await debug_callback("phase1_dense", {
                    "results": list(dense_results),
                    "top_k": dense_candidate_k,
                })
            except Exception as e:
                logger.warning(f"debug_callback phase1_dense 失败: {e}")

        # Sparse 路 (BM25)
        sparse_results: list[tuple[str, float]] = []
        bm25_available = (
            self.config.bm25_enabled
            and self.bm25 is not None
            and self.bm25._built
        )
        if bm25_available:
            sparse_candidate_k = self.config.sparse_candidate_k
            try:
                # B-160: BM25 同步打分同样丢线程池（纯读，线程安全）
                bm25_raw = await asyncio.to_thread(
                    self.bm25.search,
                    _sparse_q,
                    top_k=sparse_candidate_k,
                    metadata_filters=filters,
                    query_weights=sparse_weights,
                )
                for r in bm25_raw:
                    content = r.get("content", "")
                    score = r.get("bm25_score", 0.0)
                    meta = r.get("metadata", {})
                    chunk_id = r.get("chunk_id", "")
                    if not chunk_id:
                        chunk_id = f"sparse_{hash(content) & 0xFFFFFFFF:08x}"
                        logger.warning(f"BM25 结果缺少 chunk_id，使用临时 id: {chunk_id[:20]}")
                    sparse_results.append((chunk_id, score))
                    if chunk_id and chunk_id not in chunk_id_to_meta:
                        meta_copy = dict(meta)
                        meta_copy["_content"] = content
                        chunk_id_to_meta[chunk_id] = meta_copy
            except (RuntimeError, ValueError) as e:
                logger.warning(f"BM25 检索失败: {e}")
                bm25_available = False
        else:
            logger.info("BM25 不可用，仅使用 Dense 检索")

        # T00.07: Phase 1 Sparse 诊断回调
        if debug_callback:
            try:
                await debug_callback("phase1_sparse", {
                    "results": list(sparse_results),
                    "top_k": self.config.sparse_candidate_k if bm25_available else 0,
                })
            except Exception as e:
                logger.warning(f"debug_callback phase1_sparse 失败: {e}")

        # ================================================================
        # Phase 2: 融合
        # ================================================================
        if bm25_available and sparse_results:
            # 双路融合
            if self.config.fusion_method == "score_normalized":
                fused = _score_normalized_fusion(
                    dense_results, sparse_results,
                    dense_weight=self.config.fusion_dense_weight,
                )
            elif self.config.fusion_method == "linear_combination":
                fused = _score_normalized_fusion(
                    dense_results, sparse_results,
                    dense_weight=self.config.fusion_dense_weight,
                )
            else:
                # 默认 RRF（加权）
                fused = _reciprocal_rank_fusion(
                    dense_results, sparse_results,
                    k=self.config.fusion_k,
                    dense_weight=self.config.fusion_dense_weight,
                    sparse_weight=1.0 - self.config.fusion_dense_weight,
                )
        else:
            # CU-37: BM25 不可用 → 仅 Dense 结果
            fused = [(doc_id, score) for doc_id, score in dense_results]

        # ================================================================
        # P0: 研究文献软降权（融合后，跨 dense/sparse 统一）
        # ================================================================
        # 对 source_type == "research_literature" 的 chunk 分数乘以系数，
        # 让原文在混合候选池中优先浮出。系数由查询理解层按研究意图决定，
        # 本层只消费信号，不识别意图。
        if research_penalty < 1.0:
            fused = self._apply_research_penalty(
                fused, chunk_id_to_meta, research_penalty,
            )

        # ================================================================
        # P0: 版本软 boost（融合后，跨 dense/sparse 统一）
        # ================================================================
        # 对匹配版本约束的 chunk 融合分乘 >1 系数上浮，别版本不屏蔽不降。
        # 信号来自查询理解层 detect_version_filters，本层只消费不识别。
        if version_boost:
            fused = self._apply_version_boost(
                fused, chunk_id_to_meta, version_boost,
                self.config.version_boost_factor,
            )

        # ================================================================
        # P0: 上下文扩展（同章相邻 chunk 补召回）
        # ================================================================
        if self.config.context_window > 0:
            fused = self._apply_context_expansion(
                fused, self.config.context_window, self.config.context_decay,
            )

        if not fused:
            logger.info(f"融合后无结果，回退到纯关键词检索: {str(query)[:40]}")
            return self._keyword_only_search(query, top_k, filters=filters)

        # T00.07: Phase 2 Fusion 诊断回调
        if debug_callback:
            try:
                await debug_callback("phase2_fusion", {
                    "results": list(fused),
                    "method": self.config.fusion_method,
                    "total_candidates": len(fused),
                })
            except Exception as e:
                logger.warning(f"debug_callback phase2_fusion 失败: {e}")

        # ================================================================
        # Phase 3: Reranker 精排
        # ================================================================
        reranker_available = (
            self.config.reranker_enabled
            and self.reranker is not None
            and self.reranker.is_ready
        )
        if reranker_available:
            try:
                # 全部融合候选送入 Reranker，不做截断
                candidates = fused
                # 从 chunk_id_to_meta 提取实际文档文本，而非传入 chunk_id 字符串
                candidate_texts = [
                    chunk_id_to_meta.get(doc_id, {}).get("_content", "")
                    for doc_id, _ in candidates
                ]

                # 重排序输出数：取 reranker_output_k 和 top_k 的较大值，
                # 确保批量调用方（如 cooccurrence top_k=50）不被截断
                rerank_top_k = max(self.config.reranker_output_k, top_k)
                # B-160: rerank 是 cross-encoder 同步推理，同样丢线程池
                reranked = await asyncio.to_thread(
                    self.reranker.rerank,
                    _dense_q, candidate_texts, top_k=rerank_top_k,
                )

                # 构建最终结果
                final_results = []
                reranker_diag: list[tuple] = []  # T00.07: for phase3 callback
                for reranker_rank, (idx, rerank_score) in enumerate(reranked, start=1):
                    if idx < len(candidates):
                        chunk_id, fusion_score = candidates[idx]
                        fusion_rank = idx + 1  # 1-based
                        meta = chunk_id_to_meta.get(chunk_id, {})
                        content = meta.get("_content", "")
                        final_results.append({
                            "content": content,
                            "chunk_id": chunk_id,
                            "metadata": meta,
                            "score": fusion_score,
                            "relevance_score": rerank_score,
                        })
                        reranker_diag.append((chunk_id, rerank_score, fusion_rank, reranker_rank))

                # T00.07: Phase 3 Reranker 诊断回调
                if debug_callback:
                    try:
                        await debug_callback("phase3_reranker", {
                            "results": reranker_diag,
                            "output_k": rerank_top_k,
                        })
                    except Exception as e:
                        logger.warning(f"debug_callback phase3_reranker 失败: {e}")

                return final_results
            except (RuntimeError, ValueError) as e:
                logger.warning(f"Reranker 精排失败: {e}，使用融合结果")
                # CU-37: Reranker 不可用 → 直接返回融合候选集
                reranker_available = False

        # ================================================================
        # Phase 3: 按 RRF 融合分直接排序 — 不做语义重排序
        # ================================================================
        # 在此语料上 Dense 模型对文言文理解有限，语义重排序会使
        # BM25 召回的稀疏独有相关文档被 Dense 低分压到底部，
        # 导致 F ≈ D，完全浪费了关键词检索的召回优势。
        # 直接按 RRF 融合分排序，保留 BM25(0.7) + Dense(0.3) 的权重配比。

        results = []
        for chunk_id, fused_score in fused[:top_k]:
            meta = chunk_id_to_meta.get(chunk_id, {})
            results.append({
                "content": meta.get("_content", ""),
                "chunk_id": chunk_id,
                "metadata": meta,
                "score": fused_score,
                "relevance_score": fused_score,
            })
        return results

    # ========================================================================
    # CU-37: 逐级降级回退
    # ========================================================================
    async def _degraded_search(self, query: str, top_k: int = 5,
                                filters: Optional[dict] = None,
                                dense_query: str = "") -> list[dict]:
        """逐级降级回退。

        优先级:
        1. Reranker 不可用 → 融合结果前 top_k (已在 _three_stage_search 中处理)
        2. BM25 不可用 → 仅 Dense 粗排前 top_k
        3. BM25 + Dense 均不可用 → 纯关键词搜索 (CU-38: 最后一级)
        """
        from .embedder import Embedder
        embedder = self.vs.embedder

        _dense_q = dense_query if dense_query else query

        # 尝试纯语义检索
        if embedder.is_ready:
            try:
                query_vec = embedder.embed_query(_dense_q)
                semantic_raw = self.vs.query(
                    query_vec,
                    top_k=top_k,
                    where=filters if filters else None,
                )
                results = []
                for r in semantic_raw:
                    results.append({
                        "content": r.get("content", ""),
                        "chunk_id": r.get("chunk_id", ""),
                        "metadata": r.get("metadata", {}),
                        "score": r.get("score", 0.0),
                        "relevance_score": r.get("score", 0.0),
                    })
                if results:
                    logger.info("降级: 纯语义检索成功")
                    return results
            except (RuntimeError, ValueError) as e:
                logger.warning(f"降级语义检索失败: {e}")

        # CU-38: 最后一级回退 → 纯关键词搜索 (U11: 异步化)
        logger.info("降级: 使用纯关键词搜索 (最后一级回退)")
        return await asyncio.to_thread(
            self._keyword_only_search, _dense_q, top_k, filters
        )

    # ========================================================================
    # CU-38: _keyword_only_search 保留为最后一级回退 (U11: 移除同步阻塞)
    # ========================================================================
    def _keyword_only_search(self, query: str, top_k: int = 5,
                             filters: Optional[dict] = None) -> list[dict]:
        """纯关键词检索（BM25 和 Dense 都不可用时的最终回退）。

        U11: 移除同步 chunk_all() 调用，改为直接从 ChromaDB 已有索引中
        做关键词匹配，避免阻塞 ~120s。
        """
        logger.info("使用纯关键词检索 (final fallback)...")

        # U11: 直接从 ChromaDB 已有 collection 获取文档做关键词匹配
        if self.vs._collection is None:
            logger.warning("ChromaDB collection 不可用，_keyword_only_search 返回空")
            return []

        try:
            # 尝试通过 ChromaDB collection.get() 拉取文档
            all_data = self.vs._collection.get()
        except (RuntimeError, ValueError, OSError) as e:
            logger.warning(f"_keyword_only_search 读取 ChromaDB 失败: {e}")
            return []

        if not all_data or not all_data.get("documents"):
            return []

        scored = []
        for i, doc_content in enumerate(all_data["documents"]):
            if not doc_content:
                continue
            meta = all_data.get("metadatas", [{}])[i] if i < len(all_data.get("metadatas", [])) else {}
            chunk_id = all_data.get("ids", [""])[i] if i < len(all_data.get("ids", [])) else ""

            if filters:
                skip = False
                for fk, fv in filters.items():
                    if meta.get(fk) != fv:
                        skip = True
                        break
                if skip:
                    continue

            kw_score = self._keyword_match_score(query, doc_content)
            if kw_score > 0:
                scored.append({
                    "content": doc_content,
                    "chunk_id": chunk_id,
                    "metadata": meta,
                    "score": 0.0,
                    "relevance_score": kw_score,
                })

        scored.sort(key=lambda d: d["relevance_score"], reverse=True)
        return scored[:top_k]

    # ========================================================================
    # 辅助方法
    # ========================================================================
    def _tokenize(self, text: str) -> list[str]:
        """jieba 分词。"""
        try:
            import jieba
            return list(jieba.cut(text))
        except ImportError:
            tokens = []
            for i in range(len(text)):
                if text[i].strip():
                    tokens.append(text[i])
                if i < len(text) - 1:
                    bigram = text[i:i + 2]
                    if not any(c.isspace() for c in bigram):
                        tokens.append(bigram)
            return tokens

    def _keyword_match_score(self, query: str, doc_content: str) -> float:
        """计算关键词匹配得分（Jaccard-like）。"""
        query_tokens = set(self._tokenize(query))
        doc_tokens = set(self._tokenize(doc_content))
        if not query_tokens:
            return 0.0
        intersection = query_tokens & doc_tokens
        return len(intersection) / len(query_tokens)

    # ========================================================================
    # P0: 研究文献软降权
    # ========================================================================
    def _apply_research_penalty(
        self,
        fused: list[tuple[str, float]],
        chunk_id_to_meta: dict[str, dict],
        research_penalty: float,
    ) -> list[tuple[str, float]]:
        """对融合结果中的研究文献分数乘降权系数并重排。

        Args:
            fused: 融合后的 [(chunk_id, fused_score), ...]
            chunk_id_to_meta: chunk_id -> metadata 查找表
            research_penalty: 研究文献降权系数 (1.0 = 不降权)

        Returns:
            降权后按分数降序重新排序的列表。
            同一 source_type 内为均匀缩放，相对顺序不变；
            仅当候选池混合了 primary 与 research 时才改变跨类排序。
        """
        if research_penalty >= 1.0:
            return fused

        penalized: list[tuple[str, float]] = []
        for doc_id, score in fused:
            meta = chunk_id_to_meta.get(doc_id, {})
            if meta.get("source_type") == "research_literature":
                score = score * research_penalty
            penalized.append((doc_id, score))

        # 降权可能改变跨类排序，需重新排序
        penalized.sort(key=lambda x: x[1], reverse=True)
        return penalized

    # ========================================================================
    # P0: 版本软 boost
    # ========================================================================
    def _apply_version_boost(
        self,
        fused: list[tuple[str, float]],
        chunk_id_to_meta: dict[str, dict],
        version_boost: dict,
        boost_factor: float,
    ) -> list[tuple[str, float]]:
        """对融合结果中匹配版本约束的 chunk 分数乘 boost 系数并重排。

        只升匹配版本（× boost_factor > 1），别版本不降不屏蔽——与硬过滤
        （完全屏蔽）相反，即使 boost 判错，别版本仍在候选池中，误伤有限。

        Args:
            fused: 融合后的 [(chunk_id, fused_score), ...]
            chunk_id_to_meta: chunk_id -> metadata 查找表
            version_boost: 版本约束，如 {"genre": "opera_libretto"}
                或 {"genre": [..]}（OR）或 {"dynasty": "清"}
            boost_factor: 升权系数 (>1 升权；<=1 或空约束视为不 boost)

        Returns:
            boost 后按分数降序重新排序的列表。
        """
        if boost_factor <= 1.0 or not version_boost:
            return fused

        boosted: list[tuple[str, float]] = []
        for doc_id, score in fused:
            meta = chunk_id_to_meta.get(doc_id, {})
            if _meta_matches(meta, version_boost):
                score = score * boost_factor
            boosted.append((doc_id, score))

        # boost 可能改变跨版本排序，需重新排序
        boosted.sort(key=lambda x: x[1], reverse=True)
        return boosted

    # ========================================================================
    # P0: 上下文扩展
    # ========================================================================
    def _apply_context_expansion(
        self,
        fused: list[tuple[str, float]],
        window: int,
        decay: float,
    ) -> list[tuple[str, float]]:
        """把融合候选里每个 chunk 的同章相邻 chunk 补进候选池。

        针对「找不全」根因：评测发现相关 chunk 60%+ 是同章相邻段（连续 run），
        但检索只捞到 run 里的 1-2 个。这里给每个候选补上 seq ± window 的同章
        邻居（分数乘 decay^delta），让相邻段有机会进入 top-k。

        chunk_id 内嵌 (source_prefix, genre, chapter, seq)，seq 全局递增；同章
        相邻段的 chunk_id 前缀相同、seq 连续。构造邻居 id 后查存在性，天然只补
        同章邻居（跨章边界时构造出的 id 不存在，自动跳过）。
        """
        if window <= 0 or not fused:
            return fused

        from .chunker import parse_chunk_id

        all_ids: set[str] = set()
        if self.bm25 is not None and getattr(self.bm25, "_docs", None):
            for d in self.bm25._docs:
                cid = d.get("chunk_id", "")
                if cid:
                    all_ids.add(cid)
        if not all_ids:
            return fused

        expanded = list(fused)
        seen = {doc_id for doc_id, _ in fused}
        for doc_id, score in fused:
            try:
                info = parse_chunk_id(doc_id)
            except ValueError:
                continue
            prefix = f"{info['source_prefix']}_{info['genre']}_{info['chapter_number']}"
            for delta in range(1, window + 1):
                w = decay ** delta
                for sign in (-1, 1):
                    nb_id = f"{prefix}_{info['sequence'] + sign * delta:05d}"
                    if nb_id in all_ids and nb_id not in seen:
                        expanded.append((nb_id, score * w))
                        seen.add(nb_id)

        expanded.sort(key=lambda x: x[1], reverse=True)
        return expanded

    # ========================================================================
    # batch_retrieve
    # ========================================================================
    async def batch_retrieve(self, queries: list[str], top_k: int | None = None,
                             filters: Optional[dict] = None) -> list[list[dict]]:
        """批量并行检索。"""
        tasks = [self.retrieve(q, top_k=top_k, filters=filters) for q in queries]
        return await asyncio.gather(*tasks)

    # ========================================================================
    # B-121: 朝代维度统计查询
    # ========================================================================
    async def aggregate_by_dynasty(self, entity_name: str) -> dict[str, int]:
        """统计 entity 在每个朝代的【语义相关 chunk 分布】。

        ⚠️ 注意：这不是字面出现次数——返回的是语义检索召回的 chunk 数
        （受 top_k 上限影响），实体未被语料收录时也会因语义相近而得到
        非零数字。仅供"相关度分布"类展示（keyword_trend/sentiment_timeline）；
        需要真实频次请用 count_literal。
        """
        docs = await self.retrieve(entity_name, top_k=MAX_TOP_K)
        dynasty_counts: dict[str, int] = {}
        for d in docs:
            dynasty = d.get("metadata", {}).get("dynasty", "未知")
            dynasty_counts[dynasty] = dynasty_counts.get(dynasty, 0) + 1
        return dynasty_counts

    async def count_literal(self, entity_name: str) -> dict[str, dict]:
        """统计 entity 在全语料中的【字面出现次数】（真实频次）。

        遍历全库 chunk 对 content 做子串计数求和（chunk 切分无重叠，
        求和即全文真实次数），按朝代/章节两个维度分组。
        无 embedding/LLM 开销；实体未被语料收录时 total=0。
        """
        docs = self.vs.get_all_documents()
        # 语料库为空（索引重建中/集合损坏）时必须显式标记——否则会把
        # 「查不到」误报为「实体不存在」（2026-09-02 事故：空集合导致
        # 金山寺被报 0 次出现）
        corpus_available = len(docs) > 0
        by_dynasty: dict[str, int] = {}
        by_chapter: dict[str, int] = {}
        total = 0
        for d in docs:
            content = d.get("content", "") or ""
            n = content.count(entity_name)
            if n <= 0:
                continue
            total += n
            meta = d.get("metadata", {}) or {}
            dynasty = str(meta.get("dynasty") or "")
            by_dynasty[dynasty] = by_dynasty.get(dynasty, 0) + n
            chapter = str(meta.get("chapter_number") or "")
            by_chapter[chapter] = by_chapter.get(chapter, 0) + n
        return {"total": total, "by_dynasty": by_dynasty, "by_chapter": by_chapter,
                "corpus_available": corpus_available}

    async def find_appearances(self, entity_name: str,
                               context_chars: int = 60,
                               max_results: int = 200) -> dict:
        """定位 entity 在全语料中的所有字面出现位置（出场轨迹）。

        与 count_literal 同口径：遍历全库 chunk 做子串匹配，收集每处出现的
        章节/段落坐标 + 上下文摘录，供"XX在哪些章节哪些段落出场"类问题与
        前端跳转列表使用。超 max_results 截断并标记 truncated。

        B-163: chunk 的 paragraph_index 有 79% 为 -1（源文件单换行分段，
        \\n\\n 切不出段落）或与前端段落切分口径不一致 → 前端"跳转到原文"
        定位失败。这里用 ParagraphLocator 按 chunk 内容在源文件段落中
        重新定位，成功则覆盖；失败保留原值（前端仍有实体扫描兜底）。
        """
        docs = self.vs.get_all_documents()
        corpus_available = len(docs) > 0
        appearances: list[dict] = []
        total = 0
        truncated = False
        fixed_count = 0
        locator = get_paragraph_locator()
        for d in docs:
            content = d.get("content", "") or ""
            if entity_name not in content:
                continue
            meta = d.get("metadata", {}) or {}
            # B-163: 内容级段落重定位（locator 内部已处理无效索引与装饰前缀）
            located = locator.locate(meta.get("chapter_number", ""), content)
            start = 0
            while True:
                idx = content.find(entity_name, start)
                if idx == -1:
                    break
                total += 1
                if len(appearances) < max_results:
                    ctx_start = max(0, idx - context_chars)
                    ctx_end = min(len(content), idx + len(entity_name) + context_chars)
                    prefix = "…" if ctx_start > 0 else ""
                    suffix = "…" if ctx_end < len(content) else ""
                    orig_pidx = meta.get("paragraph_index", 0)
                    pidx = located if located is not None else orig_pidx
                    if located is not None and located != orig_pidx:
                        fixed_count += 1
                    appearances.append({
                        "chapter_number": chapter_number_to_int(
                            meta.get("chapter_number", "")),
                        "chapter_title": meta.get("chapter_title", ""),
                        "dynasty": str(meta.get("dynasty") or ""),
                        "paragraph_index": pidx,
                        "excerpt": (
                            f"{prefix}{content[ctx_start:ctx_end]}{suffix}"
                        ),
                    })
                else:
                    truncated = True
                start = idx + len(entity_name)
        if fixed_count:
            logger.info(
                f"[find_appearances] '{entity_name}' 段落号重定位: "
                f"{fixed_count} 处被修正（B-163 内容定位）"
            )
        return {"total": total, "truncated": truncated, "appearances": appearances,
                "corpus_available": corpus_available}

    # ========================================================================
    # B-120: search 方法别名
    # ========================================================================
    async def search(self, query: str, top_k: int | None = None,
                     filters: Optional[dict] = None) -> list[dict]:
        """带 metadata 过滤的搜索方法。"""
        return await self.retrieve(query, top_k=top_k, filters=filters)
