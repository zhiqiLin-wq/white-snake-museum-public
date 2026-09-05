"""两阶段混合检索管道。

Stage 1: FTS5 全文检索候选召回 (top_k=30)
Stage 2: 向量语义精排 (top_k=5)
Stage 3: 分数过滤 (score >= 0.5)

检索目标:
- long_term_memories 集合（长期记忆）
- recall_messages 集合（被淘汰的历史消息）

结果以统一格式返回，支持被动注入和 MCP 工具两种消费方式。
"""

import logging
from typing import Optional

from .memory_db import MemoryDB
from .fts_index import FTSIndex
from .memory_scorer import MemoryScorer

logger = logging.getLogger(__name__)

# 默认参数
DEFAULT_FTS_TOP_K = 50
DEFAULT_VECTOR_TOP_K = 10
DEFAULT_SCORE_THRESHOLD = 0.5
DEFAULT_INJECTION_TOKEN_BUDGET = 12000  # 8K-16K tokens，取中间

# 字面检索同义词扩展：查询命中某组词时，把该组其余词拼进查询，补漏同义表达。
# 仅用于 BM25/FTS 字面候选召回，不影响向量检索。
SYNONYM_GROUPS = [
    ["学者", "研究者", "专家"],
    ["研究", "考据", "探讨", "梳理", "探究"],
    ["喜欢", "中意", "偏好", "钟爱", "喜爱"],
    ["关注", "重视", "留意", "在意"],
    ["人物", "角色", "形象"],
    ["版本", "文本", "传本", "本子"],
    ["演变", "变迁", "流变", "变化"],
    ["认为", "觉得", "主张", "指出"],
    ["分析", "解读", "阐释", "剖析"],
    ["地点", "地方", "场所", "位置"],
    ["母题", "主题", "题材"],
    ["最早", "最初", "源头", "雏形"],
    ["情节", "桥段", "故事线"],
]


def _expand_synonyms(query: str) -> str:
    """查询同义词扩展：命中某组的词时，把该组其余词拼进查询。"""
    parts = [query]
    for group in SYNONYM_GROUPS:
        if any(w in query for w in group):
            for w in group:
                if w not in query:
                    parts.append(w)
    return " ".join(parts)


class HybridRetriever:
    """FTS5 → 向量精排 两阶段混合检索器。"""

    def __init__(self, memory_db: MemoryDB, embedder, chroma_client,
                 fts_index: Optional[FTSIndex] = None, user_id: str = "",
                 bm25_retriever=None, reranker=None, fusion_weight: float = 0.5):
        """
        Args:
            memory_db: MemoryDB 实例
            embedder: Embedder 实例（生成 query embedding）
            chroma_client: ChromaDB PersistentClient 实例
            fts_index: FTSIndex 实例（可选，默认新建）
            user_id: 用户 ID。非空时 ChromaDB 集合名加用户后缀，实现物理隔离。
            bm25_retriever: BM25Retriever 实例（可选，替代 FTS5 做字面检索）
            reranker: Reranker 实例（可选，融合后交叉编码器精排）
        """
        self._db = memory_db
        self._embedder = embedder
        self._chroma = chroma_client
        self._fts = fts_index or FTSIndex(memory_db)
        self._scorer = MemoryScorer(memory_db)
        self._bm25 = bm25_retriever
        self._reranker = reranker
        self._fusion_weight = fusion_weight
        suffix = f"_{user_id}" if user_id else ""
        self._recall_collection_name = f"recall_messages{suffix}"
        self._memory_collection_name = f"long_term_memories{suffix}"

        # 延迟加载 ChromaDB 集合引用
        self._recall_collection = None
        self._memory_collection = None
        self._init_collections()

    def _init_collections(self):
        """加载 ChromaDB 集合引用。"""
        try:
            self._recall_collection = self._chroma.get_collection(self._recall_collection_name)
        except Exception:
            self._recall_collection = None
        try:
            self._memory_collection = self._chroma.get_collection(self._memory_collection_name)
        except Exception:
            self._memory_collection = None

    def rebuild_bm25_index(self):
        """从 memory_db 读全部消息，重建 BM25 索引（含 source 元数据）。"""
        if self._bm25 is None:
            return
        msgs = self._db.get_all_messages()
        docs = [{"content": m["content"],
                 "metadata": {"message_id": m["message_id"], "source": m["source"]}}
                for m in msgs]
        self._bm25.rebuild(docs)

    def _candidate_search(self, query: str, top_k: int, source_filter: str) -> list[dict]:
        """字面候选召回：BM25 优先，未提供或未构建时回退 FTS5。

        字面检索前做同义词扩展，补漏同义表达（不影响向量检索）。
        统一返回 [{message_id, content, score}]，score 为字面相关分
        （BM25 原始分 / FTS rank），供融合阶段做 min-max 归一化。
        """
        if self._bm25 is not None and getattr(self._bm25, "_built", False):
            expanded = _expand_synonyms(query)
            results = self._bm25.search(expanded, top_k=top_k,
                                        metadata_filters={"source": source_filter})
            return [{"message_id": r["metadata"]["message_id"],
                     "content": r["content"],
                     "score": r.get("bm25_score", 0.0)}
                    for r in results]
        fts_results = self._fts.search(query, top_k=top_k, source_filter=source_filter)
        return [{"message_id": r.get("message_id", ""),
                 "content": r.get("content", ""),
                 "score": r.get("fts_rank", 0.0)}
                for r in fts_results]

    def _rerank(self, query: str, docs: list[dict], top_k: int) -> list[dict]:
        """用交叉编码器对候选精排（未提供/未就绪时原样返回）。"""
        if self._reranker is None or not self._reranker.is_ready or not docs:
            return docs
        try:
            texts = [d.get("content", "") for d in docs]
            scored = self._reranker.rerank(query, texts, top_k=min(top_k, len(docs)))
            return [docs[i] for i, _ in scored]
        except Exception as e:
            logger.warning("hybrid_retriever rerank failed: %s", e)
            return docs

    # ================================================================
    #  核心检索
    # ================================================================

    def search(self, query: str,
               source: str = "all",
               fts_top_k: int = DEFAULT_FTS_TOP_K,
               vector_top_k: int = DEFAULT_VECTOR_TOP_K,
               score_threshold: float = DEFAULT_SCORE_THRESHOLD,
               token_budget: Optional[int] = None) -> dict:
        """执行两阶段混合检索。

        Args:
            query: 搜索查询
            source: 检索来源 ("all" | "memories" | "recall")
            fts_top_k: FTS5 候选召回数量
            vector_top_k: 向量精排后保留数量
            score_threshold: 最低分数阈值
            token_budget: 注入 token 预算上限（None=不限制）

        Returns:
            {
                "memories": [{message_id, content, score, memory_type, ...}, ...],
                "recalls": [{message_id, content, score, role, timestamp, ...}, ...],
                "below_threshold": bool,
                "total_tokens": int,
            }
        """
        memories = []
        recalls = []

        if source in ("all", "memories"):
            memories = self._search_memories(query, fts_top_k, vector_top_k)

        if source in ("all", "recall"):
            recalls = self._search_recalls(query, fts_top_k, vector_top_k)

        # 分数过滤
        memories = [m for m in memories if m.get("score", 0) >= score_threshold]
        recalls = [r for r in recalls if r.get("score", 0) >= score_threshold]

        # Token 预算控制
        if token_budget is not None:
            memories, recalls = self._apply_token_budget(
                memories, recalls, token_budget,
            )

        all_below = (len(memories) + len(recalls)) == 0

        # 更新访问计数（非阻塞，best effort）
        for m in memories:
            mid = m.get("message_id")
            if mid:
                try:
                    self._db.update_memory_access(mid)
                except Exception:
                    pass

        total_tokens = self._estimate_tokens(memories, recalls)

        logger.info(
            "hybrid_retriever: query='%s' -> %d memories + %d recalls "
            "(fts_k=%d, vec_k=%d, threshold=%.2f, tokens=%d)",
            query[:60], len(memories), len(recalls),
            fts_top_k, vector_top_k, score_threshold, total_tokens,
        )

        return {
            "memories": memories,
            "recalls": recalls,
            "below_threshold": all_below,
            "total_tokens": total_tokens,
        }

    # ================================================================
    #  内部: 两阶段检索
    # ================================================================

    def _search_memories(self, query: str, fts_k: int, vec_k: int) -> list[dict]:
        """在长期记忆中执行两阶段检索（并集融合）。"""
        # Stage 0: 检查是否有数据
        if self._memory_collection is None or self._memory_collection.count() == 0:
            return []

        # Stage 1: 字面候选召回（BM25 优先，回退 FTS5）
        fts_candidates = self._candidate_search(query, fts_k, "memory_extraction")

        if not fts_candidates:
            # FTS5 无结果，降级为纯向量检索
            return self._vector_only_search(query, vec_k, "memory")

        # Stage 2: 向量检索（扩大候选覆盖，避免相关记忆被 top-N 截断丢失）
        query_embedding = self._embedder.embed([query])[0]
        try:
            chroma_results = self._memory_collection.query(
                query_embeddings=[query_embedding],
                n_results=min(vec_k * 5, self._memory_collection.count()),
            )
            chroma_docs = self._format_chroma_results(chroma_results)
        except Exception as e:
            logger.warning("hybrid_retriever vector search failed: %s, using FTS results", e)
            # 降级：只用 FTS 结果
            return self._format_fts_only([c["content"] for c in fts_candidates], fts_candidates)

        # 并集融合：字面候选 ∪ 向量候选，两路分数 min-max 归一化后按
        # fusion_weight 加权。单边命中缺的一路归一化为 0（不用拍脑袋的 0.8 降权）。
        reranked = self._fuse_candidates(fts_candidates, chroma_docs)

        # 按 score 降序排列 + 交叉编码器精排 + 截断
        reranked.sort(key=lambda x: x.get("score", 0), reverse=True)
        reranked = self._rerank(query, reranked, vec_k)[:vec_k]

        # 从 memory_db 补充元数据
        return self._enrich_with_memory_meta(reranked)

    def _search_recalls(self, query: str, fts_k: int, vec_k: int) -> list[dict]:
        """在 Recall Storage 中执行两阶段检索（并集融合）。"""
        if self._recall_collection is None or self._recall_collection.count() == 0:
            return []

        fts_candidates = self._candidate_search(query, fts_k, "recall")
        if not fts_candidates:
            return self._vector_only_search(query, vec_k, "recall")

        query_embedding = self._embedder.embed([query])[0]

        try:
            chroma_results = self._recall_collection.query(
                query_embeddings=[query_embedding],
                n_results=min(vec_k * 5, self._recall_collection.count()),
            )
            chroma_docs = self._format_chroma_results(chroma_results)
        except Exception as e:
            logger.warning("hybrid_retriever recall vector search failed: %s", e)
            return self._format_fts_only(fts_candidates, fts_candidates)

        # 并集融合（同 memories：min-max 归一化 + fusion_weight 加权）
        reranked = self._fuse_candidates(fts_candidates, chroma_docs)

        reranked.sort(key=lambda x: x.get("score", 0), reverse=True)
        return self._rerank(query, reranked, vec_k)[:vec_k]

    def _fuse_candidates(self, fts_candidates: list[dict],
                         chroma_docs: list[dict]) -> list[dict]:
        """并集融合：字面候选 ∪ 向量候选，两路分数 min-max 归一化后按 fusion_weight 加权。

        单边命中（只进一路）时缺的一路归一化为 0，等价于「该路信号缺失」，
        取代旧的「单边 0.8 降权」（拍脑袋值）。返回融合后的候选列表（未排序）。
        """
        fts_by_id = {c["message_id"]: c for c in fts_candidates}
        vec_by_id = {d["message_id"]: d for d in chroma_docs}
        all_ids = list(dict.fromkeys(list(fts_by_id.keys()) + list(vec_by_id.keys())))

        fts_scores = [c.get("score", 0.0) for c in fts_candidates]
        vec_scores = [d.get("score", 0.0) for d in chroma_docs]
        fts_min, fts_max = min(fts_scores), max(fts_scores)
        vec_min, vec_max = min(vec_scores), max(vec_scores)

        def _norm(v, lo, hi):
            if v is None or hi <= lo:
                return 0.0
            return (v - lo) / (hi - lo)

        reranked = []
        for mid in all_ids:
            fts_c = fts_by_id.get(mid)
            vec_d = vec_by_id.get(mid)
            nf = _norm(fts_c.get("score") if fts_c else None, fts_min, fts_max)
            nv = _norm(vec_d.get("score") if vec_d else None, vec_min, vec_max)
            score = self._fusion_weight * nv + (1.0 - self._fusion_weight) * nf
            src = vec_d if vec_d else fts_c
            reranked.append({
                "message_id": mid,
                "content": src.get("content", ""),
                "score": score,
                "fts_rank": nf,
                "role": src.get("role", ""),
                "timestamp": src.get("timestamp", ""),
            })
        return reranked

    def _vector_only_search(self, query: str, top_k: int, source: str) -> list[dict]:
        """纯向量检索降级路径（FTS5 无结果时）。"""
        collection = self._memory_collection if source == "memory" else self._recall_collection
        if collection is None or collection.count() == 0:
            return []

        query_embedding = self._embedder.embed([query])[0]
        try:
            results = collection.query(
                query_embeddings=[query_embedding],
                n_results=min(top_k, collection.count()),
            )
            docs = self._format_chroma_results(results)
            for d in docs:
                d["fts_rank"] = 0.0
                d["score"] = d.get("score", 0) * 0.8
            return docs
        except Exception as e:
            logger.warning("hybrid_retriever vector-only search failed: %s", e)
            return []

    def _format_fts_only(self, fts_candidates: list[dict],
                          original_candidates: list[dict]) -> list[dict]:
        """仅用 FTS 结果格式化（向量检索失败时的降级路径）。"""
        results = []
        for i, c in enumerate(original_candidates[:5]):
            score = max(0.1, 1.0 - i / 5 * 0.8)
            results.append({
                "message_id": c["message_id"],
                "content": c["content"],
                "score": score,
                "fts_rank": 1.0 - i / len(original_candidates),
                "role": c.get("role", ""),
                "timestamp": c.get("timestamp", ""),
            })
        return results

    # ================================================================
    #  辅助方法
    # ================================================================

    def _format_chroma_results(self, results) -> list[dict]:
        """将 ChromaDB 原始查询结果转为统一格式。"""
        docs = []
        if not results or not results.get("documents") or not results["documents"][0]:
            return docs
        for i, content in enumerate(results["documents"][0]):
            meta = results.get("metadatas", [[{}]])[0][i] if results.get("metadatas") else {}
            distance = results.get("distances", [[1.0]])[0][i] if results.get("distances") else 1.0
            doc_id = results.get("ids", [[""]])[0][i] if results.get("ids") else ""
            docs.append({
                "message_id": meta.get("message_id", doc_id),
                "content": content,
                "score": 1.0 - distance,
                "metadata": meta,
                "role": meta.get("role", ""),
                "timestamp": meta.get("timestamp", ""),
            })
        return docs

    def _enrich_with_memory_meta(self, docs: list[dict]) -> list[dict]:
        """从 SQLite 补充长期记忆元数据，并用 MemoryScorer 计算综合评分。

        综合评分 = 0.5*relevance + 0.3*importance + 0.2*recency，
        把记忆的绝对重要性与时间衰减纳入检索排序（P1-1）。
        relevance 即当前 vector + FTS 融合分。
        """
        for doc in docs:
            mid = doc.get("message_id", "")
            memory = self._db.get_long_term_memory(mid)
            if memory:
                doc["memory_type"] = memory.get("memory_type", "Entity")
                doc["importance"] = memory.get("importance", 0.5)
                doc["confidence"] = memory.get("confidence", 0.5)
                doc["created_at"] = memory.get("created_at")
                doc["last_accessed_at"] = memory.get("last_accessed_at")
            else:
                doc["memory_type"] = "Entity"
                doc["importance"] = 0.5
                doc["confidence"] = 0.5
                doc["created_at"] = None
                doc["last_accessed_at"] = None

            doc["score"] = self._scorer.compute_score(
                relevance_raw=float(doc.get("score", 0) or 0),
                importance_raw=float(doc.get("importance", 0.5) or 0.5),
                memory_type=doc.get("memory_type", "Entity"),
                created_at=doc.get("created_at"),
                last_accessed_at=doc.get("last_accessed_at"),
            )
        return docs

    def _apply_token_budget(self, memories: list[dict], recalls: list[dict],
                             token_budget: int) -> tuple[list[dict], list[dict]]:
        """Token 预算控制：按 score 降序逐条纳入，超预算的单条跳过（不阻断后续）。"""
        # 粗略估算: 中文约 1 char = 0.5 token, 英文约 1 char = 0.25 token
        def estimate(content: str) -> int:
            chars = len(content)
            chinese_chars = sum(1 for c in content if '一' <= c <= '鿿')
            other_chars = chars - chinese_chars
            return int(chinese_chars * 0.5 + other_chars * 0.25) + 10

        used = 0
        selected_memories = []
        for m in memories:
            cost = estimate(str(m.get("content", "")))
            if used + cost <= token_budget:
                selected_memories.append(m)
                used += cost
            else:
                continue

        selected_recalls = []
        for r in recalls:
            cost = estimate(str(r.get("content", "")))
            if used + cost <= token_budget:
                selected_recalls.append(r)
                used += cost
            else:
                continue

        return selected_memories, selected_recalls

    def _estimate_tokens(self, memories: list[dict], recalls: list[dict]) -> int:
        """估算结果的总 token 数（中文约 0.5 token/字 + 每条 10 token 开销）。"""
        total_chars = sum(len(str(m.get("content", ""))) for m in memories)
        total_chars += sum(len(str(r.get("content", ""))) for r in recalls)
        return int(total_chars * 0.5) + len(memories) * 10 + len(recalls) * 10

    # ================================================================
    #  格式化输出（注入用自然语言）
    # ================================================================

    def format_for_injection(self, result: dict) -> str:
        """将检索结果格式化为注入上下文的自然语言文本。

        Returns:
            格式化后的文本，如果无结果返回空字符串。
        """
        memories = result.get("memories", [])
        recalls = result.get("recalls", [])

        if not memories and not recalls:
            return ""

        parts = []
        parts.append(
            "[以下是与当前对话相关的历史记忆，请自然地利用这些信息回答，"
            "不要提及'根据记忆'或类似表述]\n"
        )

        if memories:
            parts.append("=== 相关长期记忆 ===")
            for i, m in enumerate(memories, 1):
                score = m.get("score", 0)
                mem_type = m.get("memory_type", "")
                type_hint = f"[{mem_type}]" if mem_type else ""
                parts.append(f"{i}. {type_hint} {m['content']} (相关度: {score:.2f})")

        if recalls:
            parts.append("\n=== 相关历史消息 ===")
            for i, r in enumerate(recalls, 1):
                role = r.get("role", "?")
                ts = r.get("timestamp", "")[:19]
                score = r.get("score", 0)
                content = str(r.get("content", ""))[:300]
                parts.append(f"{i}. [{role}] ({ts}) {content} (相关度: {score:.2f})")

        return "\n".join(parts)

    def format_for_mcp_tool(self, result: dict) -> str:
        """将检索结果格式化为 MCP 工具返回文本。"""
        memories = result.get("memories", [])
        recalls = result.get("recalls", [])

        parts = []
        parts.append("[内部参考，不要直接对用户说'根据检索到的记忆']\n")

        if memories:
            parts.append("=== 长期记忆 ===")
            for i, m in enumerate(memories, 1):
                score = m.get("score", 0)
                parts.append(f"{i}. [相似度 {score:.2f}] {m['content']}")
        else:
            parts.append("=== 长期记忆 ===\n(无相关记忆)")

        parts.append("")

        if recalls:
            parts.append("=== 历史消息 ===")
            for i, r in enumerate(recalls, 1):
                role = r.get("role", "?")
                ts = r.get("timestamp", "")[:19]
                score = r.get("score", 0)
                parts.append(
                    f"{i}. [{role}] ({ts}) [相似度 {score:.2f}] "
                    f"{str(r.get('content', ''))[:300]}"
                )
        else:
            parts.append("=== 历史消息 ===\n(无相关历史消息)")

        return "\n".join(parts)

    @property
    def is_healthy(self) -> bool:
        """检查检索器是否健康。"""
        try:
            self._fts.document_count
            return True
        except Exception:
            return False
