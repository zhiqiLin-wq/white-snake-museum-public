"""标签增强检索器：五路召回 -> 候选池 -> boost 重排 -> top-k。

严格按 rag_eval/RETRIEVAL_RULES.md。
"""
import logging

import numpy as np

from .boost import compute_boost, normalize_qtype, GENRE_FACTOR, HARD_OFFSET, PLOT_UNIT_SIM_THRESHOLD
from .boost import summary_jaccard
from .tag_store import IDF_THRESHOLD

logger = logging.getLogger(__name__)

TOP_K_CANDIDATE = 100
PLOT_UNIT_TOP_K = 50
RESEARCH_TOP_K = 50
DENSE_WEIGHT = 0.4
SPARSE_WEIGHT = 0.4
TAG_WEIGHT = 0.2


def _minmax(d):
    if not d:
        return {}
    lo, hi = min(d.values()), max(d.values())
    if hi == lo:
        return {k: 0.5 for k in d}
    return {k: (v - lo) / (hi - lo) for k, v in d.items()}


def _cosine(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-9
    return float(a @ b / denom)


def build_query_tags(query_label):
    """全标签（sparse 标签增强用，BM25 自带 IDF）。"""
    tags = []
    for field in ["persons", "locations", "plot_unit", "分析主题"]:
        for x in (query_label.get(field) or []):
            if isinstance(x, dict) and x.get("canonical"):
                tags.append(x["canonical"])
    pd = query_label.get("plot_detail", "")
    if pd:
        tags.append(pd)
    summary = query_label.get("summary", "")
    if summary:
        tags.append(summary)
    return [t for t in tags if t and t != "待删除"]


def build_query_tags_disc(query_label, idf, threshold=IDF_THRESHOLD):
    """判别性标签（过滤高频 persons/locations，标签语义用）。"""
    tags = []
    for field in ["plot_unit", "分析主题"]:
        for x in (query_label.get(field) or []):
            if isinstance(x, dict) and x.get("canonical"):
                tags.append(x["canonical"])
    for field in ["persons", "locations"]:
        for x in (query_label.get(field) or []):
            if isinstance(x, dict) and x.get("canonical") and idf(x["canonical"]) >= threshold:
                tags.append(x["canonical"])
    pd = query_label.get("plot_detail", "")
    if pd:
        tags.append(pd)
    summary = query_label.get("summary", "")
    if summary:
        tags.append(summary)
    return [t for t in tags if t and t != "待删除"]


class TagRetriever:
    """五路召回 + boost 重排的检索器。"""

    def __init__(self, vector_store, embedder, tag_store, bm25_retriever=None):
        self.vs = vector_store
        self.embedder = embedder
        self.tag_store = tag_store
        self.bm25 = bm25_retriever          # 原始 BM25（原文索引，可选）
        self._tag_bm25 = None               # 标签增强 BM25（全 chunk）
        self._research_bm25 = None          # 研究文献专用 BM25
        self._aug_matrix = None             # 标签增强 dense 矩阵（N x dim）
        self._aug_cids = None               # 标签增强 dense 的 chunk_id 顺序

    def build_tag_bm25(self, corpus):
        """构建标签增强 BM25 索引（原文 + 标签词，全 chunk）。"""
        from .bm25_retriever import BM25Retriever
        docs = []
        for cid, doc in corpus.items():
            original = doc.get("text", "")
            docs.append({
                "chunk_id": cid,
                "content": self.tag_store.aug_text(cid, original),
                "metadata": doc.get("metadata", {}),
            })
        self._tag_bm25 = BM25Retriever()
        self._tag_bm25.index(docs)

    def build_research_bm25(self, corpus):
        """构建研究文献专用 BM25 索引（只含 research_literature chunk）。"""
        from .bm25_retriever import BM25Retriever
        docs = []
        for cid, doc in corpus.items():
            if doc.get("metadata", {}).get("genre") != "research_literature":
                continue
            original = doc.get("text", "")
            docs.append({
                "chunk_id": cid,
                "content": self.tag_store.aug_text(cid, original),
                "metadata": doc.get("metadata", {}),
            })
        self._research_bm25 = BM25Retriever()
        self._research_bm25.index(docs)

    def build_tag_aug_dense(self, corpus):
        """构建标签增强 dense 矩阵（标签词拼原文开头），带缓存。

        复用 rag_eval 的 tag_aug_dense 向量缓存；缓存只覆盖旧语料时，
        为新增 chunk 实时补算并回写缓存；embedder 不可用则降级为原始向量检索
        （_aug_matrix=None，检索时走 vs.query），绝不让缺失 id 触发 KeyError。
        """
        from .tag_store import _load_vecs, _save_vecs
        self._aug_cids = list(corpus.keys())
        cached = _load_vecs("tag_aug_dense") or {}
        missing = [cid for cid in self._aug_cids if cid not in cached]
        if missing:
            logger.info(f"tag_aug_dense 缓存缺失 {len(missing)} 个新块，实时补算...")
            if not self.embedder:
                logger.warning("embedder 不可用，标签增强 dense 降级为原始向量检索")
                self._aug_matrix = None
                return
            try:
                # 等待后台加载的模型就绪（最多 180s），避免启动竞态下直接降级
                import time
                waited = 0
                while not self.embedder.is_ready and waited < 180:
                    time.sleep(3)
                    waited += 3
                if not self.embedder.is_ready:
                    logger.warning("embedder 180s 内未就绪，标签增强 dense 降级为原始向量检索")
                    self._aug_matrix = None
                    return
                aug_texts = [self.tag_store.build_tag_augmented_text(cid, corpus[cid]["text"])
                             for cid in missing]
                new_vecs = []
                # 小批次补算：低空闲内存机器上大批次会触发原生崩溃（0xC0000005）
                for i in range(0, len(aug_texts), 4):
                    new_vecs.extend(self.embedder.embed(aug_texts[i:i + 4]))
                for cid, v in zip(missing, new_vecs):
                    cached[cid] = v
                _save_vecs("tag_aug_dense", cached)
                logger.info(f"tag_aug_dense 新块向量补算完成并回写缓存（共 {len(cached)} 条）")
            except Exception as e:
                logger.warning(f"tag_aug_dense 新块补算失败，降级为原始向量检索: {e}")
                self._aug_matrix = None
                return
        self._aug_matrix = np.asarray([cached[cid] for cid in self._aug_cids], dtype=float)

    def build_tag_vectors(self):
        self.tag_store.build_tag_vectors()

    def _candidate_scores(self, query_text, query_label, meta_lookup):
        """五路召回，返回 {chunk_id: {dense_score, bm25_score, tag_score}}。"""
        qtype = normalize_qtype(query_label.get("query_type", ""))
        qtags = build_query_tags(query_label)
        qtags_disc = build_query_tags_disc(query_label, self.tag_store.idf)

        # 1. dense（标签增强：标签词拼 chunk 开头，优先；否则原始向量兜底）
        dense_scores = {}
        if self._aug_matrix is not None:
            qvec = np.asarray(self.embedder.embed_query(query_text), dtype=float)
            ds = self._aug_matrix @ qvec
            order = np.argsort(ds)[::-1][:TOP_K_CANDIDATE]
            dense_scores = {self._aug_cids[i]: float(ds[i]) for i in order}
        else:
            qvec = self.embedder.embed_query(query_text)
            dense_raw = self.vs.query(qvec, top_k=TOP_K_CANDIDATE)
            for r in dense_raw:
                cid = r.get("chunk_id")
                if cid:
                    dense_scores[cid] = r["score"]

        # 2. sparse 标签增强
        bm25_scores = {}
        if self._tag_bm25 is not None:
            aug_query = (query_text + " " + " ".join(qtags)).strip()
            for r in self._tag_bm25.search(aug_query, top_k=TOP_K_CANDIDATE):
                cid = r.get("chunk_id")
                if cid:
                    bm25_scores[cid] = r.get("bm25_score", 0.0)

        # 3. 标签语义（判别性标签，去高频）——仅参与当前语料中仍存在的 chunk，
        # 旧标签缓存里的失效 id 不进入候选（避免召回已不存在的块）
        tag_scores = {}
        tag_cids = self.tag_store.chunk_ids
        if meta_lookup:
            tag_cids = [c for c in tag_cids if c in meta_lookup]
        tag_cids = [c for c in tag_cids if self.tag_store.tag_vector(c) is not None]
        if tag_cids:
            q_tag_text = " ".join(qtags_disc)
            q_tag_vec = np.asarray(self.embedder.embed([q_tag_text])[0], dtype=float)
            mat = np.asarray([self.tag_store.tag_vector(c) for c in tag_cids], dtype=float)
            scores = mat @ q_tag_vec
            for cid, s in zip(tag_cids, scores):
                tag_scores[cid] = float(s)

        # 4. plot_unit 精确检索路
        plot_ids = []
        plot_units = [x["canonical"] for x in (query_label.get("plot_unit") or [])
                      if isinstance(x, dict) and x.get("canonical")]
        if plot_units and self._tag_bm25 is not None:
            pq = " ".join(plot_units)
            plot_ids = [r["chunk_id"] for r in self._tag_bm25.search(pq, top_k=PLOT_UNIT_TOP_K)
                        if r.get("chunk_id")]

        # 5. 研究文献专用召回
        research_ids = []
        if qtype in ("comparison题", "研究分析题") and self._research_bm25 is not None:
            rq_tags = [x["canonical"] for x in (query_label.get("分析主题") or [])
                       if isinstance(x, dict) and x.get("canonical")]
            rq_tags += [x["canonical"] for x in (query_label.get("plot_unit") or [])
                        if isinstance(x, dict) and x.get("canonical")]
            if rq_tags:
                rq = " ".join(rq_tags)
                research_ids = [r["chunk_id"] for r in self._research_bm25.search(rq, top_k=RESEARCH_TOP_K)
                                if r.get("chunk_id")]

        candidates = (set(dense_scores.keys()) | set(bm25_scores.keys())
                      | set(tag_scores.keys()) | set(plot_ids) | set(research_ids))
        out = {}
        for cid in candidates:
            out[cid] = {
                "dense_score": dense_scores.get(cid),
                "bm25_score": bm25_scores.get(cid),
                "tag_score": tag_scores.get(cid),
            }
        return out

    async def retrieve(self, query_text, query_label, top_k=10, corpus=None, meta_lookup=None,
                       max_candidates=None, fusion_weights=None,
                       plot_unit_threshold=PLOT_UNIT_SIM_THRESHOLD, return_scores=False):
        """五路召回 + boost 重排，返回 top-k chunk_id 列表（或 (chunk_id, score) 对）。"""
        if not isinstance(query_label, dict) or "_error" in query_label:
            return []

        candidates = self._candidate_scores(query_text, query_label, meta_lookup)
        if not candidates:
            return []

        # summary 相似度（dense 用预计算向量 + sparse Jaccard）
        q_summary = query_label.get("summary", "") or ""
        summary_dense = {}
        summary_sparse = {}
        if q_summary:
            q_emb = np.asarray(self.embedder.embed([q_summary])[0], dtype=float)
            for cid in candidates:
                sv = self.tag_store.summary_vector(cid)
                if sv is not None:
                    summary_dense[cid] = _cosine(q_emb, np.asarray(sv, dtype=float))
                st = self.tag_store.summary_text(cid)
                if st:
                    summary_sparse[cid] = summary_jaccard(q_summary, st)

        # fused_base（候选集内 minmax，dense + sparse + 标签语义 三路融合）
        dn = _minmax({cid: c["dense_score"] for cid, c in candidates.items() if c["dense_score"] is not None})
        sn = _minmax({cid: c["bm25_score"] for cid, c in candidates.items() if c["bm25_score"] is not None})
        tn = _minmax({cid: c["tag_score"] for cid, c in candidates.items() if c["tag_score"] is not None})

        # 候选池阈值：按 fused_base 截断（若设置）
        dw, sw, tw = fusion_weights or (DENSE_WEIGHT, SPARSE_WEIGHT, TAG_WEIGHT)
        fused_pre = {cid: dw * dn.get(cid, 0.0) + sw * sn.get(cid, 0.0) + tw * tn.get(cid, 0.0)
                     for cid in candidates}
        if max_candidates and len(candidates) > max_candidates:
            keep = sorted(fused_pre.items(), key=lambda x: x[1], reverse=True)[:max_candidates]
            candidates = {cid: candidates[cid] for cid, _ in keep}

        # boost 重排
        q_genre = query_label.get("genre", "") or ""
        final = {}
        for cid, c in candidates.items():
            c_genre = ""
            if meta_lookup and cid in meta_lookup:
                c_genre = meta_lookup[cid].get("genre", "")
            nl = self.tag_store.norm_label(cid) or {}
            boost, hard_match = compute_boost(
                query_label, nl, c_genre,
                summary_dense.get(cid, 0.0), summary_sparse.get(cid, 0.0),
                self.tag_store.idf, self.tag_store.plot_unit_sim, plot_unit_threshold,
            )
            genre_factor = GENRE_FACTOR if (c_genre == q_genre and q_genre) else 1.0
            final[cid] = (fused_pre[cid] + boost) * genre_factor
            if hard_match:
                final[cid] += HARD_OFFSET

        ranked = sorted(final.items(), key=lambda x: x[1], reverse=True)[:top_k]
        if return_scores:
            return ranked
        return [cid for cid, _ in ranked]
