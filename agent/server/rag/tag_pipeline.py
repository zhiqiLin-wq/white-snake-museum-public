"""生产问答检索管线：TagRetriever 五路召回 + boost + 在线 LLM 打标。

与 HybridRetriever 互补分工：
- TagPipeline：问答流程（literature_qa / agent_loop / comparison / location_story /
  evolution_workbench），LLM 打标 + 标签增强检索，覆盖 1339 条评测口径。
- HybridRetriever：工具类短查询（共现 / 计数 / 情感时间线等），无 LLM，支持 filters。

TagPipeline.retrieve 兼容 HybridRetriever.retrieve 的调用签名，但：
- 只接收 str 查询（问答入口均为 str），不支持 filters 硬过滤；传非 None 时 fast-fail 返回空。
- 每次检索做一次 LLM 打标（复用 provider 自带 LLM 缓存 + 本类 label 缓存）。
"""
import logging

logger = logging.getLogger(__name__)


class TagPipeline:
    """问答检索器：在线 LLM 打标 -> TagRetriever 五路召回 + boost -> dict 列表。"""

    # B-152: 本检索器不支持硬过滤（retrieve 遇 filters fast-fail 返回空）。
    # EvolutionWorkbench 据此走"过滤词并入查询 + 本地 metadata 过滤"的代理路径，
    # 避免 E2/E7 分朝代检索静默拿到 0 结果。
    supports_filters = False

    # B-155: 支持跳过在线 LLM 打标（演化分析的批量程序化检索专用——查询是
    # 拼接的关键词串而非自然语言，打标 ~15s/次 属纯浪费；问答流程不受影响）。
    supports_skip_tagging = True

    def __init__(self, tag_retriever, provider, corpus, meta_lookup):
        self.tag_retriever = tag_retriever
        self.provider = provider
        self.corpus = corpus
        self.meta_lookup = meta_lookup
        self._label_cache = {}

    async def retrieve(self, query, top_k=None, filters=None, version_boost=None,
                       debug_callback=None, skip_tagging=False):
        """问答检索入口（签名对齐 HybridRetriever.retrieve）。

        skip_tagging=True 时跳过 LLM 打标，用空标签直接走五路召回的核心三路
        （原始 dense + 原始 sparse；标签增强/情节单元/研究文献通道自然关闭）。
        """
        from .config import rag_config
        if filters:
            logger.warning("TagPipeline 不支持 filters 硬过滤，返回空（问答流程不应传 filters）")
            return []
        if top_k is None:
            top_k = rag_config.default_top_k

        label = {} if skip_tagging else await self._tag(query)
        if not isinstance(label, dict) or "_error" in label:
            return []

        ranked = await self.tag_retriever.retrieve(
            query, label, top_k=top_k,
            corpus=self.corpus, meta_lookup=self.meta_lookup,
            return_scores=True,
        )
        docs = []
        for cid, score in ranked:
            doc = self.corpus.get(cid, {})
            docs.append({
                "chunk_id": cid,
                "content": doc.get("text", ""),
                "metadata": dict(doc.get("metadata", {}) or {}),
                "score": float(score),
                "relevance_score": float(score),
            })
        return docs

    async def _tag(self, query):
        if query in self._label_cache:
            return self._label_cache[query]
        from .query_tagger import tag_query
        label = await tag_query(self.provider, query)
        self._label_cache[query] = label
        return label
