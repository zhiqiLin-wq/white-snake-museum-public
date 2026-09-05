"""BM25 稀疏检索器 —— 倒排索引 + Okapi BM25 评分 + 三种分词策略。

CU-22~CU-28: BM25Retriever 完整实现。
"""
import logging
import math
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)

# CU-28: jieba 自定义词典 —— 古籍专有名词
# 从 entity_names 别名库自动抽取，避免手工维护两份数据
def _build_jieba_custom_words() -> list[str]:
    """从 entity_names 别名库抽取所有多字词，去重后用于 jieba 分词。

    覆盖所有角色/地点/事件/概念的别名，确保 BM25 分词时能正确切分出
    古典文献中的专有名词，而非按单字拆分。
    """
    try:
        from ..utils.entity_names import (
            CHARACTER_ALIASES, LOCATION_ALIASES, EVENT_ALIASES, CONCEPT_ALIASES,
        )
    except ImportError:
        # 回退: 核心词列表（当 entity_names 模块不可用时）
        return [
            '白素贞', '白娘子', '白氏', '白蛇', '白娘娘', '白云仙姑', '白衣娘子',
            '许宣', '许汉文', '许仙', '小乙官', '许郎', '许官人', '奚宣赞',
            '小青', '青青', '青蛇', '青儿', '卯奴',
            '法海', '法海禅师', '禅师', '海公',
            '许梦蛟', '梦蛟', '许士麟', '李公甫', '李碧莲',
            '雷峰塔', '雷锋塔', '黄妃塔', '王妃塔',
            '断桥', '段桥', '段家桥', '西湖', '金山寺',
            '保叔塔', '涌金门', '四圣观', '孤山路',
            '钱塘门', '清波门', '望江门', '武林门',
            '净慈寺', '灵隐寺', '昭庆寺', '承天寺',
            '临安', '临安府', '钱塘县', '杭州',
            '镇江', '苏州', '姑苏', '汴梁', '金陵',
            '青城山', '峨眉山', '龙虎山', '终南山', '昆仑山',
            '义妖传', '太平广记', '西湖三塔记',
            '方成培', '陈遇乾', '田汉', '玉山堂主人',
            '西江月', '鹧鸪天', '临江仙', '满江红', '蝶恋花',
            '回目', '楔子', '入话', '头回',
            '白蛇传', '白蛇故事', '白蛇传说', '白娘子传奇',
            '水漫金山', '盗仙草', '端阳惊变', '合钵镇妖',
        ]

    words: set[str] = set()
    for alias_dict in (CHARACTER_ALIASES, LOCATION_ALIASES, EVENT_ALIASES, CONCEPT_ALIASES):
        for aliases in alias_dict.values():
            for a in aliases:
                a = a.strip()
                # 只保留多字词（>=2字符），单字词对分词无帮助且会增加噪音
                if len(a) >= 2:
                    words.add(a)

    # 额外补充: 文献专属术语和结构标记（entity_names 中不包含）
    extra_terms = [
        # 文献名
        '义妖传', '太平广记', '西湖三塔记', '警世通言', '雷峰塔传奇',
        # 作者名
        '方成培', '陈遇乾', '田汉', '玉山堂主人', '冯梦龙',
        # 曲牌常见词
        '西江月', '鹧鸪天', '临江仙', '满江红', '蝶恋花',
        # 结构术语
        '回目', '楔子', '入话', '头回', '正话',
        # 古籍常见用语
        '话说', '且说', '却说', '正是', '诗曰', '词云', '赞曰',
        '下回', '未知', '毕竟', '端的',
    ]
    for t in extra_terms:
        words.add(t)

    # 按长度降序排列: jieba 优先匹配更长的词
    return sorted(words, key=len, reverse=True)


_JIEBA_CUSTOM_WORDS: list[str] = _build_jieba_custom_words()


@dataclass
class BM25Config:
    """BM25 参数配置 (CU-22)。"""
    k1: float = 1.5
    b: float = 0.75
    tokenizer: str = "jieba"
    top_k_multiplier: int = 5


class BM25Retriever:
    """BM25 稀疏检索器 (CU-22~CU-28)。"""

    def __init__(self, config: Optional[BM25Config] = None):
        if config is None:
            # 默认从 RAG 统一配置读取 tokenizer 等参数。
            # 避免 BM25Config 硬编码默认值 (jieba) 在评估等未显式传 config 的
            # 路径上把 BM25_TOKENIZER 环境变量变成死配置 (与 main.py 生产接线一致)。
            from .config import rag_config
            config = BM25Config(
                k1=rag_config.bm25_k1,
                b=rag_config.bm25_b,
                tokenizer=rag_config.bm25_tokenizer,
                top_k_multiplier=rag_config.bm25_top_k_multiplier,
            )
        self.config = config

        # 延迟加载 jieba
        self._jieba = None
        if self.config.tokenizer == "jieba":
            try:
                import jieba
                self._jieba = jieba
                logger.info("BM25: jieba 分词器已加载")
            except ImportError:
                logger.warning("BM25: jieba 未安装，回退到 bigram 分词")
                self.config.tokenizer = "bigram"

        # 内部数据结构 (CU-22)
        self._docs: list[dict] = []
        self._doc_metadata: list[dict] = []
        self._inverted_index: dict[str, list[tuple[int, int]]] = {}
        self._doc_lengths: list[int] = []
        self._avg_doc_length: float = 0.0
        self._doc_freq: dict[str, int] = {}
        self._total_docs: int = 0
        self._built: bool = False

    # ------------------------------------------------------------------
    # CU-23: 三种分词策略
    # ------------------------------------------------------------------
    def _tokenize(self, text: str) -> list[str]:
        """根据配置选择分词策略。"""
        if self.config.tokenizer == "jieba" and self._jieba is not None:
            return self._tokenize_jieba(text)
        elif self.config.tokenizer == "char":
            return self._tokenize_char(text)
        else:
            return self._tokenize_bigram(text)

    def _tokenize_jieba(self, text: str) -> list[str]:
        """jieba 精确模式分词，保留单字。"""
        tokens = list(self._jieba.lcut(text))
        return [t.strip() for t in tokens if t.strip()]

    @staticmethod
    def _tokenize_char(text: str) -> list[str]:
        """逐字分词，跳过空白和换行。"""
        return [ch for ch in text if ch not in (' ', '\n', '\r', '\t')]

    @staticmethod
    def _tokenize_bigram(text: str) -> list[str]:
        """字符级二元组分词。"""
        clean = text.replace(' ', '').replace('\n', '').replace('\r', '').replace('\t', '')
        if len(clean) < 2:
            return [clean] if clean else []
        return [clean[i:i + 2] for i in range(len(clean) - 1)]

    # ------------------------------------------------------------------
    # CU-24: 倒排索引构建
    # ------------------------------------------------------------------
    def index(self, documents: list[dict]):
        """构建 BM25 倒排索引。

        Args:
            documents: [{"content": str, "metadata": dict}, ...]
        """
        if not documents:
            logger.warning("BM25: 空文档列表，跳过索引构建")
            return

        # CU-28: 加载 jieba 自定义词典
        if self._jieba is not None:
            for word in _JIEBA_CUSTOM_WORDS:
                self._jieba.add_word(word)
            logger.info(f"BM25: 已加载 {len(_JIEBA_CUSTOM_WORDS)} 个自定义词到 jieba")

        self._docs = documents
        self._doc_metadata = [d.get("metadata", {}) for d in documents]
        self._inverted_index = {}
        self._doc_lengths = []
        self._doc_freq = {}

        total_length = 0

        for doc_id, doc in enumerate(documents):
            content = doc.get("content", "")
            tokens = self._tokenize(content)
            self._doc_lengths.append(len(tokens))
            total_length += len(tokens)

            # within-doc term frequency
            term_freq: dict[str, int] = {}
            for token in tokens:
                term_freq[token] = term_freq.get(token, 0) + 1

            # 更新倒排索引: token -> [(doc_id, freq), ...]
            for token, freq in term_freq.items():
                if token not in self._inverted_index:
                    self._inverted_index[token] = []
                self._inverted_index[token].append((doc_id, freq))

        # 计算 avg_doc_length 和 doc_freq
        self._total_docs = len(documents)
        self._avg_doc_length = total_length / max(self._total_docs, 1)
        for token, postings in self._inverted_index.items():
            self._doc_freq[token] = len(postings)

        self._built = True
        logger.info(
            f"BM25 索引构建完成: {self._total_docs} 文档, "
            f"{len(self._inverted_index)} 独立 token, "
            f"avg_doc_length={self._avg_doc_length:.1f}"
        )

    # ------------------------------------------------------------------
    # CU-25: Okapi BM25 评分 (支持词权重)
    # ------------------------------------------------------------------
    def _bm25_score(self, query_tokens: list[str], doc_id: int,
                    token_weights: dict[str, float] | None = None) -> float:
        """标准 Okapi BM25 公式，支持可选词权重。

        IDF = log(1 + (N - df + 0.5) / (df + 0.5))
        TF_norm = tf * (k1 + 1) / (tf + k1 * (1 - b + b * (doc_len / avg_doc_len)))
        score = sum(IDF * TF_norm * qtf * weight) over all query tokens

        Args:
            query_tokens: 分词后的查询 token 列表
            doc_id: 文档 ID
            token_weights: 可选 {token: weight}，未指定默认 weight=1.0
        """
        if doc_id >= self._total_docs:
            return 0.0

        k1 = self.config.k1
        b = self.config.b
        doc_len = self._doc_lengths[doc_id]
        avgdl = max(self._avg_doc_length, 1.0)
        N = self._total_docs

        score = 0.0
        # 对每个查询 token 计算贡献
        token_counts: dict[str, int] = {}
        for t in query_tokens:
            token_counts[t] = token_counts.get(t, 0) + 1

        for token, qtf in token_counts.items():
            df = self._doc_freq.get(token, 0)
            if df == 0:
                continue

            # IDF
            idf = math.log(1.0 + (N - df + 0.5) / (df + 0.5))

            # TF in document
            postings = self._inverted_index.get(token, [])
            tf = 0
            for pid, freq in postings:
                if pid == doc_id:
                    tf = freq
                    break
            if tf == 0:
                continue

            # TF normalization
            tf_norm = (tf * (k1 + 1.0)) / (tf + k1 * (1.0 - b + b * (doc_len / avgdl)))

            # 词权重: 原始查询词 = 1.0, 扩展别名 = 0.5 (未指定默认 1.0)
            weight = token_weights.get(token, 1.0) if token_weights else 1.0
            score += idf * tf_norm * qtf * weight

        return score

    # ------------------------------------------------------------------
    # CU-26: search() 检索方法
    # ------------------------------------------------------------------
    def search(self, query: str, top_k: int = 10,
               metadata_filters: Optional[dict] = None,
               query_weights: Optional[dict[str, float]] = None) -> list[dict]:
        """BM25 检索。

        Args:
            query: 查询文本
            top_k: 返回结果数
            metadata_filters: 如 {"genre": "classical_chinese"} 只返回指定体裁
            query_weights: 可选 {token: weight}，原查询词 weight=1.0，别名=0.5。
                           未指定时所有词权重均为 1.0（等价于标准 BM25）。

        Returns:
            [{"content": str, "metadata": dict, "bm25_score": float}, ...]
        """
        if not self._built:
            logger.warning("BM25: 索引未构建，返回空结果")
            return []

        query_tokens = self._tokenize(query)
        if not query_tokens:
            return []

        # 遍历全部文档计算 BM25 分数
        scored: list[tuple[int, float]] = []
        for doc_id in range(self._total_docs):
            # metadata_filters 过滤（单值匹配）
            if metadata_filters:
                meta = self._doc_metadata[doc_id]
                skip = False
                for fk, fv in metadata_filters.items():
                    if meta.get(fk) != fv:
                        skip = True
                        break
                if skip:
                    continue

            score = self._bm25_score(query_tokens, doc_id, query_weights)
            if score > 0:
                scored.append((doc_id, score))

        # 按分数降序排列
        scored.sort(key=lambda x: x[1], reverse=True)
        top_results = scored[:top_k]

        results = []
        for doc_id, bm25_score in top_results:
            results.append({
                "content": self._docs[doc_id]["content"],
                "metadata": self._doc_metadata[doc_id],
                "bm25_score": bm25_score,
                "chunk_id": self._docs[doc_id].get("chunk_id", ""),
            })

        return results

    # ------------------------------------------------------------------
    # CU-27: clear() 清空索引 + rebuild 生命周期
    # ------------------------------------------------------------------
    def clear(self):
        """清空所有内部数据结构，重置 _built = False。"""
        self._docs = []
        self._doc_metadata = []
        self._inverted_index = {}
        self._doc_lengths = []
        self._avg_doc_length = 0.0
        self._doc_freq = {}
        self._total_docs = 0
        self._built = False
        logger.info("BM25: 索引已清空")

    def rebuild(self, documents: list[dict]):
        """清空后重建索引，保证数据一致性 (CU-27)。"""
        self.clear()
        self.index(documents)
