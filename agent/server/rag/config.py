"""RAG 管线统一配置，从环境变量加载所有 RAG 相关设置。"""
import os
import logging
from dataclasses import dataclass
from pathlib import Path
from dotenv import load_dotenv

# agent/ 目录，用于解析 data_dir / chroma_persist_dir 等相对路径
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
# .env 文件位于项目根目录（white-snake-museum-public/），而非 agent/ 下
_ENV_FILE = _PROJECT_ROOT.parent / ".env"

if _ENV_FILE.exists():
    load_dotenv(_ENV_FILE)


@dataclass
class RAGConfig:
    """RAG 管线统一配置，从环境变量加载。"""

    # ---- Embedding ----
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-zh-v1.5")
    embedding_local_path: str = os.getenv("EMBEDDING_LOCAL_PATH", "")
    hf_endpoint: str = os.getenv("HF_ENDPOINT", "https://hf-mirror.com")
    sentence_transformers_home: str = os.getenv("SENTENCE_TRANSFORMERS_HOME", "./models_cache")

    # ---- 向量库 ----
    chroma_persist_dir: str = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")

    # ---- 体裁感知分块 (CU-17) ----
    genre_chunking_enabled: bool = os.getenv("GENRE_CHUNKING_ENABLED", "true").lower() == "true"
    chunk_debug_enabled: bool = os.getenv("CHUNK_DEBUG_ENABLED", "false").lower() == "true"
    chunk_debug_output_dir: str = os.getenv("CHUNK_DEBUG_OUTPUT_DIR", "./debug_output")

    # ---- BM25 稀疏检索 (CU-18) ----
    bm25_enabled: bool = os.getenv("BM25_ENABLED", "true").lower() == "true"
    bm25_k1: float = float(os.getenv("BM25_K1", "1.5"))
    bm25_b: float = float(os.getenv("BM25_B", "0.75"))
    bm25_tokenizer: str = os.getenv("BM25_TOKENIZER", "jieba")
    bm25_top_k_multiplier: int = int(os.getenv("BM25_TOP_K_MULTIPLIER", "5"))

    # ---- Reranker 精排 (CU-19) ----
    reranker_model: str = os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L6-v2")
    reranker_backend: str = os.getenv("RERANKER_BACKEND", "onnx")  # "onnx" | "torch"
    # reranker 默认关闭：它与 embedder 合计 ~2.4GB 内存。在 16GB、日常已被
    # 其他应用占用 ~10GB 的机器上，双模型并发/叠加加载会撞穿 commit 上限，
    # 导致 torch 原生层崩溃（进程无声消失）。内存充裕可设 RERANKER_ENABLED=true。
    reranker_enabled: bool = os.getenv("RERANKER_ENABLED", "false").lower() == "true"
    reranker_top_k_multiplier: int = int(os.getenv("RERANKER_TOP_K_MULTIPLIER", "3"))

    # ---- 融合配置 (CU-20) ----
    # score_normalized: Min-max归一化后线性加权，保留连续分数差异
    # rrf: 基于排名的加权倒数融合（丢失分数分辨率，仅保留排序）
    fusion_method: str = os.getenv("FUSION_METHOD", "score_normalized")
    fusion_k: int = int(os.getenv("FUSION_K", "20"))
    # 语义检索权重: 0.5 = 语义/关键词均衡。权重扫描结论（4 数据集增强版 qrels，
    # dense_weight 0.0~1.0 步长 0.1）：最优权重落在 0.4~0.6，0.5 是全局更稳选择
    fusion_dense_weight: float = float(os.getenv("FUSION_DENSE_WEIGHT", "0.5"))

    # ---- 候选池与重排序 (U23) ----
    # 语料仅427个chunk，扩大候选池至100对性能几乎无影响(<5ms)，
    # 但能显著提升融合阶段的召回覆盖率
    dense_candidate_k: int = int(os.getenv("DENSE_CANDIDATE_K", "100"))
    sparse_candidate_k: int = int(os.getenv("SPARSE_CANDIDATE_K", "100"))
    reranker_output_k: int = int(os.getenv("RERANKER_OUTPUT_K", "10"))

    # ---- 默认检索 top_k (U24) ----
    # 生产中大部分调用方使用 top_k=10，少数批量/统计场景使用 50
    default_top_k: int = int(os.getenv("DEFAULT_TOP_K", "10"))
    # search_paragraphs 用户可见的搜索上限
    search_top_k: int = int(os.getenv("SEARCH_TOP_K", "20"))
    # cooccurrence / count_occurrences 批量统计检索数
    batch_top_k: int = int(os.getenv("BATCH_TOP_K", "50"))

    # ---- 研究文献降权 (P0) ----
    # 研究文献在混合候选池中的降权系数（0~1）。研究意图查询豁免（等效 1.0）。
    # 由 query_intent.research_penalty_for() 读取。
    research_penalty: float = float(os.getenv("RESEARCH_PENALTY", "0.5"))

    # ---- 版本软 boost (P0) ----
    # 版本约束 chunk 的融合分乘系数（>1 升权，不屏蔽别版本）。仅当查询明确
    # 指向单一版本时由 detect_version_filters 输出 version_boost 信号才生效。
    # 1.0 = 不 boost。扫参结论（4 数据集全量）：academic 甜点 x1.2~1.5 收益
    # 几乎相同，daily 单调负相关（x1.2 持平、x1.5 微负、x2.0 明显负），
    # 故默认温和值 1.2，保留 academic 收益且 daily 零伤害。
    version_boost_factor: float = float(os.getenv("VERSION_BOOST_FACTOR", "1.2"))

    # ---- 上下文扩展 (P0) ----
    # 把同章相邻 chunk（seq ± window）补进融合候选，0 = 关闭。
    # 针对「找不全」根因：相关 chunk 60%+ 是同章相邻段（连续 run）。
    context_window: int = int(os.getenv("CONTEXT_WINDOW", "0"))
    context_decay: float = float(os.getenv("CONTEXT_DECAY", "0.9"))

    # ---- 数据目录 ----
    data_dir: str = os.getenv("DATA_DIR", "../excel_data")

    @property
    def fusion_method_validated(self) -> str:
        """校验 fusion_method 值，非法值回退到 rrf。"""
        valid = {"rrf", "score_normalized", "linear_combination"}
        if self.fusion_method not in valid:
            logging.getLogger("agent").warning(
                f"Invalid fusion_method '{self.fusion_method}', falling back to 'rrf'"
            )
            return "rrf"
        return self.fusion_method

    @property
    def data_dir_path(self) -> Path:
        """data_dir 的绝对路径。"""
        p = Path(self.data_dir)
        if not p.is_absolute():
            p = _PROJECT_ROOT / p
        return p.resolve()

    @property
    def chroma_persist_path(self) -> Path:
        """chroma_persist_dir 的绝对路径。"""
        p = Path(self.chroma_persist_dir)
        if not p.is_absolute():
            p = _PROJECT_ROOT / p
        return p.resolve()


# 模块级单例
rag_config = RAGConfig()
