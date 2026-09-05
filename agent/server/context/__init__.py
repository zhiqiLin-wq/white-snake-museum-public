"""上下文管理与记忆系统。

模块:
- token_counter: tiktoken cl100k_base 封装
- memory_db: SQLite 记忆数据库（Source of Truth，8个表）
- fts_index: FTS5 全文索引接口
- recall_store: 双写 ChromaDB + SQLite（ChromaDB 仅作语义索引）
- memory_extractor: 长期记忆提取（规则过滤 + LLM 提取 + MD5+向量 双重去重）
- memory_scorer: 综合评分计算（Relevance + Importance + Recency）
- hybrid_retriever: FTS5 + 向量 两阶段混合检索
- context_manager: FIFO 上下文队列管理 + 递归摘要压缩 + SQLite 持久化
- background_tasks: 后台任务调度（软删除、合并、空闲提取、补偿重试）
"""

from .token_counter import count_tokens, count_message_tokens, count_messages_tokens
from .memory_db import MemoryDB
from .fts_index import FTSIndex
from .recall_store import RecallStore
from .memory_extractor import MemoryExtractor
from .memory_scorer import MemoryScorer
from .hybrid_retriever import HybridRetriever
from .context_manager import ContextManager, ContextState

__all__ = [
    "count_tokens",
    "count_message_tokens",
    "count_messages_tokens",
    "MemoryDB",
    "FTSIndex",
    "RecallStore",
    "MemoryExtractor",
    "MemoryScorer",
    "HybridRetriever",
    "ContextManager",
    "ContextState",
]
