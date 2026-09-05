"""FTS5 全文索引接口封装。

提供对 messages_fts 虚拟表的查询接口，
支持中文文本的 BM25 排序检索。
Tokenization 使用 FTS5 内置的 unicode61（对中文自动 n-gram）。
"""

import logging
from typing import Optional

from .memory_db import MemoryDB

logger = logging.getLogger(__name__)


class FTSIndex:
    """FTS5 全文检索接口。

    不直接操作 SQL，通过 MemoryDB 完成。
    """

    def __init__(self, memory_db: MemoryDB):
        self._db = memory_db

    def search(self, query: str, top_k: int = 30,
               source_filter: str = None) -> list[dict]:
        """FTS5 BM25 全文检索。

        Args:
            query: 搜索查询文本
            top_k: 返回结果数
            source_filter: 可选的消息来源过滤

        Returns:
            [{message_id, content, role, thread_id, timestamp, fts_rank}, ...]
        """
        return self._db.search_fts(query, top_k, source_filter=source_filter)

    def rebuild(self):
        """重建 FTS5 索引（全量）。

        删除 messages_fts 全部数据并从 messages 表重新填充。
        中文内容按 bigram 化后写入，与 insert_message 保持一致。
        用于修复索引不一致。
        """
        import sqlite3
        conn = sqlite3.connect(str(self._db._db_path))
        try:
            conn.execute("DELETE FROM messages_fts")
            rows = conn.execute("SELECT rowid, content FROM messages").fetchall()
            for rowid, content in rows:
                conn.execute(
                    "INSERT INTO messages_fts (rowid, content) VALUES (?, ?)",
                    (rowid, self._db._to_fts_tokens(content)),
                )
            conn.commit()
            count = conn.execute("SELECT COUNT(*) FROM messages_fts").fetchone()[0]
            logger.info("FTS index rebuilt: %d documents indexed", count)
        finally:
            conn.close()

    @property
    def document_count(self) -> int:
        """FTS5 索引中的文档数。"""
        import sqlite3
        conn = sqlite3.connect(str(self._db._db_path))
        try:
            return conn.execute("SELECT COUNT(*) FROM messages_fts").fetchone()[0]
        finally:
            conn.close()
