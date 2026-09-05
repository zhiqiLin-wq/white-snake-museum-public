"""SQLite 记忆数据库 — Source of Truth。

管理 memory.db 中所有表的创建、迁移和 CRUD 操作。
采用 WAL 模式，支持并发读写。

表结构:
- messages: 所有消息/记忆的原文（Source of Truth）
- messages_fts: FTS5 全文索引
- long_term_memories: 长期记忆元数据
- recall_messages_meta: 被淘汰消息元数据
- memory_changelog: UPDATE/MERGE/DELETE 审计日志
- core_memory_blocks: persona / human 块当前版本
- core_memory_snapshots: persona / human 历史版本
- extraction_state: 每个 thread 的提取进度追踪
"""

import json
import logging
import sqlite3
import uuid
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional

_BEIJING_TZ = timezone(timedelta(hours=8))


def _now() -> str:
    return datetime.now(_BEIJING_TZ).isoformat()


logger = logging.getLogger(__name__)

# 数据库版本（用于未来迁移）
DB_VERSION = 1


class MemoryDB:
    """SQLite 记忆数据库管理器（全局单例）。

    隔离模型：全局单例 memory.db + user 维度复合 key 隔离，
    而非每个用户独立文件（避免多文件连接管理复杂度）。
    用户隔离由 routes 层的 _ctx_thread_id(user_id, conversation_id) 复合 key 保证。
    """

    def __init__(self, db_path: Path):
        """
        Args:
            db_path: memory.db 文件的完整路径
        """
        self._db_path = db_path
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    # ================================================================
    #  初始化
    # ================================================================

    def _init_db(self):
        """创建所有表和索引（如果不存在）。"""
        conn = sqlite3.connect(str(self._db_path))
        try:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA foreign_keys=ON")

            # --- messages: Source of Truth ---
            conn.execute("""
                CREATE TABLE IF NOT EXISTS messages (
                    message_id TEXT PRIMARY KEY,
                    content TEXT NOT NULL,
                    role TEXT NOT NULL DEFAULT 'unknown',
                    thread_id TEXT DEFAULT '',
                    source TEXT NOT NULL DEFAULT 'fifo',
                    timestamp TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    pending_embedding INTEGER NOT NULL DEFAULT 0
                )
            """)

            # --- messages_fts: FTS5 全文索引 ---
            conn.execute("""
                CREATE VIRTUAL TABLE IF NOT EXISTS messages_fts USING fts5(
                    content,
                    content_rowid='rowid',
                    tokenize='unicode61'
                )
            """)

            # --- long_term_memories: 长期记忆 ---
            conn.execute("""
                CREATE TABLE IF NOT EXISTS long_term_memories (
                    memory_id TEXT PRIMARY KEY,
                    content TEXT NOT NULL,
                    memory_type TEXT NOT NULL DEFAULT 'Entity',
                    importance REAL NOT NULL DEFAULT 0.5,
                    confidence REAL NOT NULL DEFAULT 0.5,
                    last_accessed_at TEXT,
                    last_observed_at TEXT,
                    access_count INTEGER NOT NULL DEFAULT 0,
                    contradiction_count INTEGER NOT NULL DEFAULT 0,
                    is_unstable INTEGER NOT NULL DEFAULT 0,
                    status TEXT NOT NULL DEFAULT 'active',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    FOREIGN KEY (memory_id) REFERENCES messages(message_id)
                )
            """)

            # --- recall_messages_meta: 被淘汰消息元数据 ---
            conn.execute("""
                CREATE TABLE IF NOT EXISTS recall_messages_meta (
                    message_id TEXT PRIMARY KEY,
                    thread_id TEXT NOT NULL DEFAULT '',
                    role TEXT NOT NULL DEFAULT 'unknown',
                    timestamp TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (message_id) REFERENCES messages(message_id)
                )
            """)

            # --- memory_changelog: 审计日志 ---
            conn.execute("""
                CREATE TABLE IF NOT EXISTS memory_changelog (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    memory_id TEXT NOT NULL,
                    batch_id INTEGER NOT NULL,
                    event_type TEXT NOT NULL,
                    old_value TEXT,
                    new_value TEXT,
                    reason TEXT DEFAULT '',
                    changed_at TEXT NOT NULL,
                    changed_by TEXT NOT NULL DEFAULT 'system'
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_changelog_memory_id
                ON memory_changelog(memory_id)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_changelog_batch_id
                ON memory_changelog(batch_id)
            """)

            # --- core_memory_blocks: persona/human 当前版本 ---
            conn.execute("""
                CREATE TABLE IF NOT EXISTS core_memory_blocks (
                    block_name TEXT PRIMARY KEY,
                    content TEXT NOT NULL DEFAULT '{}',
                    version INTEGER NOT NULL DEFAULT 1,
                    token_count INTEGER NOT NULL DEFAULT 0,
                    updated_at TEXT NOT NULL
                )
            """)

            # --- core_memory_snapshots: 历史版本 ---
            conn.execute("""
                CREATE TABLE IF NOT EXISTS core_memory_snapshots (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    block_name TEXT NOT NULL,
                    version INTEGER NOT NULL,
                    content TEXT NOT NULL,
                    snapshot_at TEXT NOT NULL
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_snapshots_block_version
                ON core_memory_snapshots(block_name, version)
            """)

            # --- extraction_state: 提取进度追踪 ---
            conn.execute("""
                CREATE TABLE IF NOT EXISTS extraction_state (
                    thread_id TEXT PRIMARY KEY,
                    last_extracted_index INTEGER NOT NULL DEFAULT -1,
                    last_active_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
            """)

            # --- 元数据表: 记录数据库版本 ---
            conn.execute("""
                CREATE TABLE IF NOT EXISTS _meta (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                )
            """)
            conn.execute(
                "INSERT OR REPLACE INTO _meta (key, value) VALUES (?, ?)",
                ("db_version", str(DB_VERSION)),
            )

            # --- 批量操作计数器（用于生成 batch_id） ---
            conn.execute("""
                CREATE TABLE IF NOT EXISTS _batch_counter (
                    id INTEGER PRIMARY KEY CHECK (id = 1),
                    next_batch_id INTEGER NOT NULL DEFAULT 1
                )
            """)
            conn.execute(
                "INSERT OR IGNORE INTO _batch_counter (id, next_batch_id) VALUES (1, 1)"
            )

            conn.commit()
            logger.info("memory_db initialized: %s (v%d)", self._db_path, DB_VERSION)
        finally:
            conn.close()

    # ================================================================
    #  消息 CRUD（Source of Truth）
    # ================================================================

    def insert_message(self, content: str, role: str = "unknown",
                       thread_id: str = "", source: str = "fifo",
                       timestamp: Optional[str] = None,
                       pending_embedding: int = 1) -> str:
        """插入一条消息原文，返回 message_id。

        同时写入 messages_fts 全文索引。

        pending_embedding: 默认 1（待写入向量）。调用方写入 ChromaDB 成功后
        应调用 mark_embedding_done() 置 0；失败保持 1，由后台补偿重试。
        """
        message_id = f"msg_{uuid.uuid4().hex[:16]}"
        ts = timestamp or _now()
        created_at = _now()

        conn = sqlite3.connect(str(self._db_path))
        try:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute(
                """INSERT INTO messages (message_id, content, role, thread_id, source, timestamp, created_at, pending_embedding)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (message_id, content, role, thread_id, source, ts, created_at, pending_embedding),
            )
            # 同步写入 FTS5 索引（中文 bigram 化，使 FTS 检索真正命中中文）
            conn.execute(
                "INSERT INTO messages_fts (rowid, content) VALUES ((SELECT rowid FROM messages WHERE message_id = ?), ?)",
                (message_id, self._to_fts_tokens(content)),
            )
            conn.commit()
            logger.debug("memory_db insert_message: %s role=%s (%d chars)", message_id, role, len(content))
            return message_id
        finally:
            conn.close()

    def get_message(self, message_id: str) -> Optional[dict]:
        """根据 message_id 获取消息完整内容。"""
        conn = sqlite3.connect(str(self._db_path))
        try:
            row = conn.execute(
                "SELECT message_id, content, role, thread_id, source, timestamp, created_at, pending_embedding "
                "FROM messages WHERE message_id = ?",
                (message_id,),
            ).fetchone()
            if row is None:
                return None
            return {
                "message_id": row[0],
                "content": row[1],
                "role": row[2],
                "thread_id": row[3],
                "source": row[4],
                "timestamp": row[5],
                "created_at": row[6],
                "pending_embedding": bool(row[7]),
            }
        finally:
            conn.close()

    def get_messages_by_ids(self, message_ids: list[str]) -> list[dict]:
        """批量获取消息内容。"""
        if not message_ids:
            return []
        conn = sqlite3.connect(str(self._db_path))
        try:
            placeholders = ",".join("?" for _ in message_ids)
            rows = conn.execute(
                f"SELECT message_id, content, role, thread_id, source, timestamp, created_at, pending_embedding "
                f"FROM messages WHERE message_id IN ({placeholders})",
                message_ids,
            ).fetchall()
            return [
                {
                    "message_id": r[0], "content": r[1], "role": r[2],
                    "thread_id": r[3], "source": r[4], "timestamp": r[5],
                    "created_at": r[6], "pending_embedding": bool(r[7]),
                }
                for r in rows
            ]
        finally:
            conn.close()

    def mark_embedding_done(self, message_id: str):
        """标记消息的 embedding 已成功写入 ChromaDB。"""
        conn = sqlite3.connect(str(self._db_path))
        try:
            conn.execute(
                "UPDATE messages SET pending_embedding = 0 WHERE message_id = ?",
                (message_id,),
            )
            conn.commit()
        finally:
            conn.close()

    def get_pending_embeddings(self, limit: int = 100) -> list[dict]:
        """获取所有 pending_embedding=True 的消息。"""
        conn = sqlite3.connect(str(self._db_path))
        try:
            rows = conn.execute(
                "SELECT message_id, content, role, thread_id, source, timestamp, created_at "
                "FROM messages WHERE pending_embedding = 1 LIMIT ?",
                (limit,),
            ).fetchall()
            return [
                {
                    "message_id": r[0], "content": r[1], "role": r[2],
                    "thread_id": r[3], "source": r[4], "timestamp": r[5],
                    "created_at": r[6],
                }
                for r in rows
            ]
        finally:
            conn.close()

    def get_all_messages(self) -> list[dict]:
        """获取所有消息（用于全量重建向量索引）。"""
        conn = sqlite3.connect(str(self._db_path))
        try:
            rows = conn.execute(
                "SELECT message_id, content, role, thread_id, source, timestamp, created_at, pending_embedding "
                "FROM messages ORDER BY created_at ASC"
            ).fetchall()
            return [
                {
                    "message_id": r[0], "content": r[1], "role": r[2],
                    "thread_id": r[3], "source": r[4], "timestamp": r[5],
                    "created_at": r[6], "pending_embedding": bool(r[7]),
                }
                for r in rows
            ]
        finally:
            conn.close()

    # ================================================================
    #  FTS5 全文检索
    # ================================================================

    def search_fts(self, query: str, top_k: int = 30,
                   source_filter: Optional[str] = None) -> list[dict]:
        """FTS5 全文检索，返回候选 message_id 列表。

        支持中英文混合查询。中文使用 bigram 切分 + LIKE fallback。

        Args:
            query: 搜索查询
            top_k: 返回数量
            source_filter: 可选的消息来源过滤 ('memory_extraction' | 'recall' | None=all)
        """
        conn = sqlite3.connect(str(self._db_path))
        try:
            rows = []

            # ---- 策略 1: Bigram OR query for FTS5 ----
            # 对中文文本做 2-gram 切分，英文保留原词
            tokens = self._tokenize_for_fts(query)
            if tokens:
                or_query = " OR ".join(tokens)
                source_clause = ""
                params = [or_query]
                if source_filter:
                    source_clause = "AND m.source = ?"
                    params.append(source_filter)
                params.append(top_k)
                try:
                    rows = conn.execute(
                        f"""SELECT m.message_id, m.content, m.role, m.thread_id, m.timestamp,
                                  m.pending_embedding, m.source, fts.rank
                           FROM messages_fts fts
                           JOIN messages m ON m.rowid = fts.rowid
                           WHERE messages_fts MATCH ? {source_clause}
                           ORDER BY rank
                           LIMIT ?""",
                        params,
                    ).fetchall()
                except Exception:
                    rows = []

            # ---- 策略 2: LIKE fallback (FTS5 无结果时) ----
            if not rows:
                rows = self._like_search(conn, query, top_k, source_filter)

            return [
                {
                    "message_id": r[0], "content": r[1], "role": r[2],
                    "thread_id": r[3], "timestamp": r[4],
                    "pending_embedding": bool(r[5]),
                    # FTS 命中路径返回 8 列 (rank 在 r[7])；LIKE fallback 返回 7 列 (rank 在 r[6])
                    "fts_rank": r[7] if len(r) > 7 else (r[6] if len(r) > 6 else 0.5),
                }
                for r in rows
            ]
        finally:
            conn.close()

    @staticmethod
    def _is_chinese(c: str) -> bool:
        """判断字符是否为中文（含扩展区）。"""
        return '一' <= c <= '鿿' or '㐀' <= c <= '䶿'

    def _to_fts_tokens(self, text: str) -> str:
        """把文本切分为 FTS5 可索引的 token 序列（空格分隔）。

        中文: 2-gram 切分（使 FTS 检索真正命中中文）
        英文/数字: 保留原词
        用于写入 messages_fts 索引，与 _tokenize_for_fts 的查询侧 bigram 对齐。
        """
        tokens = []
        i = 0
        n = len(text)
        while i < n:
            c = text[i]
            if self._is_chinese(c):
                if i + 1 < n and self._is_chinese(text[i + 1]):
                    tokens.append(text[i:i + 2])
                    i += 1
                else:
                    tokens.append(c)
                    i += 1
            elif c.isalnum():
                start = i
                while i < n and text[i].isalnum():
                    i += 1
                tokens.append(text[start:i])
            else:
                i += 1
        return " ".join(tokens)

    def _tokenize_for_fts(self, query: str) -> list[str]:
        """将查询切分为 FTS5 可用的 token 列表。

        中文: 2-gram 字符切分
        英文/数字: 保留原词
        生成 FTS5 OR query 格式: "ab" OR "bc" OR word
        """
        tokens = []
        i = 0
        while i < len(query):
            c = query[i]
            # 中文字符范围
            if self._is_chinese(c):
                # 取 2-gram
                if i + 1 < len(query) and self._is_chinese(query[i + 1]):
                    tokens.append(f'"{query[i:i+2]}"')
                    i += 1
                else:
                    tokens.append(f'"{c}"')
                    i += 1
            elif c.isalnum():
                # 英文/数字: 收集完整词
                start = i
                while i < len(query) and query[i].isalnum():
                    i += 1
                word = query[start:i]
                if len(word) > 1:
                    tokens.append(f'"{word}"')
            else:
                i += 1
        return tokens[:60]  # 限制 token 数量避免查询过长

    def _like_search(self, conn, query: str, top_k: int,
                      source_filter: Optional[str] = None) -> list:
        """LIKE 模糊匹配降级搜索。

        对中文按 bigram 做 LIKE 匹配，用匹配 bigram 数量作为排序依据。
        """
        # 提取 bigrams
        bigrams = []
        for i in range(len(query) - 1):
            if '一' <= query[i] <= '鿿' and '一' <= query[i + 1] <= '鿿':
                bigrams.append(query[i:i + 2])

        if not bigrams:
            # 纯英文或短查询：简单 LIKE
            like_pattern = f"%{query}%"
            source_clause = "AND m.source = ?" if source_filter else ""
            params = [like_pattern]
            if source_filter:
                params.append(source_filter)
            params.append(top_k)
            return conn.execute(
                f"""SELECT m.message_id, m.content, m.role, m.thread_id, m.timestamp,
                          m.pending_embedding, 0.5 as rank
                   FROM messages m
                   WHERE m.content LIKE ? {source_clause}
                   ORDER BY m.created_at DESC
                   LIMIT ?""",
                params,
            ).fetchall()

        # 对每个 bigram 做 LIKE 匹配，匹配数越多排名越高
        # SELECT 中的 rank 计算需要 len(bigrams) 个参数
        # WHERE 中的 OR 条件也需要 len(bigrams) 个参数
        # 总共: len(bigrams) [for rank calc] + len(bigrams) [for WHERE] + source + top_k
        like_patterns = [f"%{bg}%" for bg in bigrams]
        like_parts = " + ".join(
            f"(CASE WHEN m.content LIKE ? THEN 1 ELSE 0 END)" for _ in bigrams
        )
        select_params = like_patterns[:]  # for rank computation in SELECT
        where_params = like_patterns[:]   # for WHERE OR conditions

        source_clause = "AND m.source = ?" if source_filter else ""
        all_params = select_params + where_params
        if source_filter:
            all_params.append(source_filter)
        all_params.append(top_k)

        return conn.execute(
            f"""SELECT m.message_id, m.content, m.role, m.thread_id, m.timestamp,
                      m.pending_embedding, ({like_parts}) * 1.0 / {len(bigrams)} as rank
               FROM messages m
               WHERE ({' OR '.join('m.content LIKE ?' for _ in bigrams)}) {source_clause}
               ORDER BY rank DESC
               LIMIT ?""",
            all_params,
        ).fetchall()

    # ================================================================
    #  长期记忆 CRUD
    # ================================================================

    def insert_long_term_memory(self, memory_id: str, content: str,
                                 memory_type: str = "Entity",
                                 importance: float = 0.5,
                                 confidence: float = 0.5) -> str:
        """插入一条长期记忆元数据。"""
        now = _now()
        conn = sqlite3.connect(str(self._db_path))
        try:
            conn.execute(
                """INSERT INTO long_term_memories
                   (memory_id, content, memory_type, importance, confidence,
                    last_observed_at, created_at, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (memory_id, content, memory_type, importance, confidence,
                 now, now, now),
            )
            conn.commit()
            return memory_id
        finally:
            conn.close()

    def get_long_term_memory(self, memory_id: str) -> Optional[dict]:
        """获取单条长期记忆。"""
        conn = sqlite3.connect(str(self._db_path))
        try:
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                "SELECT * FROM long_term_memories WHERE memory_id = ?",
                (memory_id,),
            ).fetchone()
            if row is None:
                return None
            return dict(row)
        finally:
            conn.close()

    def get_active_memories(self, limit: int = 100) -> list[dict]:
        """获取所有活跃的长期记忆。"""
        conn = sqlite3.connect(str(self._db_path))
        try:
            rows = conn.execute(
                "SELECT memory_id, content, memory_type, importance, confidence, "
                "last_accessed_at, last_observed_at, access_count, created_at "
                "FROM long_term_memories WHERE status = 'active' "
                "ORDER BY importance DESC, created_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
            return [
                {
                    "memory_id": r[0], "content": r[1], "memory_type": r[2],
                    "importance": r[3], "confidence": r[4],
                    "last_accessed_at": r[5], "last_observed_at": r[6],
                    "access_count": r[7], "created_at": r[8],
                }
                for r in rows
            ]
        finally:
            conn.close()

    def get_memories_by_type(self, memory_type: str, limit: int = 100) -> list[dict]:
        """按类型获取记忆。"""
        conn = sqlite3.connect(str(self._db_path))
        try:
            rows = conn.execute(
                "SELECT memory_id, content, memory_type, importance, confidence, "
                "last_accessed_at, access_count, created_at "
                "FROM long_term_memories WHERE status = 'active' AND memory_type = ? "
                "ORDER BY importance DESC LIMIT ?",
                (memory_type, limit),
            ).fetchall()
            return [
                {
                    "memory_id": r[0], "content": r[1], "memory_type": r[2],
                    "importance": r[3], "confidence": r[4],
                    "last_accessed_at": r[5], "access_count": r[6], "created_at": r[7],
                }
                for r in rows
            ]
        finally:
            conn.close()

    def update_memory_access(self, memory_id: str):
        """更新记忆的访问时间和计数。"""
        now = _now()
        conn = sqlite3.connect(str(self._db_path))
        try:
            conn.execute(
                """UPDATE long_term_memories
                   SET last_accessed_at = ?, access_count = access_count + 1, updated_at = ?
                   WHERE memory_id = ?""",
                (now, now, memory_id),
            )
            conn.commit()
        finally:
            conn.close()

    def update_memory_status(self, memory_id: str, status: str,
                              reason: str = ""):
        """更新记忆状态（active/consolidated/tombstone）。"""
        now = _now()
        conn = sqlite3.connect(str(self._db_path))
        try:
            old = conn.execute(
                "SELECT content, memory_type, importance, confidence FROM long_term_memories WHERE memory_id = ?",
                (memory_id,),
            ).fetchone()
            conn.execute(
                "UPDATE long_term_memories SET status = ?, updated_at = ? WHERE memory_id = ?",
                (status, now, memory_id),
            )
            conn.commit()
            if old:
                batch_id = self._next_batch_id(conn)
                conn.execute(
                    """INSERT INTO memory_changelog
                       (memory_id, batch_id, event_type, old_value, new_value, reason, changed_at, changed_by)
                       VALUES (?, ?, ?, ?, ?, ?, ?, 'system')""",
                    (memory_id, batch_id, status.upper(),
                     json.dumps({"content": old[0], "memory_type": old[1],
                                 "importance": old[2], "confidence": old[3]}, ensure_ascii=False),
                     json.dumps({"status": status}, ensure_ascii=False),
                     reason, now),
                )
                conn.commit()
        finally:
            conn.close()

    def get_memories_for_merge(self, min_similarity: float = 0.7, limit: int = 50) -> list[dict]:
        """获取可能需要合并的记忆列表（按创建时间排序，用于后续语义去重）。"""
        conn = sqlite3.connect(str(self._db_path))
        try:
            rows = conn.execute(
                "SELECT memory_id, content, memory_type, importance, confidence, created_at "
                "FROM long_term_memories WHERE status = 'active' "
                "ORDER BY created_at ASC LIMIT ?",
                (limit,),
            ).fetchall()
            return [
                {
                    "memory_id": r[0], "content": r[1], "memory_type": r[2],
                    "importance": r[3], "confidence": r[4], "created_at": r[5],
                }
                for r in rows
            ]
        finally:
            conn.close()

    def get_memories_for_decay(self, days_unaccessed: int = 30) -> list[dict]:
        """获取需要衰减检查的记忆候选（超过 days_unaccessed 天未被访问）。

        用 COALESCE(last_accessed_at, created_at) 判断，使从未被访问的记忆
        也能基于创建时间进入候选，而非永不衰减 (P2-5)。
        实际是否 tombstone 由调用方用 MemoryScorer 判定。
        """
        conn = sqlite3.connect(str(self._db_path))
        try:
            cutoff = datetime.now(_BEIJING_TZ) - timedelta(days=days_unaccessed)
            cutoff_str = cutoff.isoformat()
            rows = conn.execute(
                """SELECT memory_id, content, memory_type, importance, confidence,
                          last_accessed_at, access_count, created_at
                   FROM long_term_memories WHERE status = 'active'
                   AND COALESCE(last_accessed_at, created_at) < ?
                   ORDER BY importance ASC, access_count ASC""",
                (cutoff_str,),
            ).fetchall()
            return [
                {
                    "memory_id": r[0], "content": r[1], "memory_type": r[2],
                    "importance": r[3], "confidence": r[4],
                    "last_accessed_at": r[5], "access_count": r[6], "created_at": r[7],
                }
                for r in rows
            ]
        finally:
            conn.close()

    def update_memory_instability(self, memory_id: str):
        """标记记忆为不稳定（矛盾反转次数 +1）。"""
        conn = sqlite3.connect(str(self._db_path))
        try:
            conn.execute(
                """UPDATE long_term_memories
                   SET contradiction_count = contradiction_count + 1,
                       is_unstable = CASE WHEN contradiction_count + 1 >= 2 THEN 1 ELSE 0 END,
                       confidence = MAX(0.1, confidence - 0.2),
                       updated_at = ?
                   WHERE memory_id = ?""",
                (_now(), memory_id),
            )
            conn.commit()
        finally:
            conn.close()

    # ================================================================
    #  Recall Messages 元数据
    # ================================================================

    def insert_recall_message_meta(self, message_id: str, thread_id: str = "",
                                    role: str = "unknown", timestamp: Optional[str] = None):
        """记录一条被淘汰消息的元数据。"""
        conn = sqlite3.connect(str(self._db_path))
        try:
            conn.execute(
                """INSERT INTO recall_messages_meta (message_id, thread_id, role, timestamp, created_at)
                   VALUES (?, ?, ?, ?, ?)""",
                (message_id, thread_id, role, timestamp or _now(), _now()),
            )
            conn.commit()
        finally:
            conn.close()

    # ================================================================
    #  Changelog 审计
    # ================================================================

    def _next_batch_id(self, conn: sqlite3.Connection) -> int:
        """生成下一个 batch_id（数据库自增）。"""
        row = conn.execute(
            "SELECT next_batch_id FROM _batch_counter WHERE id = 1"
        ).fetchone()
        batch_id = row[0]
        conn.execute(
            "UPDATE _batch_counter SET next_batch_id = next_batch_id + 1 WHERE id = 1"
        )
        return batch_id

    def record_changelog(self, memory_id: str, event_type: str,
                          old_value: Optional[str] = None,
                          new_value: Optional[str] = None,
                          reason: str = "",
                          changed_by: str = "system") -> int:
        """记录一条 changelog 事件，返回 batch_id。"""
        now = _now()
        conn = sqlite3.connect(str(self._db_path))
        try:
            batch_id = self._next_batch_id(conn)
            conn.execute(
                """INSERT INTO memory_changelog
                   (memory_id, batch_id, event_type, old_value, new_value, reason, changed_at, changed_by)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (memory_id, batch_id, event_type, old_value, new_value, reason, now, changed_by),
            )
            conn.commit()
            return batch_id
        finally:
            conn.close()

    def get_changelog_by_batch(self, batch_id: int) -> list[dict]:
        """获取指定批次的所有 changelog 记录。"""
        conn = sqlite3.connect(str(self._db_path))
        try:
            cursor = conn.execute(
                "SELECT * FROM memory_changelog WHERE batch_id = ? ORDER BY id",
                (batch_id,),
            )
            rows = cursor.fetchall()
            cols = [d[0] for d in cursor.description] if rows else []
            return [dict(zip(cols, r)) for r in rows]
        finally:
            conn.close()

    def get_changelog_by_memory(self, memory_id: str, limit: int = 50) -> list[dict]:
        """获取指定记忆的所有变更记录。"""
        conn = sqlite3.connect(str(self._db_path))
        try:
            cursor = conn.execute(
                "SELECT * FROM memory_changelog WHERE memory_id = ? ORDER BY id DESC LIMIT ?",
                (memory_id, limit),
            )
            rows = cursor.fetchall()
            cols = [d[0] for d in cursor.description] if rows else []
            return [dict(zip(cols, r)) for r in rows]
        finally:
            conn.close()

    # ================================================================
    #  Core Memory Blocks
    # ================================================================

    def init_core_memory_block(self, block_name: str, content: str):
        """初始化 core memory 块（仅在不存在时插入）。"""
        conn = sqlite3.connect(str(self._db_path))
        try:
            existing = conn.execute(
                "SELECT block_name FROM core_memory_blocks WHERE block_name = ?",
                (block_name,),
            ).fetchone()
            if existing is None:
                conn.execute(
                    """INSERT INTO core_memory_blocks (block_name, content, version, token_count, updated_at)
                       VALUES (?, ?, 1, 0, ?)""",
                    (block_name, content, _now()),
                )
                conn.commit()
                logger.info("core_memory_block %s initialized", block_name)
        finally:
            conn.close()

    def get_core_memory_block(self, block_name: str) -> Optional[dict]:
        """获取 core memory 块的当前内容。"""
        conn = sqlite3.connect(str(self._db_path))
        try:
            row = conn.execute(
                "SELECT block_name, content, version, token_count, updated_at "
                "FROM core_memory_blocks WHERE block_name = ?",
                (block_name,),
            ).fetchone()
            if row is None:
                return None
            return {
                "block_name": row[0],
                "content": row[1],
                "version": row[2],
                "token_count": row[3],
                "updated_at": row[4],
            }
        finally:
            conn.close()

    def set_core_memory_block(self, block_name: str, content: str,
                               token_count: int = 0,
                               reason: str = "core_memory_replace") -> int:
        """更新 core memory 块内容。

        自动保存旧版本快照，递增版本号。

        Returns:
            新版本号
        """
        now = _now()
        conn = sqlite3.connect(str(self._db_path))
        try:
            old = conn.execute(
                "SELECT content, version FROM core_memory_blocks WHERE block_name = ?",
                (block_name,),
            ).fetchone()

            if old is None:
                conn.execute(
                    """INSERT INTO core_memory_blocks (block_name, content, version, token_count, updated_at)
                       VALUES (?, ?, 1, ?, ?)""",
                    (block_name, content, token_count, now),
                )
                conn.commit()
                return 1

            old_content, old_version = old
            new_version = old_version + 1

            # 保存旧版本快照
            conn.execute(
                """INSERT INTO core_memory_snapshots (block_name, version, content, snapshot_at)
                   VALUES (?, ?, ?, ?)""",
                (block_name, old_version, old_content, now),
            )

            # 更新当前版本
            conn.execute(
                """UPDATE core_memory_blocks
                   SET content = ?, version = ?, token_count = ?, updated_at = ?
                   WHERE block_name = ?""",
                (content, new_version, token_count, now, block_name),
            )
            conn.commit()

            # 记录 changelog
            batch_id = self._next_batch_id(conn)
            conn.execute(
                """INSERT INTO memory_changelog
                   (memory_id, batch_id, event_type, old_value, new_value, reason, changed_at, changed_by)
                   VALUES (?, ?, 'CORE_MEMORY_UPDATE', ?, ?, ?, ?, 'system')""",
                (f"core_{block_name}", batch_id,
                 old_content[:2000] if old_content else None,
                 content[:2000],
                 reason,
                 now),
            )
            conn.commit()

            logger.info("core_memory %s: v%d -> v%d", block_name, old_version, new_version)
            return new_version
        finally:
            conn.close()

    def append_core_memory_block(self, block_name: str, content_snippet: str) -> int:
        """向 core memory 块追加内容。

        在现有内容末尾追加新片段，自动保存快照。

        Returns:
            新版本号
        """
        now = _now()
        conn = sqlite3.connect(str(self._db_path))
        try:
            old = conn.execute(
                "SELECT content, version FROM core_memory_blocks WHERE block_name = ?",
                (block_name,),
            ).fetchone()

            old_content = old[0] if old else "{}"
            old_version = old[1] if old else 0
            new_version = old_version + 1

            # 尝试 JSON 合并
            new_content = self._merge_json_content(old_content, content_snippet)

            if old:
                conn.execute(
                    """INSERT INTO core_memory_snapshots (block_name, version, content, snapshot_at)
                       VALUES (?, ?, ?, ?)""",
                    (block_name, old_version, old_content, now),
                )
                conn.execute(
                    """UPDATE core_memory_blocks
                       SET content = ?, version = ?, token_count = 0, updated_at = ?
                       WHERE block_name = ?""",
                    (new_content, new_version, now, block_name),
                )
            else:
                conn.execute(
                    """INSERT INTO core_memory_blocks (block_name, content, version, token_count, updated_at)
                       VALUES (?, ?, 1, 0, ?)""",
                    (block_name, new_content, now),
                )

            conn.commit()

            batch_id = self._next_batch_id(conn)
            conn.execute(
                """INSERT INTO memory_changelog
                   (memory_id, batch_id, event_type, old_value, new_value, reason, changed_at, changed_by)
                   VALUES (?, ?, 'CORE_MEMORY_UPDATE', ?, ?, 'core_memory_append', ?, 'system')""",
                (f"core_{block_name}", batch_id,
                 old_content[:2000] if old_content else None,
                 content_snippet[:2000],
                 now),
            )
            conn.commit()

            return new_version
        finally:
            conn.close()

    def _merge_json_content(self, existing: str, snippet: str) -> str:
        """尝试将 snippet 合并到现有 JSON 内容中。"""
        try:
            existing_obj = json.loads(existing)
        except (json.JSONDecodeError, TypeError):
            # 非 JSON，直接拼接
            return existing + "\n" + snippet

        try:
            snippet_obj = json.loads(snippet)
        except (json.JSONDecodeError, TypeError):
            # snippet 不是 JSON，尝试存入 _notes 字段
            if "_notes" not in existing_obj:
                existing_obj["_notes"] = []
            existing_obj["_notes"].append(snippet)
            return json.dumps(existing_obj, ensure_ascii=False, indent=2)

        # 两者都是 JSON，浅层合并
        for key, value in snippet_obj.items():
            if key in existing_obj and isinstance(existing_obj[key], list) and isinstance(value, list):
                existing_obj[key].extend(value)
            elif key in existing_obj and isinstance(existing_obj[key], dict) and isinstance(value, dict):
                existing_obj[key].update(value)
            else:
                existing_obj[key] = value

        return json.dumps(existing_obj, ensure_ascii=False, indent=2)

    def get_core_memory_snapshots(self, block_name: str, limit: int = 10) -> list[dict]:
        """获取 core memory 块的版本历史。"""
        conn = sqlite3.connect(str(self._db_path))
        try:
            rows = conn.execute(
                "SELECT id, block_name, version, content, snapshot_at "
                "FROM core_memory_snapshots WHERE block_name = ? "
                "ORDER BY version DESC LIMIT ?",
                (block_name, limit),
            ).fetchall()
            return [
                {
                    "id": r[0], "block_name": r[1], "version": r[2],
                    "content": r[3], "snapshot_at": r[4],
                }
                for r in rows
            ]
        finally:
            conn.close()

    def rollback_core_memory(self, block_name: str, target_version: int) -> bool:
        """回滚 core memory 块到指定版本。"""
        conn = sqlite3.connect(str(self._db_path))
        try:
            snapshot = conn.execute(
                "SELECT content FROM core_memory_snapshots WHERE block_name = ? AND version = ?",
                (block_name, target_version),
            ).fetchone()
            if snapshot is None:
                return False

            current = conn.execute(
                "SELECT content, version FROM core_memory_blocks WHERE block_name = ?",
                (block_name,),
            ).fetchone()
            if current is None:
                return False

            now = _now()
            new_version = current[1] + 1

            conn.execute(
                """INSERT INTO core_memory_snapshots (block_name, version, content, snapshot_at)
                   VALUES (?, ?, ?, ?)""",
                (block_name, current[1], current[0], now),
            )
            conn.execute(
                """UPDATE core_memory_blocks
                   SET content = ?, version = ?, updated_at = ?
                   WHERE block_name = ?""",
                (snapshot[0], new_version, now, block_name),
            )
            conn.commit()
            logger.info("core_memory %s rolled back to v%d (now v%d)", block_name, target_version, new_version)
            return True
        finally:
            conn.close()

    def get_core_memory_token_count(self, block_name: str) -> int:
        """获取 core memory 块的 token 计数。"""
        conn = sqlite3.connect(str(self._db_path))
        try:
            row = conn.execute(
                "SELECT token_count FROM core_memory_blocks WHERE block_name = ?",
                (block_name,),
            ).fetchone()
            return row[0] if row else 0
        finally:
            conn.close()

    # ================================================================
    #  Extraction State
    # ================================================================

    def get_extraction_state(self, thread_id: str) -> dict:
        """获取指定 thread 的提取进度。"""
        conn = sqlite3.connect(str(self._db_path))
        try:
            row = conn.execute(
                "SELECT last_extracted_index, last_active_at, updated_at FROM extraction_state WHERE thread_id = ?",
                (thread_id,),
            ).fetchone()
            if row is None:
                return {
                    "thread_id": thread_id,
                    "last_extracted_index": -1,
                    "last_active_at": None,
                    "updated_at": None,
                }
            return {
                "thread_id": thread_id,
                "last_extracted_index": row[0],
                "last_active_at": row[1],
                "updated_at": row[2],
            }
        finally:
            conn.close()

    def update_extraction_state(self, thread_id: str, last_extracted_index: int,
                                 last_active_at: Optional[str] = None):
        """更新提取进度。"""
        now = _now()
        active_at = last_active_at or now
        conn = sqlite3.connect(str(self._db_path))
        try:
            conn.execute(
                """INSERT OR REPLACE INTO extraction_state
                   (thread_id, last_extracted_index, last_active_at, updated_at)
                   VALUES (?, ?, ?, ?)""",
                (thread_id, last_extracted_index, active_at, now),
            )
            conn.commit()
        finally:
            conn.close()

    def touch_active_at(self, thread_id: str):
        """更新 thread 的最后活跃时间（用于空闲检测）。"""
        now = _now()
        conn = sqlite3.connect(str(self._db_path))
        try:
            conn.execute(
                """INSERT INTO extraction_state (thread_id, last_extracted_index, last_active_at, updated_at)
                   VALUES (?, -1, ?, ?)
                   ON CONFLICT(thread_id) DO UPDATE SET last_active_at = ?, updated_at = ?""",
                (thread_id, now, now, now, now),
            )
            conn.commit()
        finally:
            conn.close()

    def get_idle_threads(self, idle_minutes: int = 30) -> list[str]:
        """获取空闲超过指定分钟的 thread_id 列表。"""
        conn = sqlite3.connect(str(self._db_path))
        try:
            cutoff = datetime.now(_BEIJING_TZ) - timedelta(minutes=idle_minutes)
            cutoff_str = cutoff.isoformat()
            rows = conn.execute(
                "SELECT thread_id FROM extraction_state WHERE last_active_at < ?",
                (cutoff_str,),
            ).fetchall()
            return [r[0] for r in rows]
        finally:
            conn.close()

    def delete_thread_data(self, thread_id: str) -> dict:
        """删除某对话（thread_id）产生的全部记忆/召回数据。

        删除范围（按外键依赖顺序，先删子表再删父表）:
        1. long_term_memories（memory_id 关联 messages）
        2. recall_messages_meta（thread_id 直接匹配）
        3. messages_fts（rowid 对齐 messages.rowid）
        4. messages（原文，Source of Truth）
        5. extraction_state（thread_id 直接匹配）

        memory_changelog 保留（审计日志，无外键约束，不影响删除）。

        Returns:
            {"deleted_messages": int, "deleted_memories": int, "deleted_recall_meta": int}
        """
        if not thread_id:
            return {"deleted_messages": 0, "deleted_memories": 0, "deleted_recall_meta": 0}
        conn = sqlite3.connect(str(self._db_path))
        try:
            conn.execute("PRAGMA journal_mode=WAL")
            cur_mem = conn.execute(
                "DELETE FROM long_term_memories WHERE memory_id IN "
                "(SELECT message_id FROM messages WHERE thread_id = ?)",
                (thread_id,),
            )
            cur_meta = conn.execute(
                "DELETE FROM recall_messages_meta WHERE thread_id = ?",
                (thread_id,),
            )
            conn.execute(
                "DELETE FROM messages_fts WHERE rowid IN "
                "(SELECT rowid FROM messages WHERE thread_id = ?)",
                (thread_id,),
            )
            cur_msg = conn.execute(
                "DELETE FROM messages WHERE thread_id = ?",
                (thread_id,),
            )
            conn.execute(
                "DELETE FROM extraction_state WHERE thread_id = ?",
                (thread_id,),
            )
            conn.commit()
            stats = {
                "deleted_messages": cur_msg.rowcount,
                "deleted_memories": cur_mem.rowcount,
                "deleted_recall_meta": cur_meta.rowcount,
            }
            logger.info("memory_db delete_thread_data thread=%s: %s", thread_id, stats)
            return stats
        finally:
            conn.close()

    # ================================================================
    #  统计
    # ================================================================

    def get_stats(self) -> dict:
        """获取数据库统计信息。"""
        conn = sqlite3.connect(str(self._db_path))
        try:
            return {
                "total_messages": conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0],
                "total_memories": conn.execute(
                    "SELECT COUNT(*) FROM long_term_memories WHERE status = 'active'"
                ).fetchone()[0],
                "total_recall_meta": conn.execute("SELECT COUNT(*) FROM recall_messages_meta").fetchone()[0],
                "pending_embeddings": conn.execute(
                    "SELECT COUNT(*) FROM messages WHERE pending_embedding = 1"
                ).fetchone()[0],
                "consolidated_memories": conn.execute(
                    "SELECT COUNT(*) FROM long_term_memories WHERE status = 'consolidated'"
                ).fetchone()[0],
                "tombstoned_memories": conn.execute(
                    "SELECT COUNT(*) FROM long_term_memories WHERE status = 'tombstone'"
                ).fetchone()[0],
                "changelog_entries": conn.execute("SELECT COUNT(*) FROM memory_changelog").fetchone()[0],
            }
        finally:
            conn.close()

    def close(self):
        """关闭数据库连接（no-op for sqlite3 每次连接自动关闭）。"""
        pass
