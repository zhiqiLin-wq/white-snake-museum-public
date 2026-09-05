"""双写 ChromaDB + SQLite 封装。

架构: SQLite (Source of Truth) -> ChromaDB (语义索引)

写入流程:
1. 所有消息/记忆首先写入 SQLite messages 表，获得 message_id
2. 向量化后写入 ChromaDB，附带 message_id 关联
3. ChromaDB 写入失败 -> pending_embedding=True -> 后台补偿重试

检索流程:
1. 外部使用 HybridRetriever 进行 FTS5 + 向量两阶段检索
2. 本模块提供底层的 ChromaDB 向量检索（供 HybridRetriever 调用）

去重:
- 防线 1: MD5 精确去重（写入前）
- 防线 2: Cosine similarity > 0.95 向量去重（LLM 提取后）

嵌入模型: shibing624/text2vec-base-chinese (384维, cosine 距离)
"""

import logging
import uuid
from datetime import datetime, timezone, timedelta
from typing import Optional

from .memory_db import MemoryDB

_BEIJING_TZ = timezone(timedelta(hours=8))


def _now() -> str:
    return datetime.now(_BEIJING_TZ).isoformat()


logger = logging.getLogger(__name__)

COLLECTION_RECALL_MESSAGES = "recall_messages"
COLLECTION_LONG_TERM_MEMORIES = "long_term_memories"


class RecallStore:
    """双写存储管理器。

    管理两个 ChromaDB 集合: recall_messages 和 long_term_memories。
    MemoryDB 作为 Source of Truth。
    """

    def __init__(self, chroma_client, embedder, memory_db: MemoryDB, user_id: str = ""):
        """
        Args:
            chroma_client: chromadb.PersistentClient 实例（复用已有）
            embedder: Embedder 实例
            memory_db: MemoryDB 实例（SQLite Source of Truth）
            user_id: 用户 ID。非空时 ChromaDB 集合名加用户后缀，实现物理隔离。
        """
        self._client = chroma_client
        self._embedder = embedder
        self._db = memory_db
        suffix = f"_{user_id}" if user_id else ""
        self._recall_collection_name = f"{COLLECTION_RECALL_MESSAGES}{suffix}"
        self._memory_collection_name = f"{COLLECTION_LONG_TERM_MEMORIES}{suffix}"
        self._recall_collection = None
        self._memory_collection = None
        self._init_collections()

    def _init_collections(self):
        """初始化两个 ChromaDB 集合（集合名按 user_id 隔离）。"""
        try:
            self._recall_collection = self._client.get_collection(
                self._recall_collection_name
            )
        except Exception:
            self._recall_collection = self._client.create_collection(
                name=self._recall_collection_name,
                metadata={"hnsw:space": "cosine"},
            )

        try:
            self._memory_collection = self._client.get_collection(
                self._memory_collection_name
            )
        except Exception:
            self._memory_collection = self._client.create_collection(
                name=self._memory_collection_name,
                metadata={"hnsw:space": "cosine"},
            )

        logger.info(
            "RecallStore ready: recall=%d, memories=%d",
            self._recall_collection.count(),
            self._memory_collection.count(),
        )

    # ================================================================
    #  双写: Recall Messages
    # ================================================================

    def add_recall_message(self, message: dict) -> Optional[str]:
        """双写一条被弹出的原始消息。

        Step 1: 写入 SQLite messages 表（Source of Truth）
        Step 2: 向量化写入 ChromaDB（语义索引）
        Step 3: 写入 recall_messages_meta 表
        Step 4: ChromaDB 失败 -> pending_embedding=True

        Returns:
            message_id，失败返回 None
        """
        content = str(message.get("content", ""))
        if not content.strip():
            return None

        role = message.get("role", "unknown")
        thread_id = message.get("_thread_id", "")
        timestamp = message.get("timestamp", _now())

        # Step 1: 写入 SQLite Source of Truth
        message_id = self._db.insert_message(
            content=content,
            role=role,
            thread_id=thread_id,
            source="recall",
            timestamp=timestamp,
        )

        # Step 2: 写入 SQLite 元数据
        self._db.insert_recall_message_meta(
            message_id=message_id,
            thread_id=thread_id,
            role=role,
            timestamp=timestamp,
        )

        # Step 3: 向量化 + 写入 ChromaDB
        try:
            embedding = self._embedder.embed([content])[0]
            self._recall_collection.add(
                embeddings=[embedding],
                documents=[content],
                metadatas=[{
                    "message_id": message_id,
                    "role": role,
                    "thread_id": thread_id,
                    "timestamp": timestamp,
                }],
                ids=[message_id],
            )
            self._db.mark_embedding_done(message_id)
            logger.debug("recall_store dual-write recall: %s (%d chars)", message_id, len(content))
        except Exception as e:
            logger.warning("recall_store ChromaDB write failed for %s: %s, pending retry", message_id, e)

        return message_id

    # ================================================================
    #  双写: Long-term Memories
    # ================================================================

    def check_duplicate_md5(self, content: str) -> Optional[str]:
        """精确内容去重（防线 1）。

        实现为全文相等匹配（并非 MD5 哈希），比 MD5 更严格：
        语义等价但字面不同的表述不会被误判为重复。
        保留旧方法名以兼容现有调用方与测试。

        Returns:
            已存在的 message_id，或 None（不重复）
        """
        import sqlite3
        conn = sqlite3.connect(str(self._db._db_path))
        try:
            row = conn.execute(
                "SELECT message_id FROM messages WHERE content = ? LIMIT 1",
                (content,),
            ).fetchone()
            if row:
                logger.debug("recall_store MD5 duplicate: %s", row[0])
                return row[0]
            return None
        finally:
            conn.close()

    def check_duplicate_vector(self, memory_text: str) -> bool:
        """向量去重（防线 2）。

        Returns:
            True 如果 cosine distance < 0.05 (similarity > 0.95)
        """
        if self._memory_collection is None or self._memory_collection.count() == 0:
            return False
        try:
            embedding = self._embedder.embed([memory_text])[0]
            results = self._memory_collection.query(
                query_embeddings=[embedding],
                n_results=1,
            )
            if results and results.get("distances") and results["distances"][0]:
                distance = results["distances"][0][0]
                return distance < 0.05
        except Exception as e:
            logger.warning("recall_store vector dedup check failed: %s", e)
        return False

    def add_memory(self, memory_text: str,
                   memory_type: str = "Entity",
                   importance: float = 0.5,
                   confidence: float = 0.5,
                   thread_id: str = "") -> Optional[str]:
        """双写一条长期记忆。

        Step 1: MD5 去重
        Step 2: 写入 SQLite messages 表
        Step 3: 写入 SQLite long_term_memories 表
        Step 4: 向量化写入 ChromaDB

        Returns:
            message_id，MD5 重复返回 None
        """
        if not memory_text.strip():
            return None

        # Step 1: MD5 精确去重（防线 1）
        if self.check_duplicate_md5(memory_text):
            return None

        # Step 2: 写入 SQLite Source of Truth
        message_id = self._db.insert_message(
            content=memory_text,
            role="system",
            source="memory_extraction",
            thread_id=thread_id,
        )

        # Step 3: 写入 long_term_memories 元数据
        self._db.insert_long_term_memory(
            memory_id=message_id,
            content=memory_text,
            memory_type=memory_type,
            importance=importance,
            confidence=confidence,
        )

        # Step 4: 向量化 + 写入 ChromaDB
        try:
            embedding = self._embedder.embed([memory_text])[0]
            self._memory_collection.add(
                embeddings=[embedding],
                documents=[memory_text],
                metadatas=[{
                    "message_id": message_id,
                    "memory_type": memory_type,
                    "importance": importance,
                    "confidence": confidence,
                    "thread_id": thread_id,
                    "timestamp": _now(),
                }],
                ids=[message_id],
            )
            self._db.mark_embedding_done(message_id)
            logger.info("recall_store new memory: %s [%s] -> %s", message_id, memory_type, memory_text[:80])
        except Exception as e:
            logger.warning("recall_store ChromaDB memory write failed for %s: %s, pending retry", message_id, e)

        return message_id

    # ================================================================
    #  检索（底层向量检索，供 HybridRetriever 调用）
    # ================================================================

    def search_memories_vector(self, query_embedding: list[float],
                                top_k: int = 5) -> list[dict]:
        """在长期记忆 ChromaDB 集合中做纯向量检索。"""
        if self._memory_collection is None or self._memory_collection.count() == 0:
            return []
        try:
            results = self._memory_collection.query(
                query_embeddings=[query_embedding],
                n_results=min(top_k, self._memory_collection.count()),
            )
            return self._format_results(results)
        except Exception as e:
            logger.warning("recall_store memory vector search failed: %s", e)
            return []

    def search_recall_vector(self, query_embedding: list[float],
                              top_k: int = 3) -> list[dict]:
        """在 Recall Storage ChromaDB 集合中做纯向量检索。"""
        if self._recall_collection is None or self._recall_collection.count() == 0:
            return []
        try:
            results = self._recall_collection.query(
                query_embeddings=[query_embedding],
                n_results=min(top_k, self._recall_collection.count()),
            )
            return self._format_results(results)
        except Exception as e:
            logger.warning("recall_store recall vector search failed: %s", e)
            return []

    # ---- 向后兼容接口 ----

    def search_memories(self, query: str, top_k: int = 5) -> list[dict]:
        """在长期记忆中语义搜索（向后兼容）。"""
        if self._memory_collection is None or self._memory_collection.count() == 0:
            return []
        embedding = self._embedder.embed([query])[0]
        return self.search_memories_vector(embedding, top_k)

    def search_recall_messages(self, query: str, top_k: int = 3) -> list[dict]:
        """在 Recall Storage 中语义搜索（向后兼容）。"""
        if self._recall_collection is None or self._recall_collection.count() == 0:
            return []
        embedding = self._embedder.embed([query])[0]
        return self.search_recall_vector(embedding, top_k)

    # ================================================================
    #  辅助
    # ================================================================

    def _format_results(self, results) -> list[dict]:
        """将 ChromaDB 查询结果转为统一格式。"""
        docs = []
        if results and results.get("documents") and results["documents"][0]:
            for i, content in enumerate(results["documents"][0]):
                meta = results.get("metadatas", [[{}]])[0][i] if results.get("metadatas") else {}
                distance = results.get("distances", [[1.0]])[0][i] if results.get("distances") else 1.0
                doc_id = results.get("ids", [[""]])[0][i] if results.get("ids") else ""
                docs.append({
                    "message_id": meta.get("message_id", doc_id),
                    "content": content,
                    "metadata": meta,
                    "score": 1.0 - distance,
                })
        return docs

    # ================================================================
    #  健康检查 & 统计
    # ================================================================

    def is_healthy(self) -> bool:
        """检查双存储是否可正常访问。"""
        try:
            self._recall_collection.count()
            self._memory_collection.count()
            self._db.get_stats()
            return True
        except Exception:
            return False

    @property
    def stats(self) -> dict:
        """返回双存储统计信息。"""
        db_stats = self._db.get_stats()
        return {
            "recall_messages_count": (
                self._recall_collection.count() if self._recall_collection else 0
            ),
            "long_term_memories_count": (
                self._memory_collection.count() if self._memory_collection else 0
            ),
            **db_stats,
        }

    # ================================================================
    #  全量重建 ChromaDB 向量索引（最终一致性兜底）
    # ================================================================

    def rebuild_chroma_index(self, batch_size: int = 100) -> dict:
        """从 SQLite 全量重建 ChromaDB 向量索引。

        ChromaDB 只是语义索引，SQLite 才是 Source of Truth。当 ChromaDB 损坏、
        长期断写、或补偿重试持续失败时，从 SQLite 完整重算索引。

        步骤:
        1. 清空 recall_messages / long_term_memories 两个集合。
        2. 读取 messages 表所有原文，重新 embed。
        3. 按 source 写回对应集合，并补全长期记忆 metadata。
        4. 全部写入后标记 pending_embedding=0。

        Returns:
            {"recall": int, "memories": int}
        """
        self._clear_collection(self._recall_collection)
        self._clear_collection(self._memory_collection)

        messages = self._db.get_all_messages()
        recall_msgs = [m for m in messages if m.get("source") != "memory_extraction"]
        memory_msgs = [m for m in messages if m.get("source") == "memory_extraction"]

        recall_ok = self._reindex_batch(recall_msgs, self._recall_collection, "recall", batch_size)
        memory_ok = self._reindex_batch(memory_msgs, self._memory_collection, "memory", batch_size)

        logger.info(
            "rebuild_chroma_index done: recall=%d, memories=%d, total=%d",
            recall_ok, memory_ok, len(messages),
        )
        return {"recall": recall_ok, "memories": memory_ok}

    @staticmethod
    def _clear_collection(collection) -> None:
        """清空一个 ChromaDB 集合（保留集合本身）。"""
        if collection is None:
            return
        try:
            existing = collection.get()
            if existing and existing.get("ids"):
                collection.delete(ids=existing["ids"])
        except Exception as e:
            logger.warning("rebuild_chroma_index: clear collection failed: %s", e)

    def _reindex_batch(self, msgs: list[dict], collection, kind: str,
                       batch_size: int) -> int:
        """把一批消息重新 embed 并写入集合，返回成功条数。"""
        if not msgs or collection is None:
            return 0
        ok = 0
        for i in range(0, len(msgs), batch_size):
            batch = msgs[i:i + batch_size]
            try:
                embeddings = self._embedder.embed([m["content"] for m in batch])
                collection.add(
                    embeddings=embeddings,
                    documents=[m["content"] for m in batch],
                    metadatas=[self._build_index_metadata(m, kind) for m in batch],
                    ids=[m["message_id"] for m in batch],
                )
                for m in batch:
                    self._db.mark_embedding_done(m["message_id"])
                ok += len(batch)
            except Exception as e:
                logger.error(
                    "rebuild_chroma_index: %s batch %d failed: %s", kind, i, e,
                )
        return ok

    def _build_index_metadata(self, m: dict, kind: str) -> dict:
        """构建写入 ChromaDB 的 metadata（与正常写入对齐）。"""
        if kind == "memory":
            mem = self._db.get_long_term_memory(m["message_id"])
            return {
                "message_id": m["message_id"],
                "memory_type": mem.get("memory_type", "Entity") if mem else "Entity",
                "importance": mem.get("importance", 0.5) if mem else 0.5,
                "confidence": mem.get("confidence", 0.5) if mem else 0.5,
                "timestamp": m.get("timestamp", ""),
            }
        return {
            "message_id": m["message_id"],
            "role": m.get("role", ""),
            "thread_id": m.get("thread_id", ""),
            "timestamp": m.get("timestamp", ""),
        }

    # ================================================================
    #  数据清空（开发阶段）
    # ================================================================

    def delete_by_thread(self, thread_id: str) -> int:
        """删除 ChromaDB 两个集合中 thread_id 匹配的向量文档。

        与 memory_db.delete_thread_data 配合：SQLite 侧删原文，本方法删向量索引。
        返回成功清理的集合数（best effort，单个集合失败不影响另一个）。
        """
        if not thread_id:
            return 0
        cleaned = 0
        for collection in (self._recall_collection, self._memory_collection):
            if collection is None:
                continue
            try:
                collection.delete(where={"thread_id": thread_id})
                cleaned += 1
            except Exception as e:
                logger.warning(
                    "recall_store delete_by_thread failed for %s (%s): %s",
                    thread_id, getattr(collection, "name", "?"), e,
                )
        return cleaned

    def clear_all(self):
        """清空所有数据（开发阶段使用）。"""
        try:
            if self._recall_collection:
                all_recall = self._recall_collection.get()
                if all_recall and all_recall.get("ids"):
                    self._recall_collection.delete(ids=all_recall["ids"])

            if self._memory_collection:
                all_mem = self._memory_collection.get()
                if all_mem and all_mem.get("ids"):
                    self._memory_collection.delete(ids=all_mem["ids"])

            import sqlite3
            conn = sqlite3.connect(str(self._db._db_path))
            try:
                tables = [
                    "messages_fts", "messages", "long_term_memories",
                    "recall_messages_meta", "memory_changelog",
                    "core_memory_blocks", "core_memory_snapshots",
                    "extraction_state",
                ]
                for table in tables:
                    try:
                        conn.execute(f"DELETE FROM {table}")
                    except Exception:
                        pass
                conn.commit()
            finally:
                conn.close()

            logger.info("RecallStore: all data cleared (dev mode)")
        except Exception as e:
            logger.error("RecallStore clear_all failed: %s", e)
