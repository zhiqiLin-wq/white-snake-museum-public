"""ChromaDB 持久化向量存储。"""
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class VectorStore:
    """ChromaDB 封装：建库、查询、健康检查。"""

    def __init__(self, persist_dir: Path, embedder, bm25_retriever=None):
        self.persist_dir = persist_dir
        self.embedder = embedder
        self.bm25_retriever = bm25_retriever  # CU-40: BM25 同步
        self._collection = None
        self._client = None
        self._init_chroma()

    def _init_chroma(self):
        try:
            import chromadb
            from chromadb.config import Settings as ChromaSettings
            self.persist_dir.mkdir(parents=True, exist_ok=True)

            # anonymized_telemetry=False: chromadb 与 posthog 版本不兼容导致每次
            # 启动/查询刷 "Failed to send telemetry event" 错误日志，关闭遥测
            _chroma_settings = ChromaSettings(anonymized_telemetry=False)

            # 尝试打开已有数据库；先重试一次，确认非瞬时错误后才清理
            try:
                self._client = chromadb.PersistentClient(path=str(self.persist_dir), settings=_chroma_settings)
            except Exception as client_err:
                err_msg = str(client_err)
                logger.warning(f"ChromaDB PersistentClient 首次打开失败 ({err_msg[:120]})，2 秒后重试...")
                import time as _time
                _time.sleep(2)
                try:
                    self._client = chromadb.PersistentClient(path=str(self.persist_dir), settings=_chroma_settings)
                except Exception as retry_err:
                    if "Component not running" in str(retry_err) or "lock" in str(retry_err).lower():
                        # 并发/锁类问题：绝不能删库，向上抛出由健康检查处理
                        logger.error(f"ChromaDB 重试仍失败且疑似并发问题，保留数据: {str(retry_err)[:150]}")
                        raise
                    logger.warning(
                        f"ChromaDB PersistentClient 重试仍失败 ({str(retry_err)[:120]})，"
                        f"判定为数据损坏，清理旧数据后重建..."
                    )
                    self._reset_persist_dir()
                    self._client = chromadb.PersistentClient(path=str(self.persist_dir), settings=_chroma_settings)

            # 尝试获取已有集合；缺失则创建。
            # 注意：不再在此处 rmtree 清理数据目录——并发打开（如另一进程持有锁）
            # 也会导致瞬时错误，误删会造成整库数据丢失（2026-09-01 事故根因）。
            try:
                self._collection = self._client.get_collection(name="literature_chunks")
                logger.info(f"ChromaDB 就绪 (已有): {self._collection.count()} 个文档")
            except Exception as get_err:
                err_msg = str(get_err)
                transient = ("Component not running" in err_msg or "lock" in err_msg.lower()
                             or "already exists" in err_msg.lower())
                if transient:
                    # 瞬时/并发错误：绝不清理数据；集合置空交给健康检查/重建流程处理
                    logger.error(
                        f"ChromaDB 集合打开失败（疑似瞬时/并发问题），保留数据、不重建: "
                        f"{err_msg[:150]}"
                    )
                    self._collection = None
                else:
                    logger.warning(f"ChromaDB 集合缺失或不可读 ({err_msg[:120]})，创建空集合等待重建...")
                    try:
                        self._client.delete_collection("literature_chunks")
                    except Exception:
                        pass
                    self._collection = self._client.create_collection(
                        name="literature_chunks",
                        metadata={"hnsw:space": "cosine"},
                    )
                    logger.info("ChromaDB 就绪 (新建空集合): 0 个文档")
        except Exception as e:
            import traceback
            logger.error(f"ChromaDB 初始化失败: {e}\n{traceback.format_exc()}")
            self._collection = None

    def _reset_persist_dir(self):
        """清理 ChromaDB 持久化目录中所有数据。"""
        import shutil
        for item in self.persist_dir.iterdir():
            try:
                if item.is_dir():
                    shutil.rmtree(item)
                else:
                    item.unlink()
            except Exception:
                pass

    def is_healthy(self) -> bool:
        """健康判定：集合存在且行数 > 0。空集合视为不健康（触发启动自动重建）。"""
        if self._collection is None:
            return False
        try:
            return self._collection.count() > 0
        except Exception:
            return False

    def collection_count(self) -> int:
        """返回集合中的文档数量，集合不可用时返回 0。"""
        if self._collection is None:
            return 0
        try:
            return self._collection.count()
        except Exception:
            return 0

    def get_all_documents(self, where: dict | None = None) -> list[dict]:
        """取全部（或按 metadata 过滤的）文档，不走向量检索。

        用于字面统计类操作（如 count_occurrences 的全文子串计数）——
        这类操作需要遍历全库 content，语义检索的 top_k 召回无法覆盖。
        返回 [{content, chunk_id, metadata}, ...]；集合不可用返回 []。
        """
        if self._collection is None:
            return []
        try:
            kwargs: dict = {}
            if where:
                kwargs["where"] = where
            results = self._collection.get(**kwargs)
        except Exception as e:
            logger.error(f"get_all_documents 失败: {e}")
            return []
        docs = []
        documents = results.get("documents") or []
        metadatas = results.get("metadatas") or []
        ids = results.get("ids") or []
        for i, content in enumerate(documents):
            docs.append({
                "content": content or "",
                "chunk_id": ids[i] if i < len(ids) else f"chunk_{i}",
                "metadata": metadatas[i] if i < len(metadatas) else {},
            })
        return docs

    @staticmethod
    def _clean_metadata(meta: dict) -> dict:
        """清理 metadata，确保值都是 chromadb 允许的类型 (str/int/float/bool)。

        chromadb 0.5.3 拒绝 None 值和复杂类型：None 转空字符串、其他转 str。
        语料中研究文献的 dynasty 等字段为 None，直接 add 会报 ValueError。
        """
        clean = {}
        for k, v in (meta or {}).items():
            if v is None:
                clean[k] = ""
            elif isinstance(v, (str, int, float, bool)):
                clean[k] = v
            else:
                clean[k] = str(v)
        return clean

    def ensure_collection(self):
        """确保集合存在（不写入数据）；返回 self._collection。"""
        if self._collection is None:
            self._init_chroma()
        if self._collection is None:
            try:
                self._collection = self._client.get_or_create_collection(
                    name="literature_chunks",
                    metadata={"hnsw:space": "cosine"},
                )
            except Exception as e:
                logger.error(f"ensure_collection 失败: {e}")
                raise
        return self._collection

    def upsert_chunks(self, documents: list, embed_batch_size: int = 4,
                      add_batch_size: int = 16) -> int:
        """增量入库：小批次 embed + chroma upsert（不删库）。

        chunk_id 由 chunker 按章节顺序确定性生成，重跑时相同 id 会被覆盖，
        因此本方法可安全用于"边分块边入库"的可续跑重建。
        """
        if not self.embedder or not self.embedder.is_ready:
            raise RuntimeError("Embedding 模型未就绪，无法构建索引")
        if self._collection is None:
            self._init_chroma()
        if self._collection is None:
            raise RuntimeError("向量集合不可用")

        texts = [d.content for d in documents]
        metadatas = [self._clean_metadata(d.metadata) for d in documents]
        ids = [d.metadata.get("chunk_id", f"chunk_{i}") for i, d in enumerate(documents)]

        embeddings = []
        for i in range(0, len(texts), embed_batch_size):
            embeddings.extend(self.embedder.embed(texts[i:i + embed_batch_size]))

        # 分批 upsert，避免 Windows + ChromaDB hnswlib 大批量写入时 segfault
        for j in range(0, len(embeddings), add_batch_size):
            end = min(j + add_batch_size, len(embeddings))
            self._collection.upsert(
                embeddings=embeddings[j:end],
                documents=texts[j:end],
                metadatas=metadatas[j:end],
                ids=ids[j:end],
            )
        return len(texts)

    def build_from_chunks(self, documents: list):
        """全量重建索引。"""
        if not self.embedder.is_ready:
            raise RuntimeError("Embedding 模型未就绪，无法构建索引")

        if self._collection is None:
            self._init_chroma()

        texts = [d.content for d in documents]
        metadatas = [self._clean_metadata(d.metadata) for d in documents]
        ids = [d.metadata.get("chunk_id", f"chunk_{i}") for i, d in enumerate(documents)]

        # 批量 embed（避免 OOM，每批 32 个）
        batch_size = 32
        embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            embeddings.extend(self.embedder.embed(batch))

        # 清空并重建
        try:
            self._client.delete_collection("literature_chunks")
        except Exception:
            pass
        self._collection = self._client.create_collection(
            name="literature_chunks",
            metadata={"hnsw:space": "cosine"},
        )
        # 分批添加，避免 ChromaDB 在 Windows 上一次添加过多向量时 segfault (crash)
        # batch_size 从 80 降到 20 —— Windows + ChromaDB 0.5.x 的 hnswlib 在大批量时不稳定
        add_batch_size = 20
        for j in range(0, len(embeddings), add_batch_size):
            end = min(j + add_batch_size, len(embeddings))
            self._collection.add(
                embeddings=embeddings[j:end],
                documents=texts[j:end],
                metadatas=metadatas[j:end],
                ids=ids[j:end],
            )
        logger.info(f"索引构建完成: {len(texts)} 个文本块")

        # CU-40: BM25 索引同步重建 (U04/U16: 统一逻辑 + chunk_id)
        self._sync_bm25(documents)

    def query(self, query_embedding: list[float], top_k: int = 5, where: dict | None = None) -> list[dict]:
        """语义检索。"""
        if self._collection is None:
            raise RuntimeError("向量存储未就绪")

        query_kwargs: dict = {
            "query_embeddings": [query_embedding],
            "n_results": max(1, min(top_k, self._collection.count())),
        }
        if where:
            query_kwargs["where"] = where

        # 空集合（索引未就绪/重建中）直接返回空结果，避免 chroma 抛
        # "Number of requested results 0" 导致整路检索崩溃
        try:
            if self._collection.count() == 0:
                logger.warning("vector_store.query: 集合为空（索引可能正在重建），返回空结果")
                return []
            results = self._collection.query(**query_kwargs)
        except Exception as e:
            logger.error(f"vector_store.query 失败: {e}")
            return []

        docs = []
        if results and results["documents"] and results["documents"][0]:
            for i, content in enumerate(results["documents"][0]):
                metadata = results["metadatas"][0][i] if results.get("metadatas") else {}
                distance = results["distances"][0][i] if results.get("distances") else 1.0
                chunk_id = results["ids"][0][i] if results.get("ids") else f"chunk_{i}"
                docs.append({
                    "content": content,
                    "chunk_id": chunk_id,
                    "metadata": metadata,
                    "score": 1.0 - distance,  # 余弦距离 → 相似度
                })
        return docs

    def rebuild(self) -> int:
        """重建索引（由 /rebuild-index 调用）。

        策略：先清空集合，再"逐章分块 → 立即 upsert 落库"。
        即使进程在某一章崩溃，已完成章节的向量仍保留在集合中（旧实现会全部丢失）；
        重跑本方法时 chunk_id 确定性生成，同名块幂等覆盖。
        """
        import gc
        from .chunker import Chunker
        from .config import rag_config

        # 清空旧集合（显式重建才删除；启动自动重建走 upsert 增量路径不删）
        try:
            self._client.delete_collection("literature_chunks")
        except Exception:
            pass
        self._collection = self._client.create_collection(
            name="literature_chunks",
            metadata={"hnsw:space": "cosine"},
        )

        chunker = Chunker(
            data_dir=rag_config.data_dir_path,
            embedder=self.embedder,
            embed_batch_size=4,  # 小批次压低峰值内存（批 16 在低空闲内存机器上原生崩溃 0xC0000005，批 4 实测稳定）
        )

        def _sink(chapter_number, chapter_title, dynasty, ch_chunks):
            n = self.upsert_chunks(ch_chunks)
            chunker._embedding_cache.clear()
            gc.collect()
            logger.info(
                f"[rebuild] 第{chapter_number}章入库 {n} 块，"
                f"集合累计 {self.collection_count()} 行"
            )

        chunks = chunker.chunk_all(chapter_sink=_sink)
        self._sync_bm25(chunks)

        return len(chunks)

    # ========================================================================
    # U16: 统一 BM25 同步逻辑
    # ========================================================================
    def _sync_bm25(self, documents: list):
        """将分块结果同步到 BM25 索引（如果 BM25 已配置）。U04: 包含 chunk_id。"""
        if self.bm25_retriever is None:
            return
        chunks_as_dicts = [
            {
                "content": doc.content,
                "metadata": doc.metadata,
                "chunk_id": doc.metadata.get("chunk_id", ""),
            }
            for doc in documents
        ]
        self.bm25_retriever.rebuild(chunks_as_dicts)
        logger.info(f"BM25 索引同步完成: {self.bm25_retriever._total_docs} 文档")
