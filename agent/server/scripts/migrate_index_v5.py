"""索引迁移脚本 —— 旧 schema → 新 schema (v5 统一 metadata 规范)。U22

用法:
    cd agent && python server/scripts/migrate_index_v5.py

行为:
    1. 备份旧 ChromaDB 数据目录（重命名为 chroma_db_backup_<timestamp>）
    2. 检查旧索引的 metadata schema 版本
    3. 运行 chunk_all() 重新分块（使用新的 metadata 规范）
    4. 重建 ChromaDB + BM25 索引
    5. 运行 validate_chunk_metadata() 验证
    6. 输出迁移报告

注意:
    - 迁移过程需要 ~120s（取决于 embedding 模型速度）
    - 旧的 chroma_db 目录会被备份而非直接删除
    - 如果迁移失败，可手动恢复备份目录
"""
import os
import sys
import shutil
import time
import logging
from datetime import datetime
from pathlib import Path

# 确保项目路径在 sys.path 中
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
_AGENT_DIR = _PROJECT_ROOT / "agent"
sys.path.insert(0, str(_AGENT_DIR))
sys.path.insert(0, str(_PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("migrate_index_v5")


def main():
    logger.info("=" * 60)
    logger.info("ChromaDB 索引迁移: 旧 schema → v5 新 schema")
    logger.info("=" * 60)

    from server.rag.config import rag_config
    from server.rag.chunker import Chunker, validate_chunk_metadata
    from server.rag.embedder import Embedder
    from server.rag.vector_store import VectorStore

    data_dir = rag_config.data_dir_path
    chroma_dir = data_dir / rag_config.chroma_persist_dir

    # ---- Step 1: 备份旧数据 ----
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_dir = data_dir / f"chroma_db_backup_{timestamp}"
    if chroma_dir.exists():
        logger.info(f"Step 1: 备份旧 ChromaDB 数据 → {backup_dir}")
        shutil.copytree(chroma_dir, backup_dir)
        logger.info(f"  备份完成: {backup_dir}")
    else:
        logger.info("Step 1: 没有旧 ChromaDB 数据需要备份")

    # ---- Step 2: 检查旧索引 ----
    logger.info("Step 2: 检查旧索引 schema...")
    needs_migration = True
    if chroma_dir.exists():
        try:
            import chromadb
            client = chromadb.PersistentClient(path=str(chroma_dir))
            try:
                collection = client.get_collection(name="literature_chunks")
                sample = collection.get(limit=1)
                if sample and sample.get("metadatas") and sample["metadatas"]:
                    first_meta = sample["metadatas"][0]
                    if "chunk_id" in first_meta:
                        logger.info("  已有 chunk_id 字段，索引可能已经是新 schema")
                        logger.info("  仍将执行重建以确保数据一致性")
                        needs_migration = True
                    else:
                        logger.info("  检测到旧 schema（无 chunk_id），需要迁移")
            except Exception:
                logger.info("  无法读取 collection，将重新构建")
        except Exception as e:
            logger.warning(f"  检查失败: {e}，将重新构建")

    # ---- Step 3: 加载 Embedding 模型 ----
    logger.info("Step 3: 加载 Embedding 模型...")
    embedder = Embedder(
        model_name=rag_config.embedding_model,
        local_path=rag_config.embedding_local_path or None,
    )
    embedder.start_loading()

    waited = 0
    while not embedder.is_ready and waited < 300:
        time.sleep(2)
        waited += 2
        if waited % 10 == 0:
            logger.info(f"  等待中... ({waited}s)")

    if not embedder.is_ready:
        logger.error(f"Embedding 模型在 300s 内未就绪: {embedder.load_error}")
        logger.info(f"若需回滚，请恢复备份目录: {backup_dir}")
        return 1

    logger.info(f"Embedding 模型就绪 (等待了 {waited}s)")

    # ---- Step 4: 重新分块 + 验证 ----
    logger.info("Step 4: 重新分块 (chunk_all)...")
    chunker = Chunker(data_dir=data_dir, embedder=embedder)
    # 迁移脚本的目的就是按新 metadata 规范重新分块，必须绕过冻结快照
    all_chunks = chunker.chunk_all(use_snapshot=False)

    logger.info(f"  分块完成: {len(all_chunks)} 个 chunk")
    validation_errors = validate_chunk_metadata(all_chunks)
    if validation_errors:
        logger.warning(f"  发现 {len(validation_errors)} 个 schema 问题:")
        for err in validation_errors[:10]:
            logger.warning(f"    - {err}")
        logger.error("迁移中止: metadata schema 验证未通过")
        logger.info(f"旧索引备份位于: {backup_dir}")
        return 1
    else:
        logger.info("  metadata schema 验证通过")

    # ---- Step 5: 重建索引 ----
    logger.info("Step 5: 重建 ChromaDB + BM25 索引...")
    t0 = time.time()
    vector_store = VectorStore(
        persist_dir=chroma_dir,
        embedder=embedder,
    )
    vector_store.build_from_chunks(all_chunks)
    elapsed = time.time() - t0
    logger.info(f"  索引重建完成 ({elapsed:.1f}s)")

    # ---- Step 6: 清理旧备份 ----
    logger.info("Step 6: 清理旧备份（保留最近3个）...")
    backup_pattern = "chroma_db_backup_"
    backups = sorted(
        [d for d in data_dir.iterdir() if d.name.startswith(backup_pattern)],
        key=lambda d: d.name,
    )
    while len(backups) > 3:
        oldest = backups.pop(0)
        shutil.rmtree(oldest)
        logger.info(f"  清理旧备份: {oldest}")

    # ---- 迁移报告 ----
    logger.info("=" * 60)
    logger.info("迁移完成!")
    logger.info(f"  - 新索引 chunk 数: {len(all_chunks)}")
    logger.info(f"  - 备份目录: {backup_dir}")
    logger.info(f"  - 总耗时: {elapsed + waited:.1f}s")
    logger.info("=" * 60)
    logger.info("请启动服务后访问 /health 端点确认索引就绪")
    return 0


if __name__ == "__main__":
    sys.exit(main())
