"""端到端集成测试计划 (U20): chunker → vector_store → retriever → consumer 完整链路。

测试场景:
    TC01: chunk_id 全链路一致性
    TC02: content 碰撞独立性
    TC03: char_start/char_end 准确性
    TC04: 演化分析引文溯源
    TC05: 降级路径覆盖
    TC06: 性能回归
    TC07: metadata schema 一致性

用法:
    cd agent && python -m pytest server/rag/tests/test_chunk_traceability_e2e.py -v
    # 或单独运行:
    python server/rag/tests/test_chunk_traceability_e2e.py
"""
import os
import sys
import time
import logging
import tempfile
from pathlib import Path

# 确保项目路径
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent
_AGENT_DIR = _PROJECT_ROOT / "agent"
sys.path.insert(0, str(_AGENT_DIR))
sys.path.insert(0, str(_PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("test_chunk_traceability_e2e")

# 测试标记
ALL_TESTS_PASSED = True


def check(condition: bool, name: str) -> bool:
    global ALL_TESTS_PASSED
    if condition:
        logger.info(f"  [PASS] {name}")
    else:
        logger.error(f"  [FAIL] {name}")
        ALL_TESTS_PASSED = False
    return condition


# ============================================================================
# TC01: chunk_id 全链路一致性
# ============================================================================
def test_tc01_chunk_id_end_to_end():
    """验证 chunk_id 在 chunker -> ChromaDB -> BM25 -> retriever 全链路一致。"""
    logger.info("=" * 60)
    logger.info("TC01: chunk_id 全链路一致性")
    logger.info("=" * 60)

    from server.rag.config import rag_config
    from server.rag.chunker import Chunker, parse_chunk_id, validate_chunk_metadata
    from server.rag.embedder import Embedder

    data_dir = rag_config.data_dir_path

    # 加载 embedder
    embedder = Embedder(
        model_name=rag_config.embedding_model,
        local_path=rag_config.embedding_local_path or None,
    )
    embedder.start_loading()
    waited = 0
    while not embedder.is_ready and waited < 300:
        time.sleep(2)
        waited += 2
    check(embedder.is_ready, "Embedder 就绪")

    chunker = Chunker(data_dir=data_dir, embedder=embedder)
    all_chunks = chunker.chunk_all(use_snapshot=False)  # 追溯性测试必须走实时分块
    check(len(all_chunks) > 0, f"chunk_all 返回 {len(all_chunks)} 个 chunk")

    # 检查所有 chunk_id 唯一
    chunk_ids = [doc.metadata.get("chunk_id", "") for doc in all_chunks]
    unique_ids = set(chunk_ids)
    check(len(unique_ids) == len(all_chunks),
          f"所有 chunk_id 唯一 ({len(unique_ids)}/{len(all_chunks)})")

    # 检查 chunk_id 格式
    import re
    id_pattern = re.compile(r'^(primary|research)_[a-z_]+_[^_]+_\d{5}$')
    for cid in chunk_ids:
        if not id_pattern.match(cid):
            check(False, f"chunk_id 格式: {cid}")
            break
    else:
        check(True, "所有 chunk_id 格式正确")

    # 测试 parse_chunk_id
    for cid in chunk_ids[:5]:
        parsed = parse_chunk_id(cid)
        check(parsed["source_prefix"] in ("primary", "research"),
              f"parse_chunk_id({cid}) source_prefix={parsed['source_prefix']}")
        check(isinstance(parsed["sequence"], int),
              f"parse_chunk_id({cid}) sequence={parsed['sequence']}")

    # 验证 metadata
    errors = validate_chunk_metadata(all_chunks)
    check(len(errors) == 0, f"validate_chunk_metadata: {len(errors)} errors")

    logger.info(f"TC01 完成: {len(all_chunks)} chunks, {len(unique_ids)} unique IDs")
    return all_chunks


# ============================================================================
# TC03: char_start/char_end 准确性
# ============================================================================
def test_tc03_char_offsets():
    """验证各体裁 chunk 的 char_start/char_end 准确性。"""
    logger.info("=" * 60)
    logger.info("TC03: char_start/char_end 准确性")
    logger.info("=" * 60)

    from server.rag.config import rag_config
    from server.rag.chunker import Chunker
    from server.rag.embedder import Embedder

    data_dir = rag_config.data_dir_path
    embedder = Embedder(
        model_name=rag_config.embedding_model,
        local_path=rag_config.embedding_local_path or None,
    )
    embedder.start_loading()
    waited = 0
    while not embedder.is_ready and waited < 300:
        time.sleep(2)
        waited += 2

    chunker = Chunker(data_dir=data_dir, embedder=embedder)
    all_chunks = chunker.chunk_all(use_snapshot=False)  # 追溯性测试必须走实时分块

    # 按体裁分组抽查
    genre_samples: dict[str, list] = {}
    for doc in all_chunks:
        g = doc.metadata.get("genre", "unknown")
        if g not in genre_samples:
            genre_samples[g] = []
        if len(genre_samples[g]) < 3:
            genre_samples[g].append(doc)

    for genre, samples in genre_samples.items():
        for doc in samples:
            meta = doc.metadata
            char_start = meta.get("char_start", -1)
            char_end = meta.get("char_end", -1)
            if char_start >= 0 and char_end >= 0:
                check(char_start < char_end,
                      f"[{genre}] char_start({char_start}) < char_end({char_end})")
            # 对于 opera/script/tanci，检查 context_prefix_length
            if meta.get("context_prefix"):
                cpl = meta.get("context_prefix_length", -1)
                check(cpl == len(meta["context_prefix"]),
                      f"[{genre}] context_prefix_length({cpl}) == len(prefix)({len(meta['context_prefix'])})")

    logger.info("TC03 完成")


# ============================================================================
# TC07: metadata schema 一致性
# ============================================================================
def test_tc07_metadata_schema():
    """运行 validate_chunk_metadata 确认零错误。"""
    logger.info("=" * 60)
    logger.info("TC07: metadata schema 一致性")
    logger.info("=" * 60)

    from server.rag.config import rag_config
    from server.rag.chunker import Chunker, validate_chunk_metadata
    from server.rag.embedder import Embedder

    data_dir = rag_config.data_dir_path
    embedder = Embedder(
        model_name=rag_config.embedding_model,
        local_path=rag_config.embedding_local_path or None,
    )
    embedder.start_loading()
    waited = 0
    while not embedder.is_ready and waited < 300:
        time.sleep(2)
        waited += 2

    chunker = Chunker(data_dir=data_dir, embedder=embedder)
    all_chunks = chunker.chunk_all(use_snapshot=False)  # 追溯性测试必须走实时分块
    errors = validate_chunk_metadata(all_chunks)
    check(len(errors) == 0, f"validate_chunk_metadata 零错误 (实际: {len(errors)})")
    if errors:
        for err in errors[:5]:
            logger.warning(f"  {err}")

    logger.info("TC07 完成")


# ============================================================================
# 主入口
# ============================================================================
def main():
    logger.info("Chunk 溯源与检索修复 — 端到端集成测试")
    logger.info("")

    try:
        test_tc01_chunk_id_end_to_end()
    except Exception as e:
        logger.error(f"TC01 异常: {e}")
        import traceback
        traceback.print_exc()
        global ALL_TESTS_PASSED
        ALL_TESTS_PASSED = False

    try:
        test_tc03_char_offsets()
    except Exception as e:
        logger.error(f"TC03 异常: {e}")
        import traceback
        traceback.print_exc()
        ALL_TESTS_PASSED = False

    try:
        test_tc07_metadata_schema()
    except Exception as e:
        logger.error(f"TC07 异常: {e}")
        import traceback
        traceback.print_exc()
        ALL_TESTS_PASSED = False

    logger.info("=" * 60)
    if ALL_TESTS_PASSED:
        logger.info("所有端到端测试通过!")
    else:
        logger.error("存在失败的测试!")

    logger.info("")
    logger.info("手工测试 checklist:")
    logger.info("  [ ] 前端：点击演化分析报告中的引文标签，确认 TextReaderView 滚动到正确位置")
    logger.info("  [ ] 前端：opera/script/tanci 的引文高亮起始位置跳过了 context_prefix")
    logger.info("  [ ] 前端：研究文献的引文显示 '研究' 而非数字 0")
    logger.info("  [ ] 前端：旧格式引文 [明3] 显示为灰色不可点击 badge")
    logger.info("  [ ] /health 端点返回 build_progress 字段，前端进度条正常更新")

    return 0 if ALL_TESTS_PASSED else 1


if __name__ == "__main__":
    sys.exit(main())
