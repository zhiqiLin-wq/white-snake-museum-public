"""体裁感知分块系统 —— 独立测试脚本。

运行方式:
    1. PyCharm 中右键此文件 → Run 'test_chunking'
    2. 或在终端: cd white-snake-museum-public && python test_chunking.py

注意:
    - 首次运行需要加载 Embedding 模型（已缓存则秒过）
    - 文言文/话本/章回体/研究文献 四种策略需要 embedding 模型
    - 拟话本/戏曲/剧本/弹词 四种策略不需要 embedding 模型
    - 若只想快速验证结构分块（不需要 embedding），设置 GENRE_CHUNKING_ENABLED=false

输出:
    - debug_output/ 目录下生成按体裁分类的调试文件
    - 每个文件包含: 体裁名 / 策略 / 参数 / 总chunk数 / 每个chunk字数 / 元数据 / 完整文本
"""

import os
import sys
import time
import logging

# ---- 设置环境变量（必须在 import config 之前） ----
os.environ.setdefault("CHUNK_DEBUG_ENABLED", "true")
os.environ.setdefault("CHUNK_DEBUG_OUTPUT_DIR", "./debug_output")
# 如果想快速测试结构分块而不加载 embedding 模型，取消下面这行的注释:
# os.environ["GENRE_CHUNKING_ENABLED"] = "false"

# 确保项目路径在 sys.path 中（多种方式兜底）
_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
_AGENT_DIR = os.path.join(_PROJECT_ROOT, "agent")
sys.path.insert(0, _AGENT_DIR)
sys.path.insert(0, _PROJECT_ROOT)

# ---- 日志配置 ----
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("test_chunking")

# ---- 导入项目模块 ----
# PyCharm 用户需确保 agent/ 已标记为 Sources Root:
#   右键 agent/ 目录 → Mark Directory as → Sources Root
_HAS_SOURCE_ROOT = False
try:
    from server.config import settings
    from server.rag.chunker import Chunker, Genre
    from server.rag.embedder import Embedder
    _HAS_SOURCE_ROOT = True
except ImportError:
    # 如果 agent/ 不是 Sources Root，手动从文件路径加载
    import importlib.util
    logger.info("通过文件路径加载模块（PyCharm 中可右键 agent/ → Mark Directory as → Sources Root 解决）")
    # config
    spec = importlib.util.spec_from_file_location("config", os.path.join(_AGENT_DIR, "server", "config.py"))
    cfg = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cfg)
    settings = cfg.settings
    # chunker
    spec = importlib.util.spec_from_file_location("chunker", os.path.join(_AGENT_DIR, "server", "rag", "chunker.py"))
    chk = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(chk)
    Chunker = chk.Chunker
    Genre = chk.Genre
    # embedder
    spec = importlib.util.spec_from_file_location("embedder", os.path.join(_AGENT_DIR, "server", "rag", "embedder.py"))
    emb = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(emb)
    Embedder = emb.Embedder
from server.rag.chunker import Chunker, Genre
from server.rag.embedder import Embedder


def main():
    logger.info("=" * 60)
    logger.info("体裁感知分块系统 —— 测试运行")
    logger.info("=" * 60)
    logger.info(f"数据目录: {settings.data_dir_path}")
    logger.info(f"体裁分块: {'启用' if settings.genre_chunking_enabled else '禁用 (纯段落分块)'}")
    logger.info(f"调试输出: {'启用' if settings.chunk_debug_enabled else '禁用'}")
    logger.info(f"调试目录: {settings.chunk_debug_output_dir}")

    # ---- 1. 检查数据文件 ----
    main_txt = settings.data_dir_path / "白蛇传文献选集.txt"
    research_txt = settings.data_dir_path / "文本景观建构研究文献.txt"
    logger.info(f"主文献: {'存在' if main_txt.exists() else '缺失'} ({main_txt})")
    logger.info(f"研究文献: {'存在' if research_txt.exists() else '缺失'} ({research_txt})")

    if not main_txt.exists() and not research_txt.exists():
        logger.error("没有找到任何数据文件，退出")
        return

    # ---- 2. 加载 Embedding 模型 ----
    embedder = None
    if settings.genre_chunking_enabled:
        logger.info(f"加载 Embedding 模型: {settings.embedding_model}")
        embedder = Embedder(
            model_name=settings.embedding_model,
            local_path=settings.embedding_local_path or None,
        )
        embedder.start_loading()
        logger.info("等待 Embedding 模型就绪...")

        waited = 0
        while not embedder.is_ready and waited < 300:
            time.sleep(2)
            waited += 2
            if waited % 10 == 0:
                logger.info(f"  等待中... ({waited}s)")

        if not embedder.is_ready:
            logger.error(f"Embedding 模型在 300s 内未就绪: {embedder.load_error}")
            logger.warning("将禁用体裁感知分块，使用纯段落分块")
            os.environ["GENRE_CHUNKING_ENABLED"] = "false"
            embedder = None
        else:
            logger.info(f"Embedding 模型就绪 (等待了 {waited}s)")

    # ---- 3. 创建 Chunker 并运行分块 ----
    logger.info("=" * 60)
    logger.info("开始分块...")
    t0 = time.time()

    chunker = Chunker(
        data_dir=settings.data_dir_path,
        embedder=embedder,
    )

    all_chunks = chunker.chunk_all()
    elapsed = time.time() - t0

    # ---- U00/U19: 验证 metadata schema + chunk_id 唯一性 ----
    from server.rag.chunker import validate_chunk_metadata
    validation_errors = validate_chunk_metadata(all_chunks)
    if validation_errors:
        logger.warning(f"[VALIDATE] 发现 {len(validation_errors)} 个 metadata schema 问题:")
        for err in validation_errors[:20]:
            logger.warning(f"  - {err}")
        if len(validation_errors) > 20:
            logger.warning(f"  ... 还有 {len(validation_errors) - 20} 个问题")
    else:
        logger.info("[VALIDATE] 所有 chunk metadata schema 验证通过")

    # ---- 4. 输出统计 ----
    logger.info("=" * 60)
    logger.info(f"分块完成! 总耗时: {elapsed:.1f}s")
    logger.info(f"总 chunk 数: {len(all_chunks)}")

    # chunk_id 统计
    chunk_ids = [doc.metadata.get("chunk_id", "") for doc in all_chunks]
    unique_ids = len(set(c for c in chunk_ids if c))
    logger.info(f"唯一 chunk_id 数: {unique_ids}")
    if unique_ids != len(all_chunks):
        logger.warning(f"chunk_id 重复! 总数={len(all_chunks)}, 唯一={unique_ids}")

    # 按体裁统计
    genre_counts: dict[str, int] = {}
    genre_total_chars: dict[str, int] = {}
    for doc in all_chunks:
        g = doc.metadata.get("genre", "unknown")
        genre_counts[g] = genre_counts.get(g, 0) + 1
        genre_total_chars[g] = genre_total_chars.get(g, 0) + len(doc.content)

    logger.info("")
    logger.info(f"{'体裁':<25} {'chunk数':>8} {'总字数':>10} {'平均字数':>8}")
    logger.info("-" * 55)
    for g in sorted(genre_counts.keys()):
        cnt = genre_counts[g]
        total_chars = genre_total_chars[g]
        avg = total_chars // cnt if cnt > 0 else 0
        logger.info(f"{g:<25} {cnt:>8} {total_chars:>10} {avg:>8}")

    # 按策略统计
    strategy_counts: dict[str, int] = {}
    for doc in all_chunks:
        s = doc.metadata.get("strategy", "unknown")
        strategy_counts[s] = strategy_counts.get(s, 0) + 1
    logger.info("")
    logger.info(f"{'策略':<35} {'chunk数':>8}")
    logger.info("-" * 45)
    for s in sorted(strategy_counts.keys()):
        logger.info(f"{s:<35} {strategy_counts[s]:>8}")

    # ---- 5. 调试文件位置 ----
    import os as _os
    debug_dir = _os.path.abspath(settings.chunk_debug_output_dir)
    logger.info("")
    logger.info("=" * 60)
    logger.info(f"调试文件已输出到: {debug_dir}")
    if _os.path.isdir(debug_dir):
        files = sorted(_os.listdir(debug_dir))
        for f in files:
            fpath = _os.path.join(debug_dir, f)
            fsize = _os.path.getsize(fpath)
            logger.info(f"  {f} ({fsize:,} bytes)")
    else:
        logger.warning(f"调试目录不存在: {debug_dir}")
        logger.warning("请确认 CHUNK_DEBUG_ENABLED=true 已设置")


if __name__ == "__main__":
    main()
