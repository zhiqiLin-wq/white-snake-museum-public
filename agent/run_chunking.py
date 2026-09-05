"""独立分块测试脚本 -- 不依赖 Web 服务，仅测试 chunker.chunk_all() 并输出 debug 日志。

用法:
    cd agent
    set CHUNK_DEBUG_ENABLED=true
    python run_chunking.py

依赖:
    - .env 文件 (位于 white-snake-museum-public/.env)
    - Embedding 模型缓存 (首次运行会自动下载，约 2-3 分钟)
"""

import os
import sys
import time
import logging
from pathlib import Path

# ---------------------------------------------------------------------------
# 0. 路径设置: 确保 agent/ 在 sys.path 中
# ---------------------------------------------------------------------------
AGENT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(AGENT_DIR))

# 加载父目录的 .env (white-snake-museum-public/.env)
from dotenv import load_dotenv
ENV_FILE = AGENT_DIR.parent / ".env"
if ENV_FILE.exists():
    load_dotenv(ENV_FILE)
    print(f"[ENV] Loaded: {ENV_FILE}")

# ---------------------------------------------------------------------------
# 1. 配置日志
# ---------------------------------------------------------------------------
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL.upper(), logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("run_chunking")

# ---------------------------------------------------------------------------
# 2. 强制启用 debug 输出 (可通过环境变量覆盖)
# ---------------------------------------------------------------------------
if not os.getenv("CHUNK_DEBUG_ENABLED"):
    os.environ["CHUNK_DEBUG_ENABLED"] = "true"
    logger.info("CHUNK_DEBUG_ENABLED 自动设置为 true")

debug_output_dir = os.getenv("CHUNK_DEBUG_OUTPUT_DIR", "./debug_output")
logger.info(f"Debug 输出目录: {debug_output_dir}")

# ---------------------------------------------------------------------------
# 3. 导入依赖
# ---------------------------------------------------------------------------
from server.rag.config import rag_config
from server.rag.embedder import Embedder
from server.rag.chunker import Chunker

# ---------------------------------------------------------------------------
# 4. 初始化 Embedder
# ---------------------------------------------------------------------------
logger.info(f"Embedding 模型: {rag_config.embedding_model}")
if rag_config.embedding_local_path:
    logger.info(f"本地路径: {rag_config.embedding_local_path}")
logger.info(f"HF 镜像: {rag_config.hf_endpoint}")
logger.info(f"缓存目录: {rag_config.sentence_transformers_home}")

embedder = Embedder(
    model_name=rag_config.embedding_model,
    local_path=rag_config.embedding_local_path or None,
)
embedder.start_loading()

# 同步等待模型就绪
logger.info("等待 Embedding 模型加载...")
max_wait = 300  # 最长等待 5 分钟
waited = 0
while not embedder.is_ready and waited < max_wait:
    progress = embedder.progress
    if embedder.load_error:
        logger.error(f"模型加载失败: {embedder.load_error}")
        sys.exit(1)
    logger.info(f"  进度: {progress:.0%} (已等待 {waited}s)")
    time.sleep(5)
    waited += 5

if not embedder.is_ready:
    logger.error(f"模型在 {max_wait}s 内未就绪")
    sys.exit(1)

logger.info(f"模型就绪 (等待了 {waited}s)")

# ---------------------------------------------------------------------------
# 5. 创建 Chunker 并执行分块
# ---------------------------------------------------------------------------
data_dir = rag_config.data_dir_path
logger.info(f"数据目录: {data_dir}")
logger.info(f"  白蛇传文献选集: {'存在' if (data_dir / '白蛇传文献选集.txt').exists() else '缺失'}")
logger.info(f"  文本景观建构研究文献: {'存在' if (data_dir / '文本景观建构研究文献.txt').exists() else '缺失'}")

chunker = Chunker(data_dir=data_dir, embedder=embedder)

logger.info("=" * 60)
logger.info("开始分块...")
start_time = time.time()

try:
    # 分块调试脚本：强制走实时语义分块（不读冻结快照）
    all_chunks = chunker.chunk_all(use_snapshot=False)
    elapsed = time.time() - start_time
    logger.info(f"分块完成: {len(all_chunks)} chunks, 耗时 {elapsed:.1f}s")
except Exception as e:
    logger.error(f"分块失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# ---------------------------------------------------------------------------
# 6. 统计输出
# ---------------------------------------------------------------------------
genre_counts: dict[str, int] = {}
for doc in all_chunks:
    g = doc.metadata.get("genre", "unknown")
    genre_counts[g] = genre_counts.get(g, 0) + 1

print()
print("=" * 60)
print("分块结果统计")
print("=" * 60)
for g, cnt in sorted(genre_counts.items()):
    print(f"  {g}: {cnt} chunks")

# 统计 figure_type
figure_image = sum(1 for doc in all_chunks if doc.metadata.get("figure_type") == "image")
figure_table = sum(1 for doc in all_chunks if doc.metadata.get("figure_type") == "table")
if figure_image or figure_table:
    print(f"  图表标记: image={figure_image}, table={figure_table}")

# 统计 merged_characters
merged_char_count = sum(1 for doc in all_chunks if len(doc.metadata.get("merged_characters", [])) > 1)
if merged_char_count:
    print(f"  多角色合并 chunks: {merged_char_count}")

print(f"\n总计: {len(all_chunks)} chunks")
print(f"Debug 输出目录: {debug_output_dir}")

# 列出生成的 debug 文件
debug_path = AGENT_DIR / debug_output_dir
if debug_path.exists():
    files = sorted(debug_path.rglob("*.txt"))
    if files:
        print(f"\nDebug 文件 ({len(files)} 个):")
        for f in files:
            size_kb = f.stat().st_size / 1024
            print(f"  {f.name} ({size_kb:.1f} KB)")
