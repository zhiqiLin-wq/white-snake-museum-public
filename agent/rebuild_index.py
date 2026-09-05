"""独立索引重建脚本（恢复/重建 Chroma literature_chunks 集合）。

为什么需要独立脚本：
  服务进程内重建时，FastAPI/Chroma/BM25/embedder 同占内存，
  第三章（最长章节）句子级 embedding 峰值内存过高会导致进程被系统
  静默杀掉（无 Python 堆栈），且旧实现"先全量分块再一次性写入"，
  中途崩溃则全部丢失。本脚本：
    1. 独立进程运行（无 Web 服务竞争内存）；
    2. 逐章分块、逐章 upsert 入库（chunk_id 确定性生成，重跑幂等覆盖）；
    3. 小批次 embedding（16 条/批）+ 每章后 gc 释放内存；
    4. 结束后打印集合实际行数并做一次检索自检。

用法（Agent 服务请先停止，避免并发占用 ChromaDB）：
    cd agent
    venv\Scripts\python.exe rebuild_index.py
"""

import gc
import logging
import os
import sys
import time
from pathlib import Path

AGENT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(AGENT_DIR))

from dotenv import load_dotenv
ENV_FILE = AGENT_DIR.parent / ".env"
if ENV_FILE.exists():
    load_dotenv(ENV_FILE)
    print(f"[ENV] Loaded: {ENV_FILE}")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("rebuild_index")
# 降低第三方库噪音
logging.getLogger("chromadb").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)

from server.rag.config import rag_config
from server.rag.embedder import Embedder
from server.rag.chunker import Chunker
from server.rag.vector_store import VectorStore

# 小批次：降低 embedding/写入的峰值内存。
# 实测：本机空闲内存 ~3GB 时，批 16 在长句语义分块阶段触发原生崩溃（0xC0000005），
# 批 4 全流程稳定（RSS ~1.7GB），434 块约 25 分钟。
EMBED_BATCH_SIZE = int(os.getenv("REBUILD_EMBED_BATCH", "4"))
ADD_BATCH_SIZE = 16


def mem_info() -> str:
    """返回进程 RSS 与系统可用内存（GB），用于定位原生层 OOM（0xC0000005）。"""
    try:
        import psutil
        p = psutil.Process()
        rss = p.memory_info().rss / 1024 ** 3
        avail = psutil.virtual_memory().available / 1024 ** 3
        return f"RSS={rss:.2f}GB 可用RAM={avail:.1f}GB"
    except Exception:
        try:
            import ctypes
            class PMC(ctypes.Structure):
                _fields_ = [("cb", ctypes.c_uint32), ("PageFaultCount", ctypes.c_uint32),
                            ("PeakWorkingSetSize", ctypes.c_size_t),
                            ("WorkingSetSize", ctypes.c_size_t),
                            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                            ("QuotaPagedPoolUsage", ctypes.c_size_t),
                            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                            ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                            ("PagefileUsage", ctypes.c_size_t),
                            ("PeakPagefileUsage", ctypes.c_size_t)]
            pmc = PMC()
            pmc.cb = ctypes.sizeof(PMC)
            ctypes.windll.psapi.GetProcessMemoryInfo(
                ctypes.windll.kernel32.GetCurrentProcess(),
                ctypes.byref(pmc), pmc.cb)
            rss = pmc.WorkingSetSize / 1024 ** 3

            class MS(ctypes.Structure):
                _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                            ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                            ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                            ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                            ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
            ms = MS()
            ms.dwLength = ctypes.sizeof(MS)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(ms))
            return f"RSS={rss:.2f}GB 可用RAM={ms.ullAvailPhys/1024**3:.1f}GB"
        except Exception:
            return "RSS=?"


def wait_for_embedder(embedder: Embedder, max_wait: int = 300) -> bool:
    logger.info("等待 Embedding 模型加载...")
    waited = 0
    while not embedder.is_ready and waited < max_wait:
        if embedder.load_error:
            logger.error(f"模型加载失败: {embedder.load_error}")
            return False
        logger.info(f"  进度: {embedder.progress:.0%} (已等待 {waited}s)")
        time.sleep(5)
        waited += 5
    return embedder.is_ready


def main() -> int:
    logger.info(f"Embedding 模型: {rag_config.embedding_model}")
    embedder = Embedder(
        model_name=rag_config.embedding_model,
        local_path=rag_config.embedding_local_path or None,
    )
    embedder.start_loading()
    if not wait_for_embedder(embedder):
        return 1
    logger.info("模型就绪")

    vs = VectorStore(
        persist_dir=rag_config.chroma_persist_path,
        embedder=embedder,
        bm25_retriever=None,  # 脚本不建 BM25（服务重启时从集合加载）
    )
    coll = vs.ensure_collection()
    existing = coll.count()
    logger.info(f"集合 literature_chunks 当前行数: {existing}（upsert 将覆盖同 id 块）")

    data_dir = rag_config.data_dir_path
    logger.info(f"数据目录: {data_dir}")
    chunker = Chunker(
        data_dir=data_dir,
        embedder=embedder,
        embed_batch_size=EMBED_BATCH_SIZE,
    )

    stats: list[dict] = []

    def sink(chapter_number: str, chapter_title: str, dynasty, chunks: list):
        logger.info(f"[内存] 第{chapter_number}章分块完成，{mem_info()}")
        n = vs.upsert_chunks(chunks, embed_batch_size=EMBED_BATCH_SIZE,
                             add_batch_size=ADD_BATCH_SIZE)
        stats.append({"chapter": chapter_number, "chunks": n, "dynasty": dynasty or ""})
        logger.info(f"[入库] 第{chapter_number}章 {dynasty or ''} "
                    f"《{chapter_title[:30]}》: {n} 块 upsert 完成，"
                    f"集合累计 {coll.count()} 行，{mem_info()}")
        # 释放本章句子 embedding 缓存与临时对象，压低峰值内存
        chunker._embedding_cache.clear()
        gc.collect()
        logger.info(f"[内存] gc 后，{mem_info()}")

    logger.info("=" * 60)
    logger.info("开始逐章分块 + 入库...")
    t0 = time.time()
    try:
        all_chunks = chunker.chunk_all(chapter_sink=sink)
    except Exception as e:
        logger.error(f"分块/入库失败: {e}")
        import traceback
        traceback.print_exc()
        logger.error("已入库章节保留在集合中，修复后重跑本脚本即可幂等续建。")
        return 2
    elapsed = time.time() - t0

    logger.info("=" * 60)
    total = coll.count()
    logger.info(f"重建完成：分块 {len(all_chunks)} 个，集合实际 {total} 行，耗时 {elapsed:.0f}s")
    for s in stats:
        logger.info(f"  第{s['chapter']}章 {s['dynasty']}: {s['chunks']} 块")

    if total < len(all_chunks) // 2:
        logger.error(f"[ALERT] 集合行数({total}) 远少于分块数({len(all_chunks)})，入库可能不完整")
        return 3

    # 自检：一次真实语义检索
    try:
        vec = embedder.embed_query("金山寺 水漫金山")
        hits = vs.query(vec, top_k=3)
        logger.info(f"[自检] 语义检索 '金山寺 水漫金山' 召回 {len(hits)} 条")
        for h in hits[:3]:
            meta = h.get("metadata", {})
            logger.info(f"  - 第{meta.get('chapter_number','?')}章 "
                        f"{h.get('content', '')[:40]}...")
        if len(hits) == 0:
            logger.error("[自检] 检索返回 0 条，请检查数据")
            return 4
    except Exception as e:
        logger.warning(f"[自检] 检索验证失败（不影响入库结果）: {e}")

    logger.info("完成。现在可以启动 Agent 服务（服务启动时 BM25 会从集合同步）。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
