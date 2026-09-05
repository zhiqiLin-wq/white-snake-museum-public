"""L3 端到端记忆保持评测（LOCOMO 风格简化版，v2：多段对话）。

把短/中/长 3 段长对话分别走完整 memory 链路（enqueue -> 压缩 -> 提取），
然后对每段的关键信息检索，用 bge 向量相似度判定是否仍可找回。
保持率 = 找回的关键信息数 / 总数；分长度维度报告。
"""

import asyncio
import shutil
import sys
from pathlib import Path

_eval_dir = Path(__file__).resolve().parent
_project_root = _eval_dir.parent.parent
sys.path.insert(0, str(_project_root))
sys.path.insert(0, str(_eval_dir))

from common import read_json
from agent.server.llm.deepseek_provider import DeepSeekProvider
from agent.server.config import settings
from agent.server.rag.config import rag_config
from agent.server.rag.embedder import Embedder
from agent.server.context.wiring import build_memory_system

MODEL = settings.memory_extraction_model or settings.effective_fast_model
SIM_THRESHOLD = 0.72
FIFO_MAX_TOKENS = 400


def cosine(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(x * x for x in b) ** 0.5
    return dot / (na * nb) if na > 0 and nb > 0 else 0.0


async def run_segment(seg, embedder, llm):
    """跑一段对话的完整链路，返回 (找回数, 关键信息总数)。"""
    import chromadb
    chroma_dir = _eval_dir / f"chroma_eval_{seg['id']}"
    if chroma_dir.exists():
        shutil.rmtree(chroma_dir, ignore_errors=True)
    chroma_dir.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(chroma_dir))

    mem_db = _eval_dir / f"eval_memory_{seg['id']}.db"
    ctx_db = _eval_dir / f"eval_context_{seg['id']}.db"
    for p in (mem_db, ctx_db):
        if p.exists():
            p.unlink()

    ms = build_memory_system(
        llm=llm, embedder=embedder, chroma_client=client,
        memory_db_path=mem_db, context_db_path=ctx_db,
        fifo_max_tokens=FIFO_MAX_TOKENS, summary_model=MODEL, memory_extraction_model=MODEL,
    )
    cm = ms.context_manager
    retriever = ms.hybrid_retriever
    db = ms.memory_db

    thread_id = f"d3-eval-{seg['id']}"
    await cm.load(thread_id)
    for msg in seg["conversation"]:
        await cm.enqueue(thread_id, {"role": msg["role"], "content": msg["content"]})
        await asyncio.sleep(0.05)

    # 收尾 flush：模拟对话结束后的收尾提取，把仍留在 FIFO 缓冲区的消息也提取成记忆。
    # 真实生产中这一步由 idle 提取（30 分钟无活动）或下次对话触发，评测显式补上，
    # 避免末尾关键信息因未触发压缩弹出而漏提取。
    state = cm.get_state(thread_id)
    if state and state.messages:
        remaining = list(state.messages)
        extractor = cm._get_memory_extractor()
        await extractor.extract_batch(remaining)

    active = db.get_active_memories(limit=200)
    print(f"    提取到的长期记忆: {len(active)} 条")

    kept = 0
    total = len(seg["key_facts"])
    for fact in seg["key_facts"]:
        res = retriever.search(fact, source="memories", score_threshold=0.0, vector_top_k=10)
        top_contents = [m["content"] for m in res["memories"]]
        fv = embedder.embed([fact])[0]
        hit = any(cosine(fv, embedder.embed([c])[0]) >= SIM_THRESHOLD for c in top_contents)
        if hit:
            kept += 1
            print(f"    [保持] {fact[:24]}...")
        else:
            print(f"    [丢失] {fact[:24]}...")

    await cm.close()
    if ms.scheduler is not None:
        await ms.scheduler.stop()
    return kept, total


async def main():
    data = read_json("d3_retention.json")
    segments = data["segments"]

    print("[1/2] 加载 embedder...")
    embedder = Embedder(model_name=rag_config.embedding_model,
                        local_path=rag_config.embedding_local_path or None)
    embedder.start_loading()
    for _ in range(60):
        if embedder.is_ready:
            break
        await asyncio.sleep(1)

    llm = DeepSeekProvider()

    print("[2/2] 逐段跑完整链路...\n")
    grand_kept = 0
    grand_total = 0
    per_seg = []
    for seg in segments:
        print(f"===== {seg['label']} ({len(seg['conversation'])} 轮) =====")
        kept, total = await run_segment(seg, embedder, llm)
        rate = kept / total if total else 0.0
        per_seg.append((seg["label"], kept, total, rate))
        grand_kept += kept
        grand_total += total
        print(f"  -> 保持率 {kept}/{total} = {rate:.4f}\n")

    print("===== L3 端到端记忆保持 =====")
    for label, kept, total, rate in per_seg:
        print(f"  {label:16s} {kept:2d}/{total:2d} = {rate:.4f}")
    print(f"  {'总计':16s} {grand_kept}/{grand_total} = {grand_kept/grand_total:.4f}")


if __name__ == "__main__":
    asyncio.run(main())
