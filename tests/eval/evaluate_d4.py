"""L4 压缩与摘要评测：压缩生成的摘要是否覆盖关键信息（语义判定版）。

复用 D3 的 3 段长对话数据，逐段走完整链路触发压缩，检查增量摘要是否在语义上
覆盖每条关键信息。语义判定用 bge 余弦：摘要按句切分，每条关键信息找最相似句，
余弦 >= 阈值即视为覆盖。摘要覆盖度 = 覆盖的关键信息数 / 总数。
"""

import asyncio
import re
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
FIFO_MAX_TOKENS = 400
SIM_THRESHOLD = 0.72


def cosine(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(x * x for x in b) ** 0.5
    return dot / (na * nb) if na > 0 and nb > 0 else 0.0


def split_sentences(text):
    """按中文句读切分摘要，返回非空片段列表。

    除句读外也按逗号/分号切，避免摘要把多条信息挤在一个长句里、
    导致单条关键信息与混合长句的余弦被稀释。
    """
    parts = re.split(r"[。！？；，、\n]+", text)
    return [p.strip() for p in parts if p.strip()]


async def run_segment(seg, embedder, llm):
    """跑一段对话的完整链路，返回 (覆盖数, 总数, 摘要文本, 缺失列表)。"""
    import chromadb
    chroma_dir = _eval_dir / f"chroma_eval_d4_{seg['id']}"
    if chroma_dir.exists():
        shutil.rmtree(chroma_dir, ignore_errors=True)
    chroma_dir.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(chroma_dir))

    mem_db = _eval_dir / f"eval_memory_d4_{seg['id']}.db"
    ctx_db = _eval_dir / f"eval_context_d4_{seg['id']}.db"
    for p in (mem_db, ctx_db):
        if p.exists():
            p.unlink()

    ms = build_memory_system(
        llm=llm, embedder=embedder, chroma_client=client,
        memory_db_path=mem_db, context_db_path=ctx_db,
        fifo_max_tokens=FIFO_MAX_TOKENS, summary_model=MODEL, memory_extraction_model=MODEL,
    )
    cm = ms.context_manager
    thread_id = f"d4-eval-{seg['id']}"
    await cm.load(thread_id)
    for msg in seg["conversation"]:
        await cm.enqueue(thread_id, {"role": msg["role"], "content": msg["content"]})
        await asyncio.sleep(0.05)

    state = cm.get_state(thread_id)
    summary = state.summary if state else ""

    # 语义判定：摘要按句切分，每条关键信息找最相似句
    summary_sents = split_sentences(summary)
    sent_vecs = [embedder.embed([s])[0] for s in summary_sents] if summary_sents else []

    covered = 0
    missing = []
    for fact in seg["key_facts"]:
        fv = embedder.embed([fact])[0]
        best = max((cosine(fv, sv) for sv in sent_vecs), default=0.0)
        if best >= SIM_THRESHOLD:
            covered += 1
        else:
            missing.append((fact, best))

    await cm.close()
    if ms.scheduler is not None:
        await ms.scheduler.stop()
    return covered, len(seg["key_facts"]), summary, missing


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

    print("[2/2] 逐段跑完整链路触发压缩，语义判定摘要覆盖...\n")
    grand_covered = 0
    grand_total = 0
    for seg in segments:
        covered, total, summary, missing = await run_segment(seg, embedder, llm)
        grand_covered += covered
        grand_total += total
        print(f"===== {seg['label']} =====")
        print(f"  摘要长度: {len(summary)} 字符")
        print(f"  覆盖度: {covered}/{total} = {covered/total:.4f}")
        for fact, sim in missing:
            print(f"    [缺失] sim={sim:.4f} <- {fact[:28]}")
        print(f"  摘要预览: {summary[:150]}...")
        print()

    print(f"===== L4 压缩与摘要（语义判定） =====")
    print(f"总计覆盖度: {grand_covered}/{grand_total} = {grand_covered/grand_total:.4f}")


if __name__ == "__main__":
    asyncio.run(main())
