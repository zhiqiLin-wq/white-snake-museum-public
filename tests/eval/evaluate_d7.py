"""D7 记忆合并评测：验证 _run_memory_merge 是否正确合并相似记忆、保留不同记忆。

构造相似对（语义等价，cosine 应 > 0.7）和不同对（语义无关，cosine 应 < 0.7），
预先用 embedder 算 cosine 作为金标准，真实写入 + 跑合并，验证：
- 相似对（金标准 > 0.7）的两条 → 都被 consolidated（合并）
- 不同对（金标准 < 0.7）的两条 → 都保持 active（保留）
"""

import asyncio
import shutil
import sys
from pathlib import Path

_eval_dir = Path(__file__).resolve().parent
_project_root = _eval_dir.parent.parent
sys.path.insert(0, str(_project_root))
sys.path.insert(0, str(_eval_dir))

from agent.server.llm.deepseek_provider import DeepSeekProvider
from agent.server.config import settings
from agent.server.rag.config import rag_config
from agent.server.rag.embedder import Embedder
from agent.server.context.wiring import build_memory_system

MODEL = settings.memory_extraction_model or settings.effective_fast_model
MERGE_THRESHOLD = 0.7

# 相似对（语义等价，应被合并）
SIMILAR_PAIRS = [
    ("用户喜欢法海这个角色", "用户最喜欢的文学人物是法海"),
    ("用户偏好文本细读与宏观历史分析相结合", "用户喜欢把文本细读和宏观分析结合起来"),
]
# 不同对（语义无关，应保留）
DIFFERENT_PAIRS = [
    ("用户不喜欢收到长篇回复", "雷峰塔位于杭州西湖边"),
    ("用户研究时要求引用文献原文并标注出处", "分析版本演变要按朝代顺序"),
]


def cosine(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(x * x for x in b) ** 0.5
    return dot / (na * nb) if na > 0 and nb > 0 else 0.0


async def main():
    embedder = Embedder(model_name=rag_config.embedding_model,
                        local_path=rag_config.embedding_local_path or None)
    embedder.start_loading()
    for _ in range(60):
        if embedder.is_ready:
            break
        await asyncio.sleep(1)

    import chromadb
    chroma_dir = _eval_dir / "chroma_eval_d7"
    if chroma_dir.exists():
        shutil.rmtree(chroma_dir, ignore_errors=True)
    chroma_dir.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(chroma_dir))

    mem_db = _eval_dir / "eval_memory_d7.db"
    ctx_db = _eval_dir / "eval_context_d7.db"
    for p in (mem_db, ctx_db):
        if p.exists():
            p.unlink()

    llm = DeepSeekProvider()
    ms = build_memory_system(
        llm=llm, embedder=embedder, chroma_client=client,
        memory_db_path=mem_db, context_db_path=ctx_db,
        fifo_max_tokens=90000, summary_model=MODEL, memory_extraction_model=MODEL,
    )
    db = ms.memory_db
    store = ms.recall_store

    all_pairs = SIMILAR_PAIRS + DIFFERENT_PAIRS

    # 预先算 cosine 金标准
    all_contents = [c for pair in all_pairs for c in pair]
    vecs = embedder.embed(all_contents)
    print("[金标准] 各对 cosine（> 0.7 应合并）:")
    pair_sims = []
    for idx, (a, b) in enumerate(all_pairs):
        sim = cosine(vecs[idx * 2], vecs[idx * 2 + 1])
        pair_sims.append(sim)
        kind = "相似对" if idx < len(SIMILAR_PAIRS) else "不同对"
        print(f"  [{kind}] {sim:.4f}  '{a[:16]}...' vs '{b[:16]}...'")

    # 写入（相似对给不同 importance，验证合并取 max）
    ids = []
    importances = []
    sim_imp = [(0.5, 0.8), (0.6, 0.9)]
    for idx, (a, b) in enumerate(all_pairs):
        if idx < len(SIMILAR_PAIRS):
            ia, ib = sim_imp[idx]
        else:
            ia, ib = 0.7, 0.7
        ids.append(store.add_memory(a, memory_type="Entity", importance=ia))
        ids.append(store.add_memory(b, memory_type="Entity", importance=ib))
        importances.append((ia, ib))

    print(f"\n写入 {len(ids)} 条记忆，跑合并...")
    await ms.scheduler._run_memory_merge()

    # 验证
    print("\n=== 验证结果 ===")
    correct = 0
    false_merge = 0
    miss_merge = 0
    for idx, sim in enumerate(pair_sims):
        a_status = db.get_long_term_memory(ids[idx * 2])["status"]
        b_status = db.get_long_term_memory(ids[idx * 2 + 1])["status"]
        should_merge = sim > MERGE_THRESHOLD
        actually_merged = (a_status == "consolidated" and b_status == "consolidated")
        if should_merge and actually_merged:
            correct += 1
            print(f"  [正确合并] 相似对{idx} sim={sim:.3f} -> consolidated")
        elif should_merge and not actually_merged:
            miss_merge += 1
            print(f"  [漏合并] 相似对{idx} sim={sim:.3f} 应合并但 status={a_status}/{b_status}")
        elif not should_merge and actually_merged:
            false_merge += 1
            print(f"  [误合并] 不同对{idx} sim={sim:.3f} 不应合并但被合并了")
        else:
            correct += 1
            print(f"  [正确保留] 不同对{idx} sim={sim:.3f} -> active")

    # 验证新合并条内容 + importance 取 max + changelog
    print("\n=== 新合并条 + changelog ===")
    active = db.get_active_memories(limit=100)
    for idx in range(len(SIMILAR_PAIRS)):
        a, b = all_pairs[idx]
        ia, ib = importances[idx]
        merged_content = f"{a}; {b}"
        new_mem = next((m for m in active if m["content"] == merged_content), None)
        if new_mem is None:
            print(f"  [新条缺失] 相似对{idx} 未找到 '{merged_content}'")
            continue
        imp_ok = abs(new_mem["importance"] - max(ia, ib)) < 1e-9
        logs = db.get_changelog_by_memory(new_mem["memory_id"])
        merge_logs = [l for l in logs if l.get("event_type") == "MERGE"]
        print(f"  [相似对{idx}] 新条 importance={new_mem['importance']} (期望 {max(ia, ib)}) "
              f"{'OK' if imp_ok else 'MISMATCH'} | changelog MERGE {len(merge_logs)} 条")

    total = len(pair_sims)
    print(f"\n===== D7 合并评测 =====")
    print(f"正确判定: {correct}/{total} = {correct/total:.4f}")
    print(f"误合并: {false_merge}（应保留却被合并，期望=0）")
    print(f"漏合并: {miss_merge}（应合并却保留）")

    if ms.scheduler is not None:
        await ms.scheduler.stop()


if __name__ == "__main__":
    asyncio.run(main())
