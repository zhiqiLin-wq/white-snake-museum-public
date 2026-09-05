"""真实运行一致性追踪 — memory 与 context 系统逐轮数据流演示。

目的：用与生产完全一致的装配与调用序列，逐轮展现对话过程中
context（FIFO/摘要/压力）、长期记忆、召回、Core Memory、被动注入、
以及底层 SQLite/ChromaDB 各表数据的真实变化。

与生产严格对齐（对照 main.py lifespan + api/routes.py _execute_chat）：
1. 装配：context.wiring.build_memory_system() 工厂（与 main.py 完全一致）
2. 调用序列（与 _execute_chat 一致）：
   cm.load(thread_id) -> cm.enqueue(user) -> cm.build_memory_context_text
   -> cm.build_history_messages -> (LLM 生成回复) -> cm.enqueue(assistant)
   -> cm.save(thread_id)
3. LLM：真实 DeepSeekProvider（deepseek-v4-flash）
4. Embedder：真实 BAAI/bge-large-zh-v1.5

唯一刻意差异：fifo_max_tokens 用演示值（DEMO_FIFO_MAX_TOKENS=600），
生产默认 90000（FIFO_MAX_TOKENS 环境变量）。目的：让「超窗口压缩 ->
召回存储 -> 摘要更新 -> 长期记忆提取」这条生产代码路径，在有限演示轮数内
真实触发并观察。代码路径与生产完全一致，仅窗口大小参数不同。

输出：tests/_test_data/trace_realtime.txt（UTF-8），控制台同步打印。
"""

import asyncio
import json
import shutil
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path

_project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_project_root))

BEIJING_TZ = timezone(timedelta(hours=8))
DEMO_FIFO_MAX_TOKENS = 300
DATA_DIR = _project_root / "tests" / "_test_data"
LOG_PATH = DATA_DIR / "trace_realtime.txt"

_out = None


def log(msg: str = ""):
    global _out
    if _out is None:
        _out = open(str(LOG_PATH), "w", encoding="utf-8")
    _out.write(msg + "\n")
    _out.flush()
    print(msg)


def hr(title: str = ""):
    log()
    log("=" * 90)
    if title:
        log(f"  {title}")
    log("=" * 90)
    log()


def _trunc(s: str, n: int = 80) -> str:
    s = str(s)
    return s if len(s) <= n else s[:n] + "..."


def dump_round(turn: int, query: str, answer: str, ms, thread_id: str,
               memory_context: str = "", history: list = None):
    """输出一轮对话后的完整数据快照（memory_context/history 为 supervisor 实际收到的值）。"""
    cm = ms.context_manager
    db = ms.memory_db
    store = ms.recall_store
    state = cm.get_state(thread_id)

    hr(f"第 {turn} 轮对话")

    log(f"[用户输入] {query}")
    log(f"[助手回复] {_trunc(answer, 120)}")

    # ---- context 状态 ----
    if state:
        ratio = cm.get_pressure_ratio(thread_id)
        log(f"[ContextState] FIFO {len(state.messages)} 条 | {state.total_tokens} tokens "
            f"/ 上限 {cm._max_tokens} ({ratio:.0%}) | 摘要 {len(state.summary)} 字符")
        log(f"[摘要内容] {_trunc(state.summary or '(空)', 200)}")
        log("  [FIFO 消息队列 (旧->新)]")
        for i, m in enumerate(state.messages):
            log(f"    [{i}] {m.get('role'):9s} | {_trunc(m.get('content'), 60)}")
    else:
        log("[ContextState] (未加载)")

    # ---- supervisor 实际收到的 memory_context（被动注入/摘要/Core Memory/压力）----
    log(f"[memory_context 注入文本] {len(memory_context)} 字符:")
    log("    " + (memory_context.replace(chr(10), chr(10) + "    ") if memory_context else "(空)"))

    # ---- supervisor 实际收到的 history ----
    log(f"[history 传给 supervisor] {len(history) if history is not None else 0} 条 user/assistant 消息")

    # ---- 底层数据表 ----
    stats = db.get_stats()
    log(f"[SQLite 各表] messages={stats['total_messages']} | "
        f"活跃长期记忆={stats['total_memories']} | 召回元数据={stats['total_recall_meta']} | "
        f"待补向量={stats['pending_embeddings']} | 已合并={stats['consolidated_memories']} | "
        f"已墓碑={stats['tombstoned_memories']} | changelog={stats['changelog_entries']}")

    # ---- 长期记忆 ----
    active = db.get_active_memories(limit=50)
    log(f"[长期记忆 long_term_memories] {len(active)} 条活跃:")
    for m in active:
        log(f"    [{m['memory_type']:8s}] importance={m['importance']:.2f} "
            f"access={m['access_count']} | {_trunc(m['content'], 55)}")

    # ---- 召回消息 ----
    rec_msg_count = store._recall_collection.count() if store._recall_collection else 0
    mem_vec_count = store._memory_collection.count() if store._memory_collection else 0
    log(f"[ChromaDB 向量] recall_messages={rec_msg_count} | long_term_memories={mem_vec_count}")

    # ---- Core Memory ----
    human = db.get_core_memory_block("human")
    persona = db.get_core_memory_block("persona")
    log(f"[Core Memory] persona v{persona['version'] if persona else '?'} | "
        f"human v{human['version'] if human else '?'}")
    log(f"    human 块内容: {_trunc(human['content'] if human else '{}', 200)}")
    log()


async def _real_gen_reply(llm, query: str, model: str) -> str:
    """真实调用 LLM 生成简短助手回复（代替完整 supervisor 图，聚焦 memory/context 层）。"""
    r = await llm.generate(
        system="你是白蛇传博物馆的学术助手。必须用一两句中文直接回答问题，禁止输出空内容。",
        user=query,
        model=model,
        max_tokens=200,
        temperature=0.3,
    )
    return (r.content or "").strip()


async def main():
    # 清空输出
    if LOG_PATH.exists():
        LOG_PATH.unlink()

    from agent.server.config import settings
    from agent.server.rag.config import rag_config
    from agent.server.rag.embedder import Embedder
    from agent.server.llm.deepseek_provider import DeepSeekProvider
    from agent.server.context.wiring import build_memory_system

    hr("memory / context 系统 — 真实运行一致性逐轮追踪")

    # ---- 1. 真实 Embedder ----
    log("[装配] 加载真实 Embedder...")
    embedder = Embedder(
        model_name=rag_config.embedding_model,
        local_path=rag_config.embedding_local_path or None,
    )
    embedder.start_loading()
    for _ in range(120):
        if embedder.is_ready:
            break
        await asyncio.sleep(1)
    if not embedder.is_ready:
        log("[错误] Embedder 加载超时")
        return
    log(f"[装配] Embedder 就绪: {rag_config.embedding_model}")

    # ---- 2. 真实 ChromaDB client（临时目录，避免污染生产数据）----
    import chromadb
    chroma_dir = DATA_DIR / "chroma_realtime"
    if chroma_dir.exists():
        shutil.rmtree(chroma_dir, ignore_errors=True)
    chroma_dir.mkdir(parents=True, exist_ok=True)
    chroma_client = chromadb.PersistentClient(path=str(chroma_dir))
    log(f"[装配] ChromaDB client: {chroma_dir}")

    # ---- 3. 真实 LLM ----
    llm = DeepSeekProvider()
    memory_model = settings.memory_extraction_model or settings.effective_fast_model
    log(f"[装配] LLM: DeepSeek ({memory_model})")

    # ---- 4. 与 main.py 一致的工厂装配 ----
    mem_db = DATA_DIR / "realtime_memory.db"
    ctx_db = DATA_DIR / "realtime_context.db"
    for p in (mem_db, ctx_db):
        if p.exists():
            p.unlink()

    ms = build_memory_system(
        llm=llm,
        embedder=embedder,
        chroma_client=chroma_client,
        memory_db_path=mem_db,
        context_db_path=ctx_db,
        fifo_max_tokens=DEMO_FIFO_MAX_TOKENS,
        summary_model=memory_model,
        memory_extraction_model=memory_model,
        core_memory_enabled=settings.core_memory_enabled,
        background_tasks_enabled=settings.background_memory_tasks_enabled,
    )
    log(f"[装配] build_memory_system 完成 (fifo_max_tokens={DEMO_FIFO_MAX_TOKENS}, "
        f"core_memory={settings.core_memory_enabled}, background_tasks={settings.background_memory_tasks_enabled})")
    log(f"[装配] 开关: recall_storage={settings.recall_storage_enabled}, "
        f"memory_extraction={settings.memory_extraction_enabled}, "
        f"passive_injection={settings.memory_passive_injection_enabled}")
    log()

    cm = ms.context_manager
    db = ms.memory_db
    thread_id = "realtime-demo-thread"

    # ---- 5. 逐轮对话（与 _execute_chat 一致的序列）----
    turns = [
        "我叫知知，是研究宋代江南民间信仰与文学互动的学者，最喜欢《雷峰塔传奇》里的法海形象。",
        "法海这个角色在话本里的形象其实很复杂，能不能帮我分析一下他的多重性？",
        "白蛇传从宋代话本到明清小说的演变里，法海形象发生了哪些关键变化？",
        "雷峰塔在杭州西湖边，它和白蛇传的关联具体是怎么建立起来的？",
        "西湖十景里的雷峰夕照，是不是也和这个传说有关？",
        "能不能把前面聊的法海形象演变，帮我梳理成一条清晰的时间线？",
        "另外请记住，我研究上偏好文本细读和宏观历史分析相结合，不喜欢长篇大论的回复。",
    ]

    for turn, query in enumerate(turns, 1):
        # 真实序列：load -> enqueue(user) -> build_memory_context_text -> build_history_messages
        await cm.load(thread_id)
        await cm.enqueue(thread_id, {"role": "user", "content": query})
        # supervisor.run 实际收到的两个值（此时 assistant 尚未入队，与生产时序一致）
        memory_context = cm.build_memory_context_text(thread_id, query)
        history = cm.build_history_messages(thread_id)

        # 模拟 supervisor 用真实 LLM 生成回复
        answer = await _real_gen_reply(llm, query, memory_model)

        # 回写 assistant + 持久化（对齐 routes.py：空回复不回写）
        if answer.strip():
            await cm.enqueue(thread_id, {"role": "assistant", "content": answer})
            await cm.save(thread_id)

        dump_round(turn, query, answer, ms, thread_id, memory_context, history)
        await asyncio.sleep(0.5)

    # ---- 6. 最终全量快照 ----
    hr("最终汇总")
    stats = db.get_stats()
    log(f"[SQLite 汇总] {json.dumps(stats, ensure_ascii=False)}")
    log(f"[ChromaDB] recall_messages={ms.recall_store._recall_collection.count()}, "
        f"long_term_memories={ms.recall_store._memory_collection.count()}")
    log(f"[FIFO 最终] {cm.get_stats(thread_id)}")
    log()
    log("说明：assistant 回复用真实 LLM 生成以代替完整 supervisor 图（supervisor 含 RAG/技能路由，")
    log("与 memory/context 层无关）；memory/context 层的装配与调用序列与生产一致。")
    log(f"完整日志已写入: {LOG_PATH}")

    await cm.close()
    if ms.scheduler is not None:
        await ms.scheduler.stop()


if __name__ == "__main__":
    asyncio.run(main())
