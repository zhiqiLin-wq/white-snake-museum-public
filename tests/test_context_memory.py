"""多轮对话上下文管理集成测试 + 详细追踪日志。

不依赖 FastAPI / Node.js / 完整 RAG 管线。
仅测试 ContextManager + RecallStore + MemoryExtractor + retrieve_memory 的端到端行为。

运行方式:
  从项目根目录 white-snake-museum-public/ 运行:
  python tests/test_context_memory.py

日志同时输出到:
  - 控制台 (INFO 级别，简要进度)
  - tests/_test_data/trace.log (DEBUG 级别，完整消息内容 + 每步详细追踪)
"""

import asyncio
import logging
import sys
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path

_project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_project_root))

# ============================================================
#  时区: 北京时间 (UTC+8)
# ============================================================
BEIJING_TZ = timezone(timedelta(hours=8))


def _beijing_now() -> datetime:
    """返回当前北京时间。"""
    return datetime.now(BEIJING_TZ)


class BeijingFormatter(logging.Formatter):
    """使用北京时间的 logging Formatter。"""

    def formatTime(self, record, datefmt=None):
        dt = datetime.fromtimestamp(record.created, tz=BEIJING_TZ)
        if datefmt:
            return dt.strftime(datefmt)
        return dt.strftime("%Y-%m-%d %H:%M:%S")


# ============================================================
#  日志配置: 控制台 (INFO) + 文件 (DEBUG 完整追踪)
# ============================================================
LOG_DIR = _project_root / "tests" / "_test_data"
LOG_DIR.mkdir(parents=True, exist_ok=True)
TRACE_LOG_PATH = LOG_DIR / "trace.log"

# 清空旧日志
if TRACE_LOG_PATH.exists():
    TRACE_LOG_PATH.unlink()

# 文件 handler: 完整 DEBUG 级别，包含所有细节
file_handler = logging.FileHandler(str(TRACE_LOG_PATH), encoding="utf-8")
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(BeijingFormatter(
    "%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
))

# 控制台 handler: INFO 级别，简要进度
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(BeijingFormatter(
    "%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
))

# 根 logger 配置
root_logger = logging.getLogger()
root_logger.setLevel(logging.DEBUG)
root_logger.handlers.clear()
root_logger.addHandler(console_handler)
root_logger.addHandler(file_handler)

logger = logging.getLogger("test_context_memory")

# ============================================================
#  专用追踪 logger: 写入 trace.log 的纯文本辅助函数
# ============================================================
_trace_file = None


def _get_trace_file():
    global _trace_file
    if _trace_file is None:
        _trace_file = open(str(TRACE_LOG_PATH), "a", encoding="utf-8")
    return _trace_file


def trace(msg: str = ""):
    """向 trace.log 追加一行纯文本（不经 logging 格式化）。"""
    ts = _beijing_now().strftime("%H:%M:%S")
    f = _get_trace_file()
    f.write(f"[{ts}] {msg}\n")
    f.flush()


def trace_separator(title: str = "", char: str = "=", width: int = 80):
    if title:
        trace()
        trace(char * width)
        trace(f"  {title}")
        trace(char * width)
        trace()
    else:
        trace(char * width)


def trace_message(index: int, role: str, content: str, tags: str = ""):
    """打印一条完整消息。"""
    tag_str = f" [{tags}]" if tags else ""
    trace(f"[MSG #{index}] {role}{tag_str}  ({len(content)} chars)")
    trace("-" * 60)
    trace(content)
    trace("-" * 60)
    trace()


def trace_fifo_snapshot(context_manager, thread_id: str, label: str = ""):
    """打印 FIFO 队列中所有消息的完整内容。"""
    state = context_manager.get_state(thread_id)
    if state is None:
        trace(f"[FIFO SNAPSHOT] {label}: (not loaded)")
        return

    trace(f"[FIFO SNAPSHOT] {label}: {len(state.messages)} msgs, {state.total_tokens} tokens")
    trace(f"  Summary ({len(state.summary)} chars): {state.summary[:300] if state.summary else '(empty)'}")
    trace()

    for i, msg in enumerate(state.messages):
        role = msg.get("role", "?")
        content = str(msg.get("content", ""))
        ephemeral = " [EPHEMERAL]" if msg.get("_ephemeral") else ""
        truncated = content[:500]
        if len(content) > 500:
            truncated += f"\n... [truncated, total {len(content)} chars]"
        trace(f"  [{i}] {role}{ephemeral}  ({len(content)} chars)")
        trace(f"      {truncated}")
        trace()


def trace_context_window(context_manager, thread_id: str, label: str = ""):
    """打印 build_context_window 的完整输出。"""
    window = context_manager.build_context_window(thread_id)
    trace(f"[CONTEXT WINDOW] {label}: {len(window)} messages total")
    trace()
    for i, msg in enumerate(window):
        role = msg.get("role", "?")
        content = str(msg.get("content", ""))
        trace(f"  [{i}] {role}  ({len(content)} chars)")
        trace(f"      {content[:600]}")
        if len(content) > 600:
            trace(f"      ... [truncated, total {len(content)} chars]")
        trace()


# ================================================================
#  测试场景定义
# ================================================================

SCENARIO = {
    "label": "端到端测试: 关键信息 + 闲聊压缩 + 检索验证",
    "thread_id": "test-memory-e2e",
    "seed_info": [
        {
            "role": "user",
            "content": (
                "我叫小研。我特别喜欢《雷峰塔传奇》里的法海形象，"
                "觉得他其实很复杂，不只是简单的反派。"
                "另外，我的研究方向是宋代江南地区的民间信仰与文学文本的互动关系。"
                "我希望你在后续对话中能记住这些偏好。"
            ),
            "label": "播种关键信息",
        },
        {
            "role": "assistant",
            "content": (
                "小研您好！您的观点非常有见地。法海在《雷峰塔传奇》中的形象确实值得深入探讨，"
                "他不仅是宗教权威的象征，更代表了宋代社会对秩序与情感的复杂态度。"
                "您的研究方向——宋代江南民间信仰与文学的互动——是一个非常精深的领域。"
                "我会记住您的这些偏好。"
            ),
            "label": "AI 确认关键信息",
        },
    ],
    "chitchat_rounds": 20,
    "chitchat_topics": [
        "今天杭州天气如何？适合出游吗？",
        "白蛇传的故事里你最喜欢哪个角色？",
        "你觉得西湖美还是洞庭湖美？",
        "中国古代四大民间传说分别是什么？",
        "唐宋时期的妇女地位有什么变化？",
        "雷峰塔是什么时候倒塌的？新塔是哪年建的？",
        "杭州有哪些必去的景点？",
        "白素贞和许仙的爱情故事为什么能流传千年？",
        "你觉得文学作品对历史研究有什么价值？",
        "江南地区的水系对文化发展有什么影响？",
        "宋代话本和白话小说有什么区别？",
        "你了解法海在历史上的真实原型吗？",
        "苏轼来过杭州吗？他写过关于西湖的诗吗？",
        "青蛇这个角色在不同版本中有什么变化？",
        "民间信仰和官方宗教在宋代的关系是怎样的？",
        "你觉得好的文学研究应该具备什么特质？",
        "文本细读和宏观历史分析哪个更重要？",
        "西湖十景中你最喜欢哪一个？",
        "白蛇传在戏曲中有哪些著名的改编版本？",
        "明清时期对白蛇故事的改写有什么特点？",
    ],
    "verification_query": "我之前提到过我喜欢哪个文学人物？我对什么研究领域感兴趣？我的名字是什么？",
    "expected_keywords": ["法海", "雷峰塔传奇", "民间信仰", "宋代", "小研"],
}


# ================================================================
#  Monkey-patch ContextManager 以捕获详细追踪
# ================================================================
_original_enqueue = None
_original_update_summary = None
_msg_counter = [0]  # 全局消息序号


def _install_tracing(context_manager):
    """给 ContextManager 安装追踪钩子，拦截关键操作输出到 trace.log。"""
    global _original_enqueue, _original_update_summary

    _original_enqueue = context_manager.enqueue
    _original_update_summary = context_manager._update_summary

    async def traced_enqueue(thread_id: str, message: dict):
        _msg_counter[0] += 1
        idx = _msg_counter[0]
        role = message.get("role", "?")
        content = str(message.get("content", ""))
        ephemeral = message.get("_ephemeral", False)

        trace_separator(f"ENQUEUE #{idx}: {role}" + (" [EPHEMERAL]" if ephemeral else ""))
        trace_message(idx, role, content, "INCOMING" if not ephemeral else "EPHEMERAL")

        # 入队前状态
        stats_before = context_manager.get_stats(thread_id)
        state_before = context_manager.get_state(thread_id)
        msgs_before = len(state_before.messages) if state_before else 0
        tokens_before = stats_before["total_tokens"]
        summary_before = stats_before["summary_preview"]

        trace(f"[BEFORE enqueue] msgs={msgs_before} tokens={tokens_before}/{context_manager._max_tokens} "
              f"summary_chars={stats_before['summary_length']}")

        # 执行原始 enqueue
        await _original_enqueue(thread_id, message)

        # 入队后状态
        stats_after = context_manager.get_stats(thread_id)
        state_after = context_manager.get_state(thread_id)
        msgs_after = len(state_after.messages) if state_after else 0
        tokens_after = stats_after["total_tokens"]

        trace(f"[AFTER enqueue] msgs={msgs_before}->{msgs_after} "
              f"tokens={tokens_before}->{tokens_after}/{context_manager._max_tokens}")

        # 如果摘要更新了
        if stats_after["summary_length"] > 0 and stats_after["summary_preview"] != summary_before:
            trace(f"[SUMMARY CHANGED] {stats_before['summary_length']} -> {stats_after['summary_length']} chars")
            trace(f"  New summary: {stats_after['summary_preview']}")

        # 如果发生了弹出
        if msgs_before > 0 and msgs_after <= msgs_before:
            trace(f"[POP DETECTED] {msgs_before - msgs_after + 1} message(s) were popped during this enqueue")

        trace()

    async def traced_update_summary(state, popped_message):
        role = popped_message.get("role", "unknown")
        content = str(popped_message.get("content", ""))[:1000]
        old_summary = state.summary or "(empty)"

        trace_separator("SUMMARY UPDATE", "-")
        trace(f"[OLD SUMMARY] ({len(state.summary)} chars):")
        trace(f"  {old_summary}")
        trace()
        trace(f"[POPPED MESSAGE] {role} ({len(content)} chars):")
        trace(f"  {content[:500]}")
        if len(content) > 500:
            trace(f"  ... [truncated for summary, full {len(content)} chars]")
        trace()

        await _original_update_summary(state, popped_message)

        trace(f"[NEW SUMMARY] ({len(state.summary)} chars):")
        trace(f"  {state.summary[:500]}")
        if len(state.summary) > 500:
            trace(f"  ... [truncated, total {len(state.summary)} chars]")
        trace()

    context_manager.enqueue = traced_enqueue
    context_manager._update_summary = traced_update_summary


def _install_memory_extractor_tracing(context_manager):
    """给 MemoryExtractor 安装追踪钩子。"""
    extractor = context_manager._get_memory_extractor()
    _original_extract = extractor.extract

    async def traced_extract(message: dict) -> list[str]:
        role = message.get("role", "?")
        content = str(message.get("content", ""))

        trace_separator("MEMORY EXTRACTION", "-")
        trace(f"[EXTRACT INPUT] {role} message ({len(content)} chars):")
        trace(f"  {content[:500]}")
        if len(content) > 500:
            trace(f"  ... [truncated, full {len(content)} chars]")
        trace()

        # 先检查规则过滤
        if not extractor._should_extract(content):
            trace("[EXTRACT FILTERED] Message did not pass rule filter (too short or noise pattern)")
            trace()
            return []

        result = await _original_extract(message)

        if result:
            trace(f"[EXTRACT RESULT] {len(result)} new memories extracted:")
            for i, stmt in enumerate(result, 1):
                trace(f"  {i}. {stmt}")
        else:
            trace("[EXTRACT RESULT] No new memories (NO or all duplicates)")
        trace()

        return result

    extractor.extract = traced_extract


def _install_recall_store_tracing(recall_store):
    """给 RecallStore 安装追踪钩子。"""
    _original_add_recall = recall_store.add_recall_message
    _original_add_memory = recall_store.add_memory
    _original_check_dup = recall_store.check_duplicate

    def traced_add_recall(message: dict) -> str:
        role = message.get("role", "?")
        content = str(message.get("content", ""))
        doc_id = _original_add_recall(message)
        trace(f"[RECALL STORE] Stored recall_message: doc_id={doc_id} role={role} ({len(content)} chars)")
        return doc_id

    def traced_add_memory(memory_text: str):
        doc_id = _original_add_memory(memory_text)
        if doc_id:
            trace(f"[MEMORY STORE] Stored long-term memory: doc_id={doc_id} text={memory_text[:120]}")
        return doc_id

    def traced_check_dup(memory_text: str) -> bool:
        is_dup = _original_check_dup(memory_text)
        if is_dup:
            trace(f"[MEMORY DEDUP] Duplicate skipped: {memory_text[:80]}")
        return is_dup

    recall_store.add_recall_message = traced_add_recall
    recall_store.add_memory = traced_add_memory
    recall_store.check_duplicate = traced_check_dup


# ================================================================
#  测试辅助函数
# ================================================================

async def run_conversation_turn(context_manager, recall_store, thread_id,
                                role: str, content: str, label: str = "") -> dict:
    """执行一轮对话（一条消息入队）。"""
    logger.info("--- %s [%s] ---", label or "turn", role)
    logger.info("  Content (%d chars): %s...", len(content), content[:120])

    message = {"role": role, "content": content}
    await context_manager.enqueue(thread_id, message)

    stats_after = context_manager.get_stats(thread_id)
    return stats_after


async def test_context_and_memory_pipeline():
    """端到端测试：上下文压缩 + 记忆提取 + 检索验证。"""

    trace_separator("CONTEXT MEMORY SYSTEM - DETAILED TRACE LOG")
    trace(f"Started at: {_beijing_now().isoformat()}")
    trace()

    # ---- 加载配置 ----
    from agent.server.config import settings
    from agent.server.rag.config import rag_config

    trace(f"[CONFIG] llm_provider={settings.llm_provider}")
    trace(f"[CONFIG] fifo_max_tokens={settings.fifo_max_tokens}")
    trace(f"[CONFIG] memory_extraction_model={settings.memory_extraction_model or settings.effective_fast_model}")
    trace(f"[CONFIG] memory_extraction_enabled={settings.memory_extraction_enabled}")
    trace(f"[CONFIG] recall_storage_enabled={settings.recall_storage_enabled}")
    trace()

    # ---- ChromaDB client ----
    import chromadb
    chroma_dir = rag_config.chroma_persist_path
    logger.info("ChromaDB persist dir: %s", chroma_dir)
    trace(f"[INIT] ChromaDB persist_dir={chroma_dir}")
    chroma_client = chromadb.PersistentClient(path=str(chroma_dir))

    # ---- Embedder ----
    from agent.server.rag.embedder import Embedder
    embedder = Embedder(
        model_name=rag_config.embedding_model,
        local_path=rag_config.embedding_local_path or None,
    )
    embedder.start_loading()
    logger.info("Waiting for Embedder to load...")
    for i in range(60):
        if embedder.is_ready:
            break
        await asyncio.sleep(1)
    if not embedder.is_ready:
        raise RuntimeError("Embedder failed to load within 60 seconds")
    logger.info("Embedder ready: %s", rag_config.embedding_model)
    trace(f"[INIT] Embedder ready: {rag_config.embedding_model}")
    trace()

    # ---- LLM provider ----
    if settings.llm_provider == "deepseek":
        from agent.server.llm.deepseek_provider import DeepSeekProvider
        llm = DeepSeekProvider()
        logger.info("LLM provider: DeepSeek (%s)", settings.deepseek_model)
        trace(f"[INIT] LLM: DeepSeek model={settings.deepseek_model}")
    else:
        from agent.server.llm.anthropic_provider import AnthropicProvider
        llm = AnthropicProvider()
        logger.info("LLM provider: Anthropic (%s)", settings.anthropic_model)
        trace(f"[INIT] LLM: Anthropic model={settings.anthropic_model}")
    trace()

    # ---- RecallStore ----
    from agent.server.context.recall_store import RecallStore
    recall_store = RecallStore(chroma_client=chroma_client, embedder=embedder)
    logger.info("RecallStore ready: %s", recall_store.stats)
    trace(f"[INIT] RecallStore: {recall_store.stats}")
    trace()

    # ---- ContextManager ----
    from agent.server.context.context_manager import ContextManager
    test_db_dir = _project_root / "tests" / "_test_data"
    test_db_dir.mkdir(parents=True, exist_ok=True)
    test_db_path = test_db_dir / "context_state_test.db"
    if test_db_path.exists():
        test_db_path.unlink()

    memory_model = settings.memory_extraction_model or settings.effective_fast_model
    TEST_FIFO_MAX_TOKENS = 1500

    context_manager = ContextManager(
        db_path=test_db_path,
        llm=llm,
        recall_store=recall_store,
        max_tokens=TEST_FIFO_MAX_TOKENS,
        summary_model=memory_model,
        memory_extraction_model=memory_model,
    )
    logger.info("ContextManager ready: model=%s, fifo_max_tokens=%d", memory_model, TEST_FIFO_MAX_TOKENS)
    trace(f"[INIT] ContextManager: db={test_db_path} fifo_max_tokens={TEST_FIFO_MAX_TOKENS}")
    trace()

    # ---- 安装追踪钩子 ----
    _install_tracing(context_manager)
    _install_memory_extractor_tracing(context_manager)
    _install_recall_store_tracing(recall_store)
    trace("[TRACING] All tracing hooks installed")
    trace()

    thread_id = SCENARIO["thread_id"]

    # ================================================================
    #  Phase 1: 播种关键信息
    # ================================================================
    trace_separator("PHASE 1: 播种关键信息")
    logger.info("Phase 1: 播种关键信息")

    for i, turn in enumerate(SCENARIO["seed_info"]):
        await run_conversation_turn(
            context_manager, recall_store, thread_id,
            role=turn["role"], content=turn["content"],
            label=f"SEED {i+1}: {turn['label']}",
        )
        await asyncio.sleep(1)

    trace_separator("Phase 1 结束: 上下文窗口")
    trace_context_window(context_manager, thread_id, "After Phase 1 (seed info)")
    trace_fifo_snapshot(context_manager, thread_id, "After Phase 1")

    # ================================================================
    #  Phase 2: 大量闲聊触发压缩
    # ================================================================
    trace_separator(f"PHASE 2: {SCENARIO['chitchat_rounds']} 轮闲聊触发压缩")
    logger.info("Phase 2: %d 轮闲聊触发压缩", SCENARIO["chitchat_rounds"])

    for i in range(SCENARIO["chitchat_rounds"]):
        topic = SCENARIO["chitchat_topics"][i % len(SCENARIO["chitchat_topics"])]
        await run_conversation_turn(
            context_manager, recall_store, thread_id,
            role="user", content=topic,
            label=f"Chitchat {i+1}/{SCENARIO['chitchat_rounds']} user",
        )

        ai_reply = (
            f"关于'{topic[:30]}'这个话题，让我来详细分析一下。"
            f"从文化角度来看，中国传统文化中有许多值得探讨的元素。"
            f"从历史角度分析，可以追溯到唐宋甚至更早的时期。"
            f"从文学研究的角度来说，这涉及到文本细读和语境分析的方法。"
            f"总的来说，这是一个有趣的话题，涉及多学科交叉的视角。" * 3
        )
        await run_conversation_turn(
            context_manager, recall_store, thread_id,
            role="assistant", content=ai_reply,
            label=f"Chitchat {i+1}/{SCENARIO['chitchat_rounds']} assistant",
        )
        await asyncio.sleep(0.8)

        if (i + 1) % 5 == 0:
            stats = context_manager.get_stats(thread_id)
            logger.info(">>> Snapshot @ round %d: msgs=%d tokens=%d/%d summary_chars=%d",
                        i + 1, stats["message_count"], stats["total_tokens"],
                        TEST_FIFO_MAX_TOKENS, stats["summary_length"])
            trace_fifo_snapshot(context_manager, thread_id, f"Round {i+1}")

    # 等待异步记忆提取完成
    await asyncio.sleep(2)

    trace_separator("Phase 2 结束: 最终上下文窗口")
    trace_context_window(context_manager, thread_id, "After Phase 2 (all chitchat)")
    trace_fifo_snapshot(context_manager, thread_id, "After Phase 2 (final)")

    # RecallStore 统计
    trace_separator("RecallStore 统计")
    trace(f"[RECALL STORE STATS] {recall_store.stats}")

    # 列出 Recall Storage 中所有已存储的消息
    trace_separator("Recall Storage: 所有已存储的消息原文")
    try:
        recall_col = recall_store._recall_collection
        if recall_col and recall_col.count() > 0:
            all_recalls = recall_col.get()
            if all_recalls and all_recalls.get("documents"):
                for j, doc in enumerate(all_recalls["documents"]):
                    meta = all_recalls.get("metadatas", [{}])[j] if all_recalls.get("metadatas") else {}
                    role = meta.get("role", "?")
                    ts = meta.get("timestamp", "")[:19]
                    trace(f"[RECALL #{j+1}] {role} @ {ts}  ({len(doc)} chars)")
                    trace(f"  {doc[:400]}")
                    if len(doc) > 400:
                        trace(f"  ... [truncated, total {len(doc)} chars]")
                    trace()
    except Exception as e:
        trace(f"[RECALL STORE] Failed to list recall messages: {e}")

    # 列出所有长期记忆
    trace_separator("Long-term Memories: 所有已提取的长期记忆")
    try:
        mem_col = recall_store._memory_collection
        if mem_col and mem_col.count() > 0:
            all_mems = mem_col.get()
            if all_mems and all_mems.get("documents"):
                for j, doc in enumerate(all_mems["documents"]):
                    trace(f"[MEMORY #{j+1}] {doc}")
    except Exception as e:
        trace(f"[MEMORY STORE] Failed to list memories: {e}")

    # ================================================================
    #  Phase 3: 验证检索
    # ================================================================
    trace_separator("PHASE 3: 验证检索")
    logger.info("Phase 3: 验证检索")

    verification_query = SCENARIO["verification_query"]

    # 检索长期记忆
    trace_separator("3a: 检索长期记忆", "-")
    memories = recall_store.search_memories(verification_query, top_k=5)
    if memories:
        logger.info("Long-term memories (%d results):", len(memories))
        trace(f"[SEARCH MEMORIES] query='{verification_query}' -> {len(memories)} results")
        for i, m in enumerate(memories):
            logger.info("  [%d] score=%.3f: %s", i + 1, m["score"], m["content"])
            trace(f"  [{i+1}] score={m['score']:.4f}: {m['content']}")
    else:
        logger.warning("No long-term memories found! Memory extraction may have failed.")
        trace("[SEARCH MEMORIES] No results found")
    trace()

    # 检索 Recall Storage
    trace_separator("3b: 检索 Recall Storage", "-")
    recalls = recall_store.search_recall_messages(verification_query, top_k=3)
    if recalls:
        logger.info("Recall Storage (%d results):", len(recalls))
        trace(f"[SEARCH RECALLS] query='{verification_query}' -> {len(recalls)} results")
        for i, r in enumerate(recalls):
            meta = r.get("metadata", {})
            logger.info("  [%d] score=%.3f [%s] @ %s: %s...",
                        i + 1, r["score"], meta.get("role", "?"),
                        meta.get("timestamp", "")[:19], r["content"][:120])
            trace(f"  [{i+1}] score={r['score']:.4f} [{meta.get('role', '?')}] @ {meta.get('timestamp', '')[:19]}")
            trace(f"  {r['content'][:400]}")
            if len(r["content"]) > 400:
                trace(f"  ... [truncated, total {len(r['content'])} chars]")
            trace()
    else:
        logger.warning("No recall messages found.")
        trace("[SEARCH RECALLS] No results found")
    trace()

    # 调用 retrieve_memory 工具（模拟 LLM 调用）
    trace_separator("3c: 模拟 LLM 调用 retrieve_memory 工具", "-")
    from agent.server.mcp.tools.retrieve_memory import set_recall_store, handler as rm_handler
    set_recall_store(recall_store)
    trace(f"[TOOL CALL] retrieve_memory(query='{verification_query}', source='all')")
    trace()
    result = await rm_handler(query=verification_query, source="all")
    trace("[TOOL RESULT]")
    trace(result["result"])
    trace()
    print("\n" + result["result"])

    # ================================================================
    #  Phase 4: 断言验证
    # ================================================================
    trace_separator("PHASE 4: 断言验证")
    logger.info("Phase 4: 断言验证")

    all_memories_text = " ".join(m["content"] for m in memories)
    expected_keywords = SCENARIO["expected_keywords"]

    passed = True
    for keyword in expected_keywords:
        if keyword in all_memories_text:
            logger.info("[PASS] Keyword '%s' found in long-term memories", keyword)
            trace(f"[PASS] Keyword '{keyword}' found in long-term memories")
        else:
            passed = False
            logger.error("[FAIL] Keyword '%s' NOT found in long-term memories!", keyword)
            trace(f"[FAIL] Keyword '{keyword}' NOT found in long-term memories")

    trace()

    if passed:
        msg = "ALL TESTS PASSED - Context Memory Pipeline is working correctly"
        logger.info(msg)
        trace(f"[RESULT] {msg}")
    else:
        msg = "SOME TESTS FAILED - Check logs for details"
        logger.error(msg)
        trace(f"[RESULT] {msg}")

    # Cleanup
    await context_manager.close()

    # 关闭 trace 文件
    global _trace_file
    if _trace_file:
        _trace_file.close()
        _trace_file = None

    logger.info("Trace log saved to: %s", TRACE_LOG_PATH)
    return passed


if __name__ == "__main__":
    start = time.monotonic()
    success = asyncio.run(test_context_and_memory_pipeline())
    elapsed = time.monotonic() - start
    logger.info("Test completed in %.1f seconds", elapsed)
    trace(f"\nTest completed in {elapsed:.1f} seconds")
    sys.exit(0 if success else 1)
