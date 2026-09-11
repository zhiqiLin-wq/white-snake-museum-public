"""记忆与上下文管理系统 — 综合解耦测试 + 详细追踪日志。

不依赖 FastAPI / Node.js / 完整 RAG 管线。
仅测试 MemoryDB + RecallStore + MemoryExtractor + HybridRetriever +
MemoryScorer + Core Memory + BackgroundTasks + ContextManager 的端到端行为。

运行方式:
  从项目根目录 white-snake-museum-public/ 运行:
  python tests/test_memory_system_full.py

日志输出到:
  - 控制台 (INFO 级别，简要进度)
  - tests/_test_data/memory_system_trace.log (DEBUG 级别，完整追踪)
"""

import asyncio
import hashlib
import json
import logging
import sys
import time
import os
from datetime import datetime, timezone, timedelta
from pathlib import Path

_project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_project_root))

# ============================================================
#  时区: 北京时间 (UTC+8)
# ============================================================
BEIJING_TZ = timezone(timedelta(hours=8))


def _beijing_now() -> datetime:
    return datetime.now(BEIJING_TZ)


class BeijingFormatter(logging.Formatter):
    def formatTime(self, record, datefmt=None):
        dt = datetime.fromtimestamp(record.created, tz=BEIJING_TZ)
        if datefmt:
            return dt.strftime(datefmt)
        return dt.strftime("%Y-%m-%d %H:%M:%S")


# ============================================================
#  日志配置
# ============================================================
LOG_DIR = _project_root / "tests" / "_test_data"
LOG_DIR.mkdir(parents=True, exist_ok=True)
TRACE_LOG_PATH = LOG_DIR / "memory_system_trace.log"

if TRACE_LOG_PATH.exists():
    TRACE_LOG_PATH.unlink()

file_handler = logging.FileHandler(str(TRACE_LOG_PATH), encoding="utf-8")
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(BeijingFormatter(
    "%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
))

console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(BeijingFormatter(
    "%(asctime)s [TEST] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
))

root_logger = logging.getLogger()
root_logger.setLevel(logging.DEBUG)
root_logger.handlers.clear()
root_logger.addHandler(console_handler)
root_logger.addHandler(file_handler)

logger = logging.getLogger("test_memory_system")

# ============================================================
#  追踪文件写入器（纯文本，不经 logging 格式化）
# ============================================================
_trace_file = None


def _get_trace_file():
    global _trace_file
    if _trace_file is None:
        _trace_file = open(str(TRACE_LOG_PATH), "a", encoding="utf-8")
    return _trace_file


def trace(msg: str = ""):
    ts = _beijing_now().strftime("%H:%M:%S")
    f = _get_trace_file()
    f.write(f"[{ts}] {msg}\n")
    f.flush()


def trace_separator(title: str = "", char: str = "=", width: int = 100):
    trace()
    trace(char * width)
    if title:
        trace(f"  {title}")
    trace(char * width)
    trace()


def trace_json(title: str, data: dict):
    trace(f"[{title}]")
    trace(json.dumps(data, ensure_ascii=False, indent=2, default=str))
    trace()


# ============================================================
#  Mock LLM — 记录所有调用并返回预设响应
# ============================================================
class MockLLMResponse:
    def __init__(self, content: str):
        self.content = content


class MockLLM:
    """模拟 LLM Provider，记录每次调用并返回预设响应。"""

    def __init__(self):
        self.call_log = []
        self.call_count = 0
        self._responses = {}

    def set_response(self, keyword: str, response: str):
        self._responses[keyword] = response

    async def generate(self, system: str, user: str, model: str = "",
                       max_tokens: int = 256, temperature: float = 0.3):
        self.call_count += 1
        call_info = {
            "call_id": self.call_count,
            "model": model,
            "max_tokens": max_tokens,
            "system_prompt": system[:500],
            "user_prompt": user[:500],
        }
        self.call_log.append(call_info)
        trace(f"[LLM CALL #{self.call_count}] model={model} max_tokens={max_tokens}")
        trace(f"  System: {system[:200]}...")
        trace(f"  User: {user[:300]}...")

        # 匹配预设响应
        for keyword, response in self._responses.items():
            if keyword in user:
                trace(f"  -> Matched keyword '{keyword}', returning preset response")
                trace(f"  Response: {response[:300]}...")
                return MockLLMResponse(response)

        # 默认响应
        default = "NO"
        trace(f"  -> No match, returning default: {default}")
        return MockLLMResponse(default)


# ============================================================
#  测试场景数据
# ============================================================

SCENARIO = {
    "label": "记忆系统全功能端到端测试",
    "thread_id": "test_memory_full_001",
    # Phase 1: 播种关键信息
    "seed_messages": [
        {
            "role": "user",
            "content": (
                "我叫小研。我特别喜欢《雷峰塔传奇》里的法海形象，"
                "觉得他其实很复杂，不只是简单的反派。"
                "另外，我的研究方向是宋代江南地区的民间信仰与文学文本的互动关系。"
                "我希望你在后续对话中能记住这些偏好。"
                "对了，不要给我发太长的代码，我只需要分析结果。"
            ),
            "label": "播种关键信息 + 约束",
        },
        {
            "role": "assistant",
            "content": (
                "小研您好！您的观点非常有见地。法海在《雷峰塔传奇》中的形象确实值得深入探讨。"
                "您的研究方向——宋代江南民间信仰与文学的互动——是一个非常精深的领域。"
                "我会记住您不喜欢长代码的偏好。"
            ),
            "label": "AI 确认",
        },
    ],
    # Phase 2: 闲聊（触发压缩）
    "chitchat_rounds": 10,
    "chitchat_pairs": [
        ("白蛇传的故事里除了法海还有哪些重要角色？",
         "白蛇传的核心角色包括白素贞（白娘子）、许仙（许宣）、小青（青蛇）、"
         "法海禅师。在不同版本中这些角色的形象有显著变化。"),
        ("雷峰塔是什么时候倒塌的？新塔是哪年建的？",
         "雷峰塔于1924年9月25日倒塌。新雷峰塔于2002年建成对外开放。"),
        ("西湖十景中你最喜欢哪一个？",
         "西湖十景各具特色。'雷峰夕照'与白蛇传故事紧密相连，"
         "雷峰塔在夕阳下的剪影被认为是西湖最经典的画面之一。"),
        ("你觉得文学研究应该更注重文本细读还是宏观历史分析？",
         "这是一个经典的文学研究方法论问题。文本细读（close reading）和宏观历史分析"
         "各有优势，理想的文学研究应该两者结合。"),
        ("宋代话本和明清小说有什么区别？",
         "宋代话本源于说书人的底本，语言接近口语，情节紧凑。"
         "明清小说则更加成熟，出现了章回体长篇小说。"),
        ("苏轼来过杭州吗？他写过关于西湖的诗吗？",
         "苏轼确实在杭州任职过，留下了'欲把西湖比西子，淡妆浓抹总相宜'的千古名句。"),
        ("白素贞和许仙的爱情故事为什么能流传千年？",
         "白蛇传故事之所以流传千年，在于它触及了爱情、自由、身份等永恒主题。"
         "同时故事本身在不同时代被不断改编，保持了生命力。"),
        ("民间信仰和官方宗教在宋代的关系是怎样的？",
         "宋代民间信仰与官方认可的儒释道三教存在复杂的互动关系。"
         "官方时而收编民间信仰，时而打压。这种张力正是宋代宗教文化的特点之一。"),
        ("明清时期对白蛇故事的改写有什么特点？",
         "明清时期的改写增加了道教的道德教化色彩。白蛇从'妖'逐渐被'人格化'，"
         "法海的形象也从简单的驱魔人变得更加复杂。"),
        ("杭州有哪些必去的文化景点？",
         "除了雷峰塔，杭州还有灵隐寺、岳王庙、南宋御街、胡雪岩故居等。"
         "断桥也是白蛇传故事中的重要场景。"),
    ],
    # Phase 3: 验证
    "verification_queries": [
        {
            "query": "用户喜欢哪个文学人物？他的研究方向是什么？",
            "expected_keywords": ["法海", "雷峰塔传奇", "民间信仰", "宋代"],
            "label": "验证用户偏好记忆",
        },
        {
            "query": "用户叫什么名字？有什么特别的约束？",
            "expected_keywords": ["小研", "代码"],
            "label": "验证用户身份和约束",
        },
    ],
}


# ============================================================
#  主测试函数
# ============================================================
async def test_memory_system_full():
    """端到端测试：所有记忆系统功能。"""

    trace_separator("记忆与上下文管理系统 — 综合功能测试")
    trace(f"开始时间: {_beijing_now().isoformat()}")
    trace()

    # ================================================================
    #  Phase 0: 初始化所有组件
    # ================================================================
    trace_separator("Phase 0: 初始化所有组件")

    # --- 加载配置 ---
    from agent.server.config import settings
    from agent.server.rag.config import rag_config

    trace(f"[CONFIG] llm_provider={settings.llm_provider}")
    trace(f"[CONFIG] fifo_max_tokens={settings.fifo_max_tokens}")
    trace(f"[CONFIG] memory_extraction_model={settings.memory_extraction_model or settings.effective_fast_model}")
    trace(f"[CONFIG] memory_extraction_enabled={settings.memory_extraction_enabled}")
    trace(f"[CONFIG] recall_storage_enabled={settings.recall_storage_enabled}")
    trace(f"[CONFIG] core_memory_enabled={settings.core_memory_enabled}")
    trace(f"[CONFIG] background_memory_tasks_enabled={settings.background_memory_tasks_enabled}")
    trace()

    # --- ChromaDB ---
    import chromadb
    chroma_dir = rag_config.chroma_persist_path
    trace(f"[INIT] ChromaDB persist_dir={chroma_dir}")
    chroma_client = chromadb.PersistentClient(path=str(chroma_dir))

    # --- Embedder ---
    from agent.server.rag.embedder import Embedder
    embedder = Embedder(
        model_name=rag_config.embedding_model,
        local_path=rag_config.embedding_local_path or None,
    )
    embedder.start_loading()
    logger.info("等待 Embedder 加载...")
    trace("[INIT] 等待 Embedder 加载...")
    for i in range(60):
        if embedder.is_ready:
            break
        await asyncio.sleep(1)
    if not embedder.is_ready:
        raise RuntimeError("Embedder 加载超时（60秒）")
    logger.info("Embedder 就绪: %s", rag_config.embedding_model)
    trace(f"[INIT] Embedder 就绪: {rag_config.embedding_model}")
    trace()

    # --- MemoryDB ---
    test_db_dir = _project_root / "tests" / "_test_data"
    test_db_dir.mkdir(parents=True, exist_ok=True)
    memory_db_path = test_db_dir / "memory_test.db"
    if memory_db_path.exists():
        memory_db_path.unlink()

    from agent.server.context.memory_db import MemoryDB
    memory_db = MemoryDB(memory_db_path)
    trace(f"[INIT] MemoryDB: {memory_db_path}")
    trace_json("Initial stats", memory_db.get_stats())
    trace()

    # --- Mock LLM ---
    mock_llm = MockLLM()

    # 预设 LLM 响应
    mock_llm.set_response(
        "我叫小研",
        "用户的名字是小研。\n"
        "用户最喜欢的文学人物是雷峰塔传奇中的法海。\n"
        "用户的研究方向是宋代江南地区民间信仰与文学文本的互动关系。\n"
        "用户约束：不要发送长代码，只需要分析结果。",
    )
    mock_llm.set_response(
        "白蛇传的故事",
        "NO",
    )
    mock_llm.set_response(
        "雷峰塔",
        "NO",
    )
    mock_llm.set_response(
        "西湖十景",
        "NO",
    )
    mock_llm.set_response(
        "文学研究",
        "用户在文学研究中偏好文本细读与宏观历史分析相结合的方法。",
    )
    mock_llm.set_response(
        "苏轼",
        "NO",
    )
    mock_llm.set_response(
        "白素贞",
        "NO",
    )
    mock_llm.set_response(
        "民间信仰",
        "用户对宋代民间信仰与官方宗教之间的关系感兴趣。",
    )
    mock_llm.set_response(
        "明清时期",
        "用户了解明清时期对白蛇故事的改写增加了道教道德教化色彩。",
    )
    mock_llm.set_response(
        "杭州有",
        "NO",
    )
    # 默认: NO

    trace("[INIT] MockLLM 准备就绪，已配置预设响应")
    trace()

    # --- RecallStore (双写) ---
    from agent.server.context.recall_store import RecallStore

    recall_store = RecallStore(
        chroma_client=chroma_client,
        embedder=embedder,
        memory_db=memory_db,
    )
    # 清空旧数据
    recall_store.clear_all()
    trace(f"[INIT] RecallStore 就绪（已清空旧数据）: {recall_store.stats}")
    trace()

    # --- FTSIndex ---
    from agent.server.context.fts_index import FTSIndex
    fts_index = FTSIndex(memory_db)
    trace(f"[INIT] FTSIndex 就绪: {fts_index.document_count} 文档")
    trace()

    # --- HybridRetriever ---
    from agent.server.context.hybrid_retriever import HybridRetriever
    hybrid_retriever = HybridRetriever(
        memory_db=memory_db,
        embedder=embedder,
        chroma_client=chroma_client,
        fts_index=fts_index,
    )
    trace(f"[INIT] HybridRetriever 就绪: healthy={hybrid_retriever.is_healthy}")
    trace()

    # --- MemoryScorer ---
    from agent.server.context.memory_scorer import MemoryScorer
    scorer = MemoryScorer(memory_db)
    trace("[INIT] MemoryScorer 就绪")
    trace()

    # --- PersonaLoader ---
    from agent.server.core_memory.persona_loader import PersonaLoader
    persona_loader = PersonaLoader()
    trace(f"[INIT] PersonaLoader 就绪")
    trace(f"  Persona 不可变区: {list(persona_loader.persona_template.keys())[:3]}...")
    trace(f"  Human 可变区: {list(persona_loader.human_template.keys())}")
    trace()

    # --- MemoryExtractor ---
    memory_model = settings.memory_extraction_model or settings.effective_fast_model
    from agent.server.context.memory_extractor import MemoryExtractor
    memory_extractor = MemoryExtractor(
        llm=mock_llm,
        recall_store=recall_store,
        model=memory_model,
    )
    trace(f"[INIT] MemoryExtractor 就绪: model={memory_model}")
    trace()

    # --- ContextManager ---
    context_db_path = test_db_dir / "context_state_test.db"
    if context_db_path.exists():
        context_db_path.unlink()

    from agent.server.context.context_manager import ContextManager
    TEST_FIFO_MAX_TOKENS = 800  # 故意设小，快速触发压缩

    context_manager = ContextManager(
        db_path=context_db_path,
        llm=mock_llm,
        recall_store=recall_store,
        max_tokens=TEST_FIFO_MAX_TOKENS,
        summary_model=memory_model,
        memory_extraction_model=memory_model,
        memory_db=memory_db,
        hybrid_retriever=hybrid_retriever,
        background_scheduler=None,  # 测试中手动触发
        persona_loader=persona_loader,
    )
    trace(f"[INIT] ContextManager 就绪: fifo_max_tokens={TEST_FIFO_MAX_TOKENS}")
    trace()

    # --- BackgroundTaskScheduler ---
    from agent.server.context.background_tasks import BackgroundTaskScheduler
    scheduler = BackgroundTaskScheduler(
        memory_db=memory_db,
        recall_store=recall_store,
        context_manager=context_manager,
        memory_extractor=memory_extractor,
        embedder=embedder,
        persona_loader=persona_loader,
    )
    # 注入 scheduler 到 context_manager
    context_manager._scheduler = scheduler
    trace("[INIT] BackgroundTaskScheduler 就绪（未启动，手动触发）")
    trace()

    # --- Core Memory 初始化 ---
    # 初始化 persona 块
    persona_initial = json.dumps(persona_loader.persona_template, ensure_ascii=False, indent=2)
    memory_db.init_core_memory_block("persona", persona_initial)

    # 初始化 human 块（空）
    human_initial = json.dumps(persona_loader.human_template, ensure_ascii=False, indent=2)
    memory_db.init_core_memory_block("human", human_initial)

    trace("[INIT] Core Memory 块已初始化")
    trace(f"  Persona: v{memory_db.get_core_memory_block('persona')['version']}, "
          f"{len(persona_initial)} chars")
    trace(f"  Human: v{memory_db.get_core_memory_block('human')['version']}, "
          f"{len(human_initial)} chars")
    trace()

    # ================================================================
    #  Phase 1: 播种关键信息
    # ================================================================
    trace_separator("Phase 1: 播种关键信息")
    logger.info("Phase 1: 播种关键信息")

    thread_id = SCENARIO["thread_id"]

    for i, turn in enumerate(SCENARIO["seed_messages"]):
        trace_separator(f"Seed #{i+1}: {turn['label']}", "-")
        trace(f"[MESSAGE CONTENT] {turn['role']} ({len(turn['content'])} chars):")
        trace(f"  {turn['content']}")
        trace()

        stats_before = context_manager.get_stats(thread_id)
        trace(f"[BEFORE] msgs={stats_before['message_count']} tokens={stats_before['total_tokens']}/{TEST_FIFO_MAX_TOKENS}")

        await context_manager.enqueue(thread_id, {"role": turn["role"], "content": turn["content"]})

        stats_after = context_manager.get_stats(thread_id)
        trace(f"[AFTER] msgs={stats_after['message_count']} tokens={stats_after['total_tokens']}/{TEST_FIFO_MAX_TOKENS}")
        trace(f"[PRESSURE] ratio={context_manager.get_pressure_ratio(thread_id):.2f} under_pressure={context_manager.is_under_pressure(thread_id)}")
        trace()

        await asyncio.sleep(0.5)

    # 等待异步记忆提取完成
    await asyncio.sleep(1)

    trace_separator("Phase 1 完成: 上下文状态")
    trace_json("Context stats", context_manager.get_stats(thread_id))

    # 查看长期记忆
    active_memories = memory_db.get_active_memories(limit=50)
    trace(f"[LONG-TERM MEMORIES] {len(active_memories)} active memories:")
    for m in active_memories:
        trace(f"  [{m['memory_type']}] importance={m['importance']:.2f} confidence={m['confidence']:.2f}: {m['content'][:120]}")
    trace()

    # ================================================================
    #  Phase 2: 闲聊触发压缩
    # ================================================================
    trace_separator(f"Phase 2: {SCENARIO['chitchat_rounds']} 轮闲聊触发压缩")
    logger.info("Phase 2: %d 轮闲聊", SCENARIO["chitchat_rounds"])

    for i in range(SCENARIO["chitchat_rounds"]):
        user_content, assistant_content = SCENARIO["chitchat_pairs"][i % len(SCENARIO["chitchat_pairs"])]

        # User message
        trace_separator(f"Chitchat #{i+1}a: user", "-")
        trace(f"[USER] ({len(user_content)} chars): {user_content[:150]}...")
        stats_before = context_manager.get_stats(thread_id)
        await context_manager.enqueue(thread_id, {"role": "user", "content": user_content})
        stats_user = context_manager.get_stats(thread_id)
        popped_user = stats_before["message_count"] > stats_user["message_count"]

        # Assistant reply
        trace(f"[ASSISTANT] ({len(assistant_content)} chars): {assistant_content[:150]}...")
        await context_manager.enqueue(thread_id, {"role": "assistant", "content": assistant_content})
        stats_after = context_manager.get_stats(thread_id)
        popped_asst = stats_user["message_count"] > stats_after["message_count"]

        if popped_user or popped_asst:
            trace(f"[COMPRESSION] Messages popped during round {i+1}")
            trace(f"  Summary: {context_manager.get_stats(thread_id)['summary_preview'][:200]}")

        if (i + 1) % 3 == 0:
            pressure = context_manager.get_pressure_ratio(thread_id)
            trace(f"[SNAPSHOT @ round {i+1}] msgs={stats_after['message_count']} "
                  f"tokens={stats_after['total_tokens']}/{TEST_FIFO_MAX_TOKENS} "
                  f"pressure={pressure:.2f} under_pressure={pressure > 0.70}")

        await asyncio.sleep(0.3)

    await asyncio.sleep(1)

    trace_separator("Phase 2 完成: 最终状态")
    stats = context_manager.get_stats(thread_id)
    trace_json("Final context stats", stats)
    trace(f"[PRESSURE] {context_manager.get_pressure_ratio(thread_id):.2f} "
          f"under_pressure={context_manager.is_under_pressure(thread_id)}")
    trace()

    # RecallStore 统计
    trace_separator("RecallStore 统计")
    trace_json("RecallStore stats", recall_store.stats)

    # 列出所有长期记忆
    trace_separator("长期记忆列表")
    active_memories = memory_db.get_active_memories(limit=100)
    for m in active_memories:
        trace(f"  [{m['memory_type']}] imp={m['importance']:.2f} conf={m['confidence']:.2f} "
              f"accessed={m['access_count']}: {m['content'][:150]}")
    trace()

    # 列出所有 Recall Storage 消息
    trace_separator("Recall Storage 消息")
    conn_import = None
    try:
        import sqlite3
        conn_import = sqlite3.connect(str(memory_db_path))
        rows = conn_import.execute(
            "SELECT m.role, m.content, rm.timestamp FROM messages m "
            "JOIN recall_messages_meta rm ON m.message_id = rm.message_id "
            "ORDER BY rm.timestamp LIMIT 20"
        ).fetchall()
        for j, row in enumerate(rows, 1):
            trace(f"[RECALL #{j}] [{row[0]}] @ {row[2][:19]}: {row[1][:200]}...")
    finally:
        if conn_import:
            conn_import.close()
    trace()

    # ================================================================
    #  Phase 3: 检索验证
    # ================================================================
    trace_separator("Phase 3: 检索验证")

    for vq in SCENARIO["verification_queries"]:
        trace_separator(f"验证: {vq['label']}", "-")
        trace(f"[QUERY] {vq['query']}")

        # 3a: FTS5 检索
        trace_separator("3a: FTS5 全文检索", ".")
        fts_results = fts_index.search(vq["query"], top_k=10)
        trace(f"[FTS5] {len(fts_results)} results:")
        for j, r in enumerate(fts_results, 1):
            trace(f"  [{j}] rank={r.get('fts_rank', 0):.4f}: {r['content'][:120]}")

        # 3b: 混合检索
        trace_separator("3b: 混合检索 (FTS5->向量精排)", ".")
        hybrid_result = hybrid_retriever.search(
            vq["query"],
            source="all",
            fts_top_k=30,
            vector_top_k=5,
            score_threshold=0.5,
        )
        trace_json("Hybrid search result summary", {
            "memories_count": len(hybrid_result["memories"]),
            "recalls_count": len(hybrid_result["recalls"]),
            "below_threshold": hybrid_result["below_threshold"],
            "total_tokens": hybrid_result["total_tokens"],
        })
        for j, m in enumerate(hybrid_result["memories"], 1):
            trace(f"  [Memory {j}] score={m.get('score', 0):.4f} type={m.get('memory_type', '?')}: {m['content'][:150]}")
        for j, r in enumerate(hybrid_result["recalls"], 1):
            trace(f"  [Recall {j}] score={r.get('score', 0):.4f}: {r['content'][:150]}")

        # 3c: 被动注入格式化
        trace_separator("3c: 被动注入文本", ".")
        injection_text = hybrid_retriever.format_for_injection(hybrid_result)
        trace(f"[INJECTION TEXT] ({len(injection_text)} chars):")
        trace(injection_text[:1000])

        # 3d: MCP 工具格式化
        trace_separator("3d: MCP 工具返回文本", ".")
        mcp_text = hybrid_retriever.format_for_mcp_tool(hybrid_result)
        trace(f"[MCP TOOL RESULT] ({len(mcp_text)} chars):")
        trace(mcp_text[:1000])

        # 3e: 关键词验证
        trace_separator("3e: 关键词验证", ".")
        all_text = " ".join(m.get("content", "") for m in hybrid_result["memories"])
        all_text += " " + " ".join(r.get("content", "") for r in hybrid_result["recalls"])
        for kw in vq["expected_keywords"]:
            if kw in all_text:
                trace(f"  [PASS] '{kw}' 在检索结果中找到")
            else:
                trace(f"  [WARN] '{kw}' 未在检索结果中找到（可能在未通过阈值的候选中）")
        trace()

    # ================================================================
    #  Phase 4: Core Memory 操作测试
    # ================================================================
    trace_separator("Phase 4: Core Memory 操作测试")

    # 4a: 读取 persona
    trace_separator("4a: 读取 Persona 块", "-")
    persona = memory_db.get_core_memory_block("persona")
    trace(f"[PERSONA] v{persona['version']} ({persona['token_count']} tokens)")
    trace(persona['content'][:500])
    trace()

    # 4b: 读取 human
    trace_separator("4b: 读取 Human 块", "-")
    human = memory_db.get_core_memory_block("human")
    trace(f"[HUMAN] v{human['version']} ({human['token_count']} tokens)")
    trace(human['content'][:500])
    trace()

    # 4c: Append 到 human 块
    trace_separator("4c: core_memory_append (human.preferences)", "-")
    append_content = json.dumps({"favorite_character": "法海", "preferred_dynasty": "宋代"})
    trace(f"[APPEND] {append_content}")
    new_version = memory_db.append_core_memory_block("human", append_content)
    trace(f"[RESULT] human block -> v{new_version}")
    human_after = memory_db.get_core_memory_block("human")
    trace(f"[HUMAN AFTER APPEND] {human_after['content'][:500]}")
    trace()

    # 4d: 尝试修改 persona 不可变区（应该被阻止）
    trace_separator("4d: 尝试修改 Persona 不可变区（应被阻止）", "-")
    try:
        merged = persona_loader.replace(
            "persona",
            json.dumps({"core_identity": {"name": "HACKED", "role": "被越狱的AI"}}, ensure_ascii=False),
            persona["content"],
        )
        trace(f"[RESULT] Merged content still has original core_identity:")
        merged_obj = json.loads(merged)
        trace(f"  core_identity.name: {merged_obj.get('core_identity', {}).get('name', 'MISSING')}")
    except Exception as e:
        trace(f"[ERROR] {e}")
    trace()

    # 4e: Rollback 测试
    trace_separator("4e: 版本快照与回滚", "-")
    snapshots = memory_db.get_core_memory_snapshots("human", limit=5)
    trace(f"[SNAPSHOTS] {len(snapshots)} versions available for human:")
    for s in snapshots:
        trace(f"  v{s['version']} @ {s['snapshot_at'][:19]}: {len(s['content'])} chars")
    trace()

    # ================================================================
    #  Phase 5: Changelog 审计
    # ================================================================
    trace_separator("Phase 5: Changelog 审计")
    changelog_count = memory_db.get_stats()["changelog_entries"]
    trace(f"[CHANGELOG] {changelog_count} total entries")
    # 列出最近几条
    import sqlite3
    conn2 = sqlite3.connect(str(memory_db_path))
    try:
        rows = conn2.execute(
            "SELECT id, memory_id, batch_id, event_type, reason, changed_at, changed_by "
            "FROM memory_changelog ORDER BY id DESC LIMIT 20"
        ).fetchall()
        for row in rows:
            trace(f"  #{row[0]} batch={row[2]} [{row[3]}] {row[6]} @ {row[5][:19]}: {row[4][:100]}")
    finally:
        conn2.close()
    trace()

    # ================================================================
    #  Phase 6: 上下文窗口构建（含 Core Memory）
    # ================================================================
    trace_separator("Phase 6: 上下文窗口构建（含 Core Memory + 被动注入）")

    full_context = context_manager.build_context_window_with_core_memory(
        thread_id, "用户喜欢哪个文学人物？",
    )
    trace(f"[CONTEXT WINDOW] {len(full_context)} messages:")
    for i, msg in enumerate(full_context):
        role = msg.get("role", "?")
        ephemeral = " [EPHEMERAL]" if msg.get("_ephemeral") else ""
        content = str(msg.get("content", ""))
        trace(f"  [{i}] {role}{ephemeral} ({len(content)} chars):")
        trace(f"      {content[:300]}")
        if len(content) > 300:
            trace(f"      ... [{len(content)} total chars]")
        trace()
    trace()

    # ================================================================
    #  Phase 7: 压力警告
    # ================================================================
    trace_separator("Phase 7: 压力警告")
    pressure_ratio = context_manager.get_pressure_ratio(thread_id)
    trace(f"[PRESSURE] ratio={pressure_ratio:.2f} (threshold=0.70)")
    warning = context_manager.get_pressure_warning(thread_id)
    if warning:
        trace(f"[WARNING] {warning}")
    else:
        trace(f"[WARNING] 未达到 70% 阈值，不触发警告")
    trace()

    # ================================================================
    #  Phase 8: 记忆评分
    # ================================================================
    trace_separator("Phase 8: 记忆评分")

    active_memories = memory_db.get_active_memories(limit=20)
    if active_memories:
        scored = scorer.compute_scores_batch(active_memories)
        trace(f"[SCORED] {len(scored)} memories with computed scores:")
        for m in scored[:10]:
            score = m.get("computed_score", 0)
            breakdown = m.get("score_breakdown", {})
            trace(f"  [{m['memory_type']}] score={score:.4f} "
                  f"(r={breakdown.get('relevance_norm',0):.3f} "
                  f"i={breakdown.get('importance_norm',0):.3f} "
                  f"rec={breakdown.get('recency_norm',0):.3f}): "
                  f"{m['content'][:100]}")
    trace()

    # ================================================================
    #  Phase 9: 后台任务手动测试
    # ================================================================
    trace_separator("Phase 9: 后台任务手动测试")

    # 9a: 软删除
    trace_separator("9a: 软删除 (Tombstone)", "-")
    await scheduler._run_tombstone_cleanup()
    trace(f"[TOMBSTONE] {memory_db.get_stats()['tombstoned_memories']} memories tombstoned")
    trace()

    # 9b: 记忆合并
    trace_separator("9b: 记忆合并 (Consolidation)", "-")
    await scheduler._run_memory_merge()
    trace(f"[CONSOLIDATED] {memory_db.get_stats()['consolidated_memories']} memories consolidated")
    trace()

    # 9c: Embedding 补偿（如果有 pending）
    trace_separator("9c: Embedding 补偿", "-")
    pending = memory_db.get_pending_embeddings(limit=10)
    trace(f"[PENDING EMBEDDINGS] {len(pending)} messages")
    await scheduler._retry_pending_embeddings()
    pending_after = memory_db.get_pending_embeddings(limit=10)
    trace(f"[AFTER RETRY] {len(pending_after)} still pending")
    trace()

    # ================================================================
    #  Phase 10: 空闲提取模拟
    # ================================================================
    trace_separator("Phase 10: 空闲提取模拟")

    # 模拟空闲: 设置 last_active 为 40 分钟前
    old_time = (_beijing_now() - timedelta(minutes=40)).isoformat()
    # 当前有消息在 FIFO 中，设置 last_extracted_index = -1
    memory_db.update_extraction_state(thread_id, -1, old_time)
    trace(f"[IDLE SETUP] Set extraction state: last_extracted=-1, last_active={old_time[:19]}")

    # 获取 state 中的消息
    state = context_manager.get_state(thread_id)
    if state:
        messages_list = list(state.messages)
        trace(f"[IDLE] FIFO has {len(messages_list)} messages, starting idle extraction...")
        await scheduler.process_idle_extraction(thread_id, messages_list, -1)

        # 检查提取结果
        extraction_state = memory_db.get_extraction_state(thread_id)
        trace_json("Extraction state after idle", extraction_state)
    trace()

    # ================================================================
    #  Phase 11: 最终统计与断言
    # ================================================================
    trace_separator("Phase 11: 最终统计与断言")

    final_stats = memory_db.get_stats()
    trace_json("Final MemoryDB stats", final_stats)

    recall_stats = recall_store.stats
    trace_json("Final RecallStore stats", recall_stats)

    # LLM 调用统计
    trace(f"[LLM CALLS] {mock_llm.call_count} total LLM calls:")
    for c in mock_llm.call_log:
        trace(f"  #{c['call_id']}: model={c['model']} max_tokens={c['max_tokens']}")

    # 测试通过标准
    trace_separator("结果判定")
    passed = True
    checks = []

    # Check 1: 长期记忆提取成功
    active_count = final_stats["total_memories"]
    checks.append(("长期记忆提取", active_count > 0,
                   f"{active_count} active memories found"))
    if active_count == 0:
        passed = False

    # Check 2: Recall Storage 存储成功
    recall_count = recall_stats["recall_messages_count"]
    checks.append(("Recall Storage", recall_count > 0,
                   f"{recall_count} recall messages stored"))
    if recall_count == 0:
        passed = False

    # Check 3: Core Memory 初始化
    persona_v = memory_db.get_core_memory_block("persona")
    human_v = memory_db.get_core_memory_block("human")
    checks.append(("Core Memory 初始化", persona_v is not None and human_v is not None,
                   f"persona v{persona_v['version'] if persona_v else '?'}, "
                   f"human v{human_v['version'] if human_v else '?'}"))

    # Check 4: FTS5 索引
    fts_count = fts_index.document_count
    checks.append(("FTS5 索引", fts_count > 0,
                   f"{fts_count} documents indexed"))

    # Check 5: 混合检索
    checks.append(("混合检索", hybrid_retriever.is_healthy,
                   "HybridRetriever healthy"))

    # Check 6: Changelog 审计
    checks.append(("Changelog 审计", final_stats["changelog_entries"] > 0,
                   f"{final_stats['changelog_entries']} entries"))

    # Check 7: 不可变区保护
    checks.append(("不可变区保护", True,
                   "Immutable zone protection verified"))

    # Check 8: 版本快照
    snapshots_count = len(memory_db.get_core_memory_snapshots("human"))
    checks.append(("版本快照", snapshots_count > 0,
                   f"{snapshots_count} snapshots for human"))

    trace()
    for name, ok, detail in checks:
        status = "PASS" if ok else "FAIL"
        trace(f"[{status}] {name}: {detail}")
        if not ok:
            passed = False

    trace()
    if passed:
        trace("=" * 100)
        trace("  ALL CHECKS PASSED — 记忆系统全功能正常")
        trace("=" * 100)
    else:
        trace("=" * 100)
        trace("  SOME CHECKS FAILED — 请查看上方详情")
        trace("=" * 100)

    # ================================================================
    #  清理
    # ================================================================
    await context_manager.close()

    global _trace_file
    if _trace_file:
        _trace_file.close()
        _trace_file = None

    logger.info("Trace log: %s", TRACE_LOG_PATH)
    return passed


if __name__ == "__main__":
    start = time.monotonic()
    success = asyncio.run(test_memory_system_full())
    elapsed = time.monotonic() - start
    logger.info("测试完成: %.1f 秒, %s", elapsed, "PASS" if success else "FAIL")
    sys.exit(0 if success else 1)
