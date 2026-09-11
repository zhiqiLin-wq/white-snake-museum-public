"""记忆与上下文管理系统 — 完整功能验证测试套件。

10 个测试组，覆盖 MemoryDB / RecallStore / MemoryExtractor /
HybridRetriever / MemoryScorer / Core Memory / ContextManager /
BackgroundTasks / MCP Tools / 端到端流水线。

解耦测试: 不依赖 FastAPI / Node.js / 完整 RAG 管线。
使用 MockLLM（中文响应）替代真实 LLM 调用。

运行方式:
  cd white-snake-museum-public
  python tests/test_memory_system_exhaustive.py

输出:
  - 控制台: 每步执行结果 (PASS/FAIL)
  - tests/_test_data/trace_exhaustive.log: 完整追踪日志
  - tests/_test_data/trace_exhaustive.txt: 纯文本追踪（消息原文）
"""

import asyncio
import hashlib
import json
import logging
import os
import shutil
import sqlite3
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


def bj_now() -> datetime:
    return datetime.now(BEIJING_TZ)


# ============================================================
#  追踪文件
# ============================================================
LOG_DIR = _project_root / "tests" / "_test_data"
LOG_DIR.mkdir(parents=True, exist_ok=True)
TRACE_PATH = LOG_DIR / "trace_exhaustive.log"
TRACE_TXT_PATH = LOG_DIR / "trace_exhaustive.txt"

for p in [TRACE_PATH, TRACE_TXT_PATH]:
    if p.exists():
        p.unlink()

_tf_log = None
_tf_txt = None


def _get_log():
    global _tf_log
    if _tf_log is None:
        _tf_log = open(str(TRACE_PATH), "a", encoding="utf-8")
    return _tf_log


def _get_txt():
    global _tf_txt
    if _tf_txt is None:
        _tf_txt = open(str(TRACE_TXT_PATH), "a", encoding="utf-8")
    return _tf_txt


def trace(msg: str = ""):
    ts = bj_now().strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    _get_log().write(line + "\n")
    _get_log().flush()


def trace_txt(msg: str = ""):
    _get_txt().write(msg + "\n")
    _get_txt().flush()


def sep(title: str = "", char: str = "=", w: int = 100):
    trace()
    trace(char * w)
    if title:
        trace(f"  {title}")
    trace(char * w)
    trace()


def trace_json(title: str, data):
    trace(f"[{title}]")
    trace(json.dumps(data, ensure_ascii=False, indent=2, default=str))
    trace()


# ============================================================
#  日志配置
# ============================================================
class BjFormatter(logging.Formatter):
    def formatTime(self, record, datefmt=None):
        dt = datetime.fromtimestamp(record.created, tz=BEIJING_TZ)
        return dt.strftime("%H:%M:%S") if datefmt is None else dt.strftime(datefmt)


console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(BjFormatter("%(asctime)s %(message)s", datefmt="%H:%M:%S"))

root_logger = logging.getLogger()
root_logger.setLevel(logging.DEBUG)
root_logger.handlers.clear()
root_logger.addHandler(console_handler)

logger = logging.getLogger("exhaustive_test")

# ============================================================
#  测试结果追踪
# ============================================================
_test_results = {"passed": 0, "failed": 0, "total": 0, "details": []}


def tc(grp: str, tc_id: str, description: str, condition: bool, detail: str = ""):
    """记录一个测试用例结果。"""
    _test_results["total"] += 1
    status = "PASS" if condition else "FAIL"
    if condition:
        _test_results["passed"] += 1
    else:
        _test_results["failed"] += 1
    entry = f"[{status}] {grp}-{tc_id}: {description}"
    if detail:
        entry += f" — {detail}"
    _test_results["details"].append(entry)
    logger.info(entry)
    trace(entry)


# ============================================================
#  Mock LLM — 中文响应
# ============================================================
class MockLLMResponse:
    def __init__(self, content: str):
        self.content = content


class MockLLM:
    def __init__(self):
        self.call_log = []
        self.call_count = 0

    def set_responses(self, mapping: dict):
        self._responses = mapping

    async def generate(self, system: str, user: str, model: str = "",
                       max_tokens: int = 256, temperature: float = 0.3):
        self.call_count += 1
        info = {"id": self.call_count, "model": model, "max_tokens": max_tokens,
                "system": system[:120], "user": user[:200]}
        self.call_log.append(info)
        trace(f"  [LLM #{self.call_count}] model={model} sys='{system[:80]}...' user='{user[:120]}...'")

        for kw, resp in getattr(self, '_responses', {}).items():
            if kw in user:
                trace(f"    -> matched '{kw}', response='{resp[:80]}...'")
                return MockLLMResponse(resp)

        trace("    -> no match, default NO")
        return MockLLMResponse("NO")


# ============================================================
#  Test Group 1: MemoryDB (数据层)
# ============================================================
async def test_group_1_memory_db(test_dir: Path) -> dict:
    sep("TG1: MemoryDB — SQLite 数据层测试", "=")
    trace_txt("\n" + "=" * 80)
    trace_txt("  TG1: MemoryDB — SQLite 数据层")
    trace_txt("=" * 80 + "\n")

    from agent.server.context.memory_db import MemoryDB

    db_path = test_dir / "tg1_memory.db"
    if db_path.exists():
        db_path.unlink()

    # TC1.1: 创建数据库
    db = MemoryDB(db_path)
    stats = db.get_stats()
    conn = sqlite3.connect(str(db_path))
    try:
        existing_tables = {row[0] for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table'"
        ).fetchall()}
    finally:
        conn.close()
    expected_tables = {
        "messages", "messages_fts", "long_term_memories", "recall_messages_meta",
        "memory_changelog", "core_memory_blocks", "core_memory_snapshots", "extraction_state",
    }
    missing_tables = expected_tables - existing_tables
    tc("TG1", "1.1", "数据库初始化（8张表全部创建）",
       not missing_tables and stats["total_messages"] == 0 and stats["total_memories"] == 0,
       f"missing_tables={sorted(missing_tables)}, messages={stats['total_messages']}, memories={stats['total_memories']}")

    # TC1.2: 插入消息
    mid1 = db.insert_message("用户名叫小研，喜欢法海这个文学人物", role="user", source="memory_extraction")
    mid2 = db.insert_message("雷峰塔于1924年倒塌", role="user", source="recall")
    mid3 = db.insert_message("宋代民间信仰与文学互动是重要研究课题", role="user", source="memory_extraction")
    fetched = db.get_message(mid1)
    tc("TG1", "1.2", "插入消息并读取",
       fetched is not None and "小研" in fetched["content"] and fetched["role"] == "user",
       f"mid={mid1}, content={fetched['content'][:50] if fetched else 'NONE'}")

    # TC1.3: 批量获取消息
    batch = db.get_messages_by_ids([mid1, mid2, mid3])
    tc("TG1", "1.3", "批量获取消息", len(batch) == 3,
       f"requested 3, got {len(batch)}")

    # TC1.4: FTS5 bigram 搜索（中文）— 验证索引真实命中，而非 LIKE fallback 兜底
    results = db.search_fts("小研喜欢谁", top_k=10)
    conn = sqlite3.connect(str(db_path))
    try:
        fts_docs = conn.execute("SELECT COUNT(*) FROM messages_fts").fetchone()[0]
        msg_docs = conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0]
        fts_match = conn.execute(
            "SELECT COUNT(*) FROM messages_fts WHERE messages_fts MATCH ?",
            ('"小研"',),
        ).fetchone()[0]
    finally:
        conn.close()
    tc("TG1", "1.4", "FTS5 bigram 中文搜索（索引真实命中）",
       len(results) > 0 and any("小研" in r["content"] for r in results)
       and fts_docs == msg_docs and fts_match > 0,
       f"{len(results)} results, fts_docs={fts_docs}/{msg_docs}, match='小研'={fts_match}")

    # TC1.5: FTS5 搜索 - source 过滤
    results_filtered = db.search_fts("民间信仰", top_k=10, source_filter="memory_extraction")
    tc("TG1", "1.5", "FTS5 source 过滤 (memory_extraction)",
       len(results_filtered) > 0 and all(
           db.get_message(r["message_id"])["source"] == "memory_extraction"
           for r in results_filtered
       ),
       f"{len(results_filtered)} filtered results")

    # TC1.6: LIKE fallback（无 FTS5 匹配时）
    results_like = db.search_fts("完全不可能匹配的奇特查询字符串xyzqwert", top_k=5)
    tc("TG1", "1.6", "LIKE fallback 降级搜索",
       len(results_like) == 0,
       f"{len(results_like)} results (expect 0 for nonsense query)")

    # TC1.7: 长期记忆 CRUD
    mem_id = db.insert_long_term_memory(mid1, "用户名叫小研", memory_type="Entity", importance=0.9, confidence=0.8)
    mem = db.get_long_term_memory(mem_id)
    tc("TG1", "1.7", "长期记忆 CRUD",
       mem is not None and mem["memory_type"] == "Entity" and abs(mem["importance"] - 0.9) < 0.01,
       f"type={mem['memory_type'] if mem else '?'} imp={mem['importance'] if mem else '?'}")

    # TC1.8: 记忆访问追踪
    db.update_memory_access(mem_id)
    mem_after = db.get_long_term_memory(mem_id)
    tc("TG1", "1.8", "记忆访问计数更新",
       mem_after is not None and mem_after["access_count"] >= 1,
       f"access_count={mem_after['access_count'] if mem_after else '?'}")

    # TC1.9: 记忆状态转换
    db.update_memory_status(mem_id, "tombstone", reason="test_decay")
    mem_tomb = db.get_long_term_memory(mem_id)
    tc("TG1", "1.9", "记忆状态转换 (active→tombstone)",
       mem_tomb is not None and mem_tomb["status"] == "tombstone",
       f"status={mem_tomb['status'] if mem_tomb else '?'}")

    # TC1.10: Changelog 审计
    changelog = db.get_changelog_by_memory(mem_id)
    tc("TG1", "1.10", "Changelog 审计记录",
       len(changelog) >= 1 and changelog[0]["event_type"] == "TOMBSTONE",
       f"{len(changelog)} entries, latest event={changelog[0]['event_type'] if changelog else '?'}")

    # TC1.11: 不稳定记忆标记
    db.update_memory_instability(mem_id)
    db.update_memory_instability(mem_id)  # 第二次触发 is_unstable
    mem_unstable = db.get_long_term_memory(mem_id)
    tc("TG1", "1.11", "不稳定记忆标记 (>=2次反转)",
       mem_unstable is not None and mem_unstable["is_unstable"] == 1,
       f"contradiction_count={mem_unstable['contradiction_count'] if mem_unstable else '?'}, "
       f"is_unstable={mem_unstable['is_unstable'] if mem_unstable else '?'}")

    # TC1.12: 获取活跃记忆
    # 先创建一条活跃记忆
    mid_active = db.insert_message("用户偏好文言文风格", role="user", source="memory_extraction")
    db.insert_long_term_memory(mid_active, "用户偏好文言文风格", memory_type="Entity", importance=0.7)
    active = db.get_active_memories(limit=50)
    tc("TG1", "1.12", "获取活跃记忆列表",
       len(active) >= 1 and any("文言文" in m["content"] for m in active),
       f"{len(active)} active memories")

    # TC1.13: 按类型获取记忆
    by_type = db.get_memories_by_type("Entity", limit=50)
    tc("TG1", "1.13", "按 memory_type 获取记忆",
       isinstance(by_type, list) and all(m["memory_type"] == "Entity" for m in by_type),
       f"{len(by_type)} Entity-type memories")

    final_stats = db.get_stats()
    trace_json("TG1 Final Stats", final_stats)
    db.close()
    return final_stats


# ============================================================
#  Test Group 2: RecallStore (双写存储)
# ============================================================
async def test_group_2_recall_store(test_dir: Path, chroma_client, embedder) -> dict:
    sep("TG2: RecallStore — 双写存储测试", "=")
    trace_txt("\n" + "=" * 80)
    trace_txt("  TG2: RecallStore — 双写存储")
    trace_txt("=" * 80 + "\n")

    from agent.server.context.memory_db import MemoryDB
    from agent.server.context.recall_store import RecallStore

    db_path = test_dir / "tg2_memory.db"
    if db_path.exists():
        db_path.unlink()
    db = MemoryDB(db_path)
    store = RecallStore(chroma_client=chroma_client, embedder=embedder, memory_db=db)
    store.clear_all()

    # TC2.1: 双写召回消息（验证 SQLite 与 ChromaDB 两侧都写入）
    msg = {"role": "user", "content": "我喜欢宋代文学", "_thread_id": "t1", "timestamp": bj_now().isoformat()}
    msg_id = store.add_recall_message(msg)
    fetched = db.get_message(msg_id) if msg_id else None
    chroma_hit = None
    if msg_id and store._recall_collection is not None:
        chroma_res = store._recall_collection.get(ids=[msg_id])
        chroma_hit = chroma_res.get("ids") if chroma_res else None
    tc("TG2", "2.1", "双写召回消息 (SQLite + ChromaDB 两侧都写入)",
       msg_id is not None and fetched is not None and "宋代文学" in fetched["content"]
       and chroma_hit is not None and msg_id in chroma_hit,
       f"msg_id={msg_id}, pending_embedding={fetched['pending_embedding'] if fetched else '?'}, chroma_hit={bool(chroma_hit)}")

    # TC2.2: MD5 精确去重
    dup = store.check_duplicate_md5("我喜欢宋代文学")
    tc("TG2", "2.2", "MD5 精确去重 (防线1)",
       dup is not None,
       f"duplicate message_id={dup}")

    # TC2.3: MD5 去重 — 不同内容不重复
    not_dup = store.check_duplicate_md5("我喜欢唐代文学")
    tc("TG2", "2.3", "MD5 去重 — 不同内容不误判",
       not_dup is None,
       f"result={'None (not duplicate)' if not_dup is None else 'UNEXPECTED DUPLICATE'}")

    # TC2.4: 向量去重 (防线2)
    # 插入一条记忆后检查相似内容
    store.add_memory("用户在杭州从事文学研究工作", memory_type="Entity", importance=0.8)
    await asyncio.sleep(0.5)  # 等待 embedding 写入
    is_dup_vector = store.check_duplicate_vector("用户在杭州从事文学研究工作")
    tc("TG2", "2.4", "向量去重 (防线2, cosine>0.95)",
       is_dup_vector,
       f"duplicate detected={is_dup_vector}")

    # TC2.5: 添加长期记忆（非重复）
    mem_id = store.add_memory("用户对白蛇传的法海形象有深入研究", memory_type="Entity", importance=0.9, confidence=0.8)
    mem_fetched = db.get_message(mem_id) if mem_id else None
    tc("TG2", "2.5", "双写长期记忆",
       mem_id is not None and mem_fetched is not None and "法海" in mem_fetched["content"],
       f"mem_id={mem_id}")

    # TC2.6: ChromaDB 写入容错 — pending_embedding 跟踪
    # 验证: 正常写入后 pending_embedding=0
    pending_before = db.get_stats()["pending_embeddings"]
    tc("TG2", "2.6", "Embedding 写入成功 (pending_embedding=0)",
       pending_before == 0,
       f"pending_embeddings={pending_before}")

    # TC2.7: 健康检查
    healthy = store.is_healthy()
    tc("TG2", "2.7", "RecallStore 健康检查", healthy, f"healthy={healthy}")

    # TC2.8: 统计信息一致性
    stats = store.stats
    tc("TG2", "2.8", "双存储统计信息一致性",
       stats["recall_messages_count"] >= 1 and stats["long_term_memories_count"] >= 1,
       f"recall={stats['recall_messages_count']}, memories={stats['long_term_memories_count']}")

    trace_json("TG2 Final Stats", stats)
    db.close()
    return stats


# ============================================================
#  Test Group 3: MemoryExtractor (记忆提取)
# ============================================================
async def test_group_3_memory_extractor(test_dir: Path, chroma_client, embedder) -> dict:
    sep("TG3: MemoryExtractor — 记忆提取测试", "=")
    trace_txt("\n" + "=" * 80)
    trace_txt("  TG3: MemoryExtractor — 记忆提取")
    trace_txt("=" * 80 + "\n")

    from agent.server.context.memory_db import MemoryDB
    from agent.server.context.recall_store import RecallStore
    from agent.server.context.memory_extractor import MemoryExtractor

    db_path = test_dir / "tg3_memory.db"
    if db_path.exists():
        db_path.unlink()
    db = MemoryDB(db_path)
    store = RecallStore(chroma_client=chroma_client, embedder=embedder, memory_db=db)
    store.clear_all()

    mock_llm = MockLLM()
    mock_llm.set_responses({
        "我叫小研": "用户的名字是小研。\n用户最喜欢的文学人物是雷峰塔传奇中的法海。\n用户的研究方向是宋代江南民间信仰与文学的互动。\n用户约束：不要发送长代码。",
        "不要给我发": "用户约束：不要发送长代码。",
        "宋代话本": "用户对宋代话本小说特别有兴趣。",
    })

    extractor = MemoryExtractor(llm=mock_llm, recall_store=store, model="test-model")

    # TC3.1: 噪声消息过滤
    noise_results = await extractor.extract({"role": "user", "content": "好的"})
    tc("TG3", "3.1", "噪声消息过滤 (太短/问候语)",
       len(noise_results) == 0,
       f"extracted {len(noise_results)} memories (expect 0)")

    # TC3.2: Mem0 策略 — 跳过 assistant 消息
    asst_results = await extractor.extract({
        "role": "assistant",
        "content": "用户的名字是小研，研究方向是宋代文学。这是很重要的个人信息应该记住。",
    })
    tc("TG3", "3.2", "Mem0 策略 — 跳过 assistant 消息",
       len(asst_results) == 0,
       f"extracted {len(asst_results)} from assistant (expect 0)")

    # TC3.3: 从 user 消息提取记忆
    user_results = await extractor.extract({
        "role": "user",
        "content": "我叫小研。我特别喜欢法海这个角色，我的研究方向是宋代江南民间信仰与文学的互动。不要给我发长代码。",
    })
    tc("TG3", "3.3", "从 user 消息提取长期记忆",
       len(user_results) >= 2,
       f"extracted {len(user_results)} memories: {user_results}")

    # TC3.4: MD5 去重 — 相同内容不重复提取
    dup_results = await extractor.extract({
        "role": "user",
        "content": "我叫小研。我特别喜欢法海这个角色。",
    })
    tc("TG3", "3.4", "MD5 去重 — 相同记忆不重复存储",
       len(dup_results) == 0,
       f"second extraction yielded {len(dup_results)} new memories")

    # TC3.5: memory_type 规则分类验证
    active_memories = db.get_active_memories(limit=50)
    entity_count = sum(1 for m in active_memories if m["memory_type"] == "Entity")
    tc("TG3", "3.5", "memory_type 规则分类 (Entity)",
       entity_count > 0,
       f"{entity_count}/{len(active_memories)} classified as Entity")

    # TC3.6: importance 启发式赋值
    if active_memories:
        high_imp = [m for m in active_memories if m["importance"] >= 0.7]
        tc("TG3", "3.6", "importance 启发式赋值",
           len(high_imp) > 0,
           f"{len(high_imp)} memories with importance>=0.7, avg={sum(m['importance'] for m in active_memories)/len(active_memories):.2f}")

    # TC3.7: 批量提取 (extract_batch)
    batch_msgs = [
        {"role": "user", "content": "我对宋代话本小说特别有兴趣，尤其是话本中的人物塑造。", "_thread_id": "t3"},
        {"role": "assistant", "content": "宋代话本确实很有趣。", "_thread_id": "t3"},
        {"role": "user", "content": "我在杭州西湖边长大，对当地民间文化很熟悉。", "_thread_id": "t3"},
    ]
    batch_results = await extractor.extract_batch(batch_msgs)
    tc("TG3", "3.7", "批量提取 (extract_batch)",
       len(batch_results) >= 1,
       f"{len(batch_results)} memories from batch (user messages only)")

    stats = db.get_stats()
    trace_json("TG3 Final Stats", stats)
    db.close()
    return stats


# ============================================================
#  Test Group 4: HybridRetriever (混合检索)
# ============================================================
async def test_group_4_hybrid_retriever(test_dir: Path, chroma_client, embedder) -> dict:
    sep("TG4: HybridRetriever — 混合检索测试", "=")
    trace_txt("\n" + "=" * 80)
    trace_txt("  TG4: HybridRetriever — 混合检索")
    trace_txt("=" * 80 + "\n")

    from agent.server.context.memory_db import MemoryDB
    from agent.server.context.recall_store import RecallStore
    from agent.server.context.fts_index import FTSIndex
    from agent.server.context.hybrid_retriever import HybridRetriever

    db_path = test_dir / "tg4_memory.db"
    if db_path.exists():
        db_path.unlink()
    db = MemoryDB(db_path)
    store = RecallStore(chroma_client=chroma_client, embedder=embedder, memory_db=db)
    store.clear_all()

    # 插入测试数据
    store.add_memory("用户名字是小研，研究方向是宋代江南民间信仰与文学互动",
                     memory_type="Entity", importance=0.9)
    store.add_memory("用户最喜欢的文学人物是雷峰塔传奇中的法海",
                     memory_type="Entity", importance=0.8)
    store.add_memory("用户不喜欢接收长代码，只需要分析结果",
                     memory_type="Entity", importance=0.7)
    store.add_memory("雷峰塔位于杭州西湖边，1924年倒塌，2002年重建",
                     memory_type="Archival", importance=0.6)

    msg_template = {"role": "user", "content": "", "_thread_id": "t4"}
    msg_template["content"] = "我在杭州西湖边做文学研究"
    store.add_recall_message(msg_template)
    msg_template["content"] = "法海在雷峰塔传奇中的形象非常复杂"
    store.add_recall_message(msg_template)
    await asyncio.sleep(0.5)

    fts_idx = FTSIndex(db)
    retriever = HybridRetriever(memory_db=db, embedder=embedder, chroma_client=chroma_client, fts_index=fts_idx)

    # TC4.1: FTS5 候选召回
    fts_results = fts_idx.search("用户喜欢谁", top_k=30, source_filter="memory_extraction")
    tc("TG4", "4.1", "FTS5 候选召回 (source_filter=memory_extraction)",
       len(fts_results) > 0,
       f"{len(fts_results)} FTS candidates")

    # TC4.2: 混合检索 — memories
    result = retriever.search("用户喜欢哪个文学人物", source="memories", score_threshold=0.3)
    tc("TG4", "4.2", "混合检索 (memories) — 两阶段管道",
       len(result["memories"]) > 0,
       f"{len(result['memories'])} memories, scores={[round(m.get('score',0),3) for m in result['memories']]}")

    # TC4.3: 混合检索 — recalls
    result_recall = retriever.search("法海", source="recall", score_threshold=0.3)
    tc("TG4", "4.3", "混合检索 (recalls) — 两阶段管道",
       len(result_recall["recalls"]) >= 1,
       f"{len(result_recall['recalls'])} recalls")

    # TC4.4: 混合检索 — all sources
    result_all = retriever.search("小研的研究方向", source="all", score_threshold=0.3)
    tc("TG4", "4.4", "混合检索 (all) — 双通道并行",
       len(result_all["memories"]) >= 1,
       f"memories={len(result_all['memories'])}, recalls={len(result_all['recalls'])}")

    # TC4.5: 分数阈值过滤 (score >= 0.5)
    result_strict = retriever.search("小研的研究方向", source="all", score_threshold=0.5)
    below = result_strict.get("below_threshold", False)
    all_strict = result_strict["memories"] + result_strict["recalls"]
    tc("TG4", "4.5", "分数阈值过滤 (score >= 0.5)",
       all(m.get("score", 0) >= 0.5 for m in all_strict),
       f"below_threshold={below}, memories={len(result_strict['memories'])}, recalls={len(result_strict['recalls'])}")

    # TC4.6: 空查询 → 全低于阈值 → below_threshold=True
    result_empty = retriever.search("xyz完全不相关查询测试abc", source="all", score_threshold=0.5)
    tc("TG4", "4.6", "不相关查询 → below_threshold=True",
       result_empty.get("below_threshold", False) or (
           len(result_empty.get("memories", [])) == 0 and len(result_empty.get("recalls", [])) == 0
       ),
       f"below_threshold={result_empty.get('below_threshold')}")

    # TC4.7: 格式化输出 — 被动注入
    injection = retriever.format_for_injection(result_all)
    tc("TG4", "4.7", "被动注入格式化 (自然语言文本)",
       len(injection) > 0 and "相关长期记忆" in injection,
       f"injection text: {len(injection)} chars")

    # TC4.8: 格式化输出 — MCP 工具
    mcp_text = retriever.format_for_mcp_tool(result_all)
    tc("TG4", "4.8", "MCP 工具返回格式化",
       len(mcp_text) > 0 and "内部参考" in mcp_text,
       f"MCP text: {len(mcp_text)} chars")

    # TC4.9: Token 预算控制
    result_budget = retriever.search("小研 法海 研究 方向 民间信仰", source="all",
                                      score_threshold=0.2, token_budget=50)
    tc("TG4", "4.9", "Token 预算控制",
       result_budget["total_tokens"] <= 50,
       f"total_tokens={result_budget['total_tokens']} (budget=50)")

    # TC4.10: 检索命中 → update_memory_access 更新访问计数（recency 重置的数据链路）
    target_mid = store.add_memory("小研最钟爱的诗人是苏轼", memory_type="Entity", importance=0.8)
    await asyncio.sleep(0.3)
    before = db.get_long_term_memory(target_mid)
    _ = retriever.search("苏轼", source="memories", score_threshold=0.3)
    after = db.get_long_term_memory(target_mid)
    tc("TG4", "4.10", "检索命中 → update_memory_access 更新访问计数与访问时间",
       before is not None and after is not None
       and after["access_count"] > before["access_count"]
       and after["last_accessed_at"] is not None,
       f"access_count {before['access_count'] if before else '?'} -> {after['access_count'] if after else '?'}, "
       f"last_accessed_at={'set' if after and after['last_accessed_at'] else 'None'}")

    trace_json("TG4 Results", {
        "fts_candidates": len(fts_results),
        "hybrid_memories": len(result["memories"]),
        "hybrid_recalls": len(result_recall.get("recalls", [])),
        "strict_below": below,
    })
    db.close()
    return {"fts_ok": len(fts_results) > 0, "hybrid_ok": len(result["memories"]) > 0}


# ============================================================
#  Test Group 5: MemoryScorer (评分)
# ============================================================
async def test_group_5_memory_scorer(test_dir: Path) -> dict:
    sep("TG5: MemoryScorer — 评分计算测试", "=")
    trace_txt("\n" + "=" * 80)
    trace_txt("  TG5: MemoryScorer — 评分计算")
    trace_txt("=" * 80 + "\n")

    from agent.server.context.memory_db import MemoryDB
    from agent.server.context.memory_scorer import MemoryScorer

    db_path = test_dir / "tg5_memory.db"
    if db_path.exists():
        db_path.unlink()
    db = MemoryDB(db_path)
    scorer = MemoryScorer(db)

    # TC5.1: 基本评分计算
    score = scorer.compute_score(relevance_raw=0.8, importance_raw=0.9, memory_type="Entity")
    tc("TG5", "5.1", "基本评分计算 (relevance=0.8, importance=0.9, type=Entity)",
       0.5 <= score <= 1.0,
       f"score={score:.4f}")

    # TC5.2: 低分场景
    low = scorer.compute_score(relevance_raw=0.1, importance_raw=0.1, memory_type="Working")
    tc("TG5", "5.2", "低分场景 (relevance=0.1, importance=0.1)",
       low < 0.5,
       f"score={low:.4f}")

    # TC5.3: 批量评分 + Min-Max 归一化
    memories = [
        {"score": 0.9, "importance": 0.9, "memory_type": "Entity", "content": "A"},
        {"score": 0.5, "importance": 0.5, "memory_type": "Archival", "content": "B"},
        {"score": 0.1, "importance": 0.1, "memory_type": "Working", "content": "C"},
    ]
    scored = scorer.compute_scores_batch(memories)
    tc("TG5", "5.3", "批量评分 + Min-Max 归一化",
       len(scored) == 3 and scored[0]["computed_score"] >= scored[-1]["computed_score"],
       f"scores: {[round(m['computed_score'],4) for m in scored]}")

    # TC5.4: 衰减计算 — 不同类型不同半衰期
    from datetime import timezone, timedelta
    now = bj_now()
    old_90d = (now - timedelta(days=90)).isoformat()
    recent = now.isoformat()

    score_old_entity = scorer.compute_score(0.8, 0.8, "Entity", created_at=old_90d)
    score_new_entity = scorer.compute_score(0.8, 0.8, "Entity", created_at=recent)
    tc("TG5", "5.4", "时间衰减 — Entity 90天半衰期",
       score_old_entity < score_new_entity,
       f"old(90d)={score_old_entity:.4f} < new(0d)={score_new_entity:.4f}")

    # TC5.5: 衰减计算 — Working 7天快速衰减
    old_7d = (now - timedelta(days=7)).isoformat()
    score_old_working = scorer.compute_score(0.8, 0.8, "Working", created_at=old_7d)
    score_old_entity_7d = scorer.compute_score(0.8, 0.8, "Entity", created_at=old_7d)
    tc("TG5", "5.5", "Working 7天衰减 > Entity 7天衰减",
       score_old_working < score_old_entity_7d,
       f"Working={score_old_working:.4f} < Entity={score_old_entity_7d:.4f}")

    # TC5.6: 访问重置衰减
    score_accessed = scorer.compute_score(0.8, 0.8, "Entity", created_at=old_90d, last_accessed_at=recent)
    tc("TG5", "5.6", "访问重置 — last_accessed_at 重设衰减计时器",
       score_accessed > score_old_entity,
       f"accessed={score_accessed:.4f} > not_accessed={score_old_entity:.4f}")

    # TC5.7: 重要性启发式估算
    imp_user_entity = scorer.get_importance_for_extraction("用户喜欢法海", role="user", memory_type="Entity")
    imp_user_constraint = scorer.get_importance_for_extraction("不要发送长代码", role="user", memory_type="Entity")
    imp_asst = scorer.get_importance_for_extraction("法海形象复杂", role="assistant", memory_type="Working")
    tc("TG5", "5.7", "重要性启发式赋值",
       imp_user_entity >= 0.7 and imp_asst <= 0.5,
       f"user_entity={imp_user_entity:.2f}, constraint={imp_user_constraint:.2f}, assistant={imp_asst:.2f}")

    # TC5.8: 衰减阈值判定（importance 保底 + TTL）
    should_del = scorer.should_tombstone(importance=0.2, memory_type="Working", days_unaccessed=8)
    should_keep_high = scorer.should_tombstone(importance=0.7, memory_type="Entity", days_unaccessed=200)
    should_keep_system = scorer.should_tombstone(importance=0.95, memory_type="System", days_unaccessed=999)
    tc("TG5", "5.8", "衰减阈值判定 (importance 保底 + TTL)",
       should_del and not should_keep_high and not should_keep_system,
       f"working/8d={should_del}, entity_high/200d={should_keep_high}, system/999d={should_keep_system}")

    trace_json("TG5 Scored Example", {
        "top_scored": {k: round(v, 4) for k, v in scored[0].items() if isinstance(v, (int, float))},
    })
    db.close()
    return {"min_max_ok": scored[0]["computed_score"] >= scored[-1]["computed_score"]}


# ============================================================
#  Test Group 6: Core Memory (核心记忆)
# ============================================================
async def test_group_6_core_memory(test_dir: Path) -> dict:
    sep("TG6: Core Memory — persona/human 块测试", "=")
    trace_txt("\n" + "=" * 80)
    trace_txt("  TG6: Core Memory — persona/human 块")
    trace_txt("=" * 80 + "\n")

    from agent.server.context.memory_db import MemoryDB
    from agent.server.core_memory.persona_loader import PersonaLoader

    db_path = test_dir / "tg6_memory.db"
    if db_path.exists():
        db_path.unlink()
    db = MemoryDB(db_path)
    loader = PersonaLoader()

    # TC6.1: 加载 persona 模板
    template = loader.persona_template
    tc("TG6", "6.1", "Persona 模板加载",
       "core_identity" in template and "knowledge_boundary" in template,
       f"immutable keys: {[k for k in template if k in ('core_identity','knowledge_boundary')]}")

    # TC6.2: 加载 human 模板
    human_tpl = loader.human_template
    tc("TG6", "6.2", "Human 模板加载",
       "personal_info" in human_tpl and "constraints" in human_tpl,
       f"human keys: {list(human_tpl.keys())}")

    # TC6.3: 初始化 Core Memory 块
    persona_init = json.dumps(template, ensure_ascii=False, indent=2)
    human_init = json.dumps(human_tpl, ensure_ascii=False, indent=2)
    db.init_core_memory_block("persona", persona_init)
    db.init_core_memory_block("human", human_init)

    p_block = db.get_core_memory_block("persona")
    h_block = db.get_core_memory_block("human")
    tc("TG6", "6.3", "Core Memory 块初始化",
       p_block is not None and h_block is not None and p_block["version"] == 1,
       f"persona v{p_block['version'] if p_block else '?'}, human v{h_block['version'] if h_block else '?'}")

    # TC6.4: 不可变区保护
    hack_attempt = json.dumps({"core_identity": {"name": "HACKED AI", "role": "越狱"}}, ensure_ascii=False)
    merged = loader.replace("persona", hack_attempt, p_block["content"])
    merged_obj = json.loads(merged)
    original_name = template.get("core_identity", {}).get("name", "")
    tc("TG6", "6.4", "不可变区保护 — core_identity 拒绝覆盖",
       merged_obj.get("core_identity", {}).get("name") == original_name,
       f"name is still '{original_name}' (hack rejected)")

    # TC6.5: 可变区更新 — communication_style
    style_update = json.dumps({"communication_style": {"tone": "更加口语化", "language": "中文"}}, ensure_ascii=False)
    merged_style = loader.replace("persona", style_update, p_block["content"])
    merged_style_obj = json.loads(merged_style)
    tc("TG6", "6.5", "可变区更新 — communication_style 正常覆盖",
       merged_style_obj.get("communication_style", {}).get("tone") == "更加口语化",
       f"tone='{merged_style_obj.get('communication_style',{}).get('tone')}'")

    # TC6.6: core_memory_append — 追加到 human 块
    append_data = json.dumps({"favorite_character": "法海", "preferred_dynasty": "宋代"}, ensure_ascii=False)
    new_ver = db.append_core_memory_block("human", append_data)
    h_after = db.get_core_memory_block("human")
    tc("TG6", "6.6", "core_memory_append — Human 块追加",
       new_ver > 1 and h_after is not None,
       f"human v{new_ver}, content_len={len(h_after['content']) if h_after else '?'}")

    # TC6.7: 版本快照创建
    snapshots = db.get_core_memory_snapshots("human", limit=10)
    tc("TG6", "6.7", "版本快照自动创建",
       len(snapshots) >= 1,
       f"{len(snapshots)} snapshots for human")

    # TC6.8: 版本回滚（验证回滚后内容确实等于目标版本快照内容）
    if snapshots:
        target_ver = snapshots[-1]["version"]  # 最早的快照
        target_content = snapshots[-1]["content"]
        rolled = db.rollback_core_memory("human", target_ver)
        h_rolled = db.get_core_memory_block("human")
        content_rolled_back = (h_rolled is not None and h_rolled["content"] == target_content)
        tc("TG6", "6.8", "版本回滚（内容恢复到目标版本）",
           rolled and content_rolled_back,
           f"rolled={rolled}, content_matches={content_rolled_back}, now at v{h_rolled['version'] if h_rolled else '?'}")

    # 回滚后再追加有意义的内容（否则格式化输出为空）
    db.append_core_memory_block("human", json.dumps({"personal_info": {"name": "小研"}}, ensure_ascii=False))
    h_final = db.get_core_memory_block("human")

    # TC6.9: Token 限制检测
    token_count = loader.estimate_tokens(persona_init)
    tc("TG6", "6.9", "Token 估算",
       token_count > 0,
       f"persona estimated tokens: {token_count}")

    # TC6.10: 格式化输出
    persona_text = loader.format_for_context(p_block["content"], "persona")
    human_text = loader.format_for_context(h_final["content"] if h_final else "{}", "human")
    tc("TG6", "6.10", "Core Memory 自然语言格式化",
       len(persona_text) > 0 and len(human_text) > 0,
       f"persona={len(persona_text)} chars, human={len(human_text)} chars")

    trace_txt(f"[PERSONA FORMATTED]\n{persona_text}\n")
    trace_txt(f"[HUMAN FORMATTED]\n{human_text}\n")

    stats = db.get_stats()
    trace_json("TG6 Stats", stats)
    db.close()
    return {"immutable_protected": True, "snapshots": len(snapshots)}


# ============================================================
#  Test Group 7: ContextManager (上下文管理)
# ============================================================
async def test_group_7_context_manager(test_dir: Path, chroma_client, embedder) -> dict:
    sep("TG7: ContextManager — 上下文管理测试", "=")
    trace_txt("\n" + "=" * 80)
    trace_txt("  TG7: ContextManager — 上下文管理")
    trace_txt("=" * 80 + "\n")

    from agent.server.context.wiring import build_memory_system

    db_path = test_dir / "tg7_memory.db"
    ctx_db_path = test_dir / "tg7_context.db"
    for p in [db_path, ctx_db_path]:
        if p.exists():
            p.unlink()

    mock_llm = MockLLM()
    mock_llm.set_responses({
        "我叫小研": "用户的名字是小研。用户特别喜欢法海。用户研究宋代民间信仰。",
    })

    # 与生产一致的工厂装配（fifo_max_tokens=2000 足够大，不触发压缩）
    ms = build_memory_system(
        llm=mock_llm, embedder=embedder, chroma_client=chroma_client,
        memory_db_path=db_path, context_db_path=ctx_db_path,
        fifo_max_tokens=2000, summary_model="test", memory_extraction_model="test",
    )
    cm = ms.context_manager
    db = ms.memory_db
    ms.recall_store.clear_all()
    # clear_all 会连 core_memory_blocks 一起清空，这里用工厂 persona_loader 重新初始化
    db.init_core_memory_block("persona", json.dumps(ms.persona_loader.persona_template, ensure_ascii=False, indent=2))
    db.init_core_memory_block("human", json.dumps(ms.persona_loader.human_template, ensure_ascii=False, indent=2))

    thread_id = "tg7-test-thread"

    # TC7.1: FIFO 入队 + 上下文窗口构建（真实序列先 load）
    await cm.load(thread_id)
    await cm.enqueue(thread_id, {"role": "user", "content": "你好，我叫小研。"})
    await cm.enqueue(thread_id, {"role": "assistant", "content": "你好小研！"})
    ctx = cm.build_context_window(thread_id)
    tc("TG7", "7.1", "FIFO 入队 + build_context_window",
       len(ctx) >= 2,
       f"context has {len(ctx)} messages")

    # TC7.2: 上下文持久化 + 工厂第二个实例重新加载（同一 context_db_path）
    await cm.save(thread_id)
    ms2 = build_memory_system(
        llm=mock_llm, embedder=embedder, chroma_client=chroma_client,
        memory_db_path=db_path, context_db_path=ctx_db_path,
        fifo_max_tokens=2000, summary_model="test", memory_extraction_model="test",
    )
    cm2 = ms2.context_manager
    loaded_state = await cm2.load(thread_id)
    tc("TG7", "7.2", "上下文 SQLite 持久化 + 加载",
       loaded_state is not None and len(loaded_state.messages) >= 2,
       f"loaded {len(loaded_state.messages) if loaded_state else '?'} messages")

    # TC7.3: 压力警告触发（工厂小窗口实例）
    pressure_mem = test_dir / "tg7_pressure_memory.db"
    pressure_ctx = test_dir / "tg7_pressure.db"
    for p in (pressure_mem, pressure_ctx):
        if p.exists():
            p.unlink()
    ms3 = build_memory_system(
        llm=mock_llm, embedder=embedder, chroma_client=chroma_client,
        memory_db_path=pressure_mem, context_db_path=pressure_ctx,
        fifo_max_tokens=400, summary_model="test", memory_extraction_model="test",
    )
    ms3.recall_store.clear_all()
    cm3 = ms3.context_manager
    thread_id_p = "tg7-pressure-thread"
    await cm3.load(thread_id_p)
    await cm3.enqueue(thread_id_p, {"role": "user", "content": "我是小研。我的研究涉及宋代民间信仰与文学互动。"})
    await cm3.enqueue(thread_id_p, {"role": "assistant", "content": "好的小研，我记住了。"})
    # 添加足够多的消息触发压力
    for i in range(8):
        await cm3.enqueue(thread_id_p, {"role": "user", "content": f"关于白蛇传第{i}章的讨论。" * 8})
        await cm3.enqueue(thread_id_p, {"role": "assistant", "content": f"这是关于第{i}章的学术分析回复。" * 8})

    # 手动把 token 用量推到 70% 阈值之上，验证阈值判定逻辑
    state_p = cm3.get_state(thread_id_p)
    if state_p is not None:
        state_p.total_tokens = 320  # 320/400 = 0.8 > 0.7
    under = cm3.is_under_pressure(thread_id_p)
    tc("TG7", "7.3", "70% 压力阈值判定 (is_under_pressure)",
       under is True,
       f"pressure_ratio={cm3.get_pressure_ratio(thread_id_p):.2f}, under_pressure={under}")

    # TC7.4: 压力警告消息生成
    warning = cm3.get_pressure_warning(thread_id_p)
    tc("TG7", "7.4", "压力警告消息生成",
       len(warning) > 0,
       f"warning={'generated' if warning else 'none (below threshold)'}")

    # TC7.5: FIFO 压缩（消息数被压缩到少于灌入总数 18）
    stats3 = cm3.get_stats(thread_id_p)
    tc("TG7", "7.5", "FIFO 压缩（消息数被压缩）",
       stats3["message_count"] < 18,
       f"msgs={stats3['message_count']}, tokens={stats3['total_tokens']}, summary={stats3['summary_length']} chars")

    await cm3.close()
    for p in (pressure_mem, pressure_ctx):
        if p.exists():
            p.unlink()

    # TC7.6: 真实消费方法 build_memory_context_text（含 Core Memory 注入）
    memory_context = cm.build_memory_context_text(thread_id, "法海在哪部作品中")
    has_core = "Core Memory" in memory_context
    tc("TG7", "7.6", "Core Memory 注入（build_memory_context_text）",
       has_core,
       f"memory_context {len(memory_context)} chars, core_memory_injected={has_core}")

    # TC7.7: 空闲检测 — 更新活跃时间
    extraction_state = db.get_extraction_state(thread_id)
    tc("TG7", "7.7", "活跃时间追踪 (touch_active_at)",
       extraction_state.get("last_active_at") is not None,
       f"last_active={extraction_state.get('last_active_at', 'NONE')[:19]}")

    # TC7.8: 获取未处理消息
    state = cm.get_state(thread_id)
    if state:
        unprocessed = cm._get_unprocessed_messages(state, -1)
        tc("TG7", "7.8", "获取未处理消息 (last_extracted_index=-1)",
           len(unprocessed) > 0,
           f"{len(unprocessed)} unprocessed messages")

    trace_json("TG7 Context Stats", cm.get_stats(thread_id))
    await cm.close()
    await cm2.close()
    db.close()
    return {"pressure_ok": True, "context_window_ok": has_core}


# ============================================================
#  Test Group 8: BackgroundTasks (后台任务)
# ============================================================
async def test_group_8_background_tasks(test_dir: Path, chroma_client, embedder) -> dict:
    sep("TG8: BackgroundTasks — 后台任务测试", "=")
    trace_txt("\n" + "=" * 80)
    trace_txt("  TG8: BackgroundTasks — 后台任务")
    trace_txt("=" * 80 + "\n")

    from agent.server.context.memory_db import MemoryDB
    from agent.server.context.recall_store import RecallStore
    from agent.server.context.background_tasks import BackgroundTaskScheduler
    from agent.server.core_memory.persona_loader import PersonaLoader

    db_path = test_dir / "tg8_memory.db"
    if db_path.exists():
        db_path.unlink()
    db = MemoryDB(db_path)
    store = RecallStore(chroma_client=chroma_client, embedder=embedder, memory_db=db)
    store.clear_all()
    loader = PersonaLoader()

    # 创建几条测试记忆
    mid1 = store.add_memory("用户叫小研，喜欢法海", memory_type="Entity", importance=0.9)
    mid2 = store.add_memory("用户研究宋代文学", memory_type="Entity", importance=0.8)
    mid3 = store.add_memory("非常相似的用户偏好法海形象", memory_type="Entity", importance=0.6)
    mid4 = store.add_memory("小研的研究方向是宋代江南民间信仰", memory_type="Entity", importance=0.7)
    mid5 = store.add_memory("临时状态：正在调试白蛇传文本", memory_type="Working", importance=0.2)
    await asyncio.sleep(0.3)

    scheduler = BackgroundTaskScheduler(
        memory_db=db, recall_store=store, embedder=embedder, persona_loader=loader,
    )

    # TC8.1: 软删除 — 低重要性 + 过期 → tombstone；高重要性豁免
    import sqlite3 as sq
    conn = sq.connect(str(db_path))
    conn.execute("UPDATE long_term_memories SET last_accessed_at = ? WHERE memory_id = ?",
                 ((bj_now() - timedelta(days=8)).isoformat(), mid5))
    conn.execute("UPDATE long_term_memories SET last_accessed_at = ? WHERE memory_id = ?",
                 ((bj_now() - timedelta(days=35)).isoformat(), mid4))
    conn.commit()
    conn.close()

    await scheduler._run_tombstone_cleanup()
    stats_after = db.get_stats()
    mem5_after = db.get_long_term_memory(mid5) if mid5 else None
    mem4_after = db.get_long_term_memory(mid4) if mid4 else None
    tc("TG8", "8.1", "软删除 — 低重要性过期记忆 tombstone，高重要性豁免",
       stats_after["tombstoned_memories"] >= 1
       and (mem5_after is not None and mem5_after.get("status") == "tombstone")
       and (mem4_after is not None and mem4_after.get("status") != "tombstone"),
       f"tombstoned={stats_after['tombstoned_memories']}, "
       f"mid5={mem5_after.get('status') if mem5_after else '?'}, "
       f"mid4={mem4_after.get('status') if mem4_after else '?'}")

    # TC8.2: 高重要性记忆不被误删
    mem1 = db.get_long_term_memory(mid1) if mid1 else None
    tc("TG8", "8.2", "高重要性记忆保护 — Entity importance=0.9 不被 tombstone",
       mem1 is not None and mem1.get("status") != "tombstone",
       f"mid1 status={mem1.get('status') if mem1 else '?'} (expect != tombstone)")

    # TC8.3: 记忆合并 — 插入高度相似的两条记忆，验证合并触发
    mid6 = store.add_memory("用户喜欢法海这个文学角色", memory_type="Entity", importance=0.6)
    mid7 = store.add_memory("用户喜欢法海这一文学角色", memory_type="Entity", importance=0.6)
    await asyncio.sleep(0.3)
    await scheduler._run_memory_merge()
    stats_merge = db.get_stats()
    tc("TG8", "8.3", "记忆合并 (cosine > 0.7)",
       stats_merge["consolidated_memories"] >= 2,
       f"consolidated={stats_merge['consolidated_memories']}")

    # TC8.4: Embedding 补偿重试
    # 手动标记一条消息为 pending_embedding
    msg_id = db.insert_message("测试补偿重试", role="user", source="memory_extraction")
    conn2 = sq.connect(str(db_path))
    conn2.execute("UPDATE messages SET pending_embedding = 1 WHERE message_id = ?", (msg_id,))
    conn2.commit()
    conn2.close()

    pending_before = db.get_stats()["pending_embeddings"]
    await scheduler._retry_pending_embeddings()
    pending_after = db.get_stats()["pending_embeddings"]
    tc("TG8", "8.4", "Embedding 补偿重试",
       pending_after < pending_before,
       f"pending: {pending_before} -> {pending_after}")

    # TC8.5: 空闲提取 → human 画像更新（验证内容真实写入）
    db.init_core_memory_block("human", json.dumps(loader.human_template, ensure_ascii=False))
    thread_id = "tg8-idle-thread"
    messages = [
        {"role": "user", "content": "我是小研，我在杭州做宋代文学研究，希望回复简洁一些。", "_thread_id": thread_id},
    ]
    # 直接测 human 画像更新环节（process_idle_extraction 的核心），
    # 避免 scheduler 未注入 context_manager 时该环节被跳过导致假通过。
    await scheduler._update_human_from_idle(thread_id, messages)
    human_block = db.get_core_memory_block("human")
    human_content = human_block["content"] if human_block else ""
    extracted_into_human = ("杭州" in human_content) or ("宋代" in human_content)
    tc("TG8", "8.5", "空闲提取 → Core Memory 更新（内容真实写入）",
       human_block is not None and extracted_into_human,
       f"human updated, profile_extracted={extracted_into_human}, version={human_block['version'] if human_block else '?'}")

    trace_json("TG8 Stats", stats_merge)
    db.close()
    return {"tombstone_ok": True, "merge_ok": True, "retry_ok": True}


# ============================================================
#  Test Group 9: MCP Tools (MCP 工具)
# ============================================================
async def test_group_9_mcp_tools(test_dir: Path, chroma_client, embedder) -> dict:
    sep("TG9: MCP Tools — 工具测试", "=")
    trace_txt("\n" + "=" * 80)
    trace_txt("  TG9: MCP Tools — MCP 工具")
    trace_txt("=" * 80 + "\n")

    from agent.server.context.memory_db import MemoryDB
    from agent.server.context.recall_store import RecallStore
    from agent.server.context.hybrid_retriever import HybridRetriever
    from agent.server.context.fts_index import FTSIndex
    from agent.server.core_memory.persona_loader import PersonaLoader

    db_path = test_dir / "tg9_memory.db"
    if db_path.exists():
        db_path.unlink()
    db = MemoryDB(db_path)
    store = RecallStore(chroma_client=chroma_client, embedder=embedder, memory_db=db)
    store.clear_all()
    loader = PersonaLoader()
    fts_idx = FTSIndex(db)
    retriever = HybridRetriever(memory_db=db, embedder=embedder, chroma_client=chroma_client, fts_index=fts_idx)

    # 初始化 Core Memory
    db.init_core_memory_block("persona", json.dumps(loader.persona_template, ensure_ascii=False))
    db.init_core_memory_block("human", json.dumps(loader.human_template, ensure_ascii=False))

    # 注入依赖到 MCP 工具（per-user 工厂模式，测试用单实例工厂）
    from agent.server.mcp.tools import retrieve_memory as rm_tool
    from agent.server.mcp.tools import core_memory_replace as cmr_tool
    from agent.server.mcp.tools import core_memory_append as cma_tool
    from agent.server.mcp.tools import save_to_memory as stm_tool

    class _FakeMemorySystem:
        def __init__(self, db, store, retriever, loader):
            self.memory_db = db
            self.recall_store = store
            self.hybrid_retriever = retriever
            self.persona_loader = loader

    _ms = _FakeMemorySystem(db, store, retriever, loader)

    async def _factory(user_id=""):
        return _ms

    rm_tool.set_memory_system_factory(_factory)
    cmr_tool.set_memory_system_factory(_factory)
    cma_tool.set_memory_system_factory(_factory)
    stm_tool.set_memory_system_factory(_factory)

    # 添加测试记忆
    store.add_memory("用户名字是小研，最喜欢法海", memory_type="Entity", importance=0.9)
    store.add_memory("用户在杭州研究宋代民间信仰与文学互动", memory_type="Entity", importance=0.8)
    await asyncio.sleep(0.3)

    # TC9.1: retrieve_memory (hybrid)
    rm_result = await rm_tool.handler(query="小研喜欢谁", source="all", user_id="test-user")
    tc("TG9", "9.1", "retrieve_memory — 混合检索",
       "长期记忆" in rm_result.get("result", ""),
       f"result_len={len(rm_result.get('result',''))}")

    # TC9.2: retrieve_memory — empty query
    rm_empty = await rm_tool.handler(query="", source="all", user_id="test-user")
    tc("TG9", "9.2", "retrieve_memory — 空查询报错",
       "Error" in rm_empty.get("result", ""),
       f"result='{rm_empty.get('result','')[:80]}'")

    # TC9.3: core_memory_replace — persona 可变区
    cmr_result = await cmr_tool.handler(
        block="persona",
        content=json.dumps({"communication_style": {"tone": "更学术化"}}, ensure_ascii=False),
        user_id="test-user",
    )
    tc("TG9", "9.3", "core_memory_replace — persona 可变区更新",
       "updated" in cmr_result.get("result", "").lower() or "version" in cmr_result,
       f"result='{cmr_result.get('result','')[:80]}'")

    # TC9.4: core_memory_replace — 不可变区被保护
    cmr_hack = await cmr_tool.handler(
        block="persona",
        content=json.dumps({"core_identity": {"name": "HACKED"}}, ensure_ascii=False),
        user_id="test-user",
    )
    persona_after = db.get_core_memory_block("persona")
    persona_obj = json.loads(persona_after["content"]) if persona_after else {}
    tc("TG9", "9.4", "core_memory_replace — 不可变区保护",
       persona_obj.get("core_identity", {}).get("name") != "HACKED",
       f"name='{persona_obj.get('core_identity',{}).get('name','?')}'")

    # TC9.5: core_memory_append — human
    cma_result = await cma_tool.handler(
        block="human", field="preferences",
        content=json.dumps({"favorite_era": "宋代"}, ensure_ascii=False),
        user_id="test-user",
    )
    tc("TG9", "9.5", "core_memory_append — human 追加",
       "version" in cma_result or "Appended" in cma_result.get("result", ""),
       f"result='{cma_result.get('result','')[:80]}'")

    # TC9.6: save_to_memory — archival
    stm_archival = await stm_tool.handler(target="archival", content="用户对白蛇传的法海形象有深入研究", importance="high", user_id="test-user")
    tc("TG9", "9.6", "save_to_memory — archival (高重要性)",
       "Saved" in stm_archival.get("result", ""),
       f"result='{stm_archival.get('result','')[:80]}'")

    # TC9.7: save_to_memory — core
    stm_core = await stm_tool.handler(
        target="core",
        content=json.dumps({"preferences": {"detail_level": "detailed"}}, ensure_ascii=False),
        importance="medium",
        user_id="test-user",
    )
    tc("TG9", "9.7", "save_to_memory — core (更新 human 块)",
       "Saved" in stm_core.get("result", "") or "token" in stm_core.get("result", "").lower(),
       f"result='{stm_core.get('result','')[:80]}'")

    stats = db.get_stats()
    trace_json("TG9 Stats", stats)
    db.close()
    return {"tools_ok": True}


# ============================================================
#  Test Group 10: 端到端 (E2E)
# ============================================================
async def test_group_10_e2e(test_dir: Path, chroma_client, embedder) -> dict:
    sep("TG10: E2E — 端到端流水线测试", "=")
    trace_txt("\n" + "=" * 80)
    trace_txt("  TG10: E2E — 端到端流水线")
    trace_txt("=" * 80 + "\n")

    from agent.server.context.wiring import build_memory_system

    db_path = test_dir / "tg10_memory.db"
    ctx_db_path = test_dir / "tg10_context.db"
    for p in [db_path, ctx_db_path]:
        if p.exists():
            p.unlink()

    mock_llm = MockLLM()
    mock_llm.set_responses({
        "我叫小研": "用户的名字是小研。\n用户最喜欢的文学人物是雷峰塔传奇中的法海。\n用户的研究方向是宋代江南民间信仰与文学的互动。\n用户约束：不要发送长代码。",
        "白蛇传": "NO",
        "雷峰塔": "NO",
        "西湖": "NO",
        "文学研究": "用户在文学研究中偏好文本细读与宏观分析相结合。",
        "宋代话本": "NO",
        "白素贞": "NO",
        "民间信仰": "用户对宋代民间信仰与官方宗教的关系有深入研究。",
        "明清": "NO",
        "杭州": "NO",
    })

    # 与生产一致的工厂装配（fifo_max_tokens=800 用于触发压缩）
    ms = build_memory_system(
        llm=mock_llm, embedder=embedder, chroma_client=chroma_client,
        memory_db_path=db_path, context_db_path=ctx_db_path,
        fifo_max_tokens=800, summary_model="test", memory_extraction_model="test",
    )
    cm = ms.context_manager
    db = ms.memory_db
    store = ms.recall_store
    retriever = ms.hybrid_retriever
    scheduler = ms.scheduler
    store.clear_all()
    # clear_all 会连 core_memory_blocks 一起清空，这里用工厂 persona_loader 重新初始化
    db.init_core_memory_block("persona", json.dumps(ms.persona_loader.persona_template, ensure_ascii=False, indent=2))
    db.init_core_memory_block("human", json.dumps(ms.persona_loader.human_template, ensure_ascii=False, indent=2))

    thread_id = "tg10-e2e-thread"

    # ====== Phase A: 播种 ======
    trace_txt("\n--- Phase A: 播种关键信息 ---\n")
    await cm.load(thread_id)
    seed_conversation = [
        ("user", "我叫小研。我特别喜欢《雷峰塔传奇》里的法海形象，觉得他其实很复杂。"
                 "我的研究方向是宋代江南地区的民间信仰与文学文本的互动关系。"
                 "不要给我发长代码，我只需要分析结果。"),
        ("assistant", "小研您好！法海在《雷峰塔传奇》中确实值得深入探讨。"
                       "我会记住您的研究方向和不发长代码的偏好。"),
    ]
    for role, content in seed_conversation:
        await cm.enqueue(thread_id, {"role": role, "content": content})
        trace_txt(f"[SEED {role}] {content}\n")
        await asyncio.sleep(0.3)

    await asyncio.sleep(1)
    trace_txt(f"[AFTER SEED] msgs={cm.get_stats(thread_id)['message_count']}, "
              f"tokens={cm.get_stats(thread_id)['total_tokens']}\n")

    # ====== Phase B: 闲聊压缩 ======
    trace_txt("\n--- Phase B: 闲聊触发压缩 ---\n")
    chitchat = [
        ("白蛇传里除了法海还有哪些重要角色？",
         "白蛇传的核心角色包括白素贞、许仙、小青、法海。各版本形象差异显著。"),
        ("雷峰塔的历史是怎样的？",
         "雷峰塔建于975年，1924年倒塌，2002年重建。是西湖标志性景观。"),
        ("西湖十景有哪些？",
         "西湖十景包括苏堤春晓、曲院风荷、平湖秋月、断桥残雪、雷峰夕照等。"),
        ("文学研究应该更注重文本细读还是宏观分析？",
         "文本细读和宏观历史分析各有优势，理想的文学研究应该是两者的结合。"),
        ("宋代话本和明清小说有什么区别？",
         "宋代话本源于说书人底本，语言接近口语。明清小说更成熟，出现章回体。"),
        ("苏轼和杭州有什么关系？",
         "苏轼曾在杭州任职，留下了'欲把西湖比西子'的千古名句。"),
        ("白素贞和许仙的爱情为何流传千年？",
         "白蛇传触及爱情、自由、身份等永恒主题，不同时代不断被改编赋予新意义。"),
        ("民间信仰和官方宗教在宋代的关系？",
         "宋代民间信仰与官方儒释道存在复杂的互动关系，有时收编有时打压。"),
        ("明清白蛇故事的改写特点？",
         "明清改写增加道教道德教化色彩，白蛇从妖逐渐被人格化。"),
        ("杭州有哪些必去的文化景点？",
         "杭州有雷峰塔、灵隐寺、岳王庙、断桥、南宋御街等丰富文化景观。"),
    ]
    for i, (q, a) in enumerate(chitchat):
        await cm.enqueue(thread_id, {"role": "user", "content": q})
        await cm.enqueue(thread_id, {"role": "assistant", "content": a})
        if (i + 1) % 3 == 0:
            stats = cm.get_stats(thread_id)
            trace_txt(f"[ROUND {i+1}] msgs={stats['message_count']} tokens={stats['total_tokens']} "
                      f"pressure={cm.get_pressure_ratio(thread_id):.2f}\n")
        await asyncio.sleep(0.2)

    await asyncio.sleep(1)

    # ====== Phase C: 验证检索 ======
    trace_txt("\n--- Phase C: 验证检索 ---\n")
    verify_queries = [
        "用户喜欢哪个文学人物？研究方向是什么？",
        "用户叫什么名字？有什么约束？",
        "法海在哪部作品中出现？",
    ]
    for vq in verify_queries:
        result = retriever.search(vq, source="all", score_threshold=0.3)
        trace_txt(f"[VERIFY] query='{vq}' -> memories={len(result['memories'])} recalls={len(result['recalls'])}\n")
        for m in result["memories"]:
            trace_txt(f"  [MEM] score={m.get('score',0):.3f}: {m['content'][:100]}\n")
        for r in result["recalls"]:
            trace_txt(f"  [REC] score={r.get('score',0):.3f}: {r['content'][:100]}\n")

    # ====== Phase D: Core Memory 操作 ======
    trace_txt("\n--- Phase D: Core Memory 操作 ---\n")
    human_before = db.get_core_memory_block("human")
    db.append_core_memory_block("human", json.dumps({
        "research_interests": {"primary": "宋代江南民间信仰与文学互动"},
        "preferences": {"favorite_character": "法海"},
        "constraints": {"no_long_code": True},
    }, ensure_ascii=False))
    human_after = db.get_core_memory_block("human")
    trace_txt(f"[CORE MEMORY] human: v{human_before['version']} -> v{human_after['version']}\n")

    # ====== Phase E: 真实消费方法 build_memory_context_text（Core Memory + 摘要 + 被动注入）======
    trace_txt("\n--- Phase E: 被动注入 + Core Memory 注入（build_memory_context_text）---\n")
    memory_context = cm.build_memory_context_text(thread_id, "小研喜欢哪个文学人物")
    has_injection = "相关长期记忆" in memory_context or "相关历史消息" in memory_context
    trace_txt(f"[MEMORY CONTEXT] {len(memory_context)} chars, has_injection={has_injection}:\n")
    trace_txt(memory_context[:800] + ("..." if len(memory_context) > 800 else "") + "\n")

    # ====== Phase F: 后台任务 ======
    trace_txt("\n--- Phase F: 后台任务 ---\n")
    await scheduler._run_tombstone_cleanup()
    await scheduler._run_memory_merge()

    # ====== Phase G: 跨对话持久化验证 ======
    trace_txt("\n--- Phase G: 跨对话持久化 ---\n")
    await cm.save(thread_id)
    ms2 = build_memory_system(
        llm=mock_llm, embedder=embedder, chroma_client=chroma_client,
        memory_db_path=db_path, context_db_path=ctx_db_path,
        fifo_max_tokens=800, summary_model="test", memory_extraction_model="test",
    )
    cm2 = ms2.context_manager
    loaded = await cm2.load(thread_id)
    trace_txt(f"[PERSISTENCE] reloaded thread: {len(loaded.messages) if loaded else 0} messages\n")
    await cm2.close()

    # ====== 最终判定 ======
    final_stats = db.get_stats()
    trace_json("TG10 Final Stats", final_stats)

    active_memories = db.get_active_memories(limit=50)
    e2e_checks = {
        "memories_extracted": len(active_memories) > 0,
        "recall_stored": store.stats["recall_messages_count"] > 0,
        "compression_triggered": cm.get_stats(thread_id)["summary_length"] > 0 or
                                 cm.get_stats(thread_id)["total_tokens"] >= 800,
        "core_memory_updated": human_after["version"] > human_before["version"],
        "fts_indexed": final_stats["total_messages"] > 0,
        "changelog_recorded": final_stats["changelog_entries"] > 0,
        "context_persisted": loaded is not None,
        "passive_injection": has_injection,
        "scheduler_wired": scheduler is not None and scheduler._cm is not None and scheduler._extractor is not None,
    }
    all_e2e_ok = all(e2e_checks.values())
    tc("TG10", "10.0", "端到端全流水线",
       all_e2e_ok,
       f"checks: {json.dumps(e2e_checks)}")

    trace_json("E2E Checks", e2e_checks)
    await cm.close()
    db.close()
    return e2e_checks


# ============================================================
#  Test Group 11: 按对话删除记忆/召回（跨存储联动）
# ============================================================
async def test_group_11_delete_by_thread(test_dir: Path, chroma_client, embedder) -> dict:
    sep("TG11: 按对话删除记忆/召回（跨存储联动）", "=")
    trace_txt("\n" + "=" * 80)
    trace_txt("  TG11: 按对话删除记忆/召回（跨存储联动）")
    trace_txt("=" * 80 + "\n")

    from agent.server.context.memory_db import MemoryDB
    from agent.server.context.recall_store import RecallStore

    db_path = test_dir / "tg11_memory.db"
    if db_path.exists():
        db_path.unlink()
    db = MemoryDB(db_path)
    store = RecallStore(chroma_client=chroma_client, embedder=embedder, memory_db=db)
    store.clear_all()

    tid_a = "tg11-thread-a"
    tid_b = "tg11-thread-b"

    # 播种：A 对话 1 记忆 + 1 召回；B 对话 1 记忆 + 1 召回
    store.add_memory("A对话记忆：用户喜欢法海", memory_type="Entity", importance=0.9, thread_id=tid_a)
    store.add_recall_message({"role": "user", "content": "A对话召回消息原文", "_thread_id": tid_a})
    store.add_memory("B对话记忆：用户研究宋代文学", memory_type="Entity", importance=0.8, thread_id=tid_b)
    store.add_recall_message({"role": "user", "content": "B对话召回消息原文", "_thread_id": tid_b})
    await asyncio.sleep(0.3)

    # TC11.1: 记忆/召回带 thread_id 追溯
    all_msgs = db.get_all_messages()
    a_msgs = [m for m in all_msgs if m["thread_id"] == tid_a]
    b_msgs = [m for m in all_msgs if m["thread_id"] == tid_b]
    tc("TG11", "11.1", "记忆/召回带 thread_id 追溯",
       len(a_msgs) == 2 and len(b_msgs) == 2,
       f"a={len(a_msgs)} msgs, b={len(b_msgs)} msgs")

    # TC11.2: delete_thread_data 删 A 保留 B
    store.delete_by_thread(tid_a)
    stats = db.delete_thread_data(tid_a)
    a_remaining = [m for m in db.get_all_messages() if m["thread_id"] == tid_a]
    b_remaining = [m for m in db.get_all_messages() if m["thread_id"] == tid_b]
    tc("TG11", "11.2", "delete_thread_data 删 A 保留 B",
       len(a_remaining) == 0 and len(b_remaining) == 2,
       f"a_remaining={len(a_remaining)}, b_remaining={len(b_remaining)}")

    # TC11.3: long_term_memories 同步删除
    active = db.get_active_memories(limit=100)
    active_contents = "".join(m["content"] for m in active)
    tc("TG11", "11.3", "长期记忆表同步删除",
       "B对话记忆" in active_contents and "A对话记忆" not in active_contents,
       f"active contents: {active_contents[:80]}")

    # TC11.4: ChromaDB 召回向量按 thread 删除
    recall_docs_a = store._recall_collection.get(where={"thread_id": tid_a}) if store._recall_collection else {"ids": []}
    recall_docs_b = store._recall_collection.get(where={"thread_id": tid_b}) if store._recall_collection else {"ids": []}
    tc("TG11", "11.4", "ChromaDB 召回向量按 thread 删除",
       len(recall_docs_a.get("ids", [])) == 0 and len(recall_docs_b.get("ids", [])) >= 1,
       f"a_docs={len(recall_docs_a.get('ids', []))}, b_docs={len(recall_docs_b.get('ids', []))}")

    # TC11.5: delete_user_conversation_memory 复用入口（per-user factory）
    from agent.server.context import memory_registry as mr

    class _FakeMS:
        def __init__(self, db, store):
            self.memory_db = db
            self.recall_store = store
            self.context_manager = None

    _fake = _FakeMS(db, store)

    async def _factory(user_id=""):
        return _fake

    mr.set_memory_system_factory(_factory)
    result = await mr.delete_user_conversation_memory("test-user", tid_b)
    b_remaining_after = [m for m in db.get_all_messages() if m["thread_id"] == tid_b]
    tc("TG11", "11.5", "delete_user_conversation_memory 复用入口",
       result.get("status") == "ok" and len(b_remaining_after) == 0,
       f"result={result}, b_remaining={len(b_remaining_after)}")

    db.close()
    return {"thread_trace_ok": len(a_msgs) == 2, "delete_ok": len(a_remaining) == 0 and len(b_remaining_after) == 0}


# ============================================================
#  主入口
# ============================================================
async def main():
    sep("记忆与上下文管理系统 — 完整功能验证", "=")
    trace(f"开始时间: {bj_now().isoformat()}")
    trace_txt(f"测试开始: {bj_now().isoformat()}\n")

    # ---- 初始化共享组件 ----
    from agent.server.config import settings as app_settings
    from agent.server.rag.config import rag_config

    import chromadb
    # 用临时 chroma 目录，避免污染/读取生产 RAG 的 chroma 数据（旧格式 collection 会导致 _type KeyError）
    chroma_dir = LOG_DIR / "chroma_exhaustive"
    if chroma_dir.exists():
        shutil.rmtree(chroma_dir, ignore_errors=True)
    chroma_dir.mkdir(parents=True, exist_ok=True)
    trace(f"[INIT] ChromaDB: {chroma_dir}")
    chroma_client = chromadb.PersistentClient(path=str(chroma_dir))

    from agent.server.rag.embedder import Embedder
    embedder = Embedder(
        model_name=rag_config.embedding_model,
        local_path=rag_config.embedding_local_path or None,
    )
    embedder.start_loading()
    logger.info("加载 Embedder...")
    trace("[INIT] 加载 Embedder...")
    for i in range(60):
        if embedder.is_ready:
            break
        await asyncio.sleep(1)
    if not embedder.is_ready:
        raise RuntimeError("Embedder 加载超时")
    logger.info("Embedder 就绪: %s", rag_config.embedding_model)
    trace(f"[INIT] Embedder 就绪: {rag_config.embedding_model}")

    test_dir = LOG_DIR

    # ---- 运行所有测试组 ----
    all_results = {}

    try:
        all_results["TG1"] = await test_group_1_memory_db(test_dir)
    except Exception as e:
        logger.error("TG1 失败: %s", e, exc_info=True)
        all_results["TG1"] = {"error": str(e)}
        tc("TG1", "FATAL", "测试组异常终止（未完成全部用例）", False, str(e)[:200])

    try:
        all_results["TG2"] = await test_group_2_recall_store(test_dir, chroma_client, embedder)
    except Exception as e:
        logger.error("TG2 失败: %s", e, exc_info=True)
        all_results["TG2"] = {"error": str(e)}
        tc("TG2", "FATAL", "测试组异常终止（未完成全部用例）", False, str(e)[:200])

    try:
        all_results["TG3"] = await test_group_3_memory_extractor(test_dir, chroma_client, embedder)
    except Exception as e:
        logger.error("TG3 失败: %s", e, exc_info=True)
        all_results["TG3"] = {"error": str(e)}
        tc("TG3", "FATAL", "测试组异常终止（未完成全部用例）", False, str(e)[:200])

    try:
        all_results["TG4"] = await test_group_4_hybrid_retriever(test_dir, chroma_client, embedder)
    except Exception as e:
        logger.error("TG4 失败: %s", e, exc_info=True)
        all_results["TG4"] = {"error": str(e)}
        tc("TG4", "FATAL", "测试组异常终止（未完成全部用例）", False, str(e)[:200])

    try:
        all_results["TG5"] = await test_group_5_memory_scorer(test_dir)
    except Exception as e:
        logger.error("TG5 失败: %s", e, exc_info=True)
        all_results["TG5"] = {"error": str(e)}
        tc("TG5", "FATAL", "测试组异常终止（未完成全部用例）", False, str(e)[:200])

    try:
        all_results["TG6"] = await test_group_6_core_memory(test_dir)
    except Exception as e:
        logger.error("TG6 失败: %s", e, exc_info=True)
        all_results["TG6"] = {"error": str(e)}
        tc("TG6", "FATAL", "测试组异常终止（未完成全部用例）", False, str(e)[:200])

    try:
        all_results["TG7"] = await test_group_7_context_manager(test_dir, chroma_client, embedder)
    except Exception as e:
        logger.error("TG7 失败: %s", e, exc_info=True)
        all_results["TG7"] = {"error": str(e)}
        tc("TG7", "FATAL", "测试组异常终止（未完成全部用例）", False, str(e)[:200])

    try:
        all_results["TG8"] = await test_group_8_background_tasks(test_dir, chroma_client, embedder)
    except Exception as e:
        logger.error("TG8 失败: %s", e, exc_info=True)
        all_results["TG8"] = {"error": str(e)}
        tc("TG8", "FATAL", "测试组异常终止（未完成全部用例）", False, str(e)[:200])

    try:
        all_results["TG9"] = await test_group_9_mcp_tools(test_dir, chroma_client, embedder)
    except Exception as e:
        logger.error("TG9 失败: %s", e, exc_info=True)
        all_results["TG9"] = {"error": str(e)}
        tc("TG9", "FATAL", "测试组异常终止（未完成全部用例）", False, str(e)[:200])

    try:
        all_results["TG10"] = await test_group_10_e2e(test_dir, chroma_client, embedder)
    except Exception as e:
        logger.error("TG10 失败: %s", e, exc_info=True)
        all_results["TG10"] = {"error": str(e)}
        tc("TG10", "FATAL", "测试组异常终止（未完成全部用例）", False, str(e)[:200])

    try:
        all_results["TG11"] = await test_group_11_delete_by_thread(test_dir, chroma_client, embedder)
    except Exception as e:
        logger.error("TG11 失败: %s", e, exc_info=True)
        all_results["TG11"] = {"error": str(e)}
        tc("TG11", "FATAL", "测试组异常终止（未完成全部用例）", False, str(e)[:200])

    # ---- 最终报告 ----
    sep("测试结果汇总", "=")
    trace()
    trace(f"  总测试用例: {_test_results['total']}")
    trace(f"  通过: {_test_results['passed']}")
    trace(f"  失败: {_test_results['failed']}")
    trace(f"  通过率: {_test_results['passed']/_test_results['total']*100:.1f}%" if _test_results['total'] > 0 else "N/A")
    trace()
    trace("详细结果:")
    for detail in _test_results["details"]:
        trace(f"  {detail}")
    trace()

    trace_txt(f"\n{'='*80}\n")
    trace_txt(f"  总测试用例: {_test_results['total']}\n")
    trace_txt(f"  通过: {_test_results['passed']}  |  失败: {_test_results['failed']}\n")
    trace_txt(f"{'='*80}\n")
    for detail in _test_results["details"]:
        trace_txt(f"  {detail}\n")

    # 关闭追踪文件
    global _tf_log, _tf_txt
    for f in [_tf_log, _tf_txt]:
        if f:
            f.close()

    logger.info("日志文件: %s", TRACE_PATH)
    logger.info("纯文本日志: %s", TRACE_TXT_PATH)

    return _test_results["failed"] == 0


if __name__ == "__main__":
    start = time.monotonic()
    success = asyncio.run(main())
    elapsed = time.monotonic() - start
    logger.info("全部测试完成: %.1f 秒, 结果: %s", elapsed,
                "全部通过" if success else f"{_test_results['failed']} 个失败")
    sys.exit(0 if success else 1)
