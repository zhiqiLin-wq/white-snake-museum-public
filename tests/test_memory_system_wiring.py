"""记忆系统接线自检测试。

目标: 验证 main.py 实际使用的接线工厂 build_memory_system / register_memory_tools
把所有组件真正接线到位（不绕过 main.py 的组装代码）。

这与 test_memory_system_exhaustive.py 的区别:
- exhaustive 测试在每个 TG 里自己构造组件，绕过了生产接线，抓不到 P0 断链
- 本测试直接调用 main.py 同款的 wiring 工厂，用轻量 mock 依赖，
  精确断言"每个组件是否被注入到它该在的地方"

覆盖的问题:
- P0-1: RecallStore 缺少 memory_db
- P0-2: ContextManager 缺少 memory_db/hybrid_retriever/persona_loader/scheduler
- P0-3: MemoryDB 未创建
- P0-4: MCP 工具未注册/未注入
- P0-5: BackgroundTaskScheduler / PersonaLoader 未创建/启动

运行方式:
  cd white-snake-museum-public
  python tests/test_memory_system_wiring.py
"""

import asyncio
import hashlib
import json
import logging
import shutil
import sys
import tempfile
from datetime import datetime, timezone, timedelta
from pathlib import Path

_project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_project_root))

BEIJING_TZ = timezone(timedelta(hours=8))

# ============================================================
#  日志：控制台 + 文件 trace_wiring.log
# ============================================================
LOG_DIR = _project_root / "tests" / "_test_data"
LOG_DIR.mkdir(parents=True, exist_ok=True)
TRACE_PATH = LOG_DIR / "trace_wiring.log"


def bj_now() -> datetime:
    return datetime.now(BEIJING_TZ)


class _BjFormatter(logging.Formatter):
    def formatTime(self, record, datefmt=None):
        dt = datetime.fromtimestamp(record.created, tz=BEIJING_TZ)
        return dt.strftime("%H:%M:%S") if datefmt is None else dt.strftime(datefmt)


root_logger = logging.getLogger()
root_logger.setLevel(logging.DEBUG)
root_logger.handlers.clear()

_fmt = _BjFormatter("%(asctime)s %(message)s", datefmt="%H:%M:%S")

_console = logging.StreamHandler(sys.stdout)
_console.setLevel(logging.INFO)
_console.setFormatter(_fmt)
root_logger.addHandler(_console)

_file = logging.FileHandler(str(TRACE_PATH), mode="a", encoding="utf-8")
_file.setLevel(logging.DEBUG)
_file.setFormatter(_fmt)
root_logger.addHandler(_file)

logger = logging.getLogger("wiring_test")

_results = {"passed": 0, "failed": 0, "total": 0, "details": []}


def tc(group: str, tc_id: str, description: str, condition: bool, detail: str = ""):
    _results["total"] += 1
    status = "PASS" if condition else "FAIL"
    if condition:
        _results["passed"] += 1
    else:
        _results["failed"] += 1
    entry = f"[{status}] {group}-{tc_id}: {description}"
    if detail:
        entry += f" — {detail}"
    _results["details"].append(entry)
    logger.info(entry)


def sep(title: str = ""):
    logger.info("")
    logger.info("=" * 100)
    if title:
        logger.info(f"  {title}")
    logger.info("=" * 100)


# ============================================================
#  轻量 mock 依赖
# ============================================================
class FakeEmbedder:
    """文本 hash 派生的确定性向量，避免加载真实 bge 模型。"""

    def __init__(self, dim: int = 16):
        self._dim = dim
        self.is_ready = True

    def start_loading(self):
        pass

    def _vec(self, text: str) -> list[float]:
        h = hashlib.md5(text.encode("utf-8")).digest()
        return [((h[i % len(h)] / 255.0) - 0.5) for i in range(self._dim)]

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._vec(str(t)) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vec(text)


class MockLLMResponse:
    def __init__(self, content: str):
        self.content = content


class MockLLM:
    """返回固定文本的 LLM，用于摘要/提取调用。"""

    async def generate(self, system: str, user: str, model: str = "",
                       max_tokens: int = 256, temperature: float = 0.3):
        return MockLLMResponse("用户喜欢法海这个文学人物")


# ============================================================
#  W1: 组件接线
# ============================================================
def test_w1_component_wiring(workdir: Path):
    sep("W1: 组件接线 — build_memory_system 是否把每个依赖注入到位")
    import chromadb
    from agent.server.context.wiring import build_memory_system

    chroma_dir = workdir / "chroma"
    chroma_client = chromadb.PersistentClient(path=str(chroma_dir))
    embedder = FakeEmbedder()
    llm = MockLLM()

    sys_ = build_memory_system(
        llm=llm,
        embedder=embedder,
        chroma_client=chroma_client,
        memory_db_path=workdir / "memory.db",
        context_db_path=workdir / "context_state.db",
        fifo_max_tokens=2000,
        summary_model="mock-model",
        memory_extraction_model="mock-model",
        core_memory_enabled=True,
        background_tasks_enabled=True,
    )

    # P0-3: MemoryDB 被创建
    tc("W1", "1", "MemoryDB 已创建",
       sys_.memory_db is not None)
    # P0-1: RecallStore 收到 memory_db
    tc("W1", "2", "RecallStore 注入了 MemoryDB (P0-1)",
       sys_.recall_store is not None and getattr(sys_.recall_store, "_db", None) is sys_.memory_db,
       f"recall_store._db is memory_db = {getattr(sys_.recall_store, '_db', None) is sys_.memory_db}")
    # HybridRetriever 已创建
    tc("W1", "3", "HybridRetriever 已创建",
       sys_.hybrid_retriever is not None)
    # P0-5: PersonaLoader 已创建
    tc("W1", "4", "PersonaLoader 已创建 (P0-5)",
       sys_.persona_loader is not None)
    # P0-5: BackgroundTaskScheduler 已创建
    tc("W1", "5", "BackgroundTaskScheduler 已创建 (P0-5)",
       sys_.scheduler is not None)

    cm = sys_.context_manager
    tc("W1", "6", "ContextManager 注入了 memory_db (P0-2)",
       cm is not None and getattr(cm, "_memory_db", None) is sys_.memory_db)
    tc("W1", "7", "ContextManager 注入了 hybrid_retriever (P0-2)",
       cm is not None and getattr(cm, "_hybrid_retriever", None) is sys_.hybrid_retriever)
    tc("W1", "8", "ContextManager 注入了 persona_loader (P0-2)",
       cm is not None and getattr(cm, "_persona_loader", None) is sys_.persona_loader)
    tc("W1", "9", "ContextManager 注入了 background_scheduler (P0-2)",
       cm is not None and getattr(cm, "_scheduler", None) is sys_.scheduler)

    # Core memory 块从 persona.yaml 模板初始化
    persona = sys_.memory_db.get_core_memory_block("persona")
    human = sys_.memory_db.get_core_memory_block("human")
    tc("W1", "10", "persona 块从模板初始化",
       persona is not None and "白蛇传文脉全息博物馆" in (persona.get("content") or ""))
    tc("W1", "11", "human 块从模板初始化",
       human is not None and human.get("content"))

    return sys_


# ============================================================
#  W2: MCP 工具注册
# ============================================================
def test_w2_tool_registration(memory_system):
    sep("W2: MCP 工具注册 — register_memory_tools 是否注册并注入依赖")
    from agent.server.mcp.registry import MCPToolRegistry
    from agent.server.context.wiring import register_memory_tools
    from agent.server.mcp.tools import retrieve_memory, save_to_memory
    from agent.server.mcp.tools import core_memory_append, core_memory_replace

    mcp_reg = MCPToolRegistry()
    register_memory_tools(mcp_reg, memory_system)

    tc("W2", "1", "retrieve_memory 已注册",
       mcp_reg.get_tool("retrieve_memory") is not None)
    tc("W2", "2", "retrieve_memory 注入了 hybrid_retriever (P0-4)",
       retrieve_memory._hybrid_retriever is memory_system.hybrid_retriever)
    tc("W2", "3", "retrieve_memory 注入了 recall_store",
       retrieve_memory._recall_store is memory_system.recall_store)

    tc("W2", "4", "save_to_memory 已注册 (P0-4)",
       mcp_reg.get_tool("save_to_memory") is not None)
    tc("W2", "5", "save_to_memory 注入了 memory_db (P0-4)",
       save_to_memory._memory_db is memory_system.memory_db)
    tc("W2", "6", "save_to_memory 注入了 recall_store",
       save_to_memory._recall_store is memory_system.recall_store)
    tc("W2", "7", "save_to_memory 注入了 persona_loader",
       save_to_memory._persona_loader is memory_system.persona_loader)

    tc("W2", "8", "core_memory_append 已注册 (P0-4)",
       mcp_reg.get_tool("core_memory_append") is not None)
    tc("W2", "9", "core_memory_append 注入了 memory_db",
       core_memory_append._memory_db is memory_system.memory_db)
    tc("W2", "10", "core_memory_replace 已注册 (P0-4)",
       mcp_reg.get_tool("core_memory_replace") is not None)
    tc("W2", "11", "core_memory_replace 注入了 persona_loader",
       core_memory_replace._persona_loader is memory_system.persona_loader)

    return mcp_reg


# ============================================================
#  W3: 功能冒烟 — 接线后组件真的能协作
# ============================================================
async def test_w3_smoke(memory_system, mcp_reg):
    sep("W3: 功能冒烟 — 接线后组件协作（存记忆 -> 检索 -> 入队）")

    # 1. 存一条记忆
    mid = memory_system.recall_store.add_memory(
        "用户的名字是小研", memory_type="Entity", importance=0.9, confidence=0.8,
    )
    tc("W3", "1", "双写长期记忆成功",
       mid is not None, f"memory_id={mid}")

    # 2. 混合检索能召回
    result = memory_system.hybrid_retriever.search("用户叫什么名字", source="memories",
                                                    score_threshold=0.0)
    memories = result.get("memories", [])
    tc("W3", "2", "混合检索能召回刚存的记忆",
       len(memories) > 0 and any("小研" in str(m.get("content", "")) for m in memories),
       f"召回 {len(memories)} 条: {[m.get('content') for m in memories]}")

    # 3. retrieve_memory 工具端到端可用（走 hybrid 路径，不报错，返回 result）
    ret = await mcp_reg.call_tool("retrieve_memory", {"query": "用户叫什么名字"})
    inner = ret.get("result") if isinstance(ret.get("result"), dict) else {}
    ok = "error" not in ret and isinstance(inner.get("result"), str) and len(inner["result"]) > 0
    tc("W3", "3", "retrieve_memory 工具端到端可用（走 hybrid 路径）",
       ok, f"result 前 80 字: {str(inner.get('result', ''))[:80]}")

    # 4. context_manager 入队 + 构建上下文窗口（含 Core Memory + 被动注入）
    await memory_system.context_manager.enqueue(
        "thread-w3", {"role": "user", "content": "小研喜欢哪个文学人物"},
    )
    ctx = memory_system.context_manager.build_context_window_with_core_memory(
        "thread-w3", "小研喜欢哪个文学人物",
    )
    has_persona = any("Core Memory" in str(m.get("content", "")) for m in ctx)
    has_user = any(m.get("role") == "user" for m in ctx)
    tc("W3", "4", "上下文窗口包含 Core Memory 注入",
       has_persona, f"上下文 {len(ctx)} 条消息")
    tc("W3", "5", "上下文窗口包含用户消息",
       has_user)


# ============================================================
#  W4: scheduler 生命周期
# ============================================================
async def test_w4_scheduler(memory_system):
    sep("W4: 后台调度器生命周期 — start/stop")
    sched = memory_system.scheduler
    tc("W4", "1", "scheduler 存在", sched is not None)
    if sched is None:
        return
    await sched.start()
    tc("W4", "2", "scheduler.start() 后 _running=True",
       getattr(sched, "_running", False) is True)
    await sched.stop()
    tc("W4", "3", "scheduler.stop() 后 _running=False",
       getattr(sched, "_running", False) is False)


# ============================================================
#  入口
# ============================================================
async def main():
    logger.info("开始时间: %s", bj_now().isoformat())
    logger.info("追踪文件: %s", TRACE_PATH)

    workdir = Path(tempfile.mkdtemp(prefix="wiring_test_"))

    try:
        memory_system = test_w1_component_wiring(workdir)
        mcp_reg = test_w2_tool_registration(memory_system)
        await test_w3_smoke(memory_system, mcp_reg)
        await test_w4_scheduler(memory_system)
    finally:
        # 清理临时目录（chroma 可能被占用，尽力而为）
        try:
            shutil.rmtree(workdir, ignore_errors=True)
        except Exception:
            pass

    sep("结果汇总")
    logger.info("总测试用例: %d", _results["total"])
    logger.info("通过: %d", _results["passed"])
    logger.info("失败: %d", _results["failed"])
    logger.info("通过率: %.1f%%", 100.0 * _results["passed"] / max(1, _results["total"]))
    logger.info("")
    for d in _results["details"]:
        logger.info("  %s", d)

    return _results["failed"] == 0


if __name__ == "__main__":
    ok = asyncio.run(main())
    sys.exit(0 if ok else 1)
