"""E7 检索并发限流 + 基础设施错误不静默降级（回归测试，测试先行）。

背景：2026-09-09 演化分析 E7 test_claims_retrieve 节点，6 claims ×
(4 朝代 + extra + 反例) 共约 30 路检索通过 asyncio.gather 无信号量并发，
每路都要跑 BGE-large 前向（9-07 提速改造后 embedding 走 asyncio.to_thread
真并行），峰值 ~30 个模型前向同时分配内存，触发 PyTorch OOM
（DefaultCPUAllocator: not enough memory；同日早些时候服务已三次因
"页面文件太小 os error 1455" 加载模型失败）。

更严重的是错误被两层静默吞掉：
1. ToolDispatcher.dispatch_retrieve 捕获 RuntimeError 后 return []，
   把基础设施故障伪装成"查无证据"；
2. E7 _cached_retrieve 再次 except Exception: docs = []。
导致 6/6 主张在 0.0s 内被误判 verdict=insufficient，用户看到的是
"全部证据不足"的假结论而非服务故障提示。

门禁：
1. E7 检索全局 asyncio 并发 ≤ E7_RETRIEVE_CONCURRENCY(4)
2. Embedder 前向线程并发 ≤ EMBEDDER_MAX_CONCURRENT_FORWARD(2)
3. dispatch_retrieve 遇基础设施错误必须抛出，禁止返回 []
4. E7 全部 claim 检索失败时节点必须抛错，禁止产出 insufficient 误判
5. E7 部分 claim 失败时，失败 claim 标记 verdict=retrieval_error，
   不进入 AI 判定

运行: .venv\\Scripts\\python tests/test_e7_retrieve_guardrails.py
   或: pytest tests/test_e7_retrieve_guardrails.py
"""
import asyncio
import sys
import threading
import time
import types
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

_project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_project_root))

# 与实现侧保持一致的并发上限常量
E7_RETRIEVE_CONCURRENCY = 4
EMBEDDER_MAX_CONCURRENT_FORWARD = 2


# ---------------------------------------------------------------------------
# 公共 stub 构造
# ---------------------------------------------------------------------------

def _make_workbench_stub(fake_retrieve, fake_opposing=None):
    """构造一个仅挂载 E7 检索节点所需属性的 workbench stub。

    _test_claims_retrieve 只用到 self._filtered_retrieve /
    self._generate_opposing_query / self._emit_live。
    """
    stub = types.SimpleNamespace()
    stub._filtered_retrieve = fake_retrieve

    async def _default_opposing(statement, target):
        return ""  # 空串 → 跳过反例检索分支，聚焦正向检索并发

    stub._generate_opposing_query = fake_opposing or _default_opposing
    stub._emit_live = lambda event: None  # 同步 fire-and-forget
    return stub


def _make_state(n_claims=6):
    return {
        "tracking_target": "白素贞",
        "claims": [
            {"statement": f"第{i}条主张：白蛇形象在各朝代文本中的具体表现差异",
             "verdict": "pending"}
            for i in range(n_claims)
        ],
    }


def _make_doc(dynasty="唐"):
    return {
        "content": f"证据片段-{dynasty}-{time.monotonic_ns()}",
        "metadata": {"dynasty": dynasty, "chapter_title": "测试章节",
                     "source_type": "primary_literature"},
    }


# ---------------------------------------------------------------------------
# 1. E7 检索并发上限
# ---------------------------------------------------------------------------

def test_e7_retrieve_concurrency_capped():
    """6 claims fan-out 约 36 次检索，全局并发必须 ≤ 4。"""
    from agent.server.langgraph.evolution_workbench import EvolutionWorkbench

    state = {"current": 0, "peak": 0}
    lock = threading.Lock()

    async def fake_retrieve(query, filters=None, top_k=None, skip_tagging=False):
        with lock:
            state["current"] += 1
            state["peak"] = max(state["peak"], state["current"])
        await asyncio.sleep(0.05)
        with lock:
            state["current"] -= 1
        dy = (filters or {}).get("dynasty") or "唐"
        return [_make_doc(dy)]

    stub = _make_workbench_stub(fake_retrieve)

    result = asyncio.run(
        EvolutionWorkbench._test_claims_retrieve(stub, _make_state(6))
    )

    assert state["peak"] <= E7_RETRIEVE_CONCURRENCY, (
        f"E7 检索并发峰值 {state['peak']} 超过上限 "
        f"{E7_RETRIEVE_CONCURRENCY}（OOM 事故根因复现）"
    )
    # 正常路径：6 条 claim 都应带证据池返回
    claims = result["claims"]
    assert len(claims) == 6
    assert all(c.get("evidence_pool") for c in claims), (
        "正常检索时每条主张都应有 evidence_pool"
    )


# ---------------------------------------------------------------------------
# 2. Embedder 前向并发上限
# ---------------------------------------------------------------------------

def test_embedder_forward_concurrency_capped():
    """16 线程并发 embed_query，模型 forward 并发必须 ≤ 2。"""
    import torch
    from agent.server.rag.embedder import Embedder

    emb = Embedder("fake-model-for-concurrency-test")

    stats = {"current": 0, "peak": 0}
    lock = threading.Lock()

    class FakeOut:
        def __init__(self, batch):
            self.last_hidden_state = torch.zeros(batch, 4, 1024)

    class FakeModel:
        def __call__(self, **enc):
            with lock:
                stats["current"] += 1
                stats["peak"] = max(stats["peak"], stats["current"])
            time.sleep(0.1)  # 模拟前向计算耗时
            with lock:
                stats["current"] -= 1
            batch = enc["input_ids"].shape[0]
            return FakeOut(batch)

    class FakeTokenizer:
        def __call__(self, texts, padding, truncation, max_length, return_tensors):
            n = len(texts)
            return {
                "input_ids": torch.zeros((n, 4), dtype=torch.long),
                "attention_mask": torch.ones((n, 4), dtype=torch.long),
                "token_type_ids": torch.zeros((n, 4), dtype=torch.long),
            }

    emb._model = FakeModel()
    emb._tokenizer = FakeTokenizer()

    with ThreadPoolExecutor(max_workers=16) as pool:
        list(pool.map(lambda i: emb.embed_query(f"测试查询{i}"), range(16)))

    assert stats["peak"] <= EMBEDDER_MAX_CONCURRENT_FORWARD, (
        f"Embedder 前向并发峰值 {stats['peak']} 超过上限 "
        f"{EMBEDDER_MAX_CONCURRENT_FORWARD}，多路检索同时前向会 OOM"
    )


# ---------------------------------------------------------------------------
# 3. dispatch_retrieve 基础设施错误必须抛出
# ---------------------------------------------------------------------------

def test_dispatch_retrieve_reraises_infra_error():
    """retriever 抛 RuntimeError（模拟 OOM）时，dispatch_retrieve 必须
    向上抛，禁止 return [] 把故障伪装成空结果。"""
    from agent.server.dispatch.tool_dispatcher import ToolDispatcher

    retriever = types.SimpleNamespace()
    retriever.supports_filters = False   # 走"过滤值并入查询词"分支（事故路径）
    retriever.supports_skip_tagging = True

    async def fake_retrieve(*a, **kw):
        raise RuntimeError(
            "[enforce fail at alloc_cpu.cpp:117] data. "
            "DefaultCPUAllocator: not enough memory"
        )

    retriever.retrieve = fake_retrieve
    dp = ToolDispatcher(mcp_registry=None, retriever=retriever)

    try:
        asyncio.run(dp.dispatch_retrieve(
            "白素贞 明代 大蟒蛇", caller="test",
            filters={"source_type": "primary_literature", "dynasty": "明"},
        ))
    except RuntimeError:
        pass  # 期望路径：基础设施错误向上传播
    else:
        raise AssertionError(
            "dispatch_retrieve 吞掉 RuntimeError 返回了空结果——"
            "基础设施故障被伪装成查无证据（E7 误判根因）"
        )


# ---------------------------------------------------------------------------
# 4. E7 全部 claim 检索失败 → 节点抛错，不产出 insufficient 误判
# ---------------------------------------------------------------------------

def test_e7_all_claims_infra_failure_raises():
    """模拟 OOM：所有检索都抛 RuntimeError，节点必须整体抛错。"""
    from agent.server.langgraph.evolution_workbench import EvolutionWorkbench

    async def fake_retrieve_oom(query, filters=None, top_k=None, skip_tagging=False):
        raise RuntimeError("DefaultCPUAllocator: not enough memory")

    stub = _make_workbench_stub(fake_retrieve_oom)

    try:
        asyncio.run(
            EvolutionWorkbench._test_claims_retrieve(stub, _make_state(6))
        )
    except RuntimeError as e:
        assert "检索" in str(e) or "重试" in str(e), (
            f"错误提示应告知用户检索故障/可重试，实际: {e}"
        )
    else:
        raise AssertionError(
            "全部 claim 检索基础设施失败时节点未抛错——"
            "旧逻辑会让 6/6 主张带着 0 证据进入判定并误判 insufficient"
        )


# ---------------------------------------------------------------------------
# 5. E7 部分 claim 失败 → 失败 claim 标 retrieval_error，不进入判定
# ---------------------------------------------------------------------------

def test_e7_partial_failure_marks_retrieval_error():
    from agent.server.langgraph.evolution_workbench import EvolutionWorkbench

    async def fake_retrieve(query, filters=None, top_k=None, skip_tagging=False):
        if "注定失败的主张" in query:
            raise RuntimeError("DefaultCPUAllocator: not enough memory")
        dy = (filters or {}).get("dynasty") or "唐"
        return [_make_doc(dy)]

    stub = _make_workbench_stub(fake_retrieve)
    state = {
        "tracking_target": "白素贞",
        "claims": [
            {"statement": "正常主张：明代文本中白娘子自称大蟒蛇", "verdict": "pending"},
            {"statement": "注定失败的主张：清代版本改写为白龙转世", "verdict": "pending"},
        ],
    }

    result = asyncio.run(EvolutionWorkbench._test_claims_retrieve(stub, state))
    claims = result["claims"]

    errored = [c for c in claims if c.get("verdict") == "retrieval_error"]
    pending = [c for c in claims if c.get("verdict") == "pending"]

    assert len(errored) == 1, (
        f"应有 1 条 claim 标记 retrieval_error，实际 {len(errored)}"
    )
    assert "RuntimeError" in errored[0].get("retrieval_error", ""), (
        "retrieval_error 字段应保留原始错误类型，便于排查"
    )
    assert len(pending) == 1 and pending[0].get("evidence_pool"), (
        "正常 claim 应保持 pending 且带证据池进入后续判定"
    )


# ---------------------------------------------------------------------------
# 独立运行入口（无 pytest 时也可跑）
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    _tests = [
        test_e7_retrieve_concurrency_capped,
        test_embedder_forward_concurrency_capped,
        test_dispatch_retrieve_reraises_infra_error,
        test_e7_all_claims_infra_failure_raises,
        test_e7_partial_failure_marks_retrieval_error,
    ]
    _passed = _failed = 0
    for t in _tests:
        try:
            t()
            print(f"PASS  {t.__name__}")
            _passed += 1
        except AssertionError as e:
            print(f"FAIL  {t.__name__}: {e}")
            _failed += 1
        except Exception as e:
            print(f"ERROR {t.__name__}: {type(e).__name__}: {e}")
            _failed += 1
    print(f"\n{_passed} passed, {_failed} failed")
    sys.exit(1 if _failed else 0)
