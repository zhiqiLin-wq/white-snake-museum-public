"""FastAPI 入口 + lifespan 事件。"""

# Monkey-patch: aiosqlite 0.22.x Connection lacks is_alive() required by langgraph-checkpoint-sqlite 3.x
# Must run BEFORE any langgraph import
import aiosqlite as _aiosqlite_patched
import threading as _threading_patched
if not hasattr(_aiosqlite_patched.Connection, 'is_alive'):
    def _conn_is_alive_patch(self):
        thread = getattr(self, '_thread', None)
        return thread is not None and thread.is_alive()
    _aiosqlite_patched.Connection.is_alive = _conn_is_alive_patch

import logging
import time
import sys
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .config import settings
from .rag.config import rag_config
from .api.routes import router
from .api.middleware import RequestLogMiddleware
from .security.rate_limiter import RateLimitMiddleware

logger = logging.getLogger("agent")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """服务生命周期管理。"""
    import asyncio

    # ========== 启动 ==========
    logger.info("=" * 60)
    logger.info("[White Snake Agent] Starting...")
    logger.info("=" * 60)

    # ---- 1. 初始化日志 ----
    # 控制台 handler（PlainFormatter，人类易读）
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    # 业务日志文件（JSON 行式，logs/agent.log，按天切分保留 30 天）
    from .observability.logger import setup_file_logging, setup_llm_logging, setup_tool_logging
    log_dir = setup_file_logging()
    # LLM 调用独立日志（单独文件 logs/llm/llm_calls.log，按天切分保留 60 天）
    setup_llm_logging(log_dir=log_dir)
    # 工具调用独立日志（单独文件 logs/tool/tool_calls.log，按天切分保留 60 天）
    setup_tool_logging(log_dir=log_dir)

    # ---- 2. LLM Provider ----
    from .api import routes as api_routes

    if settings.llm_provider == "deepseek":
        from .llm.deepseek_provider import DeepSeekProvider
        llm = DeepSeekProvider()
        api_routes.llm_provider = llm
        logger.info(f"[OK] LLM: DeepSeek ({settings.deepseek_model})")
    else:
        from .llm.anthropic_provider import AnthropicProvider
        llm = AnthropicProvider()
        api_routes.llm_provider = llm
        logger.info(f"[OK] LLM: Anthropic ({settings.anthropic_model})")

    # ---- 3. Node API 客户端 ----
    from .data.client import NodeAPIClient
    node_client = NodeAPIClient()
    api_routes.node_client = node_client
    node_ok = await node_client.health_check()
    logger.info(f"[{'OK' if node_ok else 'WARN'}] Node API: {'connected' if node_ok else 'unreachable'}")

    # ---- 4. Embedding 模型（后台加载，不阻塞启动） ----
    from .rag.embedder import Embedder
    embedder = Embedder(
        model_name=rag_config.embedding_model,
        local_path=rag_config.embedding_local_path or None,
    )
    embedder.start_loading()  # 后台线程下载，不阻塞
    api_routes.embedder = embedder  # 暴露给 health 端点
    logger.info(f"[WAIT] Embedding model loading in background: {rag_config.embedding_model}")

    # ---- 5. RAG 管线（延迟初始化，等模型就绪后构建索引） ----
    from .rag.chunker import Chunker
    from .rag.vector_store import VectorStore
    from .rag.retriever import HybridRetriever
    from .rag.index_watchdog import IndexWatchdog

    # CU-43: Chunker 新签名 Chunker(data_dir, embedder=embedder)
    # U12: 传入进度回调
    from .api.routes import _make_chunking_progress_callback
    chunker = Chunker(
        data_dir=rag_config.data_dir_path,
        embedder=embedder,
        progress_callback=_make_chunking_progress_callback(),
        embed_batch_size=4,  # 重建时小批次 embedding：本机可用内存仅 ~3GB，批 16 会原生层崩溃（0xC0000005），批 4 实测稳定
    )

    # CU-41: 初始化 BM25Retriever
    from .rag.bm25_retriever import BM25Retriever, BM25Config
    bm25_config = BM25Config(
        k1=rag_config.bm25_k1,
        b=rag_config.bm25_b,
        tokenizer=rag_config.bm25_tokenizer,
        top_k_multiplier=rag_config.bm25_top_k_multiplier,
    )
    bm25 = BM25Retriever(config=bm25_config)
    logger.info(f"[OK] BM25 Retriever initialized (tokenizer={rag_config.bm25_tokenizer})")

    # CU-42: 初始化 Reranker + 错峰后台加载
    # 1) reranker_enabled=false 时不构造不加载（省 ~1.5GB，检索走 dense+BM25 融合）
    # 2) 内存受限环境（commit 逼近上限）下与 embedder 并发加载两个 GB 级模型，
    #    commit 会瞬时超限 → torch 原生层崩溃（进程无声消失、无 Python 报错）。
    #    开启时也等 embedder 就绪后再加载，错开峰值内存。
    reranker = None
    if rag_config.reranker_enabled:
        from .rag.reranker import Reranker
        reranker = Reranker(
            model_name=rag_config.reranker_model,
            backend=getattr(rag_config, "reranker_backend", "onnx"),
        )

        def _deferred_reranker_load():
            for _ in range(240):  # 最多等 120s
                if embedder.is_ready or embedder.load_error:
                    break
                time.sleep(0.5)
            reranker.start_loading()
            logger.info(f"[WAIT] Reranker loading in background (after embedder): {rag_config.reranker_model} (backend={getattr(rag_config, 'reranker_backend', 'onnx')})")

        _threading_patched.Thread(target=_deferred_reranker_load, daemon=True, name="reranker-deferred-load").start()
    else:
        logger.info("[OK] Reranker disabled (RERANKER_ENABLED=false), 检索使用 dense+BM25 融合")

    # CU-40: VectorStore 注入 bm25_retriever
    vs = VectorStore(
        persist_dir=rag_config.chroma_persist_path,
        embedder=embedder,
        bm25_retriever=bm25,
    )

    # CU-44: HybridRetriever 注入 bm25 + reranker + query_preprocessor
    # Phase 0: 查询预处理 (P0: HyDE 段落 + P1: 别名扩展)
    # 双路分离: dense_query (自然文言) -> Dense Embedding
    #           sparse_query (关键词+别名) -> BM25
    from .rag.query_expander import QueryExpander
    from .rag.query_intent import research_penalty_for, detect_version_filters

    expander = QueryExpander()

    async def query_preprocessor(query: str) -> dict:
        """查询预处理管道：别名扩展 + 词权重。

        Returns:
            {"dense_query": str, "sparse_query": str, "sparse_weights": dict,
             "research_penalty": float, "version_boost": dict | None}
            sparse_weights: {token: weight}, 原查询词=1.0, 别名=0.5
            research_penalty: 研究文献降权系数，研究意图=1.0 否则=0.5
            version_boost: 版本软 boost 约束（{"genre": .., "dynasty": ..}），None 表示不 boost
        """
        research_penalty = research_penalty_for(query)
        version_boost = detect_version_filters(query)
        expand_result = expander.expand_weighted(query)
        return {
            "dense_query": query,
            "sparse_query": expand_result["expanded_query"],
            "sparse_weights": expand_result["token_weights"],
            "research_penalty": research_penalty,
            "version_boost": version_boost,
        }

    retriever = HybridRetriever(
        vector_store=vs,
        bm25_retriever=bm25,
        reranker=reranker,
        query_preprocessor=query_preprocessor,
    )

    # ---- 5b. 标签增强问答检索器（TagRetriever + 在线 LLM 打标，仅问答流程） ----
    # 语料与标签复用 rag_eval 静态资产（427 chunk 与 chunk_labels_full 对齐；
    # 生产 chunker 产出 chunk_id 与 corpus.jsonl 完全一致，已逐条校验）。
    from .rag.tag_store import TagStore, load_corpus
    from .rag.tag_retriever import TagRetriever
    from .rag.tag_pipeline import TagPipeline

    tag_store = TagStore(embedder=embedder)
    tag_store.load()
    tag_store.build_tag_vectors()
    tag_store.build_summary_vectors()
    tag_store.build_plot_unit_vectors()

    tag_retriever = TagRetriever(vs, embedder, tag_store)

    # 语料来源优先取向量库实际集合：BM25/标签增强必须与 dense 路 chunk_id 严格对齐，
    # corpus.jsonl 是评测基线快照，chunker 版本演进后块数会漂移（427→434），
    # 用旧文件补建 BM25 会导致新块 RRF 融合失效、旧块变成孤儿。
    # 集合为空（索引待重建）时回退 corpus.jsonl 兜底。
    live_docs = []
    try:
        live_docs = vs.get_all_documents()
    except Exception as e:
        logger.warning(f"读取向量库文档失败，BM25 回退 corpus.jsonl: {e}")
    if live_docs:
        tag_corpus = {
            d["chunk_id"]: {"text": d.get("content", ""), "metadata": d.get("metadata", {})}
            for d in live_docs
        }
        logger.info(f"[OK] 检索语料取自向量库实际集合: {len(tag_corpus)} chunks（与 dense 路对齐）")
    else:
        tag_corpus = load_corpus()
        logger.warning(f"向量库暂不可读/为空，BM25 回退 corpus.jsonl: {len(tag_corpus)} chunks")

    # CU-47: BM25 索引启动补建 —— 之前 bm25._built=True 只在全量重建向量索引时
    # （vector_store._sync_bm25）触发；正常启动（"index is healthy, no rebuild"）
    # 会跳过重建，BM25 恒为空壳 → 每次检索都打 "BM25 不可用，仅 Dense"，
    # sparse 路 + 查询词权重（别名扩展）全部失效。用 tag_corpus（与 chunk_id
    # 对齐的 427 chunk）启动时直接补建，约 1-3s。
    if rag_config.bm25_enabled and not bm25._built:
        bm25_docs = []
        for cid, doc in tag_corpus.items():
            meta = dict(doc.get("metadata", {}))
            meta.setdefault("chunk_id", cid)
            # CU-47b: chunk_id 必须放文档顶层 —— BM25Retriever.search() 从
            # self._docs[doc_id].get("chunk_id") 取顶层字段，只放 metadata 会导致
            # 融合阶段取不到 id 而生成临时 sparse_xxx id，RRF 无法与 Dense 路对齐
            bm25_docs.append({"content": doc.get("text", ""), "metadata": meta, "chunk_id": cid})
        bm25.index(bm25_docs)
    tag_meta_lookup = {cid: doc.get("metadata", {}) for cid, doc in tag_corpus.items()}
    tag_retriever.build_tag_bm25(tag_corpus)
    tag_retriever.build_tag_aug_dense(tag_corpus)
    tag_pipeline = TagPipeline(tag_retriever, llm, tag_corpus, tag_meta_lookup)
    logger.info("[OK] TagPipeline ready (标签增强 dense + 五路召回 + boost + LLM 打标)")

    # 注入到 routes（即使模型未就绪也先注入，/chat 会在调用时检查）
    # CU-45: RAGPipeline 扩展 bm25 和 reranker 属性
    class RAGPipeline:
        def __init__(self):
            self.embedder = embedder
            self.chunker = chunker
            self.vector_store = vs
            self.retriever = tag_pipeline
            self.bm25 = bm25
            self.reranker = reranker

    api_routes.rag_pipeline = RAGPipeline()
    api_routes.vector_store = vs

    # CU-46: 后台任务 —— 等模型就绪后自动构建索引
    # embedder 就绪等待逻辑已迁移到 Chunker._wait_for_embedder
    async def deferred_index_build():
        import asyncio
        import gc
        logger.info("[BUILD] Checking embedder readiness for index build...")
        watchdog = IndexWatchdog(rag_config.data_dir_path)
        if watchdog.needs_rebuild() or not vs.is_healthy():
            logger.info("[BUILD] Building vector + BM25 index (逐章增量入库)...")
            try:
                # 逐章分块 → 立即 upsert 落库 → gc；进程中途退出时已完成章节保留，
                # 重跑幂等覆盖（chunk_id 按章节顺序确定性生成）
                def _chapter_sink(chapter_number, chapter_title, dynasty, ch_chunks):
                    n = vs.upsert_chunks(ch_chunks)
                    chunker._embedding_cache.clear()
                    gc.collect()
                    logger.info(
                        f"[BUILD] 第{chapter_number}章入库 {n} 块，"
                        f"集合累计 {vs.collection_count()} 行"
                    )

                # chunker.chunk_all() 内部 _wait_for_embedder 阻塞等待
                chunks = await asyncio.to_thread(chunker.chunk_all, _chapter_sink)
                await asyncio.to_thread(vs._sync_bm25, chunks)
                # 重建后校验实际入库行数，防止"分块成功但入库失败/部分写入"假健康
                stored = vs.collection_count()
                if stored < max(1, len(chunks) // 2):
                    logger.error(
                        f"[ALERT] 索引重建异常：分块 {len(chunks)} 个但集合仅 {stored} 行，"
                        f"索引可能不完整，可用独立脚本 rebuild_index.py 重建"
                    )
                else:
                    logger.info(f"[OK] Index built: {len(chunks)} chunks (集合实际 {stored} 行)")
            except RuntimeError as e:
                logger.warning(f"[WARN] Index build failed (embedder timeout): {e}")
            except Exception as e:
                logger.warning(f"[WARN] Index build failed: {e}")
        else:
            logger.info("[OK] Vector index is healthy, no rebuild needed")

    asyncio.create_task(deferred_index_build())
    logger.info("[OK] RAG pipeline registered (index will build when model is ready)")

    # ---- 6. Conversation Memory（按用户延迟创建，不再使用共享 checkpointer）----
    from .langgraph.memory import ConversationMemory, close_all_user_memories, start_global_cleanup_scheduler
    api_routes.graph_registry = {"conversation_memory_type": "per_user_sqlite"}
    # 启动全局 TTL 清理调度器（每 6 小时清理所有用户过期会话）
    asyncio.create_task(start_global_cleanup_scheduler())
    logger.info("[OK] ConversationMemory: per-user mode (created lazily via get_user_memory)")

    # ---- 7. Prompt Registry ----
    from .prompts.registry import PromptRegistry
    from pathlib import Path
    prompt_dir = Path(__file__).parent / "prompts" / "templates"
    api_routes.prompt_registry = PromptRegistry(prompt_dir)
    logger.info(f"[OK] Prompt templates loaded: {len(api_routes.prompt_registry.templates)}")

    # ---- 8. Skills ----
    from .skills.loader import SkillLoader
    from .skills.registry import SkillRegistry
    skill_def_dir = Path(__file__).parent / "skills" / "definitions"
    loader = SkillLoader(skill_def_dir)
    skill_reg = SkillRegistry()
    for skill_def in loader.load_all():
        skill_reg.register(skill_def)
    api_routes.skill_registry = skill_reg
    logger.info(f"[OK] Skills: {len(skill_reg.list_all())}")

    # ---- 9. 可观测性 ----
    from .observability.metrics import Metrics
    from .observability.tracer import Tracer
    api_routes.metrics = Metrics()
    logger.info("[OK] Observability ready")

    # ---- 9. Amap 高德地图客户端（可选：配置 AMAP_API_KEY 后启用）----
    from .data.amap_client import AmapClient
    amap_client = AmapClient() if settings.amap_api_key else None
    if amap_client:
        logger.info("[OK] Amap POI Client ready（可选）")
    else:
        logger.info("[INFO] Amap API Key 未配置（非必须，web_search 已可用）")

    # ---- 9b. 加载景观数据 (v10 修复: generate_mutual_construction 跨端数据访问方案A) ----
    import json as _json
    landscape_info_path = Path(__file__).parent / "data" / "landscape_info.json"
    landscape_data = None
    if landscape_info_path.exists():
        with open(landscape_info_path, "r", encoding="utf-8") as _f:
            landscape_data = _json.load(_f)
        logger.info(f"[OK] Landscape data loaded: {len(landscape_data)} locations")
    else:
        logger.warning("[WARN] landscape_info.json not found, mutual_construction will use degraded mode")

    # ---- 10. 上下文管理与记忆系统 (v16, per-user) ----
    # ChromaDB client（复用 rag_config 的 persist_dir）
    import chromadb
    from chromadb.config import Settings as ChromaSettings
    # anonymized_telemetry=False: 关闭遥测，避免 posthog 不兼容刷错误日志
    chroma_client = chromadb.PersistentClient(
        path=str(rag_config.chroma_persist_path),
        settings=ChromaSettings(anonymized_telemetry=False),
    )

    from .context.wiring import build_memory_system
    from .context.memory_registry import set_memory_system_factory, get_user_memory_system
    data_dir = Path(__file__).parent.parent.parent / "data"
    memory_model = settings.memory_extraction_model or settings.effective_fast_model

    # per-user 惰性工厂：每个用户独立的 memory.db / context_state.db / ChromaDB 集合，
    # 物理隔离长期记忆与 Core Memory（隐私修复，防止 A 用户记忆被 B 用户看到）
    async def create_memory_system(user_id: str):
        user_dir = data_dir / "users" / user_id
        user_dir.mkdir(parents=True, exist_ok=True)
        ms = build_memory_system(
            llm=llm,
            embedder=embedder,
            chroma_client=chroma_client,
            memory_db_path=user_dir / "memory.db",
            context_db_path=user_dir / "context_state.db",
            fifo_max_tokens=settings.fifo_max_tokens,
            summary_model=memory_model,
            memory_extraction_model=memory_model,
            core_memory_enabled=settings.core_memory_enabled,
            background_tasks_enabled=settings.background_memory_tasks_enabled,
            user_id=user_id,
        )
        if ms.scheduler is not None:
            await ms.scheduler.start()
        return ms

    set_memory_system_factory(create_memory_system)
    api_routes.memory_system_factory = get_user_memory_system
    logger.info(f"[OK] Memory system factory ready (per-user, fifo_max_tokens={settings.fifo_max_tokens}, model={memory_model})")

    # ---- 11. MCP ----
    from .mcp.registry import MCPToolRegistry
    from .mcp.server import MCPServer
    from functools import partial

    mcp_reg = MCPToolRegistry()
    mcp_server = MCPServer()
    mcp_server.init_with_registry(mcp_reg)
    api_routes.mcp_registry = mcp_reg

    # 注册所有 MCP 工具
    from .mcp.tools.search_literature import TOOL_DEF as sl_tool, handler as sl_handler
    from .mcp.tools.get_location import TOOL_DEF as gl_tool, handler as gl_handler
    from .mcp.tools.get_keywords import TOOL_DEF as gk_tool, handler as gk_handler
    from .mcp.tools.compare_chapters import TOOL_DEF as cc_tool, handler as cc_handler
    from .mcp.tools.summarize_location import TOOL_DEF as sl2_tool, handler as sl2_handler
    from .mcp.tools.get_current_time import TOOL_DEF as ct_tool, handler as ct_handler
    from .mcp.tools.web_search import TOOL_DEF as ws_tool, handler as ws_handler
    from .mcp.tools.get_ticket_price import TOOL_DEF as tp_tool, handler as tp_handler

    mcp_reg.register(**sl_tool, handler=partial(sl_handler, node_client=node_client))
    mcp_reg.register(**gl_tool, handler=gl_handler)
    mcp_reg.register(**gk_tool, handler=gk_handler)
    mcp_reg.register(**cc_tool, handler=cc_handler)
    mcp_reg.register(**sl2_tool, handler=sl2_handler)
    mcp_reg.register(**ct_tool, handler=ct_handler)
    # ★ web_search + get_ticket_price（get_ticket_price 注入 web_search 作为 fallback）
    mcp_reg.register(**ws_tool, handler=ws_handler)
    mcp_reg.register(**tp_tool, handler=partial(tp_handler, web_search_handler=ws_handler))

    # ★ 高德地图 POI 工具（可选：配置 AMAP_API_KEY 后可用）
    from .mcp.tools.search_real_poi import TOOL_DEF as srp_tool, handler as srp_handler
    from .mcp.tools.get_attraction_ticket import TOOL_DEF as at_tool, handler as at_handler
    mcp_reg.register(**srp_tool, handler=partial(srp_handler, amap_client=amap_client))
    mcp_reg.register(**at_tool, handler=partial(at_handler, amap_client=amap_client))

    # ================================================================
    #  B-024: 注册 14 个新 MCP 工具 (v8)
    # ================================================================

    # ---- 通用工具 (B-010 ~ B-019) ----
    from .mcp.tools.search_paragraphs import TOOL_DEF as b010_tool, handler as b010_handler
    from .mcp.tools.get_chapter_full_text import TOOL_DEF as b011_tool, handler as b011_handler
    from .mcp.tools.get_paragraph_context import TOOL_DEF as b012_tool, handler as b012_handler
    from .mcp.tools.annotate_passage import TOOL_DEF as b013_tool, handler as b013_handler
    from .mcp.tools.annotate_user_request import TOOL_DEF as v901_tool, handler as v901_handler
    from .mcp.tools.suggest_annotations import TOOL_DEF as v902_tool, handler as v902_handler
    from .mcp.tools.compare_and_annotate import TOOL_DEF as v903_tool, handler as v903_handler
    from .mcp.tools.align_paragraphs import TOOL_DEF as v904_tool, handler as v904_handler
    from .mcp.tools.explain_passage import TOOL_DEF as b014_tool, handler as b014_handler
    from .mcp.tools.compare_passages import TOOL_DEF as b015_tool, handler as b015_handler
    from .mcp.tools.count_occurrences import TOOL_DEF as b016_tool, handler as b016_handler
    from .mcp.tools.keyword_trend import TOOL_DEF as b017_tool, handler as b017_handler
    from .mcp.tools.cooccurrence_matrix import TOOL_DEF as b018_tool, handler as b018_handler
    from .mcp.tools.sentiment_timeline import TOOL_DEF as b019_tool, handler as b019_handler

    # ---- v19 新增: 出场轨迹 + 共享段落定位 ----
    from .mcp.tools.entity_appearances import TOOL_DEF as ea2_tool, handler as ea2_handler
    from .mcp.tools.find_shared_passages import TOOL_DEF as fsp_tool, handler as fsp_handler

    # ---- v19 新增: 章节情节骨架 ----
    from .mcp.tools.chapter_summary import TOOL_DEF as cs_tool, handler as cs_handler

    # ---- 演化专用工具 (B-020 ~ B-023) ----
    from .mcp.tools.decompose_motifs import TOOL_DEF as b020_tool, handler as b020_handler
    from .mcp.tools.classify_change_type import TOOL_DEF as b021_tool, handler as b021_handler
    from .mcp.tools.infer_phylogeny import TOOL_DEF as b022_tool, handler as b022_handler
    from .mcp.tools.correlate_external import TOOL_DEF as b023_tool, handler as b023_handler

    # ---- v10 新增: 跨朝代对比 + 地图溯源工具 ----
    from .mcp.tools.dynasty_compare_elements import TOOL_DEF as dc_tool, handler as dc_handler
    from .mcp.tools.get_location_traceability import TOOL_DEF as glt_tool, handler as glt_handler
    from .mcp.tools.generate_mutual_construction import TOOL_DEF as gmc_tool, handler as gmc_handler
    from .mcp.tools.set_map_dynasty_layer import TOOL_DEF as smdl_tool, handler as smdl_handler
    from .mcp.tools.focus_map_location import TOOL_DEF as fmt_tool, handler as fmt_handler
    from .mcp.tools.export_annotations import TOOL_DEF as ea_tool, handler as ea_handler

    # 通用工具 — 注入 retriever, node_client, llm
    mcp_reg.register(**b010_tool, handler=partial(b010_handler, retriever=retriever))
    mcp_reg.register(**b011_tool, handler=partial(b011_handler, node_client=node_client, retriever=retriever))
    mcp_reg.register(**b012_tool, handler=partial(b012_handler, node_client=node_client))
    mcp_reg.register(**b013_tool, handler=partial(b013_handler, node_client=node_client, llm=llm))
    mcp_reg.register(**v901_tool, handler=partial(v901_handler, node_client=node_client, llm=llm))
    mcp_reg.register(**v902_tool, handler=partial(v902_handler, node_client=node_client, llm=llm))
    mcp_reg.register(**v903_tool, handler=partial(v903_handler, node_client=node_client, llm=llm))
    mcp_reg.register(**v904_tool, handler=partial(v904_handler, node_client=node_client, llm=llm))
    mcp_reg.register(**b014_tool, handler=partial(b014_handler, node_client=node_client, llm=llm))
    mcp_reg.register(**b015_tool, handler=partial(b015_handler, node_client=node_client, llm=llm))
    mcp_reg.register(**b016_tool, handler=partial(b016_handler, retriever=retriever, llm=llm))
    mcp_reg.register(**b017_tool, handler=partial(b017_handler, retriever=retriever))
    mcp_reg.register(**b018_tool, handler=partial(b018_handler, retriever=retriever))
    mcp_reg.register(**b019_tool, handler=partial(b019_handler, retriever=retriever, llm=llm))
    # v19: 出场轨迹（字面定位）+ 共享段落（跨章节文本传承）
    mcp_reg.register(**ea2_tool, handler=partial(ea2_handler, retriever=retriever))
    mcp_reg.register(**fsp_tool, handler=partial(fsp_handler, node_client=node_client))
    # v19: 章节情节骨架（LLM 生成 + 进程内缓存）
    mcp_reg.register(**cs_tool, handler=partial(cs_handler, node_client=node_client, llm=llm))

    # 演化工具 — 注入 retriever, llm, mcp_registry
    mcp_reg.register(**b020_tool, handler=partial(b020_handler, retriever=retriever, llm=llm))
    mcp_reg.register(**b021_tool, handler=partial(b021_handler, llm=llm))
    mcp_reg.register(**b022_tool, handler=partial(b022_handler, llm=llm))
    mcp_reg.register(**b023_tool, handler=partial(b023_handler, llm=llm, mcp_registry=mcp_reg))

    # ---- v10 新增: 跨朝代对比 + 地图溯源工具注册 ----
    mcp_reg.register(**dc_tool, handler=partial(dc_handler, node_client=node_client, llm=llm))
    mcp_reg.register(**glt_tool, handler=partial(glt_handler, node_client=node_client))
    mcp_reg.register(**gmc_tool, handler=partial(gmc_handler, node_client=node_client, llm=llm, landscape_data=landscape_data))
    mcp_reg.register(**smdl_tool, handler=smdl_handler)
    mcp_reg.register(**fmt_tool, handler=fmt_handler)
    mcp_reg.register(**ea_tool, handler=ea_handler)

    # ---- v12: 标注删除工具 ----
    from .mcp.tools.delete_annotations import TOOL_DEF as da_tool, handler as da_handler

    mcp_reg.register(**da_tool, handler=partial(da_handler, node_client=node_client))

    # ---- v17: 新增工具（诗词检索/多实体共现/对话提取/版本元数据） ----
    from .mcp.tools.search_poetry import TOOL_DEF as sp_tool, handler as sp_handler
    from .mcp.tools.multi_entity_cooccurrence import TOOL_DEF as mec_tool, handler as mec_handler
    from .mcp.tools.extract_dialogues import TOOL_DEF as ed_tool, handler as ed_handler
    from .mcp.tools.get_version_metadata import TOOL_DEF as gvm_tool, handler as gvm_handler

    mcp_reg.register(**sp_tool, handler=sp_handler)
    mcp_reg.register(**mec_tool, handler=partial(mec_handler, retriever=retriever))
    mcp_reg.register(**ed_tool, handler=ed_handler)
    mcp_reg.register(**gvm_tool, handler=gvm_handler)

    # ---- P0: 原有实现但未注册的 3 个标注质量工具补齐 (span_validator / coverage_tracker / annotation_knowledge_base) ----
    from .mcp.tools.span_validator import TOOL_DEF as sv_tool, handler as sv_handler
    from .mcp.tools.coverage_tracker import TOOL_DEF as cov_tool, handler as cov_handler
    from .mcp.tools.annotation_knowledge_base import TOOL_DEF as akb_tool, handler as akb_handler

    mcp_reg.register(**sv_tool, handler=partial(sv_handler, node_client=node_client))
    mcp_reg.register(**cov_tool, handler=partial(cov_handler, node_client=node_client))
    mcp_reg.register(**akb_tool, handler=akb_handler)  # 纯静态知识库，不需要外部依赖注入

    # ---- P1-1: 来源标签自动校验后处理工具 ----
    from .mcp.tools.verify_citation_tags import TOOL_DEF as vct_tool, handler as vct_handler
    mcp_reg.register(**vct_tool, handler=partial(vct_handler, node_client=node_client, retriever=retriever))

    # ---- P2: 研究广度 3 个新工具 ----
    from .mcp.tools.search_scholar_metadata import TOOL_DEF as ssm_tool, handler as ssm_handler
    from .mcp.tools.build_timeline import TOOL_DEF as bt_tool, handler as bt_handler
    from .mcp.tools.diff_passages import TOOL_DEF as dp_tool, handler as dp_handler
    mcp_reg.register(**ssm_tool, handler=ssm_handler)  # CrossRef API，无需内部依赖
    mcp_reg.register(**bt_tool, handler=partial(bt_handler, node_client=node_client))
    mcp_reg.register(**dp_tool, handler=partial(dp_handler, node_client=node_client))

    # ---- P3-11: 研究结论快照工具 ----
    from .mcp.tools.save_research_snapshot import TOOL_DEF as srs_tool, handler as srs_handler
    mcp_reg.register(**srs_tool, handler=partial(srs_handler, node_client=node_client))

    # ---- v19: 超长回答完整报告分段写入工具 ----
    from .mcp.tools.save_long_report import TOOL_DEF as slr_tool, handler as slr_handler
    mcp_reg.register(**slr_tool, handler=slr_handler)  # user_id 由 dispatcher 自动注入

    # ---- v16: 记忆相关 MCP 工具（按 user 路由，内部经工厂获取 per-user 实例） ----
    from .context.wiring import register_memory_tools
    register_memory_tools(mcp_reg, memory_system_factory=get_user_memory_system)

    logger.info(f"[OK] MCP Server ready: {len(mcp_reg.list_tools())} tools (v19: 标注质量/来源校验/学术检索/时间线/异文/快照/长报告)")

    # ---- 11. SupervisorGraph (v6 统一入口) ----
    # 注意: checkpointer 按请求通过 get_user_memory(user_id) 注入, 不在此处传入共享实例
    from .langgraph.supervisor import SupervisorGraph
    supervisor_graph = SupervisorGraph(
        llm=llm,
        retriever=tag_pipeline,
        prompt_registry=api_routes.prompt_registry,
        skill_registry=skill_reg,
        node_client=node_client,
        checkpointer=None,  # 每请求通过 _request_cp 注入 get_user_memory 的 checkpointer
        mcp_registry=mcp_reg,
    )
    api_routes.supervisor = supervisor_graph
    logger.info(f"[OK] SupervisorGraph ready (v6 unified router)")

    logger.info("=" * 60)
    logger.info(f"[READY] Agent service -> http://{settings.agent_host}:{settings.agent_port}")
    logger.info(f"  /health  -> health check")
    logger.info(f"  /chat    -> chat endpoint")
    logger.info("=" * 60)

    yield  # === 服务运行中 ===

    # ========== 关闭 ==========
    logger.info("Agent service shutting down...")
    from .context.memory_registry import close_all_memory_systems
    await close_all_memory_systems()
    await close_all_user_memories()
    await llm.close()
    await node_client.close()
    logger.info("Agent service stopped.")


# ---- FastAPI 应用 ----
app = FastAPI(
    title="White Snake AI Agent",
    description="White Snake Digital Museum - AI Agent (LangGraph + RAG + MCP + Skills)",
    version="6.0.0",
    lifespan=lifespan,
)

# CORS（开发阶段允许任意来源）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 请求日志
app.add_middleware(RequestLogMiddleware)

# 限流
try:
    app.add_middleware(RateLimitMiddleware)
    logger.info("[OK] Rate limiter enabled")
except Exception:
    pass

# 注册路由
app.include_router(router)


def main():
    """入口函数（供 uvicorn 调用）。"""
    import uvicorn
    uvicorn.run(
        "server.main:app",
        host=settings.agent_host,
        port=settings.agent_port,
        # 注意：绝不能开 reload。reloader 监视整个 agent 目录，
        # 运行时日志文件持续写入会反复触发服务自动重启（初始化卡死、请求 502 的元凶）。
        reload=False,
        log_level="info",
    )


if __name__ == "__main__":
    main()
