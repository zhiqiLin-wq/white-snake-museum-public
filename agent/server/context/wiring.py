"""记忆系统接线工厂 (v16)。

把 main.py lifespan 里记忆系统的组装逻辑抽成独立、可测试的函数。
main.py 与测试共用同一套组装代码，避免"测试正确、生产漏接"的再次发生。

职责:
1. build_memory_system() — 构造并接线全部组件
   MemoryDB -> RecallStore -> HybridRetriever -> PersonaLoader
   -> BackgroundTaskScheduler -> ContextManager
2. register_memory_tools() — 注册 4 个记忆相关 MCP 工具并注入依赖
   retrieve_memory / save_to_memory / core_memory_append / core_memory_replace

依赖注入约定:
- 所有外部重依赖 (llm / embedder / chroma_client) 作为显式参数传入，
  main.py 传真实实例，测试传 mock 实例。
- 路径 (memory_db_path / context_db_path) 显式传入，测试用临时目录。

与本项目另一套 LangGraph ConversationMemory 的边界:
- ConversationMemory (langgraph/memory.py) = LangGraph 图执行状态持久化
  (per-user agent_checkpoints.db，存 checkpoint BLOB)，是框架执行所必需。
- 本模块的 context 系统 = LLM 对话窗口管理 (FIFO + 递归摘要 + 长期记忆 +
  Core Memory + 被动注入)，服务于"给 LLM 喂什么上下文"。
两者职责不同、不冲突，只是对话原文会分别在 checkpoint 与 recall/messages 中各存一份，
这是刻意设计（checkpoint 供图状态恢复，messages/recall 供语义检索）。
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class MemorySystem:
    """记忆系统全部组件的容器。

    接线自检直接断言每个字段非 None 即可发现漏接问题。
    """
    memory_db: object
    recall_store: object
    hybrid_retriever: object
    persona_loader: object
    context_manager: object
    scheduler: object


def build_memory_system(
    llm,
    embedder,
    chroma_client,
    *,
    memory_db_path: Path,
    context_db_path: Path,
    persona_config_path: Optional[Path] = None,
    fifo_max_tokens: int = 90000,
    summary_model: str = "",
    memory_extraction_model: str = "",
    core_memory_enabled: bool = True,
    background_tasks_enabled: bool = True,
    user_id: str = "",
    bm25_retriever=None,
    reranker=None,
) -> MemorySystem:
    """构造并接线记忆系统全部组件。

    Args:
        llm: LLMProvider 实例
        embedder: Embedder 实例
        chroma_client: chromadb.PersistentClient 实例
        memory_db_path: memory.db 完整路径
        context_db_path: context_state.db 完整路径
        persona_config_path: persona.yaml 路径（None 用默认）
        fifo_max_tokens: FIFO 队列 token 上限
        summary_model: 摘要更新模型
        memory_extraction_model: 记忆提取模型
        core_memory_enabled: 是否启用 Core Memory (persona/human 块)
        background_tasks_enabled: 是否启用后台记忆维护任务

    Returns:
        MemorySystem 容器（所有组件已接线）
    """
    from .memory_db import MemoryDB
    from .recall_store import RecallStore
    from .hybrid_retriever import HybridRetriever
    from .context_manager import ContextManager
    from .background_tasks import BackgroundTaskScheduler
    from ..core_memory.persona_loader import PersonaLoader

    # 1. MemoryDB — SQLite Source of Truth
    memory_db = MemoryDB(memory_db_path)
    logger.info("[wiring] MemoryDB ready: %s", memory_db_path)

    # 2. RecallStore — 双写 (SQLite + ChromaDB)
    recall_store = RecallStore(
        chroma_client=chroma_client,
        embedder=embedder,
        memory_db=memory_db,
        user_id=user_id,
    )
    logger.info("[wiring] RecallStore ready")

    # 3. HybridRetriever — FTS5 + 向量两阶段检索
    hybrid_retriever = HybridRetriever(
        memory_db=memory_db,
        embedder=embedder,
        chroma_client=chroma_client,
        user_id=user_id,
        bm25_retriever=bm25_retriever,
        reranker=reranker,
    )
    logger.info("[wiring] HybridRetriever ready")

    # 4. PersonaLoader — Core Memory 模板 + 不可变区保护
    persona_loader = None
    if core_memory_enabled:
        persona_loader = PersonaLoader(config_path=persona_config_path)
        # 从 persona.yaml 模板初始化 persona/human 块（若尚未初始化）
        _init_core_memory_blocks(memory_db, persona_loader)
        logger.info("[wiring] PersonaLoader ready + core memory blocks initialized")

    # 5. BackgroundTaskScheduler — 衰减/合并/embedding 补偿
    scheduler = None
    if background_tasks_enabled:
        scheduler = BackgroundTaskScheduler(
            memory_db=memory_db,
            recall_store=recall_store,
            embedder=embedder,
            persona_loader=persona_loader,
        )
        logger.info("[wiring] BackgroundTaskScheduler ready")

    # 6. ContextManager — FIFO 队列 + 摘要 + 被动注入 + Core Memory 注入
    context_manager = ContextManager(
        db_path=context_db_path,
        llm=llm,
        recall_store=recall_store,
        max_tokens=fifo_max_tokens,
        summary_model=summary_model,
        memory_extraction_model=memory_extraction_model,
        memory_db=memory_db,
        hybrid_retriever=hybrid_retriever,
        background_scheduler=scheduler,
        persona_loader=persona_loader,
    )
    logger.info(
        "[wiring] ContextManager ready (fifo_max_tokens=%d, model=%s)",
        fifo_max_tokens, summary_model,
    )

    # 回填 scheduler 的 idle 提取依赖（循环依赖：scheduler 先建、ContextManager 后建）
    if scheduler is not None:
        scheduler.attach_context_manager(context_manager)
        logger.info("[wiring] BackgroundTaskScheduler attached to ContextManager")

    return MemorySystem(
        memory_db=memory_db,
        recall_store=recall_store,
        hybrid_retriever=hybrid_retriever,
        persona_loader=persona_loader,
        context_manager=context_manager,
        scheduler=scheduler,
    )


def _init_core_memory_blocks(memory_db, persona_loader) -> None:
    """用 persona.yaml 模板初始化 persona/human 块（幂等，仅首次插入）。"""
    try:
        persona_json = json.dumps(
            persona_loader.persona_template, ensure_ascii=False, indent=2,
        )
        human_json = json.dumps(
            persona_loader.human_template, ensure_ascii=False, indent=2,
        )
        memory_db.init_core_memory_block("persona", persona_json)
        memory_db.init_core_memory_block("human", human_json)
    except Exception as e:
        logger.warning("[wiring] core memory block init failed: %s", e)


def register_memory_tools(mcp_registry, memory_system_factory) -> None:
    """注册 4 个记忆相关 MCP 工具（按 user 路由）。

    Args:
        mcp_registry: MCPToolRegistry 实例
        memory_system_factory: per-user 工厂，async def factory(user_id) -> MemorySystem
    """
    from ..mcp.tools import retrieve_memory as rm_mod
    from ..mcp.tools import save_to_memory as stm_mod
    from ..mcp.tools import core_memory_append as cma_mod
    from ..mcp.tools import core_memory_replace as cmr_mod

    rm_mod.set_memory_system_factory(memory_system_factory)
    mcp_registry.register(**rm_mod.TOOL_DEF, handler=rm_mod.handler)
    logger.info("[wiring] MCP tool registered: retrieve_memory (per-user factory)")

    stm_mod.set_memory_system_factory(memory_system_factory)
    mcp_registry.register(**stm_mod.TOOL_DEF, handler=stm_mod.handler)
    logger.info("[wiring] MCP tool registered: save_to_memory (per-user factory)")

    cma_mod.set_memory_system_factory(memory_system_factory)
    cmr_mod.set_memory_system_factory(memory_system_factory)
    mcp_registry.register(**cma_mod.TOOL_DEF, handler=cma_mod.handler)
    mcp_registry.register(**cmr_mod.TOOL_DEF, handler=cmr_mod.handler)
    logger.info("[wiring] MCP tools registered: core_memory_append + core_memory_replace (per-user factory)")
