"""retrieve_memory MCP 工具 (v2): LLM 按需检索长期记忆和 Recall Storage。

v2: 内部升级为 FTS5 + 向量两阶段混合检索管道。
LLM 可在 ReAct 循环中主动调用此工具，检索之前被压缩或遗忘的信息。
"""

import logging

logger = logging.getLogger(__name__)

TOOL_NAME = "retrieve_memory"
TOOL_DESCRIPTION = (
    "检索 Agent 的长期记忆与历史对话记录，找回可能被遗忘或因上下文压缩而不可见的信息。"
    "当怀疑对话前段存在相关但当前看不到的内容时调用。"
)
TOOL_INPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "query": {
            "type": "string",
            "description": "The search query to find relevant memories or past messages."
        },
        "source": {
            "type": "string",
            "enum": ["all", "memories", "recall"],
            "description": "Which source to search: 'all' (default), 'memories' (long-term only), "
                           "or 'recall' (historical messages only)."
        },
    },
    "required": ["query"],
}

# 模块级引用（per-user 工厂）
_memory_system_factory = None


def set_memory_system_factory(factory):
    """注入 per-user MemorySystem 工厂（async def factory(user_id) -> MemorySystem）。"""
    global _memory_system_factory
    _memory_system_factory = factory


async def handler(query: str, source: str = "all", user_id: str = ""):
    """处理 retrieve_memory 工具调用。

    v2: 优先使用 HybridRetriever（FTS5+向量两阶段），
    降级到 RecallStore 纯向量检索。

    Args:
        query: 搜索查询
        source: 检索来源 ("all" | "memories" | "recall")
        user_id: 当前用户 ID（由 AgentLoop 注入，用于 per-user 记忆隔离）

    Returns:
        {"result": "自然语言格式的检索结果"}
    """
    if not query.strip():
        return {"result": "Error: query is required."}
    if _memory_system_factory is None:
        return {"result": "Error: Memory system not initialized."}
    if not user_id:
        return {"result": "Error: user_id is required for memory access."}

    ms = await _memory_system_factory(user_id)
    hybrid_retriever = ms.hybrid_retriever
    recall_store = ms.recall_store

    # 优先使用 HybridRetriever
    if hybrid_retriever is not None:
        try:
            result = hybrid_retriever.search(
                query,
                source=source,
                score_threshold=0.5,
                token_budget=12000,
            )
            if result.get("below_threshold", True):
                logger.info(
                    "retrieve_memory query='%s' -> no results above threshold",
                    query[:60],
                )
                return {
                    "result": (
                        "[内部参考] 未找到相关度足够高的记忆。\n"
                        "你可以基于当前对话内容和自己的知识直接回答。"
                    ),
                }

            formatted = hybrid_retriever.format_for_mcp_tool(result)
            logger.info(
                "retrieve_memory (hybrid) query='%s' -> %d memories, %d recalls",
                query[:60],
                len(result.get("memories", [])),
                len(result.get("recalls", [])),
            )
            return {"result": formatted}
        except Exception as e:
            logger.warning("retrieve_memory hybrid retriever failed: %s, falling back", e)

    # Fallback: 纯向量检索
    if recall_store is not None:
        memories = []
        recalls = []
        if source in ("all", "memories"):
            memories = recall_store.search_memories(query, top_k=5)
        if source in ("all", "recall"):
            recalls = recall_store.search_recall_messages(query, top_k=3)

        parts = []
        parts.append("[内部参考，不要直接对用户说'根据检索到的记忆']\n")

        if memories:
            parts.append("=== 长期记忆 ===")
            for i, m in enumerate(memories, 1):
                score = m.get("score", 0)
                parts.append(f"{i}. [相似度 {score:.2f}] {m['content']}")
        else:
            parts.append("=== 长期记忆 ===\n(无相关记忆)")

        parts.append("")
        if recalls:
            parts.append("=== 历史消息 ===")
            for i, r in enumerate(recalls, 1):
                meta = r.get("metadata", {})
                role = meta.get("role", "?")
                ts = meta.get("timestamp", "")[:19]
                score = r.get("score", 0)
                parts.append(f"{i}. [{role}] ({ts}) [相似度 {score:.2f}] {r['content'][:300]}")
        else:
            parts.append("=== 历史消息 ===\n(无相关历史消息)")

        result_text = "\n".join(parts)
        logger.info(
            "retrieve_memory (fallback) query='%s' -> %d memories, %d recalls",
            query[:60], len(memories), len(recalls),
        )
        return {"result": result_text}

    return {"result": "Error: Memory system not initialized."}


TOOL_DEF = {
    "name": TOOL_NAME,
    "description": TOOL_DESCRIPTION,
    "inputSchema": TOOL_INPUT_SCHEMA,
}
