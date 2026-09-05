"""core_memory_replace MCP 工具: 替换 Core Memory 块的整个可变区内容。

Agent 可在 ReAct 循环中调用此工具更新自己的 persona 或 human 块。
不可变区字段（core_identity, knowledge_boundary）受保护，不会被覆盖。
底层自动保存旧版本快照到 core_memory_snapshots 表。
"""

import logging

logger = logging.getLogger(__name__)

TOOL_NAME = "core_memory_replace"
TOOL_DESCRIPTION = (
    "整体替换 Core Memory 块（persona 或 human）的可变部分——用于更新用户偏好、"
    "沟通风格，或修正/深化对用户的理解。"
    "不可变字段（core_identity、knowledge_boundary）受自动保护；"
    "每次修改前自动保存版本快照，支持回滚。"
)
TOOL_INPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "block": {
            "type": "string",
            "enum": ["persona", "human"],
            "description": "Which Core Memory block to update."
        },
        "content": {
            "type": "string",
            "description": "JSON string with the new content for mutable fields. "
                           "For persona: communication_style, user_preferences, interaction_patterns. "
                           "For human: personal_info, research_interests, interaction_style, preferences, constraints."
        },
    },
    "required": ["block", "content"],
}

# 模块级引用
_memory_system_factory = None


def set_memory_system_factory(factory):
    """注入 per-user MemorySystem 工厂（async def factory(user_id) -> MemorySystem）。"""
    global _memory_system_factory
    _memory_system_factory = factory


async def handler(block: str, content: str, user_id: str = ""):
    """处理 core_memory_replace 工具调用。

    Args:
        block: "persona" | "human"
        content: JSON 字符串
        user_id: 当前用户 ID（由 AgentLoop 注入，用于 per-user 记忆隔离）

    Returns:
        {"result": "...", "version": int}
    """
    if _memory_system_factory is None:
        return {"result": "Error: Core Memory system not initialized."}
    if not user_id:
        return {"result": "Error: user_id is required for memory access."}

    ms = await _memory_system_factory(user_id)
    memory_db = ms.memory_db
    persona_loader = ms.persona_loader
    if memory_db is None or persona_loader is None:
        return {"result": "Error: Core Memory system not initialized."}

    if block not in ("persona", "human"):
        return {"result": f"Error: invalid block '{block}'. Must be 'persona' or 'human'."}

    # 读取当前内容
    current = memory_db.get_core_memory_block(block)
    current_content = current["content"] if current else "{}"

    # 合并（保护不可变区）
    try:
        merged = persona_loader.replace(block, content, current_content)
    except Exception as e:
        logger.error("core_memory_replace merge failed: %s", e)
        return {"result": f"Error merging content: {e}"}

    # 估算 token 数
    token_count = persona_loader.estimate_tokens(merged)
    if token_count > 2000:
        return {
            "result": (
                f"Warning: the resulting {block} block would be ~{token_count} tokens "
                f"(limit: 2000). Please simplify the content before replacing, "
                f"or use core_memory_append for incremental additions."
            ),
            "version": current["version"] if current else 0,
            "token_count": token_count,
        }

    # 写入（自动保存快照 + 递增版本号）
    new_version = memory_db.set_core_memory_block(block, merged, token_count)

    logger.info("core_memory_replace: %s v%d (%d tokens)", block, new_version, token_count)

    return {
        "result": f"Core memory block '{block}' updated to version {new_version}.",
        "version": new_version,
        "token_count": token_count,
    }


TOOL_DEF = {
    "name": TOOL_NAME,
    "description": TOOL_DESCRIPTION,
    "inputSchema": TOOL_INPUT_SCHEMA,
}
