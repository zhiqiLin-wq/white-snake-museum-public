"""save_to_memory MCP 工具: 统一记忆保存入口。

在上下文压力（70%）时，LLM 可调用此工具将关键信息存入记忆。
支持两个目标:
- core: 保存到 Core Memory（persona/human 块），用于用户画像更新
- archival: 保存到 Archival Memory（long_term_memories），用于长期知识存储

LLM 自行决定哪种更合适。
"""

import logging

from ...llm.json_utils import parse_llm_json

logger = logging.getLogger(__name__)

TOOL_NAME = "save_to_memory"
TOOL_DESCRIPTION = (
    "将重要信息保存到长期记忆，避免因上下文压缩而丢失。"
    "当对话接近上下文上限、发现需要保留的事实时调用。"
    "target='core' 保存用户画像更新（偏好/身份/约束），"
    "target='archival' 保存事实性知识。"
)
TOOL_INPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "target": {
            "type": "string",
            "enum": ["core", "archival"],
            "description": "Where to save: 'core' for Core Memory (persona/human blocks), "
                           "'archival' for Archival Memory (long-term fact storage)."
        },
        "content": {
            "type": "string",
            "description": "The information to save. For 'core' target, this should be "
                           "a JSON object with field-value pairs. For 'archival', a plain "
                           "text statement in third-person."
        },
        "importance": {
            "type": "string",
            "enum": ["high", "medium", "low"],
            "description": "How important this information is. Defaults to 'medium'."
        },
    },
    "required": ["target", "content"],
}

_memory_system_factory = None

IMPORTANCE_MAP = {"high": 0.9, "medium": 0.6, "low": 0.3}


def set_memory_system_factory(factory):
    """注入 per-user MemorySystem 工厂（async def factory(user_id) -> MemorySystem）。"""
    global _memory_system_factory
    _memory_system_factory = factory


async def handler(target: str, content: str, importance: str = "medium", user_id: str = ""):
    """处理 save_to_memory 工具调用。"""
    if _memory_system_factory is None:
        return {"error": "Memory system not initialized."}
    if not user_id:
        return {"error": "user_id is required for memory access."}

    ms = await _memory_system_factory(user_id)
    importance_value = IMPORTANCE_MAP.get(importance, 0.6)

    if target == "core":
        return await _save_to_core(content, ms)
    elif target == "archival":
        return await _save_to_archival(content, importance_value, ms)
    else:
        return {"error": f"unknown target '{target}'."}


async def _save_to_core(content: str, ms) -> dict:
    """保存到 Core Memory（追加到 human 块）。"""
    import json
    memory_db = ms.memory_db
    persona_loader = ms.persona_loader
    try:
        data = parse_llm_json(content)
    except (json.JSONDecodeError, TypeError):
        data = {"_notes": [content]}

    # 追加到 human 块
    current = memory_db.get_core_memory_block("human")
    current_content = current["content"] if current else "{}"

    # 合并各个字段
    try:
        current_obj = json.loads(current_content)
    except (json.JSONDecodeError, TypeError):
        current_obj = {}

    updated = False
    for key, value in data.items():
        if key in ("personal_info", "research_interests", "preferences",
                    "constraints", "interaction_style"):
            if key not in current_obj:
                current_obj[key] = {}
            if isinstance(value, dict) and isinstance(current_obj[key], dict):
                current_obj[key].update(value)
            else:
                current_obj[key] = value
            updated = True

    if not updated:
        if "_notes" not in current_obj:
            current_obj["_notes"] = []
        current_obj["_notes"].append(content)

    merged = json.dumps(current_obj, ensure_ascii=False, indent=2)
    token_count = persona_loader.estimate_tokens(merged) if persona_loader else len(merged) // 2

    if token_count > 2000:
        return {
            "error": (
                f"human block would be ~{token_count} tokens (limit: 2000), save rejected. "
                f"Consider consolidating with core_memory_replace before saving more."
            ),
        }

    memory_db.set_core_memory_block("human", merged, token_count)
    logger.info("save_to_memory: saved to core (human), %d tokens", token_count)

    return {
        "result": f"Saved to Core Memory (human block). Tokens: {token_count}/2000.",
        "token_count": token_count,
    }


async def _save_to_archival(content: str, importance_value: float, ms) -> dict:
    """保存到 Archival Memory（长期记忆）。"""
    memory_db = ms.memory_db
    recall_store = ms.recall_store
    if recall_store is None:
        return {"error": "RecallStore not initialized."}

    # 判断 memory_type
    entity_keywords = ["喜欢", "偏好", "我是", "我叫", "我的", "用户", "研究", "方向"]
    is_entity = any(kw in content for kw in entity_keywords)
    memory_type = "Entity" if is_entity else "Archival"

    doc_id = recall_store.add_memory(
        content,
        memory_type=memory_type,
        importance=importance_value,
        confidence=0.6,
    )

    if doc_id:
        memory_db.record_changelog(
            doc_id, "ADD",
            new_value=content,
            reason=f"save_to_memory (importance={importance_value})",
            changed_by="llm",
        )
        return {
            "result": f"Saved to Archival Memory as '{memory_type}' (id: {doc_id}).",
            "memory_id": doc_id,
        }
    else:
        return {"result": "Memory was not saved (duplicate or empty)."}


TOOL_DEF = {
    "name": TOOL_NAME,
    "description": TOOL_DESCRIPTION,
    "inputSchema": TOOL_INPUT_SCHEMA,
}
