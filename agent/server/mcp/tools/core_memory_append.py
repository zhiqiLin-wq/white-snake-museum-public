"""core_memory_append MCP 工具: 向 Core Memory 块的可变区字段追加内容。

用于记录零散的新事实，如"用户今天提到他喜欢喝拿铁"。
不可变区字段拒绝追加。底层自动保存旧版本快照。

P3-10: 写入前做冲突检测——如果新内容与同字段已有内容矛盾
（同一 key 不同 value），返回 conflictWarning 让 LLM/用户知晓。
"""

import json
import logging

logger = logging.getLogger(__name__)

TOOL_NAME = "core_memory_append"
TOOL_DESCRIPTION = (
    "向 Core Memory 块的指定可变字段追加信息——适合增量记录单条新事实或偏好。"
    "不可变字段（core_identity、knowledge_boundary）拒绝追加；"
    "每次修改前自动保存版本快照。"
)
TOOL_INPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "block": {
            "type": "string",
            "enum": ["persona", "human"],
            "description": "Which Core Memory block to append to."
        },
        "field": {
            "type": "string",
            "description": "Target field name in the block. "
                           "For persona: communication_style, user_preferences, interaction_patterns. "
                           "For human: personal_info, research_interests, preferences, constraints, interaction_style."
        },
        "content": {
            "type": "string",
            "description": "Content to append. Can be a JSON key-value pair like "
                           "'{\"favorite_drink\": \"latte\"}' or plain text."
        },
    },
    "required": ["block", "field", "content"],
}

_memory_system_factory = None


def set_memory_system_factory(factory):
    """注入 per-user MemorySystem 工厂（async def factory(user_id) -> MemorySystem）。"""
    global _memory_system_factory
    _memory_system_factory = factory


async def handler(block: str, field: str, content: str, user_id: str = ""):
    """处理 core_memory_append 工具调用。"""
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
        return {"result": f"Error: invalid block '{block}'."}

    current = memory_db.get_core_memory_block(block)
    current_content = current["content"] if current else "{}"

    try:
        merged = persona_loader.append(block, field, content, current_content)
    except ValueError as e:
        return {"result": str(e)}
    except Exception as e:
        logger.error("core_memory_append merge failed: %s", e)
        return {"result": f"Error appending: {e}"}

    token_count = persona_loader.estimate_tokens(merged)
    if token_count > 2000:
        return {
            "result": (
                f"Warning: the resulting {block} block would be ~{token_count} tokens "
                f"(limit: 2000). Consider consolidating before appending more."
            ),
            "version": current["version"] if current else 0,
            "token_count": token_count,
        }

    # 用 persona_loader 合并后的结果一次性写入，
    # 避免 double-write 导致版本号 +2、快照重复 (原 append_core_memory_block + set_core_memory_block 各写一次)
    new_version = memory_db.set_core_memory_block(
        block, merged, token_count, reason="core_memory_append",
    )

    logger.info("core_memory_append: %s.%s v%d", block, field, new_version)

    # P3-10: 冲突检测——检查新内容是否与同字段已有内容矛盾
    conflict_warnings = _detect_conflicts(field, content, current_content)

    result = {
        "result": f"Appended to '{field}' in '{block}' block (version {new_version}).",
        "version": new_version,
        "token_count": token_count,
    }
    if conflict_warnings:
        result["conflictWarnings"] = conflict_warnings
        result["result"] += (
            f" ⚠️ 检测到 {len(conflict_warnings)} 处潜在冲突："
            + "; ".join(w["message"] for w in conflict_warnings[:3])
        )
        logger.info(
            "core_memory_append P3-10: %d conflicts in %s.%s: %s",
            len(conflict_warnings), block, field,
            [w["key"] for w in conflict_warnings],
        )

    return result


def _detect_conflicts(field: str, new_content: str, current_content: str) -> list[dict]:
    """检测新追加的内容是否与已有内容冲突（轻量级，不调 LLM）。

    策略:
    1. 如果 content 是 JSON key-value 对，检查同 key 不同 value
    2. 如果是纯文本，跳过（纯文本冲突检测需要 NLP，此处不展开）
    """
    conflicts: list[dict] = []
    if not current_content or not new_content:
        return conflicts

    try:
        current_data = json.loads(current_content) if isinstance(current_content, str) else current_content
    except (json.JSONDecodeError, TypeError):
        return conflicts  # 当前内容不是 JSON，无法做结构化冲突检测

    try:
        new_data = json.loads(new_content) if isinstance(new_content, str) else new_content
    except (json.JSONDecodeError, TypeError):
        return conflicts  # 新内容不是 JSON，跳过

    if not isinstance(new_data, dict) or not isinstance(current_data, dict):
        return conflicts

    # 取同字段的已有数据
    field_data = current_data.get(field, {})
    if not isinstance(field_data, dict):
        # 可能是 list 或其他类型
        return conflicts

    for key, new_val in new_data.items():
        if key in field_data:
            old_val = field_data[key]
            # 值不同 → 潜在冲突
            if str(old_val).strip() != str(new_val).strip():
                conflicts.append({
                    "key": key,
                    "oldValue": old_val,
                    "newValue": new_val,
                    "message": f"字段 '{key}' 已有值 '{old_val}'，新值 '{new_val}' 与之不同。可能是用户观点更新，也可能是记录错误。",
                })

    return conflicts


TOOL_DEF = {
    "name": TOOL_NAME,
    "description": TOOL_DESCRIPTION,
    "inputSchema": TOOL_INPUT_SCHEMA,
}
