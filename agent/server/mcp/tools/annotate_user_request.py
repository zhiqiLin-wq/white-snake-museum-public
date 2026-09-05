"""v9: annotate_user_request — 解析用户自然语言标注指令，产生标注。

接收用户的标注指令（如"把人物用蓝色标出来"），由 LLM 解析自然语言
意图后调用 annotate_passage 来完成实际标注。
"""
import json
import logging
import re

from ...llm.json_utils import parse_llm_json

logger = logging.getLogger(__name__)

TOOL_DEF = {
    "name": "annotate_user_request",
    "description": (
        "解析用户的自然语言标注指令，识别用户想要标注的内容、类别和颜色偏好，"
        "然后对指定段落执行实体标注。"
    ),
    "inputSchema": {
        "type": "object",
        "properties": {
            "chapter_number": {
                "type": "integer",
                "description": "章节编号",
            },
            "paragraph_indices": {
                "type": "array",
                "items": {"type": "integer"},
                "description": "要标注的段落索引列表。不传则标注全部段落",
            },
            "user_instruction": {
                "type": "string",
                "description": "用户的自然语言标注指令（如'把人物标成蓝色'）",
            },
            "existing_annotations": {
                "type": "array",
                "items": {"type": "string"},
                "description": "已有标注ID列表，避免重复标注",
            },
            "marginalia_notes": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "paragraph_index": {
                            "type": "integer",
                            "description": "段落索引（0-based，与前端 passageKey 的 paragraphIndex 一致）。用户说的'第N段'对应 N-1",
                        },
                        "content": {
                            "type": "string",
                            "description": "批注文本内容",
                        },
                    },
                    "required": ["paragraph_index", "content"],
                },
                "description": (
                    "批注列表。每个元素包含 paragraph_index (0-based) 和批注文本 content。"
                    "当用户明确要求写批注/旁批/评注时填充此字段。"
                    "如果用户同时要求实体标注和批注，则两者都传入。"
                ),
            },
        },
        "required": ["chapter_number", "user_instruction"],
    },
}


def _sample_text(full_text: str, max_chars: int = 1000) -> str:
    """CU-7: 从全文首尾和中间均匀抽样，确保 categories 推断覆盖全文特征。

    即使实体集中在中后段，LLM 也能感知。
    """
    text_len = len(full_text)
    if text_len <= max_chars:
        return full_text

    head_size = min(350, text_len // 3)
    tail_size = min(350, text_len // 3)
    mid_size = max_chars - head_size - tail_size

    head = full_text[:head_size]
    tail = full_text[-tail_size:]

    mid_start = max(0, text_len // 3)
    mid = full_text[mid_start:mid_start + mid_size]

    return f"{head}\n...\n{mid}\n...\n{tail}"


async def handler(chapter_number: int,
                  user_instruction: str,
                  paragraph_indices: list = None,
                  existing_annotations: list = None,
                  marginalia_notes: list = None,
                  node_client=None, llm=None) -> dict:
    """解析用户标注指令并执行标注。支持实体标注和旁批评注。

    paragraph_indices 不传时默认标注全部段落。
    marginalia_notes 非空时产出旁批评注数据，与实体标注可同时存在。
    """
    if node_client is None:
        return {"error": "Node API 客户端未初始化"}
    if llm is None:
        return {"error": "LLM 未初始化"}

    ch = await node_client.get_literature_by_chapter(chapter_number)
    if ch is None:
        return {"error": f"未找到第 {chapter_number} 章"}

    content = ch.get("content", "")
    # Normalize line endings to match frontend splitParagraphs behavior
    content = re.sub(r'\r\n|\r', '\n', content)
    paragraphs = [p.strip() for p in content.split('\n\n') if p.strip()]

    # 未指定段落索引时，默认标注全部段落
    if paragraph_indices is None or len(paragraph_indices) == 0:
        paragraph_indices = list(range(len(paragraphs)))

    target_paragraphs = []
    for idx in paragraph_indices:
        if 0 <= idx < len(paragraphs):
            target_paragraphs.append(paragraphs[idx])
    if not target_paragraphs:
        return {"error": "指定的段落范围为空"}

    full_text = "\n".join(target_paragraphs)

    # C-05: 始终构建 target 信息，供 _format_annotation_data 使用
    base_target = {
        "chapter_number": chapter_number,
        "chapter_title": ch.get("title", ""),
        "paragraph_range": [min(paragraph_indices), max(paragraph_indices)],
    }

    # C-05: 构建 marginalia 列表（fast fail: 验证输入有效性）
    result_marginalia = []
    if marginalia_notes:
        if not isinstance(marginalia_notes, list):
            return {"error": "marginalia_notes 必须是数组"}
        for i, m in enumerate(marginalia_notes):
            if not isinstance(m, dict):
                return {"error": f"marginalia_notes[{i}] 必须是对象"}
            if "paragraph_index" not in m or "content" not in m:
                return {"error": f"marginalia_notes[{i}] 缺少必填字段 paragraph_index 或 content"}
            p_idx = m["paragraph_index"]
            if not isinstance(p_idx, int) or p_idx < 0 or p_idx >= len(paragraphs):
                return {"error": f"marginalia_notes[{i}].paragraph_index={p_idx} 超出段落范围 [0, {len(paragraphs)-1}]"}
            if not isinstance(m["content"], str) or not m["content"].strip():
                return {"error": f"marginalia_notes[{i}].content 不能为空"}
            result_marginalia.append({
                "chapter_number": chapter_number,
                "paragraph_index": p_idx,
                "content": m["content"].strip(),
            })

    # C-16: 判断 user_instruction 是否包含实体标注意图（关键词检查，无 LLM 调用）
    ANNOTATION_KEYWORDS = [
        "人物", "地点", "事件", "术语", "母题", "标注", "标出", "标一下",
        "划线", "高亮", "标记", "标成", "标为", "标蓝", "标红", "标绿",
        "person", "location", "event", "term", "motif",
    ]
    has_annotation_intent = any(kw in user_instruction for kw in ANNOTATION_KEYWORDS)

    # 场景 1: 纯批注（无实体标注意图）→ 跳过 LLM 调用和 annotate_passage
    if result_marginalia and not has_annotation_intent:
        return {
            "target": base_target,
            "annotations": [],
            "insights": [],
            "marginalia": result_marginalia,
            "user_instruction": user_instruction,
            "parsed_categories": [],
            "parsed_colors": {},
        }

    # 场景 2/3: 需要实体标注 → LLM 解析 categories（仅在此路径调用 LLM）
    system = (
        "你是白蛇传文本标注专家。分析用户的标注指令，确定需要标注的实体类别和颜色偏好。\n"
        "支持的类别: person(人物), location(地点), event(事件), term(术语), motif(母题), custom(自定义)\n"
        "默认颜色: person=#4A90D9, location=#50C878, event=#F5A623, term=#9B59B6, motif=#E91E90, custom=#F1C40F\n"
        "返回 JSON: {\"categories\": [\"person\", ...], \"color_overrides\": {\"person\": \"#FF0000\", ...}}"
    )
    sampled = _sample_text(full_text)
    user = f"用户指令: {user_instruction}\n\n原文抽样（首部/中部/尾部）:\n{sampled}"
    from ...config import settings as app_settings
    model = app_settings.effective_fast_model

    resp = await llm.generate(system, user, model=model, max_tokens=512, temperature=0.3, thinking_disabled=True)
    content_raw = resp.content.strip()
    if content_raw.startswith("```"):
        content_raw = content_raw.split("\n", 1)[1].rsplit("```", 1)[0]

    try:
        parsed = json.loads(content_raw)
    except json.JSONDecodeError as e:
        logger.error(f"annotate_user_request JSON 解析失败: {e}")
        return {"error": f"无法解析标注指令: {e}", "degraded": True}

    categories = parsed.get("categories", ["person", "location", "event", "term", "motif"])
    color_overrides = parsed.get("color_overrides", {})

    # 场景 2（纯实体标注）和场景 3（标注+批注）→ 调 annotate_passage
    from .annotate_passage import handler as annotate_handler

    result = await annotate_handler(
        chapter_number=chapter_number,
        paragraph_range=[min(paragraph_indices), max(paragraph_indices)],
        categories=categories,
        color_overrides=color_overrides,
        existing_user_annotations=existing_annotations or [],
        node_client=node_client,
        llm=llm,
    )

    # 场景 3: 标注 + 批注同时存在 → 附加 marginalia
    if result_marginalia:
        result["marginalia"] = result_marginalia

    result["user_instruction"] = user_instruction
    result["parsed_categories"] = categories
    result["parsed_colors"] = color_overrides
    return result
