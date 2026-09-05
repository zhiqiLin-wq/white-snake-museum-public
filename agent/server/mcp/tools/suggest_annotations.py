"""v9: suggest_annotations — 生成标注建议（不直接写入，返回 suggestions 供用户确认）。

Agent 分析段落并返回标注建议列表，前端渲染为半透明虚线 + 确认/忽略按钮。
用户确认后调用 accept_suggestion 转为正式标注。
"""
import json
import logging

logger = logging.getLogger(__name__)

TOOL_DEF = {
    "name": "suggest_annotations",
    "description": (
        "对指定段落生成标注建议列表，但不直接写入。"
        "返回结构化建议供前端渲染确认/忽略 UI。"
    ),
    "inputSchema": {
        "type": "object",
        "properties": {
            "chapter_number": {
                "type": "integer",
                "description": "章节编号",
            },
            "paragraph_range": {
                "type": "array",
                "items": {"type": "integer"},
                "minItems": 2,
                "maxItems": 2,
                "description": "起止段落索引 [start, end]",
            },
            "categories": {
                "type": "array",
                "items": {"type": "string"},
                "description": "建议标注的类别，默认全部",
            },
            "color_overrides": {
                "type": "object",
                "description": "类别→颜色覆盖映射",
            },
        },
        "required": ["chapter_number", "paragraph_range"],
    },
}

DEFAULT_CATEGORY_COLORS = {
    "person": "#68A0C8",
    "location": "#78B898",
    "event": "#68B0B8",
    "term": "#9888C0",
    "motif": "#C898B0",
    "custom": "#B8B878",
}


async def handler(chapter_number: int, paragraph_range: list,
                  categories: list = None,
                  color_overrides: dict = None,
                  node_client=None, llm=None) -> dict:
    """生成标注建议。"""
    if node_client is None:
        return {"error": "Node API 客户端未初始化"}
    if llm is None:
        return {"error": "LLM 未初始化"}

    if categories is None:
        categories = ["person", "location", "event", "term", "motif"]
    if color_overrides is None:
        color_overrides = {}

    color_map = {**DEFAULT_CATEGORY_COLORS, **color_overrides}

    ch = await node_client.get_literature_by_chapter(chapter_number)
    if ch is None:
        return {"error": f"未找到第 {chapter_number} 章"}

    from ._text_utils import split_paragraphs, compact_to_orig_start

    paragraphs = split_paragraphs(ch.get("content", ""))

    start_idx = max(0, paragraph_range[0])
    end_idx = min(len(paragraphs), paragraph_range[1] + 1)
    target_paragraphs = paragraphs[start_idx:end_idx]
    if not target_paragraphs:
        return {"error": "指定的段落范围为空"}

    # 段落间用 \n\n 连接，与下方 offset += len(p_text) + 2 的坐标系一致
    full_text = "\n\n".join(target_paragraphs)

    cat_labels = {
        "person": "人物名称",
        "location": "地点名称",
        "event": "事件名称",
        "term": "专有术语",
        "motif": "叙事母题",
    }
    cat_list = "、".join(cat_labels.get(c, c) for c in categories)

    system = (
        f"你是文本标注专家。分析以下原文，为每个可标注的实体生成一条建议。\n"
        f"目标类别: {cat_list}。\n"
        f"对于每条建议，提供: 实体文本、类别、为什么建议标注（reason）。\n"
        f"返回 JSON: {{\"suggestions\": ["
        f"{{\"text\": \"实体文本\", \"category\": \"person\", "
        f"\"start_char\": N, \"end_char\": N, "
        f"\"reason\": \"标注原因\"}}]}}"
    )
    user = f"原文:\n\n{full_text}"
    from ...config import settings as app_settings
    model = app_settings.effective_fast_model

    resp = await llm.generate(system, user, model=model, max_tokens=2048, temperature=0.2, thinking_disabled=True)
    content_raw = resp.content.strip()
    if content_raw.startswith("```"):
        content_raw = content_raw.split("\n", 1)[1].rsplit("```", 1)[0]

    try:
        data = json.loads(content_raw)
    except json.JSONDecodeError as e:
        logger.error(f"suggest_annotations JSON 解析失败: {e}")
        return {"error": f"AI 返回了无法解析的建议: {e}", "degraded": True}

    import uuid
    import re as _re

    # Build paragraph boundaries to map character offsets to paragraphs
    para_boundaries = []
    offset = 0
    for p_idx in range(start_idx, end_idx):
        p_text = paragraphs[p_idx]
        para_boundaries.append({"idx": p_idx, "start": offset, "end": offset + len(p_text)})
        offset += len(p_text) + 2  # +2 for paragraph separator (\n\n)

    def resolve_paragraph(char_pos):
        for b in para_boundaries:
            if b["start"] <= char_pos < b["end"]:
                return b["idx"], char_pos - b["start"]
        # Boundary fallback: try inclusive end
        for b in para_boundaries:
            if b["start"] <= char_pos <= b["end"]:
                return b["idx"], max(0, char_pos - b["start"])
        return start_idx, char_pos

    grouped: dict[int, list[dict]] = {}
    for item in data.get("suggestions", []):
        entity = item.get("text", "")
        if not entity:
            continue

        # 三级定位实体在 full_text 中的位置，避免段落归属错位:
        # 1) 优先采用 LLM 返回的 start_char（带文本校验，防止 LLM 数字符偏差，
        #    同时避免多段同实体时 find 只取首次出现而标错段落）
        # 2) 原文字面 find
        # 3) 忽略空白容错匹配，命中后用 compact_to_orig_start 映射回原文坐标
        pos = -1
        sc = item.get("start_char")
        if isinstance(sc, int) and 0 <= sc < len(full_text) \
                and full_text[sc:sc + len(entity)] == entity:
            pos = sc
        if pos == -1:
            pos = full_text.find(entity)
        if pos == -1:
            entity_compact = entity.replace(" ", "").replace("　", "")
            text_compact = full_text.replace(" ", "").replace("　", "")
            compact_pos = text_compact.find(entity_compact)
            if compact_pos >= 0:
                pos = compact_to_orig_start(full_text, compact_pos)
                if pos is None:
                    pos = -1
        if pos == -1:
            continue

        para_idx, rel_start = resolve_paragraph(pos)
        rel_end = rel_start + len(entity)
        cat = item.get("category", "custom")

        if para_idx not in grouped:
            grouped[para_idx] = []
        grouped[para_idx].append({
            "id": str(uuid.uuid4()),
            "text": entity,
            "category": cat,
            "span": {"startChar": rel_start, "endChar": rel_end},
            "reason": item.get("reason", ""),
            "color": color_map.get(cat, DEFAULT_CATEGORY_COLORS["custom"]),
        })

    passage_annotations = []
    for para_idx in sorted(grouped.keys()):
        passage_annotations.append({
            "passageKey": f"{chapter_number}:{para_idx}",
            "paragraphIndex": para_idx,
            "suggestions": grouped[para_idx],
        })

    return {
        "chapterNumber": chapter_number,
        "chapterTitle": ch.get("title", ""),
        "passageAnnotations": passage_annotations,
    }
