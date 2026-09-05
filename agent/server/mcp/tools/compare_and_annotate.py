"""v9: compare_and_annotate — 接收左右两个章节的文本，返回差异标注。

用于 Split View 和 Compare View 中的自动对比分析。
输出左右两栏各自的 highlights 和差异变化类型。
"""
import json
import logging

logger = logging.getLogger(__name__)

TOOL_DEF = {
    "name": "compare_and_annotate",
    "description": (
        "对左右两个章节进行自动对比分析，检测差异并标注变化类型"
        "(emergence/disappearance/substitution/amplification/diminution/transvaluation/transplantation)，"
        "结果会写入前端标注供对读视图展示。"
        "⚠️ 仅需分析不写入标注时用 compare_passages；多章节(3章以上)主题对比用 compare_chapters。"
    ),
    "inputSchema": {
        "type": "object",
        "properties": {
            "left_chapter": {
                "type": "integer",
                "description": "左侧章节编号",
            },
            "right_chapter": {
                "type": "integer",
                "description": "右侧章节编号",
            },
            "left_paragraph_range": {
                "type": "array",
                "items": {"type": "integer"},
                "minItems": 2,
                "maxItems": 2,
                "description": "左侧起止段落索引 [start, end]",
            },
            "right_paragraph_range": {
                "type": "array",
                "items": {"type": "integer"},
                "minItems": 2,
                "maxItems": 2,
                "description": "右侧起止段落索引 [start, end]",
            },
        },
        "required": ["left_chapter", "right_chapter"],
    },
}


async def handler(left_chapter: int, right_chapter: int,
                  left_paragraph_range: list = None,
                  right_paragraph_range: list = None,
                  node_client=None, llm=None) -> dict:
    """对比两个章节的文本并生成差异标注。"""
    if node_client is None:
        return {"error": "Node API 客户端未初始化"}
    if llm is None:
        return {"error": "LLM 未初始化"}

    left_ch = await node_client.get_literature_by_chapter(left_chapter)
    right_ch = await node_client.get_literature_by_chapter(right_chapter)
    if left_ch is None or right_ch is None:
        return {"error": "章节未找到"}

    from ._text_utils import split_paragraphs

    def get_text(ch, para_range):
        paragraphs = split_paragraphs(ch.get("content", ""))
        if para_range:
            start = max(0, para_range[0])
            end = min(len(paragraphs), para_range[1] + 1)
            return "\n".join(paragraphs[start:end])
        return "\n".join(paragraphs)

    left_text = get_text(left_ch, left_paragraph_range)
    right_text = get_text(right_ch, right_paragraph_range)

    system = (
        "你是比较文学分析专家。对比以下左右两个文本，识别内容层面的差异。\n"
        "变化类型: emergence(新增), disappearance(消失), substitution(替换), "
        "amplification(扩展), diminution(缩减), transvaluation(价值翻转), transplantation(移植)。\n\n"
        "对于每条差异，必须提供左右两侧的原文片段（每个片段15-40字，必须能在原文中字面找到）:\n"
        "返回 JSON:\n"
        '{"changes": [\n'
        '  {\n'
        '    "type": "substitution",\n'
        '    "description": "法海形象的转变",\n'
        '    "leftSnippet": "法海禅师慈悲为怀，以降妖除魔为己任",\n'
        '    "rightSnippet": "法海和尚铁面无情，将白娘子镇于塔下"\n'
        '  }\n'
        '],\n'
        '"narrative": "整体差异概述"}'
    )
    user = f"左侧文本:\n{left_text[:1500]}\n\n右侧文本:\n{right_text[:1500]}"
    from ...config import settings as app_settings
    model = app_settings.effective_fast_model

    resp = await llm.generate(system, user, model=model, max_tokens=2048, temperature=0.3, thinking_disabled=True)
    content_raw = resp.content.strip()
    if content_raw.startswith("```"):
        content_raw = content_raw.split("\n", 1)[1].rsplit("```", 1)[0]

    try:
        data = json.loads(content_raw)
    except json.JSONDecodeError as e:
        logger.error(f"compare_and_annotate JSON 解析失败: {e}")
        return {"error": f"AI 返回了无法解析的对比结果: {e}", "degraded": True}

    # Post-process: locate snippets in left/right texts to produce span highlights
    def _compact_to_orig_start(original: str, compact_start: int) -> int | None:
        """Map compact-start position back to original text by skipping whitespace chars."""
        count = 0
        for i, ch in enumerate(original):
            if ch not in (" ", "\n", "\r", "　"):
                if count == compact_start:
                    return i
                count += 1
        return None

    def _original_end_from_compact_len(original: str, orig_start: int, compact_len: int) -> int:
        """Compute original-text end position given start and compact character count."""
        count = 0
        for i in range(orig_start, len(original)):
            if original[i] not in (" ", "\n", "\r", "　"):
                count += 1
            if count == compact_len:
                return i + 1
        return orig_start + compact_len

    left_highlights = []
    right_highlights = []
    for change in data.get("changes", []):
        ch_type = change.get("type", "substitution")
        ch_desc = change.get("description", "")
        left_snippet = change.get("leftSnippet", "")
        right_snippet = change.get("rightSnippet", "")

        if left_snippet:
            pos = left_text.find(left_snippet)
            if pos >= 0:
                left_highlights.append({
                    "span": [pos, pos + len(left_snippet)],
                    "type": ch_type,
                    "label": ch_desc[:40],
                })
            else:
                compact_snippet = left_snippet.replace(" ", "").replace("\n", "").replace("　", "")
                compact_text = left_text.replace(" ", "").replace("\n", "").replace("　", "")
                compact_pos = compact_text.find(compact_snippet)
                if compact_pos >= 0:
                    orig_pos = _compact_to_orig_start(left_text, compact_pos)
                    if orig_pos is not None:
                        orig_end = _original_end_from_compact_len(left_text, orig_pos, len(compact_snippet))
                        left_highlights.append({
                            "span": [orig_pos, orig_end],
                            "type": ch_type,
                            "label": ch_desc[:40],
                        })

        if right_snippet:
            pos = right_text.find(right_snippet)
            if pos >= 0:
                right_highlights.append({
                    "span": [pos, pos + len(right_snippet)],
                    "type": ch_type,
                    "label": ch_desc[:40],
                })
            else:
                compact_snippet = right_snippet.replace(" ", "").replace("\n", "").replace("　", "")
                compact_text = right_text.replace(" ", "").replace("\n", "").replace("　", "")
                compact_pos = compact_text.find(compact_snippet)
                if compact_pos >= 0:
                    orig_pos = _compact_to_orig_start(right_text, compact_pos)
                    if orig_pos is not None:
                        orig_end = _original_end_from_compact_len(right_text, orig_pos, len(compact_snippet))
                        right_highlights.append({
                            "span": [orig_pos, orig_end],
                            "type": ch_type,
                            "label": ch_desc[:40],
                        })

    return {
        "left": {
            "label": left_ch.get("title", f"第{left_chapter}章"),
            "chapterNumber": str(left_chapter),
            "paragraphIndex": left_paragraph_range[0] if left_paragraph_range else 0,
            "text": left_text,
            "highlights": left_highlights,
        },
        "right": {
            "label": right_ch.get("title", f"第{right_chapter}章"),
            "chapterNumber": str(right_chapter),
            "paragraphIndex": right_paragraph_range[0] if right_paragraph_range else 0,
            "text": right_text,
            "highlights": right_highlights,
        },
        "diffSummary": {
            "changes": data.get("changes", []),
            "narrative": data.get("narrative", ""),
        },
    }
