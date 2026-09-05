"""B-014: explain_passage — 对指定段落生成 AI 注释（白话翻译 + 文化背景 + 学术争议）。"""
import json
import logging

logger = logging.getLogger(__name__)

TOOL_DEF = {
    "name": "explain_passage",
    "description": "对指定段落生成 AI 注释：白话翻译、文化背景说明、学术争议点。⚠️ 内部调用 LLM，耗时数秒；仅需原文上下文时用 get_paragraph_context（即时返回）。",
    "inputSchema": {
        "type": "object",
        "properties": {
            "chapter_number": {"type": "integer", "description": "章节编号"},
            "paragraph_index": {"type": "integer", "description": "段落索引（从 0 开始）"},
            "aspects": {
                "type": "array", "items": {"type": "string"},
                "description": "解释维度: translation / culture / scholarship，默认全部",
            },
        },
        "required": ["chapter_number", "paragraph_index"],
    },
}


async def handler(chapter_number: int, paragraph_index: int,
                  aspects: list = None, node_client=None, llm=None) -> dict:
    """对指定段落生成 AI 注释。

    fast fail: LLM JSON 格式异常 -> degraded，其余向上传播。
    """
    if node_client is None:
        return {"error": "Node API 客户端未初始化"}
    if llm is None:
        return {"error": "LLM 未初始化"}

    if aspects is None:
        aspects = ["translation", "culture", "scholarship"]

    ch = await node_client.get_literature_by_chapter(chapter_number)
    if ch is None:
        return {"error": f"未找到第 {chapter_number} 章"}

    from ._text_utils import split_paragraphs

    paragraphs = split_paragraphs(ch.get("content", ""))
    if paragraph_index < 0 or paragraph_index >= len(paragraphs):
        return {"error": f"段落索引 {paragraph_index} 超出范围 (0-{len(paragraphs) - 1})"}

    text = paragraphs[paragraph_index]
    aspect_labels = {"translation": "白话翻译", "culture": "文化背景", "scholarship": "学术争议"}
    requested = "、".join(aspect_labels.get(a, a) for a in aspects)

    system = (
        f"你是白蛇传文学研究专家。请对给定的文言/古白话段落，生成以下维度的注释: {requested}。\n"
        "只返回 JSON: {\"translation\": \"...\", \"cultural_context\": \"...\", \"scholarly_notes\": \"...\"}"
    )
    user = f"原文:\n\n{text}"

    from ...config import settings as app_settings
    model = app_settings.effective_model
    resp = await llm.generate(system, user, model=model, max_tokens=2048, temperature=0.5, thinking_disabled=True)
    content_raw = resp.content.strip()
    if content_raw.startswith("```"):
        content_raw = content_raw.split("\n", 1)[1].rsplit("```", 1)[0]

    try:
        data = json.loads(content_raw)
    except json.JSONDecodeError as e:
        logger.error(f"explain_passage JSON 解析失败: {e}")
        return {"error": f"AI 返回了无法解析的注释结果: {e}", "degraded": True}

    return {
        "paragraph_index": paragraph_index,
        "chapter_number": chapter_number,
        "chapter_title": ch.get("chapter_title", ""),
        "original_text": text,
        "translation": data.get("translation", "") if "translation" in aspects else "",
        "cultural_context": data.get("cultural_context", "") if "culture" in aspects else "",
        "scholarly_notes": data.get("scholarly_notes", "") if "scholarship" in aspects else "",
    }
