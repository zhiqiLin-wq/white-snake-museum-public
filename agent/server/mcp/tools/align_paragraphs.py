"""v9: align_paragraphs — 对齐两个章节的段落对应关系。

接收两个章节号，通过 LLM 分析段落内容的对应关系，
返回 aligned paragraph pairs 供 Split View 同步滚动/对齐使用。
"""
import json
import logging

logger = logging.getLogger(__name__)

TOOL_DEF = {
    "name": "align_paragraphs",
    "description": (
        "分析两个章节的段落对应关系，返回段落对齐映射。"
        "用于 Split View 中的段落同步滚动和对齐显示。"
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
        },
        "required": ["left_chapter", "right_chapter"],
    },
}


async def handler(left_chapter: int, right_chapter: int,
                  node_client=None, llm=None) -> dict:
    """对齐两个章节的段落。"""
    if node_client is None:
        return {"error": "Node API 客户端未初始化"}
    if llm is None:
        return {"error": "LLM 未初始化"}

    left_ch = await node_client.get_literature_by_chapter(left_chapter)
    right_ch = await node_client.get_literature_by_chapter(right_chapter)
    if left_ch is None or right_ch is None:
        return {"error": "章节未找到"}

    from ._text_utils import split_paragraphs

    def get_paragraphs(ch):
        return [{"index": i, "text": p}
                for i, p in enumerate(split_paragraphs(ch.get("content", "")))]

    left_paras = get_paragraphs(left_ch)
    right_paras = get_paragraphs(right_ch)

    # Build summary for LLM
    left_summary = "\n".join(
        f"[{p['index']}] {p['text'][:60]}" for p in left_paras[:10]
    )
    right_summary = "\n".join(
        f"[{p['index']}] {p['text'][:60]}" for p in right_paras[:10]
    )

    system = (
        "你是文本对齐专家。分析两个章节的段落，找出内容对应的段落对。\n"
        "返回 JSON: {\"alignments\": ["
        "{\"leftParagraphIndex\": 0, \"rightParagraphIndex\": 0, \"confidence\": 0.95}...]}"
    )
    user = (
        f"左侧章节 ({left_ch.get('title', '')}) 段落:\n{left_summary}\n\n"
        f"右侧章节 ({right_ch.get('title', '')}) 段落:\n{right_summary}"
    )
    from ...config import settings as app_settings
    model = app_settings.effective_fast_model

    resp = await llm.generate(system, user, model=model, max_tokens=1024, temperature=0.2, thinking_disabled=True)
    content_raw = resp.content.strip()
    if content_raw.startswith("```"):
        content_raw = content_raw.split("\n", 1)[1].rsplit("```", 1)[0]

    try:
        data = json.loads(content_raw)
    except json.JSONDecodeError as e:
        logger.error(f"align_paragraphs JSON 解析失败: {e}")
        return {"error": f"AI 返回了无法解析的对齐结果: {e}", "degraded": True}

    result = data.get("alignments", [])
    normalized = []
    for a in result:
        normalized.append({
            "leftParagraphIndex": a.get("leftParagraphIndex", a.get("left", 0)),
            "rightParagraphIndex": a.get("rightParagraphIndex", a.get("right", 0)),
            "confidence": a.get("confidence", 0.5),
        })
    return {
        "leftChapter": left_chapter,
        "rightChapter": right_chapter,
        "alignments": normalized,
    }
