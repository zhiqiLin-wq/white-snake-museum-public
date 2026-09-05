"""B-015: compare_passages — 跨版本/跨朝代对比两段文本。"""
import logging

logger = logging.getLogger(__name__)

TOOL_DEF = {
    "name": "compare_passages",
    "description": (
        "跨版本/跨朝代对比指定的两段文本，LLM 从指定维度识别差异并分类变化类型。"
        "只做分析，不写入任何标注。"
        "需要将差异结果写入前端标注时改用 compare_and_annotate；"
        "找两章之间的共享文本片段（原文级证据）请用 find_shared_passages。"
    ),
    "inputSchema": {
        "type": "object",
        "properties": {
            "passage_a": {
                "type": "object",
                "description": "对比对象 A: {chapter_number, paragraph_range: [start, end]}",
                "properties": {
                    "chapter_number": {"type": "integer"},
                    "paragraph_range": {"type": "array", "items": {"type": "integer"}, "minItems": 2, "maxItems": 2},
                },
                "required": ["chapter_number", "paragraph_range"],
            },
            "passage_b": {
                "type": "object",
                "description": "对比对象 B: {chapter_number, paragraph_range: [start, end]}",
                "properties": {
                    "chapter_number": {"type": "integer"},
                    "paragraph_range": {"type": "array", "items": {"type": "integer"}, "minItems": 2, "maxItems": 2},
                },
                "required": ["chapter_number", "paragraph_range"],
            },
            "aspect": {
                "type": "string",
                "description": "对比维度: character / plot / imagery / language",
                "default": "character",
            },
        },
        "required": ["passage_a", "passage_b"],
    },
}


async def handler(passage_a: dict, passage_b: dict, aspect: str = "character",
                  node_client=None, llm=None) -> dict:
    """对比两段文本。

    fast fail: LLM JSON 格式异常 -> degraded，其余向上传播。
    """
    if node_client is None:
        return {"error": "Node API 客户端未初始化"}
    if llm is None:
        return {"error": "LLM 未初始化"}

    text_a = await _get_passage_text(node_client, passage_a)
    text_b = await _get_passage_text(node_client, passage_b)
    if isinstance(text_a, dict) and "error" in text_a:
        return text_a
    if isinstance(text_b, dict) and "error" in text_b:
        return text_b

    system = (
        f"你是白蛇传跨版本对比分析专家。从'{aspect}'维度对比两段文本。\n"
        "识别差异并分类为 7 种变化类型之一: "
        "emergence/disappearance/substitution/amplification/diminution/transvaluation/transplantation。\n"
        "只返回 JSON: {{\"diff_summary\": \"...\", \"change_type\": \"...\", \"key_differences\": [\"...\"], \"confidence\": \"high|medium|low\"}}"
    )
    user = f"文本A:\n{text_a}\n\n文本B:\n{text_b}\n\n请从{aspect}维度进行对比分析。"

    from ...config import settings as app_settings
    model = app_settings.effective_model
    # 分类+JSON 任务：关思考防止推理吃光预算导致空响应；低温提升分类稳定性
    resp = await llm.generate(system, user, model=model, max_tokens=2048,
                              temperature=0.3, thinking_disabled=True)
    content_raw = resp.content.strip()

    from ...llm.json_utils import parse_llm_json
    try:
        data = parse_llm_json(content_raw)
    except (ValueError, KeyError) as e:
        logger.error(f"compare_passages JSON 解析失败: {e}")
        return {"error": f"AI 返回了无法解析的对比结果: {e}", "degraded": True}

    # B-153: 对齐前端 ComparisonDataEvent 协议 —
    # left/right 需要 {label, text, highlights}（并排阅读面板），diffSummary 需要
    # {changes: [{type, description}], narrative}。旧版返回平铺的 diff_summary/
    # key_differences + text_preview(200字)，前端渲染成空壳。
    ch_a, pa = passage_a["chapter_number"], passage_a["paragraph_range"]
    ch_b, pb = passage_b["chapter_number"], passage_b["paragraph_range"]
    change_type = data.get("change_type", "substitution")

    return {
        "left": {
            "label": f"第{ch_a}章 · 段落{pa[0]}-{pa[1]}",
            "text": text_a[:3000],
            "highlights": [],
        },
        "right": {
            "label": f"第{ch_b}章 · 段落{pb[0]}-{pb[1]}",
            "text": text_b[:3000],
            "highlights": [],
        },
        "diffSummary": {
            "changes": [
                {"type": change_type, "description": d}
                for d in (data.get("key_differences") or [])
                if isinstance(d, str) and d.strip()
            ],
            "narrative": data.get("diff_summary", ""),
        },
        "confidence": data.get("confidence", "medium"),
    }


async def _get_passage_text(node_client, passage_spec: dict) -> str:
    ch_num = passage_spec["chapter_number"]
    p_range = passage_spec["paragraph_range"]
    ch = await node_client.get_literature_by_chapter(ch_num)
    if ch is None:
        return f"[未找到第{ch_num}章]"
    from ._text_utils import split_paragraphs

    paragraphs = split_paragraphs(ch.get("content", ""))
    start = max(0, p_range[0])
    end = min(len(paragraphs), p_range[1] + 1)
    return "\n".join(paragraphs[start:end])
