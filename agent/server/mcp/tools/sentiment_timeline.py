"""B-019: sentiment_timeline — 情感倾向时间线。"""
import json
import logging

logger = logging.getLogger(__name__)

TOOL_DEF = {
    "name": "sentiment_timeline",
    "description": (
        "分析实体在各朝代（唐/宋/明/清）文本中的情感倾向变化（-1 负面 ~ +1 正面）。"
        "⚠️ 口径说明：基于语义检索召回的相关段落样本评分——每个朝代取语义最相关的前 3 个段落"
        "（拼接前 200 字符）由 LLM 评定，是抽样分析而非全文精读；"
        "实体未在语料中出现时，召回的可能只是语义相近内容，评分需谨慎解读。"
        "向用户转述时请说明这是『基于相关段落样本的情感分析』。"
    ),
    "inputSchema": {
        "type": "object",
        "properties": {
            "entity": {"type": "string", "description": "目标实体，如 '法海'"},
            "group_by": {"type": "string", "description": "分组维度: dynasty", "default": "dynasty"},
        },
        "required": ["entity"],
    },
}


async def handler(entity: str, group_by: str = "dynasty", retriever=None, llm=None) -> dict:
    """情感倾向时间线分析（基于语义召回的段落样本）。

    ⚠️ 数据口径：每个朝代用实体名+朝代做语义检索，取前 3 个相关段落
    （各取前 200 字符）拼接后由 LLM 评定情感分——抽样而非全文精读。
    fast fail: retriever/llm 调用错误向上传播。
    单朝代 LLM 分析失败 -> 该朝代 score=0，不阻塞整体。
    """
    if retriever is None:
        return {"error": "检索器未初始化"}
    if llm is None:
        return {"error": "LLM 未初始化"}

    dynasties = ["唐", "宋", "明", "清"]
    data = []

    from ...config import settings as app_settings
    fast_model = app_settings.effective_fast_model
    system = (
        "你是情感分析专家。分析文本中对指定实体的情感倾向。"
        "返回 -1（非常负面）到 +1（非常正面）的情感分数，以及 3-5 个关键情感词。"
        "只返回 JSON: {\"score\": 0.5, \"keywords\": [\"...\"]}"
    )

    for dynasty in dynasties:
        docs = await retriever.retrieve(f"{entity} {dynasty}", filters={"dynasty": dynasty})
        if not docs:
            data.append({"group": dynasty, "sentiment_score": 0, "keywords": []})
            continue

        combined_text = " ".join(d.get("content", "")[:200] for d in docs[:3])
        user = f"实体: {entity}\n朝代: {dynasty}\n相关文本:\n{combined_text}"

        # 单朝代分析失败 -> score=0，不阻塞（非关键路径）
        try:
            resp = await llm.generate(system, user, model=fast_model, max_tokens=256, temperature=0.2, thinking_disabled=True)
            content_raw = resp.content.strip()
            if content_raw.startswith("```"):
                content_raw = content_raw.split("\n", 1)[1].rsplit("```", 1)[0]
            result = json.loads(content_raw)
            data.append({
                "group": dynasty,
                "sentiment_score": max(-1.0, min(1.0, result.get("score", 0))),
                "keywords": result.get("keywords", []),
            })
        except (json.JSONDecodeError, KeyError) as e:
            logger.warning(f"sentiment_timeline {entity}@{dynasty} 分析失败（非关键）: {e}")
            data.append({"group": dynasty, "sentiment_score": 0, "keywords": []})

    return {
        "title": f"'{entity}' 跨朝代情感倾向（相关段落样本分析）",
        "entity": entity,
        "data_basis": "semantic_recall_sample",
        "metric_label": "情感分（基于每朝代前 3 个相关段落样本）",
        "data": data,
        "note": (
            "情感分由 LLM 基于每朝代语义最相关的前 3 个段落样本评定，"
            "为抽样分析，不代表该朝代全文的整体情感。"
        ),
    }
