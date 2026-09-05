"""B-017: keyword_trend — 关键词跨朝代趋势分析。"""
import logging

logger = logging.getLogger(__name__)

TOOL_DEF = {
    "name": "keyword_trend",
    "description": (
        "追踪关键词在白蛇传不同朝代（唐/宋/明/清）的相关段落分布趋势。"
        "⚠️ 口径说明：数字是语义检索召回的『语义相关段落数』，不是字面出现频次——"
        "关键词未在语料中出现时，语义相近的内容也会产生非零数字，且受检索上限(50)影响。"
        "需要全文逐字的真实出现次数请改用 count_occurrences。"
        "向用户转述时请使用『相关段落』而非『出现次数』表述。"
    ),
    "inputSchema": {
        "type": "object",
        "properties": {
            "keyword": {"type": "string", "description": "要追踪的关键词，如 '雷峰塔'"},
            "metric": {"type": "string", "description": "度量方式: frequency / tfidf / sentiment", "default": "frequency"},
        },
        "required": ["keyword"],
    },
}


async def handler(keyword: str, metric: str = "frequency", retriever=None) -> dict:
    """追踪关键词跨朝代的语义相关段落分布趋势。

    ⚠️ 数据口径：基于语义检索召回的 chunk 按朝代计数（aggregate_by_dynasty），
    是"相关程度分布"，不是字面出现频次。字段命名均如实标注 data_basis
    与 metric_label，防止 LLM 向用户误述为"出现次数"。
    fast fail: retriever 调用错误向上传播。
    """
    if retriever is None:
        return {"error": "检索器未初始化"}

    counts = await retriever.aggregate_by_dynasty(keyword)

    dynasties = ["唐", "宋", "明", "清"]
    data = []
    for d in dynasties:
        count = counts.get(d, 0)
        data.append({"dynasty": d, "value": count})

    total = sum(d["value"] for d in data)
    max_val = max(d["value"] for d in data) if data else 1
    for item in data:
        item["normalized"] = round(item["value"] / max_val, 3) if max_val > 0 else 0

    if len(data) >= 2:
        trend_direction = "上升" if data[-1]["value"] > data[0]["value"] else ("下降" if data[-1]["value"] < data[0]["value"] else "平稳")
    else:
        trend_direction = "数据不足"

    return {
        "title": f"'{keyword}' 跨朝代相关段落分布趋势",
        "keyword": keyword,
        "metric": metric,
        "data_basis": "semantic_recall",
        "metric_label": "语义相关段落数（非字面出现频次）",
        "data": data,
        "total_related_paragraphs": total,
        "trend_direction": trend_direction,
        "note": (
            "以上数字为与关键词语义相关的召回段落数（受检索上限影响），"
            "不代表关键词的字面出现次数。"
        ),
    }
