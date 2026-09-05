"""B-016: count_occurrences — 实体频次统计，按朝代/章节分组。"""
import logging

logger = logging.getLogger(__name__)

TOOL_DEF = {
    "name": "count_occurrences",
    "description": (
        "统计实体在白蛇传各朝代/章节的字面出现频次（全文逐字匹配计数，非语义估算），"
        "含占比和 AI 解读。实体未在语料中出现时返回 total=0 并提示可能为版本别名，"
        "此时请勿编造数字，应建议用户尝试其他写法。"
    ),
    "inputSchema": {
        "type": "object",
        "properties": {
            "entity": {"type": "string", "description": "实体名称，如 '白素贞'、'雷峰塔'"},
            "group_by": {"type": "string", "description": "分组维度: dynasty 或 chapter", "default": "dynasty"},
        },
        "required": ["entity"],
    },
}


async def handler(entity: str, group_by: str = "dynasty", retriever=None, llm=None) -> dict:
    """统计实体频次（字面计数，非语义召回）。

    使用 count_literal 遍历全库做子串计数——语义检索召回的 chunk 数
    不等于出现次数（实体不在语料中时语义相近照样召回），会误导研究者。
    0 命中时明确返回提示，LLM 不得编造频次。
    AI 解读失败不影响主统计结果（非关键路径）。
    """
    if retriever is None:
        return {"error": "检索器未初始化"}

    stats = await retriever.count_literal(entity)
    total = stats["total"]

    # 语料库不可用（索引重建中/集合为空）：明确报错，禁止把"查不了"误报成"0 次"
    if not stats.get("corpus_available", True):
        return {
            "error": (
                "语料索引当前不可用（可能正在后台重建），无法统计字面频次。"
                "请稍后重试本工具；在此之前不要向用户报告任何频次数字，"
                "可改用 search_literature（基于 Node 全文检索）临时回答。"
            ),
        }

    columns = [
        {"key": "group", "label": "朝代" if group_by == "dynasty" else "章节", "type": "string"},
        {"key": "count", "label": "频次", "type": "integer"},
        {"key": "ratio", "label": "占比", "type": "percent"},
    ]

    # 0 命中：明确告知语料中未出现该词，禁止用语义相关数据充当频次
    if total == 0:
        message = (
            f"「{entity}」在白蛇传语料全文中字面出现 0 次，无频次数据可统计。"
            "该实体可能是其他版本的译名/别名（同一人物在不同版本文本中名称不同），"
            "建议：1) 尝试该实体的其他常见写法或别名重新统计；"
            "2) 向用户如实说明本语料中未收录该名称，不要编造数字。"
        )
        return {
            "title": f"'{entity}' 出现频次统计",
            "metric": "频次",
            "entity": entity,
            "literal_match": False,
            "columns": columns,
            "rows": [],
            "total_occurrences": 0,
            "message": message,
            "aiSummary": f"「{entity}」在语料中字面出现 0 次，无频次数据（可能为其他版本的别名）。",
        }

    if group_by == "dynasty":
        counts = stats["by_dynasty"]
        _DYNASTY_ORDER = ["唐", "宋", "明", "清", "现当代"]

        def _sort_key(k: str):
            try:
                return (0, _DYNASTY_ORDER.index(k))
            except ValueError:
                return (1, k)  # 研究文献/未知排在后面
    else:
        counts = stats["by_chapter"]

        def _sort_key(k: str):
            return (0, int(k)) if k.isdigit() else (1, k)

    # B-153: 对齐前端 StatsDataEvent 协议 — columns 必须是 {key,label,type} 列定义对象，
    # rows 按列 key 取值，AI 解读字段为 aiSummary（驼峰）。
    fallback = "研究文献" if group_by == "dynasty" else "未知章节"

    def _display(k: str) -> str:
        return k if k else fallback

    rows = [
        {"group": _display(k), "count": counts[k],
         "ratio": round(counts[k] / total, 3) if total > 0 else 0}
        for k in sorted(counts.keys(), key=_sort_key)
    ]

    # AI 一句话解读（非关键路径——失败不影响主结果）
    ai_summary = ""
    if llm:
        try:
            from ...config import settings as app_settings
            fast_model = app_settings.effective_fast_model
            system = "你是数据分析助手。用一句话总结统计结果。只返回 JSON: {\"summary\": \"...\"}"
            user = f"'{entity}'在各{'朝代' if group_by == 'dynasty' else '章节'}的出现次数: {counts}"
            # 小 max_tokens(128) 必须关思考，否则推理内容吃光预算导致空响应（lessons learned）
            resp = await llm.generate(system, user, model=fast_model, max_tokens=128,
                                      temperature=0.3, thinking_disabled=True)
            from ...llm.json_utils import parse_llm_json
            data = parse_llm_json(resp.content.strip())
            ai_summary = data.get("summary", "")
        except (ValueError, KeyError, AttributeError) as e:
            logger.warning(f"count_occurrences AI 解读失败（非关键）: {e}")

    return {
        "title": f"'{entity}' 出现频次统计",
        "metric": "频次",
        "entity": entity,
        "literal_match": True,
        "columns": columns,
        "rows": rows,
        "total_occurrences": total,
        "aiSummary": ai_summary,
    }
