"""multi_entity_cooccurrence — 多实体关联出场工具。

查找多个实体（人物/地点等）在同一段落中共现的位置。
解决"金山寺和雷峰塔的关联出场""白娘子和法海的同场段落"类问题。
通过 retriever 逐字定位每个实体，再按 (chapter, paragraph) 做交集。
"""
import logging

logger = logging.getLogger(__name__)

TOOL_DEF = {
    "name": "multi_entity_cooccurrence",
    "description": (
        "查找多个实体（人物/地点/物件）在同一章节同一段落中共现的位置。"
        "返回共现段落的章节/朝代/原文摘录，并标注每个实体在该段落中的出现位置。"
        "适用于'XX和YY的关联出场''XX和YY在哪些段落同框'类问题。"
        "实体未在语料中出现时返回 0 并提示可能为别名。"
    ),
    "inputSchema": {
        "type": "object",
        "properties": {
            "entities": {
                "type": "array",
                "items": {"type": "string"},
                "description": "实体名称列表（2-4 个），如 ['白素贞', '法海']",
            },
        },
        "required": ["entities"],
    },
}

MAX_COOCCUR_RESULTS = 20


async def handler(entities: list[str], retriever=None) -> dict:
    """查找多实体共现段落。"""
    if retriever is None:
        return {"error": "检索器未初始化"}

    if not entities or len(entities) < 2:
        return {"error": "至少需要 2 个实体才能查询共现"}
    if len(entities) > 4:
        return {"error": "最多支持 4 个实体同时查询"}

    # 逐个实体定位出场
    entity_appearances = {}  # entity → {(chapter_number, paragraph_index): [appearances]}
    entity_totals = {}
    missing_entities = []

    for entity in entities:
        stats = await retriever.find_appearances(entity)
        total = stats.get("total", 0)
        entity_totals[entity] = total

        if not stats.get("corpus_available", True):
            return {"error": "语料索引当前不可用，无法定位实体出场。"}

        if total == 0:
            missing_entities.append(entity)
            continue

        # 按 (chapter_number, paragraph_index) 建索引
        index = {}
        for app in stats.get("appearances", []):
            ch = app.get("chapter_number", "")
            # paragraph_index 可能是 -1（研究文献），用 chapter + paragraph 组合
            para = app.get("paragraph_index", -1)
            # 如果没有 paragraph_index，用 chunk_id 作为 fallback
            key = (ch, para) if para >= 0 else (ch, app.get("chunk_id", ""))
            index.setdefault(key, []).append(app)
        entity_appearances[entity] = index

    if missing_entities:
        return {
            "title": f"多实体共现查询: {' + '.join(entities)}",
            "entities": entities,
            "total_cooccurrences": 0,
            "entity_totals": entity_totals,
            "message": (
                f"实体 {missing_entities} 在语料中未找到字面出现，无法查询共现。"
                "建议尝试其他写法或别名。"
            ),
            "cooccurrences": [],
        }

    # 求交集：找到所有实体都出现的 (chapter, paragraph) 组合
    common_keys = set(entity_appearances[entities[0]].keys())
    for entity in entities[1:]:
        common_keys &= set(entity_appearances[entity].keys())

    cooccurrences = []
    for key in sorted(common_keys, key=lambda k: (k[0] if k[0] != "" else "zzz", k[1] if isinstance(k[1], int) else 0)):
        # 收集每个实体在此段落的出场信息
        entity_hits = {}
        excerpt = ""
        chapter_title = ""
        dynasty = ""
        for entity in entities:
            apps = entity_appearances[entity].get(key, [])
            if apps:
                entity_hits[entity] = {
                    "count": len(apps),
                    "excerpt": apps[0].get("excerpt", ""),
                }
                if not excerpt:
                    excerpt = apps[0].get("excerpt", "")
                if not chapter_title:
                    chapter_title = apps[0].get("chapter_title", "")
                if not dynasty:
                    dynasty = apps[0].get("dynasty", "")

        cooccurrences.append({
            "chapter_number": key[0] if isinstance(key[0], str) else str(key[0]),
            "paragraph_index": key[1] if isinstance(key[1], int) else -1,
            "chapter_title": chapter_title,
            "dynasty": dynasty,
            "entity_hits": entity_hits,
            "excerpt": excerpt[:300],
        })

        if len(cooccurrences) >= MAX_COOCCUR_RESULTS:
            break

    return {
        "title": f"多实体共现: {' + '.join(entities)}",
        "entities": entities,
        "entity_totals": entity_totals,
        "total_cooccurrences": len(cooccurrences),
        "truncated": len(cooccurrences) >= MAX_COOCCUR_RESULTS,
        "cooccurrences": cooccurrences,
        "note": (
            f"共找到 {len(cooccurrences)} 个共现段落。"
            + ("结果已截断，建议缩小实体范围。" if len(cooccurrences) >= MAX_COOCCUR_RESULTS else "")
        ),
    }
