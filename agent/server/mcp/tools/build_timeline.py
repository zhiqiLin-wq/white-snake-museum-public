"""P2-6: build_timeline — 时间线结构化输出工具。

按朝代时间轴组织事件，返回结构化 JSON 供前端画时间线。
数据来源：调用方传入 entities/entities 列表 + node_client 查原文标注。

与 sentiment_timeline（情感值）/ keyword_trend（词频）不同：
本工具输出的是"叙事事件时间轴"本身——每条事件含朝代/年份/事件描述/证据摘录/章节/段落锚点。
"""
import logging
import re
from typing import List

logger = logging.getLogger(__name__)

DYNASTY_ORDER = {"唐": 1, "宋": 2, "明": 3, "清": 4, "现当代": 5}
DYNASTY_YEAR_RANGE = {
    "唐": (618, 907),
    "宋": (960, 1279),
    "明": (1368, 1644),
    "清": (1644, 1912),
    "现当代": (1912, 2026),
}


def _infer_dynasty(chapter_number: int) -> str:
    """根据章节号推断朝代（与 _text_utils.infer_dynasty_by_chapter 一致）。"""
    if chapter_number <= 4:
        return "唐"
    elif chapter_number <= 10:
        return "宋"
    elif chapter_number <= 20:
        return "明"
    elif chapter_number <= 30:
        return "清"
    else:
        return "现当代"


TOOL_DEF = {
    "name": "build_timeline",
    "description": (
        "构建叙事事件时间线。按朝代（唐/宋/明/清/现当代）组织事件，"
        "每条事件含朝代/年份范围/事件描述/证据摘录/章节号/段落索引/实体名。"
        "返回结构化 JSON 供前端渲染时间线图表。"
        "典型场景：用户问「白蛇形象演变时间线」「按时间顺序展示事件」"
        "「各朝代关键事件列表」「画一条白蛇传演变时间轴」。"
        "与 sentiment_timeline（情感值变化）和 keyword_trend（词频变化）互补。"
    ),
    "inputSchema": {
        "type": "object",
        "properties": {
            "entity": {
                "type": "string",
                "description": "追踪的实体名（如'白蛇'/'许仙'/'法海'/'雷峰塔'）。可选，不传=全部事件",
            },
            "chapter_numbers": {
                "type": "array",
                "items": {"type": "integer"},
                "description": "限定章节范围。空=全部章节",
            },
            "categories": {
                "type": "array",
                "items": {"type": "string"},
                "description": "限定标注类别（person/location/event/term/motif）。空=全部类别",
            },
            "include_excerpts": {
                "type": "boolean",
                "description": "是否在每条事件中包含原文摘录（默认 true）",
            },
        },
    },
}


async def handler(entity: str = "",
                  chapter_numbers: list = None,
                  categories: list = None,
                  include_excerpts: bool = True,
                  node_client=None) -> dict:
    """MCP handler: 构建时间线。"""
    if node_client is None:
        return {"error": "Node API 客户端未初始化"}

    # ① 获取章节清单
    try:
        chapters = await node_client.get_literature()
    except Exception as ex:
        return {"error": f"获取文献失败: {ex}"}

    target_chapters = set(chapter_numbers) if chapter_numbers else None
    category_filter = set(categories) if categories else None

    events: list[dict] = []
    dynasty_events: dict[str, list] = {d: [] for d in DYNASTY_ORDER}

    for ch in chapters:
        ch_num = ch.get("chapter_number", 0)
        if target_chapters and ch_num not in target_chapters:
            continue
        dynasty = _infer_dynasty(ch_num)
        ch_title = ch.get("title", "")
        content = ch.get("content", "")
        if not content:
            continue

        # 分段
        content_norm = re.sub(r"\r\n|\r", "\n", content)
        paragraphs = [p.strip() for p in content_norm.split("\n\n") if p.strip()]

        # ② 拉标注（user + agent）
        try:
            user_anns = await node_client.get_user_annotations(str(ch_num)) or {}
        except Exception:
            user_anns = {}
        try:
            agent_anns = await node_client.get_agent_annotations(str(ch_num)) or {}
        except Exception:
            agent_anns = {}

        merged: dict[str, list] = {}
        for d in (user_anns, agent_anns):
            if not isinstance(d, dict):
                continue
            for k, v in d.items():
                merged.setdefault(k, []).extend(v if isinstance(v, list) else [])

        # ③ 每条标注 → 事件
        for key, ann_list in merged.items():
            parts = key.split(":", 2)
            if len(parts) < 2:
                continue
            try:
                p_idx = int(parts[1])
            except (ValueError, TypeError):
                continue
            if not (0 <= p_idx < len(paragraphs)):
                continue
            para_text = paragraphs[p_idx]

            for ann in (ann_list or []):
                if not isinstance(ann, dict):
                    continue
                ann_entity = ann.get("entity", "")
                ann_category = ann.get("category", "")

                # 实体过滤
                if entity and entity not in ann_entity and ann_entity not in entity:
                    continue
                # 类别过滤
                if category_filter and ann_category not in category_filter:
                    continue

                # 摘录
                excerpt = None
                if include_excerpts:
                    sc = ann.get("start_char", 0)
                    ec = ann.get("end_char", 0)
                    if sc and ec and sc < ec <= len(para_text):
                        excerpt = para_text[sc:ec]
                    else:
                        # 没有精确 span → 取实体本身
                        excerpt = ann_entity
                    if len(excerpt) > 200:
                        excerpt = excerpt[:200] + "..."

                event = {
                    "dynasty": dynasty,
                    "dynastyOrder": DYNASTY_ORDER.get(dynasty, 0),
                    "yearRange": DYNASTY_YEAR_RANGE.get(dynasty, ("", "")),
                    "entity": ann_entity,
                    "category": ann_category,
                    "explanation": ann.get("explanation", ""),
                    "chapterNumber": ch_num,
                    "chapterTitle": ch_title,
                    "paragraphIndex": p_idx,
                    "excerpt": excerpt,
                    "source": ann.get("source", "agent"),
                }
                events.append(event)
                dynasty_events[dynasty].append(event)

    # ④ 按朝代排序 + 去重（同实体+同段落+同类别只保留一条）
    seen_keys = set()
    deduped = []
    for ev in sorted(events, key=lambda e: (e["dynastyOrder"], e["chapterNumber"], e["paragraphIndex"])):
        key = (ev["entity"], ev["chapterNumber"], ev["paragraphIndex"], ev["category"])
        if key not in seen_keys:
            seen_keys.add(key)
            deduped.append(ev)

    # ⑤ 朝代汇总
    dynasty_summary = []
    for dynasty in ["唐", "宋", "明", "清", "现当代"]:
        evs = dynasty_events.get(dynasty, [])
        if evs:
            entities_set = set(e["entity"] for e in evs)
            dynasty_summary.append({
                "dynasty": dynasty,
                "dynastyOrder": DYNASTY_ORDER[dynasty],
                "yearRange": DYNASTY_YEAR_RANGE[dynasty],
                "eventCount": len(evs),
                "uniqueEntities": len(entities_set),
                "topEntities": sorted(entities_set, key=lambda x: -sum(1 for e in evs if e["entity"] == x))[:10],
            })

    return {
        "timeline": deduped,
        "totalEvents": len(deduped),
        "dynastySummary": dynasty_summary,
        "filters": {
            "entity": entity or "(all)",
            "chapters": sorted(target_chapters) if target_chapters else "(all)",
            "categories": sorted(category_filter) if category_filter else "(all)",
        },
    }
