"""CoverageTracker — 标注覆盖率追踪。

阶段 1 — P1-09.

⚠️ 同时作为 MCP 工具 `coverage_tracker` 暴露给 LLM：
  - 输入: chapter_numbers（章节号数组，缺省=全量）+ group_by（chapter/category/dynasty）
  - 输出: 覆盖率（段落标注覆盖率、分类数量分布、未标注段落索引）+ 质量评级（excellent/good/fair/poor）
  - 典型场景: "第五章标注覆盖率多少？" "哪些章节没标注人物？" "展示标注完整度。"
"""
import logging
from typing import List, Dict

logger = logging.getLogger(__name__)


class CoverageTracker:
    """追踪标注覆盖率。"""

    def __init__(self, total_paragraphs: int, categories: List[str]):
        if total_paragraphs <= 0:
            raise ValueError(
                f"total_paragraphs must be > 0, got {total_paragraphs}"
            )
        self.total_paragraphs = total_paragraphs
        self.categories = categories
        self._annotated_paragraphs: set[int] = set()
        self._entity_counts: Dict[str, int] = {cat: 0 for cat in categories}
        self._total_entities = 0

    def record(self, paragraph_index: int, annotations: List[dict]):
        """记录一个段落的标注结果。"""
        if paragraph_index >= self.total_paragraphs:
            raise IndexError(
                f"paragraph_index {paragraph_index} >= total {self.total_paragraphs}"
            )
        self._annotated_paragraphs.add(paragraph_index)
        for ann in annotations:
            cat = ann.get("category", "custom")
            self._entity_counts[cat] = self._entity_counts.get(cat, 0) + 1
            self._total_entities += 1

    def snapshot(self) -> dict:
        """返回当前覆盖率快照。"""
        annotated = len(self._annotated_paragraphs)
        return {
            "paragraphsTotal": self.total_paragraphs,
            "paragraphsAnnotated": annotated,
            "coverageRate": round(annotated / self.total_paragraphs, 3),
            "entitiesByCategory": dict(self._entity_counts),
            "totalEntities": self._total_entities,
            "quality": self._assess_quality(),
        }

    def get_unannotated(self) -> List[int]:
        """返回尚未标注的段落索引列表。"""
        return sorted(
            i for i in range(self.total_paragraphs)
            if i not in self._annotated_paragraphs
        )

    def _assess_quality(self) -> str:
        from ...config import settings
        coverage = len(self._annotated_paragraphs) / self.total_paragraphs
        if coverage >= settings.annotation_quality_coverage_excellent:
            return "excellent"
        elif coverage >= settings.annotation_quality_coverage_good:
            return "good"
        elif coverage >= settings.annotation_quality_coverage_fair:
            return "fair"
        else:
            return "poor"


# ============================================================================
#  MCP 工具封装：coverage_tracker
# ============================================================================

TOOL_DEF = {
    "name": "coverage_tracker",
    "description": (
        "查询标注覆盖率统计。输入章节列表（空或省略=全部章节）和类别筛选（可选）。"
        "返回：总段落/已标注段落/覆盖率、按类别实体分布、未标注段落索引、"
        "综合质量评级 excellent/good/fair/poor（由 config.annotation_quality_coverage_* 阈值控制）。"
        "典型使用场景：用户问'标注覆盖率多少''第3章标注完了吗''还有哪些段没标''标签完整度如何'。"
    ),
    "inputSchema": {
        "type": "object",
        "properties": {
            "chapter_numbers": {
                "type": "array",
                "items": {"type": "integer"},
                "description": "要统计的章节号列表。不传或空数组=统计全部章节",
            },
            "categories": {
                "type": "array",
                "items": {"type": "string"},
                "description": "可选：只计算指定类别（person/location/event/term/motif/custom）。空=含全部",
            },
            "group_by": {
                "type": "string",
                "enum": ["chapter", "category", "dynasty"],
                "description": "分组维度: chapter=按章节, category=按标签类别, dynasty=按朝代。默认 chapter。",
            },
            "include_unannotated": {
                "type": "boolean",
                "description": "是否在返回中包含未标注段落索引数组（章节多时可能较长）。默认 true。",
            },
        },
    },
}


async def handler(chapter_numbers: list = None,
                  categories: list = None,
                  group_by: str = "chapter",
                  include_unannotated: bool = True,
                  node_client=None) -> dict:
    """MCP handler: 计算标注覆盖率与质量评级。"""
    if node_client is None:
        return {"error": "Node API 客户端未初始化"}

    # ① 获取章节清单
    chapters_resp = await node_client.get_chapters()
    chapters = []
    if isinstance(chapters_resp, list):
        for c in chapters_resp:
            cn = c.get("number") or c.get("chapterNumber") or c.get("id")
            if isinstance(cn, int):
                chapters.append({"number": cn,
                                 "title": c.get("title") or c.get("chapterTitle") or "",
                                 "dynasty": c.get("dynasty") or ""})
    if not chapters:
        return {"error": "无法获取章节清单"}

    target_numbers = set(int(x) for x in (chapter_numbers or [])) if chapter_numbers else None
    if target_numbers:
        chapters = [c for c in chapters if c["number"] in target_numbers]
    if not chapters:
        return {"error": "指定章节均不存在"}

    # ② 每章: 拉取文献原文 → 段落数 + 拉取标注 → 计算
    category_filter = set(categories) if categories else None
    per_chapter = []
    aggregate_para_total = 0
    aggregate_para_annotated = 0
    aggregate_entity_by_category: Dict[str, int] = {}
    by_dynasty: Dict[str, dict] = {}
    by_category_overall: Dict[str, int] = {}

    for ch in chapters:
        literature = await node_client.get_literature_by_chapter(ch["number"])
        if literature is None:
            continue
        import re as _re
        content = _re.sub(r"\r\n|\r", "\n", literature.get("content", ""))
        paragraphs = [p.strip() for p in content.split("\n\n") if p.strip()]
        total_paragraphs = len(paragraphs)
        if total_paragraphs == 0:
            continue

        tracker = CoverageTracker(total_paragraphs, list(
            {"person", "location", "event", "term", "motif", "custom"}
        ))
        # 拿到该章的 user + agent 标注（通过 passage_key: chapter:idx 匹配）
        anns_by_passage: Dict[int, list] = {}
        try:
            user_anns = await node_client.get_user_annotations(str(ch["number"])) or {}
        except Exception:
            user_anns = {}
        try:
            agent_anns = await node_client.get_agent_annotations(str(ch["number"])) or {}
        except Exception:
            agent_anns = {}
        merged: Dict[str, list] = {}
        for d in (user_anns, agent_anns):
            if not isinstance(d, dict):
                continue
            for k, v in d.items():
                merged.setdefault(k, []).extend(v if isinstance(v, list) else [])

        for key, ann_list in merged.items():
            # key 形如 "5:3" 或 "5:3:left"
            parts = key.split(":", 2)
            if len(parts) < 2:
                continue
            try:
                ch_n = int(parts[0])
                idx = int(parts[1])
            except (ValueError, TypeError):
                continue
            if ch_n != ch["number"]:
                continue
            if not (0 <= idx < total_paragraphs):
                continue
            filtered_list = []
            for a in (ann_list or []):
                if not isinstance(a, dict):
                    continue
                if category_filter and a.get("category") not in category_filter:
                    continue
                filtered_list.append(a)
            if filtered_list:
                anns_by_passage.setdefault(idx, []).extend(filtered_list)
        for idx, anns in anns_by_passage.items():
            tracker.record(idx, anns)
            for a in anns:
                cat = a.get("category") or "custom"
                aggregate_entity_by_category[cat] = aggregate_entity_by_category.get(cat, 0) + 1
                by_category_overall[cat] = by_category_overall.get(cat, 0) + 1

        snap = tracker.snapshot()
        aggregate_para_total += total_paragraphs
        aggregate_para_annotated += snap["paragraphsAnnotated"]

        dyn = ch.get("dynasty") or "未知"
        d = by_dynasty.setdefault(dyn, {
            "paragraphsTotal": 0, "paragraphsAnnotated": 0,
            "entitiesByCategory": {}, "chapters": [],
        })
        d["paragraphsTotal"] += total_paragraphs
        d["paragraphsAnnotated"] += snap["paragraphsAnnotated"]
        for k, v in (snap.get("entitiesByCategory") or {}).items():
            if v:
                d["entitiesByCategory"][k] = d["entitiesByCategory"].get(k, 0) + v
        if ch["title"]:
            d["chapters"].append({"number": ch["number"], "title": ch["title"]})

        entry = {
            "chapterNumber": ch["number"],
            "chapterTitle": ch["title"],
            "dynasty": dyn,
            **snap,
        }
        if include_unannotated:
            entry["unannotatedParagraphs"] = tracker.get_unannotated()
        per_chapter.append(entry)

    overall_coverage = (round(aggregate_para_annotated / aggregate_para_total, 4)
                        if aggregate_para_total else 0)

    def _rate(cov: float) -> str:
        from ...config import settings
        if cov >= settings.annotation_quality_coverage_excellent:
            return "excellent"
        if cov >= settings.annotation_quality_coverage_good:
            return "good"
        if cov >= settings.annotation_quality_coverage_fair:
            return "fair"
        return "poor"

    # 按分组维度返回
    groups: dict
    if group_by == "category":
        groups = {"byCategory": by_category_overall}
    elif group_by == "dynasty":
        for d in by_dynasty.values():
            t = d["paragraphsTotal"]
            cov = round(d["paragraphsAnnotated"] / t, 4) if t else 0
            d["coverageRate"] = cov
            d["quality"] = _rate(cov)
        groups = {"byDynasty": by_dynasty}
    else:  # chapter
        groups = {"byChapter": per_chapter}

    # 整理: 标注覆盖率最低的 Top10 章节（提示用户优先补标）
    # 按覆盖率升序排列，同时输出未标注段绝对数供参考
    need_attention = sorted(
        per_chapter,
        key=lambda c: (
            c.get("paragraphsAnnotated", 0) / max(1, c.get("paragraphsTotal", 1)),
            -(c.get("paragraphsTotal", 0) - c.get("paragraphsAnnotated", 0)),  # 同比例时未标注多的优先
        ),
    )[:10]

    return {
        "summary": {
            "chapters": len(per_chapter),
            "paragraphsTotal": aggregate_para_total,
            "paragraphsAnnotated": aggregate_para_annotated,
            "coverageRate": overall_coverage,
            "quality": _rate(overall_coverage),
            "entitiesByCategory": aggregate_entity_by_category,
            "categoryFilterApplied": sorted(category_filter) if category_filter else None,
        },
        "needAttentionChapters": [
            {
                "chapterNumber": c["chapterNumber"],
                "chapterTitle": c["chapterTitle"],
                "dynasty": c.get("dynasty"),
                "coverageRate": c.get("coverageRate"),
                "quality": c.get("quality"),
            }
            for c in need_attention
        ],
        **groups,
    }
