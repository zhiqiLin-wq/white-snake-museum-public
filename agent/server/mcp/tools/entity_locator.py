"""统一确定性定位器 — 标注坐标由代码计算，隔离 LLM。

定位策略（与 span_validator 的 6 级回退链保持一致，此处收敛为唯一入口）:
  Level 1: 精确匹配 (high)     — text[start:end] == entity
  Level 2: 窗口搜索 (medium)   — 候选偏移 ±50 字符内 text.find(entity)
  Level 3: 全文搜索 (low)      — text.find(entity)
  Level 4: 紧凑匹配 (low)      — 去空白后 find，再映射回原始坐标
  Level 5: 模糊匹配 (low)      — difflib 滑窗，ratio >= 0.7
  Level 6: 丢弃 (not_found)

LLM 只负责"发现实体名 + 类别"，不再负责 start/end。定位由本模块确定性地完成。
"""

import logging
from typing import List, Tuple

from .span_validator import validate_span, SpanValidationResult

logger = logging.getLogger(__name__)


def locate_entity(
    text: str,
    entity: str,
    hint_start: int | None = None,
    hint_end: int | None = None,
) -> SpanValidationResult:
    """在 text 中定位 entity 的一次出现，返回校验结果。

    hint_start/hint_end 为可选候选偏移（如 LLM 仍给了坐标时作为提示），
    但仅作为窗口搜索的起点，不作为最终坐标的信任来源。
    """
    entity = (entity or "").strip()
    if not entity:
        return SpanValidationResult("discarded", -1, -1, "not_found")

    start = hint_start if isinstance(hint_start, int) else 0
    end = hint_end if isinstance(hint_end, int) else (start + len(entity))

    # 若没有候选偏移，直接走全文搜索，避免窗口/紧凑依赖无效起点
    if hint_start is None and hint_end is None:
        vr = validate_span(text, entity, 0, len(entity))
        if vr.status != "discarded":
            return vr
        # 空偏移给出的精确匹配失败时，validate_span 内部已做窗口/全文/紧凑/模糊回退
        return vr

    return validate_span(text, entity, start, end)


def locate_entity_all(
    text: str,
    entity: str,
    exclude_ranges: List[Tuple[int, int]] | None = None,
) -> List[SpanValidationResult]:
    """在 text 中定位 entity 的全部出现（不重叠），跳过 exclude_ranges。

    用于同一实体在一段/一批里出现多次的场景，保证每个出现都生成一条标注。
    """
    entity = (entity or "").strip()
    if not entity:
        return []

    excluded = list(exclude_ranges or [])
    results: List[SpanValidationResult] = []
    search_start = 0

    while True:
        pos = text.find(entity, search_start)
        if pos < 0:
            break
        end = pos + len(entity)
        # 跳过落在排除区间内的匹配
        if any(pos < ex_end and end > ex_start for ex_start, ex_end in excluded):
            search_start = pos + 1
            continue
        results.append(SpanValidationResult("valid", pos, end, "high"))
        search_start = pos + 1

    # 精确匹配为空时，才走模糊回退（否则模糊可能误标同义/近似词）
    if not results:
        vr = locate_entity(text, entity)
        if vr.status != "discarded":
            results.append(vr)

    return results
