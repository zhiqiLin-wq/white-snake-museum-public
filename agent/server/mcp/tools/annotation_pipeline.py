"""标注流水线：两轮标注（Discovery + Precise Resolution）。

阶段 1 新建。与旧 annotate_passage.py 并行存在，零冲突。

Pass 1 — Discovery（发现阶段）:
  discovery_pass() → 滑动窗口扫描 → 候选实体列表

Pass 2 — Precise Resolution（精确定位阶段）:
  precise_resolution_pass() → 逐段用代码确定性定位（隔离 LLM 坐标幻觉）→ async generator

导出:
  discovery_pass, precise_resolution_pass
"""

import asyncio
import json
import logging
from typing import List, Dict, Optional, AsyncGenerator

from ...langgraph.state import ParagraphInfo
from ...llm.json_utils import parse_llm_json
from .span_validator import validate_span
from .entity_locator import locate_entity_all
from ...utils.window_generator import sliding_windows
from ...utils.async_executor import async_run_with_semaphore

logger = logging.getLogger(__name__)

# 已知实体知识库（从共享模块导入，非旧 annotate_passage）
from .annotation_knowledge_base import (
    _ENTITY_KB, DEFAULT_CATEGORY_COLORS, _CATEGORY_REVERSE_MAP
)


# === 工具函数 ===

def _resolve_category(raw_category: str) -> str:
    """将 LLM 返回的中文类别标签转为英文 key。"""
    return _CATEGORY_REVERSE_MAP.get(raw_category, raw_category)


# === Pass 1: Discovery ===

def _build_discovery_prompt(
    window_paragraphs: List[ParagraphInfo],
    categories: List[str],
    prompt_registry,
) -> tuple[str, str]:
    """构建 Discovery prompt 的 system 和 user 文本。"""
    # 构建类别信息
    cat_infos = []
    cat_labels = []
    for cat_key in categories:
        kb = _ENTITY_KB.get(cat_key, {})
        cat_infos.append({
            "key": cat_key,
            "label": kb.get("label", cat_key),
            "rules": kb.get("rules", "标注所有相关实体"),
            "known_entities": kb.get("known", []),
        })
        cat_labels.append(kb.get("label", cat_key))

    system, user = prompt_registry.render("annotation_discovery", {
        "categories": cat_infos,
        "paragraphs": [
            {"index": p["index"], "text": p["text"]}
            for p in window_paragraphs
        ],
        "category_labels": cat_labels,
    })
    return system, user


async def _scan_window(
    window_paragraphs: List[ParagraphInfo],
    categories: List[str],
    llm,
    prompt_registry,
) -> List[dict]:
    """扫描单个窗口，调用快速 LLM 发现候选实体。"""
    from ...config import settings

    system, user = _build_discovery_prompt(window_paragraphs, categories, prompt_registry)

    resp = await llm.generate(
        system, user,
        model=settings.effective_fast_model,
        max_tokens=settings.annotation_discovery_max_tokens,
        temperature=0.1,
        thinking_disabled=True,
    )

    try:
        data = parse_llm_json(resp.content)
    except json.JSONDecodeError as e:
        logger.error(
            "[P1-06] _scan_window JSON parse failed for window starting at para %d: %s",
            window_paragraphs[0]["index"], e
        )
        return []

    discoveries = data.get("discoveries", [])
    # 校验每个 discovery 的必填字段
    valid = []
    for d in discoveries:
        if not isinstance(d, dict):
            continue
        if not all(k in d for k in ("entity", "category", "paragraph_index")):
            logger.warning("[P1-06] discovery missing required fields: %s", d)
            continue
        # 规范化 category
        d["category"] = _resolve_category(d["category"])
        if d["category"] not in categories:
            d["category"] = "custom"
        valid.append(d)

    # 异常检测：单个窗口超过 200 条发现为异常，截断
    if len(valid) > 200:
        logger.warning(
            "[P1-06] _scan_window: excessive discoveries (%d), truncating to 200",
            len(valid)
        )
        valid = valid[:200]

    return valid


def _deduplicate_discoveries(discoveries: List[dict]) -> List[dict]:
    """去重合并跨窗口的重复发现。

    去重键: (entity.strip(), paragraph_index, category)
    合并策略: context_hint 和 explanation 保留更长的版本。
    """
    seen: Dict[tuple, dict] = {}
    for d in discoveries:
        key = (d["entity"].strip(), d["paragraph_index"], d["category"])
        if key in seen:
            existing = seen[key]
            # 合并 context_hint: 保留更长的
            if len(d.get("context_hint", "")) > len(existing.get("context_hint", "")):
                existing["context_hint"] = d["context_hint"]
            # 合并 explanation: 保留更长的
            if len(d.get("explanation", "")) > len(existing.get("explanation", "")):
                existing["explanation"] = d["explanation"]
            # 合并 summary(概括词): 保留更长的
            if len(d.get("summary", "")) > len(existing.get("summary", "")):
                existing["summary"] = d["summary"]
        else:
            seen[key] = dict(d)  # 浅拷贝

    result = list(seen.values())
    result.sort(key=lambda x: (x["paragraph_index"], x["entity"]))
    return result


async def discovery_pass(
    paragraphs: List[ParagraphInfo],
    categories: List[str],
    llm,
    prompt_registry,
    concurrency: int = 5,
    window_size: int = 3,
    overlap: int = 1,
    progress_callback=None,
) -> List[dict]:
    """Pass 1: 滑动窗口全面扫描，发现所有候选实体。

    Args:
        paragraphs: 前端传入的段落列表 [{"index": int, "text": str}]
        categories: 标注类别列表，如 ["person", "location"]
        llm: LLM 客户端（需支持 generate(system, user, model, max_tokens, temperature)）
        prompt_registry: PromptRegistry 实例
        concurrency: 并发窗口数
        window_size: 每个窗口包含的段落数
        overlap: 窗口间重叠段落数
        progress_callback: 可选 async 回调 (done: int, total: int, found: int) -> None，
            每个窗口完成后调用，传入结构化进度（已完成窗口数/总窗口数/累计候选数）

    Returns:
        discoveries: [
            {"entity": str, "category": str, "paragraph_index": int,
             "context_hint": str, "explanation": str}
        ]

    Raises:
        ValueError: paragraphs 为空 或 categories 为空
    """
    # 输入验证 (fast fail)
    if not paragraphs:
        raise ValueError("paragraphs must not be empty")
    if not categories:
        raise ValueError("categories must not be empty")

    # 生成滑动窗口
    windows = sliding_windows(paragraphs, window_size, overlap)
    logger.info(
        "[P1-06] discovery_pass: %d paragraphs -> %d windows "
        "(window_size=%d, overlap=%d, concurrency=%d)",
        len(paragraphs), len(windows), window_size, overlap, concurrency
    )

    # B2: 自建信号量 + as_completed，支持逐窗口进度回调
    _sem = asyncio.Semaphore(concurrency)

    async def _task(window: dict, idx: int) -> List[dict]:
        async with _sem:
            return await _scan_window(
                window["paragraphs"], categories, llm, prompt_registry
            )

    _pending = [asyncio.create_task(_task(w, i)) for i, w in enumerate(windows)]
    raw_results = [None] * len(windows)
    _done_count = 0
    _total_windows = len(windows)
    _total_discoveries = 0

    for coro in asyncio.as_completed(_pending):
        result = await coro
        _done_count += 1
        if result is not None:
            _total_discoveries += len(result)
        if progress_callback:
            await progress_callback(_done_count, _total_windows, _total_discoveries)

    # 收集结果（按原始顺序）
    for i, t in enumerate(_pending):
        try:
            raw_results[i] = t.result()
        except Exception:
            raw_results[i] = None

    # 合并所有窗口的结果
    all_discoveries = []
    failed_windows = 0
    for i, r in enumerate(raw_results):
        if r is None:
            failed_windows += 1
            logger.error("[P1-06] window %d failed (returned None)", i)
            continue
        all_discoveries.extend(r)

    if failed_windows > 0:
        logger.warning(
            "[P1-06] discovery_pass: %d/%d windows failed",
            failed_windows, len(windows)
        )

    # 去重
    deduped = _deduplicate_discoveries(all_discoveries)

    dedup_rate = 1.0 - len(deduped) / max(1, len(all_discoveries))
    if dedup_rate > 0.5:
        logger.warning(
            "[P1-06] discovery_pass: high dedup rate=%.1f%% "
            "before=%d after=%d",
            dedup_rate * 100, len(all_discoveries), len(deduped)
        )

    logger.info(
        "[P1-06] discovery_pass complete: %d discoveries (%d unique) "
        "from %d paragraphs",
        len(all_discoveries), len(deduped), len(paragraphs)
    )

    return deduped


# === Pass 2: Precise Resolution ===

def _deduplicate_annotations(annotations: List[dict]) -> List[dict]:
    """去重：同一 (entity, start_char, end_char, paragraph_index) 保留 confidence 最高的。"""
    confidence_order = {"high": 4, "medium": 3, "low": 2, "not_found": 1}

    groups: Dict[tuple, List[dict]] = {}
    for ann in annotations:
        key = (
            ann["entity"].strip(),
            ann["start_char"],
            ann["end_char"],
            ann.get("paragraph_index", 0),
        )
        if key not in groups:
            groups[key] = []
        groups[key].append(ann)

    result = []
    for key, group in groups.items():
        best = max(
            group,
            key=lambda a: confidence_order.get(a.get("confidence", "low"), 0)
        )
        result.append(best)

    return result


def _rule_engine_match(
    paragraph_text: str,
    paragraph_index: int,
    discoveries: List[dict],
) -> tuple[List[dict], List[dict]]:
    """规则引擎预匹配已知实体。

    对 _ENTITY_KB 中 person 和 location 类别的 known 列表中的实体，
    用 str.find 循环匹配段落中所有出现位置。长实体优先匹配，避免短实体误匹配子串。

    Returns:
        (matched, remaining):
            matched: 已匹配的标注列表（含 start_char/end_char, source="rule"）
            remaining: 未匹配的 discoveries（需送 LLM 进一步处理）
    """
    # 收集白名单实体（仅 person 和 location）
    entities_to_match: List[tuple[str, str]] = []  # [(entity, category)]
    for d in discoveries:
        cat = d["category"]
        if cat not in ("person", "location"):
            continue
        kb = _ENTITY_KB.get(cat, {})
        known = kb.get("known", [])
        if d["entity"] in known:
            entities_to_match.append((d["entity"], cat))

    # 长实体优先（避免"西子湖"被"西湖"误匹配）
    entities_to_match.sort(key=lambda x: -len(x[0]))

    # 逐段扫描
    matched = []
    matched_ranges: List[tuple[int, int]] = []
    matched_entities: set[str] = set()

    for entity, cat in entities_to_match:
        search_start = 0
        while True:
            pos = paragraph_text.find(entity, search_start)
            if pos < 0:
                break
            entity_end = pos + len(entity)

            # 检查是否与已匹配区间重叠
            overlaps = any(
                pos < m_end and entity_end > m_start
                for m_start, m_end in matched_ranges
            )
            if not overlaps:
                matched_ranges.append((pos, entity_end))
                matched.append({
                    "entity": entity,
                    "category": cat,
                    "start_char": pos,
                    "end_char": entity_end,
                    "paragraph_index": paragraph_index,
                    "explanation": f"{_ENTITY_KB[cat]['label']}（规则匹配）",
                    "confidence": "high",
                    "color": DEFAULT_CATEGORY_COLORS.get(cat, "#B8B878"),
                    "source": "agent",
                })
                matched_entities.add(entity)

            search_start = pos + 1  # 继续搜索下一个出现

    # 未被规则匹配的 discoveries 交给 LLM
    remaining = [
        d for d in discoveries
        if d["entity"] not in matched_entities
    ]

    return matched, remaining


async def precise_resolution_pass(
    paragraphs: List[ParagraphInfo],
    discoveries: List[dict],
    llm,
    prompt_registry,
    concurrency: int = 8,
    coverage_tracker=None,
) -> AsyncGenerator[dict, None]:
    """Pass 2: 逐段精确定位，async generator。

    每完成一段就 yield 该段结果，使得调用方可以流式发送 SSE。
    段落级并发由 asyncio.Semaphore 控制。
    """
    if not discoveries:
        logger.info("[P1-07] precise_resolution_pass: no discoveries, skipping")
        return

    # 按 paragraph_index 分组 discoveries
    by_paragraph: Dict[int, List[dict]] = {}
    for d in discoveries:
        p_idx = d["paragraph_index"]
        if p_idx not in by_paragraph:
            by_paragraph[p_idx] = []
        by_paragraph[p_idx].append(d)

    # 将 paragraphs 转为 dict 用于快速查找
    para_map: Dict[int, ParagraphInfo] = {
        p["index"]: p for p in paragraphs
    }

    # 按 paragraph_index 排序
    sorted_indices = sorted(by_paragraph.keys())

    # 信号量控制的段落级并发处理
    semaphore = asyncio.Semaphore(concurrency)

    async def _process_one_paragraph(p_idx: int) -> dict | None:
        """处理单个段落：规则预匹配 + LLM 精确定位 + 验证 + 去重。"""
        async with semaphore:
            para = para_map.get(p_idx)
            if para is None:
                logger.warning(
                    "[P1-07] paragraph %d not found in paragraph list, skipping", p_idx
                )
                return None

            para_discs = by_paragraph[p_idx]
            result_annotations = []

            # Step 1: 规则引擎预匹配
            rule_anns, remaining = _rule_engine_match(
                para["text"], p_idx, para_discs
            )
            # validate rule annotations
            for ann in rule_anns:
                vr = validate_span(para["text"], ann["entity"],
                                   ann["start_char"], ann["end_char"])
                if vr.status != "discarded":
                    ann["start_char"] = vr.start_char
                    ann["end_char"] = vr.end_char
                    ann["confidence"] = vr.confidence
                    result_annotations.append(ann)

            # Step 2: 剩余 discoveries 用代码确定性定位（隔离 LLM 坐标幻觉）
            if remaining:
                color_map = DEFAULT_CATEGORY_COLORS
                for disc in remaining:
                    entity = (disc.get("entity") or "").strip()
                    if not entity:
                        continue
                    cat = _resolve_category(disc.get("category", "custom"))

                    # 坐标由代码定位，忽略 discovery 里可能夹带的坐标信息。
                    # 同一实体多次出现时每个出现都生成一条标注。
                    matches = locate_entity_all(para["text"], entity)
                    for vr in matches:
                        if vr.status == "discarded":
                            continue
                        result_annotations.append({
                            "entity": entity,
                            "category": cat,
                            "start_char": vr.start_char,
                            "end_char": vr.end_char,
                            "paragraph_index": p_idx,
                            "confidence": vr.confidence,
                            "explanation": disc.get("explanation", ""),
                            "summary": disc.get("summary", ""),
                            "color": color_map.get(cat, color_map.get("custom", "#B8B878")),
                            "source": "agent",
                        })

            # Step 3: 去重
            result_annotations = _deduplicate_annotations(result_annotations)

            return {
                "paragraph_index": p_idx,
                "annotations": result_annotations,
            }

    # 并行启动所有段落任务
    pending = set()
    for i in sorted_indices:
        pending.add(asyncio.ensure_future(_process_one_paragraph(i)))

    # 逐个 yield 完成的结果（保持流式特性，先完成先发出）
    while pending:
        done, pending = await asyncio.wait(
            pending, return_when=asyncio.FIRST_COMPLETED
        )
        for task in done:
            result = task.result()
            if result is None:
                continue
            p_idx = result["paragraph_index"]
            result_annotations = result["annotations"]

            # Step 4: 序列化更新覆盖率（asyncio 事件循环单线程安全）
            if coverage_tracker is not None:
                coverage_tracker.record(p_idx, result_annotations)

            # Step 5: yield 本段结果（前端实时流式渲染）
            yield {
                "paragraph_index": p_idx,
                "annotations": result_annotations,
                "coverage": coverage_tracker.snapshot() if coverage_tracker else None,
            }

    logger.info(
        "[P1-07] precise_resolution_pass complete: %d paragraphs processed "
        "with concurrency=%d",
        len(sorted_indices), concurrency
    )
