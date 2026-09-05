"""B-013: annotate_passage — 对指定段落进行实体识别与分类标注。

关键: 字符偏移对齐（Python str 索引 = JavaScript charAt 索引）。
标注前不对原文做任何预处理。使用 str.find() 定位实体。

分批处理: 将目标段落分为 8 段一批，避免单次 LLM 调用 JSON 输出超长被截断。
每批独立调用 LLM，偏移量统一调整到全文坐标系。
"""
import json
import logging
import re

from .annotation_knowledge_base import (
    _ENTITY_KB, DEFAULT_CATEGORY_COLORS, _CATEGORY_REVERSE_MAP
)
from .entity_locator import locate_entity_all
from ...llm.json_utils import parse_llm_json

logger = logging.getLogger(__name__)

# 段落切分规则：与前端 sourceTreeStore.splitParagraphs 保持一致（空行分隔）
# 先规范化 \r，再按 \n\n 切分，确保前后端段落索引完全一致
_PARAGRAPH_SEP = re.compile(r'\n\n')
_SENT_END_RE = re.compile(r'[。！？!?\n]')


def _split_paragraphs(text: str) -> list[str]:
    """按空行切分段落，规范化行尾以对齐前端 splitParagraphs。"""
    normalized = re.sub(r'\r\n|\r', '\n', text)
    return [p.strip() for p in _PARAGRAPH_SEP.split(normalized) if p.strip()]


def _split_paragraph_into_chunks(text: str,
                                 target_max: int = 800,
                                 target_min: int = 500,
                                 overlap_sentences: int = 3,
                                 overlap_chars: int = 150) -> list[tuple[int, int]]:
    """把超长段落切成块区间 [start, end)（原段落纯切片，不 trim、不丢字符）。

    - 段落 <= target_max：不切，返回 [(0, len(text))]。
    - 有标点（句号/问号/叹号/换行）：按句切，目标 target_min~target_max 字，
      落刀在句子边界，相邻块重叠 overlap_sentences 句。
    - 无标点：按字数硬切，相邻块重叠 overlap_chars 字。

    关键：start 就是块在段落里的精确字符偏移，块 text = text[start:end]，
    段落级坐标 = start + 块内坐标，纯加法映射，避免错位。
    """
    if len(text) <= target_max:
        return [(0, len(text))]

    if _SENT_END_RE.search(text):
        return _split_by_sentence(text, target_max, overlap_sentences)
    return _split_by_chars(text, target_max, overlap_chars)


def _split_by_sentence(text: str, target_max: int, overlap_sentences: int) -> list[tuple[int, int]]:
    edges = [0]
    for m in _SENT_END_RE.finditer(text):
        edges.append(m.end())
    if edges[-1] != len(text):
        edges.append(len(text))
    edges = sorted(set(edges))

    chunks = []
    i = 0
    n = len(edges)
    while i < n - 1:
        start = edges[i]
        j = i
        while j < n - 1:
            j += 1
            if edges[j] - start >= target_max or j == n - 1:
                break
        end = edges[j]
        chunks.append((start, end))
        if j >= n - 1:
            break
        next_i = max(j - overlap_sentences, i + 1)
        if edges[next_i] >= end:
            next_i = j
        i = next_i
    return chunks


def _split_by_chars(text: str, target_max: int, overlap_chars: int) -> list[tuple[int, int]]:
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + target_max, len(text))
        chunks.append((start, end))
        if end >= len(text):
            break
        start = end - overlap_chars
    return chunks

TOOL_DEF = {
    "name": "annotate_passage",
    "description": (
        "对指定段落直接执行实体识别和分类标注（人物/地点/事件/术语/母题/自定义），"
        "返回标注数据供前端 TextReader 高亮渲染。"
        "支持颜色覆盖和跳过已有标注以避免重复。"
        "用户的自然语言标注请求（如'把这段里的人物都标出来'）请先经 annotate_user_request "
        "解析意图后再由其调用本工具；用户已明确给出类别与段落范围时可直接调用。"
    ),
    "inputSchema": {
        "type": "object",
        "properties": {
            "chapter_number": {
                "type": "integer",
                "description": "章节编号",
            },
            "paragraph_range": {
                "type": "array",
                "items": {"type": "integer"},
                "minItems": 2,
                "maxItems": 2,
                "description": "起止段落索引 [start, end]",
            },
            "categories": {
                "type": "array",
                "items": {"type": "string"},
                "description": "标注类别: person/location/event/term/motif/custom，默认全部",
            },
            "color_overrides": {
                "type": "object",
                "description": "类别→hex颜色映射，覆盖默认颜色，例如 {\"person\": \"#FF0000\"}",
            },
        },
        "required": ["chapter_number", "paragraph_range"],
    },
}


# _ENTITY_KB, DEFAULT_CATEGORY_COLORS, _CATEGORY_REVERSE_MAP
# 已提取到 annotation_knowledge_base.py (阶段 0 — P0-05)

def _rule_based_scan(paragraphs: list[str], para_start_idx: int,
                     categories: list[str]) -> tuple[list[dict], set]:
    """规则引擎预扫描：对已知人物/地点做精确字符串匹配。

    长实体优先匹配，避免短实体误匹配子串。
    例如 "西子湖" 优先于 "西湖"，"白素贞" 优先于 "白氏"。

    Args:
        paragraphs: 目标段落文本列表
        para_start_idx: 起始段落在全文中的索引（仅用于日志）
        categories: 请求的标注类别

    Returns:
        (annotations, exclusion_set)
        annotations: 规则标注列表（与 LLM 标注相同格式）
        exclusion_set: {(entity, global_start, global_end), ...} 供 LLM prompt 排除
    """
    annotations = []
    exclusion_set = set()

    # 只对 person 和 location 做规则匹配（这两类已知实体最全）
    rule_categories = [c for c in categories if c in ("person", "location")]
    if not rule_categories:
        return annotations, exclusion_set

    # 收集所有已知实体，按长度降序排列（长实体优先匹配）
    entities_to_match: list[tuple[str, str, dict]] = []  # [(entity, category, kb), ...]
    for cat in rule_categories:
        kb = _ENTITY_KB.get(cat)
        if not kb:
            continue
        for entity in kb.get("known", []):
            entities_to_match.append((entity, cat, kb))
    entities_to_match.sort(key=lambda x: -len(x[0]))

    # 逐段扫描
    offset = 0  # 在 target_paragraphs 拼接文本中的全局偏移
    for para_text in paragraphs:
        # 记录本段已被匹配的字符区间，避免重叠匹配
        matched_ranges: list[tuple[int, int]] = []

        for entity, cat, kb in entities_to_match:
            search_start = 0
            while True:
                pos = para_text.find(entity, search_start)
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
                    global_start = offset + pos
                    global_end = offset + entity_end

                    # 短实体降级 confidence
                    conf = "high"
                    if len(entity) <= 2 and cat == "person":
                        conf = "medium"

                    annotations.append({
                        "entity": entity,
                        "category": cat,
                        "start_char": global_start,
                        "end_char": global_end,
                        "explanation": f"{kb['label']}（规则匹配）",
                        "confidence": conf,
                        "color": DEFAULT_CATEGORY_COLORS.get(
                            cat, DEFAULT_CATEGORY_COLORS["custom"]
                        ),
                        "source": "rule",
                    })
                    exclusion_set.add((entity, global_start, global_end))

                search_start = pos + 1  # 继续搜索下一个出现

        offset += len(para_text) + 2  # +2 for paragraph separator (\n\n)

    return annotations, exclusion_set


def _build_annotation_prompt(categories: list[str],
                             excluded_entities: list[str] = None,
                             batch_paragraph_count: int = 1,
                             relaxed: bool = False) -> str:
    """按请求的类别构建带领域知识的标注 system prompt（question 式 + 行式 tuple 输出）。

    Args:
        categories: 标注类别列表
        excluded_entities: 已由规则引擎标注的实体名列表（本批范围内）
        batch_paragraph_count: 本批段落数，用于动态标注上限
        relaxed: True 时使用宽松模式（CU-3 空标注重试用）
    """
    parts = ["你是一位精通中国古代文学与文言文文本分析的专家。请根据我提供的原文片段，精准提取以下五类关键信息。\n"]

    parts.append("## 提取任务与规范\n")
    for cat in categories:
        kb = _ENTITY_KB.get(cat)
        if not kb:
            continue
        known_list = "、".join(kb["known"][:15])
        parts.append(f"### {kb['label']}({cat})\n{kb['rules']}\n参考实体: {known_list}\n")

    # 动态标注上限
    per_cat_max = 15 * batch_paragraph_count
    total_max = per_cat_max * len(categories)

    parts.append("## 提取要求\n"
                 "1. 实体原文 = 原文中一字不差的原词/原句，严格以原文为准\n"
                 "2. 不要返回字符位置（start_char/end_char 由程序自动计算）\n"
                 "3. 人名/地点/术语：实体原文就是原文中该名称本身\n"
                 "4. 事件句子/主题句子：实体原文必须一字不差引用原文完整原句，禁止删改字词、禁止用自己的话概括、禁止加标点断句；第3列写它的概括名（如\"端午惊变\"\"报恩\"）\n"
                 "5. 某类信息在原文中不存在时，该类不输出任何行\n"
                 f"6. 每类最多返回 {per_cat_max} 个，总共不超过 {total_max} 个；共 {batch_paragraph_count} 段，逐段提取\n")
    if relaxed:
        parts.append("7. 请尽可能全面地提取所有出现的实体，不要遗漏\n")
    else:
        parts.append("7. 优先提取最明确的实体，宁可少提取不要提取错\n")

    # 排除列表
    if excluded_entities:
        excluded_str = "、".join(excluded_entities[:30])
        parts.append(
            f"\n## 已找到实体（请勿重复）\n"
            f"以下实体已由规则引擎找到，请勿重复：{excluded_str}\n"
        )

    parts.append("## 输出格式\n"
                 "每行一个实体，四列，用 <|#|> 分隔，行尾用 <|COMPLETE|> 结束：\n"
                 "实体原文<|#|>类别<|#|>概括词<|#|>解释<|COMPLETE|>\n"
                 "- 类别用英文 key：person/location/term/event/motif\n"
                 "- 概括词列：仅事件句子/主题句子填写（概括名），人名/地点/术语留空\n"
                 "- 只输出这些行，不要任何多余文字，不要 JSON，不要代码块\n")

    parts.append("## 示例\n")
    for cat in categories:
        kb = _ENTITY_KB.get(cat)
        if not kb or not kb.get("examples"):
            continue
        for text, anns in kb["examples"][:3]:
            for ann in anns:
                entity = ann.get("entity", text)
                note = ann.get("explanation", "")
                summary = note if cat in ("event", "motif") else ""
                explain = "" if cat in ("event", "motif") else note
                parts.append(f"原文: {text}\n返回: {entity}<|#|>{cat}<|#|>{summary}<|#|>{explain}<|COMPLETE|>\n")

    parts.append("## 无标点原文示例（弹词/戏曲等连续唱词，原文没有标点，照抄即可，不要自己加标点断句）\n"
                 "原文: 师父赐名六支谨遵师训扫尽桃园落叶采取雷芝异草\n"
                 "返回: 师父赐名六支谨遵师训扫尽桃园落叶采取雷芝异草<|#|>motif<|#|>素贞扫叶自述<|#|><|COMPLETE|>\n")

    return "\n".join(parts)


def _build_gleaning_prompt(categories: list[str],
                           found_entities: list[str],
                           batch_paragraph_count: int) -> str:
    """查漏(gleaning)二遍 prompt：追问 LLM 找出第一遍漏掉的实体。"""
    parts = ["你是一位精通中国古代文学与文言文文本分析的专家。第一遍提取可能遗漏了一些信息，请仔细复查原文，找出被漏掉的实体。\n"]

    parts.append("## 提取任务与规范\n")
    for cat in categories:
        kb = _ENTITY_KB.get(cat)
        if not kb:
            continue
        known_list = "、".join(kb["known"][:15])
        parts.append(f"### {kb['label']}({cat})\n{kb['rules']}\n参考实体: {known_list}\n")

    if found_entities:
        parts.append(f"\n## 已找到实体（不要重复）\n{'、'.join(found_entities[:50])}\n")
    else:
        parts.append("\n## 已找到实体（无）\n")

    parts.append("## 要求\n"
                 "1. 只返回【漏掉的】实体，已找到的不要重复\n"
                 "2. 实体原文必须一字不差引用原文，不要字符位置，禁止改写/总结/省略/翻译/加标点断句\n"
                 "3. 事件句子/主题句子第3列给概括词\n"
                 "4. 输出格式：实体原文<|#|>类别<|#|>概括词<|#|>解释<|COMPLETE|>，类别用英文 key，每行一个\n"
                 "5. 如果确实没有漏掉的，输出空（不输出任何行）\n")
    return "\n".join(parts)


_TUPLE_SEP = "<|#|>"
_COMPLETE = "<|COMPLETE|>"


def _parse_annotation_output(content_raw: str) -> list[dict]:
    """解析 LLM 标注输出：优先行式 tuple，JSON 兜底。

    返回 [{entity, category, summary, explanation}]。坐标不在此解析，由代码定位。
    行式 tuple 无法解析且 JSON 也失败时，抛出 json.JSONDecodeError（fast fail）。
    """
    content = content_raw.strip()
    if not content:
        return []
    if content.startswith("```"):
        content = content.split("\n", 1)[1].rsplit("```", 1)[0].strip()

    # 行式 tuple
    if _TUPLE_SEP in content:
        rows = []
        for line in content.splitlines():
            line = line.strip()
            if not line or _COMPLETE not in line:
                continue
            line = line.split(_COMPLETE, 1)[0]
            fields = line.split(_TUPLE_SEP)
            if len(fields) < 2:
                continue
            entity = fields[0].strip()
            category = fields[1].strip()
            summary = fields[2].strip() if len(fields) > 2 else ""
            explanation = fields[3].strip() if len(fields) > 3 else ""
            if not entity:
                continue
            rows.append({
                "entity": entity,
                "category": category,
                "summary": summary,
                "explanation": explanation,
            })
        if rows:
            return rows

    # JSON 兜底（兼容过渡期/个别模型仍返回 JSON）
    data = parse_llm_json(content)
    rows = []
    for ann in data.get("annotations", []):
        entity = (ann.get("entity") or "").strip()
        if not entity:
            continue
        rows.append({
            "entity": entity,
            "category": ann.get("category", "custom"),
            "summary": (ann.get("summary") or ann.get("概括词") or "").strip(),
            "explanation": ann.get("explanation", ""),
        })
    return rows


async def _annotate_one_batch(batch_text: str, batch_offset: int, categories: list[str],
                               color_map: dict, llm,
                               batch_paragraph_count: int = 1,
                               excluded_entities: list[str] = None,
                               retry_mode: bool = False,
                               glean_found_entities: list[str] = None) -> list[dict]:
    """对单批文本调用 LLM 标注，返回偏移由代码定位后的 annotation dict 列表。

    LLM 只负责返回实体名 + 类别 + 解释，start/end 由代码根据 entity 精确/模糊匹配得出。
    batch_offset: 该批文本在全文拼接文本中的起始字符位置。
    LLM JSON 解析失败时 fast fail: 不静默吞错，直接抛出 JSONDecodeError。

    Args:
        batch_text: 本批段落拼接文本
        batch_offset: 本批在全文拼接文本中的起始偏移
        categories: 标注类别
        color_map: 类别→颜色映射
        llm: LLM 客户端
        batch_paragraph_count: 本批段落数（用于 prompt 动态上限）
        excluded_entities: 本批范围内已由规则引擎标注的实体名列表
        retry_mode: True 时使用宽松 prompt + 稍高 temperature
    """
    from ...config import settings as app_settings
    model = app_settings.effective_model  # 使用主力模型提升标注质量

    if glean_found_entities is not None:
        system = _build_gleaning_prompt(categories, glean_found_entities, batch_paragraph_count)
        user = f"原文:\n\n{batch_text}\n\n请找出以上原文中【漏掉的】实体。"
    else:
        system = _build_annotation_prompt(
            categories,
            excluded_entities=excluded_entities,
            batch_paragraph_count=batch_paragraph_count,
            relaxed=retry_mode,
        )
        user = f"原文:\n\n{batch_text}\n\n请标注以上原文中的实体。"

    temperature = 0.3 if retry_mode else 0.1
    resp = await llm.generate(system, user, model=model, max_tokens=2048 * max(1, batch_paragraph_count), temperature=temperature, thinking_disabled=True)
    content_raw = resp.content.strip()
    if content_raw.startswith("```"):
        content_raw = content_raw.split("\n", 1)[1].rsplit("```", 1)[0]

    rows = _parse_annotation_output(content_raw)

    annotations = []
    for ann in rows:
        entity = (ann.get("entity") or "").strip()
        if not entity:
            continue

        cat = ann.get("category", "custom")
        cat = _CATEGORY_REVERSE_MAP.get(cat, cat)
        if cat not in color_map:
            cat = "custom"

        # 坐标由代码确定性定位，忽略 LLM 返回的任何 start_char/end_char。
        # 同一个实体出现多次时，每个出现都生成一条标注。
        matches = locate_entity_all(batch_text, entity)
        if not matches:
            logger.warning(f"annotate_passage 无法在原文中定位实体: {entity[:30]}")
            continue

        for match in matches:
            pos = match.start_char
            ann_dict = {
                "entity": entity,
                "category": cat,
                "start_char": batch_offset + pos,
                "end_char": batch_offset + pos + len(entity),
                "explanation": ann.get("explanation", ""),
                "confidence": match.confidence,
                "color": color_map.get(cat, DEFAULT_CATEGORY_COLORS["custom"]),
                "source": "agent",
            }
            # 概括词仅 event/motif 有，暂存 annotation，由 handler 提取成旁注后移除
            ann_dict["summary"] = ann.get("summary", "")
            if retry_mode:
                ann_dict["is_retry"] = True
            annotations.append(ann_dict)

    return annotations


def _assess_quality(total_batches: int, failed_batches: int,
                    empty_batches: int, coverage_rate: float) -> str:
    """CU-6: 综合评估标注质量。"""
    if total_batches == 0:
        return "poor"
    if failed_batches > total_batches * 0.3:
        return "poor"
    if empty_batches > total_batches * 0.5:
        return "degraded"
    if coverage_rate < 0.80:
        return "degraded"
    return "good"


def _build_event_motif_marginalia(annotations: list[dict], chapter_number: int,
                                  boundaries: list[dict]) -> list[dict]:
    """对 event/motif 标注生成旁注：概括词去重，锚在第一次出现的 span 起点。

    概括词只进旁注 content，不从 annotation 单独存字段（调用方会 pop 掉 summary）。
    同一概括词只产一条旁注，但每个细节出现都保留为独立标注。
    """
    marginalia = []
    seen_summary = set()
    for ann in annotations:
        if ann.get("category") not in ("event", "motif"):
            continue
        summary = (ann.get("summary") or "").strip()
        if not summary or summary in seen_summary:
            continue
        sc = int(ann.get("start_char", 0))
        para_idx = None
        anchor = None
        for b in boundaries:
            if b["start"] <= sc < b["end"]:
                para_idx = b["idx"]
                anchor = sc - b["start"]
                break
        if para_idx is None:
            continue
        seen_summary.add(summary)
        content = summary
        if ann.get("explanation"):
            content = f"{summary}：{ann['explanation']}"
        marginalia.append({
            "id": f"agent_marg_{chapter_number}_{para_idx}_{len(marginalia)}",
            "chapter_number": chapter_number,
            "paragraph_index": para_idx,
            "anchor_char_offset": anchor,
            "content": content,
            "color": ann.get("color", DEFAULT_CATEGORY_COLORS["custom"]),
            "source": "agent",
        })
    return marginalia


async def handler(chapter_number: int, paragraph_range: list,
                  categories: list = None,
                  color_overrides: dict = None,
                  progress_callback=None,
                  node_client=None, llm=None) -> dict:
    """对指定段落进行实体标注（分批处理 + 规则预扫描 + 空标注重试 + 预筛选）。

    阶段 0: 规则预扫描已知 person/location 实体
    阶段 1: 动态 batch_size LLM 分批标注
    阶段 2: 空标注检测 → 单段宽松重试 (CU-3)
    阶段 3: 低价值 batch 跳过 LLM (CU-4, 不影响 offset)
    阶段 4: 质量指标 (CU-6)

    fast fail: LLM JSON 格式异常直接向上传播，不吞错。
    """
    if node_client is None:
        return {"error": "Node API 客户端未初始化"}
    if llm is None:
        return {"error": "LLM 未初始化"}

    # Fast fail: validate paragraph_range
    if not isinstance(paragraph_range, list) or len(paragraph_range) != 2:
        return {"error": f"paragraph_range 必须是包含两个整数的数组，收到: {paragraph_range}"}
    if not isinstance(paragraph_range[0], int) or not isinstance(paragraph_range[1], int):
        return {"error": f"paragraph_range 元素必须是整数"}
    if paragraph_range[0] < 0:
        return {"error": f"paragraph_range 起始索引不能为负数: {paragraph_range[0]}"}
    if paragraph_range[1] < paragraph_range[0]:
        return {"error": f"paragraph_range 结束索引 ({paragraph_range[1]}) 不能小于起始索引 ({paragraph_range[0]})"}

    from ...config import settings

    async def _report_progress(message: str):
        if progress_callback is not None:
            await progress_callback(message)

    if categories is None:
        categories = ["person", "location", "event", "term", "motif", "custom"]
    if color_overrides is None:
        color_overrides = {}

    color_map = {**DEFAULT_CATEGORY_COLORS, **color_overrides}

    ch = await node_client.get_literature_by_chapter(chapter_number)
    if ch is None:
        return {"error": f"未找到第 {chapter_number} 章"}

    content = ch.get("content", "")
    paragraphs = _split_paragraphs(content)

    start_idx = max(0, paragraph_range[0])
    end_idx = min(len(paragraphs), paragraph_range[1] + 1)
    target_paragraphs = paragraphs[start_idx:end_idx]
    if not target_paragraphs:
        return {"error": "指定的段落范围为空"}

    # ---- 阶段 0: CU-1 规则预扫描 ----
    all_annotations = []
    exclusion_set: set = set()
    if settings.annotation_rule_prescan_enabled:
        rule_annotations, exclusion_set = _rule_based_scan(
            target_paragraphs, start_idx, categories
        )
        all_annotations.extend(rule_annotations)
        logger.info(
            f"annotate_passage 规则预扫描: {len(rule_annotations)} 个实体"
        )
    await _report_progress(
        f"规则预扫描完成，已知实体 {len(all_annotations)} 个，开始 LLM 分批标注..."
    )

    # ---- 阶段 1: 段落内切块 + 逐块 LLM 标注 ----
    batch_count = 0
    failed_batches = 0
    empty_batches = 0
    retried_batches = 0
    annotated_paragraphs: set = set()  # CU-6: 被标注的段落索引

    for para_rel_idx, para_text in enumerate(target_paragraphs):
        # 段落在拼接全文中的起始偏移（与 _rule_based_scan 一致）
        para_offset = sum(len(target_paragraphs[k]) + 2 for k in range(para_rel_idx))

        # 超长段落切块（<=800 字不切，整段一个块）
        chunks = _split_paragraph_into_chunks(para_text)

        for chunk_start, chunk_end in chunks:
            chunk_text = para_text[chunk_start:chunk_end]
            # 块在拼接全文中的偏移 = 段落 offset + 块内起始（纯切片，纯加法，避免错位）
            chunk_offset = para_offset + chunk_start

            batch_count += 1

            # 块级排除列表（规则引擎已标的本块实体，供 LLM 跳过）
            chunk_excluded = [
                entity for entity, g_start, g_end in exclusion_set
                if chunk_offset <= g_start < chunk_offset + len(chunk_text)
            ]

            try:
                batch_annotations = await _annotate_one_batch(
                    chunk_text, chunk_offset, categories, color_map, llm,
                    batch_paragraph_count=1,
                    excluded_entities=chunk_excluded if chunk_excluded else None,
                )

                # ---- 查漏(gleaning)二遍 ----
                if settings.annotation_gleaning_enabled and batch_annotations:
                    found_entities = list(dict.fromkeys(
                        list(chunk_excluded) + [a["entity"] for a in batch_annotations]
                    ))
                    try:
                        glean_annotations = await _annotate_one_batch(
                            chunk_text, chunk_offset, categories, color_map, llm,
                            batch_paragraph_count=1,
                            glean_found_entities=found_entities,
                        )
                        if glean_annotations:
                            batch_annotations.extend(glean_annotations)
                            logger.info(
                                f"annotate_passage 块 {batch_count} gleaning 补回 {len(glean_annotations)} 个实体"
                            )
                    except json.JSONDecodeError:
                        logger.warning(f"annotate_passage 块 {batch_count} gleaning 解析失败，忽略")

                # ---- 空标注检测 + 宽松重试（块级）----
                if len(batch_annotations) == 0 and settings.annotation_empty_retry_enabled:
                    logger.warning(f"annotate_passage 块 {batch_count} 返回空标注，触发宽松重试")
                    try:
                        retry_annotations = await _annotate_one_batch(
                            chunk_text, chunk_offset, categories, color_map, llm,
                            batch_paragraph_count=1,
                            excluded_entities=chunk_excluded if chunk_excluded else None,
                            retry_mode=True,
                        )
                        if retry_annotations:
                            all_annotations.extend(retry_annotations)
                            retried_batches += 1
                            annotated_paragraphs.add(para_rel_idx)
                            logger.info(
                                f"annotate_passage 块 {batch_count} 重试成功: {len(retry_annotations)} 个标注"
                            )
                        else:
                            empty_batches += 1
                    except json.JSONDecodeError:
                        failed_batches += 1
                else:
                    all_annotations.extend(batch_annotations)
                    if len(batch_annotations) == 0:
                        empty_batches += 1
                    else:
                        annotated_paragraphs.add(para_rel_idx)

                logger.info(
                    f"annotate_passage 块 {batch_count}: "
                    f"段落[{para_rel_idx}] 块[{chunk_start}-{chunk_end}], "
                    f"文本{len(chunk_text)}字, 标注{len(batch_annotations)}个实体"
                )
                await _report_progress(
                    f"标注进度: 第 {batch_count} 批完成，段落 "
                    f"{start_idx + para_rel_idx + 1}/{end_idx}，"
                    f"累计 {len(all_annotations)} 个实体"
                )
            except json.JSONDecodeError as e:
                failed_batches += 1
                logger.error(
                    f"annotate_passage 块 {batch_count} JSON 解析失败 "
                    f"(段落[{para_rel_idx}] 块[{chunk_start}-{chunk_end}]): {e}"
                )

    if failed_batches > 0 and len(all_annotations) == 0:
        return {
            "error": (
                f"标注失败: {failed_batches}/{batch_count} 批次 "
                f"LLM 返回格式异常，无可用标注结果"
            ),
            "degraded": True,
        }

    # 去重（同一实体+同一位置只保留一个）
    seen = set()
    deduped = []
    for ann in all_annotations:
        key = (ann["entity"], ann["start_char"], ann["end_char"])
        if key not in seen:
            seen.add(key)
            deduped.append(ann)

    # ---- 段落边界（= P0 span_validator 兜底 + 旁注锚点共用） ----
    boundaries = []
    b_offset = 0
    for i, p in enumerate(target_paragraphs):
        boundaries.append({"idx": start_idx + i, "start": b_offset, "end": b_offset + len(p)})
        b_offset += len(p) + 2

    # ---- P0: 标注坐标强制兜底校验（span_validator 6 级回退） ----
    # 即使 locate_entity_all 本身是确定性 find，仍可能遇到:
    #   1) entity 中包含标点而原文中为不同标点变体/全半角空格差 → find 漏掉后 fallback 匹配
    #   2) 规则预扫描阶段产生的 span（词表 exact match）
    #   3) 未来其他注入路径产生的脏坐标
    # 结果: 丢弃 not_found 的标注，修正 corrected 的坐标，补 quality 统计
    from .span_validator import validate_span, SpanValidationResult
    sv_valid = sv_corrected = sv_discarded = 0
    sv_methods: Dict[str, int] = {}
    if deduped:
        validated_anns: list[dict] = []
        for ann in deduped:
            entity = (ann.get("entity") or "").strip()
            if not entity:
                sv_discarded += 1
                continue
            # start_char/end_char 在 annotate_passage 中是『章内偏移』，但 validate_span
            # 需要『段内偏移 + 段原文』。用 boundaries 定位段落再转偏移。
            sc = int(ann.get("start_char", 0) or 0)
            para_idx_local = None
            para_start_offset = 0
            para_text = ""
            for i, p in enumerate(target_paragraphs):
                b = boundaries[i]
                if b["start"] <= sc < b["end"]:
                    para_idx_local = i
                    para_start_offset = b["start"]
                    para_text = p
                    break
            if para_idx_local is None:
                # sc 越界 → 尝试找包含该 entity 的第一段（兜底）
                for i, p in enumerate(target_paragraphs):
                    if entity in p:
                        para_idx_local = i
                        para_start_offset = boundaries[i]["start"]
                        para_text = p
                        break
            if para_idx_local is None:
                sv_discarded += 1
                logger.warning("标注找不到所属段落，丢弃: entity=%r", entity[:30])
                continue
            start_in_para = sc - para_start_offset
            end_in_para = start_in_para + len(entity)
            try:
                r: SpanValidationResult = validate_span(para_text, entity, start_in_para, end_in_para)
            except ValueError as ex:
                sv_discarded += 1
                logger.warning("标注 span_validator 输入异常，丢弃: entity=%r err=%s", entity[:30], ex)
                continue
            if r.status == "discarded":
                sv_discarded += 1
                logger.warning(
                    "标注 span_validator 最终未匹配，丢弃: entity=%r para_idx=%d",
                    entity[:30], start_idx + para_idx_local,
                )
                continue
            if r.status == "corrected":
                sv_corrected += 1
                if r.correction_method:
                    sv_methods[r.correction_method] = sv_methods.get(r.correction_method, 0) + 1
                ann["start_char"] = para_start_offset + r.start_char
                ann["end_char"] = para_start_offset + r.end_char
                ann["confidence"] = r.confidence
                ann["span_corrected_by"] = r.correction_method or ""
            else:  # valid
                sv_valid += 1
            validated_anns.append(ann)
        original_len = len(deduped)
        deduped = validated_anns
        if sv_discarded or sv_corrected:
            logger.info(
                "span_validator 结果: total=%d valid=%d corrected=%d discarded=%d. "
                "修正方法: %s",
                original_len, sv_valid, sv_corrected, sv_discarded, sv_methods or "{}",
            )
    # END P0 兜底

    # ---- 旁注生成：event/motif 概括词 → marginalia（去重、锚第一次出现） ----
    marginalia = _build_event_motif_marginalia(deduped, chapter_number, boundaries)

    # 概括词只进旁注，不留在 annotation
    for ann in deduped:
        ann.pop("summary", None)

    if failed_batches > 0:
        logger.warning(
            f"annotate_passage: {failed_batches}/{batch_count} 批次失败, "
            f"返回 {len(deduped)} 个标注（部分结果）"
        )

    # ---- CU-6: 质量指标 ----
    coverage_rate = (
        len(annotated_paragraphs) / max(1, len(target_paragraphs))
    )
    rule_count = sum(1 for a in all_annotations if a.get("source") == "rule")

    batch_stats = {
        "total_batches": batch_count,
        "failed_batches": failed_batches,
        "empty_batches": empty_batches,
        "retried_batches": retried_batches,
        "total_annotations": len(deduped),
        "rule_annotations": rule_count,
        "llm_annotations": len(deduped) - rule_count,
        "paragraphs_total": len(target_paragraphs),
        "paragraphs_annotated": len(annotated_paragraphs),
        "coverage_rate": coverage_rate,
        "span_validator": {
            "valid": sv_valid,
            "corrected": sv_corrected,
            "discarded": sv_discarded,
            "correction_methods": sv_methods,
        },
    }
    if settings.annotation_quality_metrics_enabled:
        batch_stats["quality"] = _assess_quality(
            batch_count, failed_batches, empty_batches, coverage_rate,
        )
    else:
        batch_stats["quality"] = "unknown"

    return {
        "target": {
            "chapter_number": chapter_number,
            "chapter_title": ch.get("title", ""),
            "paragraph_range": [start_idx, end_idx],
        },
        "annotations": deduped,
        "marginalia": marginalia,
        "insights": [],
        "categories_applied": categories,
        "batch_stats": batch_stats,
    }
