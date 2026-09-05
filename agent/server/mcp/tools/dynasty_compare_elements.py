"""v10 MCP 工具 -- 跨朝代文本演变对读: 对比两个朝代章节中的同名元素并识别演变类型。

设计文档: A.12 节 -- 后端 Agent 工具定义
"""
import asyncio
import json
import logging
import re

from ._text_utils import infer_dynasty_by_chapter

logger = logging.getLogger(__name__)

# 7 种演变类型
CHANGE_TYPES = [
    "emergence",       # 新出现
    "disappearance",   # 消失
    "amplification",   # 放大/强化
    "diminution",      # 缩小/弱化
    "transvaluation",  # 价值翻转
    "continuity",      # 延续不变
    "transplantation", # 移植改编
]

DEFAULT_CATEGORIES = ["person", "location", "motif", "term", "event"]

TOOL_DEF = {
    "name": "dynasty_compare_elements",
    "description": "对比两个朝代章节中的同名元素，识别演变类型(emergence/disappearance/amplification/diminution/transvaluation/continuity/transplantation)，返回结构化对比数据供前端跨朝代对读面板使用。⚠️ 该工具内部调用 LLM，耗时 3-5 秒；仅需引用原文证据的两章段落对比可用 find_shared_passages 替代（更快且基于精确文本比对）",
    "inputSchema": {
        "type": "object",
        "properties": {
            "left_chapter_number": {
                "type": "integer",
                "description": "左侧章节编号(唐代=1, 宋代=2, 明代=3, 清代=4/5/6, 现当代=7)",
            },
            "right_chapter_number": {
                "type": "integer",
                "description": "右侧章节编号(唐代=1, 宋代=2, 明代=3, 清代=4/5/6, 现当代=7)",
            },
            "categories": {
                "type": "array",
                "items": {"type": "string"},
                "description": "要分析的元素类别，默认全部5类: person/location/motif/term/event",
            },
            "query": {
                "type": "string",
                "description": "用户原始查询，用于提取关键词过滤相关段落",
            },
        },
        "required": ["left_chapter_number", "right_chapter_number"],
    },
}


def _parse_llm_json(raw: str) -> dict | None:
    """解析 LLM 返回的 JSON，修复常见格式错误。

    fast fail: 每种修复失败继续下一种。全部失败返回 None。
    """
    # 尝试 1: 直接解析
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass

    # 尝试 2: 消除尾部逗号（LLM 最常见的 JSON 错误）
    fixed = re.sub(r',\s*([}\]])', r'\1', raw)
    try:
        return json.loads(fixed)
    except json.JSONDecodeError:
        pass

    # 尝试 3: 提取最外层 {} 块
    start = raw.find('{')
    end = raw.rfind('}')
    if start >= 0 and end > start:
        extracted = raw[start:end + 1]
        # 再次尝试消除尾部逗号
        extracted_fixed = re.sub(r',\s*([}\]])', r'\1', extracted)
        try:
            return json.loads(extracted_fixed)
        except json.JSONDecodeError:
            pass

    # 尝试 4: 补全未闭合的括号
    open_braces = raw.count('{') - raw.count('}')
    open_brackets = raw.count('[') - raw.count(']')
    if open_braces > 0 or open_brackets > 0:
        patched = raw.rstrip()
        patched += ']' * open_brackets + '}' * open_braces
        patched = re.sub(r',\s*([}\]])', r'\1', patched)
        try:
            return json.loads(patched)
        except json.JSONDecodeError:
            pass

    return None


def _extract_context(content: str, para_idx: int, start_char: int, end_char: int, window: int = 300) -> str:
    """从章节全文的指定段落和字符位置提取周围上下文窗口。

    fast fail: 参数无效 → 返回空字符串。
    """
    content = re.sub(r'\r\n|\r', '\n', content)
    paragraphs = [p.strip() for p in content.split('\n\n') if p.strip()]
    if para_idx < 0 or para_idx >= len(paragraphs):
        return ""
    para = paragraphs[para_idx]
    if start_char < 0:
        start_char = 0
    if end_char > len(para):
        end_char = len(para)
    if start_char >= end_char:
        return ""
    ctx_start = max(0, start_char - window)
    ctx_end = min(len(para), end_char + window)
    return para[ctx_start:ctx_end]


async def _extract_shared_elements(
    left_content: str, right_content: str,
    categories: list[str], query: str,
    llm, model: str,
) -> list[dict] | None:
    """阶段一: 在左右两章中识别同名元素及其位置。

    返回 [{elementName, category, left, right}, ...] 或 None。
    每类最多5个，总计不超过15个。

    fast fail: LLM 调用异常或 JSON 解析失败 → 返回 None。
    """
    cat_labels = {
        "person": "人物名称", "location": "地点名称",
        "motif": "叙事母题", "term": "专有术语", "event": "事件名称",
    }
    cat_list = "、".join(cat_labels.get(c, c) for c in categories)
    topic_hint = f"用户关注主题: {query}。" if query else ""

    system = (
        f"你是古籍文本分析专家。在以下两个章节中找出同名的{cat_list}。{topic_hint}\n"
        "对每个同名元素，标注它在左右两侧的段落索引(0-based)和字符范围。\n"
        "返回纯JSON数组(不要markdown代码块):\n"
        '[{"elementName":"...","category":"...","left":[paraIdx,start,end],"right":[paraIdx,start,end]},...]\n'
        "规则:\n"
        "- 元素未在某侧出现则不输出该侧的 left/right 字段\n"
        "- 段落索引从0开始，每类最多5个，总计不超过15个\n"
        "- 只返回在两侧都有对应关系的同名元素"
    )
    user = (
        f"左文章:\n{left_content[:6000]}\n\n"
        f"右文章:\n{right_content[:6000]}"
    )

    try:
        resp = await llm.generate(system, user, model=model, max_tokens=2048, temperature=0.3, thinking_disabled=True)
    except Exception as e:
        logger.warning(f"_extract_shared_elements LLM 调用失败: {e}")
        return None

    data = _parse_llm_json(resp.content.strip())
    if data is None or not isinstance(data, list):
        logger.warning(f"_extract_shared_elements JSON 解析失败, raw={resp.content[:200]}")
        return None
    if len(data) == 0:
        logger.info("_extract_shared_elements: 未找到同名元素")
        return []

    logger.info(f"_extract_shared_elements: 识别到 {len(data)} 个同名元素")
    return data


async def _analyze_evolution(
    left_content: str, right_content: str,
    shared_elements: list[dict],
    left_title: str, right_title: str,
    llm, model: str,
    progress_callback=None,
) -> list[dict] | None:
    """阶段二: 对每个共享元素并行判定演变类型。

    返回完整 elements 列表或 None（全部失败时回退）。

    fast fail: 单元素失败 → 该元素标记为 continuity。
    全部元素失败 → 返回 None，触发上层回退。
    """
    sem = asyncio.Semaphore(5)
    done_count = 0
    total = len(shared_elements)

    async def _report_progress(message: str):
        if progress_callback is not None:
            await progress_callback(message)

    async def analyze_one(el: dict) -> dict:
        name = el.get("elementName", "?")
        cat = el.get("category", "motif")
        left_pos = el.get("left")
        right_pos = el.get("right")

        # 提取上下文
        left_ctx = ""
        right_ctx = ""
        left_occ = []
        right_occ = []
        if left_pos and isinstance(left_pos, list) and len(left_pos) >= 3:
            left_ctx = _extract_context(left_content, left_pos[0], left_pos[1], left_pos[2])
            left_occ = [{"paragraphIndex": left_pos[0], "span": {"startChar": left_pos[1], "endChar": left_pos[2]}}]
        if right_pos and isinstance(right_pos, list) and len(right_pos) >= 3:
            right_ctx = _extract_context(right_content, right_pos[0], right_pos[1], right_pos[2])
            right_occ = [{"paragraphIndex": right_pos[0], "span": {"startChar": right_pos[1], "endChar": right_pos[2]}}]

        if not left_ctx and not right_ctx:
            return {
                "elementName": name, "category": cat,
                "leftOccurrences": left_occ, "rightOccurrences": right_occ,
                "evolution": {"changeType": "continuity", "summary": "", "confidence": "low"},
            }

        system = (
            f"比较以下两段文本中的'{name}'({cat})，判定演变类型，写20字内摘要。\n"
            "7种类型: emergence/disappearance/amplification/diminution/transvaluation/continuity/transplantation\n"
            '只返回JSON: {"changeType":"...","summary":"..."}'
        )
        user_parts = []
        if left_ctx:
            user_parts.append(f"左({left_title}): {left_ctx}")
        if right_ctx:
            user_parts.append(f"右({right_title}): {right_ctx}")
        user = "\n".join(user_parts)

        async with sem:
            try:
                resp = await llm.generate(system, user, model=model, max_tokens=256, temperature=0.3, thinking_disabled=True)
                data = _parse_llm_json(resp.content.strip())
                if data and isinstance(data, dict):
                    nonlocal done_count
                    done_count += 1
                    await _report_progress(
                        f"元素演变分析: {done_count}/{total}（{name}）"
                    )
                    return {
                        "elementName": name, "category": cat,
                        "leftOccurrences": left_occ, "rightOccurrences": right_occ,
                        "evolution": {
                            "changeType": data.get("changeType", "continuity"),
                            "summary": data.get("summary", ""),
                            "confidence": "medium",
                        },
                    }
            except Exception as e:
                logger.warning(f"_analyze_evolution [{name}] LLM 失败: {e}")

        return {
            "elementName": name, "category": cat,
            "leftOccurrences": left_occ, "rightOccurrences": right_occ,
            "evolution": {"changeType": "continuity", "summary": "", "confidence": "low"},
        }

    tasks = [analyze_one(el) for el in shared_elements]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    elements = []
    failed = 0
    for r in results:
        if isinstance(r, BaseException):
            failed += 1
        elif isinstance(r, dict):
            elements.append(r)
    if failed > 0:
        logger.warning(f"_analyze_evolution: {failed}/{len(shared_elements)} 元素分析失败")
    if len(elements) == 0:
        return None
    return elements


async def handler(
    left_chapter_number: int,
    right_chapter_number: int,
    categories: list[str] | None = None,
    query: str = "",
    *,
    progress_callback=None,
    node_client,
    llm,
) -> dict:
    """对比两个朝代章节中的同名元素，识别演变类型。

    fast fail: 只在 JSON 解析失败时降级（LLM 格式异常可恢复）。
    其他异常（node_client/llm 不可用）向上传播。
    """
    if node_client is None:
        return {"error": "Node API 客户端未初始化"}
    if llm is None:
        return {"error": "LLM 未初始化"}

    if categories is None:
        categories = list(DEFAULT_CATEGORIES)

    # Step 1: 获取两个章节的全文
    left_ch = await node_client.get_literature_by_chapter(left_chapter_number)
    right_ch = await node_client.get_literature_by_chapter(right_chapter_number)

    if left_ch is None:
        return {"error": f"未找到第 {left_chapter_number} 章"}
    if right_ch is None:
        return {"error": f"未找到第 {right_chapter_number} 章"}

    left_title = left_ch.get("title", f"第{left_chapter_number}章")
    right_title = right_ch.get("title", f"第{right_chapter_number}章")
    left_content = left_ch.get("content", "")
    right_content = right_ch.get("content", "")

    # Step 1.5: 关键词过滤 —— 从 query 提取关键名词，只保留包含这些词的段落
    # 这样既保留了和用户问题相关的所有内容，又大幅减少了无效 token
    if query:
        # 停用词
        STOP = {"对比", "比较", "分析", "描写", "描述", "版本", "不同", "区别",
                "变化", "演变", "差异", "如何", "什么", "怎么", "在哪", "哪个",
                "和", "与", "的", "中", "在", "了", "是", "有", "我", "你", "帮",
                "请", "一下", "这个", "那个", "第", "章", "回", "代", "明代", "清代",
                "宋代", "唐代", "版本中", "中的"}
        # 用 jieba 分词提取关键词
        try:
            import jieba
            words = jieba.lcut(query)
        except Exception:
            words = list(query)
        keywords = [w for w in words if len(w) >= 2 and w not in STOP]
        if keywords:
            left_paras = [p.strip() for p in re.split(r'\n\n|\r\n\r\n', left_content) if p.strip()]
            right_paras = [p.strip() for p in re.split(r'\n\n|\r\n\r\n', right_content) if p.strip()]
            left_filtered = [p for p in left_paras if any(k in p for k in keywords)]
            right_filtered = [p for p in right_paras if any(k in p for k in keywords)]
            # 至少保留一些内容
            if len(left_filtered) >= 1 and len(right_filtered) >= 1:
                left_content = "\n\n".join(left_filtered)
                right_content = "\n\n".join(right_filtered)
                logger.info(
                    f"dynasty_compare_elements: keyword filter keywords={keywords} "
                    f"left: {len(left_paras)}→{len(left_filtered)} paras "
                    f"right: {len(right_paras)}→{len(right_filtered)} paras"
                )
        if not keywords:
            logger.info("dynasty_compare_elements: 未提取到有效关键词，使用全文")

    # 安全截断（过滤后通常不会超，但做防御）
    MAX_CHARS = 8000
    if len(left_content) > MAX_CHARS:
        left_content = left_content[:MAX_CHARS]
        logger.info(f"dynasty_compare_elements: 左侧章节截断至 {MAX_CHARS} 字符")
    if len(right_content) > MAX_CHARS:
        right_content = right_content[:MAX_CHARS]
        logger.info(f"dynasty_compare_elements: 右侧章节截断至 {MAX_CHARS} 字符")

    from ...config import settings as app_settings
    model = app_settings.effective_fast_model

    async def _report_progress(message: str):
        if progress_callback is not None:
            await progress_callback(message)

    # === 两阶段分析 ===
    # 阶段一: 识别同名元素
    await _report_progress(
        f"正在识别两章同名元素（第{left_chapter_number}章 vs 第{right_chapter_number}章）..."
    )
    shared = await _extract_shared_elements(
        left_content, right_content, categories, query, llm, model,
    )

    if shared is not None and len(shared) > 0:
        # 阶段二: 逐元素并行分析演变
        await _report_progress(
            f"已识别 {len(shared)} 个同名元素，开始逐元素分析演变..."
        )
        elements = await _analyze_evolution(
            left_content, right_content, shared,
            left_title, right_title, llm, model,
            progress_callback=progress_callback,
        )
        if elements is not None and len(elements) > 0:
            # 生成 summary
            cat_labels = {
                "person": "人物", "location": "地点", "motif": "母题",
                "term": "术语", "event": "事件",
            }
            cat_summary = "、".join(
                f"{cat_labels.get(el.get('category',''), el.get('category',''))}:{el.get('elementName','')}"
                for el in elements[:8]
            )
            change_counts: dict[str, int] = {}
            for el in elements:
                ct = el.get("evolution", {}).get("changeType", "continuity")
                change_counts[ct] = change_counts.get(ct, 0) + 1
            overview = f"对比{left_title}与{right_title}，识别到{cat_summary}等{len(elements)}个同名元素。"
            if change_counts:
                overview += f" 其中{change_counts.get('continuity','0')}个延续、{change_counts.get('amplification','0')}个强化、{change_counts.get('emergence','0')}个新出现。"

            location_changes = []
            for el in elements:
                if el.get("category") == "location":
                    left_occ = el.get("leftOccurrences", [])
                    right_occ = el.get("rightOccurrences", [])
                    location_changes.append({
                        "locationName": el.get("elementName", ""),
                        "leftStatus": "present" if left_occ else "absent",
                        "rightStatus": "present" if right_occ else "absent",
                        "changeType": el.get("evolution", {}).get("changeType", "continuity"),
                    })

            return {
                "config": {
                    "left": {
                        "chapterNumber": left_chapter_number,
                        "chapterTitle": left_title,
                        "dynasty": infer_dynasty_by_chapter(left_chapter_number),
                    },
                    "right": {
                        "chapterNumber": right_chapter_number,
                        "chapterTitle": right_title,
                        "dynasty": infer_dynasty_by_chapter(right_chapter_number),
                    },
                },
                "elements": elements,
                "locationChanges": location_changes,
                "summary": {
                    "overview": overview,
                    "keyFindings": [
                        f"{el.get('elementName','')}: {el.get('evolution',{}).get('changeType','')} — {el.get('evolution',{}).get('summary','')}"
                        for el in elements[:5] if el.get("evolution", {}).get("summary")
                    ],
                },
            }

    # === 回退: 原单次 LLM 调用（两阶段失败时兜底） ===
    logger.info("dynasty_compare_elements: 两阶段未产生结果，回退到单次 LLM 调用")
    await _report_progress("切换到整体对比模式（单次深度分析，耗时较长）...")
    cat_labels = {
        "person": "人物名称", "location": "地点名称",
        "motif": "叙事母题", "term": "专有术语", "event": "事件名称",
    }
    cat_list = "、".join(cat_labels.get(c, c) for c in categories)

    system = (
        "你是白蛇传跨朝代文本分析专家。对以下两个章节进行对比分析:\n\n"
        "## 任务\n"
        f"1. 识别两个章节中所有的{cat_list}\n"
        "2. 通过语义匹配建立跨朝代同名元素的对应关系\n"
        "3. 对每对同名元素判定演变类型(emergence/disappearance/amplification/diminution/transvaluation/continuity/transplantation)\n"
        "4. 生成整体概述和2-3个关键发现\n\n"
        "## 输出格式\n"
        "只返回 JSON:\n"
        '{"elements":[{"elementName":"...","category":"...",'
        '"leftOccurrences":[{"paragraphIndex":0,"span":{"startChar":0,"endChar":0}}],'
        '"rightOccurrences":[...],'
        '"evolution":{"changeType":"...","summary":"...","confidence":"medium"}}],'
        '"summary":{"overview":"...","keyFindings":["...","..."]}}\n'
        "每侧occurrence最多2个，每类最多4个元素，总计不超过12个。"
    )
    user = (
        f"## 左侧: {left_title} (第{left_chapter_number}章)\n{left_content}\n\n"
        f"## 右侧: {right_title} (第{right_chapter_number}章)\n{right_content}"
    )

    try:
        resp = await llm.generate(system, user, model=model, max_tokens=8192, temperature=0.3, thinking_disabled=True)
    except Exception as e:
        logger.error(f"dynasty_compare_elements 回退 LLM 调用失败: {e}")
        return {"error": f"对比分析失败: {e}", "degraded": True}

    content_raw = resp.content.strip()
    if content_raw.startswith("```"):
        content_raw = content_raw.split("\n", 1)[1].rsplit("```", 1)[0]

    data = _parse_llm_json(content_raw)
    if data is None:
        logger.error(f"dynasty_compare_elements 回退 JSON 解析失败, raw_len={len(content_raw)}")
        return {"error": "AI 返回了无法解析的对比结果，请重试", "degraded": True}

    raw_elements = data.get("elements", [])
    summary = data.get("summary", {})
    elements = []
    location_changes = []
    for el in raw_elements:
        cat = el.get("category", "motif")
        name = el.get("elementName", "")
        evo = el.get("evolution", {})
        elements.append({
            "elementName": name,
            "category": cat,
            "leftOccurrences": el.get("leftOccurrences", []),
            "rightOccurrences": el.get("rightOccurrences", []),
            "evolution": {
                "changeType": evo.get("changeType", "continuity"),
                "summary": evo.get("summary", ""),
                "confidence": evo.get("confidence", "medium"),
            },
        })
        if cat == "location":
            left_occ = el.get("leftOccurrences", [])
            right_occ = el.get("rightOccurrences", [])
            location_changes.append({
                "locationName": name,
                "leftStatus": "present" if left_occ else "absent",
                "rightStatus": "present" if right_occ else "absent",
                "changeType": evo.get("changeType", "continuity"),
            })

    return {
        "config": {
            "left": {
                "chapterNumber": left_chapter_number,
                "chapterTitle": left_title,
                "dynasty": infer_dynasty_by_chapter(left_chapter_number),
            },
            "right": {
                "chapterNumber": right_chapter_number,
                "chapterTitle": right_title,
                "dynasty": infer_dynasty_by_chapter(right_chapter_number),
            },
        },
        "elements": elements,
        "locationChanges": location_changes,
        "summary": {
            "overview": summary.get("overview", ""),
            "keyFindings": summary.get("keyFindings", []),
        },
    }

