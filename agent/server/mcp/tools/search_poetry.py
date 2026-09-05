"""search_poetry — 诗词曲牌检索工具。

在白蛇传语料中检索诗词、唱词、曲牌等韵文内容。
通过文体标记词（诗曰/词曰/曲曰/有诗为证/唱道 等）定位韵文段落，
再按用户给出的关键词或体裁进一步过滤。
"""
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

TOOL_DEF = {
    "name": "search_poetry",
    "description": (
        "在白蛇传全文中检索诗词、唱词、曲牌等韵文内容。"
        "通过文体标记词（诗曰/词曰/曲曰/有诗为证/唱道/赋曰 等）定位韵文段落，"
        "再按关键词或体裁过滤。适用于'查找某主题的诗词''列出所有唱词''某版本的诗词风格'类问题。"
        "返回段落级匹配结果含章节/朝代/体裁/原文摘录。"
    ),
    "inputSchema": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "搜索关键词（如'断桥''西湖''离别'），留空则返回所有韵文段落",
            },
            "form_type": {
                "type": "string",
                "description": "体裁过滤: 诗/词/曲/唱词/赋/全部（默认全部）",
                "default": "全部",
            },
            "chapter": {
                "type": "integer",
                "description": (
                    "按章节过滤（1=唐/2=宋/3=明/4=清戏曲/5=清小说/6=清弹词/7=现当代）。"
                    "系统会报告各章命中数；做全量诗词汇编时若某章 0 命中，"
                    "应逐章调用本工具核对后再下结论，不得默认该章无韵文。"
                ),
            },
        },
        "required": [],
    },
}

# 韵文标记词 → 体裁分类
# 注意：严禁使用单字标记（如「折」「唱」）——会把「折杀了妾身」「齐唱和」
# 这类普通叙事误判为韵文，污染检索结果、挤占结果配额。
POETRY_MARKERS = {
    "诗": ["诗曰", "有诗为证", "后人有诗", "有诗云", "诗云", "题诗", "吟诗"],
    "词": ["词曰", "有词为证", "词云", "调寄", "词牌"],
    "曲": ["曲曰", "唱道", "曲云", "煞尾", "尾声曲", "集唐"],
    "唱词": ["唱道", "唱罢", "接唱", "内唱", "唱段", "歌曰", "道白", "念白"],
    "赋": ["赋曰", "赋云", "赞曰", "颂曰"],
}

# 京剧/戏曲唱词结构标记（如「(内唱南梆子倒板)」「(接唱小安板)」）
OPERA_SING_MARKERS = ["南梆子", "西皮", "二黄", "倒板", "慢板", "快板", "散板", "摇板"]

# 所有标记词的扁平列表
ALL_MARKERS = []
for markers in POETRY_MARKERS.values():
    ALL_MARKERS.extend(markers)

# 每章结果配额上限（防止单章把总配额吃光，导致后面章节完全缺席）
PER_CHAPTER_LIMIT = 8
MAX_RESULTS = 56

# 语料快照路径（与 chunker.py SNAPSHOT_PATH 一致）
_CORPUS_PATH = (
    Path(__file__).resolve().parent.parent.parent.parent.parent
    / "rag_eval" / "ground_truth" / "corpus.jsonl"
)


def _load_corpus() -> list[dict]:
    """读取语料快照，返回 chunk 列表。"""
    try:
        chunks = []
        with open(_CORPUS_PATH, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    chunks.append(json.loads(line))
        return chunks
    except FileNotFoundError:
        logger.warning(f"语料快照不存在: {_CORPUS_PATH}")
        return []
    except Exception as e:
        logger.error(f"读取语料快照失败: {e}")
        return []


def _detect_form(text: str) -> list[str]:
    """检测文本中包含的韵文体裁。"""
    forms = []
    for form, markers in POETRY_MARKERS.items():
        if any(m in text for m in markers):
            forms.append(form)
    # 京剧/戏曲唱词：含曲牌板式标记且带「唱/念」行文 → 归入唱词
    if any(m in text for m in OPERA_SING_MARKERS) and ("唱" in text or "念" in text):
        if "唱词" not in forms:
            forms.append("唱词")
    return forms


# 章节号（语料用「一/二/.../七」）→ 整数映射
CHAPTER_NUM_MAP = {"一": 1, "二": 2, "三": 3, "四": 4, "五": 5, "六": 6, "七": 7}


async def handler(query: str = "", form_type: str = "全部", chapter: int = 0, **kwargs) -> dict:
    """检索语料中的诗词曲牌等韵文内容。"""
    chunks = _load_corpus()
    if not chunks:
        return {
            "error": "语料快照不可读，无法检索韵文内容。请确认 corpus.jsonl 存在。",
        }

    # 过滤研究文献
    chunks = [c for c in chunks if c.get("metadata", {}).get("source_type") == "primary_literature"]

    results = []
    query_lower = query.strip() if query else ""
    per_chapter_count: dict[str, int] = {}

    for chunk in chunks:
        text = chunk.get("text", "")
        meta = chunk.get("metadata", {})

        # 章节过滤
        ch_int = CHAPTER_NUM_MAP.get(str(meta.get("chapter_number", "")), 0)
        if chapter and ch_int != chapter:
            continue

        # 检测韵文体裁
        forms = _detect_form(text)
        if not forms:
            continue

        # 体裁过滤
        if form_type and form_type != "全部":
            if form_type not in forms:
                continue

        # 关键词过滤
        if query_lower and query_lower not in text:
            continue

        # 每章配额：防止单章吃光总配额
        ch_key = str(meta.get("chapter_number", ""))
        if per_chapter_count.get(ch_key, 0) >= PER_CHAPTER_LIMIT:
            continue

        # 提取韵文摘录：从标记词处取至段落末尾（上限 900 字符）
        # 此前只取 150 字符导致长韵文被腰斩（如《西湖三塔记》开篇词），
        # LLM 照录残篇写入报告造成事实缺失。
        excerpts = []
        for marker in ALL_MARKERS:
            idx = text.find(marker)
            if idx >= 0:
                start = max(0, idx)
                end = min(len(text), idx + 900)
                excerpts.append(text[start:end])
                break

        results.append({
            "chunk_id": chunk.get("doc_id", ""),
            "chapter_number": meta.get("chapter_number", ""),
            "chapter_title": meta.get("chapter_title", ""),
            "dynasty": meta.get("dynasty", ""),
            "genre": meta.get("genre", ""),
            "forms": forms,
            "excerpt": excerpts[0] if excerpts else text[:900],
            "excerpt_truncated": len(excerpts[0] if excerpts else text[:900]) >= 900,
            "paragraph_index": meta.get("paragraph_index", -1),
        })
        per_chapter_count[ch_key] = per_chapter_count.get(ch_key, 0) + 1

        if len(results) >= MAX_RESULTS:
            break

    # 各章命中统计（让调用方一眼看到章节覆盖全景）
    chapter_dist = {k: per_chapter_count[k] for k in sorted(per_chapter_count)}
    chapters_touched = sorted(
        CHAPTER_NUM_MAP.get(k, 0) for k in per_chapter_count
    )
    missing_chapters = [c for c in range(1, 8) if c not in chapters_touched]
    missing_hint = (
        f"注意：第 {'、'.join(str(c) for c in missing_chapters)} 章 0 命中——"
        "如做全量诗词汇编，请逐章调用本工具（chapter=章号）核对，"
        "不得默认该章无韵文。"
        if missing_chapters and not chapter
        else ""
    )

    return {
        "title": f"韵文检索{' - ' + form_type if form_type != '全部' else ''}"
                 + (f"（第{chapter}章）" if chapter else ""),
        "query": query or "(全部韵文)",
        "form_type": form_type,
        "chapter": chapter,
        "total": len(results),
        "truncated": len(results) >= MAX_RESULTS,
        "chapter_distribution": chapter_dist,
        "results": results,
        "note": (
            f"共检索到 {len(results)} 个含韵文的段落，各章分布: {chapter_dist}。"
            + ("结果已按每章最多 8 条截断，建议用 chapter 参数逐章深挖。" if len(results) >= MAX_RESULTS else "")
            + missing_hint
        ),
    }
