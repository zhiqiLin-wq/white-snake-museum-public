"""体裁感知分块系统 —— 8种体裁策略 + 语义合并 + 结构感知。"""
import enum
import json
import re
import logging
import time
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

logger = logging.getLogger(__name__)

# ============================================================================
# 保留: 朝代提取
# ============================================================================
_DYNASTY_PATTERN = re.compile(r'(唐|宋|元|明|清|现当代)')


def _extract_dynasty(chapter_title: str) -> str | None:
    """从章节标题中提取朝代名，如 '一、萌芽期 唐：...' -> '唐'。"""
    m = _DYNASTY_PATTERN.search(chapter_title)
    if not m:
        return None
    dynasty = m.group(1)
    if dynasty in ('唐', '宋', '元', '明', '现当代'):
        return dynasty
    if dynasty == '清':
        return '清'
    return dynasty


# ============================================================================
# CU-01: Genre 枚举定义
# ============================================================================
class Genre(str, enum.Enum):
    CLASSICAL_CHINESE = "classical_chinese"
    HUABEN = "huaben"
    IMITATION_HUABEN = "imitation_huaben"
    OPERA_LIBRETTO = "opera_libretto"
    CHAPTER_VERNACULAR = "chapter_vernacular"
    TANCI = "tanci"
    SCRIPT = "script"
    RESEARCH_LITERATURE = "research_literature"


# ============================================================================
# CU-02: GenreChunkConfig 数据类 + GENRE_CHUNK_DEFAULTS 配置表
# ============================================================================
@dataclass
class GenreChunkConfig:
    chunk_strategy: str
    max_chunk_chars: int
    min_chunk_chars: int
    similarity_threshold: float = 0.0
    context_prefix_enabled: bool = False
    window_overlap: int = 0
    aria_lines_per_chunk: int = 0


GENRE_CHUNK_DEFAULTS = {
    Genre.CLASSICAL_CHINESE.value: GenreChunkConfig(
        chunk_strategy="sentence_semantic_merge",
        max_chunk_chars=250,
        min_chunk_chars=20,
        similarity_threshold=0.55,       # 0.50 -> 0.55
    ),
    Genre.HUABEN.value: GenreChunkConfig(
        chunk_strategy="paragraph_semantic_merge",
        max_chunk_chars=600,             # 800 -> 600
        min_chunk_chars=150,             # 40 -> 150
        similarity_threshold=0.75,
    ),
    Genre.IMITATION_HUABEN.value: GenreChunkConfig(
        chunk_strategy="semantic_paragraph",
        max_chunk_chars=600,             # 800 -> 600
        min_chunk_chars=150,             # 50 -> 150
        similarity_threshold=0.75,       # 新增语义合并阈值
    ),
    Genre.CHAPTER_VERNACULAR.value: GenreChunkConfig(
        chunk_strategy="structural_chapter_merge",
        max_chunk_chars=600,             # 800 -> 600
        min_chunk_chars=150,             # 50 -> 150
        similarity_threshold=0.75,
    ),
    Genre.RESEARCH_LITERATURE.value: GenreChunkConfig(
        chunk_strategy="paragraph_semantic_merge",
        max_chunk_chars=800,
        min_chunk_chars=150,             # 80 -> 150
        similarity_threshold=0.78,
    ),
    Genre.OPERA_LIBRETTO.value: GenreChunkConfig(
        chunk_strategy="structural_act",
        max_chunk_chars=1200,
        min_chunk_chars=60,
        context_prefix_enabled=True,
    ),
    Genre.SCRIPT.value: GenreChunkConfig(
        chunk_strategy="structural_scene",
        max_chunk_chars=1000,
        min_chunk_chars=150,             # 60 -> 150
        similarity_threshold=0.55,       # 0.60 -> 0.55
        context_prefix_enabled=True,
    ),
    Genre.TANCI.value: GenreChunkConfig(
        chunk_strategy="tanci_scene_window",
        max_chunk_chars=500,
        min_chunk_chars=60,
        context_prefix_enabled=True,
        window_overlap=0,
    ),
}


# ============================================================================
# CU-03: 通用工具函数
# ============================================================================
def _split_sentences(text: str) -> list[str]:
    """按 。！？ 切分为句子列表，保留标点在句尾。

    增强:
    1. 配对符号内部不切分 —— 仅在双引号、括号等配对符号之外才断句。
    2. 逗号不作为断句点 —— 仅 。！？ 触发断句。
    3. 句末标点后紧跟尾随后引号时, 断句点在后引号之后。
    """
    # ---- 配对符号定义 ----
    # 使用 \u 转义序列 (纯 ASCII) 避免中文引号编码问题
    # ---- 配对符号定义 ----
    # 使用 \u 转义序列 (纯ASCII) 避免编码问题
    LQ = '\u201c'   # 中文左双引号
    RQ = '\u201d'   # 中文右双引号
    LS = '\u2018'   # 中文左单引号
    RS = '\u2019'   # 中文右单引号
    LP = '\uff08'   # 全角左括号
    RP = '\uff09'   # 全角右括号
    LB = '\u3010'   # 黑左括号
    RB = '\u3011'   # 黑右括号
    LT = '\u300a'   # 左书名号
    RT = '\u300b'   # 右书名号

    _OPEN_SET = set([LQ, LS, LP, LB, LT, '(', '[', '{'])
    _CLOSE_SET = set([RQ, RS, RP, RB, RT, ')', ']', '}'])
    _CLOSE_TO_OPEN: dict[str, str] = {
        RQ: LQ, RS: LS, RP: LP, RB: LB, RT: LT,
        ')': '(', ']': '[', '}': '{',
    }

    # ASCII 双引号 / 单引号 (同字符开关) 用交替状态追踪
    _ASCII_DQUOTE = '"'
    _ASCII_SQUOTE = "'"
    # 哨兵值表示栈中的 ASCII 引号
    _SENTINEL_ASCII_DQ = '__ASCII_DQ__'
    _SENTINEL_ASCII_SQ = '__ASCII_SQ__'

    stack: list[str] = []          # 配对符号栈
    ascii_dq_open = False          # ASCII " 当前是否处于打开状态
    ascii_sq_open = False          # ASCII ' 当前是否处于打开状态

    sentences: list[str] = []
    start = 0
    i = 0
    n = len(text)

    while i < n:
        ch = text[i]

        if ch in _OPEN_SET:
            stack.append(ch)
        elif ch in _CLOSE_SET:
            expected_open = _CLOSE_TO_OPEN.get(ch)
            if expected_open is not None and stack and stack[-1] == expected_open:
                stack.pop()
        elif ch == _ASCII_DQUOTE:
            if ascii_dq_open:
                # 关闭 ASCII 双引号
                if stack and stack[-1] == _SENTINEL_ASCII_DQ:
                    stack.pop()
                ascii_dq_open = False
            else:
                # 打开 ASCII 双引号
                stack.append(_SENTINEL_ASCII_DQ)
                ascii_dq_open = True
        elif ch == _ASCII_SQUOTE:
            if ascii_sq_open:
                if stack and stack[-1] == _SENTINEL_ASCII_SQ:
                    stack.pop()
                ascii_sq_open = False
            else:
                stack.append(_SENTINEL_ASCII_SQ)
                ascii_sq_open = True
        elif ch in '。！？' and not stack:
            # 栈空 = 在所有配对符号之外, 可以断句
            end = i + 1  # 包含标点

            # 消费尾随后引号 (它们属于当前句子)
            j = i + 1
            while j < n:
                ch2 = text[j]
                if ch2 in _CLOSE_SET:
                    # 尾随闭符号: 消费它
                    expected_open = _CLOSE_TO_OPEN.get(ch2)
                    if expected_open is not None and stack:
                        # 但不应在栈中还有对应开符号时消费
                        pass  # 尾随闭符不属于任何开配对, 直接消费
                    j += 1
                elif ch2 == _ASCII_DQUOTE and ascii_dq_open:
                    # 关闭 ASCII 双引号
                    if stack and stack[-1] == _SENTINEL_ASCII_DQ:
                        stack.pop()
                    ascii_dq_open = False
                    j += 1
                elif ch2 == _ASCII_SQUOTE and ascii_sq_open:
                    if stack and stack[-1] == _SENTINEL_ASCII_SQ:
                        stack.pop()
                    ascii_sq_open = False
                    j += 1
                else:
                    break
            end = j

            sentence = text[start:end].strip()
            if sentence:
                sentences.append(sentence)
            start = end
            i = end - 1  # 循环末尾 +1 补偿

        i += 1

    # 收集剩余文本
    if start < n:
        remaining = text[start:].strip()
        if remaining:
            sentences.append(remaining)

    # 兜底: 若未产出任何句子 (文本中无 。！？ 或全部在配对符号内), 返回整段
    if not sentences and text.strip():
        sentences.append(text.strip())

    return sentences


def _split_paragraphs(text: str) -> list[str]:
    """按 \\n\\n 切分为段落列表，过滤空段落。"""
    parts = text.split('\n\n')
    return [p.strip() for p in parts if p.strip()]


def _compute_embedding(text: str, embedder) -> list[float]:
    """调用 embedder.embed([text])[0] 获取向量。"""
    return embedder.embed([text])[0]


# ============================================================================
# U13: 批量 embedding 辅助函数
# ============================================================================
def _compute_embeddings_batch(texts: list[str], embedder,
                               batch_size: int = 64) -> list:
    """批量计算 embedding，返回与 texts 等长的 embedding 列表。"""
    results = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        batch_embs = embedder.embed(batch)
        results.extend(batch_embs)
    return results


def _cosine_similarity(a, b) -> float:
    """余弦相似度计算。"""
    a_arr = np.asarray(a, dtype=np.float64)
    b_arr = np.asarray(b, dtype=np.float64)
    norm_a = np.linalg.norm(a_arr)
    norm_b = np.linalg.norm(b_arr)
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return float(np.dot(a_arr, b_arr) / (norm_a * norm_b))


def _is_outside_paired_symbols(text: str, pos: int) -> bool:
    """检查 pos 位置是否在所有配对符号之外 (栈深度为0)。"""
    # 配对符号定义 (左右不同字符)
    LQ = '\u201c'   # 中文左双引号
    RQ = '\u201d'   # 中文右双引号
    LS = '\u2018'   # 中文左单引号
    RS = '\u2019'   # 中文右单引号
    LP = '\uff08'   # 全角左括号
    RP = '\uff09'   # 全角右括号
    LB = '\u3010'   # 黑左括号
    RB = '\u3011'   # 黑右括号
    LT = '\u300a'   # 左书名号
    RT = '\u300b'   # 右书名号

    _OPEN_CHARS = set([LQ, LS, LP, LB, LT, '(', '[', '{'])
    _CLOSE_CHARS = set([RQ, RS, RP, RB, RT, ')', ']', '}'])
    _CLOSE_TO_OPEN = {RQ: LQ, RS: LS, RP: LP, RB: LB, RT: LT, ')': '(', ']': '[', '}': '{'}


def _split_long_paragraph_at_sentence_boundaries(text: str, max_chars: int) -> list[str]:
    """超长文本在句末标点处断开，优先级: 。！？；\\n。

    逗号不再作为断句点。候选断点需要位于所有配对符号之外。
    """
    if len(text) <= max_chars:
        return [text]

    result = []
    remaining = text
    while len(remaining) > max_chars:
        cutoff = max_chars
        search_start = max(int(max_chars * 0.6), 1)
        search_region = remaining[search_start:max_chars]

        best_pos = -1
        best_priority = 999
        for sep_priority, sep in enumerate(['。', '！', '？', '；', '\n']):
            rpos = search_region.rfind(sep)
            if rpos >= 0:
                candidate = search_start + rpos + 1
                if _is_outside_paired_symbols(remaining, candidate):
                    if best_pos < 0 or sep_priority < best_priority:
                        best_pos = candidate
                        best_priority = sep_priority
                    break

        if best_pos <= 0:
            for sep_priority, sep in enumerate(['。', '！', '？', '；', '\n']):
                rpos = search_region.rfind(sep)
                if rpos >= 0:
                    best_pos = search_start + rpos + 1
                    break

        if best_pos > 0:
            cutoff = best_pos

        result.append(remaining[:cutoff].strip())
        remaining = remaining[cutoff:].strip()

    if remaining.strip():
        result.append(remaining.strip())
    return result


# ============================================================================
# CU-03a: 句子级质心语义合并 (共享函数)
# ============================================================================
def _sentence_centroid_semantic_merge(
    sentences: list[str],
    sentence_embeddings: list,
    similarity_threshold: float,
    max_chunk_chars: int,
    min_chunk_chars: int,
) -> list[list[int]]:
    """句子级质心语义合并 + 短 chunk 向后合并后处理。"""
    raw_groups: list[list[int]] = []
    current_group: list[int] = []
    current_text_len = 0

    for i, s in enumerate(sentences):
        s_clean = s.strip()
        s_len = len(s_clean)
        if s_len == 0:
            continue

        if not current_group:
            current_group = [i]
            current_text_len = s_len
            continue

        emb_curr = sentence_embeddings[i]

        should_merge = False
        if emb_curr is not None and current_group:
            group_embs = [
                sentence_embeddings[j]
                for j in current_group
                if sentence_embeddings[j] is not None
            ]
            if group_embs:
                centroid = np.mean(np.asarray(group_embs, dtype=np.float64), axis=0)
                sim = _cosine_similarity(centroid, emb_curr)
                if sim >= similarity_threshold and (current_text_len + s_len) <= max_chunk_chars:
                    should_merge = True

        if should_merge:
            current_group.append(i)
            current_text_len += s_len
        else:
            raw_groups.append(current_group)
            current_group = [i]
            current_text_len = s_len

    if current_group:
        raw_groups.append(current_group)

    merged_groups: list[list[int]] = []
    for group in raw_groups:
        group_text = ''.join(sentences[i].strip() for i in group)
        if len(group_text) < min_chunk_chars and merged_groups:
            merged_groups[-1].extend(group)
        else:
            merged_groups.append(group)

    return merged_groups


# ============================================================================
# CU-04: 体裁识别逻辑
# ============================================================================
_GENRE_SUFFIX_MAP: list[tuple[re.Pattern, Genre]] = [
    (re.compile(r'（文言小说）'), Genre.CLASSICAL_CHINESE),
    (re.compile(r'（话本小说）'), Genre.HUABEN),
    (re.compile(r'（拟话本小说）'), Genre.IMITATION_HUABEN),
    (re.compile(r'（戏曲）'), Genre.OPERA_LIBRETTO),
    (re.compile(r'（章回体白话小说）'), Genre.CHAPTER_VERNACULAR),
    (re.compile(r'（弹词）'), Genre.TANCI),
    (re.compile(r'（剧本）'), Genre.SCRIPT),
    (re.compile(r'（研究文献）'), Genre.RESEARCH_LITERATURE),
]


def _detect_genre(chapter_title: str) -> Genre | None:
    """遍历 _GENRE_SUFFIX_MAP 做正则匹配，识别章节标题中的体裁标记。"""
    for pattern, genre in _GENRE_SUFFIX_MAP:
        if pattern.search(chapter_title):
            return genre
    return None


_CHAPTER_GENRE_FALLBACK: dict[str, Genre] = {
    '一': Genre.CLASSICAL_CHINESE,
    '二': Genre.HUABEN,
    '三': Genre.IMITATION_HUABEN,
    '四': Genre.OPERA_LIBRETTO,
    '五': Genre.CHAPTER_VERNACULAR,
    '六': Genre.TANCI,
    '七': Genre.SCRIPT,
}


# ============================================================================
# U17: parse_chunk_id 工具函数 —— 统一的 chunk_id 解析
# ============================================================================
def parse_chunk_id(chunk_id: str) -> dict:
    """解析 chunk_id 获取结构化信息。

    Args:
        chunk_id: 格式 "{source_prefix}_{genre}_{chapter_number}_{seq:05d}"
                  例如 "primary_huaben_三_00042"

    Returns:
        {
            "source_prefix": "primary" | "research",
            "source_type": "primary_literature" | "research_literature",
            "genre": "huaben",
            "chapter_number": "三",    # 字符串，不保证是数字
            "sequence": 42,
        }

    Raises:
        ValueError: 如果 chunk_id 格式不合法
    """
    parts = chunk_id.split("_")
    if len(parts) < 4:
        raise ValueError(f"Invalid chunk_id format: {chunk_id}")

    source_prefix = parts[0]
    # genre 可能包含下划线 (如 "opera_libretto", "classical_chinese")，
    # 因此需要从右向左解析：最后一段是 seq，倒数第二段是 chapter_number
    seq_str = parts[-1]
    chapter_number = parts[-2]
    genre = "_".join(parts[1:-2])

    try:
        sequence = int(seq_str)
    except ValueError:
        raise ValueError(f"Invalid sequence in chunk_id: {chunk_id}")

    source_type = "primary_literature" if source_prefix == "primary" else "research_literature"

    return {
        "source_prefix": source_prefix,
        "source_type": source_type,
        "genre": genre,
        "chapter_number": chapter_number,
        "sequence": sequence,
    }


# ============================================================================
# U00: metadata schema 验证常量与函数
# ============================================================================
REQUIRED_METADATA_FIELDS: list[str] = [
    "chunk_id", "chapter_number", "chapter_title",
    "dynasty", "genre", "source_type", "strategy",
    "paragraph_index", "paragraph_range", "sentence_range",
    "char_start", "char_end", "structural_unit",
    "source_file",
]

STRATEGY_REQUIRED_FIELDS: dict[str, list[str]] = {
    "opera_libretto": ["qupai_name", "act_index", "context_prefix_length"],
    "script": ["scene_index", "context_prefix_length"],
    "tanci": ["context_prefix_length"],
}

# source_file 与 source_type 的权威映射
SOURCE_FILE_MAP: dict[str, str] = {
    "primary_literature": "白蛇传文献选集.txt",
    "research_literature": "文本景观建构研究文献.txt",
}


def validate_chunk_metadata(documents: list) -> list[str]:
    """遍历所有 chunk，检查缺失字段、字段类型、chunk_id 唯一性。

    Returns:
        错误信息列表，空列表表示验证通过。
    """
    errors: list[str] = []
    seen_chunk_ids: set[str] = set()

    for i, doc in enumerate(documents):
        meta = doc.metadata
        chunk_id = meta.get("chunk_id", "")

        # 1. 必填字段检查
        for field in REQUIRED_METADATA_FIELDS:
            if field not in meta:
                errors.append(
                    f"chunk[{i}] chunk_id={chunk_id}: 缺少必填字段 '{field}'"
                )

        # 2. chunk_id 唯一性
        if chunk_id:
            if chunk_id in seen_chunk_ids:
                errors.append(
                    f"chunk[{i}]: chunk_id 重复: {chunk_id}"
                )
            seen_chunk_ids.add(chunk_id)

        # 3. chunk_id 格式校验
        if chunk_id:
            if not re.match(r'^(primary|research)_[a-z_]+_[^_]+_\d{5}$', chunk_id):
                errors.append(
                    f"chunk[{i}]: chunk_id 格式不合法: {chunk_id}"
                )

        # 4. chapter_number 类型安全
        chapter_number = meta.get("chapter_number")
        if chapter_number is not None and not isinstance(chapter_number, str):
            errors.append(
                f"chunk[{i}] chunk_id={chunk_id}: chapter_number 必须是 string 类型，"
                f"当前为 {type(chapter_number).__name__}"
            )

        # 5. source_file / source_type 一致性
        source_type = meta.get("source_type", "")
        source_file = meta.get("source_file", "")
        if source_type in SOURCE_FILE_MAP:
            expected_source_file = SOURCE_FILE_MAP[source_type]
            if source_file and source_file != expected_source_file:
                errors.append(
                    f"chunk[{i}] chunk_id={chunk_id}: source_file='{source_file}' "
                    f"与 source_type='{source_type}' 不一致，"
                    f"期望 source_file='{expected_source_file}'"
                )

        # 6. char_start < char_end
        char_start = meta.get("char_start", -1)
        char_end = meta.get("char_end", -1)
        if isinstance(char_start, int) and isinstance(char_end, int):
            if char_start > 0 and char_end > 0 and char_start >= char_end:
                errors.append(
                    f"chunk[{i}] chunk_id={chunk_id}: char_start({char_start}) >= char_end({char_end})"
                )

        # 7. context_prefix_length 一致性
        context_prefix = meta.get("context_prefix", "")
        context_prefix_length = meta.get("context_prefix_length", -1)
        if isinstance(context_prefix_length, int) and context_prefix_length >= 0:
            if context_prefix and context_prefix_length != len(context_prefix):
                errors.append(
                    f"chunk[{i}] chunk_id={chunk_id}: context_prefix_length({context_prefix_length}) "
                    f"与 len(context_prefix)({len(context_prefix)}) 不一致"
                )
            if not context_prefix and context_prefix_length != 0:
                errors.append(
                    f"chunk[{i}] chunk_id={chunk_id}: context_prefix 为空但 "
                    f"context_prefix_length={context_prefix_length} (应为0)"
                )

        # 8. paragraph_index 合法性
        # 使用 sentence_range / structural_unit 的策略不需要 paragraph_index
        # (它们跟踪句子级/结构级偏移，而非段落级)
        paragraph_index = meta.get("paragraph_index", -1)
        strategy = meta.get("strategy", "")
        _SENTENCE_LEVEL_STRATEGIES = (
            "sentence_semantic_merge", "structural_chapter_merge",
            "structural_act", "structural_scene", "tanci_scene_window",
        )
        if isinstance(paragraph_index, int) and paragraph_index < 0:
            if strategy not in _SENTENCE_LEVEL_STRATEGIES:
                errors.append(
                    f"chunk[{i}] chunk_id={chunk_id}: paragraph_index={paragraph_index} 不合法，"
                    f"策略 '{strategy}' 应为 >=0"
                )

        # 9. 策略特定字段检查
        strategy = meta.get("strategy", "")
        strategy_to_genre: dict[str, str] = {
            "structural_act": "opera_libretto",
            "structural_scene": "script",
            "tanci_scene_window": "tanci",
        }
        mapped_genre = strategy_to_genre.get(strategy, "")
        if mapped_genre and mapped_genre in STRATEGY_REQUIRED_FIELDS:
            for field in STRATEGY_REQUIRED_FIELDS[mapped_genre]:
                if field not in meta:
                    errors.append(
                        f"chunk[{i}] chunk_id={chunk_id}: 策略 '{strategy}' 缺少必填字段 '{field}'"
                    )

    return errors


# ============================================================================
# Document 数据类 (保留)
# ============================================================================
@dataclass
class Document:
    content: str
    metadata: dict = field(default_factory=dict)
    score: float = 0.0

    @property
    def page_content(self) -> str:
        return self.content


# ============================================================================
# 冻结分块快照：corpus.jsonl 是 chunk 边界与 chunk_id 的唯一事实来源
#
# 为什么需要：语义合并分块依赖 embedding 相似度，若每次重建都重新分块，
# 任何模型/代码/torch 版本变化都可能让块边界漂移 → chunk_id 变化，
# 导致以 chunk_id 为键的 LLM 标签、受控词表、向量缓存全部错位失效。
# 快照机制下，索引重建只重新 embed（确定性），分块结果永久冻结。
# ============================================================================
SNAPSHOT_PATH = (
    Path(__file__).resolve().parent.parent.parent.parent
    / "rag_eval" / "ground_truth" / "corpus.jsonl"
)


def load_chunk_snapshot(path: Optional[Path] = None) -> Optional[list]:
    """从冻结快照加载 Document 列表；快照不存在或损坏返回 None（触发重新分块）。"""
    p = Path(path) if path else SNAPSHOT_PATH
    if not p.exists():
        return None
    try:
        docs = []
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                doc_id = rec.get("doc_id") or rec.get("_id")
                if not doc_id:
                    continue
                meta = dict(rec.get("metadata") or {})
                meta["chunk_id"] = doc_id
                if rec.get("title"):
                    meta.setdefault("chapter_title", rec["title"])
                docs.append(Document(content=rec.get("text", ""), metadata=meta))
        if not docs:
            return None

        # 按章节阅读顺序（一→七→研究）+ 全局序号排序，
        # chunk_id 尾号是 chunker 的全局递增计数器，数值序即原文顺序
        chapter_order = {"一": 1, "二": 2, "三": 3, "四": 4,
                         "五": 5, "六": 6, "七": 7, "研究": 8}

        def _sort_key(d):
            cid = d.metadata.get("chunk_id", "")
            m = re.search(r"_(\d{5})$", cid)
            seq = int(m.group(1)) if m else 999999
            ch = d.metadata.get("chapter_number", "?")
            return (chapter_order.get(ch, 99), seq)

        docs.sort(key=_sort_key)
        return docs
    except Exception as e:
        logger.warning(f"分块快照读取失败({p})，将重新分块: {e}")
        return None


def save_chunk_snapshot(documents: list, path: Optional[Path] = None) -> None:
    """把分块结果写入冻结快照（doc_id == chunk_id），供后续重建直接复用。"""
    p = Path(path) if path else SNAPSHOT_PATH
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            for d in documents:
                meta = dict(d.metadata or {})
                cid = meta.get("chunk_id", "")
                f.write(json.dumps({
                    "doc_id": cid,
                    "title": meta.get("chapter_title", ""),
                    "text": d.content,
                    "metadata": meta,
                }, ensure_ascii=False, default=str) + "\n")
        logger.info(f"[CHUNK] 分块快照已保存: {p} ({len(documents)} 块)")
    except Exception as e:
        logger.warning(f"分块快照保存失败（不影响本次重建）: {e}")


# ============================================================================
# CU-05: Chunker 构造函数签名变更 + _wait_for_embedder
# ============================================================================
class Chunker:
    """体裁感知分块器：按体裁选择策略，支持语义合并和结构感知分块。"""

    # ---- 章回体白话小说正则 (CU-09) ----
    _CHAPTER_HUITITLE_PATTERN = re.compile(
        r'第[一二三四五六七八九十百千万\d]+[章节回]\s*[^\n]{2,40}'
    )
    _VERNACULAR_NARRATIVE_MARKERS = [
        '诗曰', '话说', '正是', '再表', '且说', '却说', '但见',
        '有诗为证', '词云', '赞曰', '正是那', '毕竟', '未知',
    ]

    # ---- 文言文实体识别正则 ----
    _CLASSICAL_PERSON_NAME_PATTERN = re.compile(
        r'([李王张刘赵陈杨黄周吴'
        r'徐孙马胡朱郭何罗高林'
        r'郑梁谢宋唐许邓冯韩曹'
        r'曾彭萧蔡潘田董袁于余'
        r'叶蒋杜苏魏吕丁沈任姚'
        r'卢姜崔钟谭陆汪范金石'
        r'廖贾夏付方白邹熊孟秦'
        r'阎薛侯雷龙段郝孔邵史'
        r'毛常万顾赖武康贺严尹'
        r'钱施牛洪龚]'
        r'[一-鿿]{1,3})')

    _CLASSICAL_PLACE_NAME_PATTERN = re.compile(
        r'(长安|洛阳|临安|杭州|钱塘|苏州|镇江|扬州|金陵|成都|'
        r'西湖|金山|峨眉|龙虎|终南|翠屏|昆仑|孤山|庐山|青城|'
        r'庄严寺|安化门|奉诚园|安邑里|永宁里|清波门|钱塘门|涌金门)')

    _CLASSICAL_TIME_WORD_PATTERN = re.compile(
        r'(元和|乾符|淳熙|绍兴|至正|万历|乾隆|嘉庆|'
        r'元和中|乾符间|是时|俄而|少顷|良久|翌日|明日|'
        r'须臾|黄昏|将暮|日暮|辰巳)')

    # ---- 戏曲正则 (CU-11) ----
    _OPERA_ACT_PATTERN = re.compile(r'第[一二三四五六七八九十百千万\d]+[章节出折][^\n]*')
    _OPERA_QUPAI_PATTERN = re.compile(r'【([^】]+)】')
    _OPERA_ROLE_PATTERN = re.compile(r'〔([^〕]+)〕')
    _OPERA_ROLE_NAMES = {'生', '旦', '净', '丑', '外', '贴', '末', '杂', '众', '副'}

    # ---- 戏曲裸幕标题检测 ----
    _OPERA_BARE_ACT_PATTERN = re.compile(
        r'(?<=\n\n)([一-鿿]{2,6})(?=\n\n)'
    )

    # ---- 剧本正则 (CU-12) ----
    _SCRIPT_SCENE_PATTERN = re.compile(r'第[一二三四五六七八九十百千万\d]+场[^\n]*')
    _SCRIPT_BANSHI_PATTERN = re.compile(r'[（(](内唱|内白|唱|白|说白|道白|念)[^）)]*[）)]')

    # ---- 弹词正则 (CU-13) ----
    _TANCI_SCENE_BREAK_PATTERN = re.compile(r'(不题|再云|再说)')
    _TANCI_INNER_MARKER_PATTERN = re.compile(r'(曰引|诗曰|唱|白|曰|表|云)')
    _TANCI_REPETITION_MARK = re.compile(r'[七匕]')
    _TANCI_ARIA_LINE_PATTERN = re.compile(r'[一-鿿]{6,8}[，。！？、]?')
    _TANCI_ROLE_NAMES_PATTERN = re.compile(r'(旦|生|净|丑|外|贴|末|众|副|小生|老旦)')

    def __init__(self, data_dir: Path, embedder=None,
                 progress_callback=None, embed_batch_size: int = 64):
        self.data_dir = data_dir
        self.embedder = embedder
        self.progress_callback = progress_callback  # U12: (current, total, title) -> None
        self._chunk_counter: int = 0  # U02: 全局 chunk 计数器
        self._embedding_cache: dict[str, list[float]] = {}  # U14: embedding 缓存
        # 句子 embedding 批次大小：重建索引等大内存场景调小（如 16）以降低峰值内存
        self.embed_batch_size = max(1, embed_batch_size)

    # ========================================================================
    # U02: chunk_id 生成
    # ========================================================================
    def _next_chunk_id(self, genre: str, chapter_number: str,
                       source_type: str = "primary_literature") -> str:
        """生成全局唯一 chunk_id。

        Args:
            genre: 体裁名 (如 "huaben", "opera_libretto")
            chapter_number: 章节号 (中文数字 "一"~"七" 或 "研究")
            source_type: "primary_literature" | "research_literature"

        Returns:
            格式: "{source_prefix}_{genre}_{chapter_number}_{seq:05d}"
        """
        self._chunk_counter += 1
        source_prefix = "primary" if source_type == "primary_literature" else "research"
        return f"{source_prefix}_{genre}_{chapter_number}_{self._chunk_counter:05d}"

    # ========================================================================
    # U03: 字符偏移辅助方法
    # ========================================================================
    @staticmethod
    def _find_char_range(text: str, chunk_body: str,
                         search_start: int = 0) -> tuple[int, int]:
        """在 text 中从 search_start 开始精确查找 chunk_body 的位置。

        Returns:
            (start, end) 字符偏移。未找到时返回 (-1, -1)。
        """
        pos = text.find(chunk_body, search_start)
        if pos >= 0:
            return (pos, pos + len(chunk_body))
        return (-1, -1)

    # ========================================================================
    # U14: 带实例缓存的 embedding 计算
    # ========================================================================
    def _cached_embed(self, text: str) -> list[float]:
        """带实例缓存的 embedding 计算。"""
        cache_key = f"{text[:80]}|{len(text)}|{hash(text)}"
        if cache_key not in self._embedding_cache:
            self._embedding_cache[cache_key] = self.embedder.embed([text])[0]
        return self._embedding_cache[cache_key]

    # ========================================================================
    # U01: 统一 metadata 构建辅助方法
    # ========================================================================
    @staticmethod
    def _build_unified_meta(
        genre: str,
        strategy: str,
        chapter_number: str,
        chapter_title: str,
        source_type: str,
        source_file: str,
        paragraph_index: int,
        paragraph_range,
        sentence_range,
        char_start: int,
        char_end: int,
        structural_unit: str,
        context_prefix: str,
        context_prefix_length: int,
        dynasty: str | None = None,
        **extra,
    ) -> dict:
        """构建统一 metadata dict，所有策略共用。

        所有字段显式写入，即使值为空字符串或 None。
        chunk_id 由 _make_chunk_meta 在之后设置。
        """
        meta: dict = {
            "chunk_id": "",  # 由 _make_chunk_meta 覆盖
            "chapter_number": chapter_number,
            "chapter_title": chapter_title,
            "dynasty": dynasty,
            "genre": genre,
            "source_type": source_type,
            "source_file": source_file,
            "strategy": strategy,
            "paragraph_index": paragraph_index,
            "paragraph_range": paragraph_range,
            "sentence_range": sentence_range,
            "char_start": char_start,
            "char_end": char_end,
            "structural_unit": structural_unit,
            "context_prefix": context_prefix,
            "context_prefix_length": context_prefix_length,
        }
        meta.update(extra)
        return meta

    def _make_chunk_meta(
        self,
        genre: str,
        strategy: str,
        chapter_number: str,
        chapter_title: str,
        source_type: str,
        source_file: str,
        paragraph_index: int,
        paragraph_range,
        sentence_range,
        char_start: int,
        char_end: int,
        structural_unit: str,
        context_prefix: str,
        context_prefix_length: int,
        dynasty: str | None = None,
        **extra,
    ) -> dict:
        """构建统一 metadata 并分配 chunk_id。"""
        meta = self._build_unified_meta(
            genre=genre,
            strategy=strategy,
            chapter_number=chapter_number,
            chapter_title=chapter_title,
            source_type=source_type,
            source_file=source_file,
            paragraph_index=paragraph_index,
            paragraph_range=paragraph_range,
            sentence_range=sentence_range,
            char_start=char_start,
            char_end=char_end,
            structural_unit=structural_unit,
            context_prefix=context_prefix,
            context_prefix_length=context_prefix_length,
            dynasty=dynasty,
            **extra,
        )
        meta["chunk_id"] = self._next_chunk_id(genre, chapter_number, source_type)
        return meta

    def _wait_for_embedder(self, timeout_seconds: float = 120.0):
        """阻塞等待 embedder 就绪，超时抛出 RuntimeError。"""
        if self.embedder is None:
            raise RuntimeError("embedder 未设置，无法执行语义合并分块策略")
        if self.embedder.is_ready:
            return
        logger.info(f"等待 Embedding 模型就绪 (最多 {timeout_seconds}s)...")
        waited = 0.0
        while not self.embedder.is_ready and waited < timeout_seconds:
            time.sleep(2.0)
            waited += 2.0
        if not self.embedder.is_ready:
            raise RuntimeError(
                f"Embedding 模型在 {timeout_seconds}s 内未就绪，无法执行语义合并分块"
            )
        logger.info(f"Embedding 模型就绪 (等待了 {waited:.0f}s)")

    def _clean_text(self, text: str) -> str:
        """基本清洗：统一换行符、清理 Unicode 空白、折叠多余空行。"""
        text = re.sub(r'\r\n|\r', '\n', text)
        _UNICODE_WHITESPACE_REPLACEMENTS = [
            (' ', ' '), ('　', ' '), (' ', ' '), (' ', ' '),
            (' ', ' '), (' ', ' '), (' ', ' '), (' ', ' '),
            (' ', ' '), (' ', ' '), (' ', ' '), (' ', ' '),
            (' ', ' '), (' ', ' '), (' ', ' '), ('﻿', ''),
        ]
        for char, replacement in _UNICODE_WHITESPACE_REPLACEMENTS:
            text = text.replace(char, replacement)
        text = re.sub(r' {2,}', ' ', text)
        text = re.sub(r'\n{3,}', '\n\n', text)
        text = text.strip()
        return text

    # ========================================================================
    # CU-06: 文言文策略 —— 句子级语义合并分块 (U01/U02/U03/U13 更新)
    # ========================================================================
    def _chunk_sentence_semantic_merge(
        self, text: str, config: GenreChunkConfig, chapter_number: str = "",
        chapter_title: str = "", dynasty: str | None = None,
        source_type: str = "primary_literature",
        chunk_id_prefix: str = "", source_file: str = "",
    ) -> list[Document]:
        similarity_threshold = getattr(config, 'similarity_threshold', 0.0) or 0.0
        max_chunk_chars = config.max_chunk_chars
        min_chunk_chars = config.min_chunk_chars

        self._wait_for_embedder()

        text = self._clean_text(text)
        if not text:
            return []

        sentences = _split_sentences(text)
        if not sentences:
            return []

        # Step 1: 批量计算 embedding + 追踪句子偏移 (U03/U13)
        sentence_embeddings = []
        sentence_offsets: list[int] = []  # U03: 每个句子在 text 中的起始偏移
        current_offset = 0
        cleaned_sentences: list[str] = []
        for s in sentences:
            cleaned = s.strip()
            if not cleaned:
                sentence_embeddings.append(None)
                sentence_offsets.append(current_offset)
                cleaned_sentences.append("")
                continue
            cleaned_sentences.append(cleaned)
            sentence_embeddings.append(None)  # placeholder，批量 embedding 阶段填充
            # 在 text 中定位当前句子的偏移
            pos = text.find(cleaned, current_offset)
            if pos >= 0:
                sentence_offsets.append(pos)
                current_offset = pos + len(cleaned)
            else:
                sentence_offsets.append(current_offset)

        # U13: 批量 embedding
        non_empty_indices = [i for i, s in enumerate(cleaned_sentences) if s]
        non_empty_texts = [cleaned_sentences[i] for i in non_empty_indices]
        if non_empty_texts:
            try:
                batch_embs = _compute_embeddings_batch(non_empty_texts, self.embedder, batch_size=self.embed_batch_size)
                for idx, emb in zip(non_empty_indices, batch_embs):
                    sentence_embeddings[idx] = emb
            except Exception as e:
                logger.warning(f"批量 embedding 计算失败: {e}，回退到逐句计算")
                for i in non_empty_indices:
                    try:
                        sentence_embeddings[i] = _compute_embedding(cleaned_sentences[i], self.embedder)
                    except Exception as e2:
                        logger.warning(f"Embedding 计算失败: {e2}")
                        sentence_embeddings[i] = None

        # Step 2: 贪心合并 (与原来相同)
        raw_groups: list[list[int]] = []
        current_group: list[int] = []
        current_text_len = 0

        for i, s in enumerate(sentences):
            s_clean = s.strip()
            s_len = len(s_clean)
            if s_len == 0:
                continue

            if not current_group:
                current_group = [i]
                current_text_len = s_len
                continue

            emb_curr = sentence_embeddings[i]

            should_merge = False
            if emb_curr is not None and current_group:
                group_embs = [
                    sentence_embeddings[j]
                    for j in current_group
                    if sentence_embeddings[j] is not None
                ]
                if group_embs:
                    centroid = np.mean(np.asarray(group_embs, dtype=np.float64), axis=0)
                    sim = _cosine_similarity(centroid, emb_curr)
                    if sim >= similarity_threshold and (current_text_len + s_len) <= max_chunk_chars:
                        should_merge = True

            # 实体名辅助合并
            if not should_merge and emb_curr is not None and group_embs:
                sim = _cosine_similarity(centroid, emb_curr)
                if sim >= similarity_threshold - 0.10:
                    group_text = ''.join(sentences[j] for j in current_group)
                    group_persons = set(self._CLASSICAL_PERSON_NAME_PATTERN.findall(group_text))
                    curr_persons = set(self._CLASSICAL_PERSON_NAME_PATTERN.findall(s_clean))
                    if group_persons and curr_persons and group_persons & curr_persons:
                        if (current_text_len + s_len) <= max_chunk_chars:
                            should_merge = True
                    if not should_merge:
                        group_places = set(self._CLASSICAL_PLACE_NAME_PATTERN.findall(group_text))
                        curr_places = set(self._CLASSICAL_PLACE_NAME_PATTERN.findall(s_clean))
                        if group_places and curr_places and group_places & curr_places:
                            if (current_text_len + s_len) <= max_chunk_chars:
                                should_merge = True
                    if not should_merge:
                        group_times = set(self._CLASSICAL_TIME_WORD_PATTERN.findall(group_text))
                        curr_times = set(self._CLASSICAL_TIME_WORD_PATTERN.findall(s_clean))
                        if group_times and curr_times and group_times & curr_times:
                            if (current_text_len + s_len) <= max_chunk_chars:
                                should_merge = True

            if should_merge:
                current_group.append(i)
                current_text_len += s_len
            else:
                raw_groups.append(current_group)
                current_group = [i]
                current_text_len = s_len

        if current_group:
            raw_groups.append(current_group)

        # Step 3: < min_chunk_chars 的 chunk 合并到前一个
        merged_groups: list[list[int]] = []
        for group in raw_groups:
            group_text = ''.join(sentences[i].strip() for i in group)
            if len(group_text) < min_chunk_chars and merged_groups:
                merged_groups[-1].extend(group)
            else:
                merged_groups.append(group)

        # U01/U03: 计算段落索引 (clean_text 后的 \n\n 分割)
        clean_paragraphs = _split_paragraphs(text)

        # Step 4: 创建 Document
        documents = []
        for group in merged_groups:
            sub_texts: list[str] = []
            for i in group:
                s_clean = sentences[i].strip()
                if len(s_clean) > max_chunk_chars:
                    parts = _split_long_paragraph_at_sentence_boundaries(s_clean, max_chunk_chars)
                    sub_texts.extend(parts)
                else:
                    sub_texts.append(s_clean)

            # U03: 计算 group 的 char_start (取第一个有效句子的偏移)
            char_start = -1
            for i in group:
                if i < len(sentence_offsets) and sentence_offsets[i] >= 0:
                    char_start = sentence_offsets[i]
                    break

            # U01: 计算 paragraph_index (chunk 起始句子所属段落)
            para_idx = -1
            if char_start >= 0:
                cumulative = 0
                for p_i, para in enumerate(clean_paragraphs):
                    para_end = cumulative + len(para)
                    if cumulative <= char_start < para_end + 2:  # +2 for \n\n
                        para_idx = p_i
                        break
                    cumulative = para_end + 2  # \n\n separator

            current_parts: list[str] = []
            current_len = 0
            for part in sub_texts:
                if current_len + len(part) <= max_chunk_chars:
                    current_parts.append(part)
                    current_len += len(part)
                else:
                    if current_parts:
                        content = ''.join(current_parts)
                        char_end = char_start + len(content) if char_start >= 0 else -1
                        meta = self._make_chunk_meta(
                            genre=Genre.CLASSICAL_CHINESE.value,
                            strategy="sentence_semantic_merge",
                            chapter_number=chapter_number,
                            chapter_title=chapter_title,
                            source_type=source_type,
                            source_file=source_file,
                            paragraph_index=para_idx,
                            paragraph_range=None,
                            sentence_range=[group[0], group[-1]],
                            char_start=char_start,
                            char_end=char_end,
                            structural_unit="",
                            context_prefix="",
                            context_prefix_length=0,
                            dynasty=dynasty,
                        )
                        documents.append(Document(content=content, metadata=meta))
                    current_parts = [part]
                    current_len = len(part)
            if current_parts:
                content = ''.join(current_parts)
                char_end = char_start + len(content) if char_start >= 0 else -1
                meta = self._make_chunk_meta(
                    genre=Genre.CLASSICAL_CHINESE.value,
                    strategy="sentence_semantic_merge",
                    chapter_number=chapter_number,
                    chapter_title=chapter_title,
                    source_type=source_type,
                    source_file=source_file,
                    paragraph_index=para_idx,
                    paragraph_range=None,
                    sentence_range=[group[0], group[-1]],
                    char_start=char_start,
                    char_end=char_end,
                    structural_unit="",
                    context_prefix="",
                    context_prefix_length=0,
                    dynasty=dynasty,
                )
                documents.append(Document(content=content, metadata=meta))

        return documents

    # ========================================================================
    # CU-07: 话本小说策略 —— 段落级语义合并分块 (现已改为句子级)
    # ========================================================================
    def _chunk_paragraph_semantic_merge_huaben(
        self, text: str, config: GenreChunkConfig, chapter_number: str = "",
        chapter_title: str = "", dynasty: str | None = None,
        source_type: str = "primary_literature",
        chunk_id_prefix: str = "", source_file: str = "",
    ) -> list[Document]:
        similarity_threshold = getattr(config, 'similarity_threshold', 0.0) or 0.0
        max_chunk_chars = config.max_chunk_chars
        min_chunk_chars = config.min_chunk_chars

        self._wait_for_embedder()

        text = self._clean_text(text)
        if not text:
            return []

        paragraphs = _split_paragraphs(text)
        if not paragraphs:
            return []

        all_sentences: list[str] = []
        for p in paragraphs:
            p_sentences = _split_sentences(p)
            for s in p_sentences:
                s_clean = s.strip()
                if s_clean:
                    all_sentences.append(s_clean)

        if not all_sentences:
            return []

        # U13: 批量 embedding
        sentence_embeddings = [None] * len(all_sentences)
        non_empty_indices = [i for i, s in enumerate(all_sentences) if s.strip()]
        non_empty_texts = [all_sentences[i] for i in non_empty_indices]
        if non_empty_texts:
            try:
                batch_embs = _compute_embeddings_batch(non_empty_texts, self.embedder, batch_size=self.embed_batch_size)
                for idx, emb in zip(non_empty_indices, batch_embs):
                    sentence_embeddings[idx] = emb
            except Exception as e:
                logger.warning(f"批量 embedding 失败: {e}，回退逐句")
                for i in non_empty_indices:
                    try:
                        sentence_embeddings[i] = _compute_embedding(all_sentences[i], self.embedder)
                    except Exception as e2:
                        logger.warning(f"Embedding 计算失败: {e2}")

        merged_groups = _sentence_centroid_semantic_merge(
            all_sentences, sentence_embeddings,
            similarity_threshold, max_chunk_chars, min_chunk_chars,
        )

        documents = []
        for group in merged_groups:
            sub_texts: list[str] = []
            for i in group:
                s_text = all_sentences[i]
                if len(s_text) > max_chunk_chars:
                    parts = _split_long_paragraph_at_sentence_boundaries(s_text, max_chunk_chars)
                    sub_texts.extend(parts)
                else:
                    sub_texts.append(s_text)

            current_parts: list[str] = []
            current_len = 0
            for part in sub_texts:
                if current_len + len(part) <= max_chunk_chars:
                    current_parts.append(part)
                    current_len += len(part)
                else:
                    if current_parts:
                        content = '\n\n'.join(current_parts)
                        meta = self._make_chunk_meta(
                            genre=Genre.HUABEN.value,
                            strategy="sentence_semantic_merge",
                            chapter_number=chapter_number,
                            chapter_title=chapter_title,
                            source_type=source_type,
                            source_file=source_file,
                            paragraph_index=-1,
                            paragraph_range=None,
                            sentence_range=[group[0], group[-1]],
                            char_start=-1,
                            char_end=-1,
                            structural_unit="",
                            context_prefix="",
                            context_prefix_length=0,
                            dynasty=dynasty,
                        )
                        documents.append(Document(content=content, metadata=meta))
                    current_parts = [part]
                    current_len = len(part)
            if current_parts:
                content = '\n\n'.join(current_parts)
                meta = self._make_chunk_meta(
                    genre=Genre.HUABEN.value,
                    strategy="sentence_semantic_merge",
                    chapter_number=chapter_number,
                    chapter_title=chapter_title,
                    source_type=source_type,
                    source_file=source_file,
                    paragraph_index=-1,
                    paragraph_range=None,
                    sentence_range=[group[0], group[-1]],
                    char_start=-1,
                    char_end=-1,
                    structural_unit="",
                    context_prefix="",
                    context_prefix_length=0,
                    dynasty=dynasty,
                )
                documents.append(Document(content=content, metadata=meta))

        return documents

    # ========================================================================
    # CU-08: 拟话本小说策略 —— 纯段落分块 (现已改为句子级语义合并)
    # ========================================================================
    def _chunk_semantic_paragraph(
        self, text: str, config: GenreChunkConfig, chapter_number: str = "",
        chapter_title: str = "", dynasty: str | None = None,
        source_type: str = "primary_literature",
        chunk_id_prefix: str = "", source_file: str = "",
    ) -> list[Document]:
        similarity_threshold = getattr(config, 'similarity_threshold', 0.0) or 0.0
        max_chunk_chars = config.max_chunk_chars
        min_chunk_chars = config.min_chunk_chars

        if similarity_threshold > 0:
            self._wait_for_embedder()

        text = self._clean_text(text)
        if not text:
            return []

        paragraphs = _split_paragraphs(text)
        if not paragraphs:
            return []

        all_sentences: list[str] = []
        for p in paragraphs:
            p_sentences = _split_sentences(p)
            for s in p_sentences:
                s_clean = s.strip()
                if s_clean:
                    all_sentences.append(s_clean)

        if not all_sentences:
            return []

        # U13: 批量 embedding
        sentence_embeddings = []
        if similarity_threshold > 0:
            sentence_embeddings = [None] * len(all_sentences)
            non_empty_indices = [i for i, s in enumerate(all_sentences) if s.strip()]
            non_empty_texts = [all_sentences[i] for i in non_empty_indices]
            if non_empty_texts:
                try:
                    batch_embs = _compute_embeddings_batch(non_empty_texts, self.embedder, batch_size=self.embed_batch_size)
                    for idx, emb in zip(non_empty_indices, batch_embs):
                        sentence_embeddings[idx] = emb
                except Exception as e:
                    logger.warning(f"批量 embedding 失败: {e}，回退逐句")
                    for i in non_empty_indices:
                        try:
                            sentence_embeddings[i] = _compute_embedding(all_sentences[i], self.embedder)
                        except Exception as e2:
                            logger.warning(f"Embedding 计算失败: {e2}")

            merged_groups = _sentence_centroid_semantic_merge(
                all_sentences, sentence_embeddings,
                similarity_threshold, max_chunk_chars, min_chunk_chars,
            )
        else:
            merged_groups: list[list[int]] = []
            current_group: list[int] = []
            current_len = 0
            for i, s in enumerate(all_sentences):
                s_len = len(s)
                if current_len + s_len <= max_chunk_chars:
                    current_group.append(i)
                    current_len += s_len
                else:
                    if current_group:
                        merged_groups.append(current_group)
                    current_group = [i]
                    current_len = s_len
            if current_group:
                merged_groups.append(current_group)

        documents = []
        for group in merged_groups:
            sub_texts: list[str] = []
            for i in group:
                s_text = all_sentences[i]
                if len(s_text) > max_chunk_chars:
                    parts = _split_long_paragraph_at_sentence_boundaries(s_text, max_chunk_chars)
                    sub_texts.extend(parts)
                else:
                    sub_texts.append(s_text)

            current_parts: list[str] = []
            current_len = 0
            for part in sub_texts:
                if current_len + len(part) <= max_chunk_chars:
                    current_parts.append(part)
                    current_len += len(part)
                else:
                    if current_parts:
                        content = '\n\n'.join(current_parts)
                        meta = self._make_chunk_meta(
                            genre=Genre.IMITATION_HUABEN.value,
                            strategy="sentence_semantic_merge",
                            chapter_number=chapter_number,
                            chapter_title=chapter_title,
                            source_type=source_type,
                            source_file=source_file,
                            paragraph_index=-1,
                            paragraph_range=None,
                            sentence_range=[group[0], group[-1]],
                            char_start=-1,
                            char_end=-1,
                            structural_unit="",
                            context_prefix="",
                            context_prefix_length=0,
                            dynasty=dynasty,
                        )
                        documents.append(Document(content=content, metadata=meta))
                    current_parts = [part]
                    current_len = len(part)
            if current_parts:
                content = '\n\n'.join(current_parts)
                meta = self._make_chunk_meta(
                    genre=Genre.IMITATION_HUABEN.value,
                    strategy="sentence_semantic_merge",
                    chapter_number=chapter_number,
                    chapter_title=chapter_title,
                    source_type=source_type,
                    source_file=source_file,
                    paragraph_index=-1,
                    paragraph_range=None,
                    sentence_range=[group[0], group[-1]],
                    char_start=-1,
                    char_end=-1,
                    structural_unit="",
                    context_prefix="",
                    context_prefix_length=0,
                    dynasty=dynasty,
                )
                documents.append(Document(content=content, metadata=meta))

        return documents

    # ========================================================================
    # CU-09: 章回体白话小说策略 —— 回目结构分块 + 句子语义合并
    # ========================================================================
    def _chunk_structural_chapter_merge(
        self, text: str, config: GenreChunkConfig, chapter_number: str = "",
        chapter_title: str = "", dynasty: str | None = None,
        source_type: str = "primary_literature",
        chunk_id_prefix: str = "", source_file: str = "",
    ) -> list[Document]:
        similarity_threshold = getattr(config, 'similarity_threshold', 0.0) or 0.0
        max_chunk_chars = config.max_chunk_chars
        min_chunk_chars = config.min_chunk_chars

        self._wait_for_embedder()

        text = self._clean_text(text)
        if not text:
            return []

        all_documents: list[Document] = []

        hui_parts = self._CHAPTER_HUITITLE_PATTERN.split(text)

        current_hui_title = ""
        for idx, part in enumerate(hui_parts):
            part = part.strip()
            if not part:
                continue

            if self._CHAPTER_HUITITLE_PATTERN.match(part):
                current_hui_title = part
                continue

            paragraphs = _split_paragraphs(part)
            if not paragraphs:
                continue

            all_sentences: list[str] = []
            for p in paragraphs:
                p_sentences = _split_sentences(p)
                for s in p_sentences:
                    s_clean = s.strip()
                    if s_clean:
                        all_sentences.append(s_clean)

            if not all_sentences:
                continue

            # U13: 批量 embedding
            sentence_embeddings = [None] * len(all_sentences)
            non_empty_indices = [i for i, s in enumerate(all_sentences) if s.strip()]
            non_empty_texts = [all_sentences[i] for i in non_empty_indices]
            if non_empty_texts:
                try:
                    batch_embs = _compute_embeddings_batch(non_empty_texts, self.embedder, batch_size=self.embed_batch_size)
                    for idx, emb in zip(non_empty_indices, batch_embs):
                        sentence_embeddings[idx] = emb
                except Exception as e:
                    logger.warning(f"批量 embedding 失败: {e}，回退逐句")
                    for i in non_empty_indices:
                        try:
                            sentence_embeddings[i] = _compute_embedding(all_sentences[i], self.embedder)
                        except Exception as e2:
                            logger.warning(f"Embedding 计算失败: {e2}")

            merged_groups = _sentence_centroid_semantic_merge(
                all_sentences, sentence_embeddings,
                similarity_threshold, max_chunk_chars, min_chunk_chars,
            )

            for group in merged_groups:
                sub_texts: list[str] = []
                for i in group:
                    s_text = all_sentences[i]
                    if len(s_text) > max_chunk_chars:
                        parts = _split_long_paragraph_at_sentence_boundaries(s_text, max_chunk_chars)
                        sub_texts.extend(parts)
                    else:
                        sub_texts.append(s_text)

                current_parts: list[str] = []
                current_len = 0
                for part in sub_texts:
                    if current_len + len(part) <= max_chunk_chars:
                        current_parts.append(part)
                        current_len += len(part)
                    else:
                        if current_parts:
                            content = '\n\n'.join(current_parts)
                            meta = self._make_chunk_meta(
                                genre=Genre.CHAPTER_VERNACULAR.value,
                                strategy="structural_chapter_merge",
                                chapter_number=chapter_number,
                                chapter_title=chapter_title,
                                source_type=source_type,
                                source_file=source_file,
                                paragraph_index=-1,
                                paragraph_range=None,
                                sentence_range=[group[0], group[-1]],
                                char_start=-1,
                                char_end=-1,
                                structural_unit=current_hui_title,
                                context_prefix="",
                                context_prefix_length=0,
                                dynasty=dynasty,
                            )
                            all_documents.append(Document(content=content, metadata=meta))
                        current_parts = [part]
                        current_len = len(part)

                if current_parts:
                    content = '\n\n'.join(current_parts)
                    meta = self._make_chunk_meta(
                        genre=Genre.CHAPTER_VERNACULAR.value,
                        strategy="structural_chapter_merge",
                        chapter_number=chapter_number,
                        chapter_title=chapter_title,
                        source_type=source_type,
                        source_file=source_file,
                        paragraph_index=-1,
                        paragraph_range=None,
                        sentence_range=[group[0], group[-1]],
                        char_start=-1,
                        char_end=-1,
                        structural_unit=current_hui_title,
                        context_prefix="",
                        context_prefix_length=0,
                        dynasty=dynasty,
                    )
                    all_documents.append(Document(content=content, metadata=meta))

        return all_documents

    # ========================================================================
    # CU-10: 研究文献策略 —— 段落级语义合并 + 章节标题边界识别
    # ========================================================================
    _RESEARCH_SECTION_TITLE_PATTERN = re.compile(
        r'^[\s]*('
        r'前\s*言|绪\s*论|结\s*[语论]|参考\s*文献|附\s*录'
        r'|[（(][一二三四五六七八九十]+[）)]'
        r'|[（(]\d+[）)]'                                    # 匹配 (2), （3）等阿拉伯数字标题
        r'|\d+(?:\.\d+)*[\.\、]'
        r'|第[一二三四五六七八九十\d]+[章节]'
        r'|[一二三四五六七八九十]{1,3}\s{1,3}[^\n]{1,80}'
        r')',
        re.MULTILINE,
    )

    _IMAGE_CAPTION_PATTERN = re.compile(r'^图\s*\d+')
    _TABLE_CAPTION_PATTERN = re.compile(r'^表\s*\d+')
    _TABLE_ROW_PATTERN = re.compile(r'\t|[ ]{4,}')

    @staticmethod
    def _convert_table_to_structured(table_rows: list[str]) -> str:
        """将表格原始行列表转换为结构化文本。"""
        if not table_rows:
            return ""
        split_pattern = re.compile(r'\t+| {4,}')
        parsed_rows: list[list[str]] = []
        for row in table_rows:
            cells = [c.strip() for c in split_pattern.split(row.strip()) if c.strip()]
            if cells:
                parsed_rows.append(cells)
        if len(parsed_rows) < 2:
            return '\n'.join(table_rows)
        headers = parsed_rows[0]
        data_rows = parsed_rows[1:]
        lines = ["[表格数据]"]
        for row in data_rows:
            padded = row + [''] * (len(headers) - len(row))
            parts = [f"{h}: {v}" for h, v in zip(headers, padded) if v]
            if parts:
                lines.append("- " + ", ".join(parts))
        return '\n'.join(lines)

    def _detect_and_convert_tables(self, paragraphs: list[str]) -> list[str]:
        """检测段落列表中的表格区域, 转换为结构化文本。"""
        table_title_indices: list[int] = []
        for i, p in enumerate(paragraphs):
            if self._TABLE_CAPTION_PATTERN.match(p.strip()):
                table_title_indices.append(i)
        if not table_title_indices:
            return paragraphs
        table_regions: list[tuple[int, int, int]] = []
        for title_idx in sorted(table_title_indices):
            data_end = title_idx
            data_start = title_idx - 1
            while data_start >= 0:
                p = paragraphs[data_start].strip()
                if not p:
                    data_start -= 1
                    continue
                if self._TABLE_ROW_PATTERN.search(p) or len(p) < 60:
                    data_start -= 1
                else:
                    break
            data_start += 1
            if data_start < data_end:
                table_regions.append((data_start, data_end, title_idx))
        if not table_regions:
            return paragraphs
        result: list[str] = []
        last_end = 0
        for data_start, data_end, title_idx in table_regions:
            result.extend(paragraphs[last_end:data_start])
            table_paragraphs = paragraphs[data_start:data_end]
            structured = self._convert_table_to_structured(table_paragraphs)
            result.append(structured)
            result.append(paragraphs[title_idx])
            last_end = title_idx + 1
        result.extend(paragraphs[last_end:])
        return result

    def _chunk_research_literature(
        self, text: str, config: GenreChunkConfig, chapter_number: str = "",
        chapter_title: str = "", dynasty: str | None = None,
        source_type: str = "research_literature",
        chunk_id_prefix: str = "", source_file: str = "",
    ) -> list[Document]:
        similarity_threshold = getattr(config, 'similarity_threshold', 0.0) or 0.0
        max_chunk_chars = config.max_chunk_chars
        min_chunk_chars = config.min_chunk_chars

        self._wait_for_embedder()

        text = self._clean_text(text)
        if not text:
            return []

        paragraphs = _split_paragraphs(text)
        if not paragraphs:
            return []

        paragraphs = self._detect_and_convert_tables(paragraphs)

        section_boundaries: set[int] = set()
        section_titles: dict[int, str] = {}
        image_boundaries: set[int] = set()
        table_boundaries: set[int] = set()

        for i, p in enumerate(paragraphs):
            p_stripped = p.strip()
            if self._RESEARCH_SECTION_TITLE_PATTERN.match(p_stripped):
                section_boundaries.add(i)
                section_titles[i] = p_stripped
                continue
            if self._IMAGE_CAPTION_PATTERN.match(p_stripped):
                image_boundaries.add(i)
            if self._TABLE_CAPTION_PATTERN.match(p_stripped):
                table_boundaries.add(i)

        # U13: 批量 embedding
        para_embeddings = [None] * len(paragraphs)
        para_texts = [p.strip() for p in paragraphs]
        if para_texts:
            try:
                batch_embs = _compute_embeddings_batch(para_texts, self.embedder, batch_size=self.embed_batch_size)
                for idx, emb in enumerate(batch_embs):
                    para_embeddings[idx] = emb
            except Exception as e:
                logger.warning(f"批量 embedding 失败: {e}，回退逐句")
                for i, p in enumerate(paragraphs):
                    try:
                        para_embeddings[i] = _compute_embedding(p.strip(), self.embedder)
                    except Exception as e2:
                        logger.warning(f"Embedding 计算失败: {e2}")

        raw_groups: list[list[int]] = []
        current_group: list[int] = []
        current_text_len = 0
        current_section_path: list[str] = []

        for i, p in enumerate(paragraphs):
            p_len = len(p)
            if p_len == 0:
                continue

            if i in section_boundaries:
                if current_group:
                    raw_groups.append(current_group)
                    current_group = []
                    current_text_len = 0
                raw_groups.append([i])
                current_section_path.append(section_titles[i])
                continue

            if i in image_boundaries or i in table_boundaries:
                if current_group:
                    raw_groups.append(current_group)
                    current_group = []
                    current_text_len = 0

            if not current_group:
                current_group = [i]
                current_text_len = p_len
                continue

            emb_curr = para_embeddings[i]
            prev_emb = None
            for j in reversed(current_group):
                prev_emb = para_embeddings[j]
                if prev_emb is not None:
                    break

            should_merge = False
            if emb_curr is not None and prev_emb is not None:
                sim = _cosine_similarity(prev_emb, emb_curr)
                if sim >= similarity_threshold and (current_text_len + p_len) <= max_chunk_chars:
                    should_merge = True

            if should_merge:
                current_group.append(i)
                current_text_len += p_len
            else:
                raw_groups.append(current_group)
                current_group = [i]
                current_text_len = p_len

        if current_group:
            raw_groups.append(current_group)

        # Step 3: < min_chunk_chars 的段落合并到相邻 chunk
        merged_groups: list[list[int]] = []
        for group in raw_groups:
            if len(group) == 1 and group[0] in section_boundaries:
                merged_groups.append(group)
                continue
            group_text = ''.join(paragraphs[i] for i in group)
            if len(group_text) < min_chunk_chars and merged_groups:
                merged_groups[-1].extend(group)
            else:
                merged_groups.append(group)

        # Step 3.5: 短标题向前合并
        merged_groups_forward: list[list[int]] = []
        i = 0
        while i < len(merged_groups):
            group = merged_groups[i]
            is_title_only = (len(group) == 1 and group[0] in section_boundaries)
            if is_title_only:
                title_text = paragraphs[group[0]]
                if len(title_text) < min_chunk_chars:
                    if i + 1 < len(merged_groups):
                        next_group = merged_groups[i + 1]
                        merged_groups_forward.append([group[0]] + next_group)
                        i += 2
                    else:
                        merged_groups_forward.append(group)
                        i += 1
                else:
                    merged_groups_forward.append(group)
                    i += 1
            else:
                merged_groups_forward.append(group)
                i += 1

        merged_groups = merged_groups_forward

        # Step 4: 创建 Document
        documents = []
        active_path: list[str] = []
        for group in merged_groups:
            for idx in group:
                if idx in section_boundaries:
                    title = section_titles[idx]
                    if not active_path or active_path[-1] != title:
                        active_path.append(title)

            if len(group) == 1 and group[0] in section_boundaries:
                continue

            sub_texts: list[str] = []
            for i in group:
                p_text = paragraphs[i]
                if len(p_text) > max_chunk_chars:
                    parts = _split_long_paragraph_at_sentence_boundaries(p_text, max_chunk_chars)
                    sub_texts.extend(parts)
                else:
                    sub_texts.append(p_text)

            current_parts: list[str] = []
            current_len = 0
            for part in sub_texts:
                if current_len + len(part) <= max_chunk_chars:
                    current_parts.append(part)
                    current_len += len(part)
                else:
                    if current_parts:
                        content = '\n\n'.join(current_parts)
                        has_image = any(idx in image_boundaries for idx in group)
                        has_table = any(idx in table_boundaries for idx in group)
                        extra = {}
                        if has_image:
                            extra["figure_type"] = "image"
                            extra["pending_multimodal"] = True
                            _img_captions = [paragraphs[idx].strip() for idx in group if idx in image_boundaries]
                            if _img_captions:
                                extra["image_caption"] = _img_captions[0]
                        elif has_table:
                            extra["figure_type"] = "table"
                            extra["pending_multimodal"] = False
                        meta = self._make_chunk_meta(
                            genre=Genre.RESEARCH_LITERATURE.value,
                            strategy="paragraph_semantic_merge",
                            chapter_number=chapter_number,
                            chapter_title=chapter_title,
                            source_type=source_type,
                            source_file=source_file,
                            paragraph_index=group[0],
                            paragraph_range=[group[0], group[-1]],
                            sentence_range=None,
                            char_start=-1,
                            char_end=-1,
                            structural_unit=' > '.join(active_path) if active_path else "",
                            context_prefix="",
                            context_prefix_length=0,
                            dynasty=dynasty,
                            **extra,
                        )
                        documents.append(Document(content=content, metadata=meta))
                    current_parts = [part]
                    current_len = len(part)

            if current_parts:
                content = '\n\n'.join(current_parts)
                has_image = any(idx in image_boundaries for idx in group)
                has_table = any(idx in table_boundaries for idx in group)
                extra = {}
                if has_image:
                    extra["figure_type"] = "image"
                    extra["pending_multimodal"] = True
                    _img_captions = [paragraphs[idx].strip() for idx in group if idx in image_boundaries]
                    if _img_captions:
                        extra["image_caption"] = _img_captions[0]
                elif has_table:
                    extra["figure_type"] = "table"
                    extra["pending_multimodal"] = False
                meta = self._make_chunk_meta(
                    genre=Genre.RESEARCH_LITERATURE.value,
                    strategy="paragraph_semantic_merge",
                    chapter_number=chapter_number,
                    chapter_title=chapter_title,
                    source_type=source_type,
                    source_file=source_file,
                    paragraph_index=group[0],
                    paragraph_range=[group[0], group[-1]],
                    sentence_range=None,
                    char_start=-1,
                    char_end=-1,
                    structural_unit=' > '.join(active_path) if active_path else "",
                    context_prefix="",
                    context_prefix_length=0,
                    dynasty=dynasty,
                    **extra,
                )
                documents.append(Document(content=content, metadata=meta))

        return documents

    # ========================================================================
    # CU-11: 戏曲策略 —— 按章结构分块 + 曲牌前缀
    # ========================================================================
    def _normalize_bare_acts(self, text: str) -> str:
        """将 '订盟' 这类裸幕名标准化为 '第X出 订盟' 格式。"""
        lines = text.split('\n')
        result: list[str] = []
        bare_act_counter = 0
        for i, line in enumerate(lines):
            stripped = line.strip()
            if not stripped:
                result.append(line)
                continue
            has_blank_before = i == 0 or not lines[i - 1].strip()
            has_blank_after = i == len(lines) - 1 or not lines[i + 1].strip()
            if (has_blank_before and has_blank_after
                    and 2 <= len(stripped) <= 6
                    and re.match(r'^[一-鿿]{2,6}$', stripped)
                    and not self._OPERA_QUPAI_PATTERN.search(stripped)
                    and not self._OPERA_ROLE_PATTERN.search(stripped)):
                bare_act_counter += 1
                result.append(f'第{bare_act_counter}出 {stripped}')
                logger.info(f"裸幕标题识别: '{stripped}' -> '第{bare_act_counter}出 {stripped}'")
            else:
                result.append(line)
        return '\n'.join(result)

    def _chunk_structural_act(
        self, text: str, config: GenreChunkConfig, chapter_number: str = "",
        chapter_title: str = "", dynasty: str | None = None,
        source_type: str = "primary_literature",
        chunk_id_prefix: str = "", source_file: str = "",
    ) -> list[Document]:
        max_chunk_chars = config.max_chunk_chars
        min_chunk_chars = config.min_chunk_chars
        context_prefix_enabled = config.context_prefix_enabled

        text = self._clean_text(text)
        if not text:
            return []

        text = self._normalize_bare_acts(text)

        documents: list[Document] = []

        act_parts = self._OPERA_ACT_PATTERN.split(text)
        current_act_title = ""
        act_index = -1  # U07: 0-based act index

        for idx, part in enumerate(act_parts):
            part = part.strip()
            if not part:
                continue

            if self._OPERA_ACT_PATTERN.match(part):
                current_act_title = part
                act_index += 1
                continue

            if not current_act_title:
                current_act_title = chapter_title

            qupai_segments = self._OPERA_QUPAI_PATTERN.split(part)
            current_qupai_name = ""

            pending_segments: list[dict] = []

            for q_idx, seg in enumerate(qupai_segments):
                seg = seg.strip()
                if not seg:
                    continue

                if q_idx % 2 == 1:
                    current_qupai_name = seg
                    continue

                qupai_name = current_qupai_name if current_qupai_name else ""

                if len(seg) > max_chunk_chars:
                    role_boundaries = list(self._OPERA_ROLE_PATTERN.finditer(seg))
                    if role_boundaries:
                        sub_start = 0
                        for r_idx, rb in enumerate(role_boundaries):
                            if r_idx == 0 and rb.start() > 0:
                                pre_text = seg[:rb.start()].strip()
                                if pre_text:
                                    prefix = f"{current_act_title} · 【{qupai_name}】" if (context_prefix_enabled and qupai_name) else ""
                                    pending_segments.append({
                                        "text": pre_text,
                                        "qupai": qupai_name,
                                        "prefix": prefix,
                                    })
                            if r_idx > 0:
                                sub_text = seg[sub_start:rb.start()].strip()
                                if sub_text and len(sub_text) >= min_chunk_chars:
                                    role_match = self._OPERA_ROLE_PATTERN.search(sub_text)
                                    role_name = role_match.group(1) if role_match else ""
                                    prefix = f"{current_act_title} · 【{qupai_name}】" if (context_prefix_enabled and qupai_name) else ""
                                    pending_segments.append({
                                        "text": sub_text,
                                        "qupai": qupai_name,
                                        "role": role_name,
                                        "prefix": prefix,
                                    })
                            sub_start = rb.start()
                        last_text = seg[sub_start:].strip()
                        if last_text and len(last_text) >= min_chunk_chars:
                            role_match = self._OPERA_ROLE_PATTERN.search(last_text)
                            role_name = role_match.group(1) if role_match else ""
                            prefix = f"{current_act_title} · 【{qupai_name}】" if (context_prefix_enabled and qupai_name) else ""
                            pending_segments.append({
                                "text": last_text,
                                "qupai": qupai_name,
                                "role": role_name,
                                "prefix": prefix,
                            })
                    else:
                        parts = _split_long_paragraph_at_sentence_boundaries(seg, max_chunk_chars)
                        for p in parts:
                            prefix = f"{current_act_title} · 【{qupai_name}】" if (context_prefix_enabled and qupai_name) else ""
                            pending_segments.append({
                                "text": p,
                                "qupai": qupai_name,
                                "prefix": prefix,
                            })
                else:
                    if len(seg) >= min_chunk_chars:
                        prefix = f"{current_act_title} · 【{qupai_name}】" if (context_prefix_enabled and qupai_name) else ""
                        pending_segments.append({
                            "text": seg,
                            "qupai": qupai_name,
                            "prefix": prefix,
                        })

            for seg_info in pending_segments:
                content = seg_info["text"]
                if seg_info["prefix"]:
                    content = seg_info["prefix"] + "\n" + content

                prefix = seg_info.get("prefix", "")
                prefix_len = len(prefix)
                # U03/U07: 追踪原文中正文的起始位置 (不含 prefix)
                body_text = seg_info["text"]
                body_start = text.find(body_text) if body_text else -1
                body_end = body_start + len(body_text) if body_start >= 0 else -1
                # 句子范围
                sentences_in_body = _split_sentences(body_text)
                s_start = 0
                s_end = max(len(sentences_in_body) - 1, 0) if sentences_in_body else 0
                # 段落索引 (act 内 \n\n 分割)
                act_text_for_para = part if current_act_title else text
                act_paras = _split_paragraphs(act_text_for_para)
                para_idx = -1
                if body_start >= 0:
                    cumulative = 0
                    for p_i, para in enumerate(act_paras):
                        para_end = cumulative + len(para)
                        if cumulative <= body_start < para_end + 2:
                            para_idx = p_i
                            break
                        cumulative = para_end + 2

                extra = {
                    "qupai_name": seg_info.get("qupai", ""),
                    "act_index": act_index,
                }
                if "role" in seg_info:
                    extra["role_name"] = seg_info["role"]

                meta = self._make_chunk_meta(
                    genre=Genre.OPERA_LIBRETTO.value,
                    strategy="structural_act",
                    chapter_number=chapter_number,
                    chapter_title=chapter_title,
                    source_type=source_type,
                    source_file=source_file,
                    paragraph_index=para_idx,
                    paragraph_range=[para_idx, para_idx] if para_idx >= 0 else None,
                    sentence_range=[s_start, s_end],
                    char_start=body_start,
                    char_end=body_end,
                    structural_unit=current_act_title,
                    context_prefix=prefix,
                    context_prefix_length=prefix_len,
                    dynasty=dynasty,
                    **extra,
                )
                documents.append(Document(content=content, metadata=meta))

        return documents

    # ========================================================================
    # CU-12: 剧本策略 —— 按场结构分块 + 角色/板式/舞台指示处理
    # ========================================================================
    def _merge_pending_segment(
        self, pending: dict, next_seg: dict | None,
        merged_segments: list[dict],
        similarity_threshold: float,
    ):
        """决定短 segment 向前合并还是向后合并。"""
        has_prev = len(merged_segments) > 0
        if not has_prev:
            if next_seg is not None:
                next_seg["text"] = pending["text"] + '\n' + next_seg["text"]
            else:
                merged_segments.append(pending)
            return
        if next_seg is None:
            merged_segments[-1]["text"] = merged_segments[-1]["text"] + '\n' + pending["text"]
            return
        prev_sim = 0.0
        pending_emb = None
        try:
            prev_emb = _compute_embedding(merged_segments[-1]["text"], self.embedder)
            pending_emb = _compute_embedding(pending["text"], self.embedder)
            if prev_emb is not None and pending_emb is not None:
                prev_sim = _cosine_similarity(prev_emb, pending_emb)
        except Exception:
            pass
        next_sim = 0.0
        try:
            next_emb = _compute_embedding(next_seg["text"], self.embedder)
            if pending_emb is None:
                pending_emb = _compute_embedding(pending["text"], self.embedder)
            if pending_emb is not None and next_emb is not None:
                next_sim = _cosine_similarity(pending_emb, next_emb)
        except Exception:
            pass
        if next_sim > prev_sim and next_sim >= similarity_threshold:
            next_seg["text"] = pending["text"] + '\n' + next_seg["text"]
        else:
            merged_segments[-1]["text"] = merged_segments[-1]["text"] + '\n' + pending["text"]

    def _chunk_structural_scene(
        self, text: str, config: GenreChunkConfig, chapter_number: str = "",
        chapter_title: str = "", dynasty: str | None = None,
        source_type: str = "primary_literature",
        chunk_id_prefix: str = "", source_file: str = "",
    ) -> list[Document]:
        max_chunk_chars = config.max_chunk_chars
        min_chunk_chars = config.min_chunk_chars
        context_prefix_enabled = config.context_prefix_enabled
        similarity_threshold = getattr(config, 'similarity_threshold', 0.0) or 0.0

        if similarity_threshold > 0:
            self._wait_for_embedder()

        text = self._clean_text(text)
        if not text:
            return []

        documents: list[Document] = []

        scene_parts = self._SCRIPT_SCENE_PATTERN.split(text)
        current_scene_title = ""
        scene_index = -1  # U07: 0-based scene index

        for idx, part in enumerate(scene_parts):
            part = part.strip()
            if not part:
                continue

            if self._SCRIPT_SCENE_PATTERN.match(part):
                current_scene_title = part
                scene_index += 1
                continue

            if not current_scene_title:
                current_scene_title = chapter_title

            segments: list[dict] = []
            lines = part.split('\n')

            current_seg_type = None
            current_seg_lines: list[str] = []
            current_character = ""
            current_banshi = ""

            for line in lines:
                line = line.strip()
                if not line:
                    continue

                banshi_match = self._SCRIPT_BANSHI_PATTERN.search(line)
                banshi_text = banshi_match.group(1) if banshi_match else ""

                role_match = re.match(r'^([一-鿿\s]{1,6})\s*[（(]', line)

                if banshi_match and not role_match:
                    if current_seg_type == "stage_direction":
                        current_seg_lines.append(line)
                    else:
                        if current_seg_lines:
                            seg_text = '\n'.join(current_seg_lines)
                            segments.append({
                                "type": current_seg_type or "dialogue",
                                "text": seg_text,
                                "character": current_character,
                                "banshi": current_banshi,
                            })
                        current_seg_type = "stage_direction"
                        current_seg_lines = [line]
                        current_character = ""
                        current_banshi = banshi_text
                elif role_match:
                    raw_role = role_match.group(1)
                    clean_role = re.sub(r'\s+', '', raw_role)
                    if current_seg_lines:
                        seg_text = '\n'.join(current_seg_lines)
                        segments.append({
                            "type": current_seg_type or "dialogue",
                            "text": seg_text,
                            "character": current_character,
                            "banshi": current_banshi,
                        })
                    current_seg_type = "dialogue"
                    current_seg_lines = [line]
                    current_character = clean_role
                    current_banshi = banshi_text
                else:
                    current_seg_lines.append(line)

            if current_seg_lines:
                seg_text = '\n'.join(current_seg_lines)
                segments.append({
                    "type": current_seg_type or "dialogue",
                    "text": seg_text,
                    "character": current_character,
                    "banshi": current_banshi,
                })

            merged_segments: list[dict] = []
            _pending_short: dict | None = None

            for seg in segments:
                seg_len = len(seg["text"])
                if seg_len >= min_chunk_chars:
                    if _pending_short is not None:
                        self._merge_pending_segment(
                            _pending_short, seg, merged_segments,
                            similarity_threshold,
                        )
                        _pending_short = None
                    merged_segments.append(seg)
                else:
                    if _pending_short is not None:
                        self._merge_pending_segment(
                            _pending_short, seg, merged_segments,
                            similarity_threshold,
                        )
                    _pending_short = seg

            if _pending_short is not None:
                if merged_segments:
                    merged_segments[-1]["text"] = (
                        merged_segments[-1]["text"] + '\n' + _pending_short["text"]
                    )
                else:
                    merged_segments.append(_pending_short)

            # Step 3.5: 同类型 segment 之间的语义合并
            if similarity_threshold > 0 and len(merged_segments) > 1:
                seg_embeddings = []
                for seg in merged_segments:
                    try:
                        emb = _compute_embedding(seg["text"], self.embedder)
                        seg_embeddings.append(emb)
                    except Exception as e:
                        logger.warning(f"剧本 segment embedding 计算失败: {e}")
                        seg_embeddings.append(None)

                semantic_groups: list[list[int]] = []
                current_group: list[int] = []
                current_text_len = 0

                for i, seg in enumerate(merged_segments):
                    seg_len = len(seg["text"])
                    if seg_len == 0:
                        continue
                    if not current_group:
                        current_group = [i]
                        current_text_len = seg_len
                        continue

                    prev_seg = merged_segments[current_group[-1]]
                    if seg.get("type") != prev_seg.get("type"):
                        semantic_groups.append(current_group)
                        current_group = [i]
                        current_text_len = seg_len
                        continue

                    emb_curr = seg_embeddings[i]
                    group_embs = [
                        seg_embeddings[j] for j in current_group
                        if seg_embeddings[j] is not None
                    ]

                    should_merge = False
                    if emb_curr is not None and group_embs:
                        centroid = np.mean(np.asarray(group_embs, dtype=np.float64), axis=0)
                        sim = _cosine_similarity(centroid, emb_curr)
                        if sim >= similarity_threshold and (current_text_len + seg_len) <= max_chunk_chars:
                            should_merge = True

                    if should_merge:
                        current_group.append(i)
                        current_text_len += seg_len
                    else:
                        semantic_groups.append(current_group)
                        current_group = [i]
                        current_text_len = seg_len

                if current_group:
                    semantic_groups.append(current_group)

                new_merged_segments: list[dict] = []
                for group in semantic_groups:
                    if len(group) == 1:
                        new_merged_segments.append(merged_segments[group[0]])
                    else:
                        combined_text = '\n'.join(
                            merged_segments[i]["text"] for i in group
                        )
                        all_characters: list[str] = []
                        for i in group:
                            char = merged_segments[i].get("character", "")
                            if char and char not in all_characters:
                                all_characters.append(char)
                        first_seg = merged_segments[group[0]]
                        new_seg = {
                            "type": first_seg["type"],
                            "text": combined_text,
                            "character": first_seg.get("character", ""),
                            "banshi": first_seg.get("banshi", ""),
                            "merged_characters": all_characters,
                        }
                        new_merged_segments.append(new_seg)

                merged_segments = new_merged_segments

            for seg in merged_segments:
                content_text = seg["text"]
                prefix = ""
                if context_prefix_enabled:
                    if seg["type"] == "dialogue":
                        merged_chars = seg.get("merged_characters", [])
                        if len(merged_chars) > 1:
                            char_display = ' & '.join(merged_chars[:3])
                            if len(merged_chars) > 3:
                                char_display += ' ...'
                        else:
                            char_display = seg.get("character", "")
                        banshi_part = f" · {seg['banshi']}" if seg.get('banshi') else ""
                        prefix = f"{current_scene_title} · {char_display}{banshi_part}"
                    else:
                        prefix = f"{current_scene_title} · 舞台指示"

                if len(content_text) > max_chunk_chars:
                    parts = _split_long_paragraph_at_sentence_boundaries(content_text, max_chunk_chars)
                    for p in parts:
                        full_content = f"{prefix}\n{p}" if prefix else p
                        prefix_len = len(prefix)
                        body_text = p
                        body_start = text.find(body_text) if body_text else -1
                        body_end = body_start + len(body_text) if body_start >= 0 else -1
                        sentences_in_body = _split_sentences(body_text)
                        s_start = 0
                        s_end = max(len(sentences_in_body) - 1, 0) if sentences_in_body else 0
                        extra = {
                            "character_name": seg.get("character", ""),
                            "banshi": seg.get("banshi", ""),
                            "segment_type": seg["type"],
                            "scene_index": scene_index,
                        }
                        merged_chars = seg.get("merged_characters", [])
                        if len(merged_chars) > 1:
                            extra["merged_characters"] = merged_chars
                        meta = self._make_chunk_meta(
                            genre=Genre.SCRIPT.value,
                            strategy="structural_scene",
                            chapter_number=chapter_number,
                            chapter_title=chapter_title,
                            source_type=source_type,
                            source_file=source_file,
                            paragraph_index=-1,
                            paragraph_range=None,
                            sentence_range=[s_start, s_end],
                            char_start=body_start,
                            char_end=body_end,
                            structural_unit=current_scene_title,
                            context_prefix=prefix,
                            context_prefix_length=prefix_len,
                            dynasty=dynasty,
                            **extra,
                        )
                        documents.append(Document(content=full_content, metadata=meta))
                else:
                    full_content = f"{prefix}\n{content_text}" if prefix else content_text
                    prefix_len = len(prefix)
                    body_text = content_text
                    body_start = text.find(body_text) if body_text else -1
                    body_end = body_start + len(body_text) if body_start >= 0 else -1
                    sentences_in_body = _split_sentences(body_text)
                    s_start = 0
                    s_end = max(len(sentences_in_body) - 1, 0) if sentences_in_body else 0
                    extra = {
                        "character_name": seg.get("character", ""),
                        "banshi": seg.get("banshi", ""),
                        "segment_type": seg["type"],
                        "scene_index": scene_index,
                    }
                    merged_chars = seg.get("merged_characters", [])
                    if len(merged_chars) > 1:
                        extra["merged_characters"] = merged_chars
                    meta = self._make_chunk_meta(
                        genre=Genre.SCRIPT.value,
                        strategy="structural_scene",
                        chapter_number=chapter_number,
                        chapter_title=chapter_title,
                        source_type=source_type,
                        source_file=source_file,
                        paragraph_index=-1,
                        paragraph_range=None,
                        sentence_range=[s_start, s_end],
                        char_start=body_start,
                        char_end=body_end,
                        structural_unit=current_scene_title,
                        context_prefix=prefix,
                        context_prefix_length=prefix_len,
                        dynasty=dynasty,
                        **extra,
                    )
                    documents.append(Document(content=full_content, metadata=meta))

        return documents

    # ========================================================================
    # CU-13: 弹词策略 —— 两级分块（场景大块 + 滑动窗口）
    # ========================================================================
    @staticmethod
    def _expand_tanci_repetition(text: str) -> str:
        """CU-49: 展开弹词重复记号。X七->XX, X匕->XX, 数字语境不处理。"""
        result = []
        chars = list(text)
        i = 0
        while i < len(chars):
            if i < len(chars) - 1 and chars[i + 1] in ('七', '匕'):
                prev_char = chars[i]
                if '一' <= prev_char <= '鿿':
                    if prev_char not in '一二三四五六七八九十百千万零':
                        result.append(prev_char)
                        i += 1
                        continue
            result.append(chars[i])
            i += 1
        return ''.join(result)

    @staticmethod
    def _find_best_break_point(
        text: str, marker_positions: list[int],
        preferred_end: int, search_range: int = 100,
    ) -> int:
        """CU-50: 在最佳位置断开。优先级: 标记词前 > 句末标点 > 分号逗号 > 硬截断。"""
        start = max(0, preferred_end - search_range)
        end = min(len(text), preferred_end + search_range)
        search_region = text[start:end]
        for mp in marker_positions:
            if start <= mp <= end:
                return mp
        for sep in ['。', '！', '？']:
            rpos = search_region.rfind(sep)
            if rpos >= 0:
                return start + rpos + 1
        for sep in ['；', '，']:
            rpos = search_region.rfind(sep)
            if rpos >= 0:
                return start + rpos + 1
        return preferred_end

    @staticmethod
    def _build_tanci_prefix(
        chunk_text: str, marker_positions: list[tuple[int, int, str]],
        scene_roles: list[str], chunk_start: int, chunk_end: int,
    ) -> str:
        """CU-50: 构建弹词上下文前缀。"""
        current_marker = None
        inherited_marker = None
        for mp_start, mp_end, mp_type in marker_positions:
            if chunk_start <= mp_start < chunk_end:
                if current_marker is None:
                    current_marker = mp_type
            elif mp_start < chunk_start:
                inherited_marker = mp_type
        marker = current_marker or inherited_marker
        if not marker:
            return ""
        marker_type_map = {
            '唱': '唱词', '白': '说白', '曰': '说白', '曰引': '开场白',
            '诗曰': '诗', '表': '叙述', '云': '说白',
        }
        marker_display = marker_type_map.get(marker, marker)
        roles_in_chunk = []
        for role in scene_roles:
            if role in chunk_text and role not in roles_in_chunk:
                roles_in_chunk.append(role)
                if len(roles_in_chunk) >= 2:
                    break
        if roles_in_chunk:
            return f"{marker_display} · {' · '.join(roles_in_chunk)}"
        return marker_display

    def _chunk_tanci_scene_window(
        self, text: str, config: GenreChunkConfig, chapter_number: str = "",
        chapter_title: str = "", dynasty: str | None = None,
        source_type: str = "primary_literature",
        chunk_id_prefix: str = "", source_file: str = "",
    ) -> list[Document]:
        window_size = config.max_chunk_chars
        window_overlap = config.window_overlap
        min_chunk_chars = config.min_chunk_chars
        context_prefix_enabled = config.context_prefix_enabled

        text = self._clean_text(text)
        text = self._expand_tanci_repetition(text)
        if not text:
            return []

        marker_positions: list[tuple[int, int, str]] = []
        for m in self._TANCI_INNER_MARKER_PATTERN.finditer(text):
            marker_positions.append((m.start(), m.end(), m.group(0)))

        scene_roles: list[str] = []
        for m in self._TANCI_ROLE_NAMES_PATTERN.finditer(text):
            role = m.group(0)
            if role not in scene_roles:
                scene_roles.append(role)

        documents: list[Document] = []

        scene_parts = self._TANCI_SCENE_BREAK_PATTERN.split(text)
        scene_chunks: list[str] = []
        current_scene = ""

        for part in scene_parts:
            part = part.strip()
            if not part:
                continue
            if self._TANCI_SCENE_BREAK_PATTERN.match(part):
                if current_scene:
                    scene_chunks.append(current_scene)
                current_scene = part
            else:
                current_scene += part

        if current_scene:
            scene_chunks.append(current_scene)

        merged_scenes: list[str] = []
        for sc in scene_chunks:
            if len(sc) < 300 and merged_scenes:
                merged_scenes[-1] = merged_scenes[-1] + sc
            else:
                merged_scenes.append(sc)

        marker_pos_list = [mp[0] for mp in marker_positions]

        for scene_text in merged_scenes:
            scene_offset = text.find(scene_text) if scene_text else 0
            if len(scene_text) <= window_size:
                if len(scene_text) >= min_chunk_chars:
                    prefix = ""
                    if context_prefix_enabled:
                        prefix = self._build_tanci_prefix(
                            scene_text, marker_positions, scene_roles, scene_offset,
                            scene_offset + len(scene_text),
                        )
                    content = f"{prefix}\n{scene_text}" if prefix else scene_text
                    prefix_len = len(prefix)
                    body_start = scene_offset if scene_offset >= 0 else -1
                    body_end = body_start + len(scene_text) if body_start >= 0 else -1
                    sentences_in_body = _split_sentences(scene_text)
                    s_start = 0
                    s_end = max(len(sentences_in_body) - 1, 0) if sentences_in_body else 0
                    meta = self._make_chunk_meta(
                        genre=Genre.TANCI.value,
                        strategy="tanci_scene_window",
                        chapter_number=chapter_number,
                        chapter_title=chapter_title,
                        source_type=source_type,
                        source_file=source_file,
                        paragraph_index=-1,
                        paragraph_range=None,
                        sentence_range=[s_start, s_end],
                        char_start=body_start,
                        char_end=body_end,
                        structural_unit="",
                        context_prefix=prefix,
                        context_prefix_length=prefix_len,
                        dynasty=dynasty,
                    )
                    documents.append(Document(content=content, metadata=meta))
                continue

            start = 0
            while start < len(scene_text):
                preferred_end = min(start + window_size, len(scene_text))
                chunk_end = self._find_best_break_point(
                    scene_text, marker_pos_list, preferred_end, 100,
                )
                chunk_end = min(chunk_end, len(scene_text))

                chunk_content = scene_text[start:chunk_end].strip()
                if not chunk_content:
                    start = max(start + 1, chunk_end)
                    continue

                if len(chunk_content) >= min_chunk_chars:
                    chunk_start_abs = scene_offset + start
                    chunk_end_abs = scene_offset + chunk_end
                    prefix = ""
                    if context_prefix_enabled:
                        prefix = self._build_tanci_prefix(
                            chunk_content, marker_positions, scene_roles,
                            chunk_start_abs, chunk_end_abs,
                        )
                    content = f"{prefix}\n{chunk_content}" if prefix else chunk_content
                    prefix_len = len(prefix)
                    body_start = chunk_start_abs
                    body_end = chunk_end_abs
                    sentences_in_body = _split_sentences(chunk_content)
                    s_start = 0
                    s_end = max(len(sentences_in_body) - 1, 0) if sentences_in_body else 0
                    meta = self._make_chunk_meta(
                        genre=Genre.TANCI.value,
                        strategy="tanci_scene_window",
                        chapter_number=chapter_number,
                        chapter_title=chapter_title,
                        source_type=source_type,
                        source_file=source_file,
                        paragraph_index=-1,
                        paragraph_range=None,
                        sentence_range=[s_start, s_end],
                        char_start=body_start,
                        char_end=body_end,
                        structural_unit="",
                        context_prefix=prefix,
                        context_prefix_length=prefix_len,
                        dynasty=dynasty,
                    )
                    documents.append(Document(content=content, metadata=meta))

                next_start = chunk_end - window_overlap
                if next_start <= start:
                    next_start = chunk_end
                start = next_start

        final_docs: list[Document] = []
        for doc in documents:
            if len(doc.content) < min_chunk_chars and final_docs:
                final_docs[-1].content = final_docs[-1].content + '\n' + doc.content
            else:
                final_docs.append(doc)

        return final_docs

    # ========================================================================
    # CU-14: chunk_by_genre 策略分发 + chunk_all 重写
    # ========================================================================
    def chunk_by_genre(
        self, text: str, genre: Genre, chapter_number: str = "",
        chapter_title: str = "", dynasty: str | None = None,
        source_type: str = "primary_literature",
        source_file: str = "",
    ) -> list[Document]:
        """根据体裁分发到对应策略函数。"""
        from .config import rag_config

        config = GENRE_CHUNK_DEFAULTS.get(genre.value)
        if config is None:
            logger.warning(f"未知体裁 {genre.value}，回退到拟话本纯段落分块")
            config = GENRE_CHUNK_DEFAULTS[Genre.IMITATION_HUABEN.value]

        source_prefix = "primary" if source_type == "primary_literature" else "research"
        chunk_id_prefix = f"{source_prefix}_{genre.value}_{chapter_number}"

        if not rag_config.genre_chunking_enabled:
            logger.info(f"genre_chunking 已禁用，{genre.value} 降级为纯段落分块")
            return self._chunk_semantic_paragraph(
                text, config, chapter_number, chapter_title, dynasty, source_type,
                chunk_id_prefix=chunk_id_prefix, source_file=source_file,
            )

        match genre:
            case Genre.CLASSICAL_CHINESE:
                return self._chunk_sentence_semantic_merge(
                    text, config, chapter_number, chapter_title, dynasty, source_type,
                    chunk_id_prefix=chunk_id_prefix, source_file=source_file,
                )
            case Genre.HUABEN:
                return self._chunk_paragraph_semantic_merge_huaben(
                    text, config, chapter_number, chapter_title, dynasty, source_type,
                    chunk_id_prefix=chunk_id_prefix, source_file=source_file,
                )
            case Genre.IMITATION_HUABEN:
                return self._chunk_semantic_paragraph(
                    text, config, chapter_number, chapter_title, dynasty, source_type,
                    chunk_id_prefix=chunk_id_prefix, source_file=source_file,
                )
            case Genre.CHAPTER_VERNACULAR:
                return self._chunk_structural_chapter_merge(
                    text, config, chapter_number, chapter_title, dynasty, source_type,
                    chunk_id_prefix=chunk_id_prefix, source_file=source_file,
                )
            case Genre.RESEARCH_LITERATURE:
                return self._chunk_research_literature(
                    text, config, chapter_number, chapter_title, dynasty, source_type,
                    chunk_id_prefix=chunk_id_prefix, source_file=source_file,
                )
            case Genre.OPERA_LIBRETTO:
                return self._chunk_structural_act(
                    text, config, chapter_number, chapter_title, dynasty, source_type,
                    chunk_id_prefix=chunk_id_prefix, source_file=source_file,
                )
            case Genre.SCRIPT:
                return self._chunk_structural_scene(
                    text, config, chapter_number, chapter_title, dynasty, source_type,
                    chunk_id_prefix=chunk_id_prefix, source_file=source_file,
                )
            case Genre.TANCI:
                return self._chunk_tanci_scene_window(
                    text, config, chapter_number, chapter_title, dynasty, source_type,
                    chunk_id_prefix=chunk_id_prefix, source_file=source_file,
                )
            case _:
                logger.warning(f"未实现的体裁 {genre.value}，回退到拟话本纯段落分块")
                return self._chunk_semantic_paragraph(
                    text, config, chapter_number, chapter_title, dynasty, source_type,
                    chunk_id_prefix=chunk_id_prefix, source_file=source_file,
                )

    def chunk_all(self, chapter_sink=None, use_snapshot: bool = True) -> list[Document]:
        """加载文献全部分块（含白蛇传文献选集 + 文本景观建构研究文献）。

        chapter_sink: 可选回调 (chapter_number, chapter_title, dynasty, chunks) -> None，
        每章分块完成后立即调用。用于"边分块边入库"的增量重建——
        单章失败/进程退出时已有章节落库，重跑可幂等覆盖（chunk_id 按章节顺序确定性生成）。

        use_snapshot: True（默认）时优先从冻结快照 corpus.jsonl 加载，
        跳过语义分块，保证重建时 chunk_id 与下游标签/词表/向量缓存完全对齐；
        需要重新分块（改了分块策略/语料更新）时传 False，分块完成后自动刷新快照。
        """
        if not self.data_dir:
            raise ValueError("data_dir 未设置")

        # ---- 冻结快照优先：重建时复用固定 chunk 边界，chunk_id 不漂移 ----
        if use_snapshot:
            snap = load_chunk_snapshot()
            if snap is not None:
                logger.info(
                    f"[CHUNK] 使用冻结分块快照: {len(snap)} 块"
                    f"（跳过语义分块，chunk_id 保持稳定）"
                )
                if chapter_sink is not None:
                    # 按章节顺序分组回调，保持逐章增量入库行为
                    groups: list = []
                    for d in snap:
                        ch = d.metadata.get("chapter_number", "?")
                        if not groups or groups[-1][0] != ch:
                            groups.append([
                                ch,
                                d.metadata.get("chapter_title", ""),
                                d.metadata.get("dynasty"),
                                [],
                            ])
                        groups[-1][3].append(d)
                    for ch, title, dynasty, grp in groups:
                        chapter_sink(ch, title, dynasty, grp)
                return snap

        self._chunk_counter = 0  # U02: 每次 chunk_all 重置计数器
        all_chunks: list[Document] = []

        # 总章节数用于进度回调
        main_chapters = 8  # 主文献7章 + 研究文献1章
        total_chapters = main_chapters
        completed = 0

        primary_source_file = SOURCE_FILE_MAP["primary_literature"]
        research_source_file = SOURCE_FILE_MAP["research_literature"]

        main_txt = self.data_dir / "白蛇传文献选集.txt"
        if main_txt.exists():
            text = main_txt.read_text(encoding="utf-8")
            chapter_pattern = re.compile(r'([一二三四五六七八九十]+、[^\n]+)')
            parts = chapter_pattern.split(text)

            current_chapter = "全文"
            current_title = "白蛇传文献选集"
            current_dynasty: str | None = None

            for part in parts:
                part = part.strip()
                if not part:
                    continue
                m = chapter_pattern.match(part)
                if m:
                    current_title = part
                    num_match = re.match(r'([一二三四五六七])', part)
                    current_chapter = num_match.group(1) if num_match else current_chapter
                    current_dynasty = _extract_dynasty(part)
                    if current_dynasty:
                        logger.info(f"章节 {current_chapter} 识别朝代: {current_dynasty} (标题: {current_title[:40]})")
                    else:
                        logger.warning(f"章节 {current_chapter} 未能提取朝代: {current_title[:40]}")
                else:
                    genre = _detect_genre(current_title)
                    if genre is None:
                        genre = _CHAPTER_GENRE_FALLBACK.get(current_chapter)
                        if genre:
                            logger.info(f"章节 {current_chapter} 回退体裁: {genre.value}")
                        else:
                            logger.warning(f"章节 {current_chapter} 无法识别体裁，跳过")
                            continue

                    logger.info(f"章节 {current_chapter} 体裁: {genre.value} 策略分块...")
                    try:
                        chunks = self.chunk_by_genre(
                            part,
                            genre=genre,
                            chapter_number=current_chapter,
                            chapter_title=current_title,
                            dynasty=current_dynasty,
                            source_type="primary_literature",
                            source_file=primary_source_file,
                        )
                        all_chunks.extend(chunks)
                        logger.info(f"   -> {len(chunks)} 个 chunk")
                        if chapter_sink is not None:
                            chapter_sink(current_chapter, current_title, current_dynasty, chunks)
                    except Exception as e:
                        logger.error(f"章节 {current_chapter} 分块失败: {e}")
                        import traceback
                        logger.error(traceback.format_exc())
                    completed += 1
                    if self.progress_callback:
                        try:
                            self.progress_callback(completed, total_chapters, current_title)
                        except Exception:
                            pass

            logger.info(f"主文献分块: {len(all_chunks)} 块")
        else:
            logger.warning(f"主文献文件不存在: {main_txt}")

        research_txt = self.data_dir / "文本景观建构研究文献.txt"
        if research_txt.exists():
            text = research_txt.read_text(encoding="utf-8")
            logger.info("研究文献直接指定 Genre.RESEARCH_LITERATURE")
            try:
                research_chunks = self.chunk_by_genre(
                    text,
                    genre=Genre.RESEARCH_LITERATURE,
                    chapter_number="研究",
                    chapter_title="文本景观建构研究文献",
                    dynasty=None,
                    source_type="research_literature",
                    source_file=research_source_file,
                )
                all_chunks.extend(research_chunks)
                logger.info(f"研究文献分块: {len(research_chunks)} 块")
                if chapter_sink is not None:
                    chapter_sink("研究", "文本景观建构研究文献", None, research_chunks)
            except Exception as e:
                logger.error(f"研究文献分块失败: {e}")
                import traceback
                logger.error(traceback.format_exc())
            completed += 1
            if self.progress_callback:
                try:
                    self.progress_callback(completed, total_chapters,
                                           "文本景观建构研究文献")
                except Exception:
                    pass
        else:
            logger.warning(f"研究文献文件不存在: {research_txt}")

        # U00: 验证 metadata schema
        validation_errors = validate_chunk_metadata(all_chunks)
        if validation_errors:
            for err in validation_errors:
                logger.warning(f"[VALIDATE] {err}")
            logger.warning(
                f"[VALIDATE] 发现 {len(validation_errors)} 个 metadata schema 问题"
            )
        else:
            logger.info("[VALIDATE] 所有 chunk metadata schema 验证通过")

        self._maybe_write_debug(all_chunks)

        genre_counts: dict[str, int] = {}
        for doc in all_chunks:
            g = doc.metadata.get("genre", "unknown")
            genre_counts[g] = genre_counts.get(g, 0) + 1

        logger.info(f"分块完成: 总计 {len(all_chunks)} 个文本块")
        for g, cnt in sorted(genre_counts.items()):
            logger.info(f"  {g}: {cnt} 块")

        # 新鲜分块结果落盘为冻结快照，后续重建直接复用（chunk_id 永久稳定）
        save_chunk_snapshot(all_chunks)
        return all_chunks

    # ========================================================================
    # CU-15: 分块调试日志输出
    # ========================================================================
    def _maybe_write_debug(self, all_chunks: list[Document]):
        """读取 RAG 配置后，按体裁分别输出调试文件。"""
        try:
            from .config import rag_config, _PROJECT_ROOT
        except ImportError:
            return

        if not rag_config.chunk_debug_enabled:
            return

        output_dir = Path(rag_config.chunk_debug_output_dir)
        if not output_dir.is_absolute():
            output_dir = _PROJECT_ROOT / output_dir
        output_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

        genre_buckets: dict[str, list[Document]] = {}
        for doc in all_chunks:
            genre = doc.metadata.get("genre", "unknown")
            if genre not in genre_buckets:
                genre_buckets[genre] = []
            genre_buckets[genre].append(doc)

        for genre, docs in genre_buckets.items():
            filename = output_dir / f"{genre}_{timestamp}.txt"
            try:
                with open(filename, "w", encoding="utf-8") as f:
                    f.write(f"体裁: {genre}\n")
                    config = GENRE_CHUNK_DEFAULTS.get(genre)
                    if config:
                        f.write(f"策略: {config.chunk_strategy}\n")
                        f.write(f"max_chunk_chars: {config.max_chunk_chars}\n")
                        f.write(f"min_chunk_chars: {config.min_chunk_chars}\n")
                        f.write(f"similarity_threshold: {config.similarity_threshold}\n")
                    f.write(f"总 chunk 数: {len(docs)}\n")
                    f.write("=" * 60 + "\n\n")

                    for i, doc in enumerate(docs):
                        meta = doc.metadata
                        f.write(f"--- Chunk {i} ---\n")
                        f.write(f"字数: {len(doc.content)}\n")
                        f.write(f"朝代: {meta.get('dynasty', 'N/A')}\n")
                        f.write(f"章节号: {meta.get('chapter_number', 'N/A')}\n")
                        f.write(f"章节名: {meta.get('chapter_title', 'N/A')}\n")
                        f.write(f"structural_unit: {meta.get('structural_unit', 'N/A')}\n")
                        f.write(f"context_prefix: {meta.get('context_prefix', 'N/A')}\n")
                        f.write(f"chunk_id: {meta.get('chunk_id', 'N/A')}\n")
                        f.write(f"source_file: {meta.get('source_file', 'N/A')}\n")
                        f.write(f"char_start: {meta.get('char_start', 'N/A')}\n")
                        f.write(f"char_end: {meta.get('char_end', 'N/A')}\n")
                        f.write(f"paragraph_index: {meta.get('paragraph_index', 'N/A')}\n")
                        sentence_range = meta.get('sentence_range') or meta.get('paragraph_range')
                        if sentence_range:
                            f.write(f"范围: {sentence_range}\n")
                        if meta.get('context_prefix_length', 0) > 0:
                            f.write(f"context_prefix_length: {meta['context_prefix_length']}\n")
                        if meta.get("act_index") is not None:
                            f.write(f"act_index: {meta['act_index']}\n")
                        if meta.get("scene_index") is not None:
                            f.write(f"scene_index: {meta['scene_index']}\n")
                        if meta.get("qupai_name"):
                            f.write(f"qupai_name: {meta['qupai_name']}\n")
                        f.write(f"strategy: {meta.get('strategy', 'N/A')}\n")
                        f.write(f"source_type: {meta.get('source_type', 'N/A')}\n")
                        if meta.get("figure_type"):
                            f.write(f"figure_type: {meta['figure_type']}\n")
                        if meta.get("pending_multimodal") is not None:
                            f.write(f"pending_multimodal: {meta['pending_multimodal']}\n")
                        if meta.get("image_caption"):
                            f.write(f"image_caption: {meta['image_caption']}\n")
                        f.write(f"文本:\n{doc.content}\n\n")
                logger.info(f"调试日志已写入: {filename}")
            except Exception as e:
                logger.warning(f"调试日志写入失败 {filename}: {e}")


# ============================================================================
# CU-15: 模块级调试函数 (供外部调用)
# ============================================================================
def _write_chunk_debug(all_chunks: list[Document]):
    """模块级调试输出函数。"""
    temp_chunker = Chunker(data_dir=Path("."))
    temp_chunker._maybe_write_debug(all_chunks)
