"""B-163: 段落定位器 — 运行时恢复 chunk 的真实段落号。

问题（日志与浏览器实测实锤）：
    语料 79% 的 chunk（明/清/现当代等章节）paragraph_index=-1、char_start=-1，
    因为源文件（白蛇传文献选集.txt）在这些章节里用单换行分段，分块时的
    \\n\\n 段落切分切不出有效段落号。后果：出场轨迹/共现等工具返回的
    paragraphIndex=-1 → 前端 para--1 不存在 → "跳转到原文"定位失败且高亮为空。

方案：
    复刻 Node API（server/src/services/literature.service.ts 的 parseLiterature）
    的章节切分 + 前端 splitParagraphs 的段落切分（两者即前端阅读页 DOM 的
    para-{index} 口径），把 chunk content 在对应章节段落中做子串定位，
    恢复真实段落号。段落号无效时由 retriever.find_appearances 调用本模块修正。

降级匹配策略（chunk content 可能被 _clean_text 改动或带装饰前缀）：
    1. 整串 indexOf（chunk 恰为段落全文/连续片段）
    2. 中段 20 字 indexOf（句级切分片段，避开首尾清洗差异）
    3. 前 12 字 indexOf
    全部失败返回 None（调用方保留原值，前端仍有实体扫描兜底）。

用法（retriever 侧）：
    locator = get_paragraph_locator()          # 进程级单例，惰性加载
    pno = locator.locate('三', chunk_content)  # int | None
"""
import logging
import re
import threading
from pathlib import Path

logger = logging.getLogger(__name__)

# 与 Node parseLiterature 完全一致的章节标题正则
_CHAPTER_TITLE_RE = re.compile(r'^\s*(一|二|三|四|五|六|七)\s*[、，,.\s]\s*(.+)$', re.M)

# 定位用的最小可信长度（低于此长度误匹配率高）
_MIN_SNIPPET_LEN = 6


def clean_text_for_locate(text: str) -> str:
    """与 chunker._clean_text / Node 侧对齐的最小清洗（只影响切分与匹配）。"""
    text = re.sub(r'\r\n|\r', '\n', text)
    text = text.replace('\ufeff', '')
    text = re.sub(r' {2,}', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


def _norm_ws(text: str) -> str:
    """匹配用空白归一化：删除所有空白字符。

    chunk 内容常含 \\n\\n / 换行 / 前导空白（跨段片段、诗曰体、装饰后缀），
    原样子串匹配必败；段落号定位只需要"落在哪个段落"，不需要字符位置，
    因此匹配统一在去空白空间进行。
    """
    return re.sub(r'\s+', '', text)


class ParagraphLocator:
    """章节×段落索引 + chunk 内容定位。"""

    def __init__(self, source_path: str):
        self._source_path = source_path
        self._chapters: dict[str, list[str]] | None = None
        self._lock = threading.Lock()

    # ---- 索引构建 ----

    def _ensure_loaded(self) -> None:
        if self._chapters is not None:
            return
        with self._lock:
            if self._chapters is not None:
                return
            try:
                raw = Path(self._source_path).read_text(encoding='utf-8')
                self._load_from_text(raw)
            except (OSError, UnicodeDecodeError) as e:
                logger.warning(f"[段落定位器] 源文件加载失败 ({self._source_path}): {e}")
                self._chapters = {}

    def _load_from_text(self, raw: str) -> None:
        """测试可注入的切分入口：章节切分与 Node parseLiterature 一致。"""
        text = raw  # 章节标题匹配用原始文本（Node 也不 clean）
        # (match_start, title_line_end, chinese_num)
        # 注意：^ 后的 \s* 可能吞掉标题前的空行，m.start() 会落在标题行之前 ——
        # 因此 content 起点必须用 m.end()（标题行尾，(.+)$ 不含换行），
        # 否则标题行会混入首段落导致定位错段
        titles: list[tuple[int, int, str]] = []
        for m in _CHAPTER_TITLE_RE.finditer(text):
            titles.append((m.start(), m.end(), m.group(1)))
        chapters: dict[str, list[str]] = {}
        for i, (start, title_end, num) in enumerate(titles):
            seg_start = title_end
            seg_end = titles[i + 1][0] if i + 1 < len(titles) else len(text)
            content = text[seg_start:seg_end].strip()
            # 段落切分与前端 splitParagraphs / Node search 一致
            normalized = content.replace('\r\n', '\n').replace('\r', '\n')
            paras = [p.strip() for p in normalized.split('\n\n') if p.strip()]
            chapters[num] = paras
        self._chapters = chapters
        total = sum(len(v) for v in chapters.values())
        logger.info(
            f"[段落定位器] 就绪: {len(chapters)} 章 {total} 段 "
            f"({ {k: len(v) for k, v in chapters.items()} })"
        )

    # ---- 查询 ----

    def chapters(self) -> dict[str, list[str]]:
        self._ensure_loaded()
        return self._chapters or {}

    def paragraphs(self, chapter_number: str) -> list[str]:
        return self.chapters().get(str(chapter_number), [])

    def locate(self, chapter_number, content: str) -> int | None:
        """chunk content → 该章节内的段落号。失败返回 None。

        匹配在空白归一化空间进行（见 _norm_ws），三级降级：
        整串 → 中段 20 字 → 前 12 字。
        """
        if content is None:
            return None
        ch = str(chapter_number) if chapter_number not in (None, '') else ''
        if not ch:
            return None
        paras = self.paragraphs(ch)
        if not paras:
            return None
        text = content.strip()
        # 剥掉可能混入的章节标题装饰（如 "三、成熟期 明：… · "，ch七 实锤存在）
        if ' · ' in text[:80]:
            text = text.split(' · ', 1)[1].strip()

        paras_n = [_norm_ws(p) for p in paras]
        text_n = _norm_ws(text)
        if len(text_n) < _MIN_SNIPPET_LEN:
            return None

        # 降级 1：整串定位（chunk 为某段落的连续片段，允许空白差异）
        for i, pn in enumerate(paras_n):
            if text_n in pn:
                return i
        # 降级 2：中段 20 字（避开首尾清洗差异）
        if len(text_n) >= 20:
            mid = text_n[len(text_n) // 3: len(text_n) // 3 + 20]
            if len(mid) >= _MIN_SNIPPET_LEN:
                for i, pn in enumerate(paras_n):
                    if mid in pn:
                        return i
        # 降级 3：前 12 字
        head = text_n[:12]
        if len(head) >= _MIN_SNIPPET_LEN:
            for i, pn in enumerate(paras_n):
                if head in pn:
                    return i
        return None


# ---- 进程级单例 ----

_locator: ParagraphLocator | None = None
_locator_lock = threading.Lock()


def get_paragraph_locator() -> ParagraphLocator:
    """惰性单例。源文件路径与分块时的 data_dir 口径一致（excel_data/）。"""
    global _locator
    if _locator is None:
        with _locator_lock:
            if _locator is None:
                # chunk metadata 的 source_file 为纯文件名，实际位于项目根 excel_data/
                src = Path(__file__).resolve().parents[3] / 'excel_data' / '白蛇传文献选集.txt'
                if not src.exists():
                    # 兼容不同的部署布局：从 agent/ 目录向上找
                    src = Path(__file__).resolve().parents[3] / 'data' / '白蛇传文献选集.txt'
                _locator = ParagraphLocator(source_path=str(src))
    return _locator
