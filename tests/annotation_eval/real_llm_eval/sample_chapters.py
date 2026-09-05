"""从《白蛇传文献选集.txt》提取 7 体裁，每体裁均匀抽 5 段，输出 sample_manifest.json。

生产等价逻辑：
- 切章：复刻 server/src/services/literature.service.ts 的 parseLiterature（按「一~七」标题切）
- 切段：复刻 client/src/stores/sourceTree.ts 的 splitParagraphs（空行 \n\n 切，先规范化换行）

调用链日志：每步记录「函数名 + 入参 + 出参」，说明调用了哪个生产等价函数。

运行：
  cd white-snake-museum-public
  PYTHONIOENCODING=utf-8 python -m tests.annotation_eval.real_llm_eval.sample_chapters
"""

import json
import logging
import re
import sys
from pathlib import Path

_project_root = Path(__file__).resolve().parent.parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from .segment import segment_content

LITERATURE_TXT = _project_root / "excel_data" / "白蛇传文献选集.txt"
DATA_DIR = Path(__file__).resolve().parent / "data"
OUTPUT = DATA_DIR / "sample_manifest.json"

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-5s | %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("sample_chapters")

# 章节号 -> 体裁（对应 run_chunking 的 8 体裁中的 7 个正文体裁）
GENRE_BY_CHAPTER = {
    1: "classical_chinese",
    2: "huaben",
    3: "imitation_huaben",
    4: "opera_libretto",
    5: "chapter_vernacular",
    6: "tanci",
    7: "script",
}

_CHINESE_NUM = {"一": 1, "二": 2, "三": 3, "四": 4, "五": 5, "六": 6, "七": 7}
_TITLE_RE = re.compile(r'^\s*([一二三四五六七])\s*[、，,.\s]\s*(.+)$', re.M)


def parse_chapters(text: str) -> list[dict]:
    """复刻 literature.service.ts 的 parseLiterature：按「一~七」标题切章。"""
    matches = list(_TITLE_RE.finditer(text))
    chapters = []
    for i, m in enumerate(matches):
        num = _CHINESE_NUM[m.group(1)]
        title = m.group(0).strip()
        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        content = text[start + len(title):end].strip()
        chapters.append({"chapterNumber": num, "title": title, "content": content})
    logger.info(
        "[CALL] parse_chapters(text=%r...) -> %d 章 [%s]",
        text[:30], len(chapters),
        ", ".join(f"{c['chapterNumber']}:{c['title'][:12]}" for c in chapters),
    )
    return chapters


def sample_indices(total: int, k: int = 5) -> list[int]:
    """均匀抽样 k 个索引（确定性、可复现）。"""
    if total <= k:
        return list(range(total))
    idxs = []
    for i in range(k):
        idxs.append(int(i * (total - 1) / (k - 1)))
    idxs = sorted(set(idxs))
    logger.info("[CALL] sample_indices(total=%d, k=%d) -> %s", total, k, idxs)
    return idxs


def main() -> int:
    logger.info("[STEP] 读取文献: %s", LITERATURE_TXT)
    text = LITERATURE_TXT.read_text(encoding="utf-8")

    chapters = parse_chapters(text)
    if not chapters:
        logger.error("未切出任何章节")
        return 1

    samples = []
    for ch in chapters:
        num = ch["chapterNumber"]
        genre = GENRE_BY_CHAPTER.get(num, "unknown")
        units = segment_content(ch["content"])
        logger.info("[CALL] segment_content(chapter=%d, %d字) -> %d 个标注单元", num, len(ch["content"]), len(units))
        for pi in sample_indices(len(units), k=5):
            samples.append({
                "chapter_number": num,
                "title": ch["title"],
                "genre": genre,
                "paragraph_index": pi,
                "text": units[pi],
            })

    manifest = {
        "sample_count": len(samples),
        "samples": samples,
    }
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("[RET] 抽样完成：%d 段 -> %s", len(samples), OUTPUT)
    logger.info("体裁分布：%s", json.dumps(
        {g: sum(1 for s in samples if s["genre"] == g) for g in sorted({s["genre"] for s in samples})},
        ensure_ascii=False,
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
