"""生成金标准草稿（代码提取实体名 + 真实 LLM 预标），交用户人工审校。

流程（每段）：
- 读 sample_manifest.json
- 用生产 annotate_passage.handler（真实 DeepSeek LLM）对每段跑一遍：
  - 规则词表 find 命中 → source=rule（代码精确匹配，100% 不漏）
  - LLM 标注 → source=agent
  - event/motif 概括词 → 生成 marginalia
- 输出 golden_draft.json：每段 text + candidates(实体/类别/span/来源) + marginalia(参考)

用户审校 golden_draft.json → 定稿 golden.json（测试读定稿）。

调用链日志：记录每个生产函数的函数名 + 入参 + 出参。

运行：
  cd white-snake-museum-public
  PYTHONIOENCODING=utf-8 python -m tests.annotation_eval.real_llm_eval.build_golden_draft
"""

import asyncio
import json
import logging
import sys
from pathlib import Path

_project_root = Path(__file__).resolve().parent.parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from agent.server.mcp.tools.annotate_passage import handler as annotate_passage_handler
from agent.server.config import settings

DATA_DIR = Path(__file__).resolve().parent / "data"
MANIFEST = DATA_DIR / "sample_manifest.json"
OUTPUT = DATA_DIR / "golden_draft.json"

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-5s %(name)s | %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("build_golden_draft")

CATEGORIES = ["person", "location", "event", "term", "motif"]

# 保留前 N 段人工审校结果，只重新标注后续段（设为 0 表示全部重标）
SKIP_FIRST_N = 3


def _make_llm():
    """按生产 main.py 同款逻辑实例化真实 LLM provider。"""
    if settings.llm_provider == "deepseek":
        from agent.server.llm.deepseek_provider import DeepSeekProvider
        logger.info("[CALL] DeepSeekProvider() -> 真实 LLM（provider=deepseek, model=%s）", settings.effective_model)
        return DeepSeekProvider()
    from agent.server.llm.anthropic_provider import AnthropicProvider
    logger.info("[CALL] AnthropicProvider() -> 真实 LLM（provider=anthropic）")
    return AnthropicProvider()


class SingleParaNodeClient:
    """把单个标注单元包装成 chapter content，供生产 annotate_passage.handler 读取。

    不是 mock 业务逻辑：handler 内部仍走规则预扫/LLM/定位/旁注完整流程。
    """

    def __init__(self, chapter_number: int, title: str, text: str):
        self._ch = {"chapterNumber": chapter_number, "title": title, "content": text}

    async def get_literature_by_chapter(self, chapter_number: int):
        if self._ch["chapterNumber"] == chapter_number:
            logger.info("[CALL] get_literature_by_chapter(%d) -> content=%d字", chapter_number, len(self._ch["content"]))
            return self._ch
        return None


async def _annotate_one(sample: dict, llm) -> dict:
    node = SingleParaNodeClient(sample["chapter_number"], sample["title"], sample["text"])
    result = await annotate_passage_handler(
        chapter_number=sample["chapter_number"],
        paragraph_range=[0, 0],
        categories=CATEGORIES,
        node_client=node,
        llm=llm,
    )
    logger.info(
        "[RET] annotate_passage.handler(chapter=%d, para=[0,0]) -> annotations=%d, marginalia=%d",
        sample["chapter_number"],
        len(result.get("annotations", [])),
        len(result.get("marginalia", [])),
    )
    return result


async def _main_async():
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    samples = manifest["samples"]
    logger.info("[STEP] 读 sample_manifest.json -> %d 段", len(samples))

    llm = _make_llm()

    # 保留已人工审校的前 N 段
    kept = []
    if SKIP_FIRST_N > 0 and OUTPUT.exists():
        old = json.loads(OUTPUT.read_text(encoding="utf-8"))
        kept = old.get("golden_draft", [])[:SKIP_FIRST_N]
        logger.info("[STEP] 保留前 %d 段人工审校结果", len(kept))
    draft = list(kept)

    for i, sample in enumerate(samples[SKIP_FIRST_N:], start=SKIP_FIRST_N):
        logger.info("[STEP] 处理 %d/%d: ch%d %s para=%d (%d字)",
                    i + 1, len(samples), sample["chapter_number"],
                    sample["genre"], sample["paragraph_index"], len(sample["text"]))
        result = await _annotate_one(sample, llm)

        # 生产 handler 返回的坐标是「拼接全文坐标」，这里单段就是段内坐标，直接收下
        candidates = []
        for ann in result.get("annotations", []):
            candidates.append({
                "entity": ann.get("entity", ""),
                "category": ann.get("category", "custom"),
                "start_char": ann.get("start_char", 0),
                "end_char": ann.get("end_char", 0),
                "source": ann.get("source", ""),
            })

        draft.append({
            "chapter_number": sample["chapter_number"],
            "title": sample["title"],
            "genre": sample["genre"],
            "paragraph_index": sample["paragraph_index"],
            "text": sample["text"],
            "candidates": candidates,
            "marginalia_reference": result.get("marginalia", []),
        })

    OUTPUT.write_text(json.dumps({"golden_draft": draft}, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("[RET] 金标准草稿已生成：%d 段 -> %s", len(draft), OUTPUT)
    logger.info("请人工审校 %s，定稿后存为 golden.json", OUTPUT)


def main() -> int:
    try:
        asyncio.run(_main_async())
        return 0
    except Exception:
        logger.exception("金标准草稿生成失败")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
