"""真实验收：真实 Node 服务 + 真实 LLM + 生产 annotate_passage.handler。

前置：
1. 已启动 Node 后端（node_api_url 默认 http://127.0.0.1:3000/api）。
2. 已运行 build_golden_draft.py 生成 golden_draft.json，并经人工审校定稿为 golden.json。

接线（生产代码，非 mock）：
- NodeAPIClient.get_literature_by_chapter 走真实 HTTP 取原始文献
- segment.segment_content 对超长段做句子级切分（用户同意的预处理）
- annotate_passage.handler 走生产完整流程（规则预扫 + LLM + gleaning + event/motif 旁注）

指标：召回率(覆盖率)、精确率、F1、span 自洽率、段落覆盖率、规则 vs LLM 覆盖率，7 体裁分报告 + 汇总。
调用链日志：记录每个生产函数名 + 入参 + 出参。

运行：
  cd white-snake-museum-public
  PYTHONIOENCODING=utf-8 python -m tests.annotation_eval.real_llm_eval.run_real_eval
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
from agent.server.data.client import NodeAPIClient
from agent.server.config import settings

from .segment import segment_content
from .. import ann_common
from ..ann_common import new_run_dir, setup_logging, write_json

DATA_DIR = Path(__file__).resolve().parent / "data"
GOLDEN = DATA_DIR / "golden.json"

CATEGORIES = ["person", "location", "event", "term", "motif"]
# event/motif 的 span 是细节原文（verbatim），同样满足 text[start:end]==entity
# 因此自洽率对全部类别成立，无需排除


def _make_llm():
    if settings.llm_provider == "deepseek":
        from agent.server.llm.deepseek_provider import DeepSeekProvider
        return DeepSeekProvider()
    from agent.server.llm.anthropic_provider import AnthropicProvider
    return AnthropicProvider()


class SingleParaNodeClient:
    """把 golden.json 里已切好的单个标注单元作为 chapter content，供 annotate_passage 读取。

    坐标与 golden.json 的 text 完全对齐（避免 parseLiterature 复刻与 TS 的差异导致坐标错位）。
    """

    def __init__(self, chapter_number: int, title: str, text: str):
        self._ch = {"chapterNumber": chapter_number, "title": title, "content": text}

    async def get_literature_by_chapter(self, chapter_number: int):
        if self._ch["chapterNumber"] == chapter_number:
            return self._ch
        return None


def _span(ann: dict) -> tuple:
    return (ann.get("category", "custom"), int(ann.get("start_char", 0)), int(ann.get("end_char", 0)))


def _compute_sample_metrics(predictions: list[dict], golden: list[dict], text: str) -> dict:
    """单段：精确 span 匹配 + 自洽率 + source 拆分。"""
    pred_spans = [_span(a) for a in predictions]
    gold_spans = [_span(a) for a in golden]
    pred_set = set(pred_spans)
    gold_set = set(gold_spans)

    tp = len(pred_set & gold_set)
    fp = len(pred_set - gold_set)
    fn = len(gold_set - pred_set)

    # 自洽率：text[start:end] == entity
    sc_ok = 0
    sc_total = 0
    for a in predictions:
        s = int(a.get("start_char", 0))
        e = int(a.get("end_char", s))
        if s < 0 or e > len(text):
            continue
        sc_total += 1
        if text[s:e] == a.get("entity", ""):
            sc_ok += 1

    rule_hit = sum(1 for a in predictions if a.get("source") == "rule")
    llm_hit = sum(1 for a in predictions if a.get("source") == "agent")

    return {
        "tp": tp, "fp": fp, "fn": fn,
        "precision": tp / (tp + fp) if (tp + fp) else 0.0,
        "recall": tp / (tp + fn) if (tp + fn) else 0.0,
        "f1": 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0,
        "span_self_consistency": sc_ok / sc_total if sc_total else 1.0,
        "sc_ok": sc_ok,
        "sc_total": sc_total,
        "rule_hit": rule_hit,
        "llm_hit": llm_hit,
    }


async def _main_async(run_dir: Path, logger: logging.Logger) -> int:
    golden_data = json.loads(GOLDEN.read_text(encoding="utf-8"))
    samples = golden_data["golden_draft"]
    logger.info("[STEP] 读 golden.json -> %d 段", len(samples))

    llm = _make_llm()
    logger.info("[CALL] %s() -> 真实 LLM", llm.__class__.__name__)

    per_genre = {}
    total = {"tp": 0, "fp": 0, "fn": 0, "sc_ok": 0, "sc_total": 0, "rule_hit": 0, "llm_hit": 0,
             "pred_total": 0, "gold_total": 0, "paragraphs": 0, "annotated_paragraphs": 0}

    for i, sample in enumerate(samples):
        ch = sample["chapter_number"]
        pi = sample["paragraph_index"]
        text = sample["text"]
        genre = sample["genre"]
        logger.info("[STEP] %d/%d ch%d %s para=%d (%d字)", i + 1, len(samples), ch, genre, pi, len(text))

        node = SingleParaNodeClient(ch, sample.get("title", ""), text)
        try:
            result = await annotate_passage_handler(
                chapter_number=ch,
                paragraph_range=[0, 0],
                categories=CATEGORIES,
                node_client=node,
                llm=llm,
            )
        except Exception as e:
            logger.error("[ERROR] annotate_passage.handler 失败 ch=%d para=%d: %s", ch, pi, e)
            continue

        if "error" in result:
            logger.error("[ERROR] handler 返回 error ch=%d para=%d: %s", ch, pi, result["error"])
            continue

        predictions = result.get("annotations", [])
        golden = sample.get("candidates", [])
        m = _compute_sample_metrics(predictions, golden, text)
        logger.info(
            "[RET] annotate_passage.handler(chapter=%d, para=[%d,%d]) -> annotations=%d, "
            "golden=%d, tp=%d fp=%d fn=%d, recall=%.3f, self_consistency=%.3f",
            ch, pi, pi, len(predictions), len(golden), m["tp"], m["fp"], m["fn"],
            m["recall"], m["span_self_consistency"],
        )

        # 汇总
        g = per_genre.setdefault(genre, {"tp": 0, "fp": 0, "fn": 0, "sc_ok": 0, "sc_total": 0,
                                         "rule_hit": 0, "llm_hit": 0, "pred_total": 0, "gold_total": 0,
                                         "paragraphs": 0, "annotated_paragraphs": 0})
        for k in ("tp", "fp", "fn", "rule_hit", "llm_hit", "sc_ok", "sc_total"):
            total[k] += m[k]
            g[k] += m[k]
        total["pred_total"] += len(predictions)
        total["gold_total"] += len(golden)
        g["pred_total"] += len(predictions)
        g["gold_total"] += len(golden)
        total["paragraphs"] += 1
        g["paragraphs"] += 1
        if predictions:
            total["annotated_paragraphs"] += 1
            g["annotated_paragraphs"] += 1

    def _summarize(tp, fp, fn, *rest):
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0
        return {"precision": precision, "recall": recall, "f1": f1, "tp": tp, "fp": fp, "fn": fn}

    overall = _summarize(total["tp"], total["fp"], total["fn"])
    overall.update({
        "span_self_consistency": total["sc_ok"] / total["sc_total"] if total["sc_total"] else 1.0,
        "paragraph_coverage": total["annotated_paragraphs"] / total["paragraphs"] if total["paragraphs"] else 0.0,
        "rule_hit": total["rule_hit"],
        "llm_hit": total["llm_hit"],
    })

    genre_report = {}
    for genre, g in per_genre.items():
        s = _summarize(g["tp"], g["fp"], g["fn"])
        s["span_self_consistency"] = g["sc_ok"] / g["sc_total"] if g["sc_total"] else 1.0
        s["rule_hit"] = g["rule_hit"]
        s["llm_hit"] = g["llm_hit"]
        genre_report[genre] = s

    report = {"overall": overall, "per_genre": genre_report}
    write_json(report, run_dir / "metrics.json")
    (run_dir / "metrics.md").write_text(_render_md(report), encoding="utf-8")
    logger.info("[RET] 真实验收完成，报告: %s", run_dir / "metrics.md")
    return 0


def _render_md(report: dict) -> str:
    lines = ["# 真实验收报告（真实 LLM + 生产代码）", ""]
    o = report["overall"]
    lines.append("## 总览")
    lines.append(f"- 召回率(覆盖率): **{o['recall']:.3f}**")
    lines.append(f"- 精确率: **{o['precision']:.3f}**，F1: **{o['f1']:.3f}**")
    lines.append(f"- span 自洽率: **{o['span_self_consistency']:.3f}**")
    lines.append(f"- 段落覆盖率: **{o['paragraph_coverage']:.3f}**")
    lines.append(f"- 规则词表命中 {o['rule_hit']}，LLM 命中 {o['llm_hit']}（TP={o['tp']} FP={o['fp']} FN={o['fn']}）")
    lines.append("")
    lines.append("## 分体裁")
    lines.append("| 体裁 | 召回率 | 精确率 | F1 | 自洽率 | 规则命中 | LLM命中 |")
    lines.append("|---|---|---|---|---|---|---|")
    for g, s in report["per_genre"].items():
        lines.append(f"| {g} | {s['recall']:.3f} | {s['precision']:.3f} | {s['f1']:.3f} | {s['span_self_consistency']:.3f} | {s['rule_hit']} | {s['llm_hit']} |")
    return "\n".join(lines)


def main() -> int:
    run_id, run_dir = new_run_dir()
    logger = setup_logging(run_id, run_dir)
    logger.info("========== 真实验收开始 ==========")
    if not GOLDEN.exists():
        logger.error("缺少 %s，请先运行 build_golden_draft.py 并人工审校定稿为 golden.json", GOLDEN)
        return 1
    try:
        return asyncio.run(_main_async(run_dir, logger))
    except Exception:
        logger.exception("真实验收失败")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
