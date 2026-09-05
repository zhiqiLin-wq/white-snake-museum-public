"""标注/旁批功能评测公共层。

职责:
1. 每次运行生成独立 run_id 与 output 目录（永不覆盖历史结果）。
2. 双粒度日志: 控制台(INFO) + 文件(DEBUG)，文件里记录每一步 trace。
3. 指标计算: 精确/容差 P/R/F1、span 自洽率、类别准确率、段落覆盖率、
   偏移恢复率、诊断计数。
4. 报告落盘: metrics.json（机器读） + metrics.md（人类读）。

坐标系约定（与生产一致）:
  span = (paragraph_index, category, start_char, end_char)
  start/end 均为"段内 0-based 字符偏移"，end 为开区间。
"""

import json
import logging
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path

# 北京时间，与生产日志时间观感一致
BEIJING_TZ = timezone(timedelta(hours=8))

_EVAL_DIR = Path(__file__).resolve().parent
OUTPUT_ROOT = _EVAL_DIR / "output"

# 事件/母题是"语义区间"，start/end 不保证字面等于 entity，自洽率需排除。
# 其余类别（人物/地点/术语/自定义）字面自洽是硬约束。
SELF_CONSISTENCY_EXCLUDE_CATEGORIES = {"event", "motif"}

# 类别 -> 中文标签（报告可读性）
CATEGORY_LABELS = {
    "person": "人物",
    "location": "地点",
    "event": "事件",
    "term": "术语",
    "motif": "母题",
    "custom": "自定义",
}


def bj_now() -> datetime:
    return datetime.now(BEIJING_TZ)


class _BjFormatter(logging.Formatter):
    def formatTime(self, record, datefmt=None):
        dt = datetime.fromtimestamp(record.created, tz=BEIJING_TZ)
        return dt.strftime("%H:%M:%S") if datefmt is None else dt.strftime(datefmt)


def setup_logging(run_id: str, run_dir: Path):
    """配置根 logger: 控制台 INFO + 文件 DEBUG。返回 logger。"""
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    # 清掉可能残留的 handler（多次 run 时避免重复输出）
    for h in list(root.handlers):
        root.removeHandler(h)

    fmt = _BjFormatter("%(asctime)s %(levelname)-5s %(name)s | %(message)s", datefmt="%H:%M:%S")

    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.INFO)
    console.setFormatter(fmt)
    root.addHandler(console)

    file_handler = logging.FileHandler(str(run_dir / "trace_annotate.log"), mode="a", encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(fmt)
    root.addHandler(file_handler)

    logger = logging.getLogger("annotation_eval")
    logger.info("run_id=%s output_dir=%s", run_id, run_dir)
    return logger


def new_run_dir() -> tuple[str, Path]:
    """创建带时间戳的输出目录并返回 (run_id, run_dir)。"""
    stamp = datetime.now(BEIJING_TZ).strftime("%Y%m%d_%H%M%S")
    run_dir = OUTPUT_ROOT / stamp
    run_dir.mkdir(parents=True, exist_ok=True)
    return stamp, run_dir


def write_json(data: dict, path: Path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


# ---------------------------------------------------------------------------
# 指标计算
# ---------------------------------------------------------------------------

def _span(ann: dict) -> tuple:
    return (
        int(ann["paragraph_index"]),
        ann.get("category", "custom"),
        int(ann.get("start_char", 0)),
        int(ann.get("end_char", int(ann.get("start_char", 0)))),
    )


def _match_span(p: tuple, g: tuple, delta: int) -> bool:
    p_para, p_cat, p_s, p_e = p
    g_para, g_cat, g_s, g_e = g
    return (
        p_para == g_para
        and p_cat == g_cat
        and p_s == g_s
        and abs(p_e - g_e) <= delta
    )


def compute_metrics(
    predictions: list[dict],
    golden: list[dict],
    paragraph_texts: dict[int, str],
) -> dict:
    """对预测结果计算全部指标。

    predictions: [{paragraph_index, entity, category, start_char, end_char, ...}]
    golden:      同构金标准
    paragraph_texts: {paragraph_index: text}，用于 span 自洽率与锚点合法性
    """
    if not golden or not predictions:
        # 空集合时 F1 无意义，返回哨兵
        return {
            "counts": {"predicted": len(predictions), "golden": len(golden)},
            "empty": True,
        }

    pred_spans = [_span(a) for a in predictions]
    gold_spans = [_span(a) for a in golden]

    # 去重后的跨度集合（同一 span 多条只算一次）
    pred_span_set = set(pred_spans)
    gold_span_set = set(gold_spans)

    def matched_pairs(delta: int):
        """返回 (命中金标准的预测 span 列表, 命中预测的金标准 span 列表)。"""
        hit_pred = set()
        hit_gold = set()
        for p in pred_span_set:
            for g in gold_span_set:
                if _match_span(p, g, delta):
                    hit_pred.add(p)
                    hit_gold.add(g)
                    break
        return hit_pred, hit_gold

    def prf(hit_pred, hit_gold):
        tp = len(hit_pred)
        precision = tp / len(pred_span_set) if pred_span_set else 0.0
        recall = tp / len(gold_span_set) if gold_span_set else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        return {"precision": precision, "recall": recall, "f1": f1, "tp": tp}

    delta_table = {}
    for delta in (0, 1, 2):
        hp, hg = matched_pairs(delta)
        delta_table[f"delta_{delta}"] = prf(hp, hg)

    # 类别准确率：span 位置命中（δ=0）时，类别一致的比例
    category_hit = 0
    category_total = 0
    for p in pred_span_set:
        p_para, p_cat, p_s, p_e = p
        for g in gold_span_set:
            g_para, g_cat, g_s, g_e = g
            if p_para == g_para and p_s == g_s and p_e == g_e:
                category_total += 1
                if p_cat == g_cat:
                    category_hit += 1
                break
    category_accuracy = category_hit / category_total if category_total else 0.0

    # span 自洽率：text[start:end] == entity（排除事件/母题）
    sc_total = 0
    sc_ok = 0
    for a in predictions:
        cat = a.get("category", "custom")
        if cat in SELF_CONSISTENCY_EXCLUDE_CATEGORIES:
            continue
        text = paragraph_texts.get(int(a["paragraph_index"]), "")
        s = int(a.get("start_char", 0))
        e = int(a.get("end_char", s))
        if s < 0 or e > len(text):
            continue
        sc_total += 1
        if text[s:e] == a.get("entity", ""):
            sc_ok += 1
    span_self_consistency = sc_ok / sc_total if sc_total else 1.0

    # 段落覆盖率：分母 = 金标准里有标注的段落
    gold_paras = {int(g["paragraph_index"]) for g in golden}
    pred_paras = {int(p["paragraph_index"]) for p in predictions}
    paragraph_coverage = len(pred_paras & gold_paras) / len(gold_paras) if gold_paras else 0.0

    # 分类别 F1（δ=0）
    per_category = {}
    for cat in sorted({a.get("category", "custom") for a in predictions} | {g.get("category", "custom") for g in golden}):
        pc = [s for s in pred_span_set if s[1] == cat]
        gc = [s for s in gold_span_set if s[1] == cat]
        hit = len(set(pc) & set(gc))
        precision = hit / len(pc) if pc else 0.0
        recall = hit / len(gc) if gc else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        per_category[cat] = {
            "label": CATEGORY_LABELS.get(cat, cat),
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "predicted": len(pc),
            "golden": len(gc),
        }

    return {
        "empty": False,
        "counts": {"predicted": len(pred_spans), "predicted_unique": len(pred_span_set),
                   "golden": len(gold_spans), "golden_unique": len(gold_span_set)},
        "delta": delta_table,
        "category_accuracy": category_accuracy,
        "span_self_consistency": span_self_consistency,
        "paragraph_coverage": paragraph_coverage,
        "per_category": per_category,
    }


def map_handler_annotations(result: dict, content: str) -> list[dict]:
    """把 annotate_passage 输出的全文拼接坐标，映射回段落相对坐标。

    content 为章节全文（\n\n 分隔段落），与 handler 内部切段规则一致。
    """
    paragraphs = [p.strip() for p in content.split("\n\n") if p.strip()]
    boundaries = []
    offset = 0
    for idx, text in enumerate(paragraphs):
        boundaries.append({"idx": idx, "start": offset, "end": offset + len(text)})
        offset += len(text) + 2

    predictions = []
    for ann in result.get("annotations", []):
        sc = int(ann.get("start_char", 0))
        ec = int(ann.get("end_char", sc))
        para_idx = None
        rel_start = None
        for b in boundaries:
            if b["start"] <= sc < b["end"]:
                para_idx = b["idx"]
                rel_start = sc - b["start"]
                break
        if para_idx is None:
            logging.getLogger("annotation_eval").warning(
                "[map] 无法映射标注到段落: %r", ann
            )
            continue
        predictions.append({
            "paragraph_index": para_idx,
            "entity": ann.get("entity", ""),
            "category": ann.get("category", "custom"),
            "start_char": rel_start,
            "end_char": rel_start + (ec - sc),
            "confidence": ann.get("confidence", ""),
            "source": ann.get("source", ""),
        })
    return predictions


def format_metrics_markdown(layer: str, metrics: dict) -> str:
    """把单个 layer 的指标渲染成 Markdown 表格。"""
    lines = [f"## {layer}"]
    if metrics.get("empty"):
        lines.append(f"- 预测数 {metrics['counts']['predicted']}，金标准数 {metrics['counts']['golden']}（跳过 F1）")
        return "\n".join(lines)

    c = metrics["counts"]
    lines.append(f"- 预测 {c['predicted']}（去重 {c['predicted_unique']}） / 金标准 {c['golden']}（去重 {c['golden_unique']}）")
    lines.append("")
    lines.append("| 容差 | Precision | Recall | F1 | TP |")
    lines.append("|---|---|---|---|---|")
    for delta in ("delta_0", "delta_1", "delta_2"):
        d = metrics["delta"][delta]
        lines.append(f"| {delta.replace('delta_', 'δ=')} | {d['precision']:.3f} | {d['recall']:.3f} | {d['f1']:.3f} | {d['tp']} |")
    lines.append("")
    lines.append(f"- 类别准确率: **{metrics['category_accuracy']:.3f}**")
    lines.append(f"- span 自洽率: **{metrics['span_self_consistency']:.3f}**")
    lines.append(f"- 段落覆盖率: **{metrics['paragraph_coverage']:.3f}**")
    lines.append("")
    lines.append("| 类别 | Precision | Recall | F1 | 预测 | 金标准 |")
    lines.append("|---|---|---|---|---|---|")
    for cat, m in metrics["per_category"].items():
        lines.append(f"| {m['label']}({cat}) | {m['precision']:.3f} | {m['recall']:.3f} | {m['f1']:.3f} | {m['predicted']} | {m['golden']} |")
    return "\n".join(lines)
