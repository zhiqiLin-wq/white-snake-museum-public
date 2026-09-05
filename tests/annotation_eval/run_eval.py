"""标注/旁批功能评测一键入口。

用法（Windows 下必须带 UTF-8，否则中文日志乱码）:
  cd white-snake-museum-public
  PYTHONIOENCODING=utf-8 python -m tests.annotation_eval.run_eval

产出（output/<run_id>/）:
  trace_annotate.log  每一步详细 trace（DEBUG 级）
  metrics.json        全部指标（机器读）
  metrics.md          人类可读汇总报告
"""

import json
import logging
import sys
from pathlib import Path

_project_root = Path(__file__).resolve().parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from . import ann_common
from .ann_common import new_run_dir, setup_logging, write_json, format_metrics_markdown
from . import l1_span_validator
from . import l2_annotate_passage
from . import l3_pipeline
from . import l4_coverage
from . import l5_chunking


def _render_markdown(results: dict) -> str:
    lines = ["# 标注/旁批功能评测报告", ""]
    lines.append(f"- run_id: `{results['run_id']}`")
    lines.append("")

    l1 = results["layers"]["L1"]
    lines.append("## L1 纯函数（span 校验 + 意图解析 + 段落切分）")
    lines.append("")
    v = l1["validator"]
    lines.append(f"- span 校验恢复率: **{v['recovery_rate']:.3f}**（{v['recovered']}/{v['total']}）")
    lines.append(f"- 回退层级分布: {json.dumps(v['fallback_distribution'], ensure_ascii=False)}")
    lines.append(f"- 意图解析准确率: **{l1['intent_parser']['accuracy']:.3f}**")
    lines.append(f"- 段落切分对齐准确率: **{l1['split_paragraphs']['accuracy']:.3f}**")
    lines.append("")

    for key in ("clean_offset", "corrupt_offset"):
        blk = results["layers"]["L2"][key]
        lines.append(f"## L2 annotate_passage — {key}")
        lines.append("")
        if blk.get("error"):
            lines.append(f"- **error**: {blk['error']}")
        else:
            lines.append(format_metrics_markdown("L2/" + key, blk["metrics"]))
            lines.append("")
            lines.append(f"- batch_stats: `{json.dumps(blk['batch_stats'], ensure_ascii=False)}`")
            lines.append(f"- confidence 分布: `{json.dumps(blk['confidence_distribution'], ensure_ascii=False)}`")
            lines.append(f"- LLM 调用次数: {blk['llm_calls']}")
        lines.append("")

    l3 = results["layers"]["L3"]
    lines.append("## L3 两轮流水线")
    lines.append("")
    lines.append(format_metrics_markdown("L3/pipeline", l3["metrics"]))
    lines.append("")
    lines.append(f"- coverage 快照: `{json.dumps(l3['coverage'], ensure_ascii=False)}`")
    lines.append(f"- discovery 候选数: {l3['discovery_count']}，LLM 调用次数: {l3['llm_calls']}")
    lines.append("")

    l4 = results["layers"]["L4"]
    lines.append("## L4 标注覆盖率（召回率）")
    lines.append("")
    fx = l4["fixture"]
    lines.append(f"- 样例段落: `{fx['paragraph']}`")
    lines.append(f"- 金标准人名 {fx['total_golden_persons']} 个：词表内 {fx['known_persons']}，词表外 {fx['unknown_persons']}")
    lines.append("")
    fm = l4["full_mock"]
    mm = l4["missing_mock"]
    lines.append(f"- 全量 LLM：总召回率 **{fm['recall']:.3f}**（规则引擎命中 {fm['rule_hit']} + LLM 命中 {fm['llm_hit']}，LLM 调用 {fm['llm_calls']} 次）")
    lines.append(f"- 漏标 LLM：总召回率 **{mm['recall']:.3f}**（规则引擎命中 {mm['rule_hit']} + LLM 命中 {mm['llm_hit']}，漏掉词表外实体后下降）")
    lines.append("")
    lines.append("说明：规则词表内实体由代码精确匹配兜底，100% 不漏；词表外实体的召回取决于 LLM，此测试可量化 LLM 漏标造成的覆盖率缺口。")
    lines.append("")

    l5 = results["layers"]["L5"]
    lines.append("## L5 超长段落块级切分 + 坐标映射")
    lines.append("")
    cs = l5["chunk_splitter"]
    lines.append(f"- 切块函数单测：**{cs['accuracy']:.3f}**（{cs['passed']}/{cs['total']}）")
    ba = l5["block_annotation"]
    if ba.get("error"):
        lines.append(f"- **error**: {ba['error']}")
    else:
        lines.append(f"- 超长段落切成 {ba['total_batches']} 块，标注 {ba['annotations']} 个实体")
        lines.append(f"- span 自洽率（坐标映射正确）: **{ba['span_self_consistency']:.3f}**")
        lines.append(f"- 无重复 span: **{ba['no_duplicate_span']}**")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    run_id, run_dir = new_run_dir()
    logger = setup_logging(run_id, run_dir)
    logger.info("========== 标注评测开始 ==========")

    results = {"run_id": run_id, "layers": {}}

    try:
        results["layers"]["L1"] = l1_span_validator.run()
        results["layers"]["L2"] = l2_annotate_passage.run()
        results["layers"]["L3"] = l3_pipeline.run()
        results["layers"]["L4"] = l4_coverage.run()
        results["layers"]["L5"] = l5_chunking.run()
    except Exception:
        logger.exception("评测过程抛出异常")
        return 1

    write_json(results, run_dir / "metrics.json")
    (run_dir / "metrics.md").write_text(_render_markdown(results), encoding="utf-8")

    logger.info("========== 标注评测结束 ==========")
    logger.info("输出目录: %s", run_dir)
    logger.info("报告: %s", run_dir / "metrics.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
