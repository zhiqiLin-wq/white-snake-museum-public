"""批量运行三个 LLM 的 RAG 评估。

运行方式:
    cd white-snake-museum-public
    python -m rag_eval.scripts.run_all_evals

输出:
    results/deepseek_report.json
    results/qwen_report.json
    results/ernie_report.json
    results/comparison_summary.txt
"""
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
RESULTS_DIR = PROJECT_ROOT / "rag_eval" / "results"
CONFIG_DIR = PROJECT_ROOT / "rag_eval" / "configs"

EVALS = [
    {
        "name": "deepseek",
        "label": "DeepSeek",
        "config": str(CONFIG_DIR / "test_config_deepseek.yaml"),
        "queries_file": "queries.jsonl",
        "qrels_file": "qrels_auto.tsv",
    },
    {
        "name": "qwen",
        "label": "千问 (Qwen)",
        "config": str(CONFIG_DIR / "test_config_qwen.yaml"),
        "queries_file": "queries_qwen.jsonl",
        "qrels_file": "qrels_auto_qwen.tsv",
    },
    {
        "name": "ernie",
        "label": "文心一言 (ERNIE)",
        "config": str(CONFIG_DIR / "test_config_ernie.yaml"),
        "queries_file": "queries_ernie.jsonl",
        "qrels_file": "qrels_auto_ernie.tsv",
    },
]


def run_eval(name: str, config: str) -> dict | None:
    """运行一次评估，返回 JSON 报告数据。"""
    print(f"\n{'=' * 70}")
    print(f"[EVAL] {name}")
    print(f"  Config: {config}")
    print(f"{'=' * 70}")

    # 删除旧报告确保生成新报告
    latest_report = RESULTS_DIR / "latest_report.json"
    if latest_report.exists():
        latest_report.unlink()

    result = subprocess.run(
        [
            sys.executable, "-m", "rag_eval.evaluate",
            "--config", config,
            "--verbose",
        ],
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
        timeout=1800,  # 30 min timeout
    )

    # 打印输出 (截断)
    output_lines = result.stdout.split("\n")
    for line in output_lines[-30:]:  # last 30 lines
        if line.strip():
            print(f"  {line.strip()}")

    if result.returncode != 0:
        print(f"  [WARN] Exit code: {result.returncode}")
        # 打印最后几行 stderr
        stderr_lines = result.stderr.strip().split("\n")
        for line in stderr_lines[-5:]:
            if line.strip():
                print(f"  [STDERR] {line.strip()}")

    # 读取生成的报告
    if latest_report.exists():
        with open(latest_report, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


def extract_summary(report: dict, name: str) -> dict:
    """从报告中提取关键指标摘要。"""
    summary = {"name": name, "configs": {}}
    for config_id in ["D", "S", "F", "R"]:
        result = report.get("results", {}).get(config_id, {})
        agg = result.get("aggregates", {})
        summary["configs"][config_id] = {
            "recall_at_10": agg.get("recall_at_10", {}).get("mean", 0),
            "recall_at_5": agg.get("recall_at_5", {}).get("mean", 0),
            "mrr": agg.get("mrr", {}).get("mean", 0),
            "ndcg_at_10": agg.get("ndcg_at_10", {}).get("mean", 0),
            "hit_at_1": agg.get("hit_at_1", {}).get("mean", 0),
            "average_precision": agg.get("average_precision", {}).get("mean", 0),
            "r_precision": agg.get("r_precision", {}).get("mean", 0),
            "coverage_rate": agg.get("coverage_rate", {}).get("mean", 0),
            "n_queries": result.get("n_queries", 0),
            "total_time_s": result.get("total_time_s", 0),
        }
    summary["n_queries"] = report.get("metadata", {}).get("n_queries", 0)
    return summary


def write_comparison(summaries: list[dict], output_path: Path):
    """生成对比报告。"""
    lines = [
        "=" * 80,
        "RAG 三 LLM 评估对比报告",
        f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "=" * 80,
        "",
        "## 评估说明",
        "  - 每个 LLM 使用自己的 queries + auto-qrels 进行评估",
        "  - Auto-qrels: 每条 query 的源 chunk 标记为 relevance=2",
        "  - 评估配置: D=DenseOnly, S=SparseOnly(BM25), F=Fusion, R=FullPipeline",
        "",
    ]

    # 总览对比表
    for metric in ["recall_at_10", "recall_at_5", "mrr", "ndcg_at_10", "hit_at_1", "average_precision", "r_precision", "coverage_rate"]:
        metric_label = {
            "recall_at_10": "Recall@10",
            "recall_at_5": "Recall@5",
            "mrr": "MRR",
            "ndcg_at_10": "NDCG@10",
            "hit_at_1": "Hit@1",
            "average_precision": "MAP",
            "r_precision": "R-Precision",
            "coverage_rate": "Coverage",
        }[metric]

        lines.append(f"## {metric_label} (R 配置)")
        lines.append("")
        header = f"  {'LLM':<25}"
        for s in summaries:
            header += f" {s['name']:<12}"
        lines.append(header)
        lines.append("  " + "-" * (25 + 12 * len(summaries)))

        row = f"  {'':25}"
        for s in summaries:
            val = s["configs"].get("R", {}).get(metric, 0)
            row += f" {val:<12.4f}"
        lines.append(row)
        lines.append("")

    # 按配置横向对比 (Recall@10)
    lines.append("## Recall@10 四配置对比")
    lines.append("")
    header = f"  {'LLM':<25}"
    for cid in ["D", "S", "F", "R"]:
        header += f" {cid:<10}"
    lines.append(header)
    lines.append("  " + "-" * (25 + 10 * 4))

    for s in summaries:
        row = f"  {s['name']:<25}"
        for cid in ["D", "S", "F", "R"]:
            val = s["configs"].get(cid, {}).get("recall_at_10", 0)
            row += f" {val:<10.4f}"
        row += f" (n={s.get('n_queries', '?')})"
        lines.append(row)
    lines.append("")

    # 查询数量信息
    lines.append("## 数据集规模")
    lines.append("")
    for s in summaries:
        r_n = s["configs"].get("R", {}).get("n_queries", "?")
        lines.append(f"  {s['name']:<25}: {r_n} queries")
    lines.append("")

    # 耗时
    lines.append("## 评估耗时 (R 配置)")
    lines.append("")
    for s in summaries:
        r_time = s["configs"].get("R", {}).get("total_time_s", 0)
        lines.append(f"  {s['name']:<25}: {r_time:.0f}s")

    content = "\n".join(lines)
    output_path.write_text(content, encoding="utf-8")
    print(f"\n[COMPARISON] -> {output_path}")
    return content


def main():
    print("=" * 70)
    print("RAG 三 LLM 评估批量运行")
    print("=" * 70)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    all_summaries = []
    all_reports = {}

    for eval_cfg in EVALS:
        t0 = time.monotonic()

        report = run_eval(eval_cfg["name"], eval_cfg["config"])

        elapsed = time.monotonic() - t0
        print(f"\n  [{eval_cfg['label']}] 耗时: {elapsed:.0f}s")

        if report is None:
            print(f"  [ERROR] {eval_cfg['label']} 评估失败，跳过")
            continue

        # 保存报告副本
        report_path = RESULTS_DIR / f"{eval_cfg['name']}_report.json"
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(f"  [SAVED] -> {report_path}")

        all_reports[eval_cfg['name']] = report
        summary = extract_summary(report, eval_cfg['label'])
        all_summaries.append(summary)

    # 生成对比报告
    if len(all_summaries) >= 1:
        comparison_path = RESULTS_DIR / "comparison_summary.txt"
        comparison_text = write_comparison(all_summaries, comparison_path)
        print(f"\n{comparison_text}")

    print(f"\n[DONE] 评估全部完成")
    print(f"  报告目录: {RESULTS_DIR}")
    for name in all_reports:
        print(f"    {name}_report.json")


if __name__ == "__main__":
    main()
