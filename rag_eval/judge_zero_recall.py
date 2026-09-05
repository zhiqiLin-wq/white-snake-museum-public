"""LLM-as-judge 补全 qrels 漏标，重算召回。

背景:
    qrels 是「候选池采样标注」——每条 query 生成时只对约 17 个候选 chunk 标注，
    并非对全部 427 chunk 标注。因此检索 top-10 里「确实相关但没进候选池」的
    chunk 不在 qrels 正例里，导致 recall 被系统性低估。

本脚本流程:
    1. 用生产配置跑检索，对每条 query 记录: 检索 top-10、qrels 命中情况、recall。
    2. 对 recall<1 的 query，用 LLM judge 逐一判断「检索返回但 qrels 未标相关」
       的 chunk 是否真的相关（temperature=0.2, deepseek-v4-flash）。
    3. LLM 判相关的 chunk 追加为命中（补全漏标），重算 recall@10。
    4. 汇总: 补全前 vs 补全后的 recall@10 / zero 数，量化漏标低估幅度。

    命中的 chunk 跳过不判（只在 retrieved 里 qrels 未标的 chunk 上判）。

Usage:
    python rag_eval/judge_zero_recall.py                     # 全部 4 数据集
    python rag_eval/judge_zero_recall.py --datasets academic_qwen
    python rag_eval/judge_zero_recall.py --skip-judge        # 只跑检索出日志，不调 LLM
    python rag_eval/judge_zero_recall.py --limit 10          # 只 judge 前 10 条 (调试)
"""
import argparse
import asyncio
import json
import re
import sys
import time
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
AGENT_DIR = PROJECT_ROOT / "agent"
for p in [str(PROJECT_ROOT), str(AGENT_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)

GT_DIR = Path(__file__).resolve().parent / "ground_truth"
RESULTS_DIR = Path(__file__).resolve().parent / "results"
ALL_DATASETS = ["academic_deepseek", "academic_qwen", "daily_deepseek", "daily_qwen"]

JUDGE_CONCURRENCY = 10       # judge 并发（过高会触发 flash 限流空 content）
CHUNK_TRUNC = 600            # 单个 chunk 截断长度（判相关性足够）


# ==============================================================================
# TeeLogger: 同时写 stdout + 文件
# ==============================================================================
class TeeLogger:
    def __init__(self, filepath: Path):
        self.file = open(str(filepath), "w", encoding="utf-8")
        self.stdout = sys.stdout

    def write(self, msg: str):
        self.file.write(msg)
        self.file.flush()
        self.stdout.write(msg)

    def close(self):
        self.file.close()


# ==============================================================================
# JSON 解析容错
# ==============================================================================
def _strip_json(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        text = "\n".join(lines[1:]) if len(lines) > 1 else text[3:]
    if text.endswith("```"):
        text = text[:-3].strip()
    return text


def _parse_json(text: str):
    text = _strip_json(text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    m = re.search(r'\[.*\]', text, re.DOTALL)
    if not m:
        m = re.search(r'\{.*\}', text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group())
        except json.JSONDecodeError:
            pass
    return None


# ==============================================================================
# Judge prompt（单一判定：该 chunk 是否与 query 相关，对齐 qrels 的 rel>=1 标准）
# ==============================================================================
JUDGE_SYSTEM = """你是一位白蛇传文献研究专家，正在复核一个检索评测数据集的相关性标注质量。

给定一个「查询问题」和「检索系统返回但未被标注为相关的文本块」，你的任务是逐一判断每个文本块是否「与查询问题相关，能帮助回答该问题」。

判断标准（对齐原标注的"相关"定义，即 relevance >= 1）:
- 1 (相关): 该文本块对回答该问题有帮助，包括：直接包含答案、事件的前因后果、同一情节的不同侧面、相关背景知识、跨文献的同类描写。
- 0 (无关): 该文本块对回答该问题没有帮助，或讨论完全无关的内容，或仅提到相同人物/主题但没有更多关联。

重要规则:
1. 必须逐一判断每个文本块，逐块输出，不能笼统。
2. 版本区分: 若问题明确指向某版本（如"方成培《雷峰塔》"），另一版本的同情节描写算「跨文献同类描写」，可判相关（1）。
3. 只看文本块内容本身是否相关，不要因为「这是检索返回的结果」就倾向判相关。
4. 只输出 JSON 数组，不要输出任何其他文字。

输出格式（元素数必须等于输入的文本块数）:
[{"id": 1, "related": 0, "reason": "一句话理由"}, ...]"""


def _build_judge_user_prompt(query_text: str, chunks: list[dict]) -> str:
    parts = [f"查询问题: {query_text}\n"]
    parts.append(f"以下 {len(chunks)} 个文本块是检索返回但未被标注为相关的，请逐一判断是否相关:\n")
    for i, c in enumerate(chunks):
        meta = c.get("metadata", {})
        dynasty = meta.get("dynasty", "?")
        genre = meta.get("genre", "?")
        source_file = meta.get("source_file", "?")
        content = c.get("content", "")
        parts.append(f"[{i + 1}] chunk_id={c.get('chunk_id', '')} "
                     f"(朝代={dynasty}/体裁={genre}/来源={source_file})\n{content}\n")
    parts.append(f"请逐一判断以上 {len(chunks)} 个文本块，输出 {len(chunks)} 个元素的 JSON 数组。")
    return "\n".join(parts)


# ==============================================================================
# 检索阶段
# ==============================================================================
def _load_dataset(ds: str):
    from rag_eval.loader import load_corpus, load_queries, load_qrels, load_all_qrels

    corpus = load_corpus(GT_DIR / "corpus.jsonl")
    queries = load_queries(GT_DIR / f"queries_{ds}.jsonl")
    qrels = load_qrels(GT_DIR / f"qrels_{ds}.tsv")
    all_qrels = load_all_qrels(GT_DIR / f"qrels_{ds}.tsv")
    return corpus, queries, qrels, all_qrels


async def _retrieve_prod(retriever, expander, text: str) -> list[dict]:
    from server.rag.query_intent import detect_version_filters, research_penalty_for

    expand_result = expander.expand_weighted(text)
    eq_dict = {
        "dense_query": text,
        "sparse_query": expand_result["expanded_query"],
        "sparse_weights": expand_result["token_weights"],
        "research_penalty": research_penalty_for(text),
    }
    version_boost = detect_version_filters(text)
    return await retriever.retrieve(eq_dict, top_k=10, version_boost=version_boost)


async def run_retrieval(datasets, retriever, expander, log) -> dict:
    """跑检索，输出全部 query 日志，收集 recall<1 的 judge cases。

    Returns:
        {"all_recall_old": [...], "cases": [...], "n_full": int}
    """
    from rag_eval.metrics import recall_at_k

    all_recall_old: list[float] = []
    cases: list[dict] = []
    n_full = 0

    for ds in datasets:
        corpus, queries, qrels, all_qrels = _load_dataset(ds)
        log.write(f"\n{'=' * 80}\n")
        log.write(f"  DATASET: {ds}  (queries {len(queries)} / qrels {len(qrels)})\n")
        log.write(f"{'=' * 80}\n")

        ds_recall = []
        for idx, q in enumerate(queries):
            qid = q.get("query_id", "")
            text = q.get("text", "")
            rels = qrels.get(qid, {})
            if not rels:
                continue
            rel_set = set(rels.keys())

            results = await _retrieve_prod(retriever, expander, text)
            retrieved_ids = [r.get("chunk_id", "") for r in results]
            rec = recall_at_k(retrieved_ids, rel_set, 10)
            hits = [rid for rid in retrieved_ids if rid in rel_set]
            all_recall_old.append(rec)
            ds_recall.append(rec)

            log.write(f"\n  [{ds} {idx + 1}/{len(queries)}] {qid}  recall={rec:.2f}  "
                      f"| 命中 {len(hits)}/{len(rel_set)}  |  {q.get('category', '')}\n")
            log.write(f"    Query: {text}\n")

            if rec >= 1.0:
                n_full += 1
                continue

            # recall<1: 收集「检索返回但 qrels 未标相关」的 chunk 供 judge
            unhit_chunks = []
            for r in results:
                cid = r.get("chunk_id", "")
                if cid in rel_set:
                    continue  # 已命中的跳过，不 judge
                meta = r.get("metadata", {})
                status = ("hard_negative" if cid in all_qrels.get(qid, {})
                          else "unlabeled")
                unhit_chunks.append({
                    "chunk_id": cid,
                    "content": r.get("content", "")[:CHUNK_TRUNC],
                    "metadata": meta,
                    "qrels_status": status,
                })
            cases.append({
                "dataset": ds,
                "query_id": qid,
                "query_text": text,
                "qrels_rel": sorted(rel_set),
                "hits": hits,
                "recall_old": rec,
                "unhit_chunks": unhit_chunks,
            })

        if ds_recall:
            log.write(f"\n  [{ds}] 有标注 {len(ds_recall)} 条, 原 recall@10 均值 "
                      f"{sum(ds_recall) / len(ds_recall):.4f}\n")

    total_annotated = len(all_recall_old)
    log.write(f"\n{'=' * 80}\n")
    log.write(f"  总计 {total_annotated} 条有标注 query, 其中 recall<1 待 judge {len(cases)} 条\n")
    log.write(f"{'=' * 80}\n")
    return {"all_recall_old": all_recall_old, "cases": cases, "n_full": n_full}


# ==============================================================================
# Judge 阶段
# ==============================================================================
async def _judge_one(provider, case: dict, sem: asyncio.Semaphore, log) -> dict:
    async with sem:
        chunks = case["unhit_chunks"]
        if not chunks:
            case["llm_rel"] = []
            case["recall_new"] = case["recall_old"]
            return case
        user = _build_judge_user_prompt(case["query_text"], chunks)
        # flash 并发下偶发空 content，重试最多 3 次
        parsed = None
        last_content = ""
        for attempt in range(3):
            try:
                response = await provider.generate(
                    system=JUDGE_SYSTEM,
                    user=user,
                    model=None,
                    max_tokens=8192,
                    temperature=0.2,
                )
            except Exception as e:
                if attempt < 2:
                    await asyncio.sleep(1.5 * (attempt + 1))
                    continue
                case["judge_error"] = f"{type(e).__name__}: {e}"
                log.write(f"  [JUDGE ERROR] {case['dataset']}/{case['query_id']}: {case['judge_error']}\n")
                return case

            last_content = response.content
            parsed = _parse_json(response.content)
            if isinstance(parsed, list):
                break
            if attempt < 2:
                await asyncio.sleep(1.5 * (attempt + 1))

        if parsed is None:
            case["judge_error"] = f"parse_failed: {last_content[:120]}"
            log.write(f"  [JUDGE PARSE ERROR] {case['dataset']}/{case['query_id']}: {case['judge_error']}\n")
            return case

        # 对齐 id 到 chunks 索引，收集 LLM 判相关的 chunk
        llm_rel = []
        for item in parsed:
            if not isinstance(item, dict):
                continue
            idx = item.get("id", -1)
            if not isinstance(idx, int) or idx < 1 or idx > len(chunks):
                continue
            if int(item.get("related", 0)) == 1:
                llm_rel.append({
                    "chunk_id": chunks[idx - 1]["chunk_id"],
                    "qrels_status": chunks[idx - 1]["qrels_status"],
                    "reason": item.get("reason", ""),
                })

        case["llm_rel"] = llm_rel
        # 补全后: 命中数 +L, 相关集合 +L (L=LLM 判相关的 retrieved chunk 数)
        R = len(case["qrels_rel"])
        H = len(case["hits"])
        L = len(llm_rel)
        case["recall_new"] = (H + L) / (R + L) if (R + L) > 0 else 0.0
        return case


async def run_judge(cases, provider, log, limit) -> list[dict]:
    sem = asyncio.Semaphore(JUDGE_CONCURRENCY)
    selected = cases[:limit] if limit else cases
    tasks = [_judge_one(provider, case, sem, log) for case in selected]
    results = await asyncio.gather(*tasks)

    # 第二轮：并发下 flash 偶发空 content（单条正常），串行逐个重试补上
    retry_cases = [c for c in results
                   if "judge_error" in c and str(c["judge_error"]).startswith("parse_failed")]
    if retry_cases:
        log.write(f"  并发下 {len(retry_cases)} 条空 content，串行重试...\n")
        serial_sem = asyncio.Semaphore(1)
        for case in retry_cases:
            case.pop("judge_error", None)
            await _judge_one(provider, case, serial_sem, log)

    for case in results:
        if "judge_error" in case:
            log.write(f"  [{case['dataset']}/{case['query_id']}] judge 失败: {case['judge_error']}\n")
        else:
            log.write(f"  [{case['dataset']}/{case['query_id']}] "
                      f"recall {case['recall_old']:.2f} -> {case['recall_new']:.2f} "
                      f"(LLM 补 {len(case['llm_rel'])} 个相关 chunk)\n")
    await provider.close()
    return results


# ==============================================================================
# 汇总
# ==============================================================================
def _summarize(retrieval: dict, judged: list[dict], log) -> dict:
    all_recall_old = retrieval["all_recall_old"]
    n_full = retrieval["n_full"]
    total = len(all_recall_old)

    # 组装补全后的 recall 序列
    recall_new_by_qid: dict[str, float] = {}
    judge_failed = 0
    for c in judged:
        if "judge_error" in c:
            judge_failed += 1
            recall_new_by_qid[(c["dataset"], c["query_id"])] = c["recall_old"]
        else:
            recall_new_by_qid[(c["dataset"], c["query_id"])] = c["recall_new"]

    log.write(f"\n{'=' * 80}\n")
    log.write(f"  LLM 补全 qrels 漏标后 recall 对比\n")
    log.write(f"{'=' * 80}\n")

    old_mean = sum(all_recall_old) / total if total else 0.0
    zero_old = sum(1 for r in all_recall_old if r == 0.0)

    # 补全后的整体 recall：recall<1 的用 recall_new，recall=1 的仍是 1.0
    new_recalls = [recall_new_by_qid.get((c["dataset"], c["query_id"]), c["recall_old"])
                   for c in judged]
    new_mean = (sum(new_recalls) + n_full * 1.0) / total if total else 0.0
    zero_new = zero_old - sum(
        1 for c in judged
        if c.get("recall_old") == 0.0 and not c.get("judge_error") and c.get("recall_new", 0.0) > 0.0
    )

    log.write(f"  有标注 query 总数: {total}\n")
    log.write(f"  recall@10 均值:  补全前 {old_mean:.4f}  ->  补全后 {new_mean:.4f}  "
              f"(提升 {new_mean - old_mean:+.4f})\n")
    log.write(f"  zero (recall=0) 数:  补全前 {zero_old}  ->  补全后 {zero_new}\n")
    if judge_failed:
        log.write(f"  judge 失败条数: {judge_failed} (按原 recall 计)\n")

    # 分数据集: 只对 judge 过的 case 分桶（recall=1 的未按数据集拆，另行说明）
    log.write(f"\n  {'数据集':<20}{'judge条数':>9}{'原recall均值':>11}{'补全后均值':>11}\n")
    by_ds: dict[str, list] = {}
    for c in judged:
        by_ds.setdefault(c["dataset"], []).append(c)
    for ds in ALL_DATASETS:
        if ds not in by_ds:
            continue
        cs = by_ds[ds]
        old = sum(c["recall_old"] for c in cs) / len(cs)
        new = sum(recall_new_by_qid.get((c["dataset"], c["query_id"]), c["recall_old"]) for c in cs) / len(cs)
        log.write(f"  {ds:<20}{len(cs):>9}{old:>11.4f}{new:>11.4f}\n")

    # 漏标 chunk 归因
    status_counter = Counter()
    for c in judged:
        for v in c.get("llm_rel", []):
            status_counter[v.get("qrels_status", "?")] += 1
    log.write(f"\n  LLM 判相关的 chunk 归因:\n")
    log.write(f"    候选池外漏标(unlabeled)={status_counter.get('unlabeled', 0)}  "
              f"hard-negative标错={status_counter.get('hard_negative', 0)}\n")
    log.write(f"{'=' * 80}\n")

    summary = {
        "total": total,
        "recall_old_mean": old_mean,
        "recall_new_mean": new_mean,
        "zero_old": zero_old,
        "zero_new": zero_new,
        "judge_failed": judge_failed,
        "llm_rel_status": dict(status_counter),
    }
    return summary


# ==============================================================================
# main
# ==============================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=None, help="默认全部 4 数据集")
    parser.add_argument("--skip-judge", action="store_true", help="只跑检索出日志，不调 LLM")
    parser.add_argument("--limit", type=int, default=None, help="只 judge 前 N 条 (调试)")
    args = parser.parse_args()
    datasets = args.datasets or ALL_DATASETS

    ts = time.strftime("%Y%m%dT%H%M%S")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    log_path = RESULTS_DIR / f"judge_zero_recall_{ts}.log"
    json_path = RESULTS_DIR / f"judge_zero_recall_{ts}.json"
    log = TeeLogger(log_path)

    log.write(f"LLM-as-judge 补全 qrels 漏标，重算召回\n")
    log.write(f"Time: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    log.write(f"Datasets: {datasets}\n")
    log.write(f"judge 模型: deepseek-v4-flash, temperature=0.2\n")
    log.write(f"范围: recall<1 的 query 的未命中 chunk; 命中的跳过\n")
    log.write(f"Log: {log_path}\n\n")

    from rag_eval.query_analysis import build_retriever

    log.write("[INIT] 构建检索器...\n")
    retriever, expander = build_retriever()

    log.write("\n[PHASE 1] 跑检索，记录每条 query 召回与命中情况...\n")
    retrieval = asyncio.run(run_retrieval(datasets, retriever, expander, log))
    cases = retrieval["cases"]
    log.write(f"\n  待 judge {len(cases)} 条 recall<1 query\n")

    if args.skip_judge or not cases:
        log.write("\n[SKIP JUDGE] 仅检索阶段完成。\n")
        log.close()
        return

    # 构建 provider (deepseek-v4-flash)
    from server.llm.openai_compatible_provider import GenericOpenAIProvider
    from rag_eval.scripts.generate_eval_dataset import PROVIDER_CONFIGS

    provider = GenericOpenAIProvider(
        api_key=PROVIDER_CONFIGS["deepseek"]["api_key"],
        base_url=PROVIDER_CONFIGS["deepseek"]["base_url"],
        default_model="deepseek-v4-flash",
    )
    log.write(f"\n[PHASE 2] LLM judge (deepseek-v4-flash, 并发 {JUDGE_CONCURRENCY}, temperature=0.2)...\n")
    judged = asyncio.run(run_judge(cases, provider, log, args.limit))

    summary = _summarize(retrieval, judged, log)

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({
            "datasets": datasets,
            "timestamp": ts,
            "cases": judged,
            "summary": summary,
        }, f, ensure_ascii=False, indent=2)

    log.write(f"\nDone. Log: {log_path}\nJSON: {json_path}\n")
    log.close()


if __name__ == "__main__":
    main()
