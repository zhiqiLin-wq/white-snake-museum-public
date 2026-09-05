"""补 judge 空 content 失败 + 生成增强版 qrels。

1. 对 judge_zero_recall 结果里 judge_error(parse_failed) 的 case 串行重试，
   用更长重试窗口补上空 content（flash 并发偶发，串行单条正常）。
2. 把 LLM 判相关但 qrels 没标的 chunk 追加为 rel=1，生成 qrels_<ds>_augmented.tsv
   （原 qrels 不动，另存一份）。

Usage:
    python rag_eval/backfill_augment.py
"""
import asyncio
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
AGENT_DIR = PROJECT_ROOT / "agent"
for p in [str(PROJECT_ROOT), str(AGENT_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from rag_eval.judge_zero_recall import JUDGE_SYSTEM, _build_judge_user_prompt, _parse_json
from server.llm.openai_compatible_provider import GenericOpenAIProvider
from rag_eval.scripts.generate_eval_dataset import PROVIDER_CONFIGS

GT_DIR = Path(__file__).resolve().parent / "ground_truth"
RESULTS_DIR = Path(__file__).resolve().parent / "results"

# 最终有效的 4 个结果 JSON（排除废弃的旧跑）
FINAL_JSONS = {
    "academic_qwen": "judge_zero_recall_20260814T114004.json",
    "academic_deepseek": "judge_zero_recall_20260814T114947.json",
    "daily_deepseek": "judge_zero_recall_20260814T120538.json",
    "daily_qwen": "judge_zero_recall_20260814T121935.json",
}

DATASETS = ["academic_qwen", "academic_deepseek", "daily_deepseek", "daily_qwen"]


# ==============================================================================
# 第一步: 补 judge 失败
# ==============================================================================
async def _backfill_one(provider, case: dict) -> bool:
    """串行重试一条失败 case，用更长重试窗口（flash 空 content 可能持续更久）。

    Returns:
        True=补上, False=仍失败
    """
    chunks = case["unhit_chunks"]
    if not chunks:
        return True  # 无未命中 chunk，无需 judge

    user = _build_judge_user_prompt(case["query_text"], chunks)
    for attempt in range(6):
        try:
            response = await provider.generate(
                system=JUDGE_SYSTEM,
                user=user,
                model=None,
                max_tokens=8192,
                temperature=0.2,
            )
        except Exception:
            if attempt < 5:
                await asyncio.sleep(2 * (attempt + 1))
                continue
            return False

        parsed = _parse_json(response.content)
        if isinstance(parsed, list):
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
            R = len(case["qrels_rel"])
            H = len(case["hits"])
            L = len(llm_rel)
            case["recall_new"] = (H + L) / (R + L) if (R + L) > 0 else 0.0
            case.pop("judge_error", None)
            return True
        if attempt < 5:
            await asyncio.sleep(2 * (attempt + 1))
    return False


async def backfill(provider):
    """对 4 个 JSON 里的失败 case 串行重试，写回 JSON。"""
    total_failed = 0
    total_recovered = 0
    for ds, fname in FINAL_JSONS.items():
        path = RESULTS_DIR / fname
        data = json.load(open(path, encoding="utf-8"))
        failed = [c for c in data["cases"] if "judge_error" in c]
        total_failed += len(failed)
        print(f"[{ds}] 失败 {len(failed)} 条，串行重试...")
        for case in failed:
            ok = await _backfill_one(provider, case)
            if ok:
                total_recovered += 1
            else:
                print(f"  [仍失败] {ds}/{case['query_id']}")
        json.dump(data, open(path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
        print(f"  [{ds}] 写回完成")
    print(f"\n[补失败] 共 {total_failed} 条失败，补上 {total_recovered} 条，"
          f"仍失败 {total_failed - total_recovered} 条")
    return total_failed, total_recovered


# ==============================================================================
# 第二步: 生成增强版 qrels
# ==============================================================================
def _load_qrels_tsv(path: Path) -> tuple[list[str], dict[str, dict[str, int]]]:
    """读 qrels tsv，返回 (注释行列表, {query_id: {doc_id: relevance}})。

    保留 rel=0（hard negative），跳过重复。
    """
    header: list[str] = []
    qrels: dict[str, dict[str, int]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            header.append("")
            continue
        if line.startswith("#"):
            header.append(line)
            continue
        parts = line.split("\t")
        if len(parts) < 3:
            continue
        qid, did, rel = parts[0], parts[1], int(parts[2])
        qrels.setdefault(qid, {})[did] = rel
    return header, qrels


def augment_qrels():
    """从补完的 JSON 收集 llm_rel，追加为 rel=1，写 qrels_<ds>_augmented.tsv。"""
    for ds in DATASETS:
        fname = FINAL_JSONS[ds]
        data = json.load(open(RESULTS_DIR / fname, encoding="utf-8"))
        header, qrels = _load_qrels_tsv(GT_DIR / f"qrels_{ds}.tsv")

        n_added = 0
        for case in data["cases"]:
            if case["dataset"] != ds:
                continue
            qid = case["query_id"]
            for v in case.get("llm_rel", []):
                did = v["chunk_id"]
                # 只有 unlabeled（原 qrels 里没有）或 hard_negative（原 rel=0）才算新增
                old = qrels.get(qid, {}).get(did)
                if old is None or old == 0:
                    qrels.setdefault(qid, {})[did] = 1
                    n_added += 1

        # 写增强版：注释头 + rel>=1 的行（评估只用 rel>=1，跳过 rel=0 负例）
        out_path = GT_DIR / f"qrels_{ds}_augmented.tsv"
        lines = [
            f"# qrels (augmented) — 原 qrels_{ds}.tsv + LLM judge 补全漏标 chunk (rel=1)",
            "# 格式: query_id\tdoc_id\trelevance",
            "# 新增行来源: judge_zero_recall 判相关但原 qrels 未标(unlabeled)或标0(hard_negative)的 chunk",
            "",
        ]
        for qid in sorted(qrels):
            for did, rel in sorted(qrels[qid].items()):
                if rel >= 1:
                    lines.append(f"{qid}\t{did}\t{rel}")
        out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"[{ds}] 新增 {n_added} 条漏标 chunk -> {out_path.name}")


def main():
    augment_only = "--augment-only" in sys.argv

    if not augment_only:
        provider = GenericOpenAIProvider(
            api_key=PROVIDER_CONFIGS["deepseek"]["api_key"],
            base_url=PROVIDER_CONFIGS["deepseek"]["base_url"],
            default_model="deepseek-v4-flash",
        )
        print("[1/2] 补 judge 失败...")
        asyncio.run(backfill(provider))
        # 不显式 close（跨 event loop 会 RuntimeError），进程退出自动清理

    print("\n[2/2] 生成增强版 qrels...")
    augment_qrels()


if __name__ == "__main__":
    main()
