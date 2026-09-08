"""E2 矩阵扫描提速 — embedding 性能基准与正确性验证（测试先行）。

验证三个假设（改码前的基线证据）：
  A. 短查询（E2 关键词串 ~20-80 字）在 padding="max_length"(512) 下的
     forward 耗时，远高于 padding="longest"（动态长度）——量化浪费。
  B. 两种 padding 产出的向量 cosine ≈ 1.0（动态 padding 不改变结果，
     attention mask 保证填充 token 不影响 CLS 向量）——正确性门禁。
  C. asyncio.to_thread 多线程并行 embedding 的真实加速比，以及
     torch 线程数超订（OMP=8 × N 并发）对吞吐的影响——决定并行策略。

运行: python -m agent.server.rag.tests.bench_embedder_e2
（在 agent/ 目录下: python server/rag/tests/bench_embedder_e2.py）
"""
import asyncio
import os
import sys
import time
from pathlib import Path

# 允许直接脚本运行（agent/ 目录下含 server/ 包）
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import numpy as np

from server.rag.embedder import Embedder, BGE_QUERY_INSTRUCTION


# 典型 E2 查询：关键词串（短）、关键词+母题名+core（中）、语料块（长）
SHORT_Q = "雷峰塔 镇压 白娘子 永镇 西湖 塔下 祭塔 出世"
MEDIUM_Q = ("雷峰塔 镇压 白娘子 永镇西湖 塔下祭塔 出世团圆 法海 金钵 囚禁 "
            "镇压与囚禁 母题核心：白娘子被法海镇压于雷峰塔下，后世衍生祭塔、破塔情节")
LONG_Q = "话说南宋年间，临安府钱塘县有一官人，姓许名仙，字汉文，自幼父母双亡，在西湖边开药铺为生。" * 6


def _forward(model, tokenizer, texts, padding_mode):
    import torch
    enc = tokenizer(
        texts,
        padding=padding_mode,   # "max_length" = 现状(固定512) / True = 动态(批内最长)
        truncation=True,
        max_length=512,
        return_tensors="pt",
    )
    with torch.no_grad():
        out = model(**enc)
    cls = out.last_hidden_state[:, 0, :]
    return torch.nn.functional.normalize(cls, p=2, dim=1).numpy()


def _time_n(fn, n=5):
    fn()  # warmup
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    return (time.perf_counter() - t0) / n


async def main():
    print("=" * 72)
    print("E2 embedding 性能基准（CPU, bge-large-zh-v1.5）")
    print("=" * 72)

    emb = Embedder("BAAI/bge-large-zh-v1.5")
    emb.start_loading()
    for _ in range(120):
        if emb.is_ready:
            break
        await asyncio.sleep(1)
    if not emb.is_ready:
        print(f"模型加载失败: {emb.load_error}")
        sys.exit(1)
    print("模型就绪。\n")

    model, tokenizer = emb._model, emb._tokenizer

    # ---- A. 固定 512 padding vs 动态 padding 耗时 ----
    print("【A】单次 forward 耗时对比（秒/次，5 次平均）")
    print(f"{'查询':<8}{'长度(字)':<10}{'max_length=512':<18}{'longest 动态':<18}{'加速比':<8}")
    cases = [("短", SHORT_Q), ("中", MEDIUM_Q), ("长", LONG_Q)]
    vectors = {}
    for label, q in cases:
        qtext = BGE_QUERY_INSTRUCTION + q
        t_max = _time_n(lambda: _forward(model, tokenizer, [qtext], "max_length"))
        v_max = _forward(model, tokenizer, [qtext], "max_length")[0]
        t_long = _time_n(lambda: _forward(model, tokenizer, [qtext], True))
        v_long = _forward(model, tokenizer, [qtext], True)[0]
        vectors[label] = (v_max, v_long)
        print(f"{label:<8}{len(q):<10}{t_max:<18.4f}{t_long:<18.4f}{t_max / max(t_long, 1e-9):<8.2f}x")

    # ---- B. 正确性：两种 padding 向量一致性 ----
    print("\n【B】向量一致性（cosine，门禁 >= 0.9999）")
    ok = True
    for label, (v_max, v_long) in vectors.items():
        cos = float(np.dot(v_max, v_long) /
                   (np.linalg.norm(v_max) * np.linalg.norm(v_long) + 1e-9))
        flag = "OK" if cos >= 0.9999 else "FAIL"
        if cos < 0.9999:
            ok = False
        print(f"  {label}查询: cosine={cos:.6f}  [{flag}]")
    # 批内 padding 一致性：长短混批 vs 单条动态
    v_mix = _forward(model, tokenizer,
                     [BGE_QUERY_INSTRUCTION + SHORT_Q, BGE_QUERY_INSTRUCTION + LONG_Q], True)
    cos_mix = float(np.dot(v_mix[0], vectors["短"][1]) /
                    (np.linalg.norm(v_mix[0]) * np.linalg.norm(vectors["短"][1]) + 1e-9))
    print(f"  混批短查询 vs 单条: cosine={cos_mix:.6f}  [{'OK' if cos_mix >= 0.9999 else 'FAIL'}]")
    if cos_mix < 0.9999:
        ok = False

    # ---- C. 并发吞吐：顺序 vs to_thread 并行，不同 torch 线程数 ----
    print("\n【C】并发吞吐（8 个短查询，模拟 E2 8 路单元格）")
    import torch
    queries = [BGE_QUERY_INSTRUCTION + f"{SHORT_Q} {i} 唐" for i in range(8)]

    async def run_concurrent(n_workers, torch_threads):
        torch.set_num_threads(torch_threads)
        sem = asyncio.Semaphore(n_workers)

        async def one(qtext):
            async with sem:
                return await asyncio.to_thread(
                    lambda: _forward(model, tokenizer, [qtext], True))
        # warmup
        await asyncio.gather(*[one(q) for q in queries[:2]])
        t0 = time.perf_counter()
        await asyncio.gather(*[one(q) for q in queries])
        return time.perf_counter() - t0

    seq_t = await run_concurrent(1, 8)
    print(f"  顺序(1 并发, torch=8): {seq_t:.3f}s  (基线 1.0x)")
    for nw, tt in [(8, 8), (8, 4), (8, 2), (4, 4), (6, 2)]:
        t = await run_concurrent(nw, tt)
        print(f"  并行({nw} 并发, torch={tt}): {t:.3f}s  ({seq_t / t:.2f}x)")

    print("\n" + ("=" * 72))
    print("结论: 正确性", "全部通过 ✅" if ok else "存在 FAIL ❌")
    print("=" * 72)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    asyncio.run(main())
