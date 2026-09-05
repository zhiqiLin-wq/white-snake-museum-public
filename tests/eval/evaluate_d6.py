"""L6 一致性评测：记忆提取对同一偏好的稳定性。

对每条偏好的 3 种措辞，跑 LLM 提取，用 bge 向量相似度判定提取结果
是否与偏好语义一致（绕开不稳定的 LLM judge）。
一致性 = 各偏好被成功提取的措辞数分布。
"""

import asyncio
import sys
from pathlib import Path

_eval_dir = Path(__file__).resolve().parent
_project_root = _eval_dir.parent.parent
sys.path.insert(0, str(_project_root))
sys.path.insert(0, str(_eval_dir))

from common import read_json
from agent.server.llm.deepseek_provider import DeepSeekProvider
from agent.server.context.memory_extractor import MemoryExtractor
from agent.server.config import settings
from agent.server.rag.config import rag_config
from agent.server.rag.embedder import Embedder

MODEL = settings.memory_extraction_model or settings.effective_fast_model
SIM_THRESHOLD = 0.72


def cosine(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(x * x for x in b) ** 0.5
    return dot / (na * nb) if na > 0 and nb > 0 else 0.0


# 否定归一化：统一否定词 + 剥离格式前缀，让否定式偏好的向量判定更公平。
# 例如「用户约束：不要发送长篇回复」与「不喜欢收到长篇回复」归一化后核心词对齐。
_NEG_PREFIXES = ("用户约束：", "用户约束:", "用户要求：", "用户要求:")
_NEG_WORDS = ["不要", "不许", "禁止", "拒绝", "不想", "受不了", "懒得", "避免", "讨厌"]
_NEG_TARGET = "不喜欢"


def neg_normalize(text: str) -> str:
    t = text
    for p in _NEG_PREFIXES:
        if t.startswith(p):
            t = t[len(p):]
            break
    for w in _NEG_WORDS:
        t = t.replace(w, _NEG_TARGET)
    return t


# 同义归一化：近义表述统一成规范词，让语义等价但措辞不同的判定更公平
_SYNONYM_MAP = [
    ("摘录", "引用"),
    ("原话", "原文"),
    ("写清", "标注"),
    ("标明", "标注"),
    ("注明", "标注"),
    ("来源", "出处"),
]


def normalize(text: str) -> str:
    """否定归一化 + 同义归一化。"""
    t = neg_normalize(text)
    for src, dst in _SYNONYM_MAP:
        t = t.replace(src, dst)
    return t


_VALID_TYPES = ("Entity", "Archival", "System", "Working", "Summary")


def strip_type_prefix(s: str) -> str:
    """剥离"类型：内容"前缀（LLM 直接分类格式），返回内容部分。"""
    for t in _VALID_TYPES:
        for colon in ("：", ":"):
            prefix = t + colon
            if s.startswith(prefix):
                return s[len(prefix):].strip()
    return s


async def main():
    data = read_json("d6_consistency.json")
    prefs = data["preferences"]

    print("[1/2] 加载 embedder...")
    embedder = Embedder(model_name=rag_config.embedding_model,
                        local_path=rag_config.embedding_local_path or None)
    embedder.start_loading()
    for _ in range(60):
        if embedder.is_ready:
            break
        await asyncio.sleep(1)

    llm = DeepSeekProvider()
    extractor = MemoryExtractor(llm=llm, recall_store=None, model=MODEL)

    print("[2/2] 评测提取一致性...\n")
    dist = {3: 0, 2: 0, 1: 0, 0: 0}
    for p in prefs:
        pref_vec = embedder.embed([normalize(p["preference"])])[0]
        success = 0
        for v in p["variants"]:
            raw = await extractor._llm_extract(v)
            statements = [s.strip() for s in (raw or "").split("\n")
                          if s.strip() and s.strip().upper() != "NO"]
            matched = False
            for s in statements:
                s_vec = embedder.embed([normalize(strip_type_prefix(s))])[0]
                if cosine(pref_vec, s_vec) >= SIM_THRESHOLD:
                    matched = True
                    break
            if matched:
                success += 1
            await asyncio.sleep(0.3)
        dist[success] += 1
        print(f"  [{p['id']}] {p['preference'][:24]}... 成功 {success}/3")

    total = len(prefs)
    avg = sum(k * v for k, v in dist.items()) / (total * 3)
    print("\n===== L6 一致性 =====")
    print(f"总偏好数: {total}")
    print(f"成功分布: 3/3={dist[3]}  2/3={dist[2]}  1/3={dist[1]}  0/3={dist[0]}")
    print(f"平均提取成功率（一致性得分）: {avg:.4f}")
    print(f"完全一致（3/3 措辞都提取成功）的偏好占比: {dist[3]/total:.4f}")


if __name__ == "__main__":
    asyncio.run(main())
