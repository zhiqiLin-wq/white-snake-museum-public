"""L2 提取质量评测：提取 P/R/F1 与分类准确率。

对每条用户陈述跑 MemoryExtractor.extract（真实 LLM），用 bge 向量相似度判定
提取结果是否语义匹配金标准事实，并比对 memory_type 分类。
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


class RecordingStore:
    """记录 add_memory 的调用，拿到提取的 statement 与 memory_type。"""

    def __init__(self):
        self.memories = []

    def check_duplicate_md5(self, c):
        return None

    def check_duplicate_vector(self, c):
        return False

    def add_memory(self, c, memory_type=None, **kw):
        self.memories.append((c, memory_type))
        return "x"


def cosine(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(x * x for x in b) ** 0.5
    return dot / (na * nb) if na > 0 and nb > 0 else 0.0


async def main():
    data = read_json("d2_extraction.json")
    facts = data["facts"]

    print("[1/3] 加载 embedder...")
    embedder = Embedder(model_name=rag_config.embedding_model,
                        local_path=rag_config.embedding_local_path or None)
    embedder.start_loading()
    for _ in range(60):
        if embedder.is_ready:
            break
        await asyncio.sleep(1)

    llm = DeepSeekProvider()

    print("[2/3] 跑提取...\n")
    recall = 0
    type_correct = 0
    type_total = 0
    precision_hits = 0
    precision_total = 0
    for fact in facts:
        store = RecordingStore()
        extractor = MemoryExtractor(llm=llm, recall_store=store, model=MODEL)
        await extractor.extract({"role": "user", "content": fact["statement"]})

        gv = embedder.embed([fact["gold_fact"]])[0]
        matched = None
        fact_hits = 0
        for s, mt in store.memories:
            sv = embedder.embed([s])[0]
            if cosine(gv, sv) >= SIM_THRESHOLD:
                fact_hits += 1
                if matched is None:
                    matched = (s, mt)
        precision_total += len(store.memories)
        precision_hits += fact_hits
        if matched:
            recall += 1
            type_total += 1
            if matched[1] == fact["memory_type"]:
                type_correct += 1
                print(f"  [正确] {fact['memory_type']:9s} {fact['gold_fact'][:24]}...")
            else:
                print(f"  [错类] 期望 {fact['memory_type']:9s} 实得 {matched[1]:9s} {fact['gold_fact'][:24]}...")
        else:
            print(f"  [漏提] {fact['memory_type']:9s} {fact['gold_fact'][:24]}...")
        await asyncio.sleep(0.3)

    total = len(facts)
    rec = recall / total
    acc = type_correct / type_total if type_total else 0.0
    prec = precision_hits / precision_total if precision_total else 0.0
    print(f"\n===== L2 提取质量 =====")
    print(f"金标准事实: {total} 条")
    print(f"提取召回率: {recall}/{total} = {rec:.4f}")
    print(f"分类准确率: {type_correct}/{type_total} = {acc:.4f}")
    print(f"提取精确度: {precision_hits}/{precision_total} = {prec:.4f}（输出里真该提的占比，越低=乱提越多）")


if __name__ == "__main__":
    asyncio.run(main())
