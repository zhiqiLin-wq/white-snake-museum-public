"""D6 失分诊断：定位「观点/约束类」偏好的失分根因。

对指定偏好的每个 variant 跑真实 LLM 提取，打印提取结果与余弦相似度，
区分「漏提（判 NO/空）」vs「提了但措辞差异大（余弦低）」。

用法: python diagnose_d6.py [p06,p07,p10]   # 默认诊断这三个失分偏好
"""

import asyncio
import sys
from pathlib import Path

_eval_dir = Path(__file__).resolve().parent
_project_root = _eval_dir.parent.parent
sys.path.insert(0, str(_project_root))
sys.path.insert(0, str(_eval_dir))

from common import read_json
from evaluate_d6 import cosine, normalize, strip_type_prefix, SIM_THRESHOLD
from agent.server.llm.deepseek_provider import DeepSeekProvider
from agent.server.context.memory_extractor import MemoryExtractor
from agent.server.config import settings
from agent.server.rag.config import rag_config
from agent.server.rag.embedder import Embedder

MODEL = settings.memory_extraction_model or settings.effective_fast_model


async def main():
    target = set(sys.argv[1].split(",")) if len(sys.argv) > 1 else {"p06", "p07", "p10"}
    data = read_json("d6_consistency.json")
    prefs = [p for p in data["preferences"] if p["id"] in target]

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

    print("[2/2] 诊断提取...\n")
    for p in prefs:
        pref_norm = normalize(p["preference"])
        pref_vec = embedder.embed([pref_norm])[0]
        print(f"===== {p['id']} 偏好: {p['preference']}  (归一化: {pref_norm})")
        for v in p["variants"]:
            raw = await extractor._llm_extract(v)
            print(f"\n  variant: {v}")
            print(f"  raw: {raw!r}")
            if not raw or raw.strip().upper() == "NO":
                print("    -> [漏提] 返回空或 NO")
                await asyncio.sleep(0.3)
                continue
            for s in raw.split("\n"):
                s = s.strip()
                if not s or s.upper() == "NO":
                    continue
                s_norm = normalize(strip_type_prefix(s))
                s_vec = embedder.embed([s_norm])[0]
                sim = cosine(pref_vec, s_vec)
                flag = "PASS" if sim >= SIM_THRESHOLD else "FAIL"
                print(f"    -> [{flag}] {s}  (归一化: {s_norm}, sim={sim:.4f})")
            await asyncio.sleep(0.3)
        print()


if __name__ == "__main__":
    asyncio.run(main())
