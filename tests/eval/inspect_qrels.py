"""抽检 D1 标注质量：输出查询与其相关记忆原文，供人工判断强/弱相关。

用法: python inspect_qrels.py [N]   # N 为输出的查询条数，默认 30
"""

import sys
from pathlib import Path

_eval_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(_eval_dir))

from common import read_json


def main():
    data = read_json("d1_retrieval.json")
    memories = {m["id"]: m for m in data["memories"]}
    queries = data["queries"]
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    for q in queries[:n]:
        print(f"\n[Q] {q['query']}  (difficulty={q['difficulty']})")
        for rid in q["relevant_ids"]:
            m = memories.get(rid)
            if m:
                print(f"   - [{rid}|{m.get('memory_type','')}] {m['content']}")


if __name__ == "__main__":
    main()
