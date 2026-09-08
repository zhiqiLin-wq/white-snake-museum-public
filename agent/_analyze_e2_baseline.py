"""临时分析：从 tool_calls.log 聚合 evolution_workbench filtered_retrieve 基线耗时。"""
import json
import collections
from datetime import datetime

traces = collections.defaultdict(list)
with open('logs/tool/tool_calls.log', encoding='utf-8') as f:
    for line in f:
        if 'filtered_retrieve' not in line:
            continue
        try:
            rec = json.loads(line[line.index('{'):])
        except Exception:
            continue
        if rec.get('caller') != 'evolution_workbench':
            continue
        ts = datetime.fromisoformat(rec['timestamp'])
        traces[rec['trace_id']].append((ts, rec.get('latency_ms', 0)))

rows = sorted(traces.items(), key=lambda kv: kv[1][-1][0])[-6:]
print("trace           检索次数  latency合计(s)  时间跨度(s)  平均单次(ms)")
for tid, evs in rows:
    evs.sort()
    span = (evs[-1][0] - evs[0][0]).total_seconds()
    tot = sum(l for _, l in evs) / 1000
    print(f"{tid[:12]:<16}{len(evs):<10}{tot:<16.1f}{span:<12.1f}{tot * 1000 / len(evs):<12.0f}")
