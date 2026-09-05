# 临时脚本：解析 llm_calls.log 尾部记录（用后即删）
import json
import io
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

lines = io.open(r"logs/llm/llm_calls.log", encoding="utf-8").read().splitlines()
print("total_lines:", len(lines))
for l in lines[-16:]:
    if "{" not in l:
        continue
    s = l[l.index("{"):]
    try:
        j = json.loads(s)
        r = j.get("response") or ""
        head = r[:70].replace("\n", "\\n")
        print(j.get("call_id"), j.get("timestamp"), "node=" + str(j.get("node")),
              "purpose=" + str(j.get("purpose")), "lat=" + str(j.get("latency_ms")),
              "resp_len=" + str(len(r)), repr(head))
    except Exception as e:
        print("FAIL", str(e)[:60])
