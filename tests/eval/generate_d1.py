"""D1 检索评测集生成器（连真实 DeepSeek LLM）。

流程：
1. 生成器分批生成记忆库（Entity/Archival/System/Working/Summary，共 200 条）
2. bge 语义去重
3. 为 120 条锚点记忆生成三档查询（同词/语义改写/跨对话）+ 相关性标注
4. 校验器复核标注，对抗器挑战漏标
5. 输出 tests/eval/data/d1_retrieval.json

使用真实 LLM（deepseek-v4-flash），运行约需几分钟。
"""

import asyncio
import json
import re
import sys
from pathlib import Path

_eval_dir = Path(__file__).resolve().parent
_project_root = _eval_dir.parent.parent
sys.path.insert(0, str(_project_root))
sys.path.insert(0, str(_eval_dir))

from common import write_json
from agent.server.llm.deepseek_provider import DeepSeekProvider
from agent.server.config import settings

MODEL = settings.memory_extraction_model or settings.effective_fast_model

# 项目真实的白蛇传文献语料（从 excel_data/白蛇传文献选集.txt 抽取的结构信息）
DOMAIN_CONTEXT = (
    "白蛇传历代文献版本（真实语料，共 7 版）：\n"
    "1. 唐·《太平广记》卷四五八《李黄》：文言小说，李黄于长安东市遇白衣女子（白蛇所化），归家后化水而亡。\n"
    "2. 宋·《西湖三塔记》：话本小说，奚宣赞于西湖遇白蛇、乌鸡、獭三怪（化白衣妇人、卯奴、婆婆），奚真人在四圣观降妖，镇于西湖三石塔。\n"
    "3. 明·冯梦龙《白娘子永镇雷峰塔》（《警世通言》）：拟话本，白娘子被法海镇压于雷峰塔。\n"
    "4. 清乾隆·方成培《雷峰塔传奇》：戏曲，白蛇传定型为爱情悲剧。\n"
    "5. 清·玉山堂主人《雷峰塔传奇》：白话神魔小说。\n"
    "6. 清·陈遇乾《义妖传》：苏州弹词。\n"
    "7. 现当代·田汉《白蛇传》：京剧。\n"
    "核心人物：白娘子（白素贞）、许仙（许宣）、法海、小青、奚宣赞、卯奴、奚真人、李黄。\n"
    "核心地点：西湖、雷峰塔、金山寺、钱塘门、涌金门、昭庆寺、四圣观、龙虎山、长安东市。\n"
    "核心母题：白蛇化人、人妖恋、镇塔、三塔镇妖、水漫金山、母题流变、版本演变。"
)

# 类型 -> (条数, importance 采样区间, 多样性维度引导)
TYPE_CONFIG = {
    "Entity": (100, (0.5, 0.9), "对特定版本、人物、母题、地点的研究偏好与观点"),
    "Archival": (90, (0.4, 0.7), "7 个版本各自的真实情节、人物、地点、体裁特征"),
    "System": (40, (0.9, 0.95), "输出格式、内容边界、交互方式等不同约束"),
    "Working": (15, (0.2, 0.3), "不同的临时研究状态"),
    "Summary": (15, (0.2, 0.3), "不同对话的摘要要点"),
}

# memory_type 语义定义 + few-shot 示例（避免标签与内容错配）
TYPE_SEMANTICS = {
    "Entity": ("学者的身份、偏好、主观观点",
               "该学者最关注《西湖三塔记》中三怪母题的早期形态，认为它是白蛇传故事的重要源头。"),
    "Archival": ("白蛇传文献的客观事实（版本情节、人物、地点、年代）",
                 "《西湖三塔记》中，奚真人将白蛇、乌鸡、獭三怪镇压于西湖三座石塔之下。"),
    "System": ("学者的明确约束或要求",
               "该学者要求引用文献原文时必须标注出处，不得用现代改编版替代原始文献。"),
    "Working": ("学者当前正在进行的临时研究活动",
                "该学者正在比对《西湖三塔记》与《白娘子永镇雷峰塔》中法海形象的差异。"),
    "Summary": ("对研究结论或分析的概括总结",
                "综上，该学者认为白蛇传母题经历了从妖祸主题到爱情悲剧的演变。"),
}

N_QUERIES = 120
BATCH = 20
QUERY_BATCH = 8

GENERATOR_SYSTEM = (
    "你是一个严谨的检索评测数据集生成器。你只输出合法的 JSON，不输出任何解释性文字。"
)

MEMORY_USER = (
    "请基于下面的白蛇传真实文献语料，生成 {n} 条关于一位研究白蛇传的学者的长期记忆。\n"
    "记忆内容必须贴合真实文献（引用真实的版本名、人物、地点、情节、母题），不得凭空编造版本或情节。\n"
    "本次只生成 memory_type={mtype} 的记忆，该类型的含义是：{semantics}\n"
    "格式示例（内容需替换为真实文献内容，且必须符合上述类型含义）：{example}\n"
    "真实文献语料：\n{domain}\n\n"
    "要求：\n"
    "1. 每条记忆是一句独立、具体、语义不重复的中文事实，第三人称。\n"
    "2. 每条记忆的 memory_type 都必须是 {mtype}，内容必须符合该类型的含义，不要生成其他类型的内容。\n"
    "3. importance 在 {imp_lo}~{imp_hi} 之间取一位小数。\n"
    "4. 内容不要出现序号，不要雷同。\n"
    "5. 尽量覆盖不同维度：{dims}。\n"
    "只输出 JSON 数组，元素格式："
    '{{"content": "...", "memory_type": "{mtype}", "importance": 0.8}}'
)

QUERY_USER = (
    "下面是 {n} 条记忆（带 id）。请为每条记忆生成一条能检索到它的查询，并标注相关记忆 id。\n"
    "难度分布：前 1/3 用「同词」（直接包含记忆关键词），中 1/3 用「语义改写」（同义换说法，不直接包含关键词），"
    "后 1/3 用「跨对话」（不同场景、更间接的措辞）。\n"
    "要求：每条查询是自然的中文问句；relevant_ids 包含该锚点记忆自身 id 以及语义真正相关的其他记忆 id（1~3 个）。\n"
    "注意：difficulty 只能是 exact、semantic、cross 三个值之一，不得自造其他标签。\n"
    "记忆列表：\n{memories}\n"
    "只输出 JSON 数组，元素格式（anchor_id 必须填对应记忆的 id）："
    '{{"anchor_id": "m001", "query": "...", "difficulty": "exact", "relevant_ids": ["m001"]}}'
)

VERIFY_USER = (
    "请复核下面的检索标注。对每条查询，找出其 relevant_ids 中与该查询语义不相关的错误 id。\n"
    "标注列表：\n{annotations}\n"
    "只输出 JSON 数组，每个元素形如 {{\"query_index\": 0, \"wrong_ids\": [\"m005\"]}}，"
    "没有错误时 wrong_ids 为空数组。"
)

ADVERSARY_USER = (
    "请从「反方向」检查：下面每个查询与其 relevant_ids 之外，是否还有语义确实相关、但被漏标的记忆 id。\n"
    "查询与候选记忆：\n{data}\n"
    "只输出 JSON 数组，每个元素形如 {{\"query_index\": 0, \"missed_ids\": []}}，没有漏标时 missed_ids 为空数组。"
)


def extract_json(text: str):
    """从 LLM 输出中健壮地提取 JSON（容忍 ```json 代码块和前后缀）。"""
    if text is None:
        return None
    text = text.strip()
    # 去掉 ```json ... ``` 包裹
    m = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    if m:
        text = m.group(1).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # 提取第一个 [ 到最后一个 ] 或 { 到最后一个 }
    for open_c, close_c in (("[", "]"), ("{", "}")):
        lo = text.find(open_c)
        hi = text.rfind(close_c)
        if lo != -1 and hi > lo:
            try:
                return json.loads(text[lo:hi + 1])
            except json.JSONDecodeError:
                continue
    return None


async def llm_json(llm, user: str, retries: int = 5):
    """调 LLM 并解析 JSON，失败重试（间隔等待避免限流）。"""
    for attempt in range(retries):
        try:
            resp = await llm.generate(
                system=GENERATOR_SYSTEM, user=user, model=MODEL,
                max_tokens=4096, temperature=0.7,
            )
            data = extract_json(resp.content)
            if data is not None:
                return data
        except Exception as e:
            print(f"  [重试 {attempt + 1}/{retries}] LLM 调用失败: {e}")
        if attempt < retries - 1:
            await asyncio.sleep(3)
    return None


async def generate_memories(llm) -> list[dict]:
    """分批生成记忆库。"""
    memories = []
    for mtype, (count, (imp_lo, imp_hi), dims) in TYPE_CONFIG.items():
        remaining = count
        while remaining > 0:
            n = min(BATCH, remaining)
            semantics, example = TYPE_SEMANTICS[mtype]
            user = MEMORY_USER.format(n=n, mtype=mtype, imp_lo=imp_lo, imp_hi=imp_hi,
                                      dims=dims, domain=DOMAIN_CONTEXT,
                                      semantics=semantics, example=example)
            data = None
            for _ in range(5):
                data = await llm_json(llm, user)
                if isinstance(data, list):
                    break
                await asyncio.sleep(5)
            if not isinstance(data, list):
                print(f"  [警告] {mtype} 批次连续失败，跳过该批 {n} 条")
                remaining -= n
                continue
            for item in data:
                if not isinstance(item, dict):
                    continue
                content = str(item.get("content", "")).strip()
                if not content or len(content) < 6:
                    continue
                mt = item.get("memory_type", mtype)
                try:
                    imp = float(item.get("importance", imp_lo))
                except (TypeError, ValueError):
                    imp = imp_lo
                memories.append({
                    "id": f"m{len(memories) + 1:03d}",
                    "content": content,
                    "memory_type": mt,
                    "importance": round(imp, 2),
                })
            remaining -= n
            print(f"  [{mtype}] 已生成 {count - remaining}/{count}")
            await asyncio.sleep(2)
    return memories


def semantic_dedup(memories: list[dict], embedder, threshold: float = 0.88) -> list[dict]:
    """用 bge 向量做语义去重（相似度 > threshold 视为重复，保留前者）。"""
    if not memories or embedder is None:
        return memories
    texts = [m["content"] for m in memories]
    embs = embedder.embed(texts)
    keep = []
    kept_embs = []
    for m, emb in zip(memories, embs):
        dup = False
        for ke in kept_embs:
            dot = sum(a * b for a, b in zip(emb, ke))
            na = sum(a * a for a in emb) ** 0.5
            nb = sum(b * b for b in ke) ** 0.5
            if na > 0 and nb > 0 and dot / (na * nb) > threshold:
                dup = True
                break
        if not dup:
            keep.append(m)
            kept_embs.append(emb)
    removed = len(memories) - len(keep)
    if removed:
        print(f"  [去重] 移除 {removed} 条近重复记忆")
    return keep


def _sample_anchors(memories: list[dict], n: int) -> list[dict]:
    """按类型 round-robin 均匀抽样锚点，保证各类型覆盖。"""
    from collections import defaultdict
    by_type = defaultdict(list)
    for m in memories:
        by_type[m["memory_type"]].append(m)
    types = list(by_type.keys())
    anchors = []
    idx = {t: 0 for t in types}
    while len(anchors) < n and any(idx[t] < len(by_type[t]) for t in types):
        progressed = False
        for t in types:
            if idx[t] < len(by_type[t]):
                anchors.append(by_type[t][idx[t]])
                idx[t] += 1
                progressed = True
                if len(anchors) >= n:
                    break
        if not progressed:
            break
    return anchors


async def generate_queries(llm, memories: list[dict]) -> list[dict]:
    """为锚点记忆生成三档查询 + 相关性标注（按 anchor_id 对应）。"""
    anchors = _sample_anchors(memories, N_QUERIES)
    queries = []
    for start in range(0, len(anchors), QUERY_BATCH):
        batch = anchors[start:start + QUERY_BATCH]
        mem_lines = "\n".join(
            f'- id={m["id"]} type={m["memory_type"]} content={m["content"]}' for m in batch
        )
        user = QUERY_USER.format(n=len(batch), memories=mem_lines)
        data = None
        for _ in range(5):
            data = await llm_json(llm, user)
            if isinstance(data, list):
                break
            await asyncio.sleep(5)
        if not isinstance(data, list):
            print(f"  [警告] 查询批次 {start} 连续失败")
            continue
        anchor_by_id = {m["id"]: m for m in batch}
        for item in data:
            if not isinstance(item, dict):
                continue
            q = str(item.get("query", "")).strip()
            if not q:
                continue
            anchor_id = str(item.get("anchor_id", ""))
            if anchor_id not in anchor_by_id:
                continue
            rel = item.get("relevant_ids", [])
            if not isinstance(rel, list):
                rel = []
            rel = [str(r) for r in rel if str(r).startswith("m")]
            if anchor_id not in rel:
                rel.insert(0, anchor_id)
            diff = str(item.get("difficulty", "semantic")).strip().lower()
            if "cross" in diff:
                diff = "cross"
            elif diff not in ("exact", "semantic"):
                diff = "semantic"
            queries.append({
                "id": f"q{len(queries) + 1:03d}",
                "query": q,
                "difficulty": diff,
                "relevant_ids": rel[:3],
            })
        print(f"  [查询] 已生成 {len(queries)}/{N_QUERIES}")
        await asyncio.sleep(2)
    return queries


async def verify_annotations(llm, queries: list[dict]) -> list[dict]:
    """校验器：删除标注错误的 relevant_ids。"""
    for start in range(0, len(queries), BATCH):
        batch = queries[start:start + BATCH]
        ann = json.dumps(batch, ensure_ascii=False)
        user = VERIFY_USER.format(annotations=ann)
        data = await llm_json(llm, user)
        if not isinstance(data, list):
            continue
        for item in data:
            if not isinstance(item, dict):
                continue
            idx = item.get("query_index")
            if idx is None or idx >= len(batch):
                continue
            wrong = item.get("wrong_ids", [])
            if isinstance(wrong, list):
                drop = {str(f) for f in wrong}
                batch[idx]["relevant_ids"] = [r for r in batch[idx]["relevant_ids"] if r not in drop]
        print(f"  [校验] 已复核 {min(start + BATCH, len(queries))}/{len(queries)}")
    return queries


async def adversarial_check(llm, memories: list[dict], queries: list[dict]) -> list[dict]:
    """对抗器：补回被漏标的 relevant_ids。"""
    mem_by_id = {m["id"]: m for m in memories}
    for start in range(0, len(queries), BATCH):
        batch = queries[start:start + BATCH]
        # 候选 = 该查询锚点附近的记忆（同类型优先），简化：全部记忆
        lines = []
        for q in batch:
            for mid in q["relevant_ids"]:
                if mid in mem_by_id:
                    lines.append(f'query="{q["query"]}" relevant={mid}')
        # 简化：让 LLM 直接对每个查询给出 missed ids（候选全库）
        data_payload = [{
            "query_index": i,
            "query": q["query"],
            "relevant_ids": q["relevant_ids"],
            "candidates": [m["id"] + ":" + m["content"] for m in memories[:40]],
        } for i, q in enumerate(batch)]
        user = ADVERSARY_USER.format(data=json.dumps(data_payload, ensure_ascii=False))
        data = await llm_json(llm, user)
        if not isinstance(data, list):
            continue
        for item in data:
            if not isinstance(item, dict):
                continue
            idx = item.get("query_index")
            if idx is None or idx >= len(batch):
                continue
            missed = item.get("missed_ids", [])
            if isinstance(missed, list):
                for mid in missed:
                    mid = str(mid)
                    if mid in mem_by_id and mid not in batch[idx]["relevant_ids"]:
                        if len(batch[idx]["relevant_ids"]) < 3:
                            batch[idx]["relevant_ids"].append(mid)
        print(f"  [对抗] 已挑战 {min(start + BATCH, len(queries))}/{len(queries)}")
    return queries


async def main():
    llm = DeepSeekProvider()

    from agent.server.rag.config import rag_config
    from agent.server.rag.embedder import Embedder
    print("[1/4] 加载 embedder（去重用）...")
    embedder = Embedder(model_name=rag_config.embedding_model,
                        local_path=rag_config.embedding_local_path or None)
    embedder.start_loading()
    for _ in range(60):
        if embedder.is_ready:
            break
        await asyncio.sleep(1)

    print("[2/4] 生成记忆库...")
    memories = await generate_memories(llm)
    memories = semantic_dedup(memories, embedder if embedder.is_ready else None)
    print(f"  记忆库共 {len(memories)} 条")

    print("[3/4] 生成查询 + 标注...")
    queries = await generate_queries(llm, memories)

    print("[4/4] 校验 + 对抗复核...")
    queries = await verify_annotations(llm, queries)
    queries = await adversarial_check(llm, memories, queries)

    data = {
        "meta": {
            "name": "D1 检索评测集",
            "generator": "LLM 生成 + 四道校验（规则/去重/校验器/对抗器）",
            "model": MODEL,
            "total_memories": len(memories),
            "total_queries": len(queries),
            "difficulty_breakdown": {
                d: sum(1 for q in queries if q["difficulty"] == d)
                for d in ("exact", "semantic", "cross")
            },
        },
        "memories": memories,
        "queries": queries,
    }
    path = write_json(data, "d1_retrieval.json")
    m = data["meta"]
    print(f"\n完成：记忆 {m['total_memories']} 条，查询 {m['total_queries']} 条")
    print(f"难度分布：{m['difficulty_breakdown']}")
    print(f"写入: {path}")


if __name__ == "__main__":
    asyncio.run(main())
