"""RAG 评估数据集生成脚本 — 全 LLM 管线，零检索模型参与。

核心改进 (v2):
  1. 去除 BM25/Reranker 参与标注 → 消除循环验证风险
  2. 批量标注: 一条 query 一次 LLM 调用标注全部候选 chunk
  3. Query 具体化约束: prompt 强制锚定 chunk 特有的具体信息
  4. 正例分级: 源 chunk 组 → rel=2, 同章扩展 → rel=1
  5. 两套数据集: 日常风格(500) + 学术风格(500)
  6. 双模型交叉验证: DeepSeek + Qwen(qwen-plus)

Chunk 分组策略:
  - source_chapter (55%): 同文献同章节连续 3-5 个 chunk
  - same_topic_cross (30%): 跨文献同主题 (26 个主题)
  - single_chunk (15%): 单 chunk 简单事实型

Usage:
  # DeepSeek 生成日常风格 500 条
  python -m rag_eval.scripts.generate_eval_dataset --style daily --provider deepseek

  # Qwen 生成学术风格 500 条
  python -m rag_eval.scripts.generate_eval_dataset --style academic --provider qwen

  # 测试模式
  python -m rag_eval.scripts.generate_eval_dataset --style daily --dry-run

  # 一次跑完两套 (两个 provider 各跑一遍)
  python -m rag_eval.scripts.generate_eval_dataset --style all --provider all
"""
import argparse
import asyncio
import json
import random
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Optional

AGENT_DIR = Path(__file__).resolve().parent.parent.parent / "agent"
sys.path.insert(0, str(AGENT_DIR))

import os
from dotenv import load_dotenv
load_dotenv(AGENT_DIR / ".env")

from server.utils.entity_names import (
    LOCATION_ALIASES, CHARACTER_NAMES, CHARACTER_ALIASES,
    EVENT_ALIASES, CONCEPT_ALIASES,
)
from server.llm.openai_compatible_provider import GenericOpenAIProvider

# ==============================================================================
# Provider 配置（api_key 从 agent/.env 环境变量读取，禁止硬编码）
# ==============================================================================
PROVIDER_CONFIGS = {
    "deepseek": {
        "api_key": os.getenv("DEEPSEEK_API_KEY", ""),
        "base_url": "https://api.deepseek.com/v1",
        "model": "deepseek-chat",
    },
    "qwen": {
        "api_key": os.getenv("DASHSCOPE_API_KEY", ""),
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "model": "qwen-plus",  # 性价比之选: 输入 ¥0.8/M, 128K 上下文
    },
}

# ==============================================================================
# 26 个跨文献同主题定义 (same_topic_cross 策略)
# ==============================================================================
TOPIC_DEFINITIONS = {
    # ── 人物类 (6) ──
    "白素贞形象": {
        "keywords": ["白素贞", "白娘子", "白氏", "白蛇", "白娘娘", "白珍娘"],
        "label": "白素贞在不同文献中的形象差异",
    },
    "许仙/许宣形象": {
        "keywords": ["许仙", "许宣", "奚宣赞", "许汉文", "许梦蛟之父"],
        "label": "许仙(许宣)在不同版本中的性格差异",
    },
    "小青/青蛇": {
        "keywords": ["小青", "青蛇", "青儿", "青青", "青蛇精", "青鱼"],
        "label": "小青在不同文献中的出身和能力",
    },
    "法海形象": {
        "keywords": ["法海", "法海禅师", "海公", "法海和尚", "禅师"],
        "label": "法海在不同版本中的形象演变",
    },
    "许梦蛟": {
        "keywords": ["梦蛟", "许梦蛟", "状元", "许士林"],
        "label": "许梦蛟在不同结局中的描写",
    },
    "观音/度化": {
        "keywords": ["观音", "度化", "点化", "菩萨", "大士", "世音"],
        "label": "观音在不同版本中的度化角色",
    },
    # ── 地点类 (5) ──
    "雷峰塔": {
        "keywords": ["雷峰塔", "雷锋塔", "镇压白蛇的塔", "黄妃塔", "塔"],
        "label": "雷峰塔在不同文献中的叙事功能变化",
    },
    "断桥": {
        "keywords": ["断桥", "段桥", "断桥残雪", "相遇之桥"],
        "label": "断桥在不同版本中的出现方式",
    },
    "西湖/钱塘": {
        "keywords": ["西湖", "西子湖", "钱塘湖", "钱塘", "钱塘门"],
        "label": "西湖作为故事背景的不同描写",
    },
    "金山寺": {
        "keywords": ["金山寺", "金山", "法海的寺庙", "水漫金山处"],
        "label": "金山寺在各版本中的作用",
    },
    "镇江/临安": {
        "keywords": ["镇江", "临安", "杭州", "苏州", "建康", "镇江渡口"],
        "label": "故事地理空间的变迁",
    },
    # ── 经典情节类 (10) ──
    "游湖借伞": {
        "keywords": ["借伞", "还伞", "游湖", "搭船", "艄公", "船家"],
        "label": "不同文献的相遇方式",
    },
    "盗银/库银案": {
        "keywords": ["盗银", "库银", "赃银", "发配", "公差", "周将仕", "典当"],
        "label": "不同版本的库银案细节",
    },
    "端午现形": {
        "keywords": ["端阳", "雄黄", "现形", "吓死", "原形", "雄黄酒"],
        "label": "不同版本的端午现形场景",
    },
    "盗仙草": {
        "keywords": ["仙草", "灵芝", "南极", "仙山", "救夫", "鹤童", "鹿童"],
        "label": "不同文献的盗仙草描写",
    },
    "水漫金山": {
        "keywords": ["水漫", "水淹", "水族", "法海对抗", "虾兵蟹将", "袈裟"],
        "label": "各版本的斗法高潮",
    },
    "合钵镇妖": {
        "keywords": ["合钵", "钵盂", "收服", "镇压", "金钵", "佛旨"],
        "label": "白蛇被收服的不同版本",
    },
    "状元祭塔": {
        "keywords": ["祭塔", "救母", "中举", "翰林", "修撰"],
        "label": "不同文献的结局写法",
    },
    "白蛇产子": {
        "keywords": ["产子", "分娩", "梦蛟出生", "临盆", "双胎", "药丸"],
        "label": "不同版本的生子情节",
    },
    "断桥重逢": {
        "keywords": ["断桥", "重逢", "和好", "破镜重圆", "解释"],
        "label": "不同版本的和解写法",
    },
    "白蛇报恩": {
        "keywords": ["报恩", "宿缘", "救命之恩", "前世救", "牧童"],
        "label": "不同版本的缘起叙事",
    },
    # ── 概念/母题类 (5) ──
    "人妖之恋": {
        "keywords": ["人妖", "人蛇", "色诱", "情欲", "人妖恋"],
        "label": "人妖相恋主题的跨文献讨论",
    },
    "佛道冲突": {
        "keywords": ["佛门", "道门", "茅山", "和尚", "道士", "佛法", "道法"],
        "label": "跨文献的宗教力量描写",
    },
    "科举入仕": {
        "keywords": ["科举", "状元", "中举", "翰林", "功名", "科甲"],
        "label": "科举元素在情节中的作用",
    },
    "报恩与宿缘": {
        "keywords": ["报恩", "宿缘", "前生", "因果", "三世", "夙缘"],
        "label": "跨文献的宿命论叙事",
    },
    "修行与度化": {
        "keywords": ["修行", "点化", "度化", "得道", "飞升", "仙班", "修炼"],
        "label": "修行主题在不同版本中的呈现",
    },
}


# ==============================================================================
# 辅助函数
# ==============================================================================
_query_counter: int = 0


def _qid() -> str:
    global _query_counter
    _query_counter += 1
    return f"q_{_query_counter:04d}"


def _make_query(
    text: str,
    category: str,
    difficulty: str,
    style: str,
    group_strategy: str,
    description: str = "",
    source_chunk_ids: Optional[list[str]] = None,
) -> dict:
    return {
        "query_id": _qid(),
        "text": text,
        "category": category,
        "difficulty": difficulty,
        "style": style,
        "group_strategy": group_strategy,
        "metadata_filters": None,
        "expected_genre_scope": "all",
        "eval_top_k_values": None,
        "description": description,
        "_source_chunk_ids": source_chunk_ids or [],
    }


def _make_qrel(query_id: str, doc_id: str, relevance: int) -> str:
    return f"{query_id}\t{doc_id}\t{relevance}"


def _strip_json(text: str) -> str:
    """清理 markdown 代码块标记。"""
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        text = "\n".join(lines[1:]) if len(lines) > 1 else text[3:]
    if text.endswith("```"):
        text = text[:-3].strip()
    return text


def _parse_json(text: str) -> Optional[dict | list]:
    """解析 LLM 返回的 JSON，带容错。"""
    text = _strip_json(text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    m = re.search(r'\{.*\}', text, re.DOTALL)
    if not m:
        m = re.search(r'\[.*\]', text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group())
        except json.JSONDecodeError:
            pass
    return None


# ==============================================================================
# Phase 1: 语料库加载 + 统计
# ==============================================================================
def load_corpus_raw(corpus_path: str | Path) -> list[dict]:
    """加载 corpus.jsonl 为原始文档列表。"""
    path = Path(corpus_path)
    if not path.exists():
        raise FileNotFoundError(f"语料库文件不存在: {path}")
    docs: list[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            docs.append(json.loads(line))
    return docs


# ==============================================================================
# Phase 2: Chunk 分组策略
# ==============================================================================
def _sliding_window_groups(
    bucket: list[dict],
    min_size: int = 4,
    max_size: int = 6,
    rng: random.Random | None = None,
) -> list[list[dict]]:
    """滑动窗口分组: 连续 chunk 编组。"""
    if rng is None:
        rng = random.Random(42)
    groups: list[list[dict]] = []
    n = len(bucket)
    if n < min_size:
        return groups

    stride = max(max_size // 3, 1)
    pos = 0
    while pos + min_size <= n:
        end = min(pos + rng.randint(min_size, max_size), n)
        if end - pos >= min_size:
            groups.append(bucket[pos:end])
        pos += stride
    return groups


def _group_key_from_ids(group: list[dict]) -> str:
    ids = sorted(d.get("doc_id", "") for d in group)
    return "|".join(ids)


def group_source_chapter(
    docs: list[dict],
    min_size: int = 4,
    max_size: int = 6,
    seed: int = 42,
) -> list[list[dict]]:
    """Strategy A: 同文献同章节连续 chunk 分组。"""
    rng = random.Random(seed)
    valid = [d for d in docs if len(d.get("text", "").strip()) >= 100]

    buckets: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for doc in valid:
        meta = doc.get("metadata", {})
        key = (meta.get("source_file", "unknown"), meta.get("chapter_number", "unknown"))
        buckets[key].append(doc)

    all_groups: list[list[dict]] = []
    for key, bucket in buckets.items():
        bucket.sort(key=lambda d: d.get("metadata", {}).get("char_start", 0))
        all_groups.extend(_sliding_window_groups(bucket, min_size, max_size, rng))

    sizes = Counter(len(g) for g in all_groups)
    print(f"  [source_chapter] {len(all_groups)} 组, 大小分布: {dict(sorted(sizes.items()))}")
    return all_groups


def group_same_topic_cross(
    docs: list[dict],
    min_size: int = 2,
    max_size: int = 4,
    seed: int = 42,
) -> list[list[dict]]:
    """Strategy B: 跨文献同主题分组。"""
    rng = random.Random(seed)
    valid = [d for d in docs if len(d.get("text", "").strip()) >= 100]

    # 构建关键词 → doc 索引
    keyword_docs: dict[str, dict[str, list[dict]]] = {}
    for topic_name, topic_def in TOPIC_DEFINITIONS.items():
        keyword_docs[topic_name] = defaultdict(list)
        for doc in valid:
            text = doc.get("text", "")
            for kw in topic_def["keywords"]:
                if kw in text:
                    keyword_docs[topic_name][kw].append(doc)
                    break

    all_groups: list[list[dict]] = []
    seen_keys: set[str] = set()

    for topic_name, kw_map in keyword_docs.items():
        # 收集该主题所有匹配的 chunk
        topic_docs: list[dict] = []
        seen_doc_ids: set[str] = set()
        for docs_list in kw_map.values():
            for d in docs_list:
                did = d.get("doc_id", "")
                if did not in seen_doc_ids:
                    seen_doc_ids.add(did)
                    topic_docs.append(d)

        if len(topic_docs) < min_size:
            continue

        # 按 source_file 聚类
        by_source: dict[str, list[dict]] = defaultdict(list)
        for d in topic_docs:
            sf = d.get("metadata", {}).get("source_file", "unknown")
            by_source[sf].append(d)

        source_files = sorted(by_source.keys())
        if len(source_files) < 2:
            continue  # 需要至少两个不同文献

        # 从每个 source_file 中取 1-2 个 chunk
        for _ in range(5):  # 每主题多抽几组
            chosen: list[dict] = []
            chosen_sources: set[str] = set()
            for sf in rng.sample(source_files, min(len(source_files), max_size)):
                pool = by_source[sf]
                pick = rng.choice(pool)
                chosen.append(pick)
                chosen_sources.add(sf)

            if len(chosen) >= min_size and len(chosen_sources) >= 2:
                key = _group_key_from_ids(chosen)
                if key not in seen_keys:
                    seen_keys.add(key)
                    all_groups.append(chosen)

    sizes = Counter(len(g) for g in all_groups)
    print(f"  [same_topic_cross] {len(all_groups)} 组 (来自 {len(TOPIC_DEFINITIONS)} 个主题), "
          f"大小分布: {dict(sorted(sizes.items()))}")
    return all_groups


def group_single_chunk(
    docs: list[dict],
    seed: int = 42,
) -> list[list[dict]]:
    """Strategy C: 单 chunk 分组 (信息密度高的 chunk)。"""
    rng = random.Random(seed)
    candidates: list[dict] = []
    for d in docs:
        text = d.get("text", "").strip()
        if 200 <= len(text) <= 1500:
            candidates.append(d)

    rng.shuffle(candidates)
    # 每个单 chunk 包装为只有一个元素的组
    groups = [[d] for d in candidates]
    print(f"  [single_chunk] {len(groups)} 组 (从 {len(candidates)} 个候选 chunk)")
    return groups


def build_all_groups(
    docs: list[dict],
    target_daily: int = 500,
    target_academic: int = 500,
    seed: int = 42,
) -> list[list[dict]]:
    """汇总三种策略，按比例采样，达到目标组数。

    使用 group_key → strategy 映射避免 doc 引用污染。
    """
    rng = random.Random(seed)
    total_target = target_daily + target_academic
    print(f"\n[分组] 目标总组数: {total_target} (日常 {target_daily} + 学术 {target_academic})")

    # 生成各组
    g1 = group_source_chapter(docs, seed=seed)
    g2 = group_same_topic_cross(docs, seed=seed)
    g3 = group_single_chunk(docs, seed=seed)

    # 去重 + strategy 映射 (g1 优先, 保持策略与留存组一致)
    all_groups: list[list[dict]] = []
    strategy_map: dict[str, str] = {}
    seen_keys: set[str] = set()

    for g in g1:
        key = _group_key_from_ids(g)
        if key not in seen_keys:
            seen_keys.add(key)
            strategy_map[key] = "source_chapter"
            all_groups.append(g)
    for g in g2:
        key = _group_key_from_ids(g)
        if key not in seen_keys:
            seen_keys.add(key)
            strategy_map[key] = "same_topic_cross"
            all_groups.append(g)
    for g in g3:
        key = _group_key_from_ids(g)
        if key not in seen_keys:
            seen_keys.add(key)
            strategy_map[key] = "single_chunk"
            all_groups.append(g)

    rng.shuffle(all_groups)
    print(f"  去重后总组数: {len(all_groups)}")

    # 按比例确定各策略目标数
    n1 = int(total_target * 0.55)
    n2 = int(total_target * 0.30)
    n3 = total_target - n1 - n2

    # 按策略分桶 (用 strategy_map 查找)
    s1 = [g for g in all_groups if strategy_map.get(_group_key_from_ids(g)) == "source_chapter"]
    s2 = [g for g in all_groups if strategy_map.get(_group_key_from_ids(g)) == "same_topic_cross"]
    s3 = [g for g in all_groups if strategy_map.get(_group_key_from_ids(g)) == "single_chunk"]

    rng.shuffle(s1)
    rng.shuffle(s2)
    rng.shuffle(s3)

    sampled: list[list[dict]] = []
    sampled.extend(s1[:n1])
    sampled.extend(s2[:n2])
    sampled.extend(s3[:n3])

    print(f"  source_chapter: 目标 {n1}, 可用 {len(s1)}, 采样 {min(n1, len(s1))}")
    print(f"  same_topic_cross: 目标 {n2}, 可用 {len(s2)}, 采样 {min(n2, len(s2))}")
    print(f"  single_chunk: 目标 {n3}, 可用 {len(s3)}, 采样 {min(n3, len(s3))}")

    rng.shuffle(sampled)

    # 将 strategy 存入组内每个 doc 供后续使用 (此时各组已独立，无覆盖风险)
    for g in sampled:
        key = _group_key_from_ids(g)
        strat = strategy_map.get(key, "unknown")
        for d in g:
            d["_group_strategy"] = strat

    sizes = Counter(len(g) for g in sampled)
    by_strategy = Counter(
        strategy_map.get(_group_key_from_ids(g), "?") for g in sampled
    )
    print(f"\n[分组总计] {len(sampled)} 组")
    print(f"  大小分布: {dict(sorted(sizes.items()))}")
    print(f"  策略分布: {dict(sorted(by_strategy.items()))}")

    return sampled


# ==============================================================================
# Phase 3: Query 生成 Prompt
# ==============================================================================
DAILY_QUERY_SYSTEM = """你是一个对白蛇传感兴趣的普通读者，正在搜索引擎中查找信息。

你的任务: 阅读以下文本块，生成 1 个问题。

**必须遵守的规则:**
1. 长度: 10-25 字，口语化，像普通人打字的感觉
2. 问题必须包含文本中的具体信息 —— 人名、地名、事件、物品、时间等
   使得只有该文本块及其同一情节的相邻文本能精准回答
3. 禁止笼统问题！比如 "白蛇传讲了什么" "白素贞是谁" "法海做了什么"
4. 禁止学术用语："请分析" "请比较" "试述" "如何体现" 等
5. 不要嵌套从句，只问一件事

**好问题的示例:**
  - "许仙在端阳节给白素贞喝了什么？"（具体到"喝了什么"这个细节）
  - "白娘子从什么地方来杭州找许宣？"（具体到地点）
  - "法海用什么法器收服了白蛇？"（具体到物品）
  - "李黄在杭州哪里遇到了白衣女子？"（具体到地点+人物）

**坏问题的示例:**
  - "白蛇传中端阳节发生了什么？"（太笼统，多个chunk都能回答）
  - "白素贞和许仙的关系如何？"（太宽泛）
  - "白蛇传的主题是什么？"（太笼统，无法检索）
  - "法海怎样降妖？"（太泛）

请输出JSON，source_relevance字段对每个源文本块用其ID标注:
{"text": "问题", "difficulty": "easy|medium|hard", "reasoning": "为何只有这些文本能精准回答", "source_relevance": {"ID1": 2, "ID2": 1, ...}}

source_relevance标准:
- 2: 包含回答问题的核心信息
- 1: 提供背景/上下文，单独不足以回答问题
注: 至少有一个2。"""

ACADEMIC_QUERY_SYSTEM = """你是一位白蛇传研究学者，正在学术数据库中进行文献检索。

你的任务: 阅读以下文本块，生成 1 个研究型问题。

**必须遵守的规则:**
1. 长度: 20-50 字，使用准确的专业术语
2. 问题必须锚定文本中的特有细节 —— 特定版本名、章节内容、具体概念、年代、典故来源等
   使得该问题需要检索到该文本块才能得到有依据的回答
3. 可以是版本间差异比较、典故溯源、概念分析、文本考据等
4. 禁止以 "请" "试述" "试论" 开头的指令句
5. 禁止太泛的问题，如 "白蛇传的主题是什么" "白蛇形象如何演变"

**好问题的示例:**
  - "方成培《雷峰塔传奇》中法海使用何种法器收服白娘子？"（具体到版本+物品）
  - "冯梦龙《白娘子永镇雷峰塔》中许宣如何发现白娘子是妖怪？"（具体到版本+情节细节）
  - "田汉京剧《白蛇传》中白素贞托孤时说了什么？"（具体到版本+具体对白）

**坏问题的示例:**
  - "白蛇传中法海如何降妖？"（太笼统）
  - "不同版本白蛇传的比较"（是主题不是问题）
  - "白蛇形象如何演变？"（泛化问题，多个chunk都能部分回答）

请输出JSON对象: {"text": "问题", "difficulty": "easy|medium|hard", "reasoning": "为何该问题具体且锚定于这些文本的特有信息"}"""


def _build_query_user_prompt(chunks: list[dict]) -> str:
    """构建 query 生成的 user prompt。"""
    meta0 = chunks[0].get("metadata", {})
    genre = meta0.get("genre", "未知")
    dynasty = meta0.get("dynasty", "未知")
    source_file = meta0.get("source_file", "未知")

    parts = []
    for i, chunk in enumerate(chunks):
        did = chunk.get("doc_id", f"chunk_{i}")
        text = chunk.get("text", "")
        parts.append(f"--- 文本块 {i + 1} (ID: {did}) ---\n{text[:2000]}")

    n_chunks = len(chunks)
    all_text = "\n\n".join(parts)
    return f"""{n_chunks} 个文本块，来自{dynasty}代，体裁{genre}，来源文件: {source_file}

{all_text}

请基于以上文本生成1个问题，并用source_relevance字段对上述{n_chunks}个文本块逐一标注相关性(至少一个2)。仅输出JSON。"""


async def generate_query_from_group(
    provider: GenericOpenAIProvider,
    group: list[dict],
    sem: asyncio.Semaphore,
    style: str,
    model: Optional[str] = None,
) -> Optional[dict]:
    """对一组 chunk 调用 LLM 生成 1 条 query。"""
    system_prompt = DAILY_QUERY_SYSTEM if style == "daily" else ACADEMIC_QUERY_SYSTEM

    async with sem:
        try:
            response = await provider.generate(
                system=system_prompt,
                user=_build_query_user_prompt(group),
                model=model,
                max_tokens=1024,  # Plan A: 输出含 source_relevance, 需要更多 token
                temperature=0.85,
            )
        except Exception as e:
            chunk_ids = [c.get("doc_id", "?") for c in group[:2]]
            print(f"  [LLM ERROR] style={style} group={chunk_ids}...: {e}")
            return None

    parsed = _parse_json(response.content)
    if not isinstance(parsed, dict):
        content_preview = response.content[:100]
        print(f"  [PARSE ERROR] style={style}: {content_preview}")
        return None

    text = parsed.get("text", "").strip()
    if not text:
        return None

    difficulty = parsed.get("difficulty", "medium")
    if difficulty not in ("easy", "medium", "hard"):
        difficulty = "medium"

    # 确定 category: 根据 group_strategy 推断
    strategy = next((d.get("_group_strategy", "source_chapter")
                     for d in group if d.get("_group_strategy")), "source_chapter")
    if strategy == "same_topic_cross":
        category = "comparison"
    elif strategy == "single_chunk":
        category = "entity"
    else:
        category = "plot"

    source_chunk_ids = [c.get("doc_id", "") for c in group]

    # Plan A: 解析生成时 LLM 对源 chunk 的标注
    source_relevance = parsed.get("source_relevance", {})
    source_grades: dict[str, int] = {}
    if isinstance(source_relevance, dict):
        for chunk_id, rel in source_relevance.items():
            if isinstance(rel, (int, float)) and chunk_id in set(source_chunk_ids):
                rel_int = int(rel)
                if rel_int in (1, 2):
                    source_grades[chunk_id] = rel_int

    query = _make_query(
        text=text,
        category=category,
        difficulty=difficulty,
        style=style,
        group_strategy=strategy,
        description=f"LLM生成自chunks {source_chunk_ids}: {parsed.get('reasoning', '')}",
        source_chunk_ids=source_chunk_ids,
    )
    # 将源标注附在 query 上供后续使用
    query["_source_grades"] = source_grades
    return query


# ==============================================================================
# Phase 4: 质量过滤
# ==============================================================================
FORBIDDEN_PATTERNS = [
    r'^请分析', r'^请结合', r'^请比较', r'^试述', r'^试论', r'^试比较',
    r'^请综合', r'^请说明', r'^请阐述', r'^请解释', r'^请论述',
    r'^请你', r'^请您', r'^请根据', r'^请以', r'^请从',
]


def quality_filter(query: dict) -> tuple[bool, str]:
    """质量过滤。返回 (通过?, 原因)。"""
    text = query.get("text", "").strip()
    length = len(text)

    # 1. 长度过滤
    if length < 8:
        return False, f"query过短 ({length}字): {text}"
    style = query.get("style", "daily")
    max_len = 60 if style == "daily" else 80
    if length > max_len:
        return False, f"query过长 ({length}字): {text[:50]}..."

    # 2. 指令性开头过滤
    for pat in FORBIDDEN_PATTERNS:
        if re.match(pat, text):
            return False, f"命中禁止模式 '{pat}': {text[:50]}"

    # 3. 笼统关键词检测
    vague_patterns = [
        (r'白蛇传讲了什么', '过于笼统'),
        (r'白素贞是谁', '过于笼统'),
        (r'法海做了什么', '过于笼统'),
        (r'白蛇传的主题', '过于笼统'),
    ]
    for pat, reason in vague_patterns:
        if re.search(pat, text):
            return False, f"{reason}: {text[:50]}"

    return True, ""


# ==============================================================================
# Phase 5: 批量 Relevance 标注
# ==============================================================================
BATCH_ANNOTATION_SYSTEM = """你是一位白蛇传文献研究专家，正在为检索系统的测试集标注相关性。

你的任务: 给定一个查询问题，逐个判断以下文本块能否帮助回答该问题。

判断标准:
- 2 (高度相关): 文本包含回答问题的核心信息 — 缺了它就答不完整
- 1 (部分相关): 文本对回答有帮助 — 包括: 事件前因后果、同一情节的不同侧面、相关背景知识、跨文献的同类描写
- 0 (不相关): 文本对回答该问题没有帮助，或讨论完全无关的内容

重要提示:
- 同一故事的不同段落即使不直接包含答案，只要与该问题关注的事件/人物/主题相关 → 标1
- 仅仅提到同一人物名字但没有更多关联 → 标0
- 不确定时偏向标1而非0

请对以下{n}个文本块逐一判断。输出JSON数组:
[{{"id": 1, "relevance": 0, "evidence": "引用的原文"}}, ...]"""


def _build_annotation_prompt(query_text: str, candidates: list[dict]) -> str:
    """构建批量标注的 user prompt。"""
    parts = [f"查询问题: {query_text}\n"]
    parts.append(f"以下 {len(candidates)} 个文本块，请逐一判断相关性:\n")

    for i, chunk in enumerate(candidates):
        did = chunk.get("doc_id", f"c_{i}")
        meta = chunk.get("metadata", {})
        dynasty = meta.get("dynasty", "?")
        genre = meta.get("genre", "?")
        text = chunk.get("text", "")[:500]
        parts.append(f"[{i + 1}] ID={did} ({dynasty}/{genre})\n{text}\n")

    parts.append(f"\n请输出包含 {len(candidates)} 个元素的 JSON 数组。")
    return "\n".join(parts)


async def batch_annotate_query(
    provider: GenericOpenAIProvider,
    query: dict,
    all_candidates: list[dict],
    sem: asyncio.Semaphore,
    model: Optional[str] = None,
) -> dict[str, int]:
    """对一条 query 的候选 chunk 做批量标注。

    Returns: {chunk_id: relevance}
    """
    query_text = query.get("text", "")
    n = len(all_candidates)

    async with sem:
        try:
            response = await provider.generate(
                system=BATCH_ANNOTATION_SYSTEM.format(n=n),
                user=_build_annotation_prompt(query_text, all_candidates),
                model=model,
                max_tokens=4096,
                temperature=0.3,  # 低温度提高标注一致性
            )
        except Exception as e:
            print(f"  [ANNOTATE ERROR] query={query.get('query_id')}: {type(e).__name__}: {e}")
            return {}

    parsed = _parse_json(response.content)
    # 容错: LLM 有时返回 dict (单元素) 而非 list
    if isinstance(parsed, dict):
        parsed = [parsed]
    if not isinstance(parsed, list):
        preview = response.content[:200] if hasattr(response, 'content') else str(response)[:200]
        print(f"  [ANNOTATE PARSE ERROR] query={query.get('query_id')}: type={type(parsed).__name__}, preview={preview}")
        return {}

    grades: dict[str, int] = {}
    for item in parsed:
        if not isinstance(item, dict):
            continue
        idx = item.get("id", -1)
        rel = item.get("relevance", 0)
        if not isinstance(idx, int) or idx < 1 or idx > n:
            continue
        chunk = all_candidates[idx - 1]
        chunk_id = chunk.get("doc_id", "")
        grades[chunk_id] = rel

    return grades


# ==============================================================================
# Phase 6: 候选池构建 (每条 query)
# ==============================================================================
def build_candidate_pool(
    query: dict,
    corpus_docs: list[dict],
    rng: random.Random,
) -> tuple[list[dict], list[str]]:
    """构建批量标注的候选池。

    返回: (candidates_list, negative_source_ids)

    候选池组成:
      - 源 chunk 组成员 (rel=2 候选)
      - 同章节其他 chunk (从 corpus 查找同 source_file + chapter_number)
      - 跨文献负例候选 (~10 个)
    去重后合并，前两部分在前面，负例候选在后面打乱
    """
    source_ids = set(query.get("_source_chunk_ids", []))
    style = query.get("style", "daily")

    # 源 chunk 组
    source_chunks = [d for d in corpus_docs if d.get("doc_id", "") in source_ids]

    # 同章节其他 chunk (rel=1 候选)
    source_meta = {}
    for d in source_chunks:
        meta = d.get("metadata", {})
        source_meta["source_file"] = meta.get("source_file", "")
        source_meta["chapter_number"] = meta.get("chapter_number", "")
        break
    same_chapter_candidates: list[dict] = []
    adj_chapter_candidates: list[dict] = []
    if source_meta:
        for d in corpus_docs:
            did = d.get("doc_id", "")
            if did in source_ids:
                continue
            meta = d.get("metadata", {})
            same_file = meta.get("source_file") == source_meta["source_file"]
            same_ch = meta.get("chapter_number") == source_meta["chapter_number"]
            if same_file and same_ch:
                same_chapter_candidates.append(d)
            elif same_file:
                # 相邻章节候选 (前一章末尾 / 后一章开头)
                adj_chapter_candidates.append(d)

    # 跨文献负例候选
    negative_candidates: list[dict] = []
    for d in corpus_docs:
        did = d.get("doc_id", "")
        if did in source_ids:
            continue
        meta = d.get("metadata", {})
        # 排除同 source_file 的 (留作同章扩展)
        if meta.get("source_file") == source_meta.get("source_file", ""):
            continue
        negative_candidates.append(d)

    # 从负例候选池中采样
    rng.shuffle(negative_candidates)
    n_neg = 8
    selected_negatives = negative_candidates[:n_neg]

    # 同章扩展也限制数量
    rng.shuffle(same_chapter_candidates)
    selected_same_chapter = same_chapter_candidates[:6]
    # 相邻章节候选
    rng.shuffle(adj_chapter_candidates)
    selected_adj_chapter = adj_chapter_candidates[:3]

    # Plan A: 源 chunk 已在生成时标注, 候选池不含源 chunk (省 token)
    candidates = selected_same_chapter + selected_adj_chapter + selected_negatives

    # 去重
    seen: set[str] = set()
    deduped: list[dict] = []
    for d in candidates:
        did = d.get("doc_id", "")
        if did not in seen:
            seen.add(did)
            deduped.append(d)

    return deduped


# ==============================================================================
# Phase 7: 主生成流程
# ==============================================================================
async def async_main(
    corpus_path: str,
    queries_output: str,
    qrels_output: str,
    style: str,
    provider: GenericOpenAIProvider,
    llm_model: Optional[str] = None,
    target_count: int = 500,
    dry_run: bool = False,
    concurrency: int = 8,
    seed: int = 42,
):
    """主异步生成流程。"""
    if llm_model is None:
        llm_model = provider.default_model

    print("=" * 60)
    print(f"RAG Eval Dataset Generator v2")
    print(f"  Style: {style}")
    print(f"  Provider: {type(provider).__name__} / {llm_model}")
    print(f"  Target: {target_count} queries")
    print("=" * 60)

    rng = random.Random(seed)

    # ---- Step 1: 加载语料 ----
    print("\n[Step 1/6] 加载语料库...")
    corpus_docs = load_corpus_raw(corpus_path)
    print(f"  已加载 {len(corpus_docs)} 篇文档")

    # ---- Step 2: Chunk 分组 ----
    print(f"\n[Step 2/6] Chunk 分组 (source_chapter 55% + same_topic_cross 30% + single_chunk 15%)...")
    groups = build_all_groups(
        corpus_docs,
        target_daily=target_count if style == "daily" else 0,
        target_academic=target_count if style == "academic" else 0,
        seed=seed,
    )
    print(f"  实际分组数: {len(groups)}")
    if len(groups) < target_count:
        print(f"  [WARN] 组数 ({len(groups)}) < 目标 ({target_count}), 将调整目标")
        target_count = len(groups)

    # ---- Step 3: LLM 生成 Query ----
    print(f"\n[Step 3/6] LLM Query 生成 ({style} 风格)...")
    if dry_run:
        groups = groups[:20]
        print(f"  [DRY RUN] 仅处理 {len(groups)} 组")

    sem = asyncio.Semaphore(concurrency)
    queries: list[dict] = []
    n_errors = 0
    n_filtered = 0
    start_time = time.monotonic()

    batch_size = concurrency * 2
    for batch_start in range(0, len(groups), batch_size):
        batch = groups[batch_start:batch_start + batch_size]
        tasks = [
            generate_query_from_group(provider, group, sem, style, llm_model)
            for group in batch
        ]
        results = await asyncio.gather(*tasks)

        for result in results:
            if result is None:
                n_errors += 1
                continue
            passed, reason = quality_filter(result)
            if not passed:
                n_filtered += 1
                if n_filtered <= 5:
                    print(f"  [FILTER] {reason}")
                continue
            queries.append(result)

        elapsed = time.monotonic() - start_time
        done = min(batch_start + batch_size, len(groups))
        valid = len(queries)
        print(f"  进度: {done}/{len(groups)} ({elapsed:.0f}s) — "
              f"生成 {valid} 条, 错误 {n_errors}, 过滤 {n_filtered}")

        if valid >= target_count:
            break

    print(f"  Query 生成完成: {len(queries)} 条 (错误 {n_errors}, 过滤 {n_filtered})")

    # 截断到目标数量
    if len(queries) > target_count:
        queries = queries[:target_count]

    # ---- Step 4: 构建候选池 ----
    print(f"\n[Step 4/6] 构建候选池...")
    doc_lookup: dict[str, dict] = {d["doc_id"]: d for d in corpus_docs}
    # 为每条 query 构建候选池
    candidate_pools: dict[str, list[dict]] = {}
    for q in queries:
        qid = q["query_id"]
        # 将 source_ids 转回 doc 对象
        source_ids = set(q.get("_source_chunk_ids", []))
        source_docs = [doc_lookup[sid] for sid in source_ids if sid in doc_lookup]
        candidate_pools[qid] = build_candidate_pool(q, corpus_docs, rng)
        if (len(queries) <= 5 or queries.index(q) == 0):
            n_chunks = len(candidate_pools[qid])
            n_source = len(source_ids & {d["doc_id"] for d in candidate_pools[qid]})
            print(f"  [{qid}] 候选池: {n_chunks} chunks (源={n_source})")

    # ---- Step 5: 批量 Relevance 标注 ----
    print(f"\n[Step 5/6] 批量 Relevance 标注...")
    annotation_sem = asyncio.Semaphore(concurrency)
    all_grades: dict[str, dict[str, int]] = {}

    for batch_start in range(0, len(queries), batch_size):
        batch = queries[batch_start:batch_start + batch_size]
        tasks = [
            batch_annotate_query(provider, q, candidate_pools.get(q["query_id"], []),
                                 annotation_sem, llm_model)
            for q in batch
        ]
        batch_results = await asyncio.gather(*tasks)
        for q, grades in zip(batch, batch_results):
            all_grades[q["query_id"]] = grades

        elapsed = time.monotonic() - start_time
        done = min(batch_start + batch_size, len(queries))
        print(f"  标注进度: {done}/{len(queries)} ({elapsed:.0f}s)")

    # ---- Step 6: 重新分配 query_id + 输出 ----
    print(f"\n[Step 6/6] 输出...")

    # 重新编号
    global _query_counter
    _query_counter = 0
    old_to_new: dict[str, str] = {}
    for i, q in enumerate(queries):
        old_qid = q["query_id"]
        new_qid = _qid()
        q["query_id"] = new_qid
        old_to_new[old_qid] = new_qid

    # 更新 grades 中的 query_id
    new_grades: dict[str, dict[str, int]] = {}
    for old_qid, grades in all_grades.items():
        if old_qid in old_to_new:
            new_grades[old_to_new[old_qid]] = grades

    # 写 queries
    queries_path = Path(queries_output)
    queries_path.parent.mkdir(parents=True, exist_ok=True)
    with open(queries_path, "w", encoding="utf-8") as f:
        for q in queries:
            # 移除内部字段
            q_out = {k: v for k, v in q.items() if not k.startswith("_")}
            f.write(json.dumps(q_out, ensure_ascii=False) + "\n")

    # 写 qrels
    qrels_lines: list[str] = [
        f"# qrels — LLM 批量标注的 relevance judgments ({style} / {type(provider).__name__})",
        "# 格式: query_id\tdoc_id\trelevance",
        "# relevance: 2=源chunk核心答案, 1=同章扩展辅助上下文, 0=LLM判定不相关",
        "# 标注方法: 全 LLM 管线, 零检索模型参与 (无 BM25/Embedder/Reranker)",
        "",
    ]

    rel_counts = Counter()
    query_pos_counts: list[int] = []
    skipped_queries = 0

    for q in queries:
        qid = q["query_id"]
        source_ids = set(q.get("_source_chunk_ids", []))
        grades = new_grades.get(qid, {})

        pos_count = 0
        # Plan A: 源 chunk 优先用生成时的标注 (_source_grades), 其次用批量标注 (grades)
        source_grades_from_gen = q.get("_source_grades", {})

        for sid in source_ids:
            # 优先取生成时的标注, 其次批量标注, 都没有默认 2
            rel = source_grades_from_gen.get(sid) or grades.get(sid) or 2
            if rel not in (1, 2):
                rel = 2
            qrels_lines.append(_make_qrel(qid, sid, rel))
            rel_counts[rel] += 1
            if rel > 0:
                pos_count += 1

        # 非源 chunk: 保留批量标注结果
        for chunk_id, rel in grades.items():
            if chunk_id in source_ids:
                continue  # 已在上面处理
            if rel == 0:
                # rel=0 为负例, 也写入
                qrels_lines.append(_make_qrel(qid, chunk_id, 0))
                rel_counts[0] += 1
            elif rel == 1:
                qrels_lines.append(_make_qrel(qid, chunk_id, 1))
                rel_counts[1] += 1
                pos_count += 1
            elif rel == 2:
                qrels_lines.append(_make_qrel(qid, chunk_id, 2))
                rel_counts[2] += 1
                pos_count += 1

        if pos_count == 0:
            skipped_queries += 1
        query_pos_counts.append(pos_count)

    qrels_path = Path(qrels_output)
    qrels_path.parent.mkdir(parents=True, exist_ok=True)
    with open(qrels_path, "w", encoding="utf-8") as f:
        f.write("\n".join(qrels_lines) + "\n")

    # 统计
    queries_fname = queries_path.name
    qrels_fname = qrels_path.name
    print(f"\n{'='*60}")
    print(f"[OK] 生成完成!")
    print(f"{'='*60}")
    print(f"  queries: {len(queries)} 条 → {queries_path}")
    print(f"  qrels: {len(qrels_lines) - 4} 条标注 → {qrels_path}")
    print(f"\n  Relevance 分布:")
    print(f"    rel=2 (核心): {rel_counts[2]}")
    print(f"    rel=1 (辅助): {rel_counts[1]}")
    print(f"    rel=0 (负例): {rel_counts[0]}")
    if query_pos_counts:
        avg_pos = sum(query_pos_counts) / len(query_pos_counts)
        min_pos = min(query_pos_counts)
        max_pos = max(query_pos_counts)
        pos_dist = Counter(query_pos_counts)
        print(f"\n  每 query 正例数: avg={avg_pos:.1f}, min={min_pos}, max={max_pos}")
        print(f"  分布: {dict(sorted(pos_dist.items()))}")
    print(f"  无正例查询: {skipped_queries}")
    if skipped_queries > 0:
        print(f"  [WARN] {skipped_queries} 条查询无任何正例 — 评估时会被跳过")

    by_cat = Counter(q["category"] for q in queries)
    by_diff = Counter(q["difficulty"] for q in queries)
    by_strategy = Counter(q.get("group_strategy", "?") for q in queries)
    print(f"\n  类别分布: {dict(sorted(by_cat.items()))}")
    print(f"  难度分布: {dict(sorted(by_diff.items()))}")
    print(f"  分组策略分布: {dict(sorted(by_strategy.items()))}")

    # 关闭 provider
    await provider.close()


# ==============================================================================
# CLI
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="RAG Eval Dataset Generator v2 — 全 LLM 管线"
    )
    _PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
    _default_corpus = str(_PROJECT_ROOT / "rag_eval" / "ground_truth" / "corpus.jsonl")

    parser.add_argument("--corpus", default=_default_corpus, help="语料库路径")
    parser.add_argument("--style", default="daily",
                        choices=["daily", "academic", "all"],
                        help="Query 风格 (默认: daily)")
    parser.add_argument("--provider", default="deepseek",
                        choices=list(PROVIDER_CONFIGS.keys()) + ["all"],
                        help="LLM provider (默认: deepseek)")
    parser.add_argument("--model", default=None, help="LLM 模型名 (默认使用 provider 默认模型)")
    parser.add_argument("--target", type=int, default=500, help="目标 query 数 (默认: 500)")
    parser.add_argument("--concurrency", type=int, default=10, help="LLM 并发数 (默认: 10)")
    parser.add_argument("--dry-run", action="store_true", help="测试模式: 仅处理 20 组")
    parser.add_argument("--output", default=None, help="查询输出路径")
    parser.add_argument("--qrels", default=None, help="qrels 输出路径")
    args = parser.parse_args()

    styles_to_run: list[str] = []
    if args.style == "all":
        styles_to_run = ["daily", "academic"]
    else:
        styles_to_run = [args.style]

    providers_to_run: list[str] = []
    if args.provider == "all":
        providers_to_run = list(PROVIDER_CONFIGS.keys())
    else:
        providers_to_run = [args.provider]

    for style in styles_to_run:
        for prov_name in providers_to_run:
            cfg = PROVIDER_CONFIGS[prov_name]
            provider = GenericOpenAIProvider(
                api_key=cfg["api_key"],
                base_url=cfg["base_url"],
                default_model=cfg["model"],
            )
            model = args.model or cfg["model"]

            # 输出路径
            if args.output:
                queries_out = args.output
            else:
                queries_out = str(_PROJECT_ROOT / "rag_eval" / "ground_truth"
                                  / f"queries_{style}_{prov_name}.jsonl")
            if args.qrels:
                qrels_out = args.qrels
            else:
                qrels_out = str(_PROJECT_ROOT / "rag_eval" / "ground_truth"
                                / f"qrels_{style}_{prov_name}.tsv")

            print(f"\n[Run] style={style} provider={prov_name} model={model}")
            print(f"       queries → {queries_out}")
            print(f"       qrels   → {qrels_out}")

            asyncio.run(async_main(
                corpus_path=args.corpus,
                queries_output=queries_out,
                qrels_output=qrels_out,
                style=style,
                provider=provider,
                llm_model=model,
                target_count=args.target,
                dry_run=args.dry_run,
                concurrency=args.concurrency,
            ))


if __name__ == "__main__":
    main()
