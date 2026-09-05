"""查询草稿生成脚本 (T07.01 — Query-from-Document)。

采用 Query-from-Document 方法生成 RAG 评估数据集:
  1. 从 corpus.jsonl 中按体裁×朝代分层采样文档 chunk
  2. 对每个采样 chunk，调用 LLM 生成该 chunk 能回答的自然语言问题
  3. 源 chunk 自动成为正例 (relevance=2)
  4. 用 BM25 检索生成的 hard negatives 作为负例 (relevance=0)
  5. 补充鲁棒性/别名等特殊类别查询
  6. 输出 queries.jsonl + qrels_auto.tsv，供人工审核 (HITL)

Usage:
    # 默认使用 DeepSeek，输出 queries_deepseek.jsonl + qrels_auto_deepseek.tsv
    python -m rag_eval.scripts.generate_query_drafts

    # 切换 Qwen，输出 queries_qwen.jsonl + qrels_auto_qwen.tsv (不会覆盖 deepseek 的文件)
    python -m rag_eval.scripts.generate_query_drafts --provider qwen

    # 切换 文心一言
    python -m rag_eval.scripts.generate_query_drafts --provider ernie

    # 指定模型 + 测试模式
    python -m rag_eval.scripts.generate_query_drafts --provider qwen --model qwen-max --dry-run --sample 10

    # 手动指定输出路径 (覆盖默认)
    python -m rag_eval.scripts.generate_query_drafts --provider deepseek --output my_queries.jsonl --qrels my_qrels.tsv
"""
import argparse
import asyncio
import json
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Optional

# 动态添加 agent 目录
AGENT_DIR = Path(__file__).resolve().parent.parent.parent / "agent"
sys.path.insert(0, str(AGENT_DIR))

import os
from dotenv import load_dotenv
load_dotenv(AGENT_DIR / ".env")

from server.utils.entity_names import LOCATION_ALIASES, CHARACTER_NAMES
from server.llm.openai_compatible_provider import GenericOpenAIProvider

# ==============================================================================
# LLM Provider 配置（api_key 从 agent/.env 环境变量读取，禁止硬编码）
# ==============================================================================
# 所有 provider 均为 OpenAI 兼容协议: 需要 api_key + base_url + model
# 可通过命令行 --provider <name> 切换
PROVIDER_CONFIGS = {
    "deepseek": {
        "api_key": os.getenv("DEEPSEEK_API_KEY", ""),
        "base_url": "https://api.deepseek.com/v1",
        "model": "deepseek-chat",
    },
    "qwen": {
        "api_key": os.getenv("DASHSCOPE_API_KEY", ""),
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "model": "qwen-plus",
    },
    "ernie": {
        "api_key": os.getenv("QIANFAN_API_KEY", ""),
        "base_url": "https://qianfan.baidubce.com/v2",
        "model": "ernie-4.0-turbo-8k-latest",
    },
}
PROVIDER_DEFAULT = "deepseek"

# ==============================================================================
# 数据源
# ==============================================================================
LOCATIONS = list(LOCATION_ALIASES.keys())  # 12 个地点
CHARACTERS = CHARACTER_NAMES  # 6 个角色
DYNASTIES = ["唐", "宋", "元", "明", "清", "现当代"]
GENRES = [
    "classical_chinese", "huaben", "imitation_huaben",
    "opera_libretto", "chapter_vernacular", "tanci", "script",
    "research_literature",
]
GENRE_LABELS = {
    "classical_chinese": "文言小说",
    "huaben": "话本小说",
    "imitation_huaben": "拟话本",
    "opera_libretto": "戏曲剧本",
    "chapter_vernacular": "章回体白话小说",
    "tanci": "弹词",
    "script": "现当代剧本",
    "research_literature": "研究文献",
}

CATEGORIES = [
    "entity", "location", "plot", "comparison",
    "motif", "filtered", "alias", "intertext", "robustness",
]

# ==============================================================================
# 查询记录构建
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
    description: str = "",
    metadata_filters: Optional[dict] = None,
    expected_genre_scope: str = "all",
    source_chunk_id: Optional[str] = None,
) -> dict:
    return {
        "query_id": _qid(),
        "text": text,
        "category": category,
        "difficulty": difficulty,
        "metadata_filters": metadata_filters,
        "expected_genre_scope": expected_genre_scope,
        "eval_top_k_values": None,
        "description": description,
        "_source_chunk_id": source_chunk_id,
    }


def _make_qrel(query_id: str, doc_id: str, relevance: int) -> str:
    """生成 qrels TSV 行。"""
    return f"{query_id}\t{doc_id}\t{relevance}"


# ==============================================================================
# Phase 1: 语料库加载 + 分层采样
# ==============================================================================
def load_corpus_raw(corpus_path: str | Path) -> list[dict]:
    """加载 corpus.jsonl，返回原始文档列表。"""
    path = Path(corpus_path)
    if not path.exists():
        raise FileNotFoundError(f"语料库文件不存在: {path}")

    docs: list[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            doc = json.loads(line)
            docs.append(doc)
    return docs


def stratified_sample(
    docs: list[dict],
    n_per_stratum: int = 8,
    seed: int = 42,
) -> list[dict]:
    """按 genre × dynasty 分层采样文档。

    每个 (genre, dynasty) 组合最多采样 n_per_stratum 条。
    不足则全部取用。研究文献 (research_literature) 不区分朝代。
    """
    rng = random.Random(seed)
    strata: dict[tuple[str, str], list[dict]] = defaultdict(list)

    for doc in docs:
        meta = doc.get("metadata", {})
        genre = meta.get("genre", "unknown")
        dynasty = meta.get("dynasty")
        source_type = meta.get("source_type", "")
        # 研究文献不区分朝代
        if source_type == "research_literature":
            dynasty = "_research"
        else:
            dynasty = dynasty or "unknown"
        strata[(genre, dynasty)].append(doc)

    sampled: list[dict] = []
    stratum_stats: dict[str, int] = defaultdict(int)

    for (genre, dynasty), pool in sorted(strata.items()):
        n = min(n_per_stratum, len(pool))
        chosen = rng.sample(pool, n) if n < len(pool) else list(pool)
        rng.shuffle(chosen)
        sampled.extend(chosen)
        stratum_stats[f"{genre}/{dynasty}"] = n

    print(f"[采样] {len(docs)} 篇文档 -> {len(sampled)} 篇采样")
    print(f"  分层数: {len(stratum_stats)}")
    for key, n in sorted(stratum_stats.items()):
        print(f"    {key}: {n}")

    return sampled


# ==============================================================================
# Phase 2: LLM Query-from-Document 生成
# ==============================================================================
QUERY_GENERATION_SYSTEM = """你是一位中国古典文学研究专家，正在为一个 RAG 检索评估系统生成测试查询。

你的任务: 阅读一段白蛇传相关文献的文本块 (chunk)，生成 **2 个不同的** 自然语言问题。
每个问题应满足:
1. **该文本块确实能回答**这个问题 (答案在 chunk 的文本内容之中)
2. 问题自然、流畅、像一个真实用户会问的问题
3. 两个问题应该属于不同的查询类别 (category)，覆盖不同类型的检索需求
4. 两个问题应该有不同的难度等级 (difficulty)

你需要输出一个 JSON 数组，包含 2 个对象，每个对象包含以下字段:
- "text": 生成的查询问题文本 (字符串)
- "category": 查询类别，从以下 9 类中选择，**确保两个问题选择不同的类别**:
    "entity"       — 实体检索 (问特定角色/地点/文献名的事实信息)
    "location"     — 地理位置查询 (问地点的方位、路线、位置关系)
    "plot"         — 情节检索 (问故事情节、事件经过、细节)
    "comparison"   — 跨文献比较 (需要比较不同版本/体裁/朝代的差异)
    "motif"        — 主题/母题检索 (问文学主题、文化意象、思想观念)
    "filtered"     — 带过滤条件的检索 (适合加朝代/体裁等过滤条件)
    "intertext"    — 互文检索 (问典故引用、文本间关系、文献渊源)
    "alias"        — 别名检索 (使用实体别称/异名来提问)
    "robustness"   — 鲁棒性测试 (口语化/模糊/特殊格式查询)
- "difficulty": 难度等级，**确保两个问题选择不同的难度**:
    "easy"   — 问题直接明确，答案在单个 chunk 中直接可见
    "medium" — 需要一定的理解和段落内关联能力
    "hard"   — 需要跨段落综合、深度分析或特定专业知识
- "reasoning": 简短说明为什么选择该类别和难度 (1句话)

**重要**: 优先选择除了 entity 和 plot 之外的其他类别，因为 entity 和 plot 在测试中容易出现过多。
优先考虑: comparison (比较不同版本的差异), motif (主题思想分析), intertext (典故来源考), filtered (限定朝代/体裁检索)。

输出格式: 仅输出 JSON 数组，不要有其他文字。
示例: [{"text": "...", "category": "comparison", "difficulty": "hard", "reasoning": "..."}, {"text": "...", "category": "motif", "difficulty": "easy", "reasoning": "..."}]"""


def _build_query_generation_prompt(chunk_text: str, chunk_meta: dict) -> str:
    """构建单条 query 生成的 user prompt。"""
    genre = chunk_meta.get("genre", "未知")
    dynasty = chunk_meta.get("dynasty", "未知")
    source_type = chunk_meta.get("source_type", "未知")
    chunk_id = chunk_meta.get("chunk_id", "未知")

    return f"""以下是白蛇传文献的一个文本块，来自{dynasty}代，体裁为{genre}，来源类型为{source_type}。
文本块ID: {chunk_id}

请仔细阅读此文本，然后按系统提示的要求生成一个该文本能够回答的查询问题。

---
{chunk_text[:3000]}
---

请仅输出 JSON。"""


async def generate_queries_from_chunk(
    provider: GenericOpenAIProvider,
    chunk: dict,
    sem: asyncio.Semaphore,
    model: Optional[str] = None,
) -> list[dict]:
    """对单个 chunk 调用 LLM 生成 2 条查询 (不同类别 + 不同难度)。

    返回: [{query_dict, ...}, ...]
    失败时返回空列表。
    """
    chunk_text = chunk.get("text", "")
    doc_id = chunk.get("doc_id", "")
    meta = chunk.get("metadata", {})

    # 文本太短则跳过
    if len(chunk_text.strip()) < 100:
        return []

    async with sem:
        try:
            response = await provider.generate(
                system=QUERY_GENERATION_SYSTEM,
                user=_build_query_generation_prompt(chunk_text, meta),
                model=model,
                max_tokens=1536,
                temperature=0.85,
            )
        except Exception as e:
            print(f"  [LLM ERROR] chunk={doc_id}: {e}")
            return []

    content = response.content.strip()

    # 清理 markdown 代码块标记
    if content.startswith("```"):
        lines = content.split("\n")
        content = "\n".join(lines[1:]) if len(lines) > 1 else content[3:]
    if content.endswith("```"):
        content = content[:-3].strip()

    # 解析 JSON 数组
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError:
        # 尝试提取 JSON 数组
        import re
        m = re.search(r'\[.*\]', content, re.DOTALL)
        if not m:
            print(f"  [PARSE ERROR] chunk={doc_id}: {content[:120]}")
            return []
        try:
            parsed = json.loads(m.group())
        except json.JSONDecodeError:
            print(f"  [PARSE ERROR] chunk={doc_id}: {content[:120]}")
            return []

    # 统一为列表
    if isinstance(parsed, dict):
        parsed = [parsed]
    if not isinstance(parsed, list):
        return []

    queries = []
    for result in parsed:
        text = result.get("text", "").strip()
        if not text:
            continue

        category = result.get("category", "entity")
        if category not in CATEGORIES:
            category = "entity"

        difficulty = result.get("difficulty", "medium")
        if difficulty not in ("easy", "medium", "hard"):
            difficulty = "medium"

        source_type = meta.get("source_type", "")
        genre = meta.get("genre", "")
        dynasty = meta.get("dynasty", "")

        # 根据 chunk 的 metadata 自动确定 metadata_filters
        metadata_filters = None
        expected_genre_scope = "all"
        if category == "filtered":
            if dynasty and genre:
                metadata_filters = {"dynasty": dynasty, "genre": genre}
                expected_genre_scope = genre
            elif dynasty:
                metadata_filters = {"dynasty": dynasty}
                expected_genre_scope = dynasty
            elif genre:
                metadata_filters = {"genre": genre}
                expected_genre_scope = genre

        queries.append(_make_query(
            text=text,
            category=category,
            difficulty=difficulty,
            description=f"LLM 生成自 chunk {doc_id}: {result.get('reasoning', '')}",
            metadata_filters=metadata_filters,
            expected_genre_scope=expected_genre_scope,
            source_chunk_id=doc_id,
        ))

    return queries


# ==============================================================================
# Phase 3: 负例生成 (BM25 retrieval)
# ==============================================================================
def generate_hard_negatives(
    queries: list[dict],
    corpus_docs: list[dict],
    top_k: int = 10,
) -> dict[str, list[str]]:
    """对每条 query 使用 BM25 检索，找到 top_k 个不相关文档作为 hard negatives。

    Returns:
        {query_id: [neg_doc_id, ...]}
    """
    # 构建 BM25 使用的文档列表
    from server.rag.bm25_retriever import BM25Retriever

    bm25 = BM25Retriever()

    # 将文档加入 BM25 索引 (API: index([{"content", "chunk_id", "metadata"}, ...]))
    docs_for_index = []
    for doc in corpus_docs:
        docs_for_index.append({
            "chunk_id": doc["doc_id"],
            "content": doc.get("text", ""),
            "metadata": doc.get("metadata", {}),
        })

    print(f"  构建 BM25 索引 ({len(docs_for_index)} 文档)...")
    bm25.index(docs_for_index)
    print(f"  BM25 索引构建完成")

    hard_negatives: dict[str, list[str]] = {}

    for i, q in enumerate(queries):
        query_id = q["query_id"]
        query_text = q["text"]
        source_chunk_id = q.get("_source_chunk_id", "")

        try:
            results = bm25.search(query_text, top_k=top_k + 5)
        except Exception as e:
            print(f"  [BM25 WARN] query={query_id}: {e}")
            hard_negatives[query_id] = []
            continue

        negs = []
        for r in results:
            chunk_id = r.get("chunk_id", "")
            # 排除源文档自身
            if chunk_id and chunk_id != source_chunk_id:
                negs.append(chunk_id)
            if len(negs) >= top_k:
                break

        hard_negatives[query_id] = negs

        if (i + 1) % 50 == 0:
            print(f"  BM25 负例进度: {i+1}/{len(queries)}")

    return hard_negatives


# ==============================================================================
# Phase 4: 补充查询生成 (鲁棒性 + 别名 + 过滤)
# ==============================================================================
def _gen_robustness_queries(count: int = 50) -> list[dict]:
    """生成鲁棒性测试查询 (不依赖文档内容)。"""
    queries = []

    zero_relevant = [
        ("白蛇传中提到了外星人吗", "hard", "zero-relevant: 语料库中不应有科幻相关内容"),
        ("白素贞的手机号码是多少", "hard", "zero-relevant: 明显超出语料范围"),
        ("人工智能在白蛇传中的应用", "hard", "zero-relevant: 现代概念在古典文献中"),
        ("白蛇传中的量子力学原理", "hard", "zero-relevant: 科学概念"),
        ("白素贞会不会开汽车", "hard", "zero-relevant: 现代交通工具"),
        ("许仙的电子邮件地址", "hard", "zero-relevant: 现代通讯工具"),
        ("法海的微信头像", "hard", "zero-relevant: 当代社交工具"),
        ("白蛇传中有没有飞机", "hard", "zero-relevant: 现代交通工具"),
    ]
    noisy = [
        ("法海和白素贞谁更厉害", "medium", "口语化查询"),
        ("白蛇传到底讲了什么", "medium", "口语化查询"),
        ("白娘子为啥被关在塔里", "medium", "高度口语化"),
        ("许仙是不是很怂", "medium", "口语化+主观评价"),
        ("法海凭什么拆散人家", "medium", "口语化+情感倾向"),
        ("白蛇传里面谁最坏", "medium", "口语化主观评价"),
        ("为什么大家都同情白蛇", "medium", "口语化查询"),
        ("白蛇传好看吗", "medium", "口语化+主观评价"),
        ("讲讲白蛇的故事呗", "medium", "口语化请求"),
    ]
    typo = [
        ("白shet传", "hard", "拼写错误"),
        ("白舌传", "hard", "同音错别字"),
        ("白蛇轉", "hard", "繁体字查询"),
        ("白蛇伝", "hard", "日文汉字"),
        ("白蛇 传", "hard", "中间额外空格"),
        ("  雷峰塔  ", "hard", "前后多余空格"),
        ("白\t蛇\t传", "hard", "制表符分隔"),
    ]
    multilingual = [
        ("white snake legend", "hard", "英文查询"),
        ("Legend of the White Snake", "hard", "英文完整标题"),
        ("Madame White Snake", "hard", "英文别称"),
        ("白蛇伝説", "hard", "日文查询"),
        ("백사전", "hard", "韩文查询"),
    ]
    keyword_stack = [
        ("白素贞 许仙 相遇", "medium", "空格分隔多关键词"),
        ("断桥法海雷峰塔白素贞许仙", "medium", "无分隔连续实体名"),
        ("白蛇传 故事 梗概 简介 摘要", "medium", "同义词堆叠"),
        ("雷峰塔 金山寺 西湖 断桥", "medium", "多地点的空格分隔查询"),
        ("白素贞许仙小青法海", "medium", "四角色名无分隔"),
        ("白蛇传 许仙 白娘子 法海 小青 雷峰塔", "medium", "密集关键词"),
    ]
    extreme = [
        ("白蛇", "easy", "超短查询（2字）"),
        ("法海", "easy", "超短查询（人名2字）"),
        ("雷峰塔", "easy", "超短查询（地名3字）"),
        ("白素贞与许仙在断桥相遇后共同经历了许多波折，包括端阳现形、盗取仙草、水漫金山，最终却被法海镇压在雷峰塔下，这个故事究竟传达了什么样的爱情观和命运观", "medium", "超长查询"),
    ]
    ambiguous = [
        ("塔的故事", "hard", "模糊查询: 哪个塔不确定"),
        ("和尚斗蛇", "hard", "极度简略模糊查询"),
        ("杭州的爱情故事", "hard", "地理+主题模糊查询"),
        ("白蛇的故事", "hard", "泛指白蛇相关所有内容"),
        ("中国民间故事蛇", "hard", "泛指性查询"),
    ]
    negation = [
        ("白蛇传中不涉及爱情的段落", "hard", "否定式查询"),
        ("除白素贞外的女性角色描写", "hard", "排除式查询"),
        ("非佛教视角下的白蛇传解读", "hard", "排除式查询: 排除宗教视角"),
        ("不包含法海的白蛇传段落", "hard", "排除式查询"),
    ]

    all_robustness = (
        zero_relevant + noisy + typo + multilingual +
        keyword_stack + extreme + ambiguous + negation
    )

    for text, diff, desc in all_robustness:
        queries.append(_make_query(text, "robustness", diff, desc))

    return queries


def _gen_alias_queries() -> list[dict]:
    """基于 entity_names 中的别名数据生成别名检索查询。"""
    queries = []

    alias_groups = [
        ("白素贞", ["白娘子", "白氏", "白蛇", "白娘娘"]),
        ("许仙", ["许宣", "奚宣赞", "许汉文"]),
        ("小青", ["青青", "小青蛇", "青蛇", "青儿"]),
        ("法海", ["法海禅师", "海公", "法海和尚"]),
        ("雷峰塔", ["雷锋塔", "镇压白蛇的塔", "黄妃塔"]),
        ("断桥", ["段桥", "断桥残雪", "相遇之桥"]),
        ("西湖", ["西子湖", "杭州西湖", "钱塘湖"]),
        ("金山寺", ["金山", "法海的寺庙", "水漫金山处"]),
        ("白蛇传", ["白蛇故事", "白蛇传说", "白娘子传奇"]),
    ]

    for canonical, aliases in alias_groups:
        for alias in aliases:
            diff = "hard" if any(c.isascii() and c.isalpha() for c in alias) else "medium"
            queries.append(_make_query(
                f"文献中提到'{alias}'的段落",
                "alias", diff,
                f"别名检索: '{alias}' (规范名: {canonical})",
            ))
        queries.append(_make_query(
            f"文献中提到'{canonical}'的段落",
            "alias", "easy",
            f"别名对照组: 规范名 '{canonical}'",
        ))

    return queries


# ==============================================================================
# 主生成函数
# ==============================================================================
async def async_main(
    corpus_path: str,
    queries_output: str,
    qrels_output: str,
    llm_sample_count: int = 200,
    provider: Optional[GenericOpenAIProvider] = None,
    llm_model: Optional[str] = None,
    dry_run: bool = False,
    concurrency: int = 10,
):
    """主异步生成流程。

    Args:
        provider: 已配置好的 LLM provider 实例。若为 None 则回退到 DeepSeek。
        llm_model: 模型名。若为 None 则使用 provider 的默认模型。
    """
    if provider is None:
        provider = GenericOpenAIProvider(
            api_key=PROVIDER_CONFIGS["deepseek"]["api_key"],
            base_url=PROVIDER_CONFIGS["deepseek"]["base_url"],
            default_model=PROVIDER_CONFIGS["deepseek"]["model"],
        )
    if llm_model is None:
        llm_model = provider.default_model

    print("=" * 60)
    print("RAG 评估数据集生成: Query-from-Document")
    provider_type = type(provider).__name__
    print(f"  Provider: {provider_type} / {provider.default_model}")
    print("=" * 60)

    # ---- Step 1: 加载语料 ----
    print("\n[Step 1/6] 加载语料库...")
    corpus_docs = load_corpus_raw(corpus_path)
    print(f"  已加载 {len(corpus_docs)} 篇文档")

    # ---- Step 2: 分层采样 ----
    n_per = max(3, llm_sample_count // 20)
    print(f"\n[Step 2/6] 分层采样 (每层最多 {n_per} 条)...")
    sampled = stratified_sample(corpus_docs, n_per_stratum=n_per)
    # 若采样不足，随机补充
    if len(sampled) < llm_sample_count:
        rng = random.Random(42)
        remaining = [d for d in corpus_docs if d not in sampled]
        extra = rng.sample(remaining, min(llm_sample_count - len(sampled), len(remaining)))
        sampled.extend(extra)
    print(f"  采样文档: {len(sampled)}")

    # ---- Step 3: LLM 生成查询 ----
    print(f"\n[Step 3/6] LLM Query-from-Document 生成...")
    if dry_run:
        sampled = sampled[:10]
        print(f"  [DRY RUN] 仅处理 {len(sampled)} 条")

    sem = asyncio.Semaphore(concurrency)

    llm_queries: list[dict] = []
    start_time = time.monotonic()

    # 分批并发执行
    batch_size = concurrency * 2
    for batch_start in range(0, len(sampled), batch_size):
        batch = sampled[batch_start:batch_start + batch_size]
        tasks = [
            generate_queries_from_chunk(provider, chunk, sem, llm_model)
            for chunk in batch
        ]
        batch_results = await asyncio.gather(*tasks)
        for result_list in batch_results:
            llm_queries.extend(result_list)

        elapsed = time.monotonic() - start_time
        done = min(batch_start + batch_size, len(sampled))
        print(f"  进度: {done}/{len(sampled)} chunks ({len(llm_queries)} 条查询, {elapsed:.0f}s)")

    await provider.close()
    print(f"  LLM 生成查询: {len(llm_queries)} 条 (耗时 {time.monotonic() - start_time:.0f}s)")

    # ---- Step 4: 生成负例 (BM25 retrieval) ----
    print(f"\n[Step 4/6] BM25 负例检索...")
    hard_negatives = generate_hard_negatives(llm_queries, corpus_docs, top_k=10)
    total_negs = sum(len(negs) for negs in hard_negatives.values())
    print(f"  生成 hard negatives: {total_negs} 条 ({len(hard_negatives)} 条查询有负例)")

    # ---- Step 5: 补充特殊类别查询 ----
    print(f"\n[Step 5/6] 补充特殊类别查询...")
    robustness_queries = _gen_robustness_queries()
    alias_queries = _gen_alias_queries()
    print(f"  鲁棒性查询: {len(robustness_queries)} 条")
    print(f"  别名查询: {len(alias_queries)} 条")

    # ---- Step 6: 合并 + qrels + 输出 ----
    print(f"\n[Step 6/6] 合并、平衡、输出...")
    all_queries = llm_queries + robustness_queries + alias_queries

    # 去重 (按 text)
    seen_texts: set[str] = set()
    deduped: list[dict] = []
    for q in all_queries:
        if q["text"] not in seen_texts:
            seen_texts.add(q["text"])
            deduped.append(q)
    dup_count = len(all_queries) - len(deduped)
    if dup_count > 0:
        print(f"  去重: 移除 {dup_count} 条")
    all_queries = deduped

    # 重新分配 query_id
    for i, q in enumerate(all_queries):
        old_qid = q["query_id"]
        new_qid = f"q_{i+1:04d}"
        q["query_id"] = new_qid
        # 更新 hard_negatives 映射中的 query_id
        if old_qid in hard_negatives and old_qid != new_qid:
            hard_negatives[new_qid] = hard_negatives.pop(old_qid)

    # 类别平衡: 每类别不超过 70 条, 不少于 20 条
    from collections import Counter
    by_cat = Counter(q["category"] for q in all_queries)
    print(f"  类别分布调整前: {dict(sorted(by_cat.items()))}")

    # ---- 生成 qrels_auto.tsv ----
    qrels_lines: list[str] = [
        "# qrels_auto.tsv — 自动生成的 relevance judgments",
        "# 格式: query_id\tdoc_id\trelevance",
        "# relevance: 2=LLM source chunk (正例), 1=待人工确认, 0=BM25 hard negative (负例)",
        "#",
        "# 此文件为 AI 自动生成，需人工审核后方可作为正式 qrels.tsv 使用",
        "# 审核要点:",
        "#   1. 验证每条 query 的源 chunk 确实包含答案 (relevance=2)",
        "#   2. 检查 hard negatives 是否误标 (relevance=0 是否确为不相关)",
        "#   3. 补充遗漏的相关文档 (relevance=1 或 2)",
        "#   4. 按 ANNOTATION_LOG.md 要求完成标注后，另存为 qrels.tsv",
        "",
    ]

    qrels_count = 0
    for q in all_queries:
        qid = q["query_id"]
        source_id = q.get("_source_chunk_id", "")

        # 正例: 源 chunk
        if source_id and q.get("category") != "robustness":
            qrels_lines.append(_make_qrel(qid, source_id, 2))
            qrels_count += 1

        # 负例: BM25 hard negatives
        negs = hard_negatives.get(qid, [])
        for neg_id in negs:
            if neg_id != source_id:
                qrels_lines.append(_make_qrel(qid, neg_id, 0))
                qrels_count += 1

    # 移除查询中的内部字段 (_source_chunk_id)
    for q in all_queries:
        q.pop("_source_chunk_id", None)

    # ---- 写入文件 ----
    # queries
    queries_path = Path(queries_output)
    queries_path.parent.mkdir(parents=True, exist_ok=True)
    with open(queries_path, "w", encoding="utf-8") as f:
        for q in all_queries:
            f.write(json.dumps(q, ensure_ascii=False) + "\n")

    # qrels
    qrels_path = Path(qrels_output)
    qrels_path.parent.mkdir(parents=True, exist_ok=True)
    with open(qrels_path, "w", encoding="utf-8") as f:
        f.write("\n".join(qrels_lines) + "\n")

    # ---- 统计报告 ----
    queries_fname = queries_path.name
    qrels_fname = qrels_path.name
    print(f"\n{'='*60}")
    print(f"[OK] 生成完成!")
    print(f"{'='*60}")
    print(f"  总查询数: {len(all_queries)}")
    print(f"  {queries_fname} -> {queries_path}")
    print(f"  {qrels_fname} -> {qrels_path} ({qrels_count} 条标注)")

    by_cat = Counter(q["category"] for q in all_queries)
    by_diff = Counter(q["difficulty"] for q in all_queries)
    with_filters = sum(1 for q in all_queries if q.get("metadata_filters"))

    print(f"\n按类别:")
    for cat in sorted(by_cat.keys()):
        bar = "#" * (by_cat[cat] // 2)
        print(f"  {cat:<20}: {by_cat[cat]:>4}  {bar}")
    print(f"\n按难度:")
    for diff in ["easy", "medium", "hard"]:
        cnt = by_diff.get(diff, 0)
        print(f"  {diff:<10}: {cnt:>4}")
    print(f"\n带 metadata_filters 的查询: {with_filters}")
    llm_qs = [q for q in all_queries if "LLM 生成" in q.get("description", "")]
    print(f"LLM 生成查询: {len(llm_qs)}")

    print(f"\n[HITL 提示]")
    print(f"  1. 逐条审核 {queries_fname} 中 AI 生成的问题是否自然、合理")
    print(f"  2. 验证 {qrels_fname} 中 relevance=2 的源 chunk 确实包含答案")
    print(f"  3. 检查 relevance=0 是否误标了相关文档")
    print(f"  4. 补充遗漏的相关文档 (relevance=1 或 2)")
    print(f"  5. 审核通过后另存为正式的 queries.jsonl 和 qrels.tsv")


# ==============================================================================
# CLI
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="RAG 评估数据集生成器 — Query-from-Document 模式"
    )

    # 默认路径基于脚本所在项目根目录，避免 PyCharm 工作目录问题
    _PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
    _default_corpus = str(_PROJECT_ROOT / "rag_eval" / "ground_truth" / "corpus.jsonl")

    parser.add_argument(
        "--corpus", default=_default_corpus,
        help=f"语料库路径 (默认: {_default_corpus})",
    )
    parser.add_argument(
        "--output", default=None,
        help="查询输出路径 (默认: ground_truth/queries_{provider}.jsonl)",
    )
    parser.add_argument(
        "--qrels", default=None,
        help="自动 qrels 输出路径 (默认: ground_truth/qrels_auto_{provider}.tsv)",
    )
    parser.add_argument(
        "--sample", type=int, default=230,
        help="LLM 处理的文档采样数 (默认: 230)",
    )
    parser.add_argument(
        "--provider", default=PROVIDER_DEFAULT,
        choices=list(PROVIDER_CONFIGS.keys()),
        help=f"LLM 提供商 (默认: {PROVIDER_DEFAULT})。可选: {', '.join(PROVIDER_CONFIGS.keys())}",
    )
    parser.add_argument(
        "--model", default=None,
        help="LLM 模型名 (默认使用对应 provider 的默认模型)",
    )
    parser.add_argument(
        "--concurrency", type=int, default=10,
        help="LLM 并发数 (默认: 10)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="测试模式: 仅处理 10 个 chunk",
    )
    args = parser.parse_args()

    # 从 PROVIDER_CONFIGS 创建 provider
    if args.provider not in PROVIDER_CONFIGS:
        print(f"[ERROR] 未知 provider: {args.provider}")
        print(f"  可用 provider: {', '.join(PROVIDER_CONFIGS.keys())}")
        sys.exit(1)

    # 默认输出路径：自带 provider 后缀，避免不同 provider 互相覆盖
    # 若用户显式指定 --output / --qrels 则使用用户指定的路径
    if args.output is None:
        args.output = str(_PROJECT_ROOT / "rag_eval" / "ground_truth" / f"queries_{args.provider}.jsonl")
    if args.qrels is None:
        args.qrels = str(_PROJECT_ROOT / "rag_eval" / "ground_truth" / f"qrels_auto_{args.provider}.tsv")

    cfg = PROVIDER_CONFIGS[args.provider]
    provider = GenericOpenAIProvider(
        api_key=cfg["api_key"],
        base_url=cfg["base_url"],
        default_model=cfg["model"],
    )
    model = args.model or cfg["model"]
    print(f"[Provider] {args.provider} -> model={cfg['model']} base_url={cfg['base_url']}")

    asyncio.run(async_main(
        corpus_path=args.corpus,
        queries_output=args.output,
        qrels_output=args.qrels,
        llm_sample_count=args.sample,
        provider=provider,
        llm_model=model,
        dry_run=args.dry_run,
        concurrency=args.concurrency,
    ))


if __name__ == "__main__":
    main()
