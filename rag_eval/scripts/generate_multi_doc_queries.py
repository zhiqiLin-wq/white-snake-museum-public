"""多文档 Query-from-Document 生成脚本。

与 generate_query_drafts.py (单 chunk → 1 条 query) 不同，
本脚本将 3-7 个连续相关 chunk 编为一组，让 LLM 基于整组文档生成
一条需要跨 chunk 综合信息才能回答的问题。
若 LLM 判断 chunk 间内容跨越太大不适合生成一个问题，可返回 skip 跳过该组。

优势:
    1. 每条 query 有 3-7 个正例 (relevance=2)，评估指标更有统计意义
    2. 问题需要综合多个 chunk，更贴近真实 RAG 检索场景
    3. 自动按 source_file + chapter_number + char_start 分组，保证语义连贯
    4. LLM 自主跳过不合适的组，避免硬凑低质量问题

分组逻辑:
    - 同一 source_file、同一 chapter_number 的 chunk 按 char_start 排序
    - 连续 3-7 个 chunk 编为一组
    - 跳过文本过短 (< 100 字符) 的 chunk
    - 跳过 chunk 数不足 3 的残余组
    - LLM 可返回 skip 跳过内容跨越过大的组

输出:
    rag_eval/ground_truth/queries_multi.jsonl
    rag_eval/ground_truth/qrels_multi.tsv

Usage:
    # 默认使用 DeepSeek
    python -m rag_eval.scripts.generate_multi_doc_queries

    # 切换 provider
    python -m rag_eval.scripts.generate_multi_doc_queries --provider qwen
    python -m rag_eval.scripts.generate_multi_doc_queries --provider ernie

    # 测试模式
    python -m rag_eval.scripts.generate_multi_doc_queries --dry-run --sample 10
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

# HF 配置：镜像 + 优先使用本地缓存
import os as _os
_os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
_os.environ.setdefault("OMP_NUM_THREADS", "4")   # PyTorch 2.13.0 在 Windows 上默认12线程加载大模型会 segfault
_os.environ.setdefault("MKL_NUM_THREADS", "4")   # 限制 OpenMP/MKL 线程数避免 C 层面段错误

from dotenv import load_dotenv
load_dotenv(AGENT_DIR / ".env")

from server.utils.entity_names import LOCATION_ALIASES, CHARACTER_NAMES
from server.llm.openai_compatible_provider import GenericOpenAIProvider

# ==============================================================================
# LLM Provider 配置（api_key 从 agent/.env 环境变量读取，禁止硬编码）
# ==============================================================================
PROVIDER_CONFIGS = {
    "deepseek": {
        "api_key": _os.getenv("DEEPSEEK_API_KEY", ""),
        "base_url": "https://api.deepseek.com/v1",
        "model": "deepseek-chat",
    },
    "qwen": {
        "api_key": _os.getenv("DASHSCOPE_API_KEY", ""),
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "model": "qwen3.8-max",
    },
    "ernie": {
        "api_key": _os.getenv("QIANFAN_API_KEY", ""),
        "base_url": "https://qianfan.baidubce.com/v2",
        "model": "ernie-4.0-turbo-8k-latest",
    },
}
PROVIDER_DEFAULT = "deepseek"

# 生成查询时允许的类别 (plot 排除：连续 chunk 天然适合 plot，但我们要多样性)
ALLOWED_CATEGORIES = [
    "entity", "location", "comparison", "motif",
    "filtered", "intertext", "alias", "robustness",
    "plot",  # 情节检索 —— 连续 chunk 天然适合，配额控制占比
]
CATEGORY_LABELS = {
    "entity": "实体检索",
    "location": "地理查询",
    "comparison": "跨文献比较",
    "motif": "主题/母题分析",
    "filtered": "带过滤条件检索",
    "intertext": "互文/典故检索",
    "alias": "别名/异名检索",
    "robustness": "鲁棒性测试",
    "plot": "情节检索",
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
    source_chunk_ids: Optional[list[str]] = None,
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
        "_source_chunk_ids": source_chunk_ids or [],
    }


def _make_qrel(query_id: str, doc_id: str, relevance: int) -> str:
    return f"{query_id}\t{doc_id}\t{relevance}"


# ==============================================================================
# Phase 1: 语料库加载 + 连续 chunk 分组
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
            docs.append(json.loads(line))
    return docs


def _sliding_window_groups(
    bucket: list[dict],
    min_group_size: int,
    max_group_size: int,
    rng: random.Random,
) -> list[list[dict]]:
    """对单个桶内的有序文档执行滑动窗口分组。"""
    groups: list[list[dict]] = []
    n = len(bucket)
    if n < min_group_size:
        return groups

    stride = max(max_group_size // 3, 1)
    pos = 0
    while pos + min_group_size <= n:
        end = min(pos + rng.randint(min_group_size, max_group_size), n)
        end = min(end, pos + max_group_size)
        if end - pos >= min_group_size:
            groups.append(bucket[pos:end])
        pos += stride
    return groups


def _group_key_from_ids(group: list[dict]) -> str:
    """生成组的唯一标识 (排序后的 id 列表)，用于跨策略去重。"""
    ids = sorted(d.get("doc_id", "") for d in group)
    return "|".join(ids)


def group_all_strategies(
    docs: list[dict],
    min_group_size: int = 3,
    max_group_size: int = 7,
    min_chunk_chars: int = 100,
    seed: int = 42,
) -> list[list[dict]]:
    """多策略分组: source_chapter + genre_dynasty + cross_chapter。

    三种策略互补:
        strategy 1 (source_chapter): 同文献同章节连续 chunk — 适合 motif/entity/location
        strategy 2 (genre_dynasty):  同体裁同朝代跨文献 chunk — 适合 comparison/intertext
        strategy 3 (cross_chapter):  同文献跨章节边界 chunk — 适合叙事弧线/情节发展

    最后按 chunk_id 集合去重，随机打乱。

    Returns:
        [[doc1, doc2, ...], ...]
    """
    rng = random.Random(seed)
    all_groups: list[list[dict]] = []
    seen_keys: set[str] = set()  # 去重用

    # 预过滤短文本
    valid_docs = [d for d in docs if len(d.get("text", "").strip()) >= min_chunk_chars]
    skipped_short = len(docs) - len(valid_docs)
    if skipped_short > 0:
        print(f"  [过滤] 跳过 {skipped_short} 篇短文本 (< {min_chunk_chars} chars)")

    def _add_groups(groups: list[list[dict]], strategy_name: str):
        nonlocal all_groups
        added = 0
        dup = 0
        for g in groups:
            key = _group_key_from_ids(g)
            if key not in seen_keys:
                seen_keys.add(key)
                all_groups.append(g)
                added += 1
            else:
                dup += 1
        sizes = Counter(len(g) for g in groups)
        print(f"  [{strategy_name}] 生成 {len(groups)} 组 (新增 {added}, 重复 {dup}), "
              f"分布: {dict(sorted(sizes.items()))}")

    # ========================================================================
    # Strategy 1: source_file × chapter_number (同文献同章节连续 chunk)
    # ========================================================================
    s1_buckets: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for doc in valid_docs:
        meta = doc.get("metadata", {})
        key = (meta.get("source_file", "unknown"), meta.get("chapter_number", "unknown"))
        s1_buckets[key].append(doc)

    s1_groups: list[list[dict]] = []
    for key, bucket in sorted(s1_buckets.items()):
        bucket.sort(key=lambda d: d.get("metadata", {}).get("char_start", 0))
        s1_groups.extend(_sliding_window_groups(bucket, min_group_size, max_group_size, rng))
    _add_groups(s1_groups, "source_chapter")

    # ========================================================================
    # Strategy 2: genre × dynasty (同体裁同朝代，跨文献)
    # 研究文献 dynasty 为 None，统一归入 "_research"
    # ========================================================================
    s2_buckets: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for doc in valid_docs:
        meta = doc.get("metadata", {})
        genre = meta.get("genre", "unknown")
        dynasty = meta.get("dynasty") or "_research"
        s2_buckets[(genre, dynasty)].append(doc)

    s2_groups: list[list[dict]] = []
    for key, bucket in sorted(s2_buckets.items()):
        # 按 (source_file, chapter_number, char_start) 排序，
        # 同体裁同朝代的相邻文献被排在一起
        bucket.sort(key=lambda d: (
            d.get("metadata", {}).get("source_file", ""),
            d.get("metadata", {}).get("chapter_number", ""),
            d.get("metadata", {}).get("char_start", 0),
        ))
        s2_groups.extend(_sliding_window_groups(bucket, min_group_size, max_group_size, rng))
    _add_groups(s2_groups, "genre_dynasty")

    # ========================================================================
    # Strategy 3: cross_chapter (同文献跨章节边界)
    # 取 source_file 内相邻两章的衔接处: 前章末尾 1~2 chunk + 后章开头 1~2 chunk
    # ========================================================================
    s3_groups: list[list[dict]] = []
    # 按 source_file 分组，找出所有章节对
    source_chapters: dict[str, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for doc in valid_docs:
        meta = doc.get("metadata", {})
        sf = meta.get("source_file", "unknown")
        ch = meta.get("chapter_number", "unknown")
        source_chapters[sf][ch].append(doc)

    for sf, chapters in source_chapters.items():
        # 每章内按 char_start 排序
        for ch in chapters:
            chapters[ch].sort(key=lambda d: d.get("metadata", {}).get("char_start", 0))

        sorted_chapters = sorted(chapters.keys())
        for i in range(len(sorted_chapters) - 1):
            ch_a = sorted_chapters[i]
            ch_b = sorted_chapters[i + 1]
            bucket_a = chapters[ch_a]
            bucket_b = chapters[ch_b]

            # 取前章末尾 1-2 chunk + 后章开头 1-2 chunk
            # 至少总共 min_group_size 个 chunk
            for take_a in range(1, min(3, len(bucket_a) + 1)):
                for take_b in range(1, min(3, len(bucket_b) + 1)):
                    if take_a + take_b < min_group_size:
                        continue
                    if take_a + take_b > max_group_size:
                        continue
                    tail = bucket_a[-take_a:]
                    head = bucket_b[:take_b]
                    combined = tail + head
                    if len(combined) >= min_group_size:
                        s3_groups.append(combined)

    _add_groups(s3_groups, "cross_chapter")

    # ========================================================================
    # 全局打乱
    # ========================================================================
    rng.shuffle(all_groups)

    group_sizes = Counter(len(g) for g in all_groups)
    print(f"\n[分组总计] {len(docs)} 篇文档 -> {len(all_groups)} 组 (去重后)")
    print(f"  组大小分布: {dict(sorted(group_sizes.items()))}")
    print(f"  策略贡献: source_chapter + genre_dynasty + cross_chapter")

    return all_groups


# ==============================================================================
# Phase 2: LLM 多文档问题生成
# ==============================================================================
MULTI_DOC_QUERY_SYSTEM = """你是一位中国古典文学研究专家，正在为一个 RAG 检索评估系统生成测试查询。

你的任务: 阅读以下 **{n_chunks} 个连续的文本块**（来自同一文献的相邻段落），
生成 **1 个** 需要综合理解这多个文本块才能完整回答的自然语言问题。

**必须满足的类别要求**: 本轮你必须生成一个 **{category_label}（{category}）** 类型的问题。

要求:
1. 问题的答案**分散在至少 2-3 个不同的文本块中**，需要综合各块的信息
2. 不能是仅靠某一个文本块就能完全回答的问题
3. 问题自然、流畅，像一个真实研究者会提出的问题
4. 问题应该有明确的检索价值（不能太宽泛如"这段讲了什么"）

你需要输出一个 JSON 对象，包含以下字段:
- "text": 生成的查询问题文本 (字符串)
- "category": 必须固定为 **"{category}"**
- "difficulty": 难度等级:
    "easy"   — 信息在多块中直接可见，只需简单关联
    "medium" — 需要一定的跨块理解和信息整合
    "hard"   — 需要深度分析、推理或多个维度的综合比较
- "reasoning": 简短说明问题需要哪些文本块的哪些信息，以及为何适合 {category_label} 类别 (2-3 句话)

**关键约束**:
- category 字段必须严格等于 "{category}"，不能使用其他类别。
- 严禁使用 "plot" 作为 category。即使文本内容是叙事性的，也要从 {category_label} 的角度设计。
- 避免生成仅靠常识就能回答的问题。

**chunk 内容关联度判断**: 如果这些文本块无法合理地生成一个 {category_label} 类的问题，
请输出以下 JSON 跳过该组:
{{"skip": true, "reason": "简短说明为何这些文本块不适合生成 {category_label} 类问题"}}

输出格式: 仅输出 JSON 对象，不要有其他文字。
正常示例: {{"text": "...", "category": "{category}", "difficulty": "hard", "reasoning": "..."}}"""


def _build_multi_doc_prompt(chunks: list[dict], target_category: str = "comparison") -> str:
    """构建多文档 query 生成的 user prompt。"""
    meta0 = chunks[0].get("metadata", {})
    source_type = meta0.get("source_type", "未知")
    genre = meta0.get("genre", "未知")
    dynasty = meta0.get("dynasty", "未知")
    source_file = meta0.get("source_file", "未知")
    label = CATEGORY_LABELS.get(target_category, target_category)

    chunk_texts = []
    for i, chunk in enumerate(chunks):
        doc_id = chunk.get("doc_id", f"chunk_{i}")
        text = chunk.get("text", "")
        chunk_texts.append(f"--- 文本块 {i + 1} (ID: {doc_id}) ---\n{text[:2000]}")

    all_chunks_text = "\n\n".join(chunk_texts)

    return f"""以下 {len(chunks)} 个文本块来自白蛇传文献，属于{dynasty}代，体裁为{genre}，来源类型为{source_type}。
来源文件: {source_file}

请仔细阅读所有文本块，然后生成一个 **{label}** 类别的问题。记住: category 必须为 "{target_category}"。

{all_chunks_text}

请仅输出 JSON，category="{target_category}"。"""


async def generate_query_from_group(
    provider: GenericOpenAIProvider,
    group: list[dict],
    sem: asyncio.Semaphore,
    model: Optional[str] = None,
    target_category: str = "comparison",
) -> Optional[dict]:
    """对一组 chunk 调用 LLM 生成 1 条综合问题。

    Args:
        target_category: 强制指定的查询类别 (ALLOWED_CATEGORIES 之一)

    返回: query_dict (成功), None (失败或 LLM 判断应跳过)
    """
    label = CATEGORY_LABELS.get(target_category, target_category)

    async with sem:
        try:
            system_prompt = MULTI_DOC_QUERY_SYSTEM.format(
                n_chunks=len(group),
                category=target_category,
                category_label=label,
            )
            response = await provider.generate(
                system=system_prompt,
                user=_build_multi_doc_prompt(group, target_category),
                model=model,
                max_tokens=1024,
                temperature=0.85,
            )
        except Exception as e:
            chunk_ids = [c.get("doc_id", "?") for c in group]
            print(f"  [LLM ERROR] group={chunk_ids}: {e}")
            return None

    content = response.content.strip()

    # 清理 markdown 代码块标记
    if content.startswith("```"):
        lines = content.split("\n")
        content = "\n".join(lines[1:]) if len(lines) > 1 else content[3:]
    if content.endswith("```"):
        content = content[:-3].strip()

    # 解析 JSON
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError:
        m = re.search(r'\{.*\}', content, re.DOTALL)
        if not m:
            print(f"  [PARSE ERROR] group chunks: {[c.get('doc_id','?') for c in group[:2]]}...: {content[:120]}")
            return None
        try:
            parsed = json.loads(m.group())
        except json.JSONDecodeError:
            print(f"  [PARSE ERROR] group chunks: {[c.get('doc_id','?') for c in group[:2]]}...: {content[:120]}")
            return None

    # LLM 自主跳过: chunk 内容关联度不够
    if parsed.get("skip"):
        reason = parsed.get("reason", "")
        chunk_ids = [c.get("doc_id", "?") for c in group]
        print(f"  [SKIP] group={chunk_ids[:2]}...({len(group)} chunks): {reason}")
        return None

    text = parsed.get("text", "").strip()
    if not text:
        return None

    category = parsed.get("category", "entity")
    if category not in CATEGORIES:
        category = "entity"

    difficulty = parsed.get("difficulty", "medium")
    if difficulty not in ("easy", "medium", "hard"):
        difficulty = "medium"

    source_chunk_ids = [c.get("doc_id", "") for c in group]
    meta0 = group[0].get("metadata", {})

    # 自动确定 metadata_filters
    metadata_filters = None
    expected_genre_scope = "all"
    if category == "filtered":
        dynasty = meta0.get("dynasty")
        genre = meta0.get("genre")
        if dynasty and genre:
            metadata_filters = {"dynasty": dynasty, "genre": genre}
            expected_genre_scope = genre
        elif dynasty:
            metadata_filters = {"dynasty": dynasty}
        elif genre:
            metadata_filters = {"genre": genre}
            expected_genre_scope = genre

    return _make_query(
        text=text,
        category=category,
        difficulty=difficulty,
        description=f"LLM 多文档生成自 chunks {source_chunk_ids}: {parsed.get('reasoning', '')}",
        metadata_filters=metadata_filters,
        expected_genre_scope=expected_genre_scope,
        source_chunk_ids=source_chunk_ids,
    )


# ==============================================================================
# Phase 3.5: Reranker + Reasoning 分级 relevance 标注
# ==============================================================================
def _parse_core_chunk_indices(reasoning: str, n_chunks: int) -> set[int]:
    """从 LLM reasoning 文本中解析被点名包含核心信息的文本块索引 (1-based)。

    支持的表达模式:
        - "文本块1" "文本块3" "文本块1-3" "文本块1、3、5"
        - "块1" "块2和块3" "第1个文本块"
        - "chunk 1" "chunks 2 and 3"

    Returns:
        {1-based chunk index, ...}
    """
    import re
    indices: set[int] = set()

    # 模式: "文本块N" 或 "文本块 N"
    for m in re.finditer(r'文本块\s*(\d+)', reasoning):
        idx = int(m.group(1))
        if 1 <= idx <= n_chunks:
            indices.add(idx)

    # 模式: "第N个文本块"
    for m in re.finditer(r'第\s*(\d+)\s*个文本块', reasoning):
        idx = int(m.group(1))
        if 1 <= idx <= n_chunks:
            indices.add(idx)

    # 模式: "块N" (但不匹配 "文本块" 以免重复计数; 去掉了 (?<!\w) 因为 Python3 \w 包含中文)
    # 先去除已被 "文本块N" 匹配过的文本，再匹配独立的 "块N"
    for m in re.finditer(r'(?<!文本)块\s*(\d+)', reasoning):
        idx = int(m.group(1))
        if 1 <= idx <= n_chunks:
            indices.add(idx)

    # 模式: "chunk N" "chunks N" "block N" "blocks N"
    for m in re.finditer(r'(?:chunks?|blocks?)\s*(\d+)', reasoning, re.IGNORECASE):
        idx = int(m.group(1))
        if 1 <= idx <= n_chunks:
            indices.add(idx)

    return indices


class _DirectReranker:
    """轻量 Reranker 封装 — 主线程同步加载 CrossEncoder，避免后台线程 hang。"""

    def __init__(self, model_name: str = "BAAI/bge-reranker-base"):
        from sentence_transformers import CrossEncoder
        print(f"  正在加载 CrossEncoder: {model_name} ...")
        # local_files_only=True: 模型已缓存则跳过网络请求，避免 hang
        self._model = CrossEncoder(model_name, local_files_only=True)
        print(f"  CrossEncoder 加载完成")

    def rerank(self, query: str, documents: list[str]) -> list[tuple[int, float]]:
        if not documents:
            return []
        pairs = [(query, doc) for doc in documents]
        scores = self._model.predict(pairs, batch_size=32, show_progress_bar=False)
        scored = [(i, float(scores[i])) for i in range(len(scores))]
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored


def _load_reranker_for_grading() -> Optional[object]:
    """加载 Reranker 模型用于 relevance 分级（主线程同步加载）。

    Returns:
        _DirectReranker 实例，加载失败返回 None
    """
    try:
        return _DirectReranker("BAAI/bge-reranker-base")
    except Exception as e:
        print(f"  [WARN] Reranker 加载失败: {e}，回退到统一 relevance=2")
        return None


def _grade_relevance_with_reranker(
    queries: list[dict],
    source_groups_map: dict[str, list[dict]],
    reranker,
) -> dict[str, dict[str, int]]:
    """用 LLM Reasoning (主) + Reranker (底线过滤) 对 source chunks 分级标注。

    核心原则: **LLM reasoning 做主信号，Reranker 只做底线过滤**。
    这样 relevance 标注源 (LLM) 与检索系统组件 (Reranker) 完全独立，避免循环评估。

    分级规则:
        1. 解析 LLM reasoning，提取被点名包含"核心信息"的 chunk
        2. Reranker 对所有 source chunks 打分
        3. 映射:
           - LLM reasoning 点名核心                            → relevance=2
           - 其余 source chunks (reranker >= 0.1，大概率相关)    → relevance=1
           - reranker < 0.1 (极低分，可能 LLM 标错了)           → relevance=0

    底线过滤:
        - 如果被 LLM 标记为"核心"的 chunk，Reranker 评分 < 0.05，
          打印警告但不改标注 → 供人工审核时重点关注

    Returns:
        {query_id: {chunk_id: relevance_grade}}
    """
    graded: dict[str, dict[str, int]] = {}
    n_core2 = 0      # LLM reasoning 点名 → relevance=2
    n_support1 = 0    # 其余 source chunk → relevance=1
    n_dropped = 0     # reranker < 0.1 → relevance=0
    n_warn = 0        # reasoning 核心但 reranker < 0.05 (需人工审核)

    for q in queries:
        qid = q["query_id"]
        query_text = q["text"]
        reasoning = q.get("description", "")
        source_group = source_groups_map.get(qid, [])

        if not source_group:
            graded[qid] = {}
            continue

        n = len(source_group)
        chunk_texts = [c.get("text", "") for c in source_group]
        chunk_ids = [c.get("doc_id", "") for c in source_group]

        # Step 1: 解析 reasoning 中的核心 chunk 引用 (1-based index)
        core_indices = _parse_core_chunk_indices(reasoning, n)

        # Step 2: Reranker 批量打分 (仅用于底线过滤)
        score_map: dict[int, float] = {}
        try:
            scored = reranker.rerank(query_text, chunk_texts)
            score_map = {idx: s for idx, s in scored}
        except Exception as e:
            print(f"  [RERANK WARN] query={qid}: {e}")
            # 回退: reasoning 核心 → 2, 其余 → 1
            for i, cid in enumerate(chunk_ids):
                if (i + 1) in core_indices:
                    graded.setdefault(qid, {})[cid] = 2
                else:
                    graded.setdefault(qid, {})[cid] = 1
            continue

        # Step 3: 映射分级
        grades: dict[str, int] = {}
        for i, cid in enumerate(chunk_ids):
            rerank_score = score_map.get(i, 0.0)
            is_core = (i + 1) in core_indices

            if is_core:
                grades[cid] = 2
                n_core2 += 1
                # 底线检查: 如果 LLM 说核心但 Reranker 极低分
                if rerank_score < 0.05:
                    n_warn += 1
                    if n_warn <= 3:  # 只打印前 3 条
                        print(f"  [REVIEW] query={qid} chunk={cid}: LLM标记为核心但reranker={rerank_score:.4f}")
            elif rerank_score >= 0.1:
                grades[cid] = 1
                n_support1 += 1
            else:
                grades[cid] = 0
                n_dropped += 1

        graded[qid] = grades

    print(f"  Relevance 分级完成 (LLM reasoning 为主，Reranker 底线过滤):")
    print(f"    relevance=2 (LLM reasoning 核心): {n_core2}")
    print(f"    relevance=1 (辅助上下文):         {n_support1}")
    print(f"    relevance=0 (Reranker < 0.1 剔除): {n_dropped}")
    if n_warn > 0:
        print(f"    [WARN] 需人工审核: {n_warn} 个LLM标记核心但Reranker极低分的chunk")

    return graded


# ==============================================================================
# Phase 4: 负例生成 (BM25 retrieval)
# ==============================================================================
def generate_hard_negatives(
    queries: list[dict],
    corpus_docs: list[dict],
    top_k: int = 15,
) -> dict[str, list[str]]:
    """对每条 query 使用 BM25 检索 hard negatives。

    与单文档版本不同: 排除所有 source_chunk_ids (而不仅是单个)。

    Returns:
        {query_id: [neg_doc_id, ...]}
    """
    from server.rag.bm25_retriever import BM25Retriever

    bm25 = BM25Retriever()
    docs_for_index = [
        {"chunk_id": doc["doc_id"], "content": doc.get("text", ""), "metadata": doc.get("metadata", {})}
        for doc in corpus_docs
    ]
    print(f"  构建 BM25 索引 ({len(docs_for_index)} 文档)...")
    bm25.index(docs_for_index)
    print(f"  BM25 索引构建完成")

    hard_negatives: dict[str, list[str]] = {}
    for i, q in enumerate(queries):
        query_id = q["query_id"]
        query_text = q["text"]
        source_ids = set(q.get("_source_chunk_ids", []))

        try:
            results = bm25.search(query_text, top_k=top_k + len(source_ids) + 5)
        except Exception as e:
            print(f"  [BM25 WARN] query={query_id}: {e}")
            hard_negatives[query_id] = []
            continue

        negs = []
        for r in results:
            chunk_id = r.get("chunk_id", "")
            if chunk_id and chunk_id not in source_ids:
                negs.append(chunk_id)
            if len(negs) >= top_k:
                break

        hard_negatives[query_id] = negs
        if (i + 1) % 50 == 0:
            print(f"  BM25 负例进度: {i+1}/{len(queries)}")

    return hard_negatives


# ==============================================================================
# Phase 4: 补充特殊类别查询 (同单文档版本)
# ==============================================================================
def _gen_robustness_queries(count: int = 50) -> list[dict]:
    """生成鲁棒性测试查询。"""
    queries = []
    zero_relevant = [
        ("白蛇传中提到了外星人吗", "hard", "zero-relevant"),  # +7 more in real
        ("白素贞的手机号码是多少", "hard", "zero-relevant"),
        ("人工智能在白蛇传中的应用", "hard", "zero-relevant"),
        ("白蛇传中的量子力学原理", "hard", "zero-relevant"),
        ("白素贞会不会开汽车", "hard", "zero-relevant"),
        ("许仙的电子邮件地址", "hard", "zero-relevant"),
        ("法海的微信头像", "hard", "zero-relevant"),
        ("白蛇传中有没有飞机", "hard", "zero-relevant"),
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
        ("白蛇", "easy", "超短查询"),
        ("法海", "easy", "超短查询"),
        ("雷峰塔", "easy", "超短查询"),
        ("白素贞与许仙在断桥相遇后共同经历了许多波折，包括端阳现形、盗取仙草、水漫金山，最终却被法海镇压在雷峰塔下，这个故事究竟传达了什么样的爱情观和命运观", "medium", "超长查询"),
    ]
    ambiguous = [
        ("塔的故事", "hard", "模糊查询"),
        ("和尚斗蛇", "hard", "极度简略模糊查询"),
        ("杭州的爱情故事", "hard", "地理+主题模糊查询"),
        ("白蛇的故事", "hard", "泛指白蛇相关所有内容"),
        ("中国民间故事蛇", "hard", "泛指性查询"),
    ]
    negation = [
        ("白蛇传中不涉及爱情的段落", "hard", "否定式查询"),
        ("除白素贞外的女性角色描写", "hard", "排除式查询"),
        ("非佛教视角下的白蛇传解读", "hard", "排除式查询"),
        ("不包含法海的白蛇传段落", "hard", "排除式查询"),
    ]

    all_robustness = zero_relevant + noisy + typo + multilingual + keyword_stack + extreme + ambiguous + negation
    for text, diff, desc in all_robustness:
        queries.append(_make_query(text, "robustness", diff, desc))
    return queries


def _gen_alias_queries() -> list[dict]:
    """基于实体别名生成别名检索查询。"""
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
            queries.append(_make_query(
                f"文献中提到'{alias}'的段落",
                "alias", "medium",
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
    provider: Optional[GenericOpenAIProvider] = None,
    llm_model: Optional[str] = None,
    min_group_size: int = 3,
    max_group_size: int = 7,
    target_groups: Optional[int] = None,
    dry_run: bool = False,
    concurrency: int = 10,
    seed: int = 42,
):
    """主异步生成流程。"""
    if provider is None:
        provider = GenericOpenAIProvider(
            api_key=PROVIDER_CONFIGS["deepseek"]["api_key"],
            base_url=PROVIDER_CONFIGS["deepseek"]["base_url"],
            default_model=PROVIDER_CONFIGS["deepseek"]["model"],
        )
    if llm_model is None:
        llm_model = provider.default_model

    print("=" * 60)
    print("RAG 评估数据集生成: 多文档 Query-from-Document")
    print(f"  Provider: {type(provider).__name__} / {llm_model}")
    print(f"  组大小: {min_group_size}-{max_group_size} chunks/组")
    print("=" * 60)

    # ---- Step 1: 加载语料 ----
    print("\n[Step 1/6] 加载语料库...")
    corpus_docs = load_corpus_raw(corpus_path)
    print(f"  已加载 {len(corpus_docs)} 篇文档")

    # ---- Step 2: 多策略 chunk 分组 ----
    print(f"\n[Step 2/6] 多策略 chunk 分组 (source_chapter + genre_dynasty + cross_chapter)...")
    groups = group_all_strategies(
        corpus_docs,
        min_group_size=min_group_size,
        max_group_size=max_group_size,
        seed=seed,
    )
    print(f"  得到 {len(groups)} 组 (去重后)")

    # 限制目标组数
    if target_groups is not None and len(groups) > target_groups:
        rng = random.Random(seed)
        groups = rng.sample(groups, target_groups)
        print(f"  限制目标组数: {len(groups)}")

    # ---- Step 3: LLM 多文档生成 ----
    print(f"\n[Step 3/6] LLM 多文档 Query 生成...")
    if dry_run:
        groups = groups[:10]
        print(f"  [DRY RUN] 仅处理 {len(groups)} 组")

    # ---- 类别配额轮换队列 ----
    # 每组预分配一个目标类别，按轮换方式均匀覆盖
    category_queue: list[str] = []
    for i in range(len(groups)):
        category_queue.append(ALLOWED_CATEGORIES[i % len(ALLOWED_CATEGORIES)])
    # 打乱队列顺序（保持每类别数量均匀，但消除了固定的轮换顺序偏差）
    rng = random.Random(seed + 1)  # 独立种子，不影响分组
    # 用 Fisher-Yates 在保持每 N 个一组均匀的前提下有限打乱
    # 简单策略: 将队列分成多轮 (每 8 个一组)，每轮内部打乱
    for round_start in range(0, len(category_queue), len(ALLOWED_CATEGORIES)):
        round_end = min(round_start + len(ALLOWED_CATEGORIES), len(category_queue))
        round_slice = category_queue[round_start:round_end]
        rng.shuffle(round_slice)
        category_queue[round_start:round_end] = round_slice

    print(f"  类别配额: {dict(sorted(Counter(category_queue).items()))}")

    sem = asyncio.Semaphore(concurrency)
    llm_queries: list[dict] = []
    dropped_groups: int = 0  # LLM skip + error + parse failure
    generated_cats: Counter = Counter()
    dropped_cats: Counter = Counter()
    start_time = time.monotonic()

    batch_size = concurrency * 2
    for batch_start in range(0, len(groups), batch_size):
        batch = groups[batch_start:batch_start + batch_size]
        batch_cats = category_queue[batch_start:batch_start + batch_size]
        tasks = [
            generate_query_from_group(provider, group, sem, llm_model, target_category=cat)
            for group, cat in zip(batch, batch_cats)
        ]
        batch_results = await asyncio.gather(*tasks)
        for i, result in enumerate(batch_results):
            if result is not None:
                llm_queries.append(result)
                cat = result.get("category", "unknown")
                generated_cats[cat] += 1
            else:
                dropped_groups += 1
                dropped_cats[batch_cats[i]] += 1

        elapsed = time.monotonic() - start_time
        done = min(batch_start + batch_size, len(groups))
        print(f"  进度: {done}/{len(groups)} groups ({len(llm_queries)} 条查询, {dropped_groups} 组丢弃, {elapsed:.0f}s)")

    await provider.close()
    print(f"  LLM 生成查询: {len(llm_queries)} 条, 丢弃: {dropped_groups} 组 (耗时 {time.monotonic() - start_time:.0f}s)")
    print(f"  生成类别分布: {dict(sorted(generated_cats.items()))}")
    if dropped_cats:
        print(f"  丢弃类别分布: {dict(sorted(dropped_cats.items()))}")

    # ---- Step 3.5: Reranker 分级 relevance 标注 ----
    print(f"\n[Step 3.5/6] Reranker 分级 relevance 标注...")
    # 构建 doc_id → doc 的快速查找表
    doc_lookup: dict[str, dict] = {d["doc_id"]: d for d in corpus_docs}
    # 构建 query_id → source chunks 映射
    source_groups_map: dict[str, list[dict]] = {}
    for q in llm_queries:
        qid = q["query_id"]
        source_ids = q.get("_source_chunk_ids", [])
        source_chunks = []
        for sid in source_ids:
            if sid in doc_lookup:
                source_chunks.append(doc_lookup[sid])
        source_groups_map[qid] = source_chunks

    reranker = _load_reranker_for_grading()
    if reranker is not None:
        relevance_grades = _grade_relevance_with_reranker(
            llm_queries, source_groups_map, reranker
        )
    else:
        # 回退: 全部 source chunks 标 relevance=2
        relevance_grades = {}
        for q in llm_queries:
            qid = q["query_id"]
            relevance_grades[qid] = {sid: 2 for sid in q.get("_source_chunk_ids", [])}
        print("  回退到统一 relevance=2")

    # ---- Step 4: 生成负例 (BM25 retrieval) ----
    print(f"\n[Step 4/6] BM25 负例检索...")
    hard_negatives = generate_hard_negatives(llm_queries, corpus_docs, top_k=15)
    total_negs = sum(len(negs) for negs in hard_negatives.values())
    print(f"  生成 hard negatives: {total_negs} 条 ({len(hard_negatives)} 条查询有负例)")

    # ---- Step 5: 补充特殊类别查询 ----
    print(f"\n[Step 5/6] 补充特殊类别查询...")
    robustness_queries = _gen_robustness_queries()
    alias_queries = _gen_alias_queries()
    print(f"  鲁棒性查询: {len(robustness_queries)} 条")
    print(f"  别名查询: {len(alias_queries)} 条")

    # ---- Step 6: 合并 + qrels + 输出 ----
    print(f"\n[Step 6/6] 合并、去重、输出...")
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
    old_to_new_qid: dict[str, str] = {}
    for i, q in enumerate(all_queries):
        old_qid = q["query_id"]
        new_qid = f"q_{i+1:04d}"
        q["query_id"] = new_qid
        old_to_new_qid[old_qid] = new_qid

    # 更新 hard_negatives 映射
    new_hard_negatives: dict[str, list[str]] = {}
    for old_qid, negs in hard_negatives.items():
        if old_qid in old_to_new_qid:
            new_hard_negatives[old_to_new_qid[old_qid]] = negs
    hard_negatives = new_hard_negatives

    # 类别分布
    by_cat = Counter(q["category"] for q in all_queries)
    print(f"  类别分布: {dict(sorted(by_cat.items()))}")

    # ---- 生成 qrels_multi.tsv ----
    qrels_lines: list[str] = [
        "# qrels_multi.tsv — 多文档自动生成的 relevance judgments (分级版)",
        "# 格式: query_id\tdoc_id\trelevance",
        "# relevance: 2=Reranker+Reasoning 核心答案块, 1=辅助上下文块, 0=BM25 hard negative (负例)",
        "#",
        "# 分级方法: Reranker (bge-reranker-base) 对 (query, chunk) 打分 + LLM reasoning 解析",
        "#   - 被 reasoning 点名核心信息 + reranker >= 0.2  → relevance=2",
        "#   - reranker >= 0.5 (高度相关)                   → relevance=2",
        "#   - reranker >= 0.2 非核心                       → relevance=1",
        "#   - reranker < 0.2                              → 剔除 (不作为正例)",
        "#",
        "# 此文件为 AI 自动生成，需人工审核后方可作为正式 qrels 使用",
        "# 审核要点:",
        "#   1. 验证 relevance=2 的 chunk 确实包含答案核心信息",
        "#   2. 检查 relevance=1 是否合理（辅助上下文 vs 不应相关）",
        "#   3. 检查 hard negatives (relevance=0) 是否误标了相关文档",
        "#   4. 按 ANNOTATION_LOG.md 要求完成标注后，另存为正式 qrels",
        "",
    ]

    qrels_count = 0
    pos2_count = 0
    pos1_count = 0
    neg_count = 0

    for q in all_queries:
        qid = q["query_id"]
        source_ids = q.get("_source_chunk_ids", [])
        grades = relevance_grades.get(qid, {})

        # 正例: 分级 relevance (2 或 1)
        if source_ids and q.get("category") != "robustness":
            for sid in source_ids:
                rel = grades.get(sid, 2)  # 默认 2 (回退)
                if rel > 0:
                    qrels_lines.append(_make_qrel(qid, sid, rel))
                    qrels_count += 1
                    if rel == 2:
                        pos2_count += 1
                    else:
                        pos1_count += 1

        # 负例: BM25 hard negatives
        negs = hard_negatives.get(qid, [])
        for neg_id in negs:
            if neg_id not in source_ids:
                qrels_lines.append(_make_qrel(qid, neg_id, 0))
                qrels_count += 1
                neg_count += 1

    # 移除内部字段
    for q in all_queries:
        q.pop("_source_chunk_ids", None)

    # ---- 写入文件 ----
    queries_path = Path(queries_output)
    queries_path.parent.mkdir(parents=True, exist_ok=True)
    with open(queries_path, "w", encoding="utf-8") as f:
        for q in all_queries:
            f.write(json.dumps(q, ensure_ascii=False) + "\n")

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
    print(f"    正例 relevance=2 (核心答案): {pos2_count} 条 ({pos2_count // max(len(all_queries), 1)} 条/查询)")
    print(f"    正例 relevance=1 (辅助上下文): {pos1_count} 条 ({pos1_count // max(len(all_queries), 1)} 条/查询)")
    print(f"    负例 relevance=0: {neg_count} 条 ({neg_count // max(len(all_queries), 1)} 条/查询)")

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

    llm_qs = [q for q in all_queries if "LLM 多文档生成" in q.get("description", "")]
    print(f"LLM 多文档生成查询: {len(llm_qs)}")

    # 正例数分布 (分级)
    pos2_per_query = [
        len([l for l in qrels_lines if l.startswith(q["query_id"]) and l.endswith("\t2")])
        for q in all_queries
        if q.get("category") != "robustness"
    ]
    pos1_per_query = [
        len([l for l in qrels_lines if l.startswith(q["query_id"]) and l.endswith("\t1")])
        for q in all_queries
        if q.get("category") != "robustness"
    ]
    if pos2_per_query:
        pos2_counter = Counter(pos2_per_query)
        pos1_counter = Counter(pos1_per_query)
        print(f"\nrelevance=2 分布 (核心答案): {dict(sorted(pos2_counter.items()))}")
        print(f"relevance=1 分布 (辅助上下文): {dict(sorted(pos1_counter.items()))}")

    print(f"\n[HITL 提示]")
    print(f"  1. 逐条审核 {queries_fname} 中的问题是否自然、合理")
    print(f"  2. 验证 relevance=2 的 chunk 确实是答案核心，relevance=1 是合理辅助上下文")
    print(f"  3. 检查 relevance=0 (BM25 hard negatives) 是否误标了相关文档")
    print(f"  4. 补充遗漏的相关文档 (relevance=1 或 2)")
    print(f"  5. 审核通过后另存为正式的 queries.jsonl 和 qrels.tsv")


# ==============================================================================
# CLI
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="RAG 评估数据集生成器 — 多文档 Query-from-Document 模式"
    )
    _PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
    _default_corpus = str(_PROJECT_ROOT / "rag_eval" / "ground_truth" / "corpus.jsonl")

    parser.add_argument("--corpus", default=_default_corpus, help="语料库路径")
    parser.add_argument("--output", default=None, help="查询输出路径 (默认: ground_truth/queries_multi_{provider}.jsonl)")
    parser.add_argument("--qrels", default=None, help="自动 qrels 输出路径 (默认: ground_truth/qrels_multi_{provider}.tsv)")
    parser.add_argument("--provider", default=PROVIDER_DEFAULT,
                        choices=list(PROVIDER_CONFIGS.keys()),
                        help=f"LLM 提供商 (默认: {PROVIDER_DEFAULT})")
    parser.add_argument("--model", default=None, help="LLM 模型名")
    parser.add_argument("--min-group-size", type=int, default=3, help="每组最少 chunk 数 (默认: 3)")
    parser.add_argument("--max-group-size", type=int, default=7, help="每组最多 chunk 数 (默认: 7)")
    parser.add_argument("--target", type=int, default=None, help="目标查询数 (默认: 不限)")
    parser.add_argument("--concurrency", type=int, default=10, help="LLM 并发数 (默认: 10)")
    parser.add_argument("--dry-run", action="store_true", help="测试模式: 仅处理 10 组")
    args = parser.parse_args()

    if args.provider not in PROVIDER_CONFIGS:
        print(f"[ERROR] 未知 provider: {args.provider}")
        sys.exit(1)

    if args.output is None:
        args.output = str(_PROJECT_ROOT / "rag_eval" / "ground_truth" / f"queries_multi_{args.provider}.jsonl")
    if args.qrels is None:
        args.qrels = str(_PROJECT_ROOT / "rag_eval" / "ground_truth" / f"qrels_multi_{args.provider}.tsv")

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
        provider=provider,
        llm_model=model,
        min_group_size=args.min_group_size,
        max_group_size=args.max_group_size,
        target_groups=args.target,
        dry_run=args.dry_run,
        concurrency=args.concurrency,
    ))


if __name__ == "__main__":
    main()
