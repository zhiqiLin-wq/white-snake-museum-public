"""BEIR 格式数据集加载器。

支持:
- corpus.jsonl: 语料库 (JSONL格式)
- queries.jsonl: 查询集 (JSONL格式)
- qrels.tsv: 相关性标注 (四列TSV)

向后兼容:
- 自动转换旧格式 dynasty_filter → metadata_filters
- expected_genre_scope 替代旧 genre_scope
"""
import json
import random
from pathlib import Path


# ==============================================================================
# 语料库加载
# ==============================================================================
def load_corpus(path: str | Path) -> dict[str, dict]:
    """加载语料库 JSONL 文件。

    Args:
        path: corpus.jsonl 文件路径

    Returns:
        {doc_id: {title, text, metadata}}

    Raises:
        ValueError: JSONL 行解析失败
        FileNotFoundError: 文件不存在
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"语料库文件不存在: {path}")

    corpus: dict[str, dict] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                doc = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"语料库第 {line_no} 行 JSON 解析失败: {e}") from e

            doc_id = doc.get("doc_id") or doc.get("_id")
            if not doc_id:
                raise ValueError(f"语料库第 {line_no} 行缺少 doc_id 字段")

            corpus[doc_id] = {
                "title": doc.get("title", ""),
                "text": doc.get("text", ""),
                "metadata": doc.get("metadata", {}),
            }

    return corpus


# ==============================================================================
# 查询加载
# ==============================================================================
def load_queries(path: str | Path) -> list[dict]:
    """加载查询 JSONL 文件，保持文件顺序。

    自动向后兼容: 将旧格式 ``dynasty_filter`` 转换为 ``metadata_filters``，
    将旧格式 ``genre_scope`` 转换为 ``expected_genre_scope``。

    Args:
        path: queries.jsonl 文件路径

    Returns:
        [{query_id, text, category, difficulty, metadata_filters,
          expected_genre_scope, eval_top_k_values, description}, ...]
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"查询文件不存在: {path}")

    queries: list[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                q = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"查询文件第 {line_no} 行 JSON 解析失败: {e}") from e

            # 向后兼容: dynasty_filter → metadata_filters (T00.06)
            if "metadata_filters" not in q and "dynasty_filter" in q:
                dynasty = q.pop("dynasty_filter")
                if dynasty is not None:
                    q["metadata_filters"] = {"dynasty": dynasty}
                else:
                    q["metadata_filters"] = None

            # 确保 metadata_filters 字段存在
            if "metadata_filters" not in q:
                q["metadata_filters"] = None

            # 向后兼容: genre_scope → expected_genre_scope (T00.06)
            if "genre_scope" in q and "expected_genre_scope" not in q:
                q["expected_genre_scope"] = q.pop("genre_scope")
            if "expected_genre_scope" not in q:
                q["expected_genre_scope"] = "all"

            # 确保 eval_top_k_values 字段存在
            if "eval_top_k_values" not in q:
                q["eval_top_k_values"] = None  # null = 使用默认值 [3,5,10,20,50]

            # 确保 description 字段存在
            if "description" not in q:
                q["description"] = ""

            queries.append(q)

    return queries


# ==============================================================================
# qrels 加载
# ==============================================================================
def load_qrels(path: str | Path) -> dict[str, dict[str, int]]:
    """加载 qrels TSV 文件 (四列: query_id, doc_id, relevance)。

    跳过 ``#`` 开头的注释行，跳过 relevance=0 的行。

    Args:
        path: qrels.tsv 文件路径

    Returns:
        {query_id: {doc_id: relevance}}
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"qrels 文件不存在: {path}")

    qrels: dict[str, dict[str, int]] = {}
    seen_pairs: set[tuple[str, str]] = set()

    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            parts = line.split("\t")
            if len(parts) < 3:
                raise ValueError(
                    f"qrels 第 {line_no} 行格式错误，期望至少 3 列 (TSV): {line}"
                )

            query_id = parts[0].strip()
            doc_id = parts[1].strip()
            try:
                relevance = int(parts[2].strip())
            except ValueError as e:
                raise ValueError(
                    f"qrels 第 {line_no} 行 relevance 非整数: {parts[2].strip()}"
                ) from e

            # 跳过 relevance=0 的行
            if relevance == 0:
                continue

            # 重复检测 (T01.06)
            pair = (query_id, doc_id)
            if pair in seen_pairs:
                raise ValueError(
                    f"qrels 第 {line_no} 行存在重复的 (query_id, doc_id) 对: "
                    f"({query_id}, {doc_id})"
                )
            seen_pairs.add(pair)

            if query_id not in qrels:
                qrels[query_id] = {}
            qrels[query_id][doc_id] = relevance

    return qrels


# ==============================================================================
# 全量 qrels 加载 (含 hard negatives)
# ==============================================================================
def load_all_qrels(path: str | Path) -> dict[str, dict[str, int]]:
    """加载 qrels TSV 文件，保留 relevance=0 的 hard negatives。

    与 load_qrels 不同，此函数保留全部 relevance 值 (0/1/2)，
    用于计算 false positive rate 等需要负例的指标。

    Args:
        path: qrels.tsv 文件路径

    Returns:
        {query_id: {doc_id: relevance}}  — 含 rel=0/1/2 全部标注
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"qrels 文件不存在: {path}")

    all_qrels: dict[str, dict[str, int]] = {}
    seen_pairs: set[tuple[str, str]] = set()

    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            parts = line.split("\t")
            if len(parts) < 3:
                raise ValueError(
                    f"qrels 第 {line_no} 行格式错误，期望至少 3 列 (TSV): {line}"
                )

            query_id = parts[0].strip()
            doc_id = parts[1].strip()
            try:
                relevance = int(parts[2].strip())
            except ValueError as e:
                raise ValueError(
                    f"qrels 第 {line_no} 行 relevance 非整数: {parts[2].strip()}"
                ) from e

            # 重复检测
            pair = (query_id, doc_id)
            if pair in seen_pairs:
                raise ValueError(
                    f"qrels 第 {line_no} 行存在重复的 (query_id, doc_id) 对: "
                    f"({query_id}, {doc_id})"
                )
            seen_pairs.add(pair)

            if query_id not in all_qrels:
                all_qrels[query_id] = {}
            all_qrels[query_id][doc_id] = relevance

    return all_qrels


# ==============================================================================
# 语料库导出
# ==============================================================================
def export_corpus(documents: list, output_path: str | Path) -> None:
    """将 Chunker.chunk_all() 返回的 Document 列表导出为 corpus.jsonl。

    字段映射:
        meta.chunk_id → doc_id
        meta.chapter_title → title
        doc.content → text
        metadata 保留 genre, dynasty, chapter_number, source_type, source_file,
                  strategy, paragraph_index, char_start, char_end

    Args:
        documents: Chunker Document 列表
        output_path: 导出路径 (corpus.jsonl)
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        for doc in documents:
            meta = doc.metadata if hasattr(doc, "metadata") else {}
            content = doc.content if hasattr(doc, "content") else ""

            record = {
                "doc_id": meta.get("chunk_id", ""),
                "title": meta.get("chapter_title", ""),
                "text": content,
                "metadata": {
                    "genre": meta.get("genre", ""),
                    "dynasty": meta.get("dynasty", ""),
                    "chapter_number": meta.get("chapter_number", ""),
                    "source_type": meta.get("source_type", ""),
                    "source_file": meta.get("source_file", ""),
                    "strategy": meta.get("strategy", ""),
                    "paragraph_index": meta.get("paragraph_index", ""),
                    "char_start": meta.get("char_start", ""),
                    "char_end": meta.get("char_end", ""),
                },
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


# ==============================================================================
# CI 子集采样
# ==============================================================================
# 设计文档 1.3 节 CI 子集分布 (每类别采样数)
DEFAULT_CI_DISTRIBUTION: dict[str, int] = {
    "entity": 3,
    "location": 3,
    "plot": 4,
    "comparison": 4,
    "motif": 3,
    "filtered": 3,
    "alias": 4,
    "intertext": 3,
    "robustness": 3,
}
# 总计: 30 条


def sample_ci_subset(
    queries: list[dict],
    n_per_category: dict[str, int] | None = None,
    seed: int = 42,
) -> list[dict]:
    """按类别分层，每类再按 easy/medium/hard 分层随机抽样。

    Args:
        queries: 全量查询列表
        n_per_category: 每类采样数，默认使用 DEFAULT_CI_DISTRIBUTION
        seed: 随机种子

    Returns:
        CI 子集查询列表
    """
    if n_per_category is None:
        n_per_category = DEFAULT_CI_DISTRIBUTION

    rng = random.Random(seed)

    # 按类别分组
    by_category: dict[str, list[dict]] = {}
    for q in queries:
        cat = q.get("category", "unknown")
        if cat not in by_category:
            by_category[cat] = []
        by_category[cat].append(q)

    ci_queries: list[dict] = []

    for category, n_target in n_per_category.items():
        pool = by_category.get(category, [])
        if not pool:
            continue

        # 按难度分层
        by_diff: dict[str, list[dict]] = {"easy": [], "medium": [], "hard": []}
        for q in pool:
            diff = q.get("difficulty", "medium")
            if diff not in by_diff:
                diff = "medium"
            by_diff[diff].append(q)

        # 每难度至少 1 条，剩余随机分配
        selected: list[dict] = []
        for diff in ["easy", "medium", "hard"]:
            if by_diff[diff] and len(selected) < n_target:
                chosen = rng.choice(by_diff[diff])
                selected.append(chosen)
                by_diff[diff].remove(chosen)

        # 剩余采样
        remaining_pool = [q for diff_list in by_diff.values() for q in diff_list]
        n_remain = n_target - len(selected)
        if n_remain > 0 and remaining_pool:
            extra = rng.sample(
                remaining_pool, min(n_remain, len(remaining_pool))
            )
            selected.extend(extra)

        ci_queries.extend(selected)

    # 按 query_id 排序保持确定性
    ci_queries.sort(key=lambda q: q.get("query_id", ""))

    return ci_queries
