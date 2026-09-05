"""P2-5: search_scholar_metadata — 学术元数据检索工具。

调用 CrossRef 公共 API（免费、无需 key）检索学术论文元数据。
返回 DOI / 标题 / 作者 / 年份 / 摘要 / 引用数，供 correlate_external 和用户直接查询。

与 evolve_workbench.correlate_external 配合：E5 节点原本只靠 LLM 推断外部关联，
现在可以调此工具拿到真实学术元数据，提升"外部参考"可信度。
"""
import logging
import urllib.parse
import urllib.request
import json

logger = logging.getLogger(__name__)

CROSSREF_API = "https://api.crossref.org/works"
# CrossRef 礼仪: 加 mailto 揬供 contact，走 "polite pool" 速率更高
MAILTO = "research@white-snake-museum.local"
USER_AGENT = "WhiteSnakeMuseum/1.0 (Digital Humanities Research; mailto:{})".format(MAILTO)


def _crossref_search(query: str, rows: int = 10) -> list[dict]:
    """同步调 CrossRef API（urllib，不依赖 aiohttp）。"""
    params = urllib.parse.urlencode({
        "query": query,
        "rows": min(rows, 20),
        "select": "DOI,title,author,published-print,published-online,is-referenced-by-count,abstract,container-title",
        "mailto": MAILTO,
    })
    url = f"{CROSSREF_API}?{params}"
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except Exception as ex:
        logger.warning(f"CrossRef API 调用失败: {ex}")
        return []

    items = data.get("message", {}).get("items", [])
    results = []
    for item in items:
        # 标题: CrossRef 返回 list[str]
        titles = item.get("title", [])
        title = titles[0] if titles else ""
        if not title:
            continue

        # 作者: list[{given, family}]
        authors_raw = item.get("author", [])
        authors = []
        for a in authors_raw[:8]:
            given = a.get("given", "")
            family = a.get("family", "")
            full = f"{family} {given}".strip() if family else given
            if full:
                authors.append(full)
        author_str = "; ".join(authors[:5])
        if len(authors_raw) > 5:
            author_str += " et al."

        # 年份: 优先 published-print，其次 published-online
        year = ""
        for key in ("published-print", "published-online"):
            dp = item.get(key, {})
            if dp and dp.get("date-parts"):
                parts = dp["date-parts"][0]
                if parts and parts[0]:
                    year = str(parts[0])
                    break

        # 期刊
        containers = item.get("container-title", [])
        journal = containers[0] if containers else ""

        # 摘要: CrossRef 的 abstract 通常是 JATS XML 格式，去标签
        abstract = item.get("abstract", "") or ""
        if abstract:
            import re
            abstract = re.sub(r"<[^>]+>", "", abstract).strip()
            if len(abstract) > 500:
                abstract = abstract[:500] + "..."

        results.append({
            "doi": item.get("DOI", ""),
            "title": title,
            "authors": author_str,
            "year": year,
            "journal": journal,
            "citedCount": item.get("is-referenced-by-count", 0),
            "abstract": abstract or "(无摘要)",
            "url": f"https://doi.org/{item.get('DOI', '')}" if item.get("DOI") else "",
        })
    return results


TOOL_DEF = {
    "name": "search_scholar_metadata",
    "description": (
        "检索学术论文元数据（通过 CrossRef 公共 API，免费无需 key）。"
        "返回论文 DOI/标题/作者/年份/期刊/引用数/摘要。"
        "用于补充白蛇传研究的外部学术参考，提升答案可信度。"
        "典型场景：用户问「关于白蛇传演变的学术研究」「方成培研究论文」"
        "「白蛇传比较文学研究有哪些」时调用。"
        "⚠️ 此工具返回的是真实学术元数据（非 AI 编造），引用时标注【学界共识】。"
    ),
    "inputSchema": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "检索关键词（英文效果更佳，如 'Legend of White Snake evolution'）",
            },
            "rows": {
                "type": "integer",
                "description": "返回数量上限（默认 10，最大 20）",
            },
        },
        "required": ["query"],
    },
}


async def handler(query: str, rows: int = 10) -> dict:
    """MCP handler: 检索 CrossRef 学术元数据。"""
    if not query or not query.strip():
        return {"error": "query 不能为空"}
    results = _crossref_search(query.strip(), rows=min(rows or 10, 20))
    if not results:
        return {
            "query": query,
            "results": [],
            "count": 0,
            "note": "CrossRef API 无结果或调用失败。可能是网络问题或关键词过于冷僻。",
        }
    return {
        "query": query,
        "results": results,
        "count": len(results),
        "source": "CrossRef Public API (https://api.crossref.org)",
    }
