"""MCP 工具 — 提取关键词。"""
import logging
import httpx

logger = logging.getLogger(__name__)

TOOL_DEF = {
    "name": "get_keywords",
    "description": (
        "提取关键词及权重。双数据源自动判断：target 为白蛇传地标名称时"
        "返回该地标在文献中的预计算关键词；否则将 target 整体当作一段文本"
        "做 jieba TF-IDF 分词。返回 source 字段标明实际使用的数据源。"
        "注意：传入地标名得到的不是'地标这个词的关键词'，而是'该地标相关的文献关键词'。"
    ),
    "inputSchema": {
        "type": "object",
        "properties": {
            "target": {
                "type": "string",
                "description": "目标：地标名称 或 要分析的文本",
            },
            "top_n": {
                "type": "integer",
                "description": "返回关键词数量，默认 10",
                "default": 10,
            },
        },
        "required": ["target"],
    },
}


async def handler(target: str, top_n: int = 10, node_client=None) -> dict:
    """提取关键词。"""
    # 先尝试作为地标名查询
    if node_client:
        try:
            locations = await node_client.get_locations()
            for loc in locations:
                if loc.get("name") == target:
                    keywords = loc.get("keywords", [])[:top_n]
                    return {
                        "source": f"地标: {target}",
                        "keywords": keywords,
                    }
        except httpx.HTTPError as e:
            logger.debug(f"Node API 关键词查询失败（回退到本地分词）: {e}")

    # 本地 jieba 分词
    try:
        import jieba
        import jieba.analyse
        keywords = jieba.analyse.extract_tags(target, topK=top_n, withWeight=True)
        return {
            "source": "本地分词",
            "keywords": [{"word": w, "weight": round(wt, 4)} for w, wt in keywords],
        }
    except ImportError:
        pass

    # 最简 fallback
    words = [w for w in target.replace("，", " ").replace("。", " ").split() if len(w) >= 2]
    return {
        "source": "简单分词",
        "keywords": [{"word": w, "weight": 1.0} for w in words[:top_n]],
    }
