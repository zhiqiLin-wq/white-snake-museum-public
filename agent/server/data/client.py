"""HTTP 客户端 — 调用 Node Fastify 后端 API。

B-124: TTL 缓存 GET /api/literature 响应
"""
import logging
from typing import Optional
import httpx
from ..config import settings
from .cache import TTLCache

logger = logging.getLogger(__name__)

# B-124: Node API 响应缓存 TTL（1 小时）
NODE_API_CACHE_TTL = 3600

# Node.js API 返回的章节 number 字段是中文数字，需要转换为整数
_CHINESE_NUMBER_MAP: dict[str, int] = {
    "一": 1, "二": 2, "三": 3, "四": 4, "五": 5, "六": 6, "七": 7,
    "八": 8, "九": 9, "十": 10,
}


def _normalize_chapter(ch: dict) -> dict:
    """归一化 Node.js API 返回的章节数据。

    Node.js 的 parseLiterature() 返回 {number: "三", title: "...", content: "..."}，
    其中 number 字段值是中文数字字符串。归一化后增加整数字段，让下游工具用整数匹配。
    """
    raw_number = ch.get("number", "")
    int_number = _CHINESE_NUMBER_MAP.get(str(raw_number).strip())
    if int_number is None:
        int_number = int(raw_number) if str(raw_number).isdigit() else 0
    ch["chapterNumber"] = int_number
    ch["chapter_number"] = int_number
    return ch


class NodeAPIClient:
    """调用 Node 后端的 /api/literature 和 /api/locations。"""

    def __init__(self, base_url: Optional[str] = None):
        self.base_url = base_url or settings.node_api_url
        self._client: Optional[httpx.AsyncClient] = None
        # B-124: 启用 TTL 缓存
        self._cache = TTLCache(ttl_seconds=NODE_API_CACHE_TTL, stale_seconds=600)

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=httpx.Timeout(30.0))
        return self._client

    async def get_literature(self) -> list[dict]:
        """获取文献章节列表（缓存 1 小时）。

        归一化章节号：Node.js API 返回中文数字 "一"/"二"/"三" 作为 number 字段，
        归一化后每个章节增加 chapterNumber (int) 和 chapter_number (int)。

        fast fail: 只记录 HTTP/网络异常后重新抛出。
        """
        cache_key = "literature_list"
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached

        client = await self._get_client()
        try:
            resp = await client.get(f"{self.base_url}/literature")
            resp.raise_for_status()
            data = resp.json()
            normalized = [_normalize_chapter(ch) for ch in data]
            self._cache.set(cache_key, normalized)
            return normalized
        except (httpx.HTTPError, httpx.TimeoutException, ConnectionError, ValueError) as e:
            logger.error(f"获取文献失败: {e}")
            raise

    async def get_literature_by_chapter(self, chapter_number: int) -> Optional[dict]:
        """获取指定章节的完整文献（B-011 get_chapter_full_text 使用）。

        章节号从归一化后的 chapterNumber 字段匹配（整数对整数）。
        """
        chapters = await self.get_literature()
        for ch in chapters:
            if ch.get("chapterNumber") == chapter_number:
                return ch
        return None

    async def get_locations(self) -> list[dict]:
        """获取地标数据列表（缓存 1 小时）。

        fast fail: 只记录 HTTP/网络异常后重新抛出。
        """
        cache_key = "locations_list"
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached

        client = await self._get_client()
        try:
            resp = await client.get(f"{self.base_url}/locations")
            resp.raise_for_status()
            data = resp.json()
            self._cache.set(cache_key, data)
            return data
        except (httpx.HTTPError, httpx.TimeoutException, ConnectionError, ValueError) as e:
            logger.error(f"获取地标数据失败: {e}")
            raise

    async def search_literature(self, query: str) -> dict:
        """全文搜索文献（调用 Node /api/literature/search）。

        fast fail: HTTP/网络异常直接抛出。
        """
        client = await self._get_client()
        try:
            resp = await client.get(
                f"{self.base_url}/literature/search",
                params={"q": query.strip()},
            )
            resp.raise_for_status()
            return resp.json()
        except (httpx.HTTPError, httpx.TimeoutException, ConnectionError, ValueError) as e:
            logger.error(f"文献搜索失败 [{query}]: {e}")
            raise

    async def get_annotations(self, chapter_number: int, user_id: str = "") -> dict:
        """读取指定章节已保存的标注/旁批（GET /api/annotations/:chapterNumber）。

        返回 {annotations: {passageKey: [...]}, marginalia: {passageKey: [...]}}，
        前端格式（label/span/note）。异常时返回空 dict（降级为"无已有数据"）。
        """
        client = await self._get_client()
        try:
            headers = {"x-user-id": user_id} if user_id else {}
            resp = await client.get(
                f"{self.base_url}/annotations/{chapter_number}", headers=headers
            )
            resp.raise_for_status()
            return resp.json()
        except (httpx.HTTPError, httpx.TimeoutException, ConnectionError, ValueError) as e:
            logger.warning(f"读取已有标注失败 ch={chapter_number}（降级为增量跳过）: {e}")
            return {}

    async def save_annotations(
        self, chapter_number: int,
        annotations: list[dict] | None = None,
        marginalia: list[dict] | None = None,
        user_id: str = "",
    ) -> dict:
        """调用 Node /api/annotations/save 保存标注到后端文件。

        fast fail: HTTP/网络异常 → 直接抛出。
        """
        client = await self._get_client()
        body: dict = {"chapterNumber": chapter_number}
        if annotations is not None:
            body["annotations"] = annotations
        if marginalia is not None:
            body["marginalia"] = marginalia
        headers = {}
        if user_id:
            headers["x-user-id"] = user_id
        resp = await client.post(f"{self.base_url}/annotations/save", json=body, headers=headers)
        resp.raise_for_status()
        return resp.json()

    async def upsert_conversation(
        self, conversation_id: str, title: str = "", user_id: str = ""
    ) -> dict:
        """调用 Node /api/conversations 创建或更新对话记录。

        在 Agent 对话开始和结束时调用，确保 conversation 列表与 checkpoint
        保持同步。即使 Node API 不可达也不抛异常（使用 degraded mode）。

        Args:
            conversation_id: 对话 ID（对应 LangGraph thread_id）
            title: 对话标题
            user_id: 用户 ID
        """
        client = await self._get_client()
        body = {"id": conversation_id}
        if title:
            body["title"] = title
        headers = {}
        if user_id:
            headers["x-user-id"] = user_id
        try:
            resp = await client.post(
                f"{self.base_url}/conversations", json=body, headers=headers, timeout=10.0,
            )
            resp.raise_for_status()
            return resp.json()
        except (httpx.HTTPError, httpx.TimeoutException, ConnectionError) as e:
            logger.warning(f"upsert_conversation [{conversation_id}] 失败 (degraded): {e}")
            return {"status": "degraded", "error": str(e)}

    async def delete_conversation(
        self, conversation_id: str, user_id: str = ""
    ) -> dict:
        """调用 Node /api/conversations/:id 删除对话记录。"""
        client = await self._get_client()
        headers = {}
        if user_id:
            headers["x-user-id"] = user_id
        try:
            resp = await client.request(
                "DELETE", f"{self.base_url}/conversations/{conversation_id}",
                headers=headers, timeout=10.0,
            )
            resp.raise_for_status()
            return resp.json()
        except (httpx.HTTPError, httpx.TimeoutException, ConnectionError) as e:
            logger.warning(f"delete_conversation [{conversation_id}] 失败 (degraded): {e}")
            return {"status": "degraded", "error": str(e)}

    async def delete_annotations(
        self, chapter_number: int,
        categories: list[str] | None = None,
        delete_all: bool = False,
        clean_agent: bool = False,
        user_id: str = "",
    ) -> dict:
        """调用 Node /api/annotations DELETE 端点删除标注。返回剩余标注。

        Args:
            chapter_number: 章节编号
            categories: 按类别删除（与 delete_all 互斥）
            delete_all: 删除所有用户标注
            clean_agent: 同时清理 agent 标注（默认仅删除用户来源数据）
            user_id: 用户 ID

        fast fail: 参数无效 → ValueError；HTTP/网络异常 → 直接抛出。
        """
        if not delete_all and not categories:
            raise ValueError("需要提供 categories 或 delete_all")
        client = await self._get_client()
        body: dict = {"chapterNumber": chapter_number}
        if delete_all:
            body["deleteAll"] = True
        else:
            body["categories"] = categories
        if clean_agent:
            body["cleanAgent"] = True
        headers = {}
        if user_id:
            headers["x-user-id"] = user_id
        resp = await client.request(
            "DELETE", f"{self.base_url}/annotations", json=body, headers=headers,
        )
        resp.raise_for_status()
        return resp.json()

    async def health_check(self) -> bool:
        """检查 Node API 是否可达。

        fast fail: 网络错误 -> 返回 False，不抛异常（health check 不应崩溃）。
        """
        try:
            client = await self._get_client()
            resp = await client.get(f"{self.base_url}/literature", timeout=5.0)
            return resp.status_code == 200
        except (httpx.HTTPError, httpx.TimeoutException, ConnectionError) as e:
            logger.warning(f"Node API health check 失败: {e}")
            return False

    async def close(self):
        if self._client:
            await self._client.aclose()
            self._client = None

    def invalidate_cache(self):
        """清除所有缓存。"""
        self._cache.clear()
