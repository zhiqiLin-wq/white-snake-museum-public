"""高德地图 POI 搜索客户端 — 真实景点/酒店/餐厅数据。
Fast Fail: API 异常使用具体 httpx 异常类型，绝不吞异常。
"""
import logging
from typing import Optional
from dataclasses import dataclass, field

import httpx

from ..config import settings

logger = logging.getLogger(__name__)

AMAP_BASE_URL = "https://restapi.amap.com/v3"
# POI 类型常量
POI_SCENIC = "110000|110100|110200|120000|120100"  # 风景名胜 + 公园
POI_HOTEL = "100000|100100|100200|100300"           # 酒店/宾馆/旅店
POI_RESTAURANT = "050000|050100|050200|050300"      # 餐饮
POI_TRANSPORT = "150000|150100|150200|150300|150400|150500|150600"  # 交通枢纽


@dataclass
class POIResult:
    """高德 POI 搜索结果。"""
    name: str
    address: str
    poi_type: str = ""
    tel: str = ""
    location: str = ""          # "lng,lat"
    distance: str = ""           # 距离（米）
    rating: str = ""             # 评分
    ticket_price: str = ""       # 门票价格（仅景点）
    hotel_star: str = ""         # 酒店星级
    hotel_price_range: str = ""  # 酒店价格区间
    open_time: str = ""          # 营业/开放时间
    photos: list = field(default_factory=list)
    raw_data: dict = field(default_factory=dict)


class AmapClient:
    """高德地图 Web API 客户端 — HTTP直连，Fast Fail。"""

    def __init__(self):
        self._client: Optional[httpx.AsyncClient] = None

    def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=AMAP_BASE_URL,
                timeout=httpx.Timeout(15.0, connect=5.0),
            )
        return self._client

    def _api_key_param(self) -> dict:
        return {"key": settings.amap_api_key}

    # ===== POI 文本搜索 =====

    async def search_poi(
        self,
        keywords: str,
        city: str = "",
        poi_type: str = "",
        offset: int = 10,
        page: int = 1,
    ) -> list[POIResult]:
        """搜索 POI（景点/酒店/餐厅/交通枢纽）。

        Args:
            keywords: 搜索关键词，如 "雷峰塔"、"全季酒店"
            city: 城市名，如 "杭州"、"成都"
            poi_type: POI类型代码，可为空（不限制）
            offset: 每页数量，默认10
        """
        params = {
            **self._api_key_param(),
            "keywords": keywords,
            "offset": min(offset, 25),
            "page": page,
            "extensions": "all",  # 返回详细信息含 biz_ext + deep_info
        }
        if city:
            params["city"] = city
        if poi_type:
            params["types"] = poi_type

        client = self._get_client()
        resp = await client.get("/place/text", params=params)

        if resp.status_code != 200:
            body = resp.text[:300]
            raise AmapAPIError(f"高德 POI 搜索失败 ({resp.status_code}): {body}")

        data = resp.json()
        if data.get("status") != "1":
            info = data.get("info", "unknown error")
            raise AmapAPIError(f"高德 API 错误: {info}")

        pois = data.get("pois", [])
        return [self._parse_poi(p) for p in pois]

    # ===== POI 周边搜索 =====

    async def search_around(
        self,
        location: str,       # "lng,lat"
        keywords: str = "",
        poi_type: str = "",
        radius: int = 3000,  # 米
        offset: int = 10,
    ) -> list[POIResult]:
        """周边搜索 — 以某坐标为中心搜索周边 POI。"""
        params = {
            **self._api_key_param(),
            "location": location,
            "radius": min(radius, 50000),
            "offset": min(offset, 25),
            "extensions": "all",
        }
        if keywords:
            params["keywords"] = keywords
        if poi_type:
            params["types"] = poi_type

        client = self._get_client()
        resp = await client.get("/place/around", params=params)

        if resp.status_code != 200:
            body = resp.text[:300]
            raise AmapAPIError(f"高德周边搜索失败 ({resp.status_code}): {body}")

        data = resp.json()
        if data.get("status") != "1":
            info = data.get("info", "unknown error")
            raise AmapAPIError(f"高德 API 错误: {info}")

        pois = data.get("pois", [])
        return [self._parse_poi(p) for p in pois]

    # ===== POI 详情 =====

    async def get_detail(self, poi_id: str) -> Optional[POIResult]:
        """获取 POI 详细信息。"""
        params = {
            **self._api_key_param(),
            "id": poi_id,
            "extensions": "all",
        }

        client = self._get_client()
        resp = await client.get("/place/detail", params=params)

        if resp.status_code != 200:
            body = resp.text[:300]
            raise AmapAPIError(f"高德 POI 详情失败 ({resp.status_code}): {body}")

        data = resp.json()
        if data.get("status") != "1":
            info = data.get("info", "unknown error")
            raise AmapAPIError(f"高德 API 错误: {info}")

        pois = data.get("pois", [])
        if not pois:
            return None
        return self._parse_poi(pois[0])

    # ===== 搜索景点（含门票价） =====

    async def search_scenic(self, keywords: str, city: str) -> list[dict]:
        """搜索景点 — 返回含门票价格的精简格式。"""
        results = await self.search_poi(keywords, city, poi_type=POI_SCENIC, offset=5)
        return [
            {
                "name": r.name,
                "address": r.address,
                "tel": r.tel,
                "ticket_price": r.ticket_price or "请电话咨询",
                "open_time": r.open_time or "请电话咨询",
                "rating": r.rating,
                "location": r.location,
            }
            for r in results
        ]

    # ===== 搜索酒店（含价格） =====

    async def search_hotels(self, keywords: str, city: str, limit: int = 6) -> list[dict]:
        """搜索酒店 — 返回含价格区间的精简格式。"""
        results = await self.search_poi(keywords, city, poi_type=POI_HOTEL, offset=limit)
        return [
            {
                "name": r.name,
                "address": r.address,
                "tel": r.tel,
                "star": r.hotel_star or "未标星",
                "price_range": r.hotel_price_range or "请电话咨询",
                "rating": r.rating,
                "location": r.location,
            }
            for r in results
        ]

    # ===== 搜索餐厅 =====

    async def search_restaurants(self, keywords: str, city: str, limit: int = 5) -> list[dict]:
        """搜索餐厅 — 返回含人均消费的精简格式。"""
        results = await self.search_poi(keywords, city, poi_type=POI_RESTAURANT, offset=limit)
        return [
            {
                "name": r.name,
                "address": r.address,
                "tel": r.tel,
                "avg_cost": r.ticket_price or r.hotel_price_range or "请电话咨询",
                "rating": r.rating,
                "location": r.location,
            }
            for r in results
        ]

    # ===== POI 数据解析 =====

    def _parse_poi(self, raw: dict) -> POIResult:
        """解析高德 POI JSON → POIResult。"""
        biz_ext = raw.get("biz_ext", {}) or {}
        deep_info = raw.get("deep_info", {}) or {}

        return POIResult(
            name=raw.get("name", ""),
            address=raw.get("address", ""),
            poi_type=raw.get("type", ""),
            tel=raw.get("tel", "") or "",
            location=raw.get("location", ""),
            distance=raw.get("distance", ""),
            rating=biz_ext.get("rating", "") or deep_info.get("rating", ""),
            ticket_price=biz_ext.get("cost", "") or deep_info.get("price", ""),
            hotel_star=biz_ext.get("star", ""),
            hotel_price_range=biz_ext.get("lowest_price", ""),
            open_time=deep_info.get("opentime", "") or biz_ext.get("opentime", ""),
            photos=[p.get("url", "") for p in raw.get("photos", [])[:3]],
            raw_data=raw,
        )

    async def close(self):
        if self._client:
            await self._client.aclose()
            self._client = None


class AmapAPIError(Exception):
    """高德 API 错误。"""
