"""MCP 协议服务端 — 基于 Anthropic mcp SDK。"""
import json
import logging
import httpx
from typing import Optional

logger = logging.getLogger(__name__)


class MCPServer:
    """MCP 协议服务端。

    管理工具注册和调用，通过 HTTP+SSE 与客户端交互。
    """

    def __init__(self):
        self._registry = None
        self._tools_by_name: dict[str, callable] = {}

    def init_with_registry(self, registry):
        """用 MCPToolRegistry 初始化。"""
        self._registry = registry

    def register_tool(self, name: str, handler: callable):
        """直接注册一个工具处理函数。"""
        self._tools_by_name[name] = handler

    def list_tools(self) -> list[dict]:
        """列出所有可用工具（MCP tools/list 响应）。"""
        if self._registry:
            return self._registry.list_tools()

        # 从直接注册的 handlers 返回基本信息
        return [
            {"name": name, "description": f"Tool: {name}"}
            for name in self._tools_by_name
        ]

    async def call_tool(self, name: str, arguments: dict) -> dict:
        """调用工具（MCP tools/call 响应）。

        返回格式：
        {
            "content": [{"type": "text", "text": "..."}],
            "isError": false
        }
        """
        result = None

        # 优先通过 registry 调用
        if self._registry:
            result = await self._registry.call_tool(name, arguments)
        elif name in self._tools_by_name:
            try:
                result = await self._tools_by_name[name](**arguments)
                result = {"result": result}
            except (RuntimeError, ValueError, TypeError, httpx.HTTPError, ConnectionError) as e:
                result = {"error": str(e)}
        else:
            return {
                "content": [{"type": "text", "text": f"工具 '{name}' 未注册"}],
                "isError": True,
            }

        if "error" in result:
            return {
                "content": [{"type": "text", "text": result["error"]}],
                "isError": True,
            }

        return {
            "content": [{"type": "text", "text": json.dumps(result.get("result", result), ensure_ascii=False, indent=2)}],
            "isError": False,
        }

    async def handle_request(self, method: str, params: Optional[dict] = None) -> dict:
        """处理 MCP JSON-RPC 风格请求。"""
        if method == "tools/list":
            return {"tools": self.list_tools()}
        elif method == "tools/call":
            name = params.get("name", "") if params else ""
            arguments = params.get("arguments", {}) if params else {}
            return await self.call_tool(name, arguments)
        elif method == "initialize":
            return {
                "protocolVersion": "2024-11-05",
                "capabilities": {"tools": {}},
                "serverInfo": {
                    "name": "white-snake-museum-mcp",
                    "version": "1.0.0",
                },
            }
        else:
            return {"error": f"未知方法: {method}"}
