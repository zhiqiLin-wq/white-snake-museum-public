# 临时复现脚本: 'str' object has no attribute 'get'（验证后删除）
import asyncio
import json
import sys
import traceback
from functools import partial
from pathlib import Path

sys.path.insert(0, ".")


class MockRetriever:
    """快路径必不足，强制走 ReAct 多轮工具调用。"""
    supports_filters = False
    supports_skip_tagging = False

    async def retrieve(self, query, top_k=None, filters=None, **kwargs):
        return []


async def test_failed_tools_direct() -> None:
    """直接复现用户会话中失败工具 + 成功工具的完整 _call_tool 处理链。"""
    from server.llm.deepseek_provider import DeepSeekProvider
    from server.data.client import NodeAPIClient
    from server.mcp.registry import MCPToolRegistry as MCPRegistry
    from server.prompts.registry import PromptRegistry
    from server.agent_loop import AgentLoop

    llm = DeepSeekProvider()
    node_client = NodeAPIClient()
    mcp = MCPRegistry()

    from server.mcp.tools.search_literature import TOOL_DEF as sl, handler as sl_h
    from server.mcp.tools.get_location import TOOL_DEF as gl, handler as gl_h
    from server.mcp.tools.search_paragraphs import TOOL_DEF as b010, handler as b010_h
    from server.mcp.tools.cooccurrence_matrix import TOOL_DEF as b018, handler as b018_h

    mcp.register(**sl, handler=partial(sl_h, node_client=node_client))
    mcp.register(**gl, handler=gl_h)
    mcp.register(**b010, handler=partial(b010_h, retriever=MockRetriever()))
    mcp.register(**b018, handler=partial(b018_h, node_client=node_client))

    prompts = PromptRegistry(Path("server/prompts/templates"))
    agent = AgentLoop(llm, MockRetriever(), prompts,
                      mcp_registry=mcp, node_client=node_client,
                      user_id="ef0554ea-2b64-4b0e-8e3e-ec099b43df26")

    cases = [
        ("cooccurrence_matrix", {"entities": ["白素贞", "小青"]}),   # 用户会话中失败 int('七')
        ("get_location_info", {"location_name": "西湖"}),            # 用户会话中 {"error": "Node A..."}
        ("search_literature", {"query": "小青"}),                    # 成功
        ("search_paragraphs", {"query": "小青 侍女", "top_k": 10}),  # 成功（新格式）
    ]
    for name, args in cases:
        r = await agent._call_tool(name, args)
        print(f"{name}: success={r.success} error={str(r.error)[:60]} sse={len(r.sse_events)} result_type={type(r.result).__name__}")
        for ev in r.sse_events:
            if not isinstance(ev, dict):
                print(f"  !!! 非 dict sse_event: {type(ev)} -> {str(ev)[:200]}")
                return
            ev.get("type"); ev.get("data")
        content = json.dumps(r.result, ensure_ascii=False) if r.success else r.error
        print(f"  tool_msg.content type={type(content).__name__}")

    print("DIRECT TEST OK")


asyncio.run(test_failed_tools_direct())
