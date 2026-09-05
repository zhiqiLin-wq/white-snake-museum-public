"""认证模块 — 从 Node.js 共享的 auth.db 验证 session token。

认证策略 (双层):
1. 外部请求: 通过 Authorization header / session_token cookie 验证
2. 内部请求 (来自 Node.js 代理的 localhost 调用): 通过 x-user-id header 信任传递
   (Node.js 中间件已完成认证，此处不再重复查 auth.db)
"""
import re
import sqlite3
from pathlib import Path
from fastapi import Request, HTTPException
from ..config import PROJECT_ROOT

# UUID v4 格式校验，防止路径遍历攻击
_UUID_RE = re.compile(
    r'^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$',
    re.IGNORECASE,
)

AUTH_DB = PROJECT_ROOT.parent / "data" / "auth.db"

# Token 格式: 64 个十六进制字符 (对应 Node.js randomBytes(32).toString('hex'))
_TOKEN_REGEX = re.compile(r'^[0-9a-f]{64}$', re.IGNORECASE)

PUBLIC_PATHS = {"/health", "/docs", "/openapi.json", "/skills"}

# 内部代理路径: 这些端点由 Node.js 代理层调用，Node 已完成鉴权
# 仅允许来自 localhost 的内部调用，通过 x-user-id header 标识用户
INTERNAL_PROXY_PATHS = {
    "/chat", "/chat/resume", "/chat/cancel", "/chat/sync",
    "/rebuild-index", "/reload-templates",
    "/memory/conversation",
    "/reports",  # v19: 报告文件库（Node 代理已完成鉴权）
}


def _is_localhost(request: Request) -> bool:
    """检查请求是否来自本机 (IPv4/IPv6/localhost)。"""
    if request.client is None:
        return False
    host = request.client.host
    return host in ("127.0.0.1", "::1", "localhost")


def _extract_token(request: Request) -> str | None:
    auth = request.headers.get("Authorization", "")
    if auth.startswith("Bearer "):
        raw = auth[7:].strip()
        # 校验 token 格式 (64 位 hex)，与 Node.js 端保持一致
        if _TOKEN_REGEX.match(raw):
            return raw
        return None
    token = request.cookies.get("session_token")
    if token:
        # Cookie 中的 token 同样需要格式校验
        if _TOKEN_REGEX.match(token):
            return token
        return None
    return None


def _validate_session(token: str) -> str | None:
    if not AUTH_DB.is_file():
        return None
    conn = None
    try:
        conn = sqlite3.connect(str(AUTH_DB))
        conn.execute("PRAGMA journal_mode=WAL")
        row = conn.execute(
            "SELECT user_id FROM sessions WHERE token = ? AND expires_at > datetime('now')",
            (token,),
        ).fetchone()
        return row[0] if row else None
    except Exception:
        return None
    finally:
        if conn:
            conn.close()


async def get_current_user(request: Request) -> str:
    # 公共路径无需认证
    if request.url.path in PUBLIC_PATHS:
        return ""

    # 内部代理路径: Node.js 已完成鉴权，通过 x-user-id 信任传递
    # 仅允许来自 localhost 的内部调用使用此机制
    # x-user-id 必须是合法 UUID v4 格式
    if request.url.path in INTERNAL_PROXY_PATHS or any(
        request.url.path.startswith(p + "/") for p in INTERNAL_PROXY_PATHS
    ):
        if _is_localhost(request):
            internal_user_id = request.headers.get("x-user-id")
            if internal_user_id and _UUID_RE.match(internal_user_id):
                return internal_user_id

    # 标准认证流程: 从 Authorization header 或 session_token cookie 提取 token
    token = _extract_token(request)
    if not token:
        raise HTTPException(status_code=401, detail="Not authenticated")

    user_id = _validate_session(token)
    if not user_id:
        raise HTTPException(status_code=401, detail="Session expired")

    return user_id


def get_user_db_path(user_id: str) -> Path | None:
    if not user_id:
        return None
    if not _UUID_RE.match(user_id):
        return None
    return PROJECT_ROOT.parent / "data" / "users" / user_id / "user.db"


def get_checkpoints_db_path(user_id: str) -> Path | None:
    if not user_id:
        return None
    if not _UUID_RE.match(user_id):
        return None
    return PROJECT_ROOT.parent / "data" / "users" / user_id / "agent_checkpoints.db"
