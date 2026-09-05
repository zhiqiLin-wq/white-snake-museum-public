import type { FastifyRequest, FastifyReply } from 'fastify'
import { validateSession } from '../services/auth.service.js'

// 精确公共路径: 无需认证即可访问
const PUBLIC_PATHS = new Set([
  '/api/auth/login',
  '/api/auth/register',
  '/api/auth/logout',
  '/api/literature',
  '/api/literature/search',
  '/api/research-literature',
  '/api/locations',
  '/api/agent/health',
  '/health',
])

// 需要认证的路径前缀
const PROTECTED_PREFIXES = [
  '/api/agent',
  '/api/annotations',
  '/api/conversations',
]

// Token 格式: 64 个十六进制字符 (secrets.token_hex(32) 的产物)
const TOKEN_REGEX = /^[0-9a-f]{64}$/i

function extractToken(req: FastifyRequest): string | null {
  const auth = req.headers.authorization
  if (auth && auth.startsWith('Bearer ')) {
    const raw = auth.slice(7).trim()
    if (TOKEN_REGEX.test(raw)) return raw
    return null
  }
  const cookies = req.headers.cookie
  if (cookies) {
    // 更稳健的 cookie 解析: 匹配 session_token=<value>，值只包含合法 token 字符
    const match = cookies.match(/(?:^|;\s*)session_token=([0-9a-fA-F]{64})(?:;|$)/)
    if (match) return match[1]
  }
  return null
}

function isLocalhost(req: FastifyRequest): boolean {
  const ip = req.ip
  return ip === '127.0.0.1' || ip === '::1' || ip === 'localhost'
}

async function authMiddleware(req: FastifyRequest, reply: FastifyReply): Promise<void> {
  // 去掉 query string，只匹配路径部分
  const rawUrl = req.url
  const path = rawUrl.includes('?') ? rawUrl.split('?')[0] : rawUrl

  // 精确匹配公共路径: 如 /api/literature 和 /api/literature/search 都是独立条目
  if (PUBLIC_PATHS.has(path)) {
    return
  }
  // 公共前缀保护: 如果请求路径以某公共路径 + '/' 开头，也是公开的
  // 例如 /api/literature/xxx (未来可能添加的子路径)
  for (const p of PUBLIC_PATHS) {
    if (path.startsWith(p + '/')) {
      return
    }
  }

  const needsAuth = PROTECTED_PREFIXES.some(p => path.startsWith(p))
  if (!needsAuth) {
    return
  }

  // 内部服务（Python Agent）通过 x-user-id 头标识用户，无需 session token
  // 仅允许来自 localhost 的内部调用使用此机制
  // x-user-id 必须是合法 UUID v4 格式，防止路径遍历
  if (isLocalhost(req)) {
    const internalUserId = req.headers['x-user-id'] as string | undefined
    if (internalUserId && /^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/i.test(internalUserId)) {
      req.userId = internalUserId
      return
    }
  }

  const token = extractToken(req)
  if (!token) {
    reply.status(401).send({ detail: 'Not authenticated' })
    return
  }

  const session = validateSession(token)
  if (!session) {
    reply.status(401).send({ detail: 'Session expired, please login again' })
    return
  }

  req.userId = session.userId
}

export { authMiddleware, extractToken }
