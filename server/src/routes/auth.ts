import type { FastifyInstance, FastifyRequest, FastifyReply } from 'fastify'
import {
  registerUser,
  loginUser,
  validateSession,
  deleteSession,
  getUserById,
  UserExistsError,
  InvalidCredentialsError,
  AccountLockedError,
} from '../services/auth.service.js'
import { closeUserDB, openUserDB, getConversationRows, getAnnotationRows } from '../services/user-data.service.js'
import { extractToken } from '../middleware/auth.js'

async function authRoutes(app: FastifyInstance): Promise<void> {

  app.post('/api/auth/register', async (req: FastifyRequest, reply: FastifyReply) => {
    const body = req.body as { username?: string; password?: string; display_name?: string }
    const username = body.username || ''
    const password = body.password || ''
    const displayName = body.display_name

    if (!username || !password) {
      return reply.status(422).send({ detail: 'username and password are required' })
    }

    try {
      const result = await registerUser(username, password, displayName)
      return reply.status(201).send(result)
    } catch (err: unknown) {
      if (err instanceof UserExistsError) {
        return reply.status(409).send({ detail: err.message || '' })
      }
      if (err instanceof Error) {
        return reply.status(422).send({ detail: err.message })
      }
      return reply.status(500).send({ detail: 'Internal server error' })
    }
  })

  app.post('/api/auth/login', async (req: FastifyRequest, reply: FastifyReply) => {
    const body = req.body as { username?: string; password?: string }
    const username = body.username || ''
    const password = body.password || ''

    if (!username || !password) {
      return reply.status(422).send({ detail: 'username and password are required' })
    }

    try {
      const ip = req.ip
      const userAgent = req.headers['user-agent']
      const result = await loginUser(username, password, ip, userAgent)
      return reply.status(200).send(result)
    } catch (err: unknown) {
      if (err instanceof InvalidCredentialsError) {
        return reply.status(401).send({ detail: '' })
      }
      if (err instanceof AccountLockedError) {
        return reply.status(423).send({ detail: `${err.message}` })
      }
      return reply.status(500).send({ detail: 'Internal server error' })
    }
  })

  app.post('/api/auth/logout', async (req: FastifyRequest, reply: FastifyReply) => {
    const token = extractToken(req)
    if (token) {
      const session = validateSession(token)
      if (session) {
        // 关闭用户缓存数据库连接，释放资源
        closeUserDB(session.userId)
      }
      deleteSession(token)
    }
    return reply.status(200).send({ status: 'ok' })
  })

  app.get('/api/auth/me', async (req: FastifyRequest, reply: FastifyReply) => {
    const token = extractToken(req)
    if (!token) {
      return reply.status(401).send({ detail: 'Not authenticated' })
    }

    const session = validateSession(token)
    if (!session) {
      return reply.status(401).send({ detail: 'Not authenticated' })
    }

    const user = getUserById(session.userId)
    if (!user) {
      return reply.status(401).send({ detail: 'User not found' })
    }

    return reply.status(200).send(user)
  })

  app.get('/api/auth/stats', async (req: FastifyRequest, reply: FastifyReply) => {
    const token = extractToken(req)
    if (!token) {
      return reply.status(401).send({ detail: 'Not authenticated' })
    }

    const session = validateSession(token)
    if (!session) {
      return reply.status(401).send({ detail: 'Not authenticated' })
    }

    const userId = session.userId
    const db = openUserDB(userId)

    // 统计对话数量
    const conversations = getConversationRows(db)
    const conversationCount = conversations.length

    // 统计标注数量（需要遍历所有章节）
    let annotationCount = 0
    try {
      for (let ch = 1; ch <= 7; ch++) {
        const rows = getAnnotationRows(db, ch)
        // 统计所有标注（包含用户手动标注和 agent 自动标注）
        annotationCount += rows.length
      }
    } catch {
      // 统计失败不阻塞其他数据
      annotationCount = 0
    }

    const lastActive = conversations.length > 0
      ? conversations.reduce((max, row) => {
          const t = (row as Record<string, unknown>).updated_at as string || ''
          return t > max ? t : max
        }, '')
      : null

    return reply.status(200).send({
      conversationCount,
      annotationCount,
      lastActive,
    })
  })
}

export { authRoutes }
