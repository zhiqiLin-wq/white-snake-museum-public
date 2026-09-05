import type { FastifyInstance, FastifyRequest, FastifyReply } from 'fastify'
import {
  openUserDB,
  getConversationRows,
  upsertConversation,
  deleteConversationRow,
  deleteCheckpointsForConversation,
  insertMessage,
  getMessagesByConversation,
  deleteMessagesByConversation,
  sqliteDatetimeToISO,
} from '../services/user-data.service.js'

/** Python Agent 服务地址（删除对话时回调清理记忆） */
const AGENT_URL = process.env.AGENT_URL || 'http://127.0.0.1:8000'

/** UUID v4 格式校验，conversation ID 对应 LangGraph thread_id */
const CONVERSATION_ID_REGEX = /^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/i

function validateConversationId(id: string): void {
  if (!CONVERSATION_ID_REGEX.test(id)) {
    throw new Error(`Invalid conversation ID format: ${id}`)
  }
}

async function conversationRoutes(app: FastifyInstance): Promise<void> {

  app.get('/api/conversations', async (req: FastifyRequest, reply: FastifyReply) => {
    const userId = req.userId
    if (!userId) {
      return reply.status(401).send({ detail: 'Not authenticated' })
    }

    const db = openUserDB(userId)
    const rows = getConversationRows(db)
    const conversations = rows.map(row => ({
      id: row.id,
      title: row.title,
      createdAt: Date.parse(sqliteDatetimeToISO(row.created_at)),
      updatedAt: Date.parse(sqliteDatetimeToISO(row.updated_at)),
    }))
    return reply.status(200).send({ conversations })
  })

  app.post('/api/conversations', async (req: FastifyRequest, reply: FastifyReply) => {
    const userId = req.userId
    if (!userId) {
      return reply.status(401).send({ detail: 'Not authenticated' })
    }

    const body = req.body as { id?: string; title?: string }
    if (!body.id) {
      return reply.status(400).send({ detail: 'id is required' })
    }

    // 校验 conversation ID 格式 (UUID v4)，防止异常数据写入
    try {
      validateConversationId(body.id)
    } catch {
      return reply.status(400).send({ detail: `Invalid conversation ID format: ${body.id}` })
    }

    const db = openUserDB(userId)
    upsertConversation(db, body.id, body.title || null)
    return reply.status(200).send({ status: 'ok' })
  })

  app.delete('/api/conversations/:id', async (req: FastifyRequest, reply: FastifyReply) => {
    const userId = req.userId
    if (!userId) {
      return reply.status(401).send({ detail: 'Not authenticated' })
    }

    const { id } = req.params as { id: string }

    // 校验 conversation ID 格式 (UUID v4)，防止异常数据写入
    try {
      validateConversationId(id)
    } catch {
      return reply.status(400).send({ detail: `Invalid conversation ID format: ${id}` })
    }

    const db = openUserDB(userId)
    deleteConversationRow(db, id)
    // 同步清理 agent_checkpoints.db 中的检查点数据（失败不阻断删除）
    try {
      deleteCheckpointsForConversation(userId, id)
    } catch (err) {
      console.warn(`[conversations] 清理 checkpoints 失败 (conversation=${id}):`, err)
    }
    // 同步清理 Python 侧记忆/召回（长期记忆 + 召回消息 + FIFO + 向量，失败不阻断删除）
    try {
      const resp = await fetch(`${AGENT_URL}/memory/conversation/${id}`, {
        method: 'DELETE',
        headers: { 'x-user-id': userId },
      })
      if (!resp.ok) {
        console.warn(`[conversations] 清理 Python 记忆失败 (conversation=${id}): HTTP ${resp.status}`)
      }
    } catch (err) {
      console.warn(`[conversations] 清理 Python 记忆失败 (conversation=${id}):`, err)
    }
    return reply.status(200).send({ status: 'ok' })
  })

  // ===== Messages CRUD =====

  app.get('/api/conversations/:id/messages', async (req: FastifyRequest, reply: FastifyReply) => {
    const userId = req.userId
    if (!userId) {
      return reply.status(401).send({ detail: 'Not authenticated' })
    }

    const { id } = req.params as { id: string }
    try {
      validateConversationId(id)
    } catch {
      return reply.status(400).send({ detail: `Invalid conversation ID format: ${id}` })
    }

    const db = openUserDB(userId)
    const rows = getMessagesByConversation(db, id)
    const messages = rows.map(row => ({
      id: row.id,
      conversationId: row.conversation_id,
      role: row.role,
      content: row.content,
      sources: row.sources ? JSON.parse(row.sources) : undefined,
      toolCalls: row.tool_calls ? JSON.parse(row.tool_calls) : undefined,
      evolutionCard: row.evolution_card ? JSON.parse(row.evolution_card) : undefined,
      reportCard: row.report_card ? JSON.parse(row.report_card) : undefined,
      reasoning: row.reasoning || undefined,
      timestamp: Date.parse(sqliteDatetimeToISO(row.created_at)),
    }))
    return reply.status(200).send({ messages })
  })

  app.post('/api/conversations/:id/messages', async (req: FastifyRequest, reply: FastifyReply) => {
    const userId = req.userId
    if (!userId) {
      return reply.status(401).send({ detail: 'Not authenticated' })
    }

    const { id } = req.params as { id: string }
    try {
      validateConversationId(id)
    } catch {
      return reply.status(400).send({ detail: `Invalid conversation ID format: ${id}` })
    }

    const body = req.body as {
      messageId?: string
      role?: string
      content?: string
      sources?: string
      toolCalls?: string
      evolutionCard?: string
      reportCard?: string
      reasoning?: string
    }
    if (!body.messageId || !body.role || body.content === undefined) {
      return reply.status(400).send({ detail: 'messageId, role, and content are required' })
    }
    if (!['user', 'assistant', 'system'].includes(body.role)) {
      return reply.status(400).send({ detail: 'role must be user, assistant, or system' })
    }

    const db = openUserDB(userId)
    // 确保 conversation 存在
    upsertConversation(db, id, null)
    insertMessage(
      db,
      body.messageId,
      id,
      body.role as 'user' | 'assistant' | 'system',
      body.content,
      body.sources || null,
      body.toolCalls || null,
      body.evolutionCard || null,
      body.reasoning || null,
      body.reportCard || null,
    )
    return reply.status(200).send({ status: 'ok' })
  })

  app.delete('/api/conversations/:id/messages', async (req: FastifyRequest, reply: FastifyReply) => {
    const userId = req.userId
    if (!userId) {
      return reply.status(401).send({ detail: 'Not authenticated' })
    }

    const { id } = req.params as { id: string }
    try {
      validateConversationId(id)
    } catch {
      return reply.status(400).send({ detail: `Invalid conversation ID format: ${id}` })
    }

    const db = openUserDB(userId)
    deleteMessagesByConversation(db, id)
    return reply.status(200).send({ status: 'ok' })
  })
}

export { conversationRoutes }
