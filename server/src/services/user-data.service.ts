import { DatabaseSync } from 'node:sqlite'
import { mkdirSync, existsSync, readdirSync } from 'node:fs'
import { dirname, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'

const __filename = fileURLToPath(import.meta.url)
const __dirname = dirname(__filename)

export const USER_DATA_BASE = resolve(__dirname, '..', '..', '..', 'data', 'users')

/** UUID v4 格式校验，防止路径遍历攻击 */
const UUID_REGEX = /^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/i

function validateUserId(userId: string): void {
  if (!UUID_REGEX.test(userId)) {
    throw new Error(`Invalid user ID format: ${userId}`)
  }
}

// ===== 连接缓存: 按 userId 缓存 DatabaseSync 实例，减少重复打开/关闭开销 =====
const _dbCache = new Map<string, DatabaseSync>()
const CACHE_CLEANUP_INTERVAL_MS = 10 * 60 * 1000 // 10 分钟清理闲置连接
const CACHE_IDLE_MAX_MS = 5 * 60 * 1000 // 5 分钟未使用则关闭

// 记录每个缓存连接的最后使用时间
const _dbLastUsed = new Map<string, number>()

// 启动定时清理
let _cleanupTimer: ReturnType<typeof setInterval> | null = null
let _cleanupTimerActive = false  // 防止竞态：clearInterval 后 _cleanupTimer 仍非 null 的窗口期
function _ensureCleanupTimer(): void {
  if (_cleanupTimerActive) return
  _cleanupTimerActive = true
  _cleanupTimer = setInterval(() => {
    const now = Date.now()
    for (const [userId, lastUsed] of _dbLastUsed) {
      if (now - lastUsed > CACHE_IDLE_MAX_MS) {
        const db = _dbCache.get(userId)
        if (db) {
          try { db.close() } catch { console.warn(`[db-cache] cleanup: 关闭 ${userId} 连接失败，已从缓存移除`) }
        }
        _dbCache.delete(userId)
        _dbLastUsed.delete(userId)
      }
    }
    // 如果缓存为空则停止定时器，下次使用时重新启动
    if (_dbCache.size === 0) {
      clearInterval(_cleanupTimer!)
      _cleanupTimer = null
      _cleanupTimerActive = false
    }
  }, CACHE_CLEANUP_INTERVAL_MS)
}

/** 将 Date 转为 SQLite datetime 兼容格式 (YYYY-MM-DD HH:MM:SS)，使用 UTC 时间。
 *  与 datetime('now') 输出格式一致，确保字符串可比较。
 *  所有时间存储统一使用 UTC，与时区无关，确保：
 *  1. Node.js 与 Python 端时间比较一致性
 *  2. 服务器迁移时区后不会产生时间偏差
 *  3. 与 Python TTL 清理 (datetime.now(timezone.utc)) 对齐 */
export function toSQLiteDatetime(d: Date): string {
  const pad = (n: number) => String(n).padStart(2, '0')
  return `${d.getUTCFullYear()}-${pad(d.getUTCMonth() + 1)}-${pad(d.getUTCDate())} ${pad(d.getUTCHours())}:${pad(d.getUTCMinutes())}:${pad(d.getUTCSeconds())}`
}

/** 将 SQLite datetime 字符串转为前端可安全解析的 ISO 8601 格式（UTC）。
 *  兼容两种输入格式: SQLite "YYYY-MM-DD HH:MM:SS" 和 ISO "YYYY-MM-DDTHH:MM:SS*"
 *
 *  所有时间统一使用 UTC 存储 (datetime('now'))，转换时追加 'Z' 后缀。
 *  浏览器 new Date("YYYY-MM-DDTHH:MM:SSZ") 会正确解析为 UTC 并显示为本地时间。
 *
 *  对于不含 'Z' 的历史数据（升级前以 localtime 存储），追加 'Z' 会导致
 *  浏览器显示偏移，但历史数据量极小（仅测试环境），手动校正即可。
 *
 *  无效输入返回 epoch (1970-01-01T00:00:00Z)，避免前端产生 NaN。 */
export function sqliteDatetimeToISO(raw: unknown): string {
  if (typeof raw !== 'string' || !raw) {
    console.warn('[sqliteDatetimeToISO] received non-string or empty value:', raw)
    return '1970-01-01T00:00:00Z'
  }
  // 如果包含 'T'，已经是 ISO 格式（可能带 Z 后缀，兼容处理）
  if (raw.includes('T')) {
    return raw.endsWith('Z') ? raw : raw + 'Z'
  }
  // SQLite 格式 "YYYY-MM-DD HH:MM:SS" -> "YYYY-MM-DDTHH:MM:SSZ"
  const m = raw.match(/^(\d{4}-\d{2}-\d{2}) (\d{2}:\d{2}:\d{2})$/)
  if (m) return `${m[1]}T${m[2]}Z`
  // 无法识别的格式，记录警告并返回安全哨兵值
  console.warn('[sqliteDatetimeToISO] unrecognized format:', raw)
  return '1970-01-01T00:00:00Z'
}

// ===== User DB Schema（单一定义，供 createUserDB 和 openUserDB 共用）=====
// 修改 schema 时只需改此处，createUserDB 和 openUserDB 会自动同步。
const USER_DB_SCHEMA = `
  CREATE TABLE IF NOT EXISTS annotations (
      id              TEXT PRIMARY KEY,
      chapter_number  INTEGER NOT NULL,
      paragraph_index INTEGER NOT NULL,
      category        TEXT NOT NULL,
      entity          TEXT NOT NULL,
      start_char      INTEGER NOT NULL,
      end_char        INTEGER NOT NULL,
      quote           TEXT,
      explanation     TEXT,
      confidence      TEXT DEFAULT 'medium' CHECK(confidence IN ('high', 'medium', 'low', 'unverified')),
      color           TEXT,
      source          TEXT NOT NULL DEFAULT 'user',
      created_at      TEXT NOT NULL DEFAULT (datetime('now')),
      updated_at      TEXT NOT NULL DEFAULT (datetime('now'))
  );
  CREATE INDEX IF NOT EXISTS idx_annotations_chapter_paragraph
      ON annotations(chapter_number, paragraph_index);
  CREATE INDEX IF NOT EXISTS idx_annotations_category ON annotations(category);
  CREATE INDEX IF NOT EXISTS idx_annotations_chapter_source
      ON annotations(chapter_number, source);

  CREATE TABLE IF NOT EXISTS marginalia (
      id                 TEXT PRIMARY KEY,
      annotation_id      TEXT REFERENCES annotations(id) ON DELETE SET NULL,
      chapter_number     INTEGER NOT NULL,
      paragraph_index    INTEGER NOT NULL,
      anchor_char_offset INTEGER,
      content            TEXT NOT NULL,
      source             TEXT DEFAULT 'user',
      created_at         TEXT NOT NULL DEFAULT (datetime('now')),
      updated_at         TEXT NOT NULL DEFAULT (datetime('now'))
  );
  CREATE INDEX IF NOT EXISTS idx_marginalia_chapter_paragraph
      ON marginalia(chapter_number, paragraph_index);
  CREATE INDEX IF NOT EXISTS idx_marginalia_chapter_source
      ON marginalia(chapter_number, source);

  CREATE TABLE IF NOT EXISTS user_preferences (
      key        TEXT PRIMARY KEY,
      value      TEXT,
      created_at TEXT NOT NULL DEFAULT (datetime('now')),
      updated_at TEXT NOT NULL DEFAULT (datetime('now'))
  );

  CREATE TABLE IF NOT EXISTS conversations (
      id          TEXT PRIMARY KEY,
      title       TEXT,
      created_at  TEXT NOT NULL DEFAULT (datetime('now')),
      updated_at  TEXT NOT NULL DEFAULT (datetime('now'))
  );

  CREATE TABLE IF NOT EXISTS messages (
      id              TEXT PRIMARY KEY,
      conversation_id TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
      role            TEXT NOT NULL CHECK(role IN ('user', 'assistant', 'system')),
      content         TEXT NOT NULL,
      sources         TEXT,
      tool_calls      TEXT,
      evolution_card  TEXT,
      reasoning       TEXT,
      created_at      TEXT NOT NULL DEFAULT (datetime('now'))
  );
  CREATE INDEX IF NOT EXISTS idx_messages_conversation
      ON messages(conversation_id, created_at);
`

/**
 * 迁移：为旧版本 user_preferences 表补充时间戳列。
 * 使用 PRAGMA table_info 预检列是否存在，避免每次都尝试 ALTER TABLE 再捕获异常。
 * 在 openUserDB 和 migrateAllUserDBs 中共用，确保无论从哪个路径进入都能触发迁移。
 */
export function ensureUserDBSchema(db: DatabaseSync): void {
  // Step 1: 补充表结构（CREATE TABLE IF NOT EXISTS / CREATE INDEX IF NOT EXISTS）
  db.exec(USER_DB_SCHEMA)

  // Step 2: 迁移 user_preferences 表 - 补充时间戳列
  const cols = db.prepare("PRAGMA table_info(user_preferences)").all() as Array<{ name: string }>
  const colNames = new Set(cols.map(c => c.name))

  if (!colNames.has('created_at')) {
    db.exec("ALTER TABLE user_preferences ADD COLUMN created_at TEXT NOT NULL DEFAULT (datetime('now'))")
    console.warn('[user-data] migration: user_preferences.created_at 列已补充')
  }
  if (!colNames.has('updated_at')) {
    db.exec("ALTER TABLE user_preferences ADD COLUMN updated_at TEXT NOT NULL DEFAULT (datetime('now'))")
    console.warn('[user-data] migration: user_preferences.updated_at 列已补充')
  }

  // v17: messages 表补充 reasoning 列（Agent 思考过程，历史消息可展开查看）
  const msgCols = db.prepare("PRAGMA table_info(messages)").all() as Array<{ name: string }>
  if (!msgCols.some(c => c.name === 'reasoning')) {
    db.exec("ALTER TABLE messages ADD COLUMN reasoning TEXT")
    console.warn('[user-data] migration: messages.reasoning 列已补充')
  }

  // v19: messages 表补充 report_card 列（长报告完整 Markdown，随消息持久化）
  const msgColsV19 = db.prepare("PRAGMA table_info(messages)").all() as Array<{ name: string }>
  if (!msgColsV19.some(c => c.name === 'report_card')) {
    db.exec("ALTER TABLE messages ADD COLUMN report_card TEXT")
    console.warn('[user-data] migration: messages.report_card 列已补充')
  }

  // Step 3: 修正 annotations 表中可能存在的非法 confidence 值
  // SQLite 不支持 ALTER TABLE ADD CHECK，旧数据库依赖此迁移 + normalizeConfidence() 双重保障
  const validValues = ['high', 'medium', 'low', 'unverified']
  const placeholders = validValues.map(() => '?').join(',')
  const result = db.prepare(
    `UPDATE annotations SET confidence = 'medium'
     WHERE confidence NOT IN (${placeholders}) OR confidence IS NULL`
  ).run(...validValues)
  if (result.changes > 0) {
    console.warn(`[user-data] 修正 ${result.changes} 条非法 confidence 值（含 NULL）-> 'medium'`)
  }
}

/**
 * 启动时全局迁移：遍历 data/users/ 下所有用户目录，
 * 对每个 user.db 执行 schema 升级（补列、补索引、修正数据）。
 *
 * 设计意图：openUserDB 内部也调用相同的 ensureUserDBSchema，
 * 此函数确保在服务器启动时主动扫描所有现存用户数据库并完成迁移，
 * 不依赖用户登录/访问来触发。
 */
export function migrateAllUserDBs(): void {
  if (!existsSync(USER_DATA_BASE)) return

  let entries: string[]
  try {
    entries = readdirSync(USER_DATA_BASE)
  } catch {
    console.warn('[user-data] migrateAllUserDBs: 无法读取 users 目录')
    return
  }

  let migrated = 0
  let errors = 0

  for (const name of entries) {
    const userDir = resolve(USER_DATA_BASE, name)
    // 跳过非目录条目和非 UUID 命名的目录
    if (!UUID_REGEX.test(name)) continue
    if (!existsSync(userDir)) continue

    const dbPath = resolve(userDir, 'user.db')
    if (!existsSync(dbPath)) continue

    try {
      const db = new DatabaseSync(dbPath)
      try {
        db.exec('PRAGMA journal_mode=WAL')
        db.exec('PRAGMA foreign_keys=ON')
        ensureUserDBSchema(db)
        migrated++
      } finally {
        db.close()
      }
    } catch (err) {
      errors++
      console.warn(`[user-data] migrateAllUserDBs: 迁移 ${name} 失败:`, err)
    }
  }

  if (migrated > 0 || errors > 0) {
    console.log(`[user-data] migrateAllUserDBs: 已扫描 ${migrated + errors} 个用户数据库, 成功 ${migrated}, 失败 ${errors}`)
  }
}

/**
 * 创建新用户的 user.db 文件（仅在注册时调用）。
 * 从模板创建，包含完整的表结构和索引。
 */
export function createUserDB(dbPath: string): DatabaseSync {
  mkdirSync(dirname(dbPath), { recursive: true })

  const db = new DatabaseSync(dbPath)
  db.exec('PRAGMA journal_mode=WAL')
  db.exec('PRAGMA foreign_keys=ON')

  db.exec(USER_DB_SCHEMA)

  return db
}

/**
 * 打开用户的 user.db 连接（带缓存复用和自动 schema 初始化）。
 *
 * 如果用户目录或数据库文件因意外丢失，此函数会自动重建目录和 schema。
 * 缓存连接在闲置 5 分钟后自动关闭。
 */
export function openUserDB(userId: string): DatabaseSync {
  validateUserId(userId)

  // 检查缓存
  const cached = _dbCache.get(userId)
  if (cached) {
    // 健康检查：验证缓存连接仍然可用（DB 文件未被外部删除等）
    try {
      cached.prepare('SELECT 1').get()
      _dbLastUsed.set(userId, Date.now())
      return cached
    } catch {
      // 连接已失效，关闭并从缓存移除，重新打开
      console.warn(`[db-cache] ${userId} 连接健康检查失败，重建连接`)
      try { cached.close() } catch { console.warn(`[db-cache] ${userId} 旧连接关闭失败，跳过`) }
      _dbCache.delete(userId)
      _dbLastUsed.delete(userId)
    }
  }

  const dbPath = resolve(USER_DATA_BASE, userId, 'user.db')

  // 确保目录存在（即使因意外被删除也能恢复）
  mkdirSync(dirname(dbPath), { recursive: true })

  // 检测是否为意外重建：如果 DB 文件之前不存在，重建后发出告警
  const dbExisted = existsSync(dbPath)

  const db = new DatabaseSync(dbPath)
  db.exec('PRAGMA journal_mode=WAL')
  db.exec('PRAGMA foreign_keys=ON')

  // 确保 schema 存在并执行迁移（补列、补索引、修正数据）
  ensureUserDBSchema(db)

  // 如果 DB 文件不存在（目录丢失导致重建），发出告警
  // 此时 schema 已恢复但用户原有数据已永久丢失
  if (!dbExisted) {
    console.warn(`[user-data] WARNING: user.db 文件丢失后自动重建: ${dbPath}。用户标注数据可能已丢失。`)
  }

  // 加入缓存
  _dbCache.set(userId, db)
  _dbLastUsed.set(userId, Date.now())
  _ensureCleanupTimer()

  return db
}

/**
 * 关闭指定用户的缓存连接（用于用户登出等场景）。
 */
export function closeUserDB(userId: string): void {
  const db = _dbCache.get(userId)
  if (db) {
    try { db.close() } catch { console.warn(`[db-cache] closeUserDB: ${userId} 连接关闭失败`) }
  }
  _dbCache.delete(userId)
  _dbLastUsed.delete(userId)
}

// ===== 查询辅助函数 =====

export function getAnnotationRows(db: DatabaseSync, chapterNumber: number): Array<Record<string, unknown>> {
  return db.prepare(
    'SELECT * FROM annotations WHERE chapter_number = ? ORDER BY paragraph_index, start_char'
  ).all(chapterNumber) as Array<Record<string, unknown>>
}

export function getMarginaliaRows(db: DatabaseSync, chapterNumber: number): Array<Record<string, unknown>> {
  return db.prepare(
    'SELECT * FROM marginalia WHERE chapter_number = ? ORDER BY paragraph_index'
  ).all(chapterNumber) as Array<Record<string, unknown>>
}

export function getPreference(db: DatabaseSync, key: string): string | null {
  const row = db.prepare(
    'SELECT value FROM user_preferences WHERE key = ?'
  ).get(key) as { value: string } | undefined
  return row ? row.value : null
}

export function savePreference(db: DatabaseSync, key: string, value: string): void {
  db.prepare(
    `INSERT INTO user_preferences (key, value, created_at, updated_at)
     VALUES (?, ?, datetime('now'), datetime('now'))
     ON CONFLICT(key) DO UPDATE SET value=excluded.value, updated_at=datetime('now')`
  ).run(key, value)
}

export function getConversationRows(db: DatabaseSync): Array<Record<string, unknown>> {
  return db.prepare(
    'SELECT * FROM conversations ORDER BY updated_at DESC'
  ).all() as Array<Record<string, unknown>>
}

export function upsertConversation(db: DatabaseSync, id: string, title: string | null): void {
  // NULL 表示未知标题，比空字符串语义更准确
  const titleValue = title || null
  db.prepare(
    `INSERT INTO conversations (id, title, updated_at) VALUES (?, ?, datetime('now'))
     ON CONFLICT(id) DO UPDATE SET
       title=CASE WHEN excluded.title IS NOT NULL THEN excluded.title ELSE conversations.title END,
       updated_at=excluded.updated_at`
  ).run(id, titleValue)
}

export function deleteConversationRow(db: DatabaseSync, id: string): void {
  db.prepare('DELETE FROM conversations WHERE id = ?').run(id)
}

// ===== Messages CRUD =====

export interface MessageRow {
  id: string
  conversation_id: string
  role: 'user' | 'assistant' | 'system'
  content: string
  sources: string | null
  tool_calls: string | null
  evolution_card: string | null
  reasoning: string | null
  report_card: string | null
  created_at: string
}

export function insertMessage(
  db: DatabaseSync,
  id: string,
  conversationId: string,
  role: 'user' | 'assistant' | 'system',
  content: string,
  sources: string | null,
  toolCalls: string | null,
  evolutionCard: string | null,
  reasoning: string | null = null,
  reportCard: string | null = null,
): void {
  db.prepare(
    `INSERT INTO messages (id, conversation_id, role, content, sources, tool_calls, evolution_card, reasoning, report_card)
     VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)`
  ).run(id, conversationId, role, content, sources, toolCalls, evolutionCard, reasoning, reportCard)
}

export function getMessagesByConversation(db: DatabaseSync, conversationId: string): MessageRow[] {
  return db.prepare(
    'SELECT * FROM messages WHERE conversation_id = ? ORDER BY created_at ASC'
  ).all(conversationId) as unknown as MessageRow[]
}

export function deleteMessagesByConversation(db: DatabaseSync, conversationId: string): void {
  db.prepare('DELETE FROM messages WHERE conversation_id = ?').run(conversationId)
}

/**
 * 清理 agent_checkpoints.db 中指定 thread_id 的检查点数据。
 *
 * 与 Python 端 cleanup_expired_sessions 的清理逻辑保持一致，
 * 确保通过 Node.js API 删除对话时能够同步清理检查点数据，
 * 避免孤儿 checkpoints 占用磁盘直到 30 天 TTL 清理。
 *
 * 如果 agent_checkpoints.db 不存在或表尚未创建（用户从未使用 Agent），
 * 此函数静默跳过，不视为错误。
 */
export function deleteCheckpointsForConversation(userId: string, threadId: string): void {
  validateUserId(userId)
  validateUserId(threadId)  // threadId 也是 UUID v4 格式，复用校验

  const cpPath = resolve(USER_DATA_BASE, userId, 'agent_checkpoints.db')

  if (!existsSync(cpPath)) return

  const cpDb = new DatabaseSync(cpPath)
  try {
    cpDb.exec('PRAGMA journal_mode=WAL')

    // 仅在表存在时执行删除：新用户可能从未使用 Agent，表尚未创建
    const tableCheck = cpDb.prepare(
      "SELECT name FROM sqlite_master WHERE type='table' AND name='checkpoints'"
    ).get() as { name: string } | undefined

    if (!tableCheck) return

    cpDb.prepare('DELETE FROM checkpoint_blobs WHERE thread_id = ?').run(threadId)
    cpDb.prepare('DELETE FROM checkpoint_writes WHERE thread_id = ?').run(threadId)
    cpDb.prepare('DELETE FROM checkpoints WHERE thread_id = ?').run(threadId)
  } finally {
    cpDb.close()
  }
}

export function confidenceNumberToText(n: number): string {
  if (n >= 0.8) return 'high'
  if (n >= 0.5) return 'medium'
  if (n > 0) return 'low'
  return 'unverified'
}

const VALID_CONFIDENCE_VALUES = new Set(['high', 'medium', 'low', 'unverified'])

/** 校验并规范化 confidence 值，非法值统一降级为 'medium' */
export function normalizeConfidence(raw: unknown): string {
  if (typeof raw === 'string' && VALID_CONFIDENCE_VALUES.has(raw)) return raw
  if (typeof raw === 'number') return confidenceNumberToText(raw)
  return 'medium'
}

export function confidenceTextToNumber(t: string): number {
  switch (t) {
    case 'high': return 0.9
    case 'medium': return 0.6
    case 'low': return 0.3
    default: return 0
  }
}
