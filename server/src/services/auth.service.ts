import { DatabaseSync } from 'node:sqlite'
import bcrypt from 'bcryptjs'
const { compare, hash } = bcrypt
import { randomBytes, randomUUID } from 'node:crypto'
import { mkdirSync, unlinkSync, rmdirSync } from 'node:fs'
import { dirname, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'
import { toSQLiteDatetime } from './user-data.service.js'

const __filename = fileURLToPath(import.meta.url)
const __dirname = dirname(__filename)
const AUTH_DB_PATH = resolve(__dirname, '..', '..', '..', 'data', 'auth.db')
const BCRYPT_ROUNDS = 12

interface UserRow {
  id: string
  username: string
  password_hash: string
  display_name: string
  created_at: string
  updated_at: string
  failed_attempts: number
  locked_until: string | null
}

interface SessionRow {
  token: string
  user_id: string
  created_at: string
  expires_at: string
}

interface AuditRow {
  id: number
  user_id: string | null
  username: string | null
  success: number
  ip_address: string | null
  user_agent: string | null
  created_at: string
}

function openAuthDB(): DatabaseSync {
  mkdirSync(dirname(AUTH_DB_PATH), { recursive: true })
  const db = new DatabaseSync(AUTH_DB_PATH)
  db.exec('PRAGMA journal_mode=WAL')
  db.exec('PRAGMA foreign_keys=ON')
  db.exec(`
    CREATE TABLE IF NOT EXISTS users (
      id TEXT PRIMARY KEY,
      username TEXT NOT NULL UNIQUE,
      password_hash TEXT NOT NULL,
      display_name TEXT,
      created_at TEXT NOT NULL DEFAULT (datetime('now')),
      updated_at TEXT NOT NULL DEFAULT (datetime('now')),
      failed_attempts INTEGER DEFAULT 0,
      locked_until TEXT
    );
    CREATE TABLE IF NOT EXISTS sessions (
      token TEXT PRIMARY KEY,
      user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
      created_at TEXT NOT NULL DEFAULT (datetime('now')),
      expires_at TEXT NOT NULL
    );
    CREATE INDEX IF NOT EXISTS idx_sessions_user_id ON sessions(user_id);
    CREATE INDEX IF NOT EXISTS idx_sessions_expires_at ON sessions(expires_at);
    CREATE TABLE IF NOT EXISTS login_audit (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      user_id TEXT,
      username TEXT,
      success INTEGER NOT NULL,
      ip_address TEXT,
      user_agent TEXT,
      created_at TEXT DEFAULT (datetime('now'))
    );
  `)
  return db
}

function initAuthDB(): void {
  const db = openAuthDB()
  db.close()
}

class UserExistsError extends Error {
  constructor(msg: string) {
    super(msg)
    this.name = 'UserExistsError'
  }
}

class InvalidCredentialsError extends Error {
  constructor(msg: string) {
    super(msg)
    this.name = 'InvalidCredentialsError'
  }
}

class AccountLockedError extends Error {
  constructor(msg: string) {
    super(msg)
    this.name = 'AccountLockedError'
  }
}

function createSession(db: DatabaseSync, userId: string): string {
  const token = randomBytes(32).toString('hex')
  const expiresAt = toSQLiteDatetime(new Date(Date.now() + 7 * 24 * 3600_000))
  db.prepare(
    'INSERT INTO sessions (token, user_id, expires_at) VALUES (?, ?, ?)'
  ).run(token, userId, expiresAt)
  return token
}

async function registerUser(
  username: string,
  password: string,
  displayName?: string
): Promise<{ user: Record<string, string>; token: string }> {
  if (!/^[a-zA-Z0-9_]{3,20}$/.test(username)) {
    throw new Error('3-20')
  }
  if (password.length < 8 || password.length > 128) {
    throw new Error('8-128')
  }

  const db = openAuthDB()
  const existing = db.prepare('SELECT id FROM users WHERE username = ?').get(username) as { id: string } | undefined
  if (existing) {
    db.close()
    throw new UserExistsError('用户名已存在，请直接登录')
  }

  const userId = randomUUID()
  const passwordHash = await hash(password, BCRYPT_ROUNDS)

  // 先创建 user.db，再写 auth.db。若 user.db 创建失败则不产生孤儿 auth 记录。
  // 若 auth.db 写入失败，清理已创建的 user.db 文件，防止孤儿数据目录。
  const { createUserDB } = await import('./user-data.service.js')
  const userDir = resolve(__dirname, '..', '..', '..', 'data', 'users', userId)
  const userDBPath = resolve(userDir, 'user.db')
  mkdirSync(userDir, { recursive: true })
  const userDB = createUserDB(userDBPath)
  userDB.close()

  try {
    db.exec('BEGIN')
    db.prepare(
      'INSERT INTO users (id, username, password_hash, display_name) VALUES (?, ?, ?, ?)'
    ).run(userId, username, passwordHash, displayName || username)

    const token = createSession(db, userId)
    db.exec('COMMIT')
    db.close()

    return {
      user: {
        id: userId,
        username,
        displayName: displayName || username,
        createdAt: new Date().toISOString(),
      },
      token,
    }
  } catch (err) {
    // auth.db 写入失败，回滚事务并清理已创建的孤儿 user.db
    try { db.exec('ROLLBACK') } catch { console.warn('[auth] registerUser: ROLLBACK 失败') }
    db.close()
    // WAL 模式下 SQLite 会创建 .db-wal 和 .db-shm 伴随文件，需一并清理
    // 否则 rmdirSync 会因目录非空而失败
    try { unlinkSync(userDBPath) } catch { console.warn(`[auth] registerUser: 清理孤儿 user.db 失败: ${userDBPath}`) }
    try { unlinkSync(userDBPath + '-wal') } catch { /* WAL 文件可能不存在，忽略 */ }
    try { unlinkSync(userDBPath + '-shm') } catch { /* SHM 文件可能不存在，忽略 */ }
    try { rmdirSync(userDir) } catch { console.warn(`[auth] registerUser: 清理孤儿用户目录失败: ${userDir}`) }
    throw err
  }
}

async function loginUser(
  username: string,
  password: string,
  ip?: string,
  userAgent?: string
): Promise<{ user: Record<string, string>; token: string }> {
  const db = openAuthDB()
  const row = db.prepare(
    'SELECT id, username, password_hash, display_name, created_at, failed_attempts, locked_until FROM users WHERE username = ?'
  ).get(username) as UserRow | undefined

  if (!row) {
    db.prepare(
      'INSERT INTO login_audit (username, success, ip_address, user_agent) VALUES (?, 0, ?, ?)'
    ).run(username, ip || null, userAgent || null)
    db.close()
    throw new InvalidCredentialsError('')
  }

  const lockedUntil = row.locked_until
  // locked_until 是 UTC 时间格式 "YYYY-MM-DD HH:MM:SS"，与 datetime('now') 一致
  // Date.now() 返回 UTC epoch ms，lockedDate 解析为 UTC 后直接比较
  if (lockedUntil && lockedUntil > toSQLiteDatetime(new Date())) {
    const lockedDate = new Date(lockedUntil.replace(' ', 'T') + 'Z')
    const remainingMin = Math.ceil((lockedDate.getTime() - Date.now()) / 60000)
    db.close()
    throw new AccountLockedError(`${remainingMin}`)
  }

  const valid = await compare(password, row.password_hash)
  if (!valid) {
    const newAttempts = (row.failed_attempts || 0) + 1
    if (newAttempts >= 5) {
      db.prepare(
        "UPDATE users SET failed_attempts = ?, locked_until = datetime('now', '+10 minutes') WHERE id = ?"
      ).run(newAttempts, row.id)
    } else {
      db.prepare(
        'UPDATE users SET failed_attempts = ? WHERE id = ?'
      ).run(newAttempts, row.id)
    }
    db.prepare(
      'INSERT INTO login_audit (user_id, username, success, ip_address, user_agent) VALUES (?, ?, 0, ?, ?)'
    ).run(row.id, username, ip || null, userAgent || null)
    db.close()
    throw new InvalidCredentialsError('')
  }

  db.prepare(
    'UPDATE users SET failed_attempts = 0, locked_until = NULL WHERE id = ?'
  ).run(row.id)

  db.prepare(
    'INSERT INTO login_audit (user_id, username, success, ip_address, user_agent) VALUES (?, ?, 1, ?, ?)'
  ).run(row.id, username, ip || null, userAgent || null)

  const token = createSession(db, row.id)
  db.close()

  return {
    user: {
      id: row.id,
      username: row.username,
      displayName: row.display_name,
      createdAt: row.created_at,
    },
    token,
  }
}

function validateSession(token: string): { userId: string } | null {
  const db = openAuthDB()
  const row = db.prepare(
    "SELECT user_id FROM sessions WHERE token = ? AND expires_at > datetime('now')"
  ).get(token) as { user_id: string } | undefined
  db.close()
  if (!row) return null
  return { userId: row.user_id }
}

function deleteSession(token: string): void {
  const db = openAuthDB()
  db.prepare('DELETE FROM sessions WHERE token = ?').run(token)
  db.close()
}

function getUserById(userId: string): Record<string, string> | null {
  const db = openAuthDB()
  const row = db.prepare(
    'SELECT id, username, display_name, created_at FROM users WHERE id = ?'
  ).get(userId) as { id: string; username: string; display_name: string; created_at: string } | undefined
  db.close()
  if (!row) return null
  return {
    id: row.id,
    username: row.username,
    displayName: row.display_name,
    createdAt: row.created_at,
  }
}

function startSessionCleanup(): void {
  // 每小时清理过期 sessions（TTL=7天，在 createSession 中设定）
  // 注意：此清理不联动 agent_checkpoints.db 和 user.db。
  // Python 端 ConversationMemory.cleanup_expired_sessions 负责清理
  // agent_checkpoints.db（TTL=30天），并同步清理 user.db 的 conversations 表。
  // 两个 TTL 不一致（7天 vs 30天）是有意为之：
  // sessions 短 TTL 为了安全，checkpoints 长 TTL 为了方便用户恢复对话。
  const cleanup = () => {
    const db = openAuthDB()
    db.exec("DELETE FROM sessions WHERE expires_at < datetime('now')")
    db.close()
  }
  cleanup()
  setInterval(cleanup, 3600_000)
}

export {
  AUTH_DB_PATH,
  BCRYPT_ROUNDS,
  initAuthDB,
  openAuthDB,
  registerUser,
  loginUser,
  validateSession,
  deleteSession,
  getUserById,
  startSessionCleanup,
  UserExistsError,
  InvalidCredentialsError,
  AccountLockedError,
}
