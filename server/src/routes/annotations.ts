import type { FastifyInstance, FastifyRequest, FastifyReply } from 'fastify'
import {
  openUserDB,
  getAnnotationRows,
  getMarginaliaRows,
  getPreference,
  savePreference,
  normalizeConfidence,
  toSQLiteDatetime,
  sqliteDatetimeToISO,
} from '../services/user-data.service.js'

/** 将行数据转为单行 CSV（RFC 4180 转义规则） */
function _rowToCSV(fields: unknown[]): string {
  return fields.map(f => {
    const s = f === null || f === undefined ? '' : String(f)
    if (s.includes(',') || s.includes('"') || s.includes('\n') || s.includes('\r')) {
      return '"' + s.replace(/"/g, '""') + '"'
    }
    return s
  }).join(',')
}

/** 构建 TEI standOff 格式 XML */
function _toTEIStandOff(
  annotations: Array<Record<string, unknown>>,
  chapterNumber: number,
): string {
  const header = `<?xml version="1.0" encoding="UTF-8"?>
<TEI xmlns="http://www.tei-c.org/ns/1.0">
  <teiHeader>
    <fileDesc>
      <titleStmt><title>Chapter ${chapterNumber} Annotations (Stand-off)</title></titleStmt>
      <publicationStmt><p>Exported from White Snake Museum</p></publicationStmt>
    </fileDesc>
  </teiHeader>
  <standOff>`
  const footer = `  </standOff>\n</TEI>`

  const entities = annotations.map(a => {
    const id = a.id || ''
    const entity = String(a.entity || '')
    const category = String(a.category || '').replace(/&/g, '&amp;').replace(/"/g, '&quot;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
    const note = String(a.explanation || '')
    const startChar = a.start_char ?? 0
    const endChar = a.end_char ?? 0
    return `    <span type="${category}" ana="#${id}" target="#paragraph_${a.paragraph_index}">
      <fs>
        <f name="entity"><string>${entity.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')}</string></f>
        <f name="start_char"><numeric value="${startChar}"/></f>
        <f name="end_char"><numeric value="${endChar}"/></f>
        <f name="note"><string>${note.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')}</string></f>
      </fs>
    </span>`
  }).join('\n')

  return header + '\n' + entities + '\n' + footer
}

/** 构建 TEI inline 格式 XML */
function _toTEIInline(
  annotations: Array<Record<string, unknown>>,
  chapterNumber: number,
): string {
  const header = `<?xml version="1.0" encoding="UTF-8"?>
<TEI xmlns="http://www.tei-c.org/ns/1.0">
  <teiHeader>
    <fileDesc>
      <titleStmt><title>Chapter ${chapterNumber} Annotations (Inline)</title></titleStmt>
      <publicationStmt><p>Exported from White Snake Museum</p></publicationStmt>
    </fileDesc>
  </teiHeader>
  <text>
    <body>`
  const footer = `    </body>\n  </text>\n</TEI>`

  // 按段落分组
  const byPara = new Map<number, Array<Record<string, unknown>>>()
  for (const a of annotations) {
    const pi = (a.paragraph_index as number) ?? 0
    if (!byPara.has(pi)) byPara.set(pi, [])
    byPara.get(pi)!.push(a)
  }

  const paragraphs = Array.from(byPara.entries())
    .sort(([a], [b]) => a - b)
    .map(([pi, anns]) => {
      const spans = anns.map(a => {
        const entity = String(a.entity || '').replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
        const category = String(a.category || '').replace(/&/g, '&amp;').replace(/"/g, '&quot;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
        const note = String(a.explanation || '').replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
        return `      <name type="${category}" ana="${note ? '#note_' + a.id : ''}">${entity}</name>`
      }).join('\n')
      return `    <p n="${pi}">\n${spans}\n    </p>`
    }).join('\n')

  return header + '\n' + paragraphs + '\n' + footer
}

/**
 * 将各种时间格式统一转为 SQLite datetime 格式 (YYYY-MM-DD HH:MM:SS)。
 * 兼容 ISO 8601 (含 T)、SQLite 本地时间格式、以及无效输入（返回 null 让调用方降级）。
 */
function _normalizeDatetime(val: unknown): string | null {
  if (typeof val !== 'string' || !val) return null
  // 已经是 "YYYY-MM-DD HH:MM:SS" 格式
  if (/^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}$/.test(val)) return val
  // ISO 8601 "YYYY-MM-DDTHH:MM:SS*" → 转为 SQLite 格式
  const m = val.match(/^(\d{4}-\d{2}-\d{2})T(\d{2}:\d{2}:\d{2})/)
  if (m) return `${m[1]} ${m[2]}`
  // 无法识别的格式，返回 null 让调用方降级到下一个候选值
  console.warn(`[_normalizeDatetime] unrecognized format: ${val}`)
  return null
}

function _tsToSQLite(val: unknown): string | null {
  if (typeof val === 'number' && val > 0) return toSQLiteDatetime(new Date(val))
  return null
}

function _isValidChapter(n: unknown): n is number {
  return typeof n === 'number' && Number.isFinite(n) && n >= 1
}

interface FrontendAnnotation {
  id: string
  label?: string
  category: string
  span: { startChar: number; endChar: number }
  color: string
  note: string
  createdAt: number
  updatedAt: number
  passageKey?: string
  paragraphIndex?: number
}

interface FrontendMarginalia {
  id: string
  annotationId?: string
  chapterNumber?: number
  paragraphIndex?: number
  anchorCharOffset?: number
  content: string
  source?: string
  createdAt?: number
  updatedAt?: number
}

async function annotationRoutes(app: FastifyInstance): Promise<void> {

  // 必须在 :chapterNumber 之前注册，防止 "stats" 被捕获为路径参数。
  // 如果重构此文件，请务必保持此路由在 "/api/annotations/:chapterNumber" 之前注册。
  app.get('/api/annotations/stats', async (req: FastifyRequest, reply: FastifyReply) => {
    const userId = req.userId
    if (!userId) {
      return reply.status(401).send({ detail: 'Not authenticated' })
    }

    const db = openUserDB(userId)

    // 按章节统计（区分 user / agent 来源）
    const chapters = db.prepare(
      "SELECT chapter_number, source, COUNT(*) as count FROM annotations GROUP BY chapter_number, source"
    ).all() as Array<{ chapter_number: number; source: string; count: number }>

    const totalRow = db.prepare(
      "SELECT source, COUNT(*) as total FROM annotations GROUP BY source"
    ).all() as Array<{ source: string; total: number }>

    // byChapter: Record<string, number> 保持向前兼容（前端依赖此格式做 > 0 判断）
    const byChapter: Record<string, number> = {}
    for (const row of chapters) {
      byChapter[String(row.chapter_number)] = (byChapter[String(row.chapter_number)] || 0) + row.count
    }

    const bySource = { user: 0, agent: 0 }
    let grandTotal = 0
    for (const row of totalRow) {
      if (row.source === 'user' || row.source === 'agent') {
        bySource[row.source] = row.total
      }
      grandTotal += row.total
    }

    return reply.status(200).send({
      totalAnnotations: grandTotal,
      bySource,
      byChapter,
    })
  })

  // D3-export: 服务端导出标注数据（直接从 SQLite 查询，保证数据完整性）
  // 支持格式: csv / json / html / plain / tei-standoff / tei-inline
  // source 过滤: user / agent / all
  app.get('/api/annotations/export/:chapterNumber', async (req: FastifyRequest, reply: FastifyReply) => {
    const userId = req.userId
    if (!userId) {
      return reply.status(401).send({ detail: 'Not authenticated' })
    }

    const { chapterNumber } = req.params as { chapterNumber: string }
    const chNum = parseInt(chapterNumber, 10)
    if (isNaN(chNum) || chNum < 1) {
      return reply.status(400).send({ detail: 'Invalid chapter number' })
    }

    const query = req.query as { format?: string; source?: string }
    const format = (query.format || 'csv').toLowerCase()
    const sourceFilter = (query.source || 'all').toLowerCase()

    const VALID_FORMATS = new Set(['csv', 'json', 'html', 'plain', 'tei-standoff', 'tei-inline'])
    if (!VALID_FORMATS.has(format)) {
      return reply.status(400).send({ detail: `Unsupported format: ${format}. Supported: csv, json, html, plain, tei-standoff, tei-inline` })
    }

    const VALID_SOURCES = new Set(['user', 'agent', 'all'])
    if (!VALID_SOURCES.has(sourceFilter)) {
      return reply.status(400).send({ detail: `Unsupported source: ${sourceFilter}. Supported: user, agent, all` })
    }

    const db = openUserDB(userId)

    // 使用参数化查询（即使 sourceFilter 已经白名单校验）
    let rows: Array<Record<string, unknown>>
    let margRows: Array<Record<string, unknown>>
    if (sourceFilter === 'all') {
      rows = db.prepare(
        'SELECT * FROM annotations WHERE chapter_number = ? ORDER BY paragraph_index, start_char'
      ).all(chNum) as Array<Record<string, unknown>>
      margRows = db.prepare(
        'SELECT * FROM marginalia WHERE chapter_number = ? ORDER BY paragraph_index'
      ).all(chNum) as Array<Record<string, unknown>>
    } else {
      rows = db.prepare(
        'SELECT * FROM annotations WHERE chapter_number = ? AND source = ? ORDER BY paragraph_index, start_char'
      ).all(chNum, sourceFilter) as Array<Record<string, unknown>>
      margRows = db.prepare(
        'SELECT * FROM marginalia WHERE chapter_number = ? AND source = ? ORDER BY paragraph_index'
      ).all(chNum, sourceFilter) as Array<Record<string, unknown>>
    }

    switch (format) {
      case 'csv': {
        const csvHeader = _rowToCSV(['id', 'entity', 'category', 'chapter_number', 'paragraph_index', 'start_char', 'end_char', 'quote', 'explanation', 'confidence', 'color', 'source', 'created_at', 'updated_at'])
        const csvBody = rows.map(r => _rowToCSV([
          r.id, r.entity, r.category, r.chapter_number, r.paragraph_index,
          r.start_char, r.end_char, r.quote, r.explanation, r.confidence,
          r.color, r.source, r.created_at, r.updated_at,
        ])).join('\n')
        const csvData = csvHeader + '\n' + csvBody
        return reply
          .header('Content-Type', 'text/csv; charset=utf-8')
          .header('Content-Disposition', `attachment; filename="chapter-${chNum}-annotations.csv"`)
          .send(csvData)
      }
      case 'json': {
        return reply
          .header('Content-Type', 'application/json; charset=utf-8')
          .header('Content-Disposition', `attachment; filename="chapter-${chNum}-annotations.json"`)
          .send({ chapterNumber: chNum, annotations: rows, marginalia: margRows, exportedAt: new Date().toISOString() })
      }
      case 'html': {
        const rows_html = rows.map(r =>
          `<tr><td>${r.entity}</td><td>${r.category}</td><td>第${r.paragraph_index}段</td><td>${r.start_char}-${r.end_char}</td><td>${r.explanation || ''}</td><td>${r.confidence}</td><td>${r.source}</td></tr>`
        ).join('\n')
        const html = `<!DOCTYPE html>
<html lang="zh-CN">
<head><meta charset="UTF-8"><title>Chapter ${chNum} Annotations</title>
<style>body{font-family:sans-serif;margin:2em}table{border-collapse:collapse;width:100%}th,td{border:1px solid #ccc;padding:8px;text-align:left}th{background:#f5f5f5}</style></head>
<body>
<h1>Chapter ${chNum} Annotations</h1>
<table>
<thead><tr><th>Entity</th><th>Category</th><th>Paragraph</th><th>Span</th><th>Note</th><th>Confidence</th><th>Source</th></tr></thead>
<tbody>${rows_html}</tbody>
</table>
</body></html>`
        return reply
          .header('Content-Type', 'text/html; charset=utf-8')
          .header('Content-Disposition', `attachment; filename="chapter-${chNum}-annotations.html"`)
          .send(html)
      }
      case 'plain': {
        const lines = rows.map(r =>
          `[${r.category}] ${r.entity} (第${r.paragraph_index}段, ${r.start_char}-${r.end_char})${r.explanation ? ': ' + r.explanation : ''}`
        )
        return reply
          .header('Content-Type', 'text/plain; charset=utf-8')
          .header('Content-Disposition', `attachment; filename="chapter-${chNum}-annotations.txt"`)
          .send(lines.join('\n'))
      }
      case 'tei-standoff': {
        const xml = _toTEIStandOff(rows, chNum)
        return reply
          .header('Content-Type', 'application/xml; charset=utf-8')
          .header('Content-Disposition', `attachment; filename="chapter-${chNum}-annotations-tei-standoff.xml"`)
          .send(xml)
      }
      case 'tei-inline': {
        const xml = _toTEIInline(rows, chNum)
        return reply
          .header('Content-Type', 'application/xml; charset=utf-8')
          .header('Content-Disposition', `attachment; filename="chapter-${chNum}-annotations-tei-inline.xml"`)
          .send(xml)
      }
      default:
        return reply.status(400).send({ detail: `Unsupported format: ${format}` })
    }
  })

  // D2-4: Visibility 持久化 — GET
  app.get('/api/annotations/visibility', async (req: FastifyRequest, reply: FastifyReply) => {
    const userId = req.userId
    if (!userId) {
      return reply.status(401).send({ detail: 'Not authenticated' })
    }

    const db = openUserDB(userId)
    const raw = getPreference(db, 'visibility')
    let visibility = null
    if (raw) {
      try { visibility = JSON.parse(raw) } catch { console.warn('[annotations] visibility JSON 解析失败，降级为 null') }
    }
    return reply.status(200).send({ visibility })
  })

  // D2-4: Visibility 持久化 — PUT
  app.put('/api/annotations/visibility', async (req: FastifyRequest, reply: FastifyReply) => {
    const userId = req.userId
    if (!userId) {
      return reply.status(401).send({ detail: 'Not authenticated' })
    }

    const body = req.body as { visibility?: unknown }
    if (body.visibility === undefined) {
      return reply.status(400).send({ detail: 'visibility is required' })
    }

    const db = openUserDB(userId)
    savePreference(db, 'visibility', JSON.stringify(body.visibility))
    return reply.status(200).send({ status: 'ok' })
  })

  // D2-5b: Agent 保存标注（追加/合并，仅操作 agent 来源数据，保护用户手工标注）
  app.post('/api/annotations/save', async (req: FastifyRequest, reply: FastifyReply) => {
    const userId = req.userId
    if (!userId) {
      return reply.status(401).send({ detail: 'Not authenticated' })
    }

    const body = req.body as {
      chapterNumber?: number
      annotations?: Array<Record<string, unknown>>
      marginalia?: Array<Record<string, unknown>>
    }

    const chapterNumber = body.chapterNumber
    if (!_isValidChapter(chapterNumber)) {
      return reply.status(400).send({ detail: 'chapterNumber is required and must be >= 1' })
    }

    const annotations = body.annotations || []
    const marginalia = body.marginalia || []

    if (annotations.length === 0 && marginalia.length === 0) {
      return reply.status(200).send({ saved: 0 })
    }

    const db = openUserDB(userId)
    try {
      db.exec('BEGIN')

      // 清理旧 agent 来源数据 — 按"本批涉及的段落"范围删除（而非整章），
      // 支持多批次增量保存：Python 流式标注逐段 save、前端 saveAgentToServer
      // 全量上传，两者交错执行时互不清除对方已保存的其他段落。
      // 同段重跑时旧实体（ID 基于序号生成，可能变化）随段级 DELETE 一并清理，无孤儿。
      // user 来源数据不受影响，由下方的 userAnnotationIds/userMarginaliaIds 保护。
      if (annotations.length > 0) {
        const paraSet = new Set<number>()
        for (const a of annotations) {
          const r = a as Record<string, unknown>
          const pi = r.paragraph_index ?? r.paragraphIndex
          paraSet.add(typeof pi === 'number' && Number.isFinite(pi) ? pi : 0)
        }
        const paras = Array.from(paraSet)
        const ph = paras.map(() => '?').join(',')
        db.prepare(
          `DELETE FROM annotations WHERE chapter_number = ? AND source = 'agent' AND paragraph_index IN (${ph})`
        ).run(chapterNumber, ...paras)
      }
      if (marginalia.length > 0) {
        const paraSet = new Set<number>()
        for (const m of marginalia) {
          const r = m as Record<string, unknown>
          const pi = r.paragraph_index ?? r.paragraphIndex
          paraSet.add(typeof pi === 'number' && Number.isFinite(pi) ? pi : 0)
        }
        const paras = Array.from(paraSet)
        const ph = paras.map(() => '?').join(',')
        db.prepare(
          `DELETE FROM marginalia WHERE chapter_number = ? AND source = 'agent' AND paragraph_index IN (${ph})`
        ).run(chapterNumber, ...paras)
      }

      // 预先查出已存在的 user 来源标注/旁批 ID，避免 UPSERT 时因 WHERE 冲突导致整批回滚。
      // 保护用户手工标注不被 agent 覆盖：如果某 ID 已被用户手工创建，agent 不可覆写。
      const userAnnotationIds = new Set<string>()
      const userMarginaliaIds = new Set<string>()

      if (annotations.length > 0) {
        const annIds = annotations.map(a => (a as Record<string, unknown>).id as string).filter(Boolean)
        if (annIds.length > 0) {
          const placeholders = annIds.map(() => '?').join(',')
          const rows = db.prepare(
            `SELECT id FROM annotations WHERE source = 'user' AND id IN (${placeholders})`
          ).all(...annIds) as Array<{ id: string }>
          for (const row of rows) userAnnotationIds.add(row.id)
        }
      }

      if (marginalia.length > 0) {
        const margIds = marginalia.map(m => (m as Record<string, unknown>).id as string).filter(Boolean)
        if (margIds.length > 0) {
          const placeholders = margIds.map(() => '?').join(',')
          const rows = db.prepare(
            `SELECT id FROM marginalia WHERE source = 'user' AND id IN (${placeholders})`
          ).all(...margIds) as Array<{ id: string }>
          for (const row of rows) userMarginaliaIds.add(row.id)
        }
      }

      let savedCount = 0

      if (annotations.length > 0) {
        const upsertAnn = db.prepare(
          `INSERT INTO annotations (id, chapter_number, paragraph_index, category, entity, start_char, end_char, quote, explanation, confidence, color, source, created_at, updated_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
           ON CONFLICT(id) DO UPDATE SET
             chapter_number=excluded.chapter_number,
             paragraph_index=excluded.paragraph_index,
             category=excluded.category,
             entity=excluded.entity,
             start_char=excluded.start_char,
             end_char=excluded.end_char,
             quote=excluded.quote,
             explanation=excluded.explanation,
             confidence=excluded.confidence,
             color=excluded.color,
             source=excluded.source,
             updated_at=excluded.updated_at`
        )

        for (const ann of annotations) {
          const r = ann as Record<string, unknown>
          const annId = r.id as string
          // 跳过与用户手工标注冲突的记录，保护用户数据
          if (userAnnotationIds.has(annId)) continue

          const paraIdx = (r.paragraph_index ?? r.paragraphIndex ?? 0) as number
          const source = (r.source as string) || 'agent'
          const now = toSQLiteDatetime(new Date())
          const entity = (r.entity as string) || (r.label as string) || ''
          const startChar = (r.start_char ?? r.startChar ?? 0) as number
          const endChar = (r.end_char ?? r.endChar ?? (startChar + entity.length)) as number
          const createdAt = _normalizeDatetime(r.created_at) || _tsToSQLite(r.createdAt) || now
          const updatedAt = _normalizeDatetime(r.updated_at) || _tsToSQLite(r.updatedAt) || now

          upsertAnn.run(
            annId, chapterNumber, paraIdx,
            (r.category as string) || 'custom', entity,
            startChar, endChar,
            (r.quote as string) || entity,
            (r.explanation as string) || (r.note as string) || '',
            normalizeConfidence(r.confidence),
            (r.color as string) || '',
            source,
            createdAt, updatedAt
          )
          savedCount++
        }
      }

      if (marginalia.length > 0) {
        const upsertMarg = db.prepare(
          `INSERT INTO marginalia (id, annotation_id, chapter_number, paragraph_index, anchor_char_offset, content, source, created_at, updated_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
           ON CONFLICT(id) DO UPDATE SET
             annotation_id=excluded.annotation_id,
             chapter_number=excluded.chapter_number,
             paragraph_index=excluded.paragraph_index,
             anchor_char_offset=excluded.anchor_char_offset,
             content=excluded.content,
             source=excluded.source,
             updated_at=excluded.updated_at`
        )

        for (const marg of marginalia) {
          const m = marg as Record<string, unknown>
          const margId = m.id as string
          // 跳过与用户手工旁批冲突的记录，保护用户数据
          if (userMarginaliaIds.has(margId)) continue

          // 跳过空内容的旁批，避免语义无效数据
          const margContent = (m.content as string) || ''
          if (!margContent.trim()) continue

          const now = toSQLiteDatetime(new Date())
          const paraIdx = (m.paragraph_index ?? m.paragraphIndex ?? 0) as number
          const createdAt = _normalizeDatetime(m.created_at) || _tsToSQLite(m.createdAt) || now
          const updatedAt = _normalizeDatetime(m.updated_at) || _tsToSQLite(m.updatedAt) || now

          upsertMarg.run(
            margId, (m.annotation_id as string) || (m.annotationId as string) || null,
            chapterNumber, paraIdx,
            (m.anchor_char_offset as number) ?? (m.anchorCharOffset as number) ?? null,
            margContent,
            (m.source as string) || 'agent',
            createdAt, updatedAt
          )
          savedCount++
        }
      }

      db.exec('COMMIT')
      return reply.status(200).send({ saved: savedCount })
    } catch (err) {
      try { db.exec('ROLLBACK') } catch { console.error('[annotations] ROLLBACK 失败，事务可能处于未定义状态') }
      throw err
    }
  })

  // D2-5c: 删除指定章节的标注（按类别或全量，默认仅删除用户来源数据）
  // cleanAgent=true 时同步清理 agent 来源标注（需显式传参，防止误删）
  app.delete('/api/annotations', async (req: FastifyRequest, reply: FastifyReply) => {
    const userId = req.userId
    if (!userId) {
      return reply.status(401).send({ detail: 'Not authenticated' })
    }

    const body = req.body as {
      chapterNumber?: number
      categories?: string[]
      deleteAll?: boolean
      cleanAgent?: boolean
    }

    const chapterNumber = body.chapterNumber
    if (!_isValidChapter(chapterNumber)) {
      return reply.status(400).send({ detail: 'chapterNumber is required and must be >= 1' })
    }

    // 构建 source 过滤条件: 默认只删除 user 来源，cleanAgent=true 时删除所有
    const sourceFilter = body.cleanAgent
      ? "source IN ('user', 'agent')"
      : "source = 'user'"

    const db = openUserDB(userId)
    try {
      db.exec('BEGIN')

      let deletedAnns = 0
      let deletedMargs = 0

      if (body.deleteAll) {
        const r1 = db.prepare(
          `DELETE FROM annotations WHERE chapter_number = ? AND ${sourceFilter}`
        ).run(chapterNumber)
        const r2 = db.prepare(
          `DELETE FROM marginalia WHERE chapter_number = ? AND ${sourceFilter}`
        ).run(chapterNumber)
        deletedAnns = Number(r1.changes)
        deletedMargs = Number(r2.changes)
        // 防御性清理：兼容未启用外键的旧数据库，或 ON DELETE SET NULL 后仍残留引用
        // 已不存在 annotation_id 的孤儿旁批
        if (deletedAnns > 0) {
          const r3 = db.prepare(
            `DELETE FROM marginalia WHERE chapter_number = ?
             AND annotation_id IS NOT NULL
             AND annotation_id NOT IN (SELECT id FROM annotations WHERE chapter_number = ?)`
          ).run(chapterNumber, chapterNumber)
          deletedMargs += Number(r3.changes)
        }
      } else if (body.categories && body.categories.length > 0) {
        const validCategories = ['person', 'location', 'event', 'term', 'motif', 'custom']
        const filteredCategories = body.categories.filter(c => validCategories.includes(c))
        if (filteredCategories.length === 0) {
          db.exec('COMMIT')
          return reply.status(200).send({ deleted: 0, deletedAnnotations: 0, deletedMarginalia: 0 })
        }
        const placeholders = filteredCategories.map(() => '?').join(',')
        const r1 = db.prepare(
          `DELETE FROM annotations WHERE chapter_number = ? AND ${sourceFilter} AND category IN (${placeholders})`
        ).run(chapterNumber, ...filteredCategories)
        deletedAnns = Number(r1.changes)
        // 外键约束 (ON DELETE SET NULL) 会在启用 foreign_keys 后自动处理
        // 但为了兼容未启用外键的旧数据库，手动同步清理孤儿旁批
        if (deletedAnns > 0) {
          const r2 = db.prepare(
            `DELETE FROM marginalia WHERE chapter_number = ? AND ${sourceFilter}
             AND annotation_id IS NOT NULL
             AND annotation_id NOT IN (SELECT id FROM annotations WHERE chapter_number = ?)`
          ).run(chapterNumber, chapterNumber)
          deletedMargs = Number(r2.changes)
        }
      }

      db.exec('COMMIT')
      return reply.status(200).send({
        deleted: deletedAnns + deletedMargs,
        deletedAnnotations: deletedAnns,
        deletedMarginalia: deletedMargs,
      })
    } catch (err) {
      try { db.exec('ROLLBACK') } catch { console.error('[annotations] ROLLBACK 失败，事务可能处于未定义状态') }
      throw err
    }
  })

  // D2-1: 获取指定章节的标注数据
  app.get('/api/annotations/:chapterNumber', async (req: FastifyRequest, reply: FastifyReply) => {
    const userId = req.userId
    if (!userId) {
      return reply.status(401).send({ detail: 'Not authenticated' })
    }

    const { chapterNumber } = req.params as { chapterNumber: string }
    const chNum = parseInt(chapterNumber, 10)
    if (isNaN(chNum) || chNum < 1) {
      return reply.status(400).send({ detail: 'Invalid chapter number' })
    }

    const db = openUserDB(userId)
    const annotationRows = getAnnotationRows(db, chNum)
    const marginaliaRows = getMarginaliaRows(db, chNum)
    const visibilityJson = getPreference(db, 'visibility')

    const annotations: Record<string, Array<Record<string, unknown>>> = {}
    for (const row of annotationRows) {
      const passageKey = `${row.chapter_number}:${row.paragraph_index}`
      if (!annotations[passageKey]) {
        annotations[passageKey] = []
      }
      annotations[passageKey].push({
        id: row.id,
        label: row.entity,
        category: row.category,
        span: { startChar: row.start_char as number, endChar: row.end_char as number },
        color: row.color,
        note: row.explanation,
        confidence: row.confidence,
        source: row.source,
        createdAt: Date.parse(sqliteDatetimeToISO(row.created_at)),
        updatedAt: Date.parse(sqliteDatetimeToISO(row.updated_at)),
      })
    }

    const marginalia: Record<string, Array<Record<string, unknown>>> = {}
    for (const row of marginaliaRows) {
      const passageKey = `${row.chapter_number}:${row.paragraph_index}`
      if (!marginalia[passageKey]) {
        marginalia[passageKey] = []
      }
      marginalia[passageKey].push({
        id: row.id,
        annotationId: row.annotation_id,
        chapterNumber: row.chapter_number,
        paragraphIndex: row.paragraph_index,
        anchorCharOffset: row.anchor_char_offset,
        content: row.content,
        source: row.source,
        createdAt: Date.parse(sqliteDatetimeToISO(row.created_at)),
        updatedAt: Date.parse(sqliteDatetimeToISO(row.updated_at)),
      })
    }

    let visibility = null
    if (visibilityJson) {
      try {
        visibility = JSON.parse(visibilityJson)
      } catch {
        visibility = null
      }
    }

    return reply.status(200).send({ annotations, marginalia, visibility })
  })

  // D2-2: 保存指定章节的标注数据（仅覆盖用户来源数据，保护 agent 标注不被误删）
  app.post('/api/annotations/:chapterNumber', async (req: FastifyRequest, reply: FastifyReply) => {
    const userId = req.userId
    if (!userId) {
      return reply.status(401).send({ detail: 'Not authenticated' })
    }

    const { chapterNumber } = req.params as { chapterNumber: string }
    const chNum = parseInt(chapterNumber, 10)
    if (isNaN(chNum) || chNum < 1) {
      return reply.status(400).send({ detail: 'Invalid chapter number' })
    }

    const body = req.body as {
      annotations?: FrontendAnnotation[]
      marginalia?: FrontendMarginalia[]
      visibility?: Record<string, unknown>
    }
    const annotations = body.annotations || []
    const marginalia = body.marginalia || []

    const db = openUserDB(userId)
    try {
      db.exec('BEGIN')

      // 预先查出 agent 来源的标注/旁批 ID，保护 agent 数据不被用户保存覆盖。
      // 在 DELETE 之前查询，确保拿到完整的 agent ID 集合。
      const agentAnnotationIds = new Set<string>()
      const agentMarginaliaIds = new Set<string>()

      if (annotations.length > 0) {
        const annIds = annotations.map(a => a.id).filter(Boolean)
        if (annIds.length > 0) {
          const placeholders = annIds.map(() => '?').join(',')
          const rows = db.prepare(
            `SELECT id FROM annotations WHERE source = 'agent' AND id IN (${placeholders})`
          ).all(...annIds) as Array<{ id: string }>
          for (const row of rows) agentAnnotationIds.add(row.id)
        }
      }

      if (marginalia.length > 0) {
        const margIds = marginalia.map(m => m.id).filter(Boolean)
        if (margIds.length > 0) {
          const placeholders = margIds.map(() => '?').join(',')
          const rows = db.prepare(
            `SELECT id FROM marginalia WHERE source = 'agent' AND id IN (${placeholders})`
          ).all(...margIds) as Array<{ id: string }>
          for (const row of rows) agentMarginaliaIds.add(row.id)
        }
      }

      // 只删除当前用户来源的标注和旁批，保护 agent 来源数据。
      // 此 DELETE + INSERT 在 BEGIN/COMMIT 事务中执行，原子性有保证。
      // 如果中途异常，ROLLBACK 会恢复已删除的数据。
      db.prepare("DELETE FROM annotations WHERE chapter_number = ? AND source = 'user'").run(chNum)

      const upsertAnn = db.prepare(
        `INSERT INTO annotations (id, chapter_number, paragraph_index, category, entity, start_char, end_char, quote, explanation, confidence, color, source, created_at, updated_at)
         VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
         ON CONFLICT(id) DO UPDATE SET
           chapter_number=excluded.chapter_number,
           paragraph_index=excluded.paragraph_index,
           category=excluded.category,
           entity=excluded.entity,
           start_char=excluded.start_char,
           end_char=excluded.end_char,
           quote=excluded.quote,
           explanation=excluded.explanation,
           confidence=excluded.confidence,
           color=excluded.color,
           source=excluded.source,
           updated_at=excluded.updated_at`
      )

      let savedCount = 0

      for (const ann of annotations) {
        // 跳过与 agent 标注 ID 冲突的记录，保护 agent 数据
        if (agentAnnotationIds.has(ann.id)) continue

        // 优先使用显式的 paragraphIndex 字段，passageKey 仅作为 fallback
        const annRecord = ann as unknown as Record<string, unknown>
        const paraIdx = (annRecord.paragraphIndex as number | undefined)
          ?? (() => {
            const parts = (ann.passageKey || '').split(':')
            const fallbackIdx = parts.length === 2 ? parseInt(parts[1], 10) : 0
            // 仅在 passageKey 无法解析时才报警，避免段落 0 的合法标注误触发
            if (parts.length !== 2 || isNaN(parseInt(parts[1], 10))) {
              console.warn(`[annotations] paragraph_index 缺失且 passageKey 无法解析: annId=${ann.id}, passageKey="${ann.passageKey}" — 前端可能未发送 paragraphIndex`)
            }
            return fallbackIdx
          })()
        const createdAt = toSQLiteDatetime(new Date(ann.createdAt))
        const updatedAt = toSQLiteDatetime(new Date(ann.updatedAt || ann.createdAt))
        const confidenceText = normalizeConfidence(
          (ann as unknown as Record<string, unknown>).confidence
        )
        const source = (ann as unknown as Record<string, unknown>).source as string || 'user'

        upsertAnn.run(
          ann.id, chNum, paraIdx,
          ann.category, ann.label || '', ann.span.startChar, ann.span.endChar,
          ann.label || '', ann.note || '', confidenceText, ann.color,
          source,
          createdAt, updatedAt
        )
        savedCount++
      }

      // 只删除当前用户来源的旁批，保护 agent 来源数据
      db.prepare("DELETE FROM marginalia WHERE chapter_number = ? AND source = 'user'").run(chNum)

      const upsertMarg = db.prepare(
        `INSERT INTO marginalia (id, annotation_id, chapter_number, paragraph_index, anchor_char_offset, content, source, created_at, updated_at)
         VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
         ON CONFLICT(id) DO UPDATE SET
           annotation_id=excluded.annotation_id,
           chapter_number=excluded.chapter_number,
           paragraph_index=excluded.paragraph_index,
           anchor_char_offset=excluded.anchor_char_offset,
           content=excluded.content,
           source=excluded.source,
           updated_at=excluded.updated_at`
      )

      for (const marg of marginalia) {
        // 跳过与 agent 旁批 ID 冲突的记录，保护 agent 数据
        if (agentMarginaliaIds.has(marg.id)) continue

        // 跳过空内容的旁批，避免语义无效数据
        if (!marg.content || !marg.content.trim()) continue

        const paraIdx = marg.paragraphIndex ?? 0
        const now = toSQLiteDatetime(new Date())
        const createdAt = marg.createdAt ? toSQLiteDatetime(new Date(marg.createdAt)) : now
        const updatedAt = marg.updatedAt ? toSQLiteDatetime(new Date(marg.updatedAt)) : now

        upsertMarg.run(
          marg.id, marg.annotationId || null, chNum, paraIdx,
          marg.anchorCharOffset || null, marg.content, marg.source || 'user',
          createdAt, updatedAt
        )
        savedCount++
      }

      if (body.visibility) {
        savePreference(db, 'visibility', JSON.stringify(body.visibility))
      }

      db.exec('COMMIT')
      return reply.status(200).send({ saved: savedCount })
    } catch (err) {
      try { db.exec('ROLLBACK') } catch { console.error('[annotations] ROLLBACK 失败，事务可能处于未定义状态') }
      throw err
    }
  })

  // D2-3: 批量导入标注数据（从 localStorage 迁移，仅替换用户来源数据）
  app.post('/api/annotations/import', async (req: FastifyRequest, reply: FastifyReply) => {
    const userId = req.userId
    if (!userId) {
      return reply.status(401).send({ detail: 'Not authenticated' })
    }

    const body = req.body as {
      annotationSets?: Record<string, { annotations?: Array<Record<string, unknown>>; marginalia?: Array<Record<string, unknown>>; savedAt?: number }>
    }
    const sets = body.annotationSets || {}

    const db = openUserDB(userId)
    const imported: Record<string, number> = {}

    try {
      db.exec('BEGIN')

      for (const [chKey, set] of Object.entries(sets)) {
        const chNum = parseInt(chKey, 10)
        if (isNaN(chNum) || chNum < 1) continue

        const anns = set.annotations || []
        const margs = set.marginalia || []

        // 预先查出 agent 来源的标注/旁批 ID，保护 agent 数据不被导入覆盖
        const agentAnnIds = new Set<string>()
        const agentMargIds = new Set<string>()

        if (anns.length > 0) {
          const ids = anns.map(a => (a as Record<string, unknown>).id as string).filter(Boolean)
          if (ids.length > 0) {
            const placeholders = ids.map(() => '?').join(',')
            const rows = db.prepare(
              `SELECT id FROM annotations WHERE source = 'agent' AND id IN (${placeholders})`
            ).all(...ids) as Array<{ id: string }>
            for (const row of rows) agentAnnIds.add(row.id)
          }
        }

        if (margs.length > 0) {
          const ids = margs.map(m => (m as Record<string, unknown>).id as string).filter(Boolean)
          if (ids.length > 0) {
            const placeholders = ids.map(() => '?').join(',')
            const rows = db.prepare(
              `SELECT id FROM marginalia WHERE source = 'agent' AND id IN (${placeholders})`
            ).all(...ids) as Array<{ id: string }>
            for (const row of rows) agentMargIds.add(row.id)
          }
        }

        // 仅删除用户来源数据再导入，保护 agent 标注不被误删
        db.prepare("DELETE FROM annotations WHERE chapter_number = ? AND source = 'user'").run(chNum)
        db.prepare("DELETE FROM marginalia WHERE chapter_number = ? AND source = 'user'").run(chNum)

        let chSaved = 0

        const insertAnn = db.prepare(
          `INSERT OR IGNORE INTO annotations (id, chapter_number, paragraph_index, category, entity, start_char, end_char, quote, explanation, confidence, color, source, created_at, updated_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`
        )

        for (const ann of anns) {
          const annRecord = ann as Record<string, unknown>
          if (agentAnnIds.has(annRecord.id as string)) continue

          const paraIdx = (annRecord.paragraphIndex as number) ?? 0
          const label = (annRecord.label as string) || ''
          const span = (annRecord.span as { startChar: number; endChar: number }) || { startChar: 0, endChar: 0 }
          const note = (annRecord.explanation as string) || (annRecord.note as string) || ''
          const source = (annRecord.source as string) || 'user'
          const confidenceText = normalizeConfidence(annRecord.confidence)
          const createdAt = annRecord.createdAt
            ? toSQLiteDatetime(new Date(annRecord.createdAt as number))
            : toSQLiteDatetime(new Date())
          const updatedAt = annRecord.updatedAt
            ? toSQLiteDatetime(new Date(annRecord.updatedAt as number))
            : createdAt

          insertAnn.run(
            annRecord.id as string, chNum, paraIdx,
            annRecord.category as string, label, span.startChar, span.endChar,
            label, note, confidenceText,
            (annRecord.color as string) || '', source, createdAt, updatedAt
          )
          chSaved++
        }

        const insertMarg = db.prepare(
          `INSERT OR IGNORE INTO marginalia (id, annotation_id, chapter_number, paragraph_index, anchor_char_offset, content, source, created_at, updated_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)`
        )

        for (const marg of margs) {
          const m = marg as Record<string, unknown>
          if (agentMargIds.has(m.id as string)) continue

          // 跳过空内容的旁批，避免语义无效数据
          const margContent = (m.content as string) || ''
          if (!margContent.trim()) continue

          const paraIdx = (m.paragraphIndex as number) ?? 0
          const now = toSQLiteDatetime(new Date())
          const mCreatedAt = m.createdAt
            ? toSQLiteDatetime(new Date(m.createdAt as number))
            : now
          const mUpdatedAt = m.updatedAt
            ? toSQLiteDatetime(new Date(m.updatedAt as number))
            : mCreatedAt

          insertMarg.run(
            m.id as string, (m.annotationId as string) || null, chNum, paraIdx,
            (m.anchorCharOffset as number) || null, margContent,
            (m.source as string) || 'user', mCreatedAt, mUpdatedAt
          )
          chSaved++
        }

        imported[chKey] = chSaved
      }

      db.exec('COMMIT')
      return reply.status(200).send({ imported })
    } catch (err) {
      try { db.exec('ROLLBACK') } catch { console.error('[annotations] ROLLBACK 失败，事务可能处于未定义状态') }
      throw err
    }
  })
}

export { annotationRoutes }
