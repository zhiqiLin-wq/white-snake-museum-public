/**
 * 跳转原文高亮计算逻辑测试（test-first）
 *
 * 验证场景：
 * 1. chapters 已加载 → 实体扫描兜底命中
 * 2. chapters 未加载（targetPara=undefined）→ 返回空 + 标记需要延迟重算
 * 3. 摘录含省略号 → 剥离后命中
 * 4. 多实体共现（『白素贞 + 许仙』）→ 两个实体都扫描
 * 5. paragraphMatchPositions 已有 → 直接使用后端精确位置
 *
 * 运行: npx vitest run tests/highlight-computation.test.ts
 * 或: npx tsx tests/highlight-computation.test.ts
 */
import { describe, it, expect } from 'vitest'

// --- 被测类型 ---
interface Paragraph {
  index: number
  text: string
}

interface SearchResultLike {
  chapterNumber: number | string
  paragraphIndex: number
  excerpt?: string
  paragraphMatchPositions?: [number, number][]
}

// --- 被测函数（从 SearchResultsView 提取的纯逻辑，将在修复后导入） ---
// 当前先内联副本测试，修复后改为 import { computeJumpHighlights } from '...'
function computeJumpHighlights(
  query: string,
  result: SearchResultLike,
  targetPara: Paragraph | undefined,
): { start: number; end: number }[] {
  let positions: [number, number][] =
    result.paragraphMatchPositions && result.paragraphMatchPositions.length > 0
      ? result.paragraphMatchPositions
      : []

  // 兜底 1：query 直接在段落原文中找
  if (positions.length === 0) {
    const q = (query || '').trim()
    if (targetPara && q.length >= 2) {
      const lowerPara = targetPara.text.toLowerCase()
      const lowerQ = q.toLowerCase()
      let from = 0
      while (from < lowerPara.length) {
        const idx = lowerPara.indexOf(lowerQ, from)
        if (idx < 0) break
        positions.push([idx, idx + q.length])
        from = idx + 1
      }
    }
  }

  // 兜底 2：excerpt 剥省略号后在段落原文中找
  if (positions.length === 0) {
    const excerptText = (result.excerpt || '').trim().replace(/^[…⋯]+|[…⋯]+$/g, '').trim()
    if (targetPara && excerptText.length >= 4) {
      const idx = targetPara.text.indexOf(excerptText)
      if (idx >= 0) {
        positions.push([idx, idx + excerptText.length])
      } else {
        const head = excerptText.slice(0, 10)
        if (head.length >= 4) {
          const hi = targetPara.text.indexOf(head)
          if (hi >= 0) positions.push([hi, hi + Math.min(excerptText.length, targetPara.text.length - hi)])
        }
      }
    }
  }

  // 兜底 3：从 『实体名』 提取实体，在段落原文中扫描
  if (positions.length === 0 && targetPara) {
    const m = query.match(/『([^』]+)』/)
    if (m?.[1]) {
      const entities = m[1].split(' + ').map(s => s.trim()).filter(s => s.length >= 2)
      for (const ent of entities) {
        let from = 0
        while (from < targetPara.text.length) {
          const idx = targetPara.text.indexOf(ent, from)
          if (idx < 0) break
          positions.push([idx, idx + ent.length])
          from = idx + 1
        }
      }
    }
  }

  return positions.map(([s, e]) => ({ start: s, end: e }))
}

// --- 测试用例 ---

describe('computeJumpHighlights', () => {
  const samplePara: Paragraph = {
    index: 5,
    text: '许仙在断桥上遇见了白素贞，两人一见钟情。许仙心中暗喜，上前搭话。',
  }

  it('场景1: chapters 已加载 → 实体扫描兜底命中', () => {
    const result: SearchResultLike = {
      chapterNumber: 7,
      paragraphIndex: 5,
      excerpt: '',  // 空摘录：强制走兜底3（实体扫描）
    }
    const positions = computeJumpHighlights('『许仙』出场轨迹', result, samplePara)
    expect(positions.length).toBeGreaterThanOrEqual(2) // "许仙"出现2次
    // 验证第一个位置确实指向"许仙"
    const first = positions[0]
    expect(samplePara.text.slice(first.start, first.end)).toBe('许仙')
  })

  it('场景2: chapters 未加载（targetPara=undefined）→ 返回空', () => {
    const result: SearchResultLike = {
      chapterNumber: 7,
      paragraphIndex: 5,
      excerpt: '…许仙心中暗喜…',
    }
    const positions = computeJumpHighlights('『许仙』出场轨迹', result, undefined)
    // 关键断言：chapters 没加载时，三层兜底都依赖 targetPara，必然返回空
    // 这就是 bug 的根因 — 后续修复需让 TextReaderView 在 chapters 加载后重算
    expect(positions).toHaveLength(0)
  })

  it('场景3: 摘录含省略号 → 剥离后命中', () => {
    const result: SearchResultLike = {
      chapterNumber: 7,
      paragraphIndex: 5,
      excerpt: '…断桥上遇见了白素贞…',
    }
    const positions = computeJumpHighlights('查询词', result, samplePara)
    expect(positions.length).toBeGreaterThanOrEqual(1)
    const first = positions[0]
    expect(samplePara.text.slice(first.start, first.end)).toContain('断桥上遇见了白素贞')
  })

  it('场景4: 多实体共现 → 两个实体都扫描', () => {
    const result: SearchResultLike = {
      chapterNumber: 7,
      paragraphIndex: 5,
      excerpt: '',
    }
    const positions = computeJumpHighlights('『白素贞 + 许仙』共现段落', result, samplePara)
    // 白素贞 出现1次 + 许仙 出现2次 = 3处
    expect(positions.length).toBeGreaterThanOrEqual(3)
  })

  it('场景5: paragraphMatchPositions 已有 → 直接使用后端精确位置', () => {
    const result: SearchResultLike = {
      chapterNumber: 7,
      paragraphIndex: 5,
      paragraphMatchPositions: [[10, 12], [20, 22]],
    }
    const positions = computeJumpHighlights('anything', result, samplePara)
    expect(positions).toEqual([{ start: 10, end: 12 }, { start: 20, end: 22 }])
  })

  it('场景6: 单字实体名被过滤（length >= 2）', () => {
    const para: Paragraph = { index: 0, text: '蛇蛇蛇' }
    const result: SearchResultLike = { chapterNumber: 1, paragraphIndex: 0 }
    // 『蛇』出场轨迹 — 实体名"蛇"长度1，被 filter(s => s.length >= 2) 过滤
    const positions = computeJumpHighlights('『蛇』出场轨迹', result, para)
    expect(positions).toHaveLength(0)
  })
})
