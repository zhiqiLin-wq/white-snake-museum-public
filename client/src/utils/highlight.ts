/**
 * 跳转原文高亮位置计算（纯函数，供 SearchResultsView 和 TextReaderView 复用）
 *
 * 三层兜底策略（按优先级）：
 * 1. 后端精确 paragraphMatchPositions（search_literature 提供）
 * 2. query 前缀直接匹配段落原文
 * 3. excerpt 剥省略号后匹配段落原文
 * 4. 从『实体名』提取实体扫描段落原文（出场轨迹/共现类查询的兜底）
 *
 * B-162: 当 SearchResultsView 调用时 chapters 可能尚未加载（targetPara=undefined），
 * 三层兜底全跳过 → highlights 为空。TextReaderView 在 chapters 加载后
 * 用此函数重算（watch paragraphs + activeSearch.ephemeral）。
 */

export interface ParagraphLike {
  index: number
  text: string
}

export interface SearchResultLike {
  chapterNumber: number | string
  paragraphIndex: number
  excerpt?: string
  paragraphMatchPositions?: [number, number][]
}

export interface CharRange {
  start: number
  end: number
}

/**
 * 计算跳转高亮的字符区间列表。
 *
 * @param query       查询词（如 "『许仙』出场轨迹"）
 * @param result      搜索结果条目（含 excerpt、paragraphMatchPositions）
 * @param targetPara  目标段落（可能为 undefined — chapters 未加载时）
 * @returns 字符区间数组（空 = 所有兜底都没命中）
 */
export function computeJumpHighlights(
  query: string,
  result: SearchResultLike,
  targetPara: ParagraphLike | undefined,
): CharRange[] {
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
          if (hi >= 0)
            positions.push([hi, hi + Math.min(excerptText.length, targetPara.text.length - hi)])
        }
      }
    }
  }

  // 兜底 3：从『实体名』提取实体，在段落原文中扫描
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

  return positions.map(([start, end]) => ({ start, end }))
}
