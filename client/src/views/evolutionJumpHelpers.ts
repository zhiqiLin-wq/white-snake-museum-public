/**
 * 演化分析引文跳转参数构造辅助函数（2026-09-09）。
 *
 * 从 EvolutionWorkspaceView.jumpToSourceFromModal 提取为纯函数，
 * 使其可被单元测试覆盖，并补齐 SearchResultsView 的 C06 兜底机制：
 * - flashParagraph: 一次性闪烁标记（TextReaderView 消费后删除，段落底色暂时高亮）
 * - citationHighlight: 归一化引文高亮（金色 mark），TextReaderView 用 NFC+前缀兜底
 * - paragraphIndex=-1: 显式传 -1，让 TextReaderView 走文本匹配路径而非段落号路径
 *
 * 与 SearchResultsView.navigateToSource 对齐：
 * - citationHighlight.text 优先 LLM 引文，回退 excerpt 前 12 字（与 TextReaderView 12 字前缀兜底对齐）
 * - < 4 字不传（TextReaderView 的 normQuote.length >= 4 门禁）
 */

export interface EvolutionJumpInput {
  /** 章节号（1-7），无效则不跳转 */
  chapterNumber: number
  /** 段落号，-1 表示未知（chunk 级缺失） */
  paragraphIndex: number
  /** LLM 生成的弯引号引文（从叙述文本正则提取），跳转高亮首选 */
  llmQuote: string
  /** 证据 chunk 文本前 25 字（fallback 高亮匹配素材） */
  excerptHead: string
}

export interface EvolutionJumpParams {
  chapterNumber: number
  paragraphIndex: number
  /** 一次性闪烁标记，TextReaderView 消费后删除 */
  flashParagraph: number
  /** 归一化引文高亮，< 4 字时省略 */
  citationHighlight?: { text: string }
}

const MIN_HIGHLIGHT_LEN = 4
const FALLBACK_EXCERPT_LEN = 12  // 与 TextReaderView 的 12 字前缀兜底对齐

/**
 * 构造跳转到原文的参数对象。
 * 返回 null 表示不跳转（章节号无效）。
 */
export function buildEvolutionJumpParams(input: EvolutionJumpInput): EvolutionJumpParams | null {
  const { chapterNumber, paragraphIndex, llmQuote, excerptHead } = input

  // 章节号无效 → 不跳转
  if (!Number.isFinite(chapterNumber) || chapterNumber <= 0) {
    return null
  }

  // citationHighlight.text: 优先 LLM 引文，回退 excerpt 前 12 字
  // 与 SearchResultsView 对齐：excerpt 剥省略号后取前缀
  const llmQuoteTrimmed = (llmQuote || '').trim()
  const excerptCore = (excerptHead || '').replace(/^[…⋯]+|[…⋯]+$/g, '').trim()
  const fallbackText = excerptCore.slice(0, FALLBACK_EXCERPT_LEN)

  const highlightText = llmQuoteTrimmed || (fallbackText.length >= MIN_HIGHLIGHT_LEN ? fallbackText : '')
  const shouldHighlight = highlightText.length >= MIN_HIGHLIGHT_LEN

  return {
    chapterNumber,
    paragraphIndex: normalizeParagraphIndex(paragraphIndex),  // -1 原样透传
    flashParagraph: Date.now(),  // 一次性闪烁标记
    citationHighlight: shouldHighlight ? { text: highlightText } : undefined,
  }
}

/**
 * B-159: 标签锚定式引文配对（前端版，与后端 validate_narrative_citations 的回溯
 * 配对同构）。
 *
 * 实锤 bug（浏览器实测）：LLM 输出几乎全用 ASCII 直引号 "（一篇报告 154 个直引号、
 * 0 个弯引号）。旧 citeRegex /["“]([^"”]{4,200})["”][^[]{0,60}\[label\]/ 从左
 * 向右匹配，而直引号同属开/闭两个字符类——前一对的闭引号会被当成下一对的开引号，
 * 把两段引号之间的散文（实测 data-quote 抓到“感气所化…开篇即以”而非真正引文
 * “可惜他千载焚修也那一旦抛”）误捕获；TextReader 拿散文去原文匹配，金色高亮与
 * 滚动定位全灭。此外旧正则把闭引号→标签之间最多 60 字的正文一并吞掉不渲染。
 *
 * 改为从标签向左回溯：先找最近闭引号（gap 内不得越过其他标签），再从闭引号前
 * 找最近开引号；开-闭片段内出现任何其他引号字符即判定配对歧义、放弃提取。
 */
const QUOTE_CLOSE_CHARS = ['”', '」', '"']
const QUOTE_OPEN_CHARS = ['“', '「', '"']
const QUOTE_ANY_CHARS = QUOTE_CLOSE_CHARS.concat(QUOTE_OPEN_CHARS)
const MAX_CLOSE_GAP = 60     // 闭引号 → 标签 的最大字符距离（对应旧正则 [^[]{0,60}）
const MAX_QUOTE_FRAGMENT = 200
const MIN_QUOTE_FRAGMENT = 4

export interface MatchedQuote {
  /** 引文正文（已 trim） */
  quote: string
  /** 开引号在原文中的下标 */
  openPos: number
  /** 闭引号在原文中的下标 */
  closePos: number
}

export function extractQuoteBeforeLabel(text: string, labelStart: number): MatchedQuote | null {
  // 从标签前 MAX_CLOSE_GAP 字符内向左回溯配对：
  // 1) 找最近闭引号候选；gap 内遇到 '['（越过其他标签）即放弃；
  // 2) 从闭引号前找最近引号字符作为开引号，片段长度需 4-200 字；
  // 3) 短片段（"堕落""宿缘"这类强调用短引号对，非引文）跳过；
  //    关键：跳过时外层扫描必须越过已配对的开引号继续向左——直引号 " 同时属于
  //    开/闭两个字符类，否则该开引号会被当成下一个闭引号候选、与更前面的开引号
  //    错位配对（实测把散文"的叹息定调，将化形视为"当成引文）。
  let p = labelStart - 1
  while (p >= Math.max(0, labelStart - MAX_CLOSE_GAP)) {
    if (text[p] === '[') return null
    if (!QUOTE_CLOSE_CHARS.includes(text[p])) { p--; continue }
    const closePos = p

    let openPos = -1
    for (let i = closePos - 1; i >= Math.max(0, closePos - MAX_QUOTE_FRAGMENT - 2); i--) {
      if (QUOTE_ANY_CHARS.includes(text[i])) {
        if (QUOTE_OPEN_CHARS.includes(text[i])) openPos = i
        break
      }
    }
    if (openPos < 0) { p--; continue }

    const frag = text.slice(openPos + 1, closePos)
    if (frag.length < MIN_QUOTE_FRAGMENT || frag.length > MAX_QUOTE_FRAGMENT) {
      p = openPos - 1  // 跳过已消费的引号对，避免直引号角色错位
      continue
    }
    const quote = frag.trim()
    if (quote.length < MIN_QUOTE_FRAGMENT) { p = openPos - 1; continue }

    return { quote, openPos, closePos }
  }
  return null
}

/**
 * 归一化段落号。
 *
 * 实锤 bug：旧代码在 navigateToSource 里把 paragraphIndex=-1（93% 的 chunk
 * 段落号缺失，后端显式给 -1）用 Math.max(0, idx) 钳成 0，导致"跳转到原文"
 * 定位到章节首段（错误位置），且 TextReaderView 走段落号路径而非
 * citationHighlight 文本匹配路径，用户看不到定位与金色高亮。
 *
 * 规则：非负整数原样返回；-1（未知）保留；null/undefined/NaN/<-1/
 * 无法解析的字符串一律返回 -1（走文本匹配兜底）；数字字符串解析为整数。
 */
export function normalizeParagraphIndex(raw: unknown): number {
  if (raw === null || raw === undefined || raw === '') return -1
  if (typeof raw === 'number') {
    if (!Number.isFinite(raw)) return -1
    if (raw < -1) return -1
    return Math.trunc(raw)
  }
  if (typeof raw === 'string') {
    const trimmed = raw.trim()
    if (!/^-?\d+$/.test(trimmed)) return -1
    const n = parseInt(trimmed, 10)
    if (n < -1) return -1
    return n
  }
  return -1
}
