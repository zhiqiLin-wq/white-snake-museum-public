/**
 * 演化分析引文跳转参数构造回归测试（测试先行）。
 *
 * 覆盖 2026-09-09 修复：演化分析"跳转到原文"按钮点击后只跳到文献开头，
 * 没定位到 chunk 也没高亮。根因：EvolutionWorkspaceView 没复用
 * SearchResultsView 的 C06 兜底机制（flashParagraph 闪烁标记 +
 * citationHighlight 短前缀回退 + paragraphIndex=-1 显式传递）。
 *
 * 测试目标:
 * 1. 跳转参数必含 flashParagraph（一次性闪烁标记，TextReaderView 消费后删除）
 * 2. citationHighlight.text 优先用 LLM 引文，回退到 excerpt 前 12 字（非 25 字）
 * 3. citationHighlight.text 长度 < 4 时不传（避免无效匹配）
 * 4. paragraphIndex=-1 时仍传 -1（让 TextReaderView 走文本匹配路径而非段落号路径）
 * 5. chapterNumber 为 NaN/<=0 时不跳转（返回 null）
 *
 * 运行: cd client && npx vitest run src/views/evolutionJump.test.ts
 */
import { describe, it, expect } from 'vitest'
import {
  buildEvolutionJumpParams,
  normalizeParagraphIndex,
  extractQuoteBeforeLabel,
} from './evolutionJumpHelpers'

describe('buildEvolutionJumpParams', () => {
  // ---- 1. 必含 flashParagraph 闪烁标记 ----

  it('includes flashParagraph timestamp for one-shot flash highlight', () => {
    const params = buildEvolutionJumpParams({
      chapterNumber: 4,
      paragraphIndex: 5,
      llmQuote: '白娘子永镇雷峰塔下',
      excerptHead: '...',
    })
    expect(params).not.toBeNull()
    expect(params!.flashParagraph).toBeDefined()
    expect(typeof params!.flashParagraph).toBe('number')
    expect(params!.flashParagraph).toBeGreaterThan(0)
  })

  // ---- 2. citationHighlight 优先 LLM 引文，回退 excerpt 前 12 字 ----

  it('prefers LLM quote for citationHighlight text', () => {
    const params = buildEvolutionJumpParams({
      chapterNumber: 4,
      paragraphIndex: 5,
      llmQuote: '白娘子永镇雷峰塔下',
      excerptHead: '其他不相关文本',
    })
    expect(params).not.toBeNull()
    expect(params!.citationHighlight).toBeDefined()
    expect(params!.citationHighlight!.text).toBe('白娘子永镇雷峰塔下')
  })

  it('falls back to excerpt head 12 chars (not 25) when LLM quote is empty', () => {
    const longExcerpt = '许仙在断桥亭中与白氏相遇风雨同舟而归这段描写...'
    const params = buildEvolutionJumpParams({
      chapterNumber: 4,
      paragraphIndex: 5,
      llmQuote: '',
      excerptHead: longExcerpt,
    })
    expect(params).not.toBeNull()
    expect(params!.citationHighlight).toBeDefined()
    // 前 12 字，与 TextReaderView 的 12 字前缀兜底对齐
    expect(params!.citationHighlight!.text).toBe('许仙在断桥亭中与白氏相遇')
    expect(params!.citationHighlight!.text.length).toBe(12)
  })

  // ---- 3. citationHighlight.text < 4 字时不传 ----

  it('omits citationHighlight when text shorter than 4 chars', () => {
    const params = buildEvolutionJumpParams({
      chapterNumber: 4,
      paragraphIndex: 5,
      llmQuote: '降',  // 单字
      excerptHead: '塔',  // 单字
    })
    expect(params).not.toBeNull()
    expect(params!.citationHighlight).toBeUndefined()
  })

  // ---- 4. paragraphIndex=-1 时仍传 -1 ----

  it('passes paragraphIndex=-1 explicitly so TextReaderView uses text matching path', () => {
    const params = buildEvolutionJumpParams({
      chapterNumber: 4,
      paragraphIndex: -1,  // chunk 级段落号缺失（93% 的 chunk）
      llmQuote: '白娘子永镇雷峰塔下',
      excerptHead: '...',
    })
    expect(params).not.toBeNull()
    expect(params!.paragraphIndex).toBe(-1)
    // -1 时仍要有 citationHighlight 让 TextReaderView 走文本匹配
    expect(params!.citationHighlight).toBeDefined()
  })

  // ---- 5. chapterNumber 无效时返回 null ----

  it('returns null when chapterNumber is NaN or <= 0', () => {
    expect(buildEvolutionJumpParams({
      chapterNumber: 0, paragraphIndex: 5, llmQuote: 'x', excerptHead: 'y',
    })).toBeNull()
    expect(buildEvolutionJumpParams({
      chapterNumber: NaN, paragraphIndex: 5, llmQuote: 'x', excerptHead: 'y',
    })).toBeNull()
    expect(buildEvolutionJumpParams({
      chapterNumber: -1, paragraphIndex: 5, llmQuote: 'x', excerptHead: 'y',
    })).toBeNull()
  })

  // ---- 6. 空引文 + 空 excerpt → 不传 citationHighlight 但仍跳转 ----

  it('still jumps without citationHighlight when both quote and excerpt are empty', () => {
    const params = buildEvolutionJumpParams({
      chapterNumber: 4,
      paragraphIndex: 5,
      llmQuote: '',
      excerptHead: '',
    })
    expect(params).not.toBeNull()
    expect(params!.chapterNumber).toBe(4)
    expect(params!.citationHighlight).toBeUndefined()
    // flashParagraph 仍要有
    expect(params!.flashParagraph).toBeDefined()
  })
})

/**
 * normalizeParagraphIndex 回归测试（2026-09-09）。
 *
 * 实锤 bug：EvolutionWorkspaceView.navigateToSource 把 paragraphIndex=-1
 * （93% chunk 段落号缺失）钳成 0，导致"跳转到原文"定位到章节首段（错误位置），
 * 且 TextReaderView 走段落号路径而非 citationHighlight 文本匹配路径，
 * 用户看不到定位与金色高亮。-1 必须原样透传。
 */
describe('normalizeParagraphIndex', () => {
  it('keeps valid non-negative integers as-is', () => {
    expect(normalizeParagraphIndex(0)).toBe(0)
    expect(normalizeParagraphIndex(5)).toBe(5)
    expect(normalizeParagraphIndex(120)).toBe(120)
  })

  it('preserves -1 (unknown paragraph) instead of clamping to 0', () => {
    expect(normalizeParagraphIndex(-1)).toBe(-1)
  })

  it('returns -1 for null/undefined/NaN/invalid', () => {
    expect(normalizeParagraphIndex(undefined)).toBe(-1)
    expect(normalizeParagraphIndex(null)).toBe(-1)
    expect(normalizeParagraphIndex(NaN)).toBe(-1)
    expect(normalizeParagraphIndex(-99)).toBe(-1)
  })

  it('parses numeric strings, falls back to -1 for garbage', () => {
    expect(normalizeParagraphIndex('7')).toBe(7)
    expect(normalizeParagraphIndex('0')).toBe(0)
    expect(normalizeParagraphIndex('abc')).toBe(-1)
    expect(normalizeParagraphIndex('')).toBe(-1)
  })
})

describe('extractQuoteBeforeLabel', () => {
  // B-159 浏览器实锤：直引号场景下旧正则把散文误当引文

  it('pairs the NEAREST straight-quote pair before label (not prose between pairs)', () => {
    // 实测真实叙述片段：前一对 "东海白龙" 与标签相距 >60 字，真正引文是
    // "可惜他千载焚修也那一旦抛"；旧正则会把中间散文捕获为 quote
    const text = '此处白蛇是"东海白龙"感气所化的灵物，其化形被纳入天命叙事。方成培《雷峰塔传奇》中，白蛇的妖性与人性并存，开篇即以"可惜他千载焚修也那一旦抛"的叹息定调，将化形视为"堕落"与"宿缘"的双重起点[清-ch4:0]'
    const labelStart = text.indexOf('[清-ch4:0]')
    const mq = extractQuoteBeforeLabel(text, labelStart)
    expect(mq).not.toBeNull()
    expect(mq!.quote).toBe('可惜他千载焚修也那一旦抛')
  })

  it('extracts curly-quote citation directly adjacent to label', () => {
    const text = '李黄归家后“才及家，便觉脑疼”[唐-ch1:0]，随后不治。'
    const labelStart = text.indexOf('[唐-ch1:0]')
    const mq = extractQuoteBeforeLabel(text, labelStart)
    expect(mq).not.toBeNull()
    expect(mq!.quote).toBe('才及家，便觉脑疼')
  })

  it('returns null when label has no quote before it', () => {
    const text = '这是一句没有引文的陈述[清-ch4:1]'
    expect(extractQuoteBeforeLabel(text, text.indexOf('[清-ch4:1]'))).toBeNull()
  })

  it('returns null when closest quote pair is too far (>60 char gap)', () => {
    // 闭引号距标签 64 字（中文 60+），超出旧正则容忍度
    const text = '引文“脑裂而卒”' + '的' .repeat(64) + '[唐-ch1:2]'
    expect(extractQuoteBeforeLabel(text, text.indexOf('[唐-ch1:2]'))).toBeNull()
  })

  it('returns null when nearest quote char is an orphan close quote', () => {
    // 只有一个孤闭引号（实际是开引号缺失），配对歧义
    const text = '某段散文之后”没有配对开引号[宋-ch2:3]'
    expect(extractQuoteBeforeLabel(text, text.indexOf('[宋-ch2:3]'))).toBeNull()
  })

  it('does not cross another label while scanning backwards', () => {
    const text = '前文[唐-ch0:0]之后还有散文"某段引文是也"[清-ch4:0]'
    // 清标签前 60 字内含 [唐-ch0:0]，扫描越过引文后遇到 [ 即终止；
    // 引文"某段引文是也"位于 [唐-ch0:0] 之后，仍可正常配对
    const labelStart = text.indexOf('[清-ch4:0]')
    const mq = extractQuoteBeforeLabel(text, labelStart)
    expect(mq).not.toBeNull()
    expect(mq!.quote).toBe('某段引文是也')
  })
})
