import { defineStore } from 'pinia'
import { ref, computed } from 'vue'
import type { Chapter, Paragraph, SearchResult, SearchHighlight, ActiveSearch } from '@/types/literature'

export interface NavigationTarget {
  chapterNumber: number
  paragraphIndex: number
  timestamp: number
}

export const useSourceTreeStore = defineStore('sourceTree', () => {
  const chapters = ref<Chapter[]>([])
  const expandedChapters = ref<Set<number>>(new Set([1]))
  const selectedNode = ref<{ chapterNumber: number; paragraphIndex?: number } | null>(null)
  const searchQuery = ref('')
  const searchResults = ref<SearchResult[]>([])
  const isSearching = ref(false)
  const loading = ref(false)
  const error = ref<string | null>(null)
  const pendingCompareChapter = ref<number | null>(null)
  const navigationTarget = ref<NavigationTarget | null>(null)
  const activeSearch = ref<ActiveSearch | null>(null)

  // ===== Getters =====

  const chapterCount = computed(() => chapters.value.length)

  const filteredSearchResults = computed(() => {
    if (!searchQuery.value.trim()) return []
    return searchResults.value
  })

  // ===== Actions =====

  interface RawApiChapter {
    number: string
    title: string
    content: string
    paragraphs?: { index: number; text: string }[]
  }

  const CHINESE_NUMBERS: Record<string, number> = {
    '一': 1, '二': 2, '三': 3, '四': 4, '五': 5,
    '六': 6, '七': 7, '八': 8, '九': 9, '十': 10,
    '十一': 11, '十二': 12, '十三': 13, '十四': 14, '十五': 15,
    '零': 0,
  }

  function parseChapterNumber(raw: string): number {
    // raw can be "", "", ... "", "", or already numeric
    const trimmed = raw.trim()
    if (CHINESE_NUMBERS[trimmed] !== undefined) return CHINESE_NUMBERS[trimmed]
    return parseInt(trimmed, 10) || 0
  }

  function extractDynasty(title: string): string {
    const match = title.match(/(唐|宋|元|明|清)/)
    return match ? match[0] : ''
  }

  function splitParagraphs(content: string): { index: number; text: string }[] {
    // Normalize line endings to \n first, matching chunker's _clean_text behavior.
    // This ensures paragraph indices are consistent between backend (chunker) and frontend.
    const normalized = content.replace(/\r\n|\r/g, '\n')
    return normalized
      .split(/\n\n/)
      .map(p => p.trim())
      .filter(p => p.length > 0)
      .map((text, i) => ({ index: i, text }))
  }

  async function fetchChapters(): Promise<void> {
    loading.value = true
    error.value = null
    try {
      const res = await fetch('/api/literature')
      if (!res.ok) throw new Error(`HTTP ${res.status}`)
      const rawData: RawApiChapter[] = await res.json()
      chapters.value = rawData.map((raw) => ({
        number: parseChapterNumber(raw.number),
        title: raw.title,
        dynasty: extractDynasty(raw.title),
        paragraphs: splitParagraphs(raw.content),
      }))
    } catch (err: unknown) {
      const message = err instanceof Error ? err.message : ''
      error.value = message
      chapters.value = []
    } finally {
      loading.value = false
    }
  }

  function toggleChapter(chapterNumber: number) {
    if (expandedChapters.value.has(chapterNumber)) {
      expandedChapters.value.delete(chapterNumber)
    } else {
      expandedChapters.value.add(chapterNumber)
    }
  }

  function selectChapter(chapterNumber: number) {
    selectedNode.value = { chapterNumber }
  }

  function selectParagraph(chapterNumber: number, paragraphIndex: number) {
    selectedNode.value = { chapterNumber, paragraphIndex }
  }

  async function search(query: string): Promise<void> {
    searchQuery.value = query
    if (query.trim().length < 2) {
      searchResults.value = []
      activeSearch.value = null
      return
    }

    isSearching.value = true
    const q = query.trim().toLowerCase()
    const results: SearchResult[] = []
    const highlights: SearchHighlight[] = []

    for (const ch of chapters.value) {
      for (const para of ch.paragraphs) {
        const lowerText = para.text.toLowerCase()
        const matchPositions: [number, number][] = []
        const matchPositionsInExcerpt: [number, number][] = []

        let searchFrom = 0
        while (searchFrom < lowerText.length) {
          const idx = lowerText.indexOf(q, searchFrom)
          if (idx === -1) break
          matchPositions.push([idx, idx + q.length])
          searchFrom = idx + 1
        }

        if (matchPositions.length > 0) {
          const firstMatch = matchPositions[0]
          const excerptOffset = Math.max(0, firstMatch[0] - 20)
          const excerptEnd = Math.min(para.text.length, firstMatch[1] + 40)
          const excerpt = para.text.slice(excerptOffset, excerptEnd)

          for (const [s, e] of matchPositions) {
            matchPositionsInExcerpt.push([s - excerptOffset, e - excerptOffset])
          }

          results.push({
            chapterNumber: ch.number,
            chapterTitle: ch.title,
            dynasty: ch.dynasty,
            paragraphIndex: para.index,
            excerpt,
            excerptOffset,
            matchPositions,
            matchPositionsInExcerpt,
            score: matchPositions.length / Math.pow(para.text.length, 0.3),
          })

          for (const [start, end] of matchPositions) {
            highlights.push({
              query: query.trim(),
              chapterNumber: ch.number,
              paragraphIndex: para.index,
              startChar: start,
              endChar: end,
              text: para.text.slice(start, end),
            })
          }
        }
      }
    }

    results.sort((a, b) => b.score - a.score)
    searchResults.value = results
    activeSearch.value = {
      query: query.trim(),
      highlights,
      source: 'user',
    }
    isSearching.value = false
  }

  function clearSearch() {
    searchQuery.value = ''
    searchResults.value = []
    activeSearch.value = null
  }

  function setPendingCompare(chapterNumber: number) {
    pendingCompareChapter.value = chapterNumber
  }

  function clearPendingCompare() {
    pendingCompareChapter.value = null
  }

  function executeCompare(currentChapterNumber: number): { leftChapter: number; rightChapter: number } | null {
    if (pendingCompareChapter.value === null) return null
    const result = {
      leftChapter: currentChapterNumber,
      rightChapter: pendingCompareChapter.value,
    }
    clearPendingCompare()
    return result
  }

  function getParagraph(chapterNumber: number, paragraphIndex: number): Paragraph | null {
    const ch = chapters.value.find(c => c.number === chapterNumber)
    if (!ch) return null
    return ch.paragraphs.find(p => p.index === paragraphIndex) || null
  }

  function getChapterParagraphs(chapterNumber: number): Paragraph[] {
    const ch = chapters.value.find(c => c.number === chapterNumber)
    return ch?.paragraphs || []
  }

  /**
   * 阶段 0 — P0-03: 返回标准化段落列表，供 Agent 标注请求使用。
   * 返回的 text 与 TextReaderView 渲染的段落文本完全一致（同一切分结果）。
   */
  function getParagraphList(chapterNumber: number): { index: number; text: string }[] {
    const ch = chapters.value.find(c => c.number === chapterNumber)
    if (!ch || !ch.paragraphs) return []
    return ch.paragraphs.map(p => ({ index: p.index, text: p.text }))
  }

  function setNavigationTarget(target: { chapterNumber: number; paragraphIndex: number }) {
    navigationTarget.value = {
      chapterNumber: target.chapterNumber,
      paragraphIndex: target.paragraphIndex,
      timestamp: Date.now(),
    }
  }

  function clearNavigationTarget() {
    navigationTarget.value = null
  }

  return {
    chapters,
    expandedChapters,
    selectedNode,
    searchQuery,
    searchResults,
    isSearching,
    loading,
    error,
    pendingCompareChapter,
    chapterCount,
    filteredSearchResults,
    fetchChapters,
    toggleChapter,
    selectChapter,
    selectParagraph,
    search,
    clearSearch,
    setPendingCompare,
    clearPendingCompare,
    executeCompare,
    getParagraph,
    getChapterParagraphs,
    getParagraphList,
    navigationTarget,
    setNavigationTarget,
    clearNavigationTarget,
    activeSearch,
  }
})
