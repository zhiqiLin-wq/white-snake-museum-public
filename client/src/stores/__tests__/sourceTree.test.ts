import { describe, it, expect, beforeEach } from 'vitest'
import { setActivePinia, createPinia } from 'pinia'
import { useSourceTreeStore } from '@/stores/sourceTree'

describe('useSourceTreeStore', () => {
  beforeEach(() => {
    setActivePinia(createPinia())
  })

  it('should start with empty chapters', () => {
    const store = useSourceTreeStore()
    expect(store.chapters.length).toBe(0)
    expect(store.chapterCount).toBe(0)
  })

  it('should toggle chapter expansion', () => {
    const store = useSourceTreeStore()
    expect(store.expandedChapters.has(1)).toBe(true)
    store.toggleChapter(1)
    expect(store.expandedChapters.has(1)).toBe(false)
    store.toggleChapter(2)
    expect(store.expandedChapters.has(2)).toBe(true)
  })

  it('should select chapter and paragraph', () => {
    const store = useSourceTreeStore()
    store.selectChapter(4)
    expect(store.selectedNode?.chapterNumber).toBe(4)
    store.selectParagraph(4, 12)
    expect(store.selectedNode?.paragraphIndex).toBe(12)
  })

  it('should search across chapters', async () => {
    const store = useSourceTreeStore()
    store.chapters = [
      {
        number: 1,
        title: '李黄',
        dynasty: '唐',
        paragraphs: [
          { index: 0, text: '有白衣老妪，居于路侧。' },
          { index: 1, text: '此蛇修炼千年，已通人性。' },
        ],
      },
    ]
    await store.search('修炼')
    expect(store.searchResults.length).toBeGreaterThan(0)
    expect(store.searchResults[0].chapterTitle).toBe('李黄')
  })

  it('should clear search', async () => {
    const store = useSourceTreeStore()
    await store.search('test')
    store.clearSearch()
    expect(store.searchQuery).toBe('')
    expect(store.searchResults.length).toBe(0)
  })
})
