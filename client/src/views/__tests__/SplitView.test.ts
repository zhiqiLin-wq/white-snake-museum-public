import { describe, it, expect, beforeEach } from 'vitest'
import { mount } from '@vue/test-utils'
import { setActivePinia, createPinia } from 'pinia'
import { useWorkspaceStore } from '@/stores/workspace'

// Minimal component test for SplitView structure
describe('SplitView', () => {
  beforeEach(() => {
    setActivePinia(createPinia())
  })

  it('should have split state initialized', () => {
    const store = useWorkspaceStore()
    expect(store.split.enabled).toBe(false)
    expect(store.split.splitRatio).toBe(0.5)
    expect(store.split.syncScroll).toBe(false)
    expect(store.split.alignParagraphs).toBe(false)
  })

  it('should enter and exit split mode via workspace store', () => {
    const store = useWorkspaceStore()
    const tab1 = store.openTab('text-reader', 'Chapter 1', { chapterNumber: 1 })
    const tab2 = store.openTab('text-reader', 'Chapter 2', { chapterNumber: 2 })

    store.enterSplitMode(tab1.id, tab2.id)
    expect(store.split.enabled).toBe(true)
    expect(store.split.leftTabId).toBe(tab1.id)
    expect(store.split.rightTabId).toBe(tab2.id)

    store.exitSplitMode()
    expect(store.split.enabled).toBe(false)
  })

  it('should clamp split ratio between 0.3 and 0.7', () => {
    const store = useWorkspaceStore()

    store.setSplitRatio(0.5)
    expect(store.split.splitRatio).toBe(0.5)

    store.setSplitRatio(0.1)
    expect(store.split.splitRatio).toBe(0.3)

    store.setSplitRatio(0.9)
    expect(store.split.splitRatio).toBe(0.7)
  })

  it('should toggle syncScroll and alignParagraphs', () => {
    const store = useWorkspaceStore()

    store.toggleSyncScroll()
    expect(store.split.syncScroll).toBe(true)
    store.toggleSyncScroll()
    expect(store.split.syncScroll).toBe(false)

    store.toggleAlignParagraphs()
    expect(store.split.alignParagraphs).toBe(true)
    store.toggleAlignParagraphs()
    expect(store.split.alignParagraphs).toBe(false)
  })

  it('should swap split sides', () => {
    const store = useWorkspaceStore()
    const tab1 = store.openTab('text-reader', 'Left', { chapterNumber: 1 })
    const tab2 = store.openTab('text-reader', 'Right', { chapterNumber: 2 })
    store.enterSplitMode(tab1.id, tab2.id)

    store.swapSplitSides()
    expect(store.split.leftTabId).toBe(tab2.id)
    expect(store.split.rightTabId).toBe(tab1.id)
  })
})
