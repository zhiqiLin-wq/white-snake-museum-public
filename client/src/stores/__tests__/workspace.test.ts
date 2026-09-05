import { describe, it, expect, beforeEach } from 'vitest'
import { setActivePinia, createPinia } from 'pinia'
import { useWorkspaceStore } from '@/stores/workspace'

describe('useWorkspaceStore', () => {
  beforeEach(() => {
    setActivePinia(createPinia())
  })

  it('should open a tab', () => {
    const store = useWorkspaceStore()
    store.openTab('text-reader', '唐 李黄')
    expect(store.tabs.length).toBe(1)
    expect(store.activeTab).not.toBeNull()
    expect(store.activeTab!.title).toBe('唐 李黄')
  })

  it('should deduplicate tabs with same type and target', () => {
    const store = useWorkspaceStore()
    store.openTab('text-reader', '唐 李黄', { target: '1' })
    store.openTab('text-reader', '唐 李黄', { target: '1' })
    expect(store.tabs.length).toBe(1)
  })

  it('should LRU evict when exceeding maxTabs', () => {
    const store = useWorkspaceStore()
    for (let i = 0; i < 6; i++) {
      store.openTab('text-reader', `Tab ${i}`, { target: `${i}` })
    }
    expect(store.tabs.length).toBe(6)
    store.openTab('text-reader', 'Tab 7', { target: '7' })
    expect(store.tabs.length).toBe(6)
  })

  it('should close tab and activate adjacent', () => {
    const store = useWorkspaceStore()
    store.openTab('text-reader', 'Tab 1', { target: '1' })
    const tab2 = store.openTab('text-reader', 'Tab 2', { target: '2' })
    store.openTab('text-reader', 'Tab 3', { target: '3' })
    store.setActiveTab(tab2.id)
    store.closeTab(tab2.id)
    expect(store.tabs.length).toBe(2)
    expect(store.activeTab).not.toBeNull()
  })

  it('should enforce evolution-workspace uniqueness', () => {
    const store = useWorkspaceStore()
    store.openTab('evolution-workspace', 'Evolution 1')
    store.openTab('evolution-workspace', 'Evolution 2')
    expect(store.tabs.length).toBe(1)
    expect(store.hasEvolutionWorkspace).toBe(true)
  })

  // J-03: SplitState tests

  it('should start with split disabled', () => {
    const store = useWorkspaceStore()
    expect(store.split.enabled).toBe(false)
    expect(store.split.leftTabId).toBeNull()
    expect(store.split.rightTabId).toBeNull()
    expect(store.split.splitRatio).toBe(0.5)
  })

  it('should enter and exit split mode', () => {
    const store = useWorkspaceStore()
    const tab1 = store.openTab('text-reader', 'Chapter 1', { chapterNumber: 1 })
    const tab2 = store.openTab('text-reader', 'Chapter 2', { chapterNumber: 2 })

    store.enterSplitMode(tab1.id, tab2.id)
    expect(store.split.enabled).toBe(true)
    expect(store.split.leftTabId).toBe(tab1.id)
    expect(store.split.rightTabId).toBe(tab2.id)

    store.exitSplitMode()
    expect(store.split.enabled).toBe(false)
    expect(store.split.leftTabId).toBeNull()
    expect(store.split.rightTabId).toBeNull()
  })

  it('should set split ratio (clamped)', () => {
    const store = useWorkspaceStore()
    store.setSplitRatio(0.6)
    expect(store.split.splitRatio).toBe(0.6)

    store.setSplitRatio(0.1)
    expect(store.split.splitRatio).toBe(0.3)  // clamped to min 30%

    store.setSplitRatio(0.9)
    expect(store.split.splitRatio).toBe(0.7)  // clamped to max 70%
  })

  it('should toggle sync scroll and align paragraphs', () => {
    const store = useWorkspaceStore()
    expect(store.split.syncScroll).toBe(false)
    store.toggleSyncScroll()
    expect(store.split.syncScroll).toBe(true)

    expect(store.split.alignParagraphs).toBe(false)
    store.toggleAlignParagraphs()
    expect(store.split.alignParagraphs).toBe(true)
  })

  it('should swap split sides', () => {
    const store = useWorkspaceStore()
    const tab1 = store.openTab('text-reader', 'Chapter 1', { chapterNumber: 1 })
    const tab2 = store.openTab('text-reader', 'Chapter 2', { chapterNumber: 2 })
    store.enterSplitMode(tab1.id, tab2.id)
    store.swapSplitSides()
    expect(store.split.leftTabId).toBe(tab2.id)
    expect(store.split.rightTabId).toBe(tab1.id)
  })
})
