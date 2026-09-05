import { defineStore } from 'pinia'
import { ref, computed } from 'vue'
import type { WorkspaceTab, TabType, SplitState } from '@/types/workspace'
import { useEvolutionStore } from './evolution'

function genId(): string {
  return Date.now().toString(36) + Math.random().toString(36).slice(2, 8)
}

export const useWorkspaceStore = defineStore('workspace', () => {
  const tabs = ref<WorkspaceTab[]>([])
  const activeTabId = ref<string | null>(null)
  const maxTabs = 6

  const split = ref<SplitState>({
    enabled: false,
    leftTabId: null,
    rightTabId: null,
    splitRatio: 0.5,
    syncScroll: false,
    alignParagraphs: false,
  })

  const activeTab = computed(() =>
    tabs.value.find(t => t.id === activeTabId.value) || null
  )

  const tabCount = computed(() => tabs.value.length)

  const hasEvolutionWorkspace = computed(() =>
    tabs.value.some(t => t.type === 'evolution-workspace')
  )

  function findTabByType(type: TabType, dataTarget?: string): WorkspaceTab | undefined {
    return tabs.value.find(t => {
      if (t.type !== type) return false
      if (dataTarget && (t.data as Record<string, unknown>).target !== dataTarget) return false
      return true
    })
  }

  function setActiveTab(tabId: string) {
    activeTabId.value = tabId
  }

  function openTab(type: TabType, title: string, data: Record<string, unknown> = {}): WorkspaceTab {
    const tab = createTab(type, title, data)
    activeTabId.value = tab.id
    return tab
  }

  /**  DynastyCompareView  */
  function createTab(type: TabType, title: string, data: Record<string, unknown> = {}): WorkspaceTab {
    // Dedup: same type + same chapterNumber already exists -> merge data and return existing
    if (type === 'text-reader') {
      const chNum = data.chapterNumber as number | undefined
      if (chNum !== undefined) {
        const existing = tabs.value.find(t => {
          if (t.type !== 'text-reader') return false
          return (t.data as Record<string, unknown>).chapterNumber === chNum
        })
        if (existing) {
          // Merge new data into existing tab so paragraphIndex etc. survive dedup
          existing.data = { ...existing.data, ...data }
          if (title) {
            existing.title = title
          }
          return existing
        }
      }
    } else if (type === 'evolution-workspace') {
      // Evolution workspace: dedup by threadId (allow multiple tabs for different analyses)
      const dataThreadId = data.threadId as string | undefined
      if (dataThreadId) {
        const existing = tabs.value.find(t => {
          if (t.type !== 'evolution-workspace') return false
          return (t.data as Record<string, unknown>).threadId === dataThreadId
        })
        if (existing) {
          existing.data = { ...existing.data, ...data }
          if (title) existing.title = title
          return existing
        }
      }
    } else {
      const dataTarget = data.target as string | undefined
      const existing = findTabByType(type, dataTarget)
      if (existing) {
        // v19.2: 复用时合并新 data（如 report-library 的 reportId），
        // 否则二次"预览"会被旧 data 覆盖、watch 收不到变化
        existing.data = { ...existing.data, ...data }
        if (title) existing.title = title
        return existing
      }
    }

    // Dynasty-compare dedup by left+right chapterNumber combination
    if (type === 'dynasty-compare') {
      const leftNum = (data.config as Record<string, unknown> | undefined)?.left as Record<string, unknown> | undefined
      const rightNum = (data.config as Record<string, unknown> | undefined)?.right as Record<string, unknown> | undefined
      const leftCh = leftNum?.chapterNumber as number | undefined
      const rightCh = rightNum?.chapterNumber as number | undefined
      if (leftCh !== undefined && rightCh !== undefined) {
        const existing = tabs.value.find(t => {
          if (t.type !== 'dynasty-compare') return false
          const td = t.data as Record<string, unknown>
          const tdConfig = td.config as Record<string, unknown> | undefined
          const tdLeft = (tdConfig?.left as Record<string, unknown> | undefined)?.chapterNumber
          const tdRight = (tdConfig?.right as Record<string, unknown> | undefined)?.chapterNumber
          return tdLeft === leftCh && tdRight === rightCh
        })
        if (existing) return existing
      }
    }

    // LRU eviction
    if (tabs.value.length >= maxTabs) {
      const sorted = [...tabs.value].sort((a, b) => a.createdAt - b.createdAt)
      const oldest = sorted[0]
      closeTab(oldest.id)
    }

    const newTab: WorkspaceTab = {
      id: genId(),
      type,
      title,
      data,
      dirty: false,
      createdAt: Date.now(),
    }

    tabs.value.push(newTab)
    return newTab
  }

  function closeTab(tabId: string) {
    const idx = tabs.value.findIndex(t => t.id === tabId)
    if (idx === -1) return

    const tab = tabs.value[idx]

    // 
    if (tab.type === 'evolution-workspace') {
      const tid = (tab.data as Record<string, unknown>).threadId as string | undefined
      if (tid) {
        useEvolutionStore().removeCompletedAnalysis(tid)
      }
    }

    // 
    if (split.value.enabled) {
      if (tabId === split.value.leftTabId || tabId === split.value.rightTabId) {
        const otherSideId = tabId === split.value.leftTabId
          ? split.value.rightTabId : split.value.leftTabId
        const otherSideExists = otherSideId && tabs.value.some(t => t.id === otherSideId && t.id !== tabId)
        if (otherSideExists) {
          // 
          if (tabId === split.value.leftTabId) {
            split.value.leftTabId = null
          } else {
            split.value.rightTabId = null
          }
        } else {
          // 
          exitSplitMode()
        }
      }
    }

    tabs.value.splice(idx, 1)

    if (activeTabId.value === tabId) {
      if (tabs.value.length === 0) {
        activeTabId.value = null
      } else if (idx < tabs.value.length) {
        activeTabId.value = tabs.value[idx].id
      } else {
        activeTabId.value = tabs.value[tabs.value.length - 1].id
      }
    }
  }

  function reorderTabs(fromIndex: number, toIndex: number) {
    if (fromIndex === toIndex) return
    if (fromIndex < 0 || fromIndex >= tabs.value.length) return
    if (toIndex < 0 || toIndex >= tabs.value.length) return
    const [moved] = tabs.value.splice(fromIndex, 1)
    tabs.value.splice(toIndex, 0, moved)
  }

  function getTabIndex(tabId: string): number {
    return tabs.value.findIndex(t => t.id === tabId)
  }

  function updateTabData(tabId: string, data: Record<string, unknown>) {
    const tab = tabs.value.find(t => t.id === tabId)
    if (tab) {
      tab.data = { ...tab.data, ...data }
      tab.dirty = true
    }
  }

  // ===== Split mode =====

  function enterSplitMode(leftTabId: string, rightTabId: string) {
    split.value.enabled = true
    split.value.leftTabId = leftTabId
    split.value.rightTabId = rightTabId
  }

  function exitSplitMode() {
    split.value.enabled = false
    split.value.leftTabId = null
    split.value.rightTabId = null
  }

  function setSplitRatio(ratio: number) {
    const clamped = Math.max(0.3, Math.min(0.7, ratio))
    split.value.splitRatio = clamped
  }

  function toggleSyncScroll() {
    split.value.syncScroll = !split.value.syncScroll
  }

  function toggleAlignParagraphs() {
    split.value.alignParagraphs = !split.value.alignParagraphs
  }

  function swapSplitSides() {
    const left = split.value.leftTabId
    split.value.leftTabId = split.value.rightTabId
    split.value.rightTabId = left
  }

  const activeSplitSide = ref<'left' | 'right' | null>(null)

  function setActiveSplitSide(side: 'left' | 'right' | null) {
    activeSplitSide.value = side
  }

  return {
    tabs,
    activeTabId,
    maxTabs,
    split,
    activeTab,
    tabCount,
    hasEvolutionWorkspace,
    openTab,
    createTab,
    closeTab,
    setActiveTab,
    updateTabData,
    findTabByType,
    reorderTabs,
    getTabIndex,
    enterSplitMode,
    exitSplitMode,
    setSplitRatio,
    toggleSyncScroll,
    toggleAlignParagraphs,
    swapSplitSides,
    activeSplitSide,
    setActiveSplitSide,
  }
})
