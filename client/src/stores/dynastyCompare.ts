import { defineStore } from 'pinia'
import { ref, computed } from 'vue'
import type {
  DynastyCompareConfig,
  ElementEvolution,
  LocationEvolution,
} from '@/types/dynasty-compare'
// LocationEvolution used in locationChanges state

// U-14: Five-state machine for dynasty compare lifecycle
export type CompareState = 'manual' | 'analyzing' | 'ready' | 'error' | 'timeout'

export const useDynastyCompareStore = defineStore('dynastyCompare', () => {
  // State
  const compareConfig = ref<DynastyCompareConfig | null>(null)
  const elements = ref<ElementEvolution[]>([])
  const locationChanges = ref<LocationEvolution[]>([])
  const summary = ref<{ overview: string; keyFindings: string[] } | null>(null)
  const isLoading = ref(false)
  const error = ref<string | null>(null)
  const compareState = ref<CompareState>('manual')
  const categoryFilter = ref<Set<string>>(new Set(['person', 'location', 'motif', 'term', 'event']))
  const selectedElement = ref<ElementEvolution | null>(null)
  const isDrawerOpen = ref(false)
  const linkedHighlight = ref<{ elementId: string; sourceSide: 'left' | 'right' } | null>(null)
  const syncScroll = ref(true)
  const pendingCompareDynasties = ref<{ left: string; right: string } | null>(null)

  // Getters
  const filteredElements = computed(() => {
    return elements.value.filter(e => categoryFilter.value.has(e.category))
  })

  const locationChangeCount = computed(() => locationChanges.value.length)

  const hasData = computed(() => elements.value.length > 0 || locationChanges.value.length > 0)

  // Actions
  function loadCompareData(data: {
    config: DynastyCompareConfig
    elements: ElementEvolution[]
    locationChanges: LocationEvolution[]
    summary: { overview: string; keyFindings: string[] }
  }) {
    compareConfig.value = data.config
    elements.value = data.elements || []
    locationChanges.value = data.locationChanges || []
    summary.value = data.summary || null
    isLoading.value = false
    error.value = null
    compareState.value = 'ready'
  }

  function setLoading(loading: boolean) {
    isLoading.value = loading
    if (loading) {
      error.value = null
      compareState.value = 'analyzing'
    }
  }

  function setError(message: string) {
    error.value = message
    isLoading.value = false
    compareState.value = 'error'
  }

  function setTimeoutState() {
    compareState.value = 'timeout'
  }

  function resetToManual() {
    compareState.value = 'manual'
    isLoading.value = false
    error.value = null
  }

  function openDrawer(element: ElementEvolution) {
    selectedElement.value = element
    isDrawerOpen.value = true
  }

  function closeDrawer() {
    isDrawerOpen.value = false
    //  selectedElement250ms
    setTimeout(() => { selectedElement.value = null }, 250)
  }

  function toggleCategoryFilter(category: string) {
    const s = new Set(categoryFilter.value)
    if (s.has(category)) {
      // 1 active
      if (s.size > 1) {
        s.delete(category)
      }
    } else {
      s.add(category)
    }
    categoryFilter.value = s
  }

  function toggleSyncScroll() {
    syncScroll.value = !syncScroll.value
  }

  function swapSides() {
    if (!compareConfig.value) return
    compareConfig.value = {
      ...compareConfig.value,
      left: compareConfig.value.right,
      right: compareConfig.value.left,
    }
  }

  function clearCompareData() {
    compareConfig.value = null
    elements.value = []
    locationChanges.value = []
    summary.value = null
    isLoading.value = false
    error.value = null
    compareState.value = 'manual'
    selectedElement.value = null
    isDrawerOpen.value = false
  }

  function setPendingCompareDynasties(dynasties: { left: string; right: string }) {
    pendingCompareDynasties.value = dynasties
  }

  function clearPendingCompareDynasties() {
    pendingCompareDynasties.value = null
  }

  return {
    // State
    compareConfig,
    elements,
    locationChanges,
    summary,
    isLoading,
    error,
    compareState,
    categoryFilter,
    selectedElement,
    isDrawerOpen,
    linkedHighlight,
    syncScroll,
    pendingCompareDynasties,

    // Getters
    filteredElements,
    locationChangeCount,
    hasData,

    // Actions
    loadCompareData,
    setLoading,
    setError,
    setTimeoutState,
    resetToManual,
    openDrawer,
    closeDrawer,
    toggleCategoryFilter,
    toggleSyncScroll,
    swapSides,
    clearCompareData,
    setPendingCompareDynasties,
    clearPendingCompareDynasties,
  }
})
