import { ref, computed } from 'vue'
import type { LocationData, Chapter, ExcelRecord, Keyword } from '../types'

// ==================== Locations ====================
const allLocations = ref<Map<string, LocationData>>(new Map())
const selectedLocation = ref<string | null>(null)
const loading = ref(false)
const error = ref<string | null>(null)

export function useLocations() {
  const locationNames = computed(() =>
    Array.from(allLocations.value.keys()).sort()
  )

  const selectedData = computed(() => {
    const name = selectedLocation.value
    if (!name) return null
    return allLocations.value.get(name) ?? null
  })

  const totalMatchSum = computed(() => {
    let sum = 0
    allLocations.value.forEach(l => (sum += l.totalMatch))
    return sum
  })

  const totalWordsSum = computed(() => {
    let sum = 0
    allLocations.value.forEach(l => (sum += l.totalWords))
    return sum
  })

  const locationCount = computed(() => allLocations.value.size)

  const locationsArray = computed(() =>
    Array.from(allLocations.value.values())
  )

  async function fetchLocations() {
    loading.value = true
    error.value = null
    try {
      const res = await fetch('/api/locations')
      if (!res.ok) throw new Error(`HTTP ${res.status}`)
      const data: LocationData[] = await res.json()
      allLocations.value.clear()
      for (const loc of data) {
        const records: ExcelRecord[] = loc.records.map(r => ({
          切词数: r.切词数 || 0,
          匹配词数: r.匹配词数 || 1,
          文本字数: r.文本字数 || 0,
          摘要: r.摘要 || '',
        }))
        allLocations.value.set(loc.name, {
          name: loc.name,
          totalMatch: loc.totalMatch,
          totalWords: loc.totalWords,
          records,
          keywords: loc.keywords || [],
        })
      }
      // 
      const def = allLocations.value.has('')
        ? ''
        : Array.from(allLocations.value.keys())[0]
      if (def) selectedLocation.value = def
    } catch (e: any) {
      error.value = e.message || ''
    } finally {
      loading.value = false
    }
  }

  function selectLocation(name: string) {
    selectedLocation.value = name
  }

  return {
    allLocations,
    selectedLocation,
    selectedData,
    loading,
    error,
    locationNames,
    totalMatchSum,
    totalWordsSum,
    locationCount,
    locationsArray,
    fetchLocations,
    selectLocation,
  }
}

// ==================== Literature ====================
const chapters = ref<Chapter[]>([])
const searchQuery = ref('')
const litLoading = ref(false)

export function useLiterature() {
  const queryWords = computed(() =>
    searchQuery.value.trim().split(/\s+/).filter(w => w.length > 0)
  )

  async function fetchLiterature() {
    litLoading.value = true
    try {
      const res = await fetch('/api/literature')
      if (!res.ok) throw new Error('')
      chapters.value = await res.json()
    } catch {
      chapters.value = []
    } finally {
      litLoading.value = false
    }
  }

  function setSearchQuery(q: string) {
    searchQuery.value = q
  }

  function clearSearch() {
    searchQuery.value = ''
  }

  return {
    chapters,
    searchQuery,
    queryWords,
    litLoading,
    fetchLiterature,
    setSearchQuery,
    clearSearch,
  }
}
