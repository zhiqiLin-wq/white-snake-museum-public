import { defineStore } from 'pinia'
import { ref, computed } from 'vue'
import type { LocationTraceabilityCard, MapMarkerStyle } from '@/types/map'
import landscapeInfoData from '@/data/landscape-info.json'
import locationLiteraryData from '@/data/location-literary-data.json'
import { coords } from '@/utils/coordinates'

export interface MapHighlightItem {
  name: string
  lat: number
  lng: number
  action: 'fly_to' | 'highlight' | 'add_marker'
  era?: string
}

export interface MapPopupData {
  name: string
  lat: number
  lng: number
  matchCount?: number
  eras?: string[]
  keywords?: string[]
  summary?: string
}

export const useMapStore = defineStore('map', () => {
  // ===== Legacy state =====
  const highlightedLocations = ref<MapHighlightItem[]>([])
  const activePopup = ref<MapPopupData | null>(null)
  const flyToQueue = ref<MapHighlightItem[]>([])
  const isAnimating = ref(false)
  const tileLayer = ref<string>('amap')
  const maxQueueSize = 10
  // U-18: Compare summary for traceability card in compare mode
  const compareSummary = ref<{ overview: string; keyFindings: string[] } | null>(null)

  // ===== New state =====
  const selectedLocation = ref<string | null>(null)
  const traceabilityCardCache = ref<Record<string, LocationTraceabilityCard>>({})
  const isCardLoading = ref(false)
  const layerPanelExpanded = ref(false)
  const timeLayer = ref<Set<string>>(new Set(['all']))
  const compareDynasties = ref<{ left: string; right: string } | null>(null)
  const compareLocationName = ref<string | null>(null)

  // ===== Getters =====
  const filteredLocations = computed(() => {
    const names = new Set<string>()
    const landscapeInfo = landscapeInfoData as { name: string }[]

    // Resolve effective dynasty filter set
    const effectiveDynasties: Set<string> = (!timeLayer.value.has('all') && timeLayer.value.size > 0)
      ? new Set(timeLayer.value)
      : new Set(['唐', '宋', '元', '明', '清'])

    for (const loc of landscapeInfo) {
      const card = traceabilityCardCache.value[loc.name]
      if (card && card.literaryRecords.length > 0) {
        const locDynasties = new Set(card.literaryRecords.map(r => r.dynasty))
        if ([...effectiveDynasties].some(d => locDynasties.has(d))) {
          names.add(loc.name)
        }
      } else {
        // No card data yet — check static data
        const staticData = (locationLiteraryData as Record<string, { literaryRecords: { dynasty: string }[] }>)[loc.name]
        if (staticData && staticData.literaryRecords.length > 0) {
          const locDynasties = new Set(staticData.literaryRecords.map(r => r.dynasty))
          if ([...effectiveDynasties].some(d => locDynasties.has(d))) {
            names.add(loc.name)
          }
        } else {
          // Fallback: show all when no data available
          names.add(loc.name)
        }
      }
    }
    return names
  })

  const markerStyles = computed<Record<string, MapMarkerStyle>>(() => {
    const styles: Record<string, MapMarkerStyle> = {}
    const landscapeInfo = landscapeInfoData as { name: string }[]
    const DEFAULT_STYLE: MapMarkerStyle = { fillColor: '#5B9BD5', radius: 12, opacity: 0.32, dashed: false, pulse: false }

    // Resolve effective dynasties for style computation
    const effectiveDynasties: string[] = (!timeLayer.value.has('all') && timeLayer.value.size > 0)
      ? Array.from(timeLayer.value)
      : ['唐', '宋', '元', '明', '清']

    for (const loc of landscapeInfo) {
      let style = { ...DEFAULT_STYLE }

      // Single dynasty mode: differentiate new vs existing vs disappeared locations
      if (effectiveDynasties.length === 1) {
        const singleDynasty = effectiveDynasties[0]
        let locDynasties: Set<string> | null = null
        const card = traceabilityCardCache.value[loc.name]
        if (card && card.literaryRecords.length > 0) {
          locDynasties = new Set(card.literaryRecords.map(r => r.dynasty))
        } else {
          const staticData = (locationLiteraryData as Record<string, { literaryRecords: { dynasty: string }[] }>)[loc.name]
          if (staticData && staticData.literaryRecords.length > 0) {
            locDynasties = new Set(staticData.literaryRecords.map(r => r.dynasty))
          }
        }

        if (locDynasties) {
          const hasMention = locDynasties.has(singleDynasty)
          const allDynasties = ['唐', '宋', '元', '明', '清']
          const dynIdx = allDynasties.indexOf(singleDynasty)
          const earlierDynasties = allDynasties.slice(0, dynIdx)
          const existedEarlier = earlierDynasties.some(d => locDynasties.has(d))

          if (!hasMention) {
            style = { fillColor: '#C8CED6', radius: 8, opacity: 0.2, dashed: true, pulse: false }
          } else if (!existedEarlier && dynIdx > 0) {
            style = { fillColor: '#5B9BD5', radius: 14, opacity: 0.4, dashed: false, pulse: true }
          }
        }
      }

      styles[loc.name] = style
    }
    return styles
  })

  const hasExactlyTwoDynastiesSelected = computed(() => {
    return timeLayer.value.size === 2 && !timeLayer.value.has('all')
  })

  const selectedDynastyArray = computed(() => {
    return Array.from(timeLayer.value).filter(d => d !== 'all')
  })

  // ===== Legacy actions =====
  function addHighlight(item: MapHighlightItem) {
    if (item.action === 'fly_to') {
      flyToQueue.value.push(item)
      if (flyToQueue.value.length > maxQueueSize) {
        flyToQueue.value.shift()
      }
    } else {
      highlightedLocations.value.push(item)
    }
  }

  function clearHighlights() {
    highlightedLocations.value = []
    flyToQueue.value = []
  }

  function processNextFlyTo(): MapHighlightItem | null {
    if (flyToQueue.value.length === 0) {
      isAnimating.value = false
      return null
    }
    isAnimating.value = true
    const item = flyToQueue.value.shift()!
    return item
  }

  function markAnimationComplete() {
    if (flyToQueue.value.length > 0) {
      const next = processNextFlyTo()
      return next
    }
    isAnimating.value = false
    return null
  }

  function openPopup(popup: MapPopupData) {
    activePopup.value = popup
  }

  function closePopup() {
    activePopup.value = null
  }

  // ===== v10 New actions =====

  /**  flyTo coords  */
  function addToFlyToQueue(locationName: string) {
    const ll = coords[locationName]
    if (!ll) {
      console.warn(`[mapStore] addToFlyToQueue:  "${locationName}" `)
      return
    }
    addHighlight({ name: locationName, lat: ll[0], lng: ll[1], action: 'fly_to' })
  }

  function openTraceabilityCard(locationName: string) {
    selectedLocation.value = locationName
    if (!traceabilityCardCache.value[locationName]) {
      fetchTraceabilityCard(locationName)
    }
  }

  function closeTraceabilityCard() {
    selectedLocation.value = null
  }

  function fetchTraceabilityCard(locationName: string) {
    if (traceabilityCardCache.value[locationName]) return

    isCardLoading.value = true
    const landscapeInfo = landscapeInfoData as {
      name: string; historicalName: string; currentName: string
      builtYear: number | null; destroyedYear: number | null; rebuiltYear: number | null
      historicalChanges: string; panoramaUrl: string
      poi: { address: string | null; openTime: string | null; ticketPrice: string | null } | null
      // also support legacy POI format during transition
    }[]
    const info = landscapeInfo.find(l => l.name === locationName)

    if (info) {
      // Normalize POI to flat format
      let poiData: { address: string | null; openTime: string | null; ticketPrice: string | null } | null = null
      if (info.poi) {
        if (Array.isArray(info.poi)) {
          // Legacy format: [{ label, value }] — convert to flat
          const poiObj: Record<string, string> = {}
          for (const item of info.poi as { label: string; value: string }[]) {
            if (item.label.includes('')) poiObj['address'] = item.value
            else if (item.label.includes('') || item.label.includes('')) poiObj['openTime'] = item.value
            else if (item.label.includes('') || item.label.includes('')) poiObj['ticketPrice'] = item.value
          }
          poiData = {
            address: poiObj['address'] || null,
            openTime: poiObj['openTime'] || null,
            ticketPrice: poiObj['ticketPrice'] || null,
          }
        } else {
          // New flat format
          const flat = info.poi as { address: string | null; openTime: string | null; ticketPrice: string | null }
          poiData = {
            address: flat.address || null,
            openTime: flat.openTime || null,
            ticketPrice: flat.ticketPrice || null,
          }
        }
      }

      // Load static literary data if available
      const staticLitData = (locationLiteraryData as Record<string, {
        literaryRecords: {
          dynasty: string; dynastyOrder: number
          chapterNumber: number; chapterTitle: string
          paragraphIndex: number; excerpt: string
          span: { startChar: number; endChar: number }
          descriptionStyle: string
        }[]
        totalMentions: Record<string, number>
        mutualConstruction?: { landscapeToText: string; textToLandscape: string } | null
        descriptionStyles?: Record<string, string>
      }>)[locationName]

      traceabilityCardCache.value[locationName] = {
        locationName,
        physicalLandscape: {
          name: info.name,
          historicalName: info.historicalName,
          currentName: info.currentName,
          builtYear: info.builtYear,
          destroyedYear: info.destroyedYear,
          rebuiltYear: info.rebuiltYear,
          historicalChanges: info.historicalChanges,
          panoramaUrl: info.panoramaUrl,
          poi: poiData,
        },
        literaryRecords: staticLitData?.literaryRecords || [],
        totalMentions: staticLitData?.totalMentions || {},
        mutualConstruction: staticLitData?.mutualConstruction || null,
        descriptionStyles: staticLitData?.descriptionStyles || {},
        loadedAt: Date.now(),
      }
    }

    isCardLoading.value = false
  }

  function cacheTraceabilityData(locationName: string, data: {
    literaryRecords?: Array<{
      dynasty: string; dynastyOrder: number
      chapterNumber: number; chapterTitle: string
      paragraphIndex: number; excerpt: string
      span: { startChar: number; endChar: number }
      descriptionStyle: string
    }>
    totalMentions?: Record<string, number>
  }) {
    const card = traceabilityCardCache.value[locationName]
    if (card) {
      if (data.literaryRecords) {
        card.literaryRecords = data.literaryRecords
      }
      if (data.totalMentions) {
        card.totalMentions = data.totalMentions
      }
    }
  }

  function cacheMutualConstruction(locationName: string, data: {
    mutualConstruction?: { landscapeToText: string; textToLandscape: string }
    descriptionStyles?: Record<string, string>
  }) {
    const card = traceabilityCardCache.value[locationName]
    if (card) {
      if (data.mutualConstruction) {
        card.mutualConstruction = data.mutualConstruction
      }
      if (data.descriptionStyles) {
        card.descriptionStyles = data.descriptionStyles
      }
    }
  }

  function setTimeLayer(dynasty: string) {
    if (dynasty === 'all') {
      timeLayer.value = new Set(['all'])
    } else {
      if (timeLayer.value.has('all')) {
        timeLayer.value = new Set([dynasty])
      } else {
        toggleTimeLayer(dynasty)
      }
    }
  }

  function toggleTimeLayer(dynasty: string) {
    if (dynasty === 'all') {
      timeLayer.value = new Set(['all'])
      return
    }
    const current = new Set(timeLayer.value)
    //  'all' 
    current.delete('all')
    if (current.has(dynasty)) {
      current.delete(dynasty)
    } else {
      // 2——UI LayerPanel2""
      current.add(dynasty)
    }
    if (current.size === 0) {
      current.add('all')
    }
    timeLayer.value = current
  }

  function resetLayers() {
    timeLayer.value = new Set(['all'])
  }

  function setCompareDynasties(dynasties: { left: string; right: string } | null) {
    compareDynasties.value = dynasties
    if (dynasties) {
      timeLayer.value = new Set([dynasties.left, dynasties.right])
    }
  }

  function setCompareLocationName(name: string | null) {
    compareLocationName.value = name
  }

  // U-18: Set agent compare summary for traceability card
  function setCompareSummary(summary: { overview: string; keyFindings: string[] }) {
    compareSummary.value = summary
  }

  function clearCompareSummary() {
    compareSummary.value = null
  }

  function clearCompareDynasties() {
    compareDynasties.value = null
    compareLocationName.value = null
    compareSummary.value = null
    //  timeLayer 
    timeLayer.value = new Set(['all'])
  }

  return {
    // Legacy
    highlightedLocations,
    activePopup,
    flyToQueue,
    isAnimating,
    tileLayer,
    maxQueueSize,
    addHighlight,
    clearHighlights,
    processNextFlyTo,
    markAnimationComplete,
    openPopup,
    closePopup,

    // New state
    selectedLocation,
    traceabilityCardCache,
    isCardLoading,
    layerPanelExpanded,
    timeLayer,
    compareDynasties,
    compareLocationName,

    // New getters
    filteredLocations,
    markerStyles,
    hasExactlyTwoDynastiesSelected,
    selectedDynastyArray,

    // New actions
    addToFlyToQueue,
    openTraceabilityCard,
    closeTraceabilityCard,
    fetchTraceabilityCard,
    cacheTraceabilityData,
    cacheMutualConstruction,
    setTimeLayer,
    toggleTimeLayer,
    resetLayers,
    compareSummary,
    setCompareDynasties,
    setCompareSummary,
    clearCompareSummary,
    setCompareLocationName,
    clearCompareDynasties,
  }
})
