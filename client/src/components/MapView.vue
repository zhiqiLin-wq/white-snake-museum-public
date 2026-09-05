<script setup lang="ts">
import { ref, watch, onMounted, onUnmounted, nextTick, inject, type Ref } from 'vue'
import type { LocationData } from '../types'
import { useMap } from '../composables/useMap'
import { computeBaseFontSize, coords } from '../utils/coordinates'

const props = defineProps<{
  locationsArray: LocationData[]
  selectedLocation: string | null
}>()

const mapEl = ref<HTMLDivElement | null>(null)
const { map, initMap, updateMap } = useMap(mapEl)

// U48 —  evolution_map_highlight 
const highlightLocation = inject<Ref<{ name: string; era: string } | null>>('evolutionMapHighlight', ref(null))

let resizeTimer: ReturnType<typeof setTimeout> | null = null
let resizeHandlerRef: (() => void) | null = null
let firstLoad = true

async function refreshMap(animate = false) {
  // fast fail: 0
  const el = mapEl.value
  if (!el || el.clientWidth === 0 || el.clientHeight === 0) return

  const fontSize = computeBaseFontSize()
  document.documentElement.style.fontSize = fontSize + 'px'
  updateMap(props.locationsArray, fontSize, animate)
}

//  → 
watch(() => props.locationsArray, async () => {
  if (props.locationsArray.length > 0) {
    await nextTick()
    refreshMap(firstLoad ? false : true)
  }
}, { deep: true })

//  → 
watch(() => props.selectedLocation, (name) => {
  if (!name || !map.value) return
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  if (!(map.value as any)._loaded) return  // fast fail: map 
  const coord = coords[name]
  if (coord) {
    map.value.flyTo(coord, 11, { animate: true, duration: 1.2 })
  }
})

//  U48: evolution_map_highlight → MapView 
watch(() => highlightLocation.value, (hl) => {
  if (!hl || !map.value) return
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  if (!(map.value as any)._loaded) return  // fast fail
  const coord = coords[hl.name]
  if (coord) {
    map.value.flyTo(coord, 13, { animate: true, duration: 1.0 })
  }
}, { deep: true })

onMounted(async () => {
  await nextTick()
  initMap()
  //  map 
  const tryRefresh = () => {
    if (map.value) {
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      if ((map.value as any)._loaded) {
        refreshMap(false)
        firstLoad = false
        return
      }
    }
    // map 
    setTimeout(tryRefresh, 300)
  }
  setTimeout(tryRefresh, 200)

  // resize  fitBounds
  resizeTimer = null
  resizeHandlerRef = () => {
    if (resizeTimer) clearTimeout(resizeTimer)
    resizeTimer = setTimeout(() => refreshMap(false), 200)
  }
  window.addEventListener('resize', resizeHandlerRef)
})

onUnmounted(() => {
  if (resizeHandlerRef) window.removeEventListener('resize', resizeHandlerRef)
  if (resizeTimer) clearTimeout(resizeTimer)
  map.value?.remove()
})
</script>

<template>
  <div ref="mapEl" id="map" style="width:100%;height:100%;min-height:100vh;"></div>
</template>
