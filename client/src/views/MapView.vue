<script setup lang="ts">
import { ref, computed, onMounted, onUnmounted, watch, nextTick } from 'vue'
import L from 'leaflet'

// Leaflet's Tooltip._animateZoom and Tooltip._updatePosition lack null-guards
// for this._map. Under rare race conditions (rapid zoom + marker removal), a
// tooltip can receive animation events after its _map has been cleared.
// This patch mirrors the safety that Leaflet's own Marker.update() and
// DivOverlay._updatePosition already implement.
// We try both Tooltip and Popup (both extend DivOverlay and override these
// methods without guards), plus the base DivOverlay for any other subclass.
(function installDivOverlayNullGuards() {
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const targets: any[] = []
  if ((L as any).Tooltip) targets.push((L as any).Tooltip.prototype)
  if ((L as any).Popup) targets.push((L as any).Popup.prototype)
  if ((L as any).DivOverlay) targets.push((L as any).DivOverlay.prototype)

  for (const proto of targets) {
    if (proto._animateZoom) {
      const _orig = proto._animateZoom
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      proto._animateZoom = function (e: any) {
        if (!this._map) return
        return _orig.call(this, e)
      }
    }
    if (proto._updatePosition) {
      const _orig = proto._updatePosition
      proto._updatePosition = function () {
        if (!this._map) return
        return _orig.call(this)
      }
    }
  }
  if (targets.length === 0) {
    console.warn('[MapView] Failed to install DivOverlay null-guards: no targets found')
  }
})()

import { coords, panoramaUrls, storySegments } from '@/utils/coordinates'
import { useMapStore, type MapHighlightItem } from '@/stores/map'
import { useWorkspaceStore } from '@/stores/workspace'
import { useAgentStore } from '@/stores/agent'
import { useAnnotationStore } from '@/stores/annotations'
import { useResearchNotesStore } from '@/stores/researchNotes'
import LayerPanel from '@/components/map/LayerPanel.vue'
import TraceabilityCard from '@/components/map/TraceabilityCard.vue'

interface RawLocation {
  name: string
  totalMatch: number
  totalWords: number
  records: unknown[]
  keywords: unknown[]
}

const mapContainer = ref<HTMLElement | null>(null)
// eslint-disable-next-line @typescript-eslint/no-explicit-any
const mapInstance = ref<any>(null)
const locations = ref<RawLocation[]>([])
const errorMessage = ref<string | null>(null)
const loading = ref(true)

const mapStore = useMapStore()
const workspaceStore = useWorkspaceStore()
const agentStore = useAgentStore()
const researchNotesStore = useResearchNotesStore()
const annotationStore = useAnnotationStore()

// P0-06: Dynamic card position from marker screen coordinates
const cardPosition = ref<{ x: number; y: number }>({ x: 170, y: 200 })

const TILE_URL = 'https://webrd0{s}.is.autonavi.com/appmaptile?lang=zh_cn&size=1&scale=1&style=7&x={x}&y={y}&z={z}'

function containerHasSize(el: HTMLElement): boolean {
  return el.clientWidth > 0 && el.clientHeight > 0
}

// eslint-disable-next-line @typescript-eslint/no-explicit-any
function initMap(): any {
  const el = mapContainer.value
  if (!el || !containerHasSize(el)) return null
  if (mapInstance.value) return mapInstance.value

  const map = L.map(el, {
    attributionControl: false,
    zoomControl: false,
  })

  L.control.zoom({ position: 'bottomright' }).addTo(map)
  L.tileLayer(TILE_URL, {
    subdomains: ['1', '2', '3', '4'],
    attribution: '',
  }).addTo(map)
  map.setView([30.5, 118.0], 6.5)

  // Draw story segment polylines
  storySegments.forEach(([a, b]) => {
    const p1 = coords[a]
    const p2 = coords[b]
    if (p1 && p2) {
      L.polyline([p1, p2], {
        color: '#A8C8D8', weight: 1.5, opacity: 0.45, dashArray: '5,6',
      }).addTo(map)
    }
  })

  // Track zoom AND move state to prevent marker/tooltip removal during
  // ANY map animation. Removing markers mid-animation is the root cause of:
  //  - _latLngToNewLayerPoint crash (marker._map null during zoomanim)
  //  - latLngToLayerPoint crash (tooltip._map null during zoom/move event)
  //  - markers "flying around" (re-created markers get wrong positions)
  map.on('zoomstart movestart', () => { isZooming = true })
  map.on('zoomend moveend', () => {
    isZooming = false
    if (pendingFilterRender) safeRenderMarkers()
  })

  mapInstance.value = map
  return map
}

// P1-09: Render markers using mapStore.markerStyles and mapStore.filteredLocations.
// Uses INCREMENTAL updates: existing markers have their styles updated in-place
// instead of being destroyed and recreated. This eliminates the root cause of
// all zombie-tooltip crashes — markers are never removed during animations.
//
// eslint-disable-next-line @typescript-eslint/no-explicit-any
function renderMarkers(map: any) {
  if (locations.value.length === 0) return

  const styles = mapStore.markerStyles
  const visibleLocations = mapStore.filteredLocations
  const filteredLocs = locations.value.filter(loc => visibleLocations.has(loc.name))
  const newVisibleSet = new Set(filteredLocs.map(l => l.name))

  // Step 1: Remove markers for locations that are no longer visible.
  // We rely on Leaflet's built-in lifecycle: map.removeLayer(marker) calls
  // marker.onRemove → marker.fire('remove') → closeTooltip → tooltip.close()
  // → map.removeLayer(tooltip) → tooltip.onRemove detaches all listeners.
  // Explicit unbindTooltip is NOT called here — it removes the
  // remove:closeTooltip handler BEFORE fire('remove'), defeating the cleanup.
  for (const [name, entry] of markerEntries) {
    if (!newVisibleSet.has(name)) {
      try {
        map.removeLayer(entry.marker)
      } catch (_e) { /* */ }
      try {
        map.removeLayer(entry.cm)
      } catch (_e) { /* */ }
      markerEntries.delete(name)
    }
  }

  // Step 2: Update existing markers (styles only, no removal)
  for (const loc of filteredLocs) {
    const existing = markerEntries.get(loc.name)
    if (existing) {
      const style = styles[loc.name] || { fillColor: '#5B9BD5', radius: 12, opacity: 0.35, dashed: false, pulse: false }
      try {
        existing.cm.setStyle({
          radius: style.radius,
          fillColor: style.fillColor,
          color: style.dashed ? '#C8CED6' : style.fillColor,
          weight: style.dashed ? 1 : 1.2,
          fillOpacity: style.opacity,
          dashArray: style.dashed ? '4,4' : undefined,
        })
        if (style.pulse) {
          L.DomUtil.addClass((existing.cm as any)._path || (existing.cm as any)._container, 'mv-pulse-marker')
        } else {
          L.DomUtil.removeClass((existing.cm as any)._path || (existing.cm as any)._container, 'mv-pulse-marker')
        }
        existing.cm.setTooltipContent(`${loc.name} · ${loc.totalMatch}`)
        existing.marker.setTooltipContent(
          `<b>${loc.name}</b><br>匹配词数: ${loc.totalMatch}<br>文本字数: ${loc.totalWords}<br>记录数: ${loc.records.length}`
        )
        // Update marker icon color if needed (avoid full icon rebuild)
        const markerColor = loc.name.includes('塔') || loc.name.includes('寺') ? '#5B9BD5' : '#7EC4A0'
        const iconEl = (existing.marker as any)._icon as HTMLElement | null
        if (iconEl) {
          const div = iconEl.querySelector('div') as HTMLElement | null
          if (div) {
            div.style.color = markerColor
            div.style.borderColor = markerColor
          }
        }
      } catch (_e) { /* marker state stale — skip update */ }
    }
  }

  // Step 3: Create new markers for locations not yet tracked
  for (const loc of filteredLocs) {
    if (markerEntries.has(loc.name)) continue

    const ll = coords[loc.name]
    if (!ll) continue

    const style = styles[loc.name] || { fillColor: '#5B9BD5', radius: 12, opacity: 0.35, dashed: false, pulse: false }

    const cm = L.circleMarker(ll, {
      radius: style.radius,
      fillColor: style.fillColor,
      color: style.dashed ? '#C8CED6' : style.fillColor,
      weight: style.dashed ? 1 : 1.2,
      fillOpacity: style.opacity,
      dashArray: style.dashed ? '4,4' : undefined,
      className: style.pulse ? 'mv-pulse-marker' : '',
    }).addTo(map)
    cm.bindTooltip(`${loc.name} · ${loc.totalMatch}`, { sticky: true, direction: 'top' })
    cm.on('click', () => {
      mapStore.openTraceabilityCard(loc.name)
      const point = map.latLngToContainerPoint(ll)
      const mapEl = mapContainer.value
      if (mapEl && point) {
        cardPosition.value = {
          x: Math.min(point.x + 20, mapEl.clientWidth - 440),
          y: Math.max(point.y - 300, 10),
        }
      }
      mapStore.addToFlyToQueue(loc.name)
    })

    const markerColor = loc.name.includes('塔') || loc.name.includes('寺') ? '#5B9BD5' : '#7EC4A0'
    const icon = L.divIcon({
      html: `<div style="width:28px;height:28px;border-radius:50%;display:flex;align-items:center;justify-content:center;font-size:13px;font-weight:600;color:${markerColor};background:#FFFFFF;border:1px solid ${markerColor};box-shadow:0 1px 3px rgba(0,0,0,.06);">${loc.name.charAt(0)}</div>`,
      iconSize: [28, 28],
      popupAnchor: [0, -14],
      className: '',
    })
    const marker = L.marker(ll, { icon }).addTo(map)
    marker.bindTooltip(
      `<b>${loc.name}</b><br>匹配词数: ${loc.totalMatch}<br>文本字数: ${loc.totalWords}<br>记录数: ${loc.records.length}`,
      { direction: 'top' }
    )
    marker.on('click', () => {
      mapStore.openTraceabilityCard(loc.name)
      const point = map.latLngToContainerPoint(ll)
      const mapEl = mapContainer.value
      if (mapEl && point) {
        cardPosition.value = {
          x: Math.min(point.x + 20, mapEl.clientWidth - 440),
          y: Math.max(point.y - 300, 10),
        }
      }
      mapStore.addToFlyToQueue(loc.name)
    })

    markerEntries.set(loc.name, { cm, marker })
  }

  // Step 4: Fit bounds on initial render only
  if (!hasInitiallyFit) {
    hasInitiallyFit = true
    try {
      const bounds = filteredLocs
        .map(l => coords[l.name])
        .filter((c): c is [number, number] => c !== undefined)
      if (bounds.length > 0 && containerHasSize(mapContainer.value!)) {
        map.invalidateSize()
        map.fitBounds(L.latLngBounds(bounds), { padding: [20, 20], animate: false })
      }
    } catch { /* not ready — skip */ }
  }
}

// P1-11: Event handlers for TraceabilityCard
function handleTrackEvolution(locationName: string) {
  mapStore.setCompareLocationName(locationName)
  // Send Agent message to track evolution
  agentStore.sendMessage(`${locationName}`)
}

function handleAddToNotes(locationName: string) {
  researchNotesStore.addNote(locationName)
}

async function fetchLocations() {
  loading.value = true
  errorMessage.value = null
  try {
    const res = await fetch('/api/locations')
    if (!res.ok) throw new Error(`HTTP ${res.status}`)
    locations.value = await res.json()
    if (destroyed) return
    void nextTick(() => {
      if (destroyed) return
      const map = initMap()
      if (map) safeRenderMarkers()
    })
  } catch (err: unknown) {
    errorMessage.value = err instanceof Error ? err.message : ''
  } finally {
    loading.value = false
  }
}

// Retry init after container has size
let initTimer: ReturnType<typeof setInterval> | null = null

function tryInit() {
  const el = mapContainer.value
  if (!el || !containerHasSize(el)) return
  if (initTimer) {
    clearInterval(initTimer)
    initTimer = null
  }
  const map = initMap()
  if (map && locations.value.length > 0) safeRenderMarkers()
}

// Guard: prevent flyTo during renderMarkers to avoid Leaflet animation conflicts
let isRendering = false
let pendingFlyTo: MapHighlightItem[] = []

// Guard: prevent re-render during zoom animation (Leaflet bug: removing markers mid-zoom
// causes _latlng null pointer crashes and marker position desync)
let isZooming = false
let isFlying = false
let pendingFilterRender = false

// Track whether initial fitBounds has been applied — subsequent renders must not
// change the map view, or they will fight against user zoom/focus operations.
let hasInitiallyFit = false

// Incremental marker tracking: maps location name -> { circleMarker, marker }.
// renderMarkers updates styles in-place instead of destroying/recreating markers,
// which eliminates zombie-tooltip crashes entirely.
// eslint-disable-next-line @typescript-eslint/no-explicit-any
const markerEntries = new Map<string, { cm: any; marker: any }>()

// Debounce timer for safeRenderMarkers triggered by watchers (rapid store changes
// during traceability card fetch can cause hundreds of deep-watch firings)
let renderDebounceTimer: ReturnType<typeof setTimeout> | null = null

function safeRenderMarkers() {
  const map = mapInstance.value
  if (!map || locations.value.length === 0) return

  // Re-entrance guard: if renderMarkers is already on the call stack, defer.
  // Also defer if any map animation is in progress. Critical: Leaflet fires
  // zoomend SYNCHRONOUSLY during _animateZoom (before the CSS transition
  // completes), so isZooming can be false even though the CSS zoom animation
  // is still running. The internal _animatingZoom flag stays true until
  // _onZoomTransitionEnd fires 250ms later — we must check it explicitly.
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const m = map as any
  if (isRendering || isZooming || isFlying || m._animatingZoom) {
    pendingFilterRender = true
    return
  }

  isRendering = true
  try {
    renderMarkers(map)
  } finally {
    isRendering = false
  }

  // After rendering, check whether another render was requested (e.g. a
  // filter change arrived during renderMarkers). Use setTimeout to break
  // any recursion chain — fitBounds on first render can trigger zoomstart/
  // zoomend which call safeRenderMarkers synchronously.
  if (pendingFilterRender) {
    pendingFilterRender = false
    setTimeout(() => safeRenderMarkers(), 0)
  }

  // Process any flyTo requests that were queued during rendering
  if (pendingFlyTo.length > 0) {
    setTimeout(() => processFlyToQueue(), 100)
  }
}

// Respond to map_highlight SSE events and addToFlyToQueue calls
let flyToDebounceTimer: ReturnType<typeof setTimeout> | null = null
watch(
  () => mapStore.flyToQueue.length,
  () => {
    if (flyToDebounceTimer) {
      clearTimeout(flyToDebounceTimer)
      flyToDebounceTimer = null
    }
    flyToDebounceTimer = setTimeout(() => {
      // If already rendering or flying, drain the store queue into
      // pendingFlyTo so it is processed after the current operation.
      if (isRendering || isFlying) {
        while (mapStore.flyToQueue.length > 0) {
          const item = mapStore.processNextFlyTo()
          if (item) pendingFlyTo.push(item)
        }
        return
      }
      processFlyToQueue()
    }, 300)
  }
)

function processFlyToQueue() {
  const map = mapInstance.value
  if (!map || destroyed || isFlying) return
  const items = [...pendingFlyTo, ...mapStore.flyToQueue.splice(0)]
  pendingFlyTo = []
  if (items.length === 0) return
  // Only process the last item (most recent flyTo request)
  const item = items[items.length - 1]
  const ll = coords[item.name]
  if (ll && map) {
    try {
      isFlying = true
      // flyTo naturally cancels any in-progress animation — calling
      // map.stop() beforehand is unnecessary and triggers moveend/
      // zoomend events that can reset isFlying/isZooming prematurely
      // and cause concurrent renderMarkers calls.
      map.flyTo(ll, 14, { animate: true, duration: 1.2 })
      map.once('moveend', () => {
        // Use a short delay before clearing isFlying: Leaflet's internal
        // _onZoomTransitionEnd fires via setTimeout(..., 250) and may
        // still be running when moveend fires. If we allow renderMarkers
        // during that window, newly-added markers get wrong positions.
        setTimeout(() => {
          isFlying = false
          if (pendingFilterRender) {
            pendingFilterRender = false
            safeRenderMarkers()
          }
        }, 300)
      })
      mapStore.openTraceabilityCard(item.name)
    } catch (_e) {
      isFlying = false
      // Leaflet internal state inconsistent — ignore this flyTo
    }
  }
}

// P1-09: Re-render when map store filter state changes.
// Debounced because fetchTraceabilityCard modifies traceabilityCardCache which
// can cause filteredLocations and markerStyles computed properties to change
// multiple times in a single tick — without debouncing, renderMarkers would
// be called hundreds of times during card data loading, each time destroying
// and recreating all map markers.
watch(
  () => [mapStore.filteredLocations, mapStore.markerStyles],
  () => {
    if (renderDebounceTimer) {
      clearTimeout(renderDebounceTimer)
    }
    renderDebounceTimer = setTimeout(() => {
      renderDebounceTimer = null
      safeRenderMarkers()
    }, 150)
  },
  { deep: true }
)

// Resize handling
let destroyed = false

function onResize() {
  const map = mapInstance.value
  const el = mapContainer.value
  if (map && !destroyed && el && containerHasSize(el)) {
    try {
      map.invalidateSize({ animate: false } as any)
    } catch {
      // Leaflet  invalidateSize  _animateZoom 
    }
  }
}

function viewLocationSources() {
  const popup = mapStore.activePopup
  if (!popup) return
  workspaceStore.openTab('search-results', `: ${popup.name}`, {
    query: popup.name,
    totalMatches: popup.matchCount || 0,
    results: [],
  })
  mapStore.closePopup()
}

function trackLocationEvolution() {
  const popup = mapStore.activePopup
  if (!popup) return
  agentStore.sendMessage(`${popup.name}`)
  mapStore.closePopup()
}

function analyzeLocationImagery() {
  const popup = mapStore.activePopup
  if (!popup) return
  agentStore.addContextTag({
    id: `ctx_${Date.now()}`,
    label: `${popup.name}`,
    data: { location: popup.name },
    sourceType: 'map_location',
  })
  mapStore.closePopup()
}

onMounted(() => {
  destroyed = false
  initTimer = setInterval(tryInit, 200)
  window.addEventListener('resize', onResize)
  fetchLocations()
})

onUnmounted(() => {
  destroyed = true
  if (initTimer) {
    clearInterval(initTimer)
    initTimer = null
  }
  if (flyToDebounceTimer) {
    clearTimeout(flyToDebounceTimer)
    flyToDebounceTimer = null
  }
  if (renderDebounceTimer) {
    clearTimeout(renderDebounceTimer)
    renderDebounceTimer = null
  }
  pendingFlyTo = []
  markerEntries.clear()
  hasInitiallyFit = false
  window.removeEventListener('resize', onResize)
  const map = mapInstance.value
  if (map) {
    try {
      // Stop all animations and destroy the map. Leaflet's map.remove()
      // internally calls onRemove for every layer, which properly detaches
      // all zoomanim listeners. Calling layer.off() first is unnecessary
      // and can interfere with onRemove cleanup.
      map.stop()
      map.off()
      map.remove()
    } catch (_e) {
      // Map already partially destroyed
    }
  }
  mapInstance.value = null
})
</script>

<template>
  <div class="map-view">
    <!-- Loading state -->
    <div v-if="loading" class="mv-loading">...</div>

    <!-- Error state -->
    <div v-else-if="errorMessage" class="mv-error">
      <p>{{ errorMessage }}</p>
      <button @click="fetchLocations">重试加载</button>
    </div>

    <!-- Map container -->
    <div
      ref="mapContainer"
      class="mv-map-container"
    />

    <!-- P1-10: Layer toggle button with icon -->
    <button
      class="mv-layer-toggle"
      @click="mapStore.layerPanelExpanded = !mapStore.layerPanelExpanded"
      title=""
    ></button>

    <!-- LayerPanel -->
    <LayerPanel
      v-if="mapStore.layerPanelExpanded"
      @close="mapStore.layerPanelExpanded = false"
    />

    <!-- TraceabilityCard (P0-06: dynamic positioning) -->
    <TraceabilityCard
      v-if="mapStore.selectedLocation"
      :card="mapStore.traceabilityCardCache[mapStore.selectedLocation] || null"
      :loading="mapStore.isCardLoading"
      :position="cardPosition"
      @close="mapStore.closeTraceabilityCard()"
      @view-text="(chNum, paraIdx) => { /* handled in card */ }"
      @track-evolution="handleTrackEvolution"
      @add-to-notes="handleAddToNotes"
    />
  </div>
</template>

<style scoped>
.map-view {
  width: 100%;
  height: 100%;
  position: relative;
  overflow: hidden;
}

.mv-map-container {
  width: 100%;
  height: 100%;
}

.mv-loading,
.mv-error {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  height: 100%;
  color: var(--color-text-secondary);
  font-size: 0.85rem;
  gap: var(--space-3);
}

.mv-error {
  color: var(--color-error);
}

.mv-error button {
  padding: var(--space-1) var(--space-3);
  background: var(--color-bg-primary);
  color: var(--color-text-primary);
  border: 1px solid var(--color-border);
  border-radius: var(--radius);
  cursor: pointer;
  font-family: var(--font-sans);
  font-size: 0.8rem;
  transition:
    background-color var(--dur-fast) var(--ease),
    border-color var(--dur-fast) var(--ease),
    transform var(--dur) var(--ease),
    box-shadow var(--dur) var(--ease);
}

.mv-error button:hover {
  background: var(--color-bg-hover);
  border-color: var(--color-border-dark);
  transform: translateY(-1px);
  box-shadow: var(--shadow-sm);
}

.mv-layer-toggle {
  position: absolute;
  top: var(--space-3);
  left: var(--space-3);
  z-index: 999;
  width: 32px;
  height: 32px;
  background: var(--color-bg-primary);
  border: 1px solid var(--color-border);
  border-radius: var(--radius-sm);
  font-size: 0.85rem;
  cursor: pointer;
  display: flex;
  align-items: center;
  justify-content: center;
  box-shadow: var(--shadow);
  transition:
    background-color var(--dur-fast) var(--ease),
    border-color var(--dur-fast) var(--ease),
    box-shadow var(--dur) var(--ease),
    transform var(--dur) var(--ease);
}

.mv-layer-toggle:hover {
  background: var(--color-bg-hover);
  border-color: var(--color-border-dark);
  box-shadow: var(--shadow-md);
  transform: translateY(-1px);
}
</style>
