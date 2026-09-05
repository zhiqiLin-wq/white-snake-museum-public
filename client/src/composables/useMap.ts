import { ref, type Ref } from 'vue'
import L from 'leaflet'
import { coords, panoramaUrls, storySegments } from '../utils/coordinates'
import type { LocationData } from '../types'

const TILE_URL = 'https://webrd0{s}.is.autonavi.com/appmaptile?lang=zh_cn&size=1&scale=1&style=7&x={x}&y={y}&z={z}'
const TILE_OPTS: L.TileLayerOptions = {
  subdomains: ['1', '2', '3', '4'],
  attribution: '',
}

// eslint-disable-next-line @typescript-eslint/no-explicit-any
function addTo(layer: any, map: any): any { layer.addTo(map); return layer }

const VIEW_CENTER: L.LatLngTuple = [30.5, 118.0]
const VIEW_ZOOM = 6.5

function containerHasSize(el: HTMLElement): boolean {
  return el.clientWidth > 0 && el.clientHeight > 0
}

export interface UseMapOptions {
  timeLayer?: Set<string>
  filteredLocations?: Set<string>
  markerStyles?: Record<string, { fillColor: string; radius: number; opacity: number; dashed: boolean; pulse: boolean }>
  onMarkerClick?: (locationName: string) => void
}

export function useMap(mapContainer: Ref<HTMLElement | null>) {
  const map = ref<L.Map | null>(null)
  const baseFontSize = ref(16)
  let _initRetries = 0
  const MAX_RETRIES = 10

  function initMap() {
    if (map.value) return
    const el = mapContainer.value
    if (!el || !containerHasSize(el)) return

    _initRetries = 0

    map.value = L.map(el, {
      attributionControl: false,
      zoomControl: false,
    })

    const m = map.value
    addTo(L.control.zoom({ position: 'bottomright' }), m)
    addTo(L.tileLayer(TILE_URL, TILE_OPTS), m)
    m.setView(VIEW_CENTER, VIEW_ZOOM)

    storySegments.forEach(([a, b]) => {
      const p1 = coords[a], p2 = coords[b]
      if (p1 && p2) {
        addTo(L.polyline([p1, p2], {
          color: '#A8C8D8', weight: 1.5, opacity: 0.45, dashArray: '5,6',
        }), m)
      }
    })
  }

  function isMapReady(): boolean {
    return map.value !== null
  }

  function updateMap(
    locations: LocationData[],
    fontSize: number,
    animate = true,
    options?: UseMapOptions
  ) {
    if (!isMapReady()) {
      if (_initRetries >= MAX_RETRIES) return
      _initRetries++
      initMap()
      const el = mapContainer.value
      if (!el || !containerHasSize(el)) return
      setTimeout(() => updateMap(locations, fontSize, animate, options), 300)
      return
    }

    const m = map.value!
    baseFontSize.value = fontSize

    // Clear old markers
    m.eachLayer(layer => {
      if (layer instanceof L.Marker || layer instanceof L.CircleMarker) m.removeLayer(layer)
    })

    if (!locations.length) return

    // P1-07: Apply filtering based on options
    let filteredLocations = locations
    const visibleSet = options?.filteredLocations
    const styles = options?.markerStyles || {}

    if (visibleSet && visibleSet.size > 0) {
      filteredLocations = locations.filter(loc => visibleSet.has(loc.name))
    } else if (options?.timeLayer && options.timeLayer.size > 0 && !options.timeLayer.has('all')) {
      // If no pre-filtered set, apply time layer filter
      filteredLocations = locations.filter(() => true) // pass-through when no cache
    }

    const locsToRender = filteredLocations
    if (!locsToRender.length) return

    const maxM = Math.max(...locsToRender.map(l => l.totalMatch), 1)
    const minM = Math.min(...locsToRender.map(l => l.totalMatch), 1)

    locsToRender.forEach(loc => {
      const ll = coords[loc.name]
      if (!ll) return

      const locStyle = styles[loc.name]
      const radius = locStyle ? locStyle.radius : (5 + ((loc.totalMatch - minM) / (maxM - minM)) * 22) * (fontSize / 16)
      const url = panoramaUrls[loc.name]

      const cm = addTo(L.circleMarker(ll, {
        radius: Math.max(radius, 4),
        fillColor: locStyle?.fillColor || '#5B9BD5',
        color: locStyle?.dashed ? '#C8CED6' : (locStyle?.fillColor || '#BFD4E8'),
        weight: locStyle?.dashed ? 1 : 1.2,
        fillOpacity: locStyle?.opacity ?? 0.35,
        dashArray: locStyle?.dashed ? '4,4' : undefined,
        className: locStyle?.pulse ? 'mv-pulse-marker' : '',
      }), m)
      cm.bindTooltip(`${loc.name} · ${loc.totalMatch}`, { sticky: true, direction: 'top' })
      // P1-08: Call openTraceabilityCard instead of opening panorama URL
      cm.on('click', () => {
        if (options?.onMarkerClick) {
          options.onMarkerClick(loc.name)
        } else if (url) {
          window.open(url, '_blank')
        }
      })

      const iconSize = Math.round(fontSize * 1.8)
      const markerColor = loc.name.includes('') || loc.name.includes('') ? '#5B9BD5' : '#7EC4A0'
      const icon = L.divIcon({
        html: `<div style="width:${iconSize}px;height:${iconSize}px;border-radius:50%;display:flex;align-items:center;justify-content:center;font-size:${fontSize - 2}px;font-weight:600;color:${markerColor};background:#FFFFFF;border:1px solid ${markerColor};box-shadow:0 1px 3px rgba(0,0,0,.06);">${loc.name.charAt(0)}</div>`,
        iconSize: [iconSize, iconSize], popupAnchor: [0, -iconSize / 2], className: '',
      })
      const marker = addTo(L.marker(ll, { icon }), m)
      marker.bindTooltip(
        `<b>${loc.name}</b><br>匹配词数: ${loc.totalMatch}<br>文本字数: ${loc.totalWords}<br>记录数: ${loc.records.length}`,
        { direction: 'top' }
      )
      marker.on('click', () => {
        if (options?.onMarkerClick) {
          options.onMarkerClick(loc.name)
        } else if (url) {
          window.open(url, '_blank')
        }
      })
    })

    // Draw story segment polylines
    storySegments.forEach(([a, b]) => {
      const p1 = coords[a], p2 = coords[b]
      if (p1 && p2) {
        addTo(L.polyline([p1, p2], {
          color: '#A8C8D8', weight: 1.5, opacity: 0.45, dashArray: '5,6',
        }), m)
      }
    })

    // Fit bounds using locsToRender
    const bounds = locsToRender.map(l => coords[l.name]).filter((c): c is [number, number] => c !== undefined)
    if (bounds.length) {
      m.whenReady(() => {
        const el = mapContainer.value
        if (!el || !containerHasSize(el)) return
        m.invalidateSize()
        const padLeft = Math.round(window.innerWidth * 0.4)
        m.setView(L.latLngBounds(bounds).getCenter(), VIEW_ZOOM, { animate: false })
        m.panBy([-padLeft / 3, 0], { animate: false })
      })
    }
  }

  return { map, initMap, updateMap }
}
