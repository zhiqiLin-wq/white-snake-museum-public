import { describe, it, expect, beforeEach } from 'vitest'
import { setActivePinia, createPinia } from 'pinia'
import { useMapStore } from '@/stores/map'

describe('useMapStore', () => {
  beforeEach(() => {
    setActivePinia(createPinia())
  })

  it('should add fly_to highlight and process queue', () => {
    const store = useMapStore()
    store.addHighlight({ name: '雷峰塔', lat: 30.23, lng: 120.14, action: 'fly_to' })
    expect(store.flyToQueue.length).toBe(1)

    const item = store.processNextFlyTo()
    expect(item).not.toBeNull()
    expect(item!.name).toBe('雷峰塔')
    expect(store.isAnimating).toBe(true)

    const next = store.markAnimationComplete()
    expect(next).toBeNull()
    expect(store.isAnimating).toBe(false)
  })

  it('should add highlight markers', () => {
    const store = useMapStore()
    store.addHighlight({ name: '金山寺', lat: 32.21, lng: 119.42, action: 'highlight' })
    expect(store.highlightedLocations.length).toBe(1)
  })

  it('should manage popups', () => {
    const store = useMapStore()
    store.openPopup({ name: '雷峰塔', lat: 30.23, lng: 120.14 })
    expect(store.activePopup).not.toBeNull()
    expect(store.activePopup!.name).toBe('雷峰塔')
    store.closePopup()
    expect(store.activePopup).toBeNull()
  })

  it('should respect max queue size', () => {
    const store = useMapStore()
    for (let i = 0; i < 15; i++) {
      store.addHighlight({ name: `Location ${i}`, lat: 30, lng: 120, action: 'fly_to' })
    }
    expect(store.flyToQueue.length).toBeLessThanOrEqual(store.maxQueueSize)
  })
})
