<script setup lang="ts">
import { computed } from 'vue'
import { useMapStore } from '@/stores/map'
import { useDynastyCompareStore } from '@/stores/dynastyCompare'
import { useSourceTreeStore } from '@/stores/sourceTree'
import { useWorkspaceStore } from '@/stores/workspace'
import landscapeInfoData from '@/data/landscape-info.json'
import locationLiteraryData from '@/data/location-literary-data.json'

const mapStore = useMapStore()
const dynastyCompareStore = useDynastyCompareStore()
const sourceTreeStore = useSourceTreeStore()
const workspaceStore = useWorkspaceStore()

const emit = defineEmits<{
  close: []
}>()

const dynasties = [
  { key: 'all', label: '全部' },
  { key: '唐', label: '唐 (618-907)' },
  { key: '宋', label: '宋 (960-1279)' },
  { key: '明', label: '明 (1368-1644)' },
  { key: '清', label: '清 (1644-1911)' },
]

// P2-06: Compute location counts per dynasty
function locationCountForDynasty(dynasty: string): number {
  if (dynasty === 'all') {
    return (landscapeInfoData as { name: string }[]).length
  }
  let count = 0
  const landscapeInfo = landscapeInfoData as { name: string }[]
  const staticLitData = locationLiteraryData as Record<string, { totalMentions: Record<string, number> }>
  for (const loc of landscapeInfo) {
    const card = mapStore.traceabilityCardCache[loc.name]
    if (card && Object.keys(card.totalMentions).length > 0) {
      if (card.totalMentions[dynasty] && card.totalMentions[dynasty] > 0) {
        count++
      }
    } else {
      // Fallback to static data
      const staticData = staticLitData[loc.name]
      if (staticData && staticData.totalMentions[dynasty] && staticData.totalMentions[dynasty] > 0) {
        count++
      }
    }
  }
  return count
}

const showCompareEntry = computed(() => mapStore.hasExactlyTwoDynastiesSelected)

const selectedDynasties = computed(() => mapStore.selectedDynastyArray)

function isDynastyActive(key: string): boolean {
  return key === 'all' ? mapStore.timeLayer.has('all') : mapStore.timeLayer.has(key)
}

// P2-07: Complete openDynastyCompare — find chapters and open tab
function openDynastyCompare() {
  const dyns = mapStore.selectedDynastyArray
  if (dyns.length !== 2) return

  const leftCh = sourceTreeStore.chapters.find(c => c.dynasty === dyns[0])
  const rightCh = sourceTreeStore.chapters.find(c => c.dynasty === dyns[1])

  if (leftCh && rightCh) {
    const title = `${leftCh.dynasty} vs ${rightCh.dynasty}`
    workspaceStore.openTab('dynasty-compare', title, {
      config: {
        left: { chapterNumber: leftCh.number, chapterTitle: leftCh.title, dynasty: leftCh.dynasty },
        right: { chapterNumber: rightCh.number, chapterTitle: rightCh.title, dynasty: rightCh.dynasty },
        syncScroll: true,
        alignParagraphs: false,
      },
    })
    emit('close')
  } else {
    // Fallback: just set pending and let DynastyCompareView handle it
    dynastyCompareStore.setPendingCompareDynasties({ left: dyns[0], right: dyns[1] })
  }
}
</script>

<template>
  <div class="layer-panel">
    <div class="lp-header">
      <span class="lp-title">图层管理</span>
      <button class="lp-close" @click="emit('close')">&times;</button>
    </div>

    <div class="lp-body">
      <!-- Base map selection -->
      <div class="lp-section">
        <div class="lp-section-title">底图选择</div>
        <div class="lp-options">
          <label class="lp-radio">
            <input type="radio" :checked="mapStore.tileLayer === 'amap'" @change="mapStore.tileLayer = 'amap'" /> 高德地图
          </label>
          <label class="lp-radio lp-disabled">
            <input type="radio" disabled /> 卫星图 (待开放)
          </label>
        </div>
      </div>

      <!-- Time layer multi-select (checkbox, max 2 per design doc B.5.2) -->
      <div class="lp-section">
        <div class="lp-section-title">时间图层 (多选, 最多2个)</div>
        <div class="lp-options">
          <label
            v-for="d in dynasties"
            :key="d.key"
            class="lp-checkbox"
            :class="{
              'lp-active': isDynastyActive(d.key),
              'lp-disabled': d.key !== 'all' && selectedDynasties.length >= 2 && !isDynastyActive(d.key)
            }"
          >
            <input
              type="checkbox"
              :checked="isDynastyActive(d.key)"
              :disabled="d.key !== 'all' && selectedDynasties.length >= 2 && !isDynastyActive(d.key)"
              @change="d.key === 'all' ? mapStore.setTimeLayer('all') : mapStore.toggleTimeLayer(d.key)"
            />
            {{ d.label }}
            <span class="lp-dynasty-count">{{ locationCountForDynasty(d.key) }}</span>
          </label>
        </div>
      </div>

      <!-- Dynasty compare entry (shown when exactly 2 dynasties selected) -->
      <div v-if="showCompareEntry" class="lp-section lp-compare-entry">
        <div class="lp-compare-label">朝代对比: {{ selectedDynasties.join(' + ') }}</div>
        <button class="lp-compare-btn" @click="openDynastyCompare">
          发起跨朝代对比
        </button>
      </div>

      <!-- Reset button -->
      <div class="lp-section">
        <button class="lp-reset-btn" @click="mapStore.resetLayers()">重置图层</button>
      </div>
    </div>
  </div>
</template>

<style scoped>
.layer-panel {
  position: absolute;
  top: 10px;
  left: 10px;
  z-index: 1000;
  background: var(--color-bg-secondary, #F7F8FA);
  border: 1px solid var(--color-border, #E5E7EB);
  border-radius: var(--radius, 8px);
  box-shadow: var(--shadow, 0 4px 16px rgba(0, 0, 0, 0.1));
  width: 280px;
  max-height: 80vh;
  display: flex;
  flex-direction: column;
}

.lp-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 8px 12px;
  border-bottom: 1px solid var(--color-border, #E5E7EB);
  flex-shrink: 0;
}

.lp-title {
  font-size: 0.78rem;
  color: var(--color-text-primary, #1A1A1A);
  font-weight: 500;
}

.lp-close {
  background: none;
  border: none;
  font-size: 1rem;
  color: var(--color-text-secondary, #4B5563);
  cursor: pointer;
  transition: color var(--dur-fast, 120ms) var(--ease, cubic-bezier(.22,.61,.36,1));
}

.lp-close:hover {
  color: var(--color-text-primary, #1A1A1A);
}

.lp-body {
  flex: 1;
  overflow-y: auto;
  padding: 8px 12px;
}

.lp-section {
  margin-bottom: 10px;
}

.lp-section-title {
  font-size: 0.68rem;
  color: var(--color-text-secondary, #4B5563);
  margin-bottom: 4px;
}

.lp-options {
  display: flex;
  flex-direction: column;
  gap: 3px;
}

.lp-radio, .lp-checkbox {
  font-size: 0.72rem;
  color: var(--color-text-primary, #1A1A1A);
  cursor: pointer;
  display: flex;
  align-items: center;
  gap: 4px;
  padding: 2px 6px;
  border-radius: var(--radius-sm, 4px);
  border: 1px solid transparent;
  transition: background var(--dur-fast, 120ms) var(--ease, cubic-bezier(.22,.61,.36,1)),
              border-color var(--dur-fast, 120ms) var(--ease, cubic-bezier(.22,.61,.36,1));
}

.lp-radio:hover:not(.lp-disabled),
.lp-checkbox:hover:not(.lp-disabled) {
  background: var(--color-bg-hover, #F2F4F7);
  border-color: var(--color-border, #E5E7EB);
}

.lp-checkbox.lp-active {
  background: var(--color-accent-light, #EBF4FC);
  border-color: var(--color-accent, #5B9BD5);
}

.lp-disabled {
  opacity: 0.5;
  cursor: not-allowed;
}

.lp-dynasty-count {
  font-size: 0.62rem;
  color: var(--color-text-tertiary, #6B7280);
  margin-left: auto;
}

.lp-compare-entry {
  padding: 8px;
  background: var(--color-accent-light, #EBF4FC);
  border: 1px dashed var(--color-accent, #5B9BD5);
  border-radius: var(--radius-sm, 4px);
  text-align: center;
}

.lp-compare-label {
  font-size: 0.65rem;
  color: var(--color-text-secondary, #4B5563);
  margin-bottom: 6px;
}

.lp-compare-btn {
  padding: 5px 12px;
  font-size: 0.72rem;
  background: var(--color-accent, #5B9BD5);
  color: #FFFFFF;
  border: 1px solid var(--color-accent, #5B9BD5);
  border-radius: var(--radius-sm, 4px);
  cursor: pointer;
  transition: background var(--dur-fast, 120ms) var(--ease, cubic-bezier(.22,.61,.36,1)),
              border-color var(--dur-fast, 120ms) var(--ease, cubic-bezier(.22,.61,.36,1));
}

.lp-compare-btn:hover {
  background: var(--color-accent-hover, #4A8BC5);
  border-color: var(--color-accent-hover, #4A8BC5);
}

.lp-reset-btn {
  width: 100%;
  padding: 5px;
  font-size: 0.7rem;
  background: var(--color-bg-tertiary, #F0F2F5);
  color: var(--color-text-secondary, #4B5563);
  border: 1px solid var(--color-border, #E5E7EB);
  border-radius: var(--radius-sm, 4px);
  cursor: pointer;
  transition: background var(--dur-fast, 120ms) var(--ease, cubic-bezier(.22,.61,.36,1)),
              color var(--dur-fast, 120ms) var(--ease, cubic-bezier(.22,.61,.36,1)),
              border-color var(--dur-fast, 120ms) var(--ease, cubic-bezier(.22,.61,.36,1));
}

.lp-reset-btn:hover {
  background: var(--color-bg-primary, #FFFFFF);
  color: var(--color-text-primary, #1A1A1A);
  border-color: var(--color-border-dark, #D1D5DB);
}
</style>
