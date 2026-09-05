<script setup lang="ts">
import { computed } from 'vue'
import { useMapStore } from '@/stores/map'
import { useSourceTreeStore } from '@/stores/sourceTree'
import { useWorkspaceStore } from '@/stores/workspace'
import type { LocationTraceabilityCard } from '@/types/map'

const props = defineProps<{
  card: LocationTraceabilityCard | null
  loading: boolean
  position: { x: number; y: number } | null
}>()

const emit = defineEmits<{
  close: []
  'view-text': [chapterNumber: number, paragraphIndex: number]
  'track-evolution': [locationName: string]
  'add-to-notes': [locationName: string]
}>()

const mapStore = useMapStore()
const sourceTreeStore = useSourceTreeStore()
const workspaceStore = useWorkspaceStore()

const compareDynasties = computed(() => mapStore.compareDynasties)
const isCompareMode = computed(() => compareDynasties.value !== null)

// P2-10: Dynasty mention overview
const dynastyMentionsSummary = computed(() => {
  if (!props.card) return ''
  const mentions = props.card.totalMentions
  return ['唐', '宋', '元', '明', '清']
    .map(d => `${d}${mentions[d] || 0}`)
    .join(' · ')
})

function viewOriginalText(record: { chapterNumber: number; paragraphIndex: number; chapterTitle?: string }) {
  sourceTreeStore.setNavigationTarget({
    chapterNumber: record.chapterNumber,
    paragraphIndex: record.paragraphIndex,
  })
  workspaceStore.openTab('text-reader', `${props.card?.locationName || ''} - ${record.chapterTitle}`, {
    chapterNumber: record.chapterNumber,
    paragraphIndex: record.paragraphIndex,
  })
}

function isDynastyCompared(dynasty: string): boolean {
  if (!compareDynasties.value) return false
  return compareDynasties.value.left === dynasty || compareDynasties.value.right === dynasty
}

// P2-08: Panorama thumbnail URL
const panoramaThumbUrl = computed(() => {
  if (!props.card?.physicalLandscape?.panoramaUrl) return ''
  return props.card.physicalLandscape.panoramaUrl
})

function openPanorama() {
  const url = panoramaThumbUrl.value
  if (url) window.open(url, '_blank')
}

// P2-09: Compare mode title
const cardTitle = computed(() => {
  if (!props.card) return ''
  if (isCompareMode.value && compareDynasties.value) {
    return `${props.card.locationName} · ${compareDynasties.value.left} vs ${compareDynasties.value.right} 对比`
  }
  return props.card.locationName
})
</script>

<template>
  <div v-if="card" class="traceability-card" :style="position ? { left: position.x + 'px', top: position.y + 'px' } : {}">
    <div class="tc-header">
      <h4>{{ cardTitle }}</h4>
      <button class="tc-close" @click="emit('close')">&times;</button>
    </div>

    <div class="tc-body">
      <!-- Loading state -->
      <div v-if="loading" class="tc-loading">加载中...</div>

      <template v-else>
        <!-- P2-10: Dynasty mention overview -->
        <div class="tc-dynasty-mentions">
          朝代提及: {{ dynastyMentionsSummary }}
        </div>

        <!-- Physical landscape section -->
        <div class="tc-section">
          <div class="tc-section-title">物理景观</div>
          <div class="tc-info-grid">
            <div class="tc-info-item">
              <span class="tc-info-label">古名</span>
              <span class="tc-info-value">{{ card.physicalLandscape.historicalName }}</span>
            </div>
            <div class="tc-info-item">
              <span class="tc-info-label">现名</span>
              <span class="tc-info-value">{{ card.physicalLandscape.currentName }}</span>
            </div>
            <div v-if="card.physicalLandscape.builtYear" class="tc-info-item">
              <span class="tc-info-label">始建</span>
              <span class="tc-info-value">{{ card.physicalLandscape.builtYear }}</span>
            </div>
            <div v-if="card.physicalLandscape.destroyedYear" class="tc-info-item">
              <span class="tc-info-label">毁于</span>
              <span class="tc-info-value">{{ card.physicalLandscape.destroyedYear }}</span>
            </div>
          </div>
          <div class="tc-changes">{{ card.physicalLandscape.historicalChanges }}</div>

          <!-- P2-08: Panorama thumbnail -->
          <div v-if="panoramaThumbUrl" class="tc-panorama">
            <img
              :src="panoramaThumbUrl"
              alt="全景照片"
              class="tc-panorama-thumb"
              @click="openPanorama"
            />
            <a :href="panoramaThumbUrl" target="_blank" class="tc-panorama-link">查看全景</a>
          </div>

          <!-- POI (flat format per design doc B.2) -->
          <div v-if="card.physicalLandscape.poi" class="tc-poi-list">
            <div v-if="card.physicalLandscape.poi.address" class="tc-poi-item">
              <span class="tc-poi-label">地址:</span>
              <span class="tc-poi-value">{{ card.physicalLandscape.poi.address }}</span>
            </div>
            <div v-if="card.physicalLandscape.poi.openTime" class="tc-poi-item">
              <span class="tc-poi-label">开放时间:</span>
              <span class="tc-poi-value">{{ card.physicalLandscape.poi.openTime }}</span>
            </div>
            <div v-if="card.physicalLandscape.poi.ticketPrice" class="tc-poi-item">
              <span class="tc-poi-label">门票:</span>
              <span class="tc-poi-value">{{ card.physicalLandscape.poi.ticketPrice }}</span>
            </div>
          </div>
        </div>

        <!-- Literary records section (scrollable) -->
        <div class="tc-section">
          <div class="tc-section-title">
            文献记录
            <span class="tc-mention-count">{{ card.literaryRecords.length }}条</span>
          </div>
          <div class="tc-records">
            <div
              v-for="(record, idx) in card.literaryRecords"
              :key="idx"
              class="tc-record-item"
              :class="{ 'tc-compared': isDynastyCompared(record.dynasty), 'tc-non-compared': isCompareMode && !isDynastyCompared(record.dynasty) }"
            >
              <div class="tc-record-header">
                <span class="tc-record-dynasty">{{ record.dynasty }}</span>
                <span class="tc-record-chapter">{{ record.chapterTitle }}</span>
                <span class="tc-record-style">{{ record.descriptionStyle }}</span>
              </div>
              <div class="tc-record-excerpt">"{{ record.excerpt.slice(0, 100) }}{{ record.excerpt.length > 100 ? '...' : '' }}"</div>
              <button class="tc-view-text-btn" @click="viewOriginalText(record)">查看原文</button>
            </div>
            <div v-if="card.literaryRecords.length === 0" class="tc-no-records">
              暂无文献记录，请使用Agent检索
            </div>
          </div>
        </div>

        <!-- U-18: Compare mode evolution summary (Agent generated or basic mention counts) -->
        <div v-if="isCompareMode" class="tc-section tc-compare-summary">
          <div class="tc-section-title">朝代对比</div>
          <div v-if="mapStore.compareSummary" class="tc-mutual">
            <p class="tc-compare-overview">{{ mapStore.compareSummary.overview }}</p>
            <ul v-if="mapStore.compareSummary.keyFindings.length > 0" class="tc-findings-list">
              <li v-for="(kf, idx) in mapStore.compareSummary.keyFindings" :key="idx">{{ kf }}</li>
            </ul>
          </div>
          <div v-else class="tc-mutual">
            <div class="tc-mutual-item">
              <span class="tc-mutual-label">{{ compareDynasties?.left }} vs {{ compareDynasties?.right }}</span>
              <p>{{ card.totalMentions[compareDynasties?.left || ''] || 0 }} vs {{ card.totalMentions[compareDynasties?.right || ''] || 0 }}</p>
            </div>
          </div>
        </div>

        <!-- Mutual construction section -->
        <div v-if="card.mutualConstruction" class="tc-section">
          <div class="tc-section-title">景观-文本互构</div>
          <div class="tc-mutual">
            <div class="tc-mutual-item">
              <span class="tc-mutual-label">景观→文本</span>
              <p>{{ card.mutualConstruction.landscapeToText }}</p>
            </div>
            <div class="tc-mutual-item">
              <span class="tc-mutual-label">文本→景观</span>
              <p>{{ card.mutualConstruction.textToLandscape }}</p>
            </div>
          </div>
        </div>

      </template>
    </div>

    <div class="tc-footer">
      <button class="tc-action-btn" @click="emit('track-evolution', card.locationName)">
        追踪演化
      </button>
      <button class="tc-action-btn" @click="emit('add-to-notes', card.locationName)">
        添加笔记
      </button>
    </div>
  </div>
</template>

<style scoped>
.traceability-card {
  position: absolute;
  z-index: 1010;
  width: 420px;
  max-height: min(600px, calc(100vh - 40px));
  background: var(--color-bg-secondary, #F7F8FA);
  border: 1px solid var(--color-border, #E5E7EB);
  border-radius: var(--radius, 8px);
  box-shadow: var(--shadow-md, 0 8px 24px rgba(0, 0, 0, 0.12));
  display: flex;
  flex-direction: column;
}

.tc-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 10px 14px;
  border-bottom: 1px solid var(--color-border, #E5E7EB);
  flex-shrink: 0;
}

.tc-header h4 {
  font-size: 0.88rem;
  color: var(--color-text-primary, #1A1A1A);
  margin: 0;
  font-weight: 500;
}

.tc-close {
  background: none;
  border: none;
  font-size: 1.1rem;
  color: var(--color-text-secondary, #4B5563);
  cursor: pointer;
  transition: color var(--dur-fast, 120ms) var(--ease, cubic-bezier(.22,.61,.36,1));
}

.tc-close:hover {
  color: var(--color-text-primary, #1A1A1A);
}

.tc-body {
  flex: 1;
  overflow-y: auto;
  padding: 10px 14px;
}

.tc-loading {
  text-align: center;
  padding: 20px;
  font-size: 0.75rem;
  color: var(--color-text-secondary, #4B5563);
}

/* P2-10: Dynasty mention overview */
.tc-dynasty-mentions {
  font-size: 0.65rem;
  color: var(--color-text-secondary, #4B5563);
  margin-bottom: 10px;
  padding: 4px 8px;
  background: var(--color-bg-tertiary, #F0F2F5);
  border-radius: var(--radius-sm, 4px);
}

.tc-section {
  margin-bottom: 12px;
}

.tc-section-title {
  font-size: 0.75rem;
  color: var(--color-text-primary, #1A1A1A);
  font-weight: 500;
  margin-bottom: 6px;
  display: flex;
  align-items: center;
  gap: 6px;
}

.tc-mention-count {
  font-size: 0.65rem;
  color: var(--color-text-tertiary, #6B7280);
  font-weight: 400;
}

.tc-info-grid {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 4px;
  margin-bottom: 6px;
}

.tc-info-item {
  display: flex;
  gap: 4px;
  font-size: 0.7rem;
}

.tc-info-label {
  color: var(--color-text-secondary, #4B5563);
  flex-shrink: 0;
}

.tc-info-value {
  color: var(--color-text-primary, #1A1A1A);
}

.tc-changes {
  font-size: 0.7rem;
  color: var(--color-text-secondary, #4B5563);
  line-height: 1.4;
}

/* P2-08: Panorama thumbnail */
.tc-panorama {
  margin-top: 6px;
  display: flex;
  align-items: center;
  gap: 8px;
}

.tc-panorama-thumb {
  width: 120px;
  height: 80px;
  object-fit: cover;
  border-radius: var(--radius-sm, 4px);
  border: 1px solid var(--color-border, #E5E7EB);
  cursor: pointer;
  transition: border-color var(--dur-fast, 120ms) var(--ease, cubic-bezier(.22,.61,.36,1));
}

.tc-panorama-thumb:hover {
  border-color: var(--color-accent, #5B9BD5);
}

.tc-panorama-link {
  font-size: 0.65rem;
  color: var(--color-accent, #5B9BD5);
  text-decoration: none;
  transition: color var(--dur-fast, 120ms) var(--ease, cubic-bezier(.22,.61,.36,1));
}

.tc-panorama-link:hover {
  text-decoration: underline;
  color: var(--color-accent-hover, #4A8BC5);
}

.tc-poi-list {
  margin-top: 6px;
  display: flex;
  flex-direction: column;
  gap: 2px;
}

.tc-poi-item {
  display: flex;
  gap: 6px;
  font-size: 0.68rem;
}

.tc-poi-label {
  color: var(--color-text-secondary, #4B5563);
  flex-shrink: 0;
}

.tc-poi-value {
  color: var(--color-text-secondary, #4B5563);
}

.tc-records {
  max-height: 200px;
  overflow-y: auto;
  display: flex;
  flex-direction: column;
  gap: 6px;
}

.tc-record-item {
  padding: 6px 8px;
  background: var(--color-bg-tertiary, #F0F2F5);
  border-radius: var(--radius-sm, 4px);
  border: 1px solid transparent;
  transition: border-color var(--dur-fast, 120ms) var(--ease, cubic-bezier(.22,.61,.36,1)),
              background var(--dur-fast, 120ms) var(--ease, cubic-bezier(.22,.61,.36,1));
}

.tc-record-item.tc-compared {
  border-color: var(--color-accent, #5B9BD5);
  background: var(--color-accent-light, #EBF4FC);
}

/* P2-09: Non-compared items fade */
.tc-record-item.tc-non-compared {
  opacity: 0.4;
}

.tc-record-header {
  display: flex;
  gap: 6px;
  margin-bottom: 2px;
  flex-wrap: wrap;
}

.tc-record-dynasty {
  font-size: 0.65rem;
  color: var(--color-text-secondary, #4B5563);
  font-weight: 500;
}

.tc-record-chapter {
  font-size: 0.65rem;
  color: var(--color-text-secondary, #4B5563);
}

.tc-record-style {
  font-size: 0.6rem;
  color: var(--color-text-tertiary, #6B7280);
  margin-left: auto;
}

.tc-record-excerpt {
  font-size: 0.72rem;
  color: var(--color-text-primary, #1A1A1A);
  line-height: 1.4;
  margin: 4px 0;
}

.tc-view-text-btn {
  font-size: 0.65rem;
  padding: 2px 8px;
  background: var(--color-bg-primary, #FFFFFF);
  color: var(--color-text-secondary, #4B5563);
  border: 1px solid var(--color-border, #E5E7EB);
  border-radius: var(--radius-sm, 4px);
  cursor: pointer;
  transition: color var(--dur-fast, 120ms) var(--ease, cubic-bezier(.22,.61,.36,1)),
              border-color var(--dur-fast, 120ms) var(--ease, cubic-bezier(.22,.61,.36,1));
}

.tc-view-text-btn:hover {
  color: var(--color-accent, #5B9BD5);
  border-color: var(--color-accent, #5B9BD5);
}

.tc-no-records {
  font-size: 0.7rem;
  color: var(--color-text-secondary, #4B5563);
  text-align: center;
  padding: 12px;
}

/* P2-09: Compare summary */
.tc-compare-summary {
  padding: 8px;
  background: var(--color-accent-light, #EBF4FC);
  border-radius: var(--radius-sm, 4px);
  border: 1px dashed var(--color-accent, #5B9BD5);
}

/* U-18: Agent compare summary */
.tc-compare-overview {
  font-size: 0.7rem;
  color: var(--color-text-primary, #1A1A1A);
  line-height: 1.5;
  margin: 0 0 6px;
}

.tc-findings-list {
  margin: 0;
  padding: 0 0 0 14px;
  list-style: disc;
}

.tc-findings-list li {
  font-size: 0.68rem;
  color: var(--color-text-secondary, #4B5563);
  line-height: 1.4;
  margin-bottom: 2px;
}

.tc-mutual {
  display: flex;
  flex-direction: column;
  gap: 6px;
}

.tc-mutual-item {
  padding: 6px 8px;
  background: var(--color-bg-tertiary, #F0F2F5);
  border-radius: var(--radius-sm, 4px);
  border: 1px solid var(--color-border, #E5E7EB);
}

.tc-mutual-label {
  font-size: 0.65rem;
  color: var(--color-accent, #5B9BD5);
  display: block;
  margin-bottom: 2px;
  font-weight: 500;
}

.tc-mutual-item p {
  font-size: 0.7rem;
  color: var(--color-text-secondary, #4B5563);
  line-height: 1.4;
  margin: 0;
}

.tc-footer {
  padding: 10px 14px;
  border-top: 1px solid var(--color-border, #E5E7EB);
  display: flex;
  flex-direction: column;
  gap: 4px;
  flex-shrink: 0;
}

.tc-action-btn {
  padding: 5px 10px;
  font-size: 0.7rem;
  background: var(--color-bg-tertiary, #F0F2F5);
  color: var(--color-text-primary, #1A1A1A);
  border: 1px solid var(--color-border, #E5E7EB);
  border-radius: var(--radius-sm, 4px);
  cursor: pointer;
  text-align: center;
  transition: background var(--dur-fast, 120ms) var(--ease, cubic-bezier(.22,.61,.36,1)),
              border-color var(--dur-fast, 120ms) var(--ease, cubic-bezier(.22,.61,.36,1)),
              color var(--dur-fast, 120ms) var(--ease, cubic-bezier(.22,.61,.36,1));
}

.tc-action-btn:hover {
  background: var(--color-bg-primary, #FFFFFF);
  border-color: var(--color-accent, #5B9BD5);
  color: var(--color-accent, #5B9BD5);
}
</style>
