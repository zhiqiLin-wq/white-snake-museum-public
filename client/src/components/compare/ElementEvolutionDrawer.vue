<script setup lang="ts">
import { computed } from 'vue'
import { useDynastyCompareStore } from '@/stores/dynastyCompare'
import { useSourceTreeStore } from '@/stores/sourceTree'
import { useWorkspaceStore } from '@/stores/workspace'
import { useMapStore } from '@/stores/map'
import type { ElementOccurrence, ElementEvolution } from '@/types/dynasty-compare'

const store = useDynastyCompareStore()
const sourceTreeStore = useSourceTreeStore()
const workspaceStore = useWorkspaceStore()
const mapStore = useMapStore()

const emit = defineEmits<{
  close: []
  'view-on-map': [locationName: string]
}>()

const selected = computed(() => store.selectedElement)
const isOpen = computed(() => store.isDrawerOpen)
const isLocation = computed(() => selected.value?.category === 'location')

const elementName = computed(() => {
  if (!selected.value) return ''
  return selected.value.elementName
})

const confidence = computed(() => {
  if (!selected.value) return 0
  const sel = selected.value as Record<string, unknown>
  if (sel.evolution && typeof sel.evolution === 'object') {
    return (sel.evolution as Record<string, number>).confidence || 0
  }
  return 0
})

const confidencePercent = computed(() => Math.round(confidence.value * 100))

const confidenceBar = computed(() => {
  const pct = confidencePercent.value
  const filled = Math.round(pct / 10)
  return '█'.repeat(filled) + '░'.repeat(10 - filled)
})

const evolutionSummary = computed(() => {
  if (!selected.value) return ''
  const sel = selected.value as Record<string, unknown>
  if (sel.evolution && typeof sel.evolution === 'object') {
    return (sel.evolution as Record<string, string>).summary || ''
  }
  return ''
})

function getOccurrences(side: 'left' | 'right'): ElementOccurrence[] {
  if (!selected.value) return []
  return side === 'left' ? selected.value.leftOccurrences : selected.value.rightOccurrences
}

function getChapterTitle(chapterNumber: number): string {
  return sourceTreeStore.chapters.find(c => c.number === chapterNumber)?.title || `${chapterNumber}`
}

function scrollToOccurrence(occ: ElementOccurrence) {
  sourceTreeStore.setNavigationTarget({
    chapterNumber: occ.chapterNumber,
    paragraphIndex: occ.paragraphIndex,
  })
  // Open text-reader in current view (don't open new tab)
  const ch = sourceTreeStore.chapters.find(c => c.number === occ.chapterNumber)
  const chapTitle = ch ? `${ch.dynasty} · ${ch.title}` : `章节${occ.chapterNumber}`
  workspaceStore.openTab('text-reader', chapTitle, {
    chapterNumber: occ.chapterNumber,
    paragraphIndex: occ.paragraphIndex,
  })
}

function viewOnMap() {
  if (!selected.value || selected.value.category !== 'location') return
  emit('view-on-map', elementName.value)
}

function trackAcrossDynasties() {
  if (!selected.value) return
  store.closeDrawer()
}

function addToResearchNotes() {
  if (!selected.value) return
  store.closeDrawer()
}
</script>

<template>
  <Teleport to="body">
    <Transition name="drawer-fade">
      <div v-if="isOpen" class="drawer-backdrop" @click="store.closeDrawer()" />
    </Transition>
    <Transition name="drawer-slide">
      <div v-if="isOpen && selected" class="element-evolution-drawer">
        <div class="eed-header">
          <h3>{{ elementName }}</h3>
          <button class="eed-close" @click="store.closeDrawer()">&times;</button>
        </div>

        <div class="eed-body">
          <!-- Confidence bar (P1-02) -->
          <div v-if="confidence > 0" class="eed-confidence">
            <div class="eed-conf-label">置信度: {{ confidenceBar }} {{ confidencePercent }}%</div>
          </div>

          <div class="eed-section">
            <div class="eed-label">分类: {{ selected.category }}</div>
            <div class="eed-label">变化类型: {{ selected.evolution.changeType }}</div>
          </div>

          <div class="eed-section">
            <div class="eed-subtitle">{{ store.compareConfig?.left.dynasty || '' }}</div>
            <div class="eed-occurrences">
              <div
                v-for="occ in getOccurrences('left')"
                :key="`${occ.chapterNumber}:${occ.paragraphIndex}:${occ.span.startChar}`"
                class="eed-occ-item"
                @click="scrollToOccurrence(occ)"
              >
                <span class="eed-occ-chapter">{{ getChapterTitle(occ.chapterNumber) }}</span>
                <span class="eed-occ-excerpt">{{ (occ.text || '').slice(0, 80) }}{{ (occ.text || '').length > 80 ? '...' : '' }}</span>
              </div>
              <div class="eed-occ-summary">共 {{ getOccurrences('left').length }} 处出现</div>
            </div>
          </div>

          <div class="eed-section">
            <div class="eed-subtitle">{{ store.compareConfig?.right.dynasty || '' }}</div>
            <div class="eed-occurrences">
              <div
                v-for="occ in getOccurrences('right')"
                :key="`${occ.chapterNumber}:${occ.paragraphIndex}:${occ.span.startChar}`"
                class="eed-occ-item"
                @click="scrollToOccurrence(occ)"
              >
                <span class="eed-occ-chapter">{{ getChapterTitle(occ.chapterNumber) }}</span>
                <span class="eed-occ-excerpt">{{ (occ.text || '').slice(0, 80) }}{{ (occ.text || '').length > 80 ? '...' : '' }}</span>
              </div>
              <div class="eed-occ-summary">共 {{ getOccurrences('right').length }} 处出现</div>
            </div>
          </div>

          <div v-if="evolutionSummary" class="eed-section eed-analysis">
            <div class="eed-subtitle">AI 演变分析</div>
            <p>{{ evolutionSummary }}</p>
          </div>
        </div>

        <div class="eed-footer">
          <button v-if="isLocation" class="eed-action-btn" @click="viewOnMap">
            在地图上查看
          </button>
          <button class="eed-action-btn" @click="trackAcrossDynasties">
            跨朝代追踪
          </button>
          <button class="eed-action-btn eed-notes-btn" @click="addToResearchNotes">
            添加到研究笔记
          </button>
        </div>
      </div>
    </Transition>
  </Teleport>
</template>

<style scoped>
.drawer-backdrop {
  position: fixed;
  inset: 0;
  z-index: 8999;
  background: rgba(26, 26, 26, 0.35);
}

.element-evolution-drawer {
  position: fixed;
  right: 0;
  top: 0;
  width: 380px;
  height: 100vh;
  z-index: 9000;
  background: var(--color-bg-secondary, #F7F8FA);
  border-left: 1px solid var(--color-border, #E5E7EB);
  box-shadow: var(--shadow-md, 0 -4px 24px rgba(0, 0, 0, 0.12));
  display: flex;
  flex-direction: column;
}

.eed-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 12px 16px;
  border-bottom: 1px solid var(--color-border, #E5E7EB);
  flex-shrink: 0;
}

.eed-header h3 {
  font-size: 0.9rem;
  color: var(--color-text-primary, #1A1A1A);
  margin: 0;
  font-weight: 500;
}

.eed-close {
  background: none;
  border: none;
  font-size: 1.2rem;
  color: var(--color-text-secondary, #4B5563);
  cursor: pointer;
  transition: color var(--dur-fast, 120ms) var(--ease, cubic-bezier(.22,.61,.36,1));
}

.eed-close:hover {
  color: var(--color-text-primary, #1A1A1A);
}

.eed-body {
  flex: 1;
  overflow-y: auto;
  padding: 12px 16px;
}

.eed-confidence {
  padding: 6px 8px;
  background: var(--color-accent-light, #EBF4FC);
  border-radius: var(--radius-sm, 4px);
  margin-bottom: 10px;
  text-align: center;
}

.eed-conf-label {
  font-size: 0.7rem;
  color: var(--color-accent, #5B9BD5);
  font-family: monospace;
}

.eed-section {
  margin-bottom: 14px;
}

.eed-label {
  font-size: 0.72rem;
  color: var(--color-text-secondary, #4B5563);
}

.eed-subtitle {
  font-size: 0.78rem;
  color: var(--color-text-primary, #1A1A1A);
  font-weight: 500;
  margin-bottom: 6px;
}

.eed-occurrences {
  display: flex;
  flex-direction: column;
  gap: 6px;
}

.eed-occ-item {
  padding: 6px 8px;
  background: var(--color-bg-tertiary, #F0F2F5);
  border: 1px solid var(--color-border, #E5E7EB);
  border-radius: var(--radius-sm, 4px);
  cursor: pointer;
  transition: background var(--dur-fast, 120ms) var(--ease, cubic-bezier(.22,.61,.36,1)),
              border-color var(--dur-fast, 120ms) var(--ease, cubic-bezier(.22,.61,.36,1));
}

.eed-occ-item:hover {
  background: var(--color-bg-primary, #FFFFFF);
  border-color: var(--color-accent, #5B9BD5);
}

.eed-occ-chapter {
  display: block;
  font-size: 0.65rem;
  color: var(--color-text-secondary, #4B5563);
  margin-bottom: 2px;
}

.eed-occ-excerpt {
  font-size: 0.75rem;
  color: var(--color-text-primary, #1A1A1A);
  line-height: 1.4;
}

.eed-occ-summary {
  font-size: 0.65rem;
  color: var(--color-text-tertiary, #6B7280);
  text-align: right;
  padding-right: 4px;
}

.eed-analysis {
  padding: 10px;
  background: var(--color-bg-tertiary, #F0F2F5);
  border-radius: var(--radius, 8px);
  border: 1px solid var(--color-border, #E5E7EB);
}

.eed-analysis p {
  font-size: 0.75rem;
  color: var(--color-text-secondary, #4B5563);
  line-height: 1.5;
  margin: 0;
}

.eed-footer {
  padding: 12px 16px;
  border-top: 1px solid var(--color-border, #E5E7EB);
  display: flex;
  flex-direction: column;
  gap: 6px;
  flex-shrink: 0;
}

.eed-action-btn {
  padding: 6px 12px;
  font-size: 0.72rem;
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

.eed-action-btn:hover {
  background: var(--color-bg-primary, #FFFFFF);
  border-color: var(--color-accent, #5B9BD5);
}

.eed-notes-btn {
  color: var(--color-text-secondary, #4B5563);
  border-color: var(--color-border, #E5E7EB);
}

.eed-notes-btn:hover {
  color: var(--color-accent, #5B9BD5);
  border-color: var(--color-accent, #5B9BD5);
}

.drawer-fade-enter-active { transition: opacity var(--dur, 220ms) var(--ease, cubic-bezier(.22,.61,.36,1)); }
.drawer-fade-leave-active { transition: opacity var(--dur-fast, 120ms) var(--ease, cubic-bezier(.22,.61,.36,1)); }
.drawer-fade-enter-from, .drawer-fade-leave-to { opacity: 0; }

.drawer-slide-enter-active { transition: transform var(--dur, 220ms) var(--ease, cubic-bezier(.22,.61,.36,1)); }
.drawer-slide-leave-active { transition: transform var(--dur-fast, 120ms) var(--ease, cubic-bezier(.22,.61,.36,1)); }
.drawer-slide-enter-from { transform: translateX(100%); }
.drawer-slide-leave-to { transform: translateX(100%); }
</style>
