<script setup lang="ts">
import { ref, computed, watch, nextTick } from 'vue'
import type { WorkspaceTab } from '@/types/workspace'
import TabItem from './TabItem.vue'

const props = defineProps<{
  tabs: WorkspaceTab[]
  activeTabId: string | null
}>()

const emit = defineEmits<{
  'select-tab': [tabId: string]
  'close-tab': [tabId: string]
  'split-drop': [tabId: string, side: 'left' | 'right']
  'drag-back': [tabId: string]
  'reorder-tabs': [fromIndex: number, toIndex: number]
  'open-compare-selector': []
}>()

const scrollContainer = ref<HTMLElement | null>(null)
const showLeftArrow = ref(false)
const showRightArrow = ref(false)
const leftDropZone = ref(false)
const rightDropZone = ref(false)
const dragOverIndex = ref(-1)
const dragFromIndex = ref(-1)

function updateArrows() {
  const el = scrollContainer.value
  if (!el) return
  showLeftArrow.value = el.scrollLeft > 0
  showRightArrow.value = el.scrollLeft + el.clientWidth < el.scrollWidth - 1
}
function scrollLeft() { scrollContainer.value?.scrollBy({ left: -150, behavior: 'smooth' }) }
function scrollRight() { scrollContainer.value?.scrollBy({ left: 150, behavior: 'smooth' }) }

function onDragStart(event: DragEvent, tabId: string) {
  dragFromIndex.value = props.tabs.findIndex(t => t.id === tabId)
}
function onDragOver(event: DragEvent) {
  event.preventDefault()
  if (!event.dataTransfer) return
  event.dataTransfer.dropEffect = 'move'
  const rect = (event.currentTarget as HTMLElement).getBoundingClientRect()
  const x = event.clientX - rect.left
  const width = rect.width
  leftDropZone.value = x < width * 0.2
  rightDropZone.value = x > width * 0.8
  if (scrollContainer.value && !leftDropZone.value && !rightDropZone.value) {
    const tabEls = scrollContainer.value.querySelectorAll('[data-tab-id]')
    let closestIdx = -1
    let closestDist = Infinity
    const mouseX = event.clientX
    tabEls.forEach((el) => {
      const r = el.getBoundingClientRect()
      const centerX = r.left + r.width / 2
      const dist = Math.abs(mouseX - centerX)
      if (dist < closestDist) {
        closestDist = dist
        const tid = el.getAttribute('data-tab-id')
        closestIdx = props.tabs.findIndex(t => t.id === tid)
      }
    })
    dragOverIndex.value = closestIdx
  } else {
    dragOverIndex.value = -1
  }
}
function onDragLeave() {
  leftDropZone.value = false; rightDropZone.value = false; dragOverIndex.value = -1
}
function onDragEnd() {
  leftDropZone.value = false; rightDropZone.value = false; dragOverIndex.value = -1; dragFromIndex.value = -1
}
function onDrop(event: DragEvent) {
  event.preventDefault()
  const tabId = event.dataTransfer?.getData('text/plain')
  if (!tabId) { onDragEnd(); return }
  const rect = (event.currentTarget as HTMLElement).getBoundingClientRect()
  const x = event.clientX - rect.left
  const width = rect.width
  if (x < width * 0.2) {
    emit('split-drop', tabId, 'left')
  } else if (x > width * 0.8) {
    emit('split-drop', tabId, 'right')
  } else if (dragFromIndex.value >= 0 && dragOverIndex.value >= 0 && dragFromIndex.value !== dragOverIndex.value) {
    emit('reorder-tabs', dragFromIndex.value, dragOverIndex.value)
  }
  onDragEnd()
}

watch(() => props.tabs.length, () => void nextTick(updateArrows))
</script>

<template>
  <div class="tab-bar" @dragover="onDragOver" @dragleave="onDragLeave" @drop="onDrop">
    <div class="drop-zone left"  :class="{ active: leftDropZone }" />
    <button v-if="showLeftArrow" class="scroll-arrow" @click="scrollLeft" aria-label="向左滚动">
      <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><polyline points="15 18 9 12 15 6"/></svg>
    </button>

    <div ref="scrollContainer" class="tab-list" @scroll="updateArrows">
      <TabItem
        v-for="(tab, idx) in tabs"
        :key="tab.id"
        :tab-id="tab.id"
        :title="tab.title"
        :active="tab.id === activeTabId"
        :dirty="tab.dirty"
        :class="{ 'drag-over': idx === dragOverIndex && idx !== dragFromIndex }"
        @click="emit('select-tab', tab.id)"
        @close="emit('close-tab', tab.id)"
        @drag-start="(event: DragEvent, id: string) => onDragStart(event, id)"
      />
    </div>

    <button v-if="showRightArrow" class="scroll-arrow" @click="scrollRight" aria-label="向右滚动">
      <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><polyline points="9 18 15 12 9 6"/></svg>
    </button>

    <button
      class="compare-btn"
      :class="{ disabled: tabs.length < 1 }"
      :disabled="tabs.length < 1"
      :title="tabs.length < 1 ? '请先打开一个章节' : '打开对比选择器'"
      @click="tabs.length >= 1 && emit('open-compare-selector')"
      aria-label="打开对比选择器"
    >
      <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
        <path d="M8 3H5a2 2 0 0 0-2 2v3"/><path d="M21 8V5a2 2 0 0 0-2-2h-3"/>
        <path d="M3 16v3a2 2 0 0 0 2 2h3"/><path d="M16 21h3a2 2 0 0 0 2-2v-3"/>
      </svg>
    </button>

    <div class="drop-zone right" :class="{ active: rightDropZone }" />
  </div>
</template>

<style scoped>
/* ============================================================
   TabBar · 极简黑白 · 激活态仅 2px 淡蓝底线
============================================================ */
.tab-bar {
  display: flex;
  align-items: stretch;
  background: var(--color-bg-primary);
  border-bottom: 1px solid var(--color-border);
  height: 40px;
  overflow: hidden;
  position: relative;
}

.tab-list {
  display: flex;
  flex: 1;
  overflow-x: auto;
  overflow-y: hidden;
  scrollbar-width: none;
  padding: 0 2px;
}
.tab-list::-webkit-scrollbar { display: none; }

.scroll-arrow {
  flex-shrink: 0;
  width: 28px;
  border: none;
  border-left: 1px solid var(--color-border-light);
  border-right: 1px solid var(--color-border-light);
  background: var(--color-bg-primary);
  color: var(--color-text-tertiary);
  cursor: pointer;
  display: flex;
  align-items: center;
  justify-content: center;
  z-index: 1;
  transition: background var(--dur-fast) var(--ease), color var(--dur-fast) var(--ease);
}
.scroll-arrow:hover { background: var(--color-bg-hover); color: var(--color-text-primary); }

.compare-btn {
  flex-shrink: 0;
  width: 36px;
  border: none;
  border-left: 1px solid var(--color-border-light);
  background: var(--color-bg-primary);
  color: var(--color-text-tertiary);
  cursor: pointer;
  display: flex;
  align-items: center;
  justify-content: center;
  z-index: 1;
  transition: background var(--dur-fast) var(--ease), color var(--dur-fast) var(--ease);
}
.compare-btn:hover:not(.disabled) {
  color: var(--color-accent);
  background: var(--color-accent-light);
}
.compare-btn.disabled { color: var(--color-text-disabled); cursor: not-allowed; }

.drop-zone {
  width: 20%;
  position: absolute;
  inset: 0 auto 0 0;
  z-index: 10;
  pointer-events: none;
  transition: background var(--dur-fast) var(--ease);
}
.drop-zone.right { left: auto; right: 0; }
.drop-zone.active {
  background: var(--color-accent-light);
  box-shadow: inset 0 -2px 0 var(--color-accent);
  pointer-events: auto;
}
</style>
