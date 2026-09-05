<script setup lang="ts">
import { ref, computed, watch, onMounted, onUnmounted } from 'vue'
import { useWorkspaceStore } from '@/stores/workspace'
import { useSourceTreeStore } from '@/stores/sourceTree'
import { useAnnotationStore } from '@/stores/annotations'
import { useAgentStore } from '@/stores/agent'
import { useDynastyCompareStore } from '@/stores/dynastyCompare'
import SplitDivider from '@/components/layout/SplitDivider.vue'
import TextReaderView from '@/views/TextReaderView.vue'

const workspaceStore = useWorkspaceStore()
const sourceTreeStore = useSourceTreeStore()
const annotationStore = useAnnotationStore()
const agentStore = useAgentStore()
const dynastyCompareStore = useDynastyCompareStore()

const splitState = computed(() => workspaceStore.split)

const leftTab = computed(() =>
  workspaceStore.tabs.find(t => t.id === splitState.value.leftTabId) || null
)
const rightTab = computed(() =>
  workspaceStore.tabs.find(t => t.id === splitState.value.rightTabId) || null
)

const syncScroll = computed(() => splitState.value.syncScroll)
const alignParagraphs = computed(() => splitState.value.alignParagraphs)
const splitRatio = computed(() => splitState.value.splitRatio)

function onRatioChange(ratio: number) {
  workspaceStore.setSplitRatio(ratio)
}

function toggleSyncScroll() {
  workspaceStore.toggleSyncScroll()
}

function toggleAlignParagraphs() {
  workspaceStore.toggleAlignParagraphs()
}

function swapSides() {
  workspaceStore.swapSplitSides()
}

function analyzeDifferences() {
  const leftTitle = leftTab.value?.title || ''
  const rightTitle = rightTab.value?.title || ''
  agentStore.sendMessage(`请对比分析 ${leftTitle} 和 ${rightTitle} 的差异`)
}

// INF-05: Open dynasty compare for split view
function openDynastyCompare() {
  const leftData = leftTab.value?.data as Record<string, unknown> | undefined
  const rightData = rightTab.value?.data as Record<string, unknown> | undefined
  const leftChNum = leftData?.chapterNumber as number | undefined
  const rightChNum = rightData?.chapterNumber as number | undefined

  const leftCh = leftChNum ? sourceTreeStore.chapters.find(c => c.number === leftChNum) : null
  const rightCh = rightChNum ? sourceTreeStore.chapters.find(c => c.number === rightChNum) : null

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
  } else {
    dynastyCompareStore.setPendingCompareDynasties({
      left: leftCh?.dynasty || '',
      right: rightCh?.dynasty || '',
    })
  }
}

// Alignment data from SSE event
interface ParagraphAlignmentEntry {
  left_paragraph: number
  right_paragraph: number
  confidence: number
}

const alignmentMap = computed<Map<number, number>>(() => {
  const map = new Map<number, number>()
  // Check left and right tabs for alignment data
  for (const tab of [leftTab.value, rightTab.value]) {
    if (!tab) continue
    const data = tab.data as Record<string, unknown>
    const alignments = data.paragraphAlignments as ParagraphAlignmentEntry[] | undefined
    if (alignments && alignments.length > 0) {
      for (const entry of alignments) {
        if (tab.id === leftTab.value?.id) {
          map.set(entry.left_paragraph, entry.right_paragraph)
        } else {
          map.set(entry.right_paragraph, entry.left_paragraph)
        }
      }
      return map
    }
  }
  return map
})

function findVisibleParagraphIndex(container: HTMLElement): number {
  const paraEls = container.querySelectorAll('[id^="para-"]')
  for (const el of paraEls) {
    const rect = el.getBoundingClientRect()
    const containerRect = container.getBoundingClientRect()
    // First paragraph whose top is inside the visible viewport
    if (rect.top >= containerRect.top - 10 && rect.top < containerRect.bottom) {
      const id = el.getAttribute('id') || ''
      const match = id.match(/^para-(\d+)$/)
      if (match) return parseInt(match[1], 10)
    }
  }
  return -1
}

// Sync scroll logic
const leftContentRef = ref<HTMLElement | null>(null)
const rightContentRef = ref<HTMLElement | null>(null)
let syncScrollLock = false

function onLeftScroll() {
  if (!syncScroll.value || syncScrollLock) return
  if (!leftContentRef.value || !rightContentRef.value) return
  syncScrollLock = true

  if (alignParagraphs.value && alignmentMap.value.size > 0) {
    const visibleIdx = findVisibleParagraphIndex(leftContentRef.value)
    const mapped = alignmentMap.value.get(visibleIdx)
    if (mapped !== undefined) {
      const targetEl = rightContentRef.value.querySelector(`#para-${mapped}`)
      if (targetEl) {
        targetEl.scrollIntoView({ block: 'start', behavior: 'auto' })
      }
    }
  } else {
    rightContentRef.value.scrollTop = leftContentRef.value.scrollTop
  }

  setTimeout(() => { syncScrollLock = false }, 200)
}

function onRightScroll() {
  if (!syncScroll.value || syncScrollLock) return
  if (!leftContentRef.value || !rightContentRef.value) return
  syncScrollLock = true

  if (alignParagraphs.value && alignmentMap.value.size > 0) {
    const visibleIdx = findVisibleParagraphIndex(rightContentRef.value)
    const mapped = -1
    // Reverse lookup from right to left
    let leftIdx = -1
    for (const [l, r] of alignmentMap.value.entries()) {
      if (r === visibleIdx) { leftIdx = l; break }
    }
    if (leftIdx >= 0) {
      const targetEl = leftContentRef.value.querySelector(`#para-${leftIdx}`)
      if (targetEl) {
        targetEl.scrollIntoView({ block: 'start', behavior: 'auto' })
      }
    }
  } else {
    leftContentRef.value.scrollTop = rightContentRef.value.scrollTop
  }

  setTimeout(() => { syncScrollLock = false }, 200)
}

// Expose split side info for child components
const leftSplitSide = ref('left')
const rightSplitSide = ref('right')
</script>

<template>
  <div class="split-view">
    <!-- Split toolbar -->
    <div class="sv-toolbar">
      <div class="sv-toolbar-left">
        <span class="sv-label">{{ leftTab?.title || 'Left' }}</span>
      </div>
      <div class="sv-toolbar-controls">
        <button
          class="sv-control-btn"
          :class="{ active: syncScroll }"
          @click="toggleSyncScroll"
        >同步滚动</button>
        <button
          class="sv-control-btn"
          :class="{ active: alignParagraphs }"
          @click="toggleAlignParagraphs"
        >对齐段落</button>
        <button class="sv-control-btn" @click="swapSides">交换位置</button>
        <button
          class="sv-control-btn sv-dc-btn"
          @click="openDynastyCompare"
          title=""
        >跨朝代对比</button>
        <button
          class="sv-control-btn sv-analyze-btn"
          :disabled="agentStore.isStreaming"
          @click="analyzeDifferences"
        >AI 差异分析</button>
      </div>
      <div class="sv-toolbar-right">
        <span class="sv-label">{{ rightTab?.title || 'Right' }}</span>
      </div>
    </div>

    <!-- Dual panels -->
    <div class="sv-panels">
      <div class="sv-panel sv-left" :style="{ flex: `0 0 ${splitRatio * 100}%` }">
        <div class="sv-panel-content" ref="leftContentRef" @scroll="onLeftScroll">
          <TextReaderView
            v-if="leftTab"
            :tab-id="leftTab.id"
            split-side="left"
            :key="leftTab.id + '-left'"
          />
        </div>
      </div>

      <SplitDivider
        @ratio-change="onRatioChange"
      />

      <div class="sv-panel sv-right" :style="{ flex: `0 0 ${(1 - splitRatio) * 100}%` }">
        <div class="sv-panel-content" ref="rightContentRef" @scroll="onRightScroll">
          <TextReaderView
            v-if="rightTab"
            :tab-id="rightTab.id"
            split-side="right"
            :key="rightTab.id + '-right'"
          />
        </div>
      </div>
    </div>
  </div>
</template>

<style scoped>
.split-view {
  display: flex;
  flex-direction: column;
  height: 100%;
  overflow: hidden;
}

.sv-toolbar {
  display: flex;
  align-items: center;
  padding: var(--space-2) var(--space-3);
  background: var(--color-bg-secondary);
  border-bottom: 1px solid var(--color-border);
  gap: var(--space-2);
}

.sv-toolbar-left,
.sv-toolbar-right {
  flex: 1;
  min-width: 0;
}

.sv-toolbar-right {
  text-align: right;
}

.sv-label {
  font-size: 0.8rem;
  color: var(--color-text-primary);
  font-weight: 500;
}

.sv-toolbar-controls {
  display: flex;
  gap: var(--space-1);
}

.sv-control-btn {
  padding: var(--space-1) var(--space-2);
  font-size: 0.7rem;
  background: var(--color-bg-primary);
  color: var(--color-text-secondary);
  border: 1px solid var(--color-border);
  border-radius: var(--radius-sm);
  cursor: pointer;
  font-family: var(--font-sans);
  font-weight: 500;
  transition:
    background-color var(--dur-fast) var(--ease),
    border-color var(--dur-fast) var(--ease),
    color var(--dur-fast) var(--ease),
    transform var(--dur) var(--ease),
    box-shadow var(--dur) var(--ease);
}

.sv-control-btn:hover:not(:disabled) {
  background: var(--color-bg-hover);
  border-color: var(--color-border-dark);
  color: var(--color-text-primary);
  transform: translateY(-1px);
  box-shadow: var(--shadow-sm);
}

.sv-control-btn:active:not(:disabled) {
  background: var(--color-bg-active);
  transform: translateY(0);
  box-shadow: none;
}

.sv-control-btn.active {
  background: var(--color-accent-light);
  color: var(--color-accent);
  border-color: var(--color-accent);
}

.sv-analyze-btn {
  border-color: var(--color-border);
  color: var(--color-text-secondary);
}

.sv-dc-btn {
  border-color: var(--color-border);
  color: var(--color-text-secondary);
}

.sv-analyze-btn:hover:not(:disabled) {
  background: var(--color-bg-hover);
  color: var(--color-text-primary);
}

.sv-analyze-btn:disabled {
  background: var(--color-bg-tertiary);
  color: var(--color-text-disabled);
  border-color: var(--color-border);
  cursor: not-allowed;
  transform: none !important;
  box-shadow: none !important;
}

.sv-panels {
  display: flex;
  flex: 1;
  overflow: hidden;
}

.sv-panel {
  min-width: 0;
  overflow: hidden;
}

.sv-panel-content {
  height: 100%;
  overflow-y: auto;
}
</style>
