<script setup lang="ts">
import { ref, computed, onMounted, onUnmounted, watch } from 'vue'
import { useDynastyCompareStore } from '@/stores/dynastyCompare'
import { useWorkspaceStore } from '@/stores/workspace'
import { useSourceTreeStore } from '@/stores/sourceTree'
import { useAnnotationStore } from '@/stores/annotations'
import { useMapStore } from '@/stores/map'
import { useAgentStore } from '@/stores/agent'
import type { DynastyCompareConfig, ElementEvolution } from '@/types/dynasty-compare'
import type { Annotation } from '@/types/annotation'
import CompareToolbar from '@/components/compare/CompareToolbar.vue'
import ElementEvolutionDrawer from '@/components/compare/ElementEvolutionDrawer.vue'
import CompareReportExport from '@/components/compare/CompareReportExport.vue'
import SplitDivider from '@/components/layout/SplitDivider.vue'
import TextReaderView from '@/views/TextReaderView.vue'

const props = defineProps<{
  tabId?: string
}>()

const workspaceStore = useWorkspaceStore()
const sourceTreeStore = useSourceTreeStore()
const annotationStore = useAnnotationStore()
const mapStore = useMapStore()
const agentStore = useAgentStore()
const store = useDynastyCompareStore()

const currentTab = computed(() => {
  if (props.tabId) {
    return workspaceStore.tabs.find(t => t.id === props.tabId) || workspaceStore.activeTab
  }
  return workspaceStore.activeTab
})

const config = computed((): DynastyCompareConfig | null => {
  const data = currentTab.value?.data as Record<string, unknown>
  return (data?.config as DynastyCompareConfig) || null
})

const splitRatio = ref(0.5)
const exportModalVisible = ref(false)
// B3: 对比页旁注默认折叠（徽章），用户在工具栏切换展开
const margExpanded = ref(false)
// U-14: Removed local errorState; now using store.compareState directly
const analyzeTimeoutTimer = ref<ReturnType<typeof setTimeout> | null>(null)
const timeoutRetryCount = ref(0) // U-14: Track timeout retries (max 1 retry, total 60s)
const hoveredElementId = ref<string | null>(null)

// U-14: State machine now driven by store.compareState — local watch removed

// Left/right tab IDs for sub-reader views
const leftTabId = ref<string>('')
const rightTabId = ref<string>('')

function ensureReaderTabs(): void {
  const cfg = config.value
  if (!cfg) return

  //  createTab  openTab
  if (!leftTabId.value && cfg.left.chapterNumber) {
    const tab = workspaceStore.createTab('text-reader', `${cfg.left.dynasty} - ${cfg.left.chapterTitle}`, {
      chapterNumber: cfg.left.chapterNumber,
    })
    leftTabId.value = tab.id
  }
  if (!rightTabId.value && cfg.right.chapterNumber) {
    const tab = workspaceStore.createTab('text-reader', `${cfg.right.dynasty} - ${cfg.right.chapterTitle}`, {
      chapterNumber: cfg.right.chapterNumber,
    })
    rightTabId.value = tab.id
  }
}

// P0-02: injectElementAnnotations — convert ElementEvolution occurrences to annotations
function injectElementAnnotations(): void {
  const cfg = config.value
  if (!cfg) return

  // Build annotation map keyed by passageKey
  const leftAnnMap = new Map<string, Annotation[]>()
  const rightAnnMap = new Map<string, Annotation[]>()

  for (const element of store.elements) {
    const el = element as ElementEvolution
    const elementId = el.elementId || `elem_${el.elementName}`
    const category = el.category
    const color = el.color

    // Process left occurrences
    for (const occ of el.leftOccurrences) {
      const key = `${occ.chapterNumber}:${occ.paragraphIndex}`
      if (!leftAnnMap.has(key)) leftAnnMap.set(key, [])
      leftAnnMap.get(key)!.push({
        id: `${elementId}_l_${occ.chapterNumber}_${occ.paragraphIndex}_${occ.span.startChar}`,
        category,
        label: el.elementName,
        span: occ.span,
        explanation: occ.contextNote || '',
        confidence: el.evolution.confidence,
        color,
        source: 'agent' as const,
        text: occ.text || '',
        createdAt: Date.now(),
        updatedAt: Date.now(),
      })
    }

    // Process right occurrences
    for (const occ of el.rightOccurrences) {
      const key = `${occ.chapterNumber}:${occ.paragraphIndex}`
      if (!rightAnnMap.has(key)) rightAnnMap.set(key, [])
      rightAnnMap.get(key)!.push({
        id: `${elementId}_r_${occ.chapterNumber}_${occ.paragraphIndex}_${occ.span.startChar}`,
        category,
        label: el.elementName,
        span: occ.span,
        explanation: occ.contextNote || '',
        confidence: el.evolution.confidence,
        color,
        source: 'agent' as const,
        text: occ.text || '',
        createdAt: Date.now(),
        updatedAt: Date.now(),
      })
    }
  }

  // Load into annotation store
  for (const [key, anns] of leftAnnMap.entries()) {
    annotationStore.loadAgentAnnotations(anns, key)
  }
  for (const [key, anns] of rightAnnMap.entries()) {
    annotationStore.loadAgentAnnotations(anns, key)
  }
}

function clearInjectedAnnotations(): void {
  const cfg = config.value
  if (!cfg) return
  //  Agent 
  if (cfg.left.chapterNumber) {
    annotationStore.clearAgentAnnotationsForChapter(cfg.left.chapterNumber)
  }
  if (cfg.right.chapterNumber && cfg.right.chapterNumber !== cfg.left.chapterNumber) {
    annotationStore.clearAgentAnnotationsForChapter(cfg.right.chapterNumber)
  }
}

// P2-03: Hover linkage — watch hoveredElementId and highlight linked occurrences
watch(hoveredElementId, (newId) => {
  // U-15: Set linked highlight with elementId only (sourceSide determined by which panel)
  store.linkedHighlight = newId ? { elementId: newId, sourceSide: 'left' } : null
})

onMounted(() => {
  if (config.value) {
    store.compareConfig = config.value
    ensureReaderTabs()
    injectElementAnnotations()
  }

  // Check pendingCompareDynasties for auto-fill (P1-05: also trigger Agent)
  if (store.pendingCompareDynasties) {
    const pending = store.pendingCompareDynasties
    const leftCh = sourceTreeStore.chapters.find(c => c.dynasty === pending.left)
    const rightCh = sourceTreeStore.chapters.find(c => c.dynasty === pending.right)
    if (leftCh && rightCh) {
      store.compareConfig = {
        left: { chapterNumber: leftCh.number, chapterTitle: leftCh.title, dynasty: leftCh.dynasty },
        right: { chapterNumber: rightCh.number, chapterTitle: rightCh.title, dynasty: rightCh.dynasty },
        syncScroll: true,
        alignParagraphs: false,
      }
      ensureReaderTabs()
      // P1-05: Auto-trigger Agent analysis
      triggerAnalyze()
    }
    store.clearPendingCompareDynasties()
  }
})

onUnmounted(() => {
  //  store 
  clearInjectedAnnotations()
  if (analyzeTimeoutTimer.value) {
    clearTimeout(analyzeTimeoutTimer.value)
    analyzeTimeoutTimer.value = null
  }
  // 
  if (leftTabId.value) {
    workspaceStore.closeTab(leftTabId.value)
    leftTabId.value = ''
  }
  if (rightTabId.value) {
    workspaceStore.closeTab(rightTabId.value)
    rightTabId.value = ''
  }
  //  workspace 
  if (workspaceStore.split.enabled) {
    workspaceStore.exitSplitMode()
  }
})

function setSplitRatio(ratio: number) {
  splitRatio.value = Math.max(0.3, Math.min(0.7, ratio))
}

// P0-03: handleViewOnMap 5-step sequence per design doc A.9.1
function handleViewOnMap(locationName: string) {
  // Step 1: closeDrawer
  store.closeDrawer()

  // Step 2: switchToMap — open map tab
  workspaceStore.openTab('map', `地图: ${locationName}`, { highlightLocation: locationName })

  // Step 3: setCompareDynasties (syncs timeLayer too) + U-18: set compare summary
  const cfg = config.value
  if (cfg) {
    mapStore.setCompareDynasties({ left: cfg.left.dynasty, right: cfg.right.dynasty })
    if (store.summary) {
      mapStore.setCompareSummary(store.summary)
    }
  }

  // Step 4: openTraceabilityCard
  mapStore.openTraceabilityCard(locationName)

  // Step 5: flyTo correct coordinates via addToFlyToQueue
  mapStore.addToFlyToQueue(locationName)
}

// P0-04: analyze — trigger Agent SSE request
function triggerAnalyze() {
  const cfg = config.value || store.compareConfig
  if (!cfg) return

  store.setLoading(true)
  timeoutRetryCount.value = 0

  // U-14: Set timeout (30 seconds for first attempt)
  if (analyzeTimeoutTimer.value) clearTimeout(analyzeTimeoutTimer.value)
  analyzeTimeoutTimer.value = setTimeout(() => {
    if (store.isLoading) {
      if (timeoutRetryCount.value === 0) {
        // First timeout: offer continue waiting (additional 30s)
        store.setTimeoutState()
      } else {
        // Second timeout: final error
        store.setError('分析超时(超过60秒未响应)')
      }
    }
  }, 30000)

  const prompt = `请对比分析 ${cfg.left.dynasty}(${cfg.left.chapterTitle}) 和 ${cfg.right.dynasty}(${cfg.right.chapterTitle}) 的差异`
  agentStore.sendMessage(prompt)
}

function retryAnalyze() {
  // U-14: If timed out once, add 30s more (total max 60s)
  if (store.compareState === 'timeout') {
    timeoutRetryCount.value = 1
    store.setLoading(true)
    if (analyzeTimeoutTimer.value) clearTimeout(analyzeTimeoutTimer.value)
    analyzeTimeoutTimer.value = setTimeout(() => {
      if (store.isLoading) {
        store.setError('分析超时(超过60秒未响应)')
      }
    }, 30000)
    // Resume waiting via the existing SSE connection
    return
  }
  // Regular retry from error state
  store.setLoading(true)
  triggerAnalyze()
}

function cancelAnalyze() {
  // U-14: Cancel resets to manual state and clears injected annotations
  if (analyzeTimeoutTimer.value) clearTimeout(analyzeTimeoutTimer.value)
  analyzeTimeoutTimer.value = null
  timeoutRetryCount.value = 0
  store.resetToManual()
  clearInjectedAnnotations()
}

// P1-06: Export logic
function handleExport(format: 'html' | 'pdf' | 'json') {
  exportModalVisible.value = false
  const cfg = store.compareConfig
  if (!cfg) return

  const content = generateExportContent(format)
  if (!content) return

  const ext = format === 'html' ? 'html' : format === 'json' ? 'json' : 'pdf'
  const mime = format === 'html' ? 'text/html' : format === 'json' ? 'application/json' : 'application/pdf'
  const blob = new Blob([content], { type: mime })
  const url = URL.createObjectURL(blob)
  const a = document.createElement('a')
  a.href = url
  a.download = `compare-report-${cfg.left.dynasty}-vs-${cfg.right.dynasty}.${ext}`
  a.click()
  URL.revokeObjectURL(url)
}

function generateExportContent(format: 'html' | 'pdf' | 'json'): string {
  const cfg = store.compareConfig
  if (!cfg) return ''

  if (format === 'json') {
    return JSON.stringify({
      config: cfg,
      elements: store.elements,
      locationChanges: store.locationChanges,
      summary: store.summary,
    }, null, 2)
  }

  if (format === 'html') {
    let html = `<!DOCTYPE html><html><head><meta charset="utf-8"><title>对比报告 - ${cfg.left.dynasty} vs ${cfg.right.dynasty}</title>
<style>body{font-family:"Noto Serif SC",serif;max-width:900px;margin:0 auto;padding:20px;line-height:1.8;color:#333}
h1{text-align:center;border-bottom:2px solid #ccc;padding-bottom:10px}
h3{color:#555;margin-top:24px}
.elem{border:1px solid #ddd;border-radius:6px;padding:12px;margin:8px 0}
.occ{background:#f9f9f9;padding:6px 10px;margin:4px 0;border-left:3px solid #aaa;border-radius:3px}
.cat{display:inline-block;padding:1px 8px;border-radius:10px;font-size:0.75rem;color:#fff;margin-right:6px}
.kf{background:#e8f4e8;padding:8px 12px;border-radius:4px;margin:6px 0}
</style></head><body>`
    html += `<h1>${cfg.left.dynasty} vs ${cfg.right.dynasty} 对比报告</h1>`
    if (store.summary) {
      html += `<p>${store.summary.overview}</p>`
    }
    for (const elem of store.elements) {
      html += `<div class="elem"><h3><span class="cat" style="background:${elem.color}">${elem.category}</span>${elem.elementName} - ${elem.evolution.changeType}</h3>`
      html += `<p><strong>${cfg.left.dynasty}:</strong> ${elem.leftOccurrences.length} / <strong>${cfg.right.dynasty}:</strong> ${elem.rightOccurrences.length}</p>`
      html += `</div>`
    }
    if (store.summary?.keyFindings?.length) {
      html += '<div><h3>关键发现</h3>'
      for (const kf of store.summary.keyFindings) {
        html += `<div class="kf">${kf}</div>`
      }
      html += '</div>'
    }
    html += '</body></html>'
    return html
  }

  return ''
}
</script>

<template>
  <div class="dynasty-compare-view">
    <CompareToolbar
      v-if="config"
      :left-title="`${config.left.dynasty || ''} · ${config.left.chapterTitle || ''}`"
      :right-title="`${config.right.dynasty || ''} · ${config.right.chapterTitle || ''}`"
      :left-ai-ready="store.compareState === 'ready'"
      :right-ai-ready="store.compareState === 'ready'"
      v-model:marginalia-expanded="margExpanded"
      @analyze="triggerAnalyze"
      @export="exportModalVisible = true"
    />
    <CompareToolbar
      v-else
      @analyze="triggerAnalyze"
      @export="exportModalVisible = true"
    />

    <!-- U-14: Loading banner driven by compareState -->
    <div v-if="store.compareState === 'analyzing'" class="dc-banner dc-loading-banner">
      <div class="dc-spinner" />
      <span>分析中... 预计10-30秒</span>
      <button class="dc-cancel-btn" @click="cancelAnalyze">取消</button>
    </div>

    <!-- U-14: Timeout banner -->
    <div v-else-if="store.compareState === 'timeout'" class="dc-banner dc-error-banner">
      <span>分析超时(超过30秒未响应)</span>
      <button @click="retryAnalyze">重试</button>
      <button @click="cancelAnalyze">取消</button>
    </div>

    <!-- U-14: Error banner -->
    <div v-else-if="store.compareState === 'error'" class="dc-banner dc-error-banner">
      <span>{{ store.error || '' }}</span>
      <button @click="retryAnalyze">重试</button>
    </div>

    <!-- Panels: show as soon as config is available (even without AI analysis) -->
    <div v-if="config" class="dc-split">
      <div class="dc-panel" :style="{ flex: `0 0 ${splitRatio * 100}%` }">
        <div class="dc-panel-body">
          <TextReaderView
            v-if="config.left.chapterNumber && leftTabId"
            :key="`dc-left-${config.left.chapterNumber}`"
            :tab-id="leftTabId"
            split-side="left"
            :linked-highlight-id="hoveredElementId"
            :hide-toolbar="true"
            :marginalia-expanded="margExpanded"
          />
        </div>
      </div>

      <SplitDivider @ratio-change="setSplitRatio" />

      <div class="dc-panel" :style="{ flex: `0 0 ${(1 - splitRatio) * 100}%` }">
        <div class="dc-panel-body">
          <TextReaderView
            v-if="config.right.chapterNumber && rightTabId"
            :key="`dc-right-${config.right.chapterNumber}`"
            :tab-id="rightTabId"
            split-side="right"
            :linked-highlight-id="hoveredElementId"
            :hide-toolbar="true"
            :marginalia-expanded="margExpanded"
          />
        </div>
      </div>
    </div>

    <!-- No config: show hint -->
    <div v-else class="dc-state dc-empty">
      <p>请从左侧文献目录右键选择两章进行跨朝代对比</p>
      <p class="dc-empty-hint">或使用地图图层面板选择两个朝代发起对比</p>
    </div>

    <!-- Element Evolution Drawer -->
    <ElementEvolutionDrawer
      @close="store.closeDrawer()"
      @view-on-map="handleViewOnMap"
    />

    <!-- Export Modal -->
    <CompareReportExport
      :visible="exportModalVisible"
      :config="store.compareConfig"
      :elements="store.elements"
      :location-changes="store.locationChanges"
      :summary="store.summary"
      @close="exportModalVisible = false"
      @export="handleExport"
    />
  </div>
</template>

<style scoped>
.dynasty-compare-view {
  display: flex;
  flex-direction: column;
  height: 100%;
  overflow: hidden;
  background: var(--color-bg-primary);
}

/* ---------- Status banners ---------- */
.dc-banner {
  display: flex;
  align-items: center;
  gap: var(--space-3);
  padding: var(--space-2) var(--space-4);
  font-size: 0.78rem;
  flex-shrink: 0;
  border-bottom: 1px solid var(--color-border);
  font-weight: 500;
}

.dc-loading-banner {
  background: var(--color-accent-light);
  color: var(--color-accent);
  border-bottom: 1px solid var(--color-accent);
}

.dc-error-banner {
  background: var(--color-error-light);
  color: var(--color-error);
  border-bottom: 1px solid var(--color-error);
}

.dc-error-banner button {
  padding: var(--space-1) var(--space-3);
  font-size: 0.7rem;
  background: var(--color-bg-primary);
  color: var(--color-text-primary);
  border: 1px solid var(--color-border);
  border-radius: var(--radius-sm);
  cursor: pointer;
  font-weight: 500;
  transition: background-color var(--dur-fast) var(--ease),
              color var(--dur-fast) var(--ease),
              border-color var(--dur-fast) var(--ease);
}

.dc-error-banner button:hover {
  background: var(--color-bg-hover);
  border-color: var(--color-border-dark);
}

.dc-spinner {
  width: 16px;
  height: 16px;
  border: 1px solid var(--color-border);
  border-top: 1px solid var(--color-accent);
  box-shadow: inset 0 0 0 1px var(--color-accent-ring);
  border-radius: var(--radius-full);
  animation: dc-spin 0.8s linear infinite;
  flex-shrink: 0;
}

@keyframes dc-spin {
  to { transform: rotate(360deg); }
}

.dc-cancel-btn {
  padding: var(--space-1) var(--space-3);
  font-size: 0.7rem;
  background: var(--color-bg-primary);
  color: var(--color-text-primary);
  border: 1px solid var(--color-border);
  border-radius: var(--radius-sm);
  cursor: pointer;
  margin-left: auto;
  font-weight: 500;
  transition: background-color var(--dur-fast) var(--ease),
              color var(--dur-fast) var(--ease),
              border-color var(--dur-fast) var(--ease);
}

.dc-cancel-btn:hover {
  background: var(--color-error-light);
  color: var(--color-error);
  border-color: var(--color-error);
}

/* ---------- Empty / state screens ---------- */
.dc-state {
  flex: 1;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  color: var(--color-text-tertiary);
  font-size: 0.85rem;
  gap: var(--space-2);
  padding: var(--space-8);
  text-align: center;
}

.dc-empty-hint {
  font-size: 0.75rem;
  color: var(--color-text-tertiary);
  background: var(--color-bg-tertiary);
  padding: var(--space-2) var(--space-3);
  border-radius: var(--radius-sm);
  margin-top: var(--space-2);
}

/* ---------- Split panels ---------- */
.dc-split {
  display: flex;
  flex: 1;
  overflow: hidden;
}

.dc-panel {
  min-width: 0;
  min-width: 300px;
  overflow: hidden;
  display: flex;
  flex-direction: column;
  background: var(--color-bg-primary);
  border: 1px solid var(--color-border);
  border-radius: 0;
  margin: 2px 4px;
  box-shadow: none;
}

.dc-panel:first-child {
  margin-right: 2px;
}

.dc-panel:last-child {
  margin-left: 2px;
}

.dc-panel-body {
  flex: 1;
  overflow: hidden;
}
</style>
