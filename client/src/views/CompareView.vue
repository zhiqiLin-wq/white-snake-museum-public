<script setup lang="ts">
import { computed } from 'vue'
import { useWorkspaceStore } from '@/stores/workspace'
import { useAgentStore } from '@/stores/agent'
import { useAnnotationStore } from '@/stores/annotations'
import type { ComparisonDataEvent, DiffChange, ComparisonHighlight } from '@/types/sse-events'

const workspaceStore = useWorkspaceStore()
const agentStore = useAgentStore()
const annotationStore = useAnnotationStore()

const tabData = computed(() => {
  const tab = workspaceStore.activeTab
  return tab?.data as unknown as ComparisonDataEvent | undefined
})

const left = computed(() => tabData.value?.left)
const right = computed(() => tabData.value?.right)
const diffSummary = computed(() => tabData.value?.diffSummary)
const source = computed(() => tabData.value?.source || 'agent')

const CHANGE_COLORS: Record<string, string> = {
  emergence: '#80C0A0',
  disappearance: '#C89898',
  substitution: '#C8C870',
  amplification: '#58A8A8',
  diminution: '#B8B8B0',
  transvaluation: '#C898B0',
  transplantation: '#9888C0',
}

function highlightSentimentColor(sentiment: string): string {
  switch (sentiment) {
    case 'positive': return 'rgba(128, 192, 160, 0.25)'
    case 'negative': return 'rgba(200, 152, 152, 0.25)'
    default: return 'rgba(88, 168, 168, 0.25)'
  }
}

function renderHighlightedText(text: string, highlights: ComparisonHighlight[]): string {
  if (!highlights.length) return escapeHtml(text)

  const sorted = [...highlights].sort((a, b) => a.span[0] - b.span[0])
  const segments: string[] = []
  let lastEnd = 0

  for (const hl of sorted) {
    const start = hl.span[0]
    const end = Math.min(hl.span[1], text.length)
    if (start > lastEnd) {
      segments.push(escapeHtml(text.slice(lastEnd, start)))
    }
    if (start < end) {
      const bg = highlightSentimentColor(hl.sentiment)
      segments.push(`<span style="background:${bg};border-bottom:1.5px solid ${hl.sentiment === 'positive' ? '#80C0A0' : hl.sentiment === 'negative' ? '#C89898' : '#58A8A8'};" title="${escapeHtml(hl.label)}">${escapeHtml(text.slice(start, end))}<sup style="font-size:0.55rem">*</sup></span>`)
    }
    lastEnd = end
  }
  if (lastEnd < text.length) {
    segments.push(escapeHtml(text.slice(lastEnd)))
  }
  return segments.join('')
}

function escapeHtml(text: string): string {
  return text
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
}

function askAboutChange(change: DiffChange) {
  agentStore.addContextTag({
    id: `ctx_${Date.now()}`,
    label: `${change.type}: ${change.description.slice(0, 30)}`,
    data: { type: 'comparison', change },
    sourceType: 'comparison',
  })
}

function convertToSplitView() {
  if (!left.value || !right.value) return
  // Open left chapter in a tab
  const leftTab = workspaceStore.openTab('text-reader', left.value.label, {
    chapterNumber: parseInt(left.value.chapterNumber, 10),
  })
  const rightTab = workspaceStore.openTab('text-reader', right.value.label, {
    chapterNumber: parseInt(right.value.chapterNumber, 10),
  })
  workspaceStore.enterSplitMode(leftTab.id, rightTab.id)
}
</script>

<template>
  <div class="compare-view">
    <div v-if="!left && !right" class="cv-empty">
      
    </div>

    <div v-else class="cv-content">
      <!-- Toolbar -->
      <div class="cv-toolbar">
        <span class="cv-source-badge">{{ source === 'split-view' ? 'Split View' : 'Agent' }}</span>
        <button class="cv-split-btn" @click="convertToSplitView">
           Split View 
        </button>
      </div>

      <!-- Side-by-side text panels -->
      <div class="cv-panels">
        <div v-if="left" class="cv-panel cv-left">
          <div class="cv-panel-header">{{ left.label }}</div>
          <div
            class="cv-panel-text"
            v-html="renderHighlightedText(left.text, left.highlights)"
          />
        </div>

        <div class="cv-divider" />

        <div v-if="right" class="cv-panel cv-right">
          <div class="cv-panel-header">{{ right.label }}</div>
          <div
            class="cv-panel-text"
            v-html="renderHighlightedText(right.text, right.highlights)"
          />
        </div>
      </div>

      <!-- AI diff summary -->
      <div v-if="diffSummary" class="cv-diff-summary">
        <h3 class="cv-diff-title">AI </h3>

        <div v-if="diffSummary.changes.length > 0" class="cv-changes">
          <div
            v-for="change in diffSummary.changes"
            :key="change.type + change.description"
            class="cv-change-item"
          >
            <span
              class="cv-change-badge"
              :style="{ background: CHANGE_COLORS[change.type] || '#666' }"
            >{{ change.type }}</span>
            <span class="cv-change-desc">{{ change.description }}</span>
            <button class="cv-ask-btn" @click="askAboutChange(change)">追问</button>
          </div>
        </div>

        <p v-if="diffSummary.narrative" class="cv-narrative">{{ diffSummary.narrative }}</p>
      </div>
    </div>
  </div>
</template>

<style scoped>
.compare-view {
  padding: var(--space-5);
  height: 100%;
  overflow-y: auto;
  background: var(--color-bg-primary);
}

.cv-empty {
  text-align: center;
  padding: var(--space-8);
  color: var(--color-text-secondary);
  font-size: 0.85rem;
}

.cv-toolbar {
  display: flex;
  align-items: center;
  gap: var(--space-2);
  margin-bottom: var(--space-3);
}

.cv-source-badge {
  font-size: 0.65rem;
  padding: var(--space-1) var(--space-2);
  border-radius: var(--radius-sm);
  background: var(--color-bg-tertiary);
  color: var(--color-text-secondary);
  border: 1px solid var(--color-border-light);
}

.cv-split-btn {
  padding: var(--space-1) var(--space-3);
  font-size: 0.7rem;
  background: var(--color-accent);
  color: #fff;
  border: 1px solid var(--color-accent);
  border-radius: var(--radius-sm);
  cursor: pointer;
  transition: background-color var(--dur) var(--ease),
              border-color var(--dur) var(--ease),
              box-shadow var(--dur) var(--ease);
}

.cv-split-btn:hover {
  background: var(--color-accent-hover);
  border-color: var(--color-accent-hover);
  box-shadow: var(--shadow-focus);
}

.cv-split-btn:active {
  background: var(--color-accent-active);
  border-color: var(--color-accent-active);
}

.cv-panels {
  display: flex;
  gap: 0;
  margin-bottom: var(--space-6);
  box-shadow: var(--shadow-sm);
  border-radius: var(--radius);
  overflow: hidden;
}

.cv-panel {
  flex: 1;
  min-width: 0;
  padding: var(--space-4);
  background: var(--color-bg-primary);
  border: 1px solid var(--color-border);
  border-radius: var(--radius);
}

.cv-left { border-right: none; border-top-right-radius: 0; border-bottom-right-radius: 0; }
.cv-right { border-left: none; border-top-left-radius: 0; border-bottom-left-radius: 0; }

.cv-divider {
  width: 1px;
  background: var(--color-border);
  flex-shrink: 0;
}

.cv-panel-header {
  font-size: 0.78rem;
  color: var(--color-text-secondary);
  margin-bottom: var(--space-3);
  font-weight: 500;
  padding-bottom: var(--space-2);
  border-bottom: 1px solid var(--color-border-light);
}

.cv-panel-text {
  font-size: 0.85rem;
  line-height: 1.8;
  color: var(--color-text-primary);
}

.cv-diff-summary {
  padding: var(--space-4);
  background: var(--color-bg-secondary);
  border: 1px solid var(--color-border);
  border-radius: var(--radius);
  box-shadow: var(--shadow-sm);
}

.cv-diff-title {
  font-size: 0.9rem;
  color: var(--color-text-primary);
  margin: 0 0 var(--space-3);
  font-weight: 600;
  padding-bottom: var(--space-2);
  border-bottom: 1px solid var(--color-border-light);
}

.cv-changes {
  display: flex;
  flex-direction: column;
  gap: var(--space-2);
  margin-bottom: var(--space-3);
}

.cv-change-item {
  display: flex;
  align-items: center;
  gap: var(--space-2);
  padding: var(--space-2) var(--space-3);
  background: var(--color-bg-primary);
  border: 1px solid var(--color-border);
  border-radius: var(--radius-sm);
  transition: background-color var(--dur-fast) var(--ease),
              border-color var(--dur-fast) var(--ease);
}

.cv-change-item:hover {
  background: var(--color-bg-hover);
  border-color: var(--color-border-dark);
}

.cv-change-badge {
  padding: var(--space-1) var(--space-2);
  border-radius: var(--radius-sm);
  font-size: 0.65rem;
  color: #fff;
  flex-shrink: 0;
  font-weight: 500;
}

.cv-change-desc {
  font-size: 0.78rem;
  color: var(--color-text-primary);
  flex: 1;
  line-height: 1.5;
}

.cv-ask-btn {
  padding: var(--space-1) var(--space-2);
  font-size: 0.68rem;
  background: var(--color-bg-primary);
  color: var(--color-text-secondary);
  border: 1px solid var(--color-border);
  border-radius: var(--radius-sm);
  cursor: pointer;
  flex-shrink: 0;
  transition: background-color var(--dur-fast) var(--ease),
              color var(--dur-fast) var(--ease),
              border-color var(--dur-fast) var(--ease);
}

.cv-ask-btn:hover {
  background: var(--color-accent-light);
  color: var(--color-accent);
  border-color: var(--color-accent);
}

.cv-narrative {
  font-size: 0.8rem;
  color: var(--color-text-secondary);
  line-height: 1.6;
  padding: var(--space-3);
  background: var(--color-bg-primary);
  border: 1px solid var(--color-border-light);
  border-radius: var(--radius-sm);
}
</style>
