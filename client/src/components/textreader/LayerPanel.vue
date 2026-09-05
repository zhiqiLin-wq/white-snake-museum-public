<script setup lang="ts">
import { computed } from 'vue'
import { useAnnotationStore } from '@/stores/annotations'
import type { AnnotationVisibility, AnnotationCategory } from '@/types/annotation'

const props = defineProps<{
  // U-10: Chapter number for adopt-all button
  chapterNumber?: number
  agentAnnotationCount?: number
}>()

const annotationStore = useAnnotationStore()

const layers: { key: 'user' | 'agent'; label: string }[] = [
  { key: 'user', label: '' },
  { key: 'agent', label: 'Agent ' },
]

// U-10: Adopt all handler
function onAdoptAndSaveAll() {
  if (props.chapterNumber !== undefined) {
    annotationStore.saveAndAdoptAllAgentAnnotations(props.chapterNumber)
  }
}

const categories: { key: keyof AnnotationVisibility['user']; label: string; color: string }[] = [
  { key: 'person', label: '', color: 'var(--color-annotation-person)' },
  { key: 'location', label: '', color: 'var(--color-annotation-location)' },
  { key: 'event', label: '', color: 'var(--color-annotation-event)' },
  { key: 'term', label: '', color: 'var(--color-annotation-term)' },
  { key: 'motif', label: '', color: 'var(--color-annotation-motif)' },
  { key: 'custom', label: '', color: 'var(--color-annotation-custom)' },
]

function isChecked(layer: 'user' | 'agent', cat: keyof AnnotationVisibility['user']): boolean {
  return annotationStore.visibility[layer][cat]
}

function toggle(layer: 'user' | 'agent', cat: keyof AnnotationVisibility['user']) {
  annotationStore.toggleVisibility(layer, cat)
}
</script>

<template>
  <div class="layer-panel">
    <div class="lp-header">
      <span class="lp-title">标注图层</span>
    </div>

    <div v-for="layer in layers" :key="layer.key" class="lp-layer">
      <div class="lp-layer-title">{{ layer.label }}</div>
      <label
        v-for="cat in categories"
        :key="`${layer.key}-${cat.key}`"
        class="lp-cat-item"
      >
        <input
          type="checkbox"
          :checked="isChecked(layer.key, cat.key)"
          @change="toggle(layer.key, cat.key)"
        />
        <span class="lp-cat-dot" :style="{ background: cat.color }" />
        {{ cat.label }}
      </label>
    </div>

    <div v-if="agentAnnotationCount && agentAnnotationCount > 0" class="lp-adopt-section">
      <button class="lp-adopt-all-btn" @click="onAdoptAndSaveAll">
         ({{ agentAnnotationCount }})
      </button>
    </div>

    <div class="lp-divider" />

    <div class="lp-batch-actions">
      <button class="lp-batch-btn" @click="annotationStore.showAll()">显示全部</button>
      <button class="lp-batch-btn" @click="annotationStore.hideAll()">全部隐藏</button>
      <button class="lp-batch-btn" @click="annotationStore.showUserOnly()">仅用户标注</button>
      <button class="lp-batch-btn" @click="annotationStore.showAgentOnly()">仅 Agent 标注</button>
    </div>
  </div>
</template>

<style scoped>
.layer-panel {
  background: var(--color-bg-primary);
  border: 1px solid var(--color-border);
  border-radius: var(--radius);
  padding: 12px;
  min-width: 200px;
  max-width: 260px;
  box-shadow: var(--shadow-sm);
}

.lp-header {
  margin-bottom: 8px;
}

.lp-title {
  font-size: 0.85rem;
  color: var(--color-text-primary);
  font-weight: 500;
}

.lp-layer {
  margin-bottom: 10px;
}

.lp-layer-title {
  font-size: 0.72rem;
  color: var(--color-text-tertiary);
  margin-bottom: 4px;
  letter-spacing: 0.02em;
}

.lp-cat-item {
  display: flex;
  align-items: center;
  gap: 6px;
  padding: 3px 4px;
  font-size: 0.75rem;
  color: var(--color-text-primary);
  cursor: pointer;
  border-radius: var(--radius-sm);
  transition: background-color var(--dur-fast) var(--ease);
}

.lp-cat-item:hover {
  background: var(--color-bg-hover);
}

.lp-cat-item input {
  accent-color: var(--color-accent);
}

.lp-cat-dot {
  width: 8px;
  height: 8px;
  border-radius: 50%;
  flex-shrink: 0;
  box-shadow: 0 0 0 1px var(--color-border-light);
}

.lp-divider {
  height: 1px;
  background: var(--color-border-light);
  margin: 8px 0;
}

.lp-adopt-section {
  margin-bottom: 4px;
}

.lp-adopt-all-btn {
  width: 100%;
  padding: 5px 10px;
  font-size: 0.7rem;
  background: var(--color-verdict-supported);
  color: #fff;
  border: none;
  border-radius: var(--radius);
  cursor: pointer;
  transition: background-color var(--dur-fast) var(--ease), box-shadow var(--dur-fast) var(--ease);
}

.lp-adopt-all-btn:hover {
  background: #4f9770;
  box-shadow: var(--shadow-sm);
}

.lp-batch-actions {
  display: flex;
  flex-wrap: wrap;
  gap: 4px;
}

.lp-batch-btn {
  padding: 3px 8px;
  font-size: 0.65rem;
  background: var(--color-bg-tertiary);
  color: var(--color-text-secondary);
  border: 1px solid var(--color-border);
  border-radius: var(--radius-sm);
  cursor: pointer;
  transition: background-color var(--dur-fast) var(--ease), border-color var(--dur-fast) var(--ease), color var(--dur-fast) var(--ease);
}

.lp-batch-btn:hover {
  color: var(--color-accent);
  border-color: var(--color-accent);
  background: var(--color-accent-light);
}
</style>
