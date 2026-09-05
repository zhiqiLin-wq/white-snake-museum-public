<script setup lang="ts">
import { ref, computed } from 'vue'
import type { Marginalia } from '@/types/annotation'

const props = defineProps<{
  marginalia: Marginalia
  editing: boolean
}>()

const emit = defineEmits<{
  click: [id: string]
  edit: [id: string]
  delete: [id: string]
}>()

const expanded = ref(false)

function toggleExpand() {
  expanded.value = !expanded.value
}

const previewLines = computed(() => {
  const lines = props.marginalia.content.split('\n')
  return lines.slice(0, 2).join('\n')
})

const hasMore = computed(() => {
  const lines = props.marginalia.content.split('\n')
  return lines.length > 2 || props.marginalia.content.length > 100
})
</script>

<template>
  <div
    class="marginalia-bubble"
    :class="{ expanded, editing }"
    :style="{ borderLeftColor: marginalia.color || 'var(--marginalia-border)', borderLeftWidth: '3px' }"
    @click="toggleExpand"
  >
    <div class="mb-content">
      <template v-if="expanded">
        {{ marginalia.content }}
      </template>
      <template v-else>
        {{ previewLines }}
        <span v-if="hasMore" class="mb-more">...</span>
      </template>
    </div>
    <div class="mb-meta">
      <span class="mb-time">{{ new Date(marginalia.updatedAt).toLocaleString('zh-CN', { month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' }) }}</span>
      <div class="mb-actions">
        <button class="mb-btn" @click.stop="emit('edit', marginalia.id)">编辑</button>
        <button class="mb-btn mb-btn-del" @click.stop="emit('delete', marginalia.id)">删除</button>
      </div>
    </div>
  </div>
</template>

<style scoped>
.marginalia-bubble {
  background: var(--marginalia-bg);
  border: 1px solid var(--marginalia-border);
  border-radius: var(--radius);
  padding: var(--marginalia-padding);
  max-width: var(--marginalia-max-width);
  font-size: var(--marginalia-font-size);
  color: var(--color-text-primary);
  cursor: pointer;
  transition: border-color var(--dur-fast) var(--ease), background-color var(--dur-fast) var(--ease), box-shadow var(--dur-fast) var(--ease);
  margin-bottom: 6px;
}

.marginalia-bubble:hover {
  border-color: var(--color-accent);
  background: var(--color-bg-secondary);
  box-shadow: var(--shadow-sm);
}

.marginalia-bubble.editing {
  border-color: var(--color-accent);
  box-shadow: var(--shadow-focus);
}

.marginalia-bubble.expanded {
  max-width: 280px;
}

.mb-content {
  line-height: 1.5;
  white-space: pre-wrap;
  word-break: break-word;
}

.mb-more {
  color: var(--color-accent);
  font-size: 0.65rem;
}

.mb-meta {
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin-top: 4px;
}

.mb-time {
  font-size: 0.6rem;
  color: var(--color-text-tertiary);
}

.mb-actions {
  display: flex;
  gap: 4px;
}

.mb-btn {
  padding: 1px 6px;
  font-size: 0.6rem;
  background: var(--color-bg-tertiary);
  color: var(--color-text-secondary);
  border: 1px solid var(--color-border);
  border-radius: var(--radius-sm);
  cursor: pointer;
  transition: background-color var(--dur-fast) var(--ease), border-color var(--dur-fast) var(--ease), color var(--dur-fast) var(--ease);
}

.mb-btn:hover {
  color: var(--color-accent);
  border-color: var(--color-accent);
  background: var(--color-accent-light);
}

.mb-btn-del:hover {
  color: var(--color-verdict-refuted);
  border-color: var(--color-verdict-refuted);
  background: var(--color-bg-tertiary);
}
</style>
