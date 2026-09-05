<script setup lang="ts">
defineProps<{
  summary: string
  resultRefs?: { label: string; type: string; data: Record<string, unknown> }[]
}>()

const emit = defineEmits<{
  'navigate': [type: string, data: Record<string, unknown>]
}>()
</script>

<template>
  <div class="workflow-card">
    <div class="wc-header">
      <span class="wc-icon">OK</span>
      <span>任务完成</span>
    </div>
    <div class="wc-summary">{{ summary }}</div>
    <div v-if="resultRefs && resultRefs.length > 0" class="wc-actions">
      <button
        v-for="ref in resultRefs"
        :key="ref.label"
        class="wc-action-btn"
        @click="emit('navigate', ref.type, ref.data)"
      >
        {{ ref.label }}
      </button>
    </div>
  </div>
</template>

<style scoped>
.workflow-card {
  margin: 8px 0;
  padding: 10px 12px;
  border: 1px solid var(--color-border);
  border-left: 3px solid var(--color-verdict-supported);
  border-radius: var(--radius);
  background: var(--color-bg-primary);
  box-shadow: var(--shadow-sm);
}

.wc-header {
  display: flex;
  align-items: center;
  gap: 6px;
  font-family: var(--font-sans);
  font-size: 0.82rem;
  color: var(--color-verdict-supported);
  font-weight: 500;
}

.wc-icon {
  font-family: var(--font-mono);
  font-size: 0.7rem;
  width: 18px;
  height: 18px;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  background: var(--color-verdict-supported);
  color: #fff;
  border-radius: 50%;
}

.wc-summary {
  margin-top: 6px;
  font-size: 0.78rem;
  color: var(--color-text-primary);
  line-height: 1.5;
}

.wc-actions {
  margin-top: 8px;
  display: flex;
  gap: 6px;
  flex-wrap: wrap;
}

.wc-action-btn {
  padding: 4px 10px;
  font-size: 0.72rem;
  background: var(--color-bg-tertiary);
  color: var(--color-text-primary);
  border: 1px solid var(--color-border);
  border-radius: var(--radius-sm);
  cursor: pointer;
  transition: background-color var(--dur-fast) var(--ease), border-color var(--dur-fast) var(--ease), color var(--dur-fast) var(--ease);
}

.wc-action-btn:hover {
  background: var(--color-accent-light);
  border-color: var(--color-accent);
  color: var(--color-accent);
}
</style>
