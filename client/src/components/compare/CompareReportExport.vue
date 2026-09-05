<script setup lang="ts">
import { ref } from 'vue'
import type { DynastyCompareConfig, ElementEvolution, LocationEvolution } from '@/types/dynasty-compare'

defineProps<{
  config: DynastyCompareConfig | null
  elements: ElementEvolution[]
  locationChanges: LocationEvolution[]
  summary: { overview: string; keyFindings: string[] } | null
  visible: boolean
}>()

const emit = defineEmits<{
  close: []
  export: [format: 'html' | 'pdf' | 'json']
}>()

const format = ref<'html' | 'pdf' | 'json'>('html')
</script>

<template>
  <Teleport to="body">
    <div v-if="visible" class="report-export-backdrop" @click="emit('close')" />
    <div v-if="visible" class="report-export-modal">
      <div class="re-header">导出对比报告</div>

      <div class="re-section">
        <div class="re-label">选择格式</div>
        <div class="re-options">
          <label class="re-option">
            <input type="radio" v-model="format" value="html" /> HTML
          </label>
          <label class="re-option">
            <input type="radio" v-model="format" value="pdf" /> PDF
          </label>
          <label class="re-option">
            <input type="radio" v-model="format" value="json" /> JSON
          </label>
        </div>
      </div>

      <div class="re-actions">
        <button class="re-btn-cancel" @click="emit('close')">取消</button>
        <button class="re-btn-export" @click="emit('export', format)">导出</button>
      </div>
    </div>
  </Teleport>
</template>

<style scoped>
.report-export-backdrop {
  position: fixed;
  inset: 0;
  z-index: 9100;
  background: rgba(26, 26, 26, 0.35);
}

.report-export-modal {
  position: fixed;
  top: 50%;
  left: 50%;
  transform: translate(-50%, -50%);
  z-index: 9101;
  background: var(--color-bg-secondary, #F7F8FA);
  border: 1px solid var(--color-border, #E5E7EB);
  border-radius: var(--radius, 8px);
  padding: 16px;
  min-width: 300px;
  box-shadow: var(--shadow-md, 0 8px 24px rgba(0, 0, 0, 0.12));
}

.re-header {
  font-size: 0.9rem;
  color: var(--color-text-primary, #1A1A1A);
  margin-bottom: 12px;
  font-weight: 500;
}

.re-section { margin-bottom: 10px; }

.re-label {
  font-size: 0.72rem;
  color: var(--color-text-secondary, #4B5563);
  margin-bottom: 4px;
}

.re-options { display: flex; gap: 10px; }

.re-option {
  display: flex;
  align-items: center;
  gap: 4px;
  font-size: 0.78rem;
  color: var(--color-text-primary, #1A1A1A);
  cursor: pointer;
  padding: 4px 6px;
  border-radius: var(--radius-sm, 4px);
  transition: background var(--dur-fast, 120ms) var(--ease, cubic-bezier(.22,.61,.36,1));
}

.re-option:hover {
  background: var(--color-bg-hover, #F2F4F7);
}

.re-actions {
  display: flex;
  justify-content: flex-end;
  gap: 8px;
  margin-top: 14px;
}

.re-btn-cancel {
  padding: 5px 14px;
  font-size: 0.75rem;
  background: var(--color-bg-tertiary, #F0F2F5);
  color: var(--color-text-primary, #1A1A1A);
  border: 1px solid var(--color-border, #E5E7EB);
  border-radius: var(--radius-sm, 4px);
  cursor: pointer;
  transition: background var(--dur-fast, 120ms) var(--ease, cubic-bezier(.22,.61,.36,1)),
              border-color var(--dur-fast, 120ms) var(--ease, cubic-bezier(.22,.61,.36,1));
}

.re-btn-cancel:hover {
  background: var(--color-bg-hover, #F2F4F7);
  border-color: var(--color-border-dark, #D1D5DB);
}

.re-btn-export {
  padding: 5px 14px;
  font-size: 0.75rem;
  background: var(--color-accent, #5B9BD5);
  color: #FFFFFF;
  border: 1px solid var(--color-accent, #5B9BD5);
  border-radius: var(--radius-sm, 4px);
  cursor: pointer;
  transition: background var(--dur-fast, 120ms) var(--ease, cubic-bezier(.22,.61,.36,1)),
              border-color var(--dur-fast, 120ms) var(--ease, cubic-bezier(.22,.61,.36,1));
}

.re-btn-export:hover {
  background: var(--color-accent-hover, #4A8BC5);
  border-color: var(--color-accent-hover, #4A8BC5);
}
</style>
