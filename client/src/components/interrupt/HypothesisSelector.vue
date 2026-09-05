<script setup lang="ts">
import InterruptTimeoutOverlay from './InterruptTimeoutOverlay.vue'

export interface HypothesisItem {
  id: string
  statement: string
  confidence: number
  reasoning: string
  selected?: boolean
}

defineProps<{
  hypotheses: HypothesisItem[]
  allowMultiSelect: boolean
  countdownRemaining: number
  isTimedOut: boolean
}>()

const emit = defineEmits<{
  toggle: [id: string]
  confirm: []
  restart: []
}>()
</script>

<template>
  <div class="ew-interrupt" style="position: relative">
    <InterruptTimeoutOverlay
      :visible="true"
      :countdown-remaining="countdownRemaining"
      :is-timed-out="isTimedOut"
      @restart="emit('restart')"
    />

    <template v-if="!isTimedOut">
      <h3>选择演化假设</h3>
      <p class="ew-instruction">AI已完成假设生成，请选择一条假设进行验证:</p>

      <div class="ew-hypothesis-list">
        <div
          v-for="h in hypotheses"
          :key="h.id"
          class="ew-hypothesis-item"
          :class="{ selected: h.selected }"
          @click="emit('toggle', h.id)"
        >
          <div class="ew-hypothesis-id">{{ h.id }}</div>
          <div class="ew-hypothesis-statement">{{ h.statement }}</div>
          <div class="ew-hypothesis-confidence">
            置信度: {{ ''.padStart(h.confidence, '★') }}{{ ''.padStart(5 - h.confidence, '☆') }} ({{ h.confidence }}/5)
          </div>
          <div class="ew-hypothesis-reasoning">{{ h.reasoning }}</div>
        </div>
      </div>

      <div class="ew-interrupt-actions">
        <button class="ew-confirm-btn" @click="emit('confirm')">
          {{ allowMultiSelect ? '确认所选假设' : '确认此假设' }}
        </button>
      </div>
    </template>
  </div>
</template>

<style scoped>
.ew-interrupt { padding: 12px; }
.ew-interrupt h3 {
  font-size: 0.85rem;
  margin: 0 0 4px;
  color: var(--color-text-primary);
  font-weight: 500;
}
.ew-instruction { font-size: 0.7rem; color: var(--color-text-tertiary); margin-bottom: 8px; }
.ew-hypothesis-list { display: flex; flex-direction: column; gap: 6px; }
.ew-hypothesis-item {
  padding: 8px 10px;
  border: 1px solid var(--color-border);
  border-radius: var(--radius);
  cursor: pointer;
  background: var(--color-bg-primary);
  transition: border-color var(--dur-fast) var(--ease), background-color var(--dur-fast) var(--ease), box-shadow var(--dur-fast) var(--ease);
}
.ew-hypothesis-item:hover {
  border-color: var(--color-accent);
  background: var(--color-bg-secondary);
  box-shadow: var(--shadow-sm);
}
.ew-hypothesis-item.selected {
  border-color: var(--color-accent);
  background: var(--color-accent-light);
}
.ew-hypothesis-id {
  font-family: var(--font-mono);
  font-size: 0.6rem;
  color: var(--color-text-tertiary);
}
.ew-hypothesis-statement {
  font-size: 0.75rem;
  font-weight: 500;
  margin: 2px 0;
  color: var(--color-text-primary);
}
.ew-hypothesis-confidence {
  font-size: 0.65rem;
  color: var(--color-accent);
}
.ew-hypothesis-reasoning {
  font-size: 0.68rem;
  color: var(--color-text-secondary);
  margin-top: 2px;
  line-height: 1.5;
}
.ew-interrupt-actions { margin-top: 12px; display: flex; justify-content: flex-end; }
.ew-confirm-btn {
  padding: 6px 16px;
  font-size: 0.75rem;
  background: var(--color-verdict-supported);
  color: #fff;
  border: none;
  border-radius: var(--radius);
  cursor: pointer;
  transition: background-color var(--dur-fast) var(--ease), box-shadow var(--dur-fast) var(--ease);
}
.ew-confirm-btn:hover {
  background: #4f9770;
  box-shadow: var(--shadow-sm);
}
</style>
