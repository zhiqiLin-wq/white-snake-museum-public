<script setup lang="ts">
import InterruptTimeoutOverlay from './InterruptTimeoutOverlay.vue'

export interface DecisionOption {
  id: string
  label: string
  description: string
  type: string
}

defineProps<{
  options: DecisionOption[]
  summary: string
  totalClaims: number
  insufficientCount: number
  selectedOption: string | null
  countdownRemaining: number
  isTimedOut: boolean
}>()

const emit = defineEmits<{
  select: [id: string]
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
      <h3>决定后续方向</h3>
      <p class="ew-decision-summary">{{ summary }}</p>
      <p v-if="totalClaims > 0" class="ew-decision-stats">
         总断言 {{ totalClaims }} 条，其中 {{ insufficientCount }} 条证据不足
      </p>

      <!-- U-13: Decision options rendered as 2x2 grid cards -->
      <div class="ew-decision-grid">
        <div
          v-for="opt in options"
          :key="opt.id"
          class="ew-decision-card"
          :class="{ selected: selectedOption === opt.id }"
          @click="emit('select', opt.id)"
        >
          <div class="ew-decision-label">{{ opt.label }}</div>
          <div class="ew-decision-desc">{{ opt.description }}</div>
        </div>
      </div>

      <div class="ew-interrupt-actions">
        <button
          class="ew-confirm-btn"
          :disabled="!selectedOption"
          @click="emit('confirm')"
        >
          确认选择
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
.ew-decision-summary {
  font-size: 0.72rem;
  color: var(--color-text-secondary);
  margin-bottom: 4px;
  line-height: 1.5;
}
.ew-decision-stats {
  font-size: 0.65rem;
  color: var(--color-text-tertiary);
  margin-bottom: 8px;
}

/* U-13: 2x2 grid layout */
.ew-decision-grid {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 6px;
}

.ew-decision-card {
  padding: 10px;
  border: 1px solid var(--color-border);
  border-radius: var(--radius);
  cursor: pointer;
  background: var(--color-bg-primary);
  transition: border-color var(--dur-fast) var(--ease), background-color var(--dur-fast) var(--ease), box-shadow var(--dur-fast) var(--ease);
}

.ew-decision-card:hover {
  border-color: var(--color-accent);
  background: var(--color-bg-secondary);
  box-shadow: var(--shadow-sm);
}

.ew-decision-card.selected {
  border-color: var(--color-accent);
  background: var(--color-accent-light);
  box-shadow: var(--shadow-sm);
}

.ew-decision-label {
  font-size: 0.78rem;
  font-weight: 500;
  color: var(--color-text-primary);
  margin-bottom: 4px;
}

.ew-decision-desc {
  font-size: 0.65rem;
  color: var(--color-text-secondary);
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
  transition: background-color var(--dur-fast) var(--ease), box-shadow var(--dur-fast) var(--ease), opacity var(--dur-fast) var(--ease);
}
.ew-confirm-btn:hover:not(:disabled) {
  background: #4f9770;
  box-shadow: var(--shadow-sm);
}
.ew-confirm-btn:disabled { opacity: 0.4; cursor: not-allowed; }
</style>
