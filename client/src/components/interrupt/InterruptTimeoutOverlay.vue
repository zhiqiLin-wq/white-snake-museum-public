<script setup lang="ts">
defineProps<{
  visible: boolean
  countdownRemaining: number
  isTimedOut: boolean
}>()

const emit = defineEmits<{
  restart: []
}>()

function formatCountdown(seconds: number): string {
  const m = Math.floor(seconds / 60)
  const s = seconds % 60
  return `${m}:${s.toString().padStart(2, '0')}`
}
</script>

<template>
  <div v-if="visible && !isTimedOut" class="ew-countdown">
    剩余时间: {{ formatCountdown(countdownRemaining) }}
    <span v-if="countdownRemaining < 60" class="ew-countdown-warn">即将超时!</span>
  </div>
  <div v-if="visible && isTimedOut" class="ew-timeout-overlay">
    <div class="ew-timeout-card">
      <h3>操作已超时</h3>
      <p>超过10分钟未响应，中断已自动失效</p>
      <button class="ew-restart-btn" @click="emit('restart')">重新开始</button>
    </div>
  </div>
</template>

<style scoped>
.ew-countdown {
  position: absolute;
  top: 8px;
  right: 8px;
  z-index: 5;
  padding: 3px 8px;
  font-size: 0.7rem;
  color: var(--color-text-secondary);
  background: var(--color-bg-tertiary);
  border: 1px solid var(--color-border);
  border-radius: var(--radius-sm);
  pointer-events: none;
  font-family: var(--font-mono);
}

.ew-countdown-warn {
  color: var(--color-verdict-refuted);
  margin-left: 4px;
  animation: pulse 1s var(--ease) infinite;
}

@keyframes pulse {
  0%, 100% { opacity: 1; }
  50% { opacity: 0.5; }
}

.ew-timeout-overlay {
  position: absolute;
  inset: 0;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  background: rgba(0, 0, 0, 0.25);
  backdrop-filter: blur(2px);
  z-index: 10;
  border-radius: var(--radius-lg);
}

.ew-timeout-card {
  background: var(--color-bg-primary);
  border: 1px solid var(--color-border);
  border-radius: var(--radius-lg);
  padding: 24px;
  text-align: center;
  max-width: 340px;
  box-shadow: var(--shadow-lg);
}

.ew-timeout-card h3 {
  margin: 0 0 8px;
  font-size: 0.9rem;
  color: var(--color-text-primary);
  font-weight: 500;
}

.ew-timeout-card p {
  margin: 0 0 12px;
  font-size: 0.75rem;
  color: var(--color-text-secondary);
}

.ew-restart-btn {
  padding: 6px 16px;
  font-size: 0.75rem;
  background: var(--color-accent);
  color: #fff;
  border: none;
  border-radius: var(--radius);
  cursor: pointer;
  transition: background-color var(--dur-fast) var(--ease), box-shadow var(--dur-fast) var(--ease);
}

.ew-restart-btn:hover {
  background: var(--color-accent-hover);
  box-shadow: var(--shadow-sm);
}
</style>
