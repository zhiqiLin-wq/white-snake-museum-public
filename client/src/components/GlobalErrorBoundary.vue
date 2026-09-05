<script setup lang="ts">
import { ref, onErrorCaptured } from 'vue'

const hasError = ref(false)
const errorMessage = ref('')

onErrorCaptured((err: Error, instance, info) => {
  console.error('[GlobalErrorBoundary] 捕获到错误:', err)
  // v19.2 调试增强：输出组件名与生命周期阶段，定位渲染错误源头
  const compName = instance && (instance as { type?: { name?: string; __name?: string } }).type
    ? ((instance as { type?: { name?: string; __name?: string } }).type?.__name
      || (instance as { type?: { name?: string; __name?: string } }).type?.name
      || 'anonymous')
    : 'none'
  console.warn('[EB-DEBUG] component=', compName, 'info=', info, 'stack=', err.stack)
  hasError.value = true
  errorMessage.value = err.message || ''
  return false // Prevent propagation
})

function refresh() {
  window.location.reload()
}
</script>

<template>
  <div v-if="hasError" class="error-boundary">
    <div class="eb-card">
      <h2>应用出现错误</h2>
      <p>{{ errorMessage }}</p>
      <button @click="refresh">刷新页面</button>
    </div>
  </div>
  <slot v-else />
</template>

<style scoped>
.error-boundary {
  display: flex;
  align-items: center;
  justify-content: center;
  height: 100vh;
  background: var(--color-bg-primary);
}

.eb-card {
  text-align: center;
  padding: 40px;
  background: var(--color-bg-secondary);
  border: 1px solid var(--color-border);
  border-radius: 8px;
  max-width: 400px;
}

.eb-card h2 {
  font-family: var(--font-sans);
  font-size: 1.1rem;
  color: var(--color-verdict-refuted);
  margin: 0 0 8px;
}

.eb-card p {
  font-size: 0.82rem;
  color: var(--color-text-secondary);
  margin-bottom: 16px;
}

.eb-card button {
  padding: 8px 20px;
  font-size: 0.82rem;
  background: var(--color-phase-active);
  color: #fff;
  border: none;
  border-radius: 4px;
  cursor: pointer;
}
</style>
