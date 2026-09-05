<script setup lang="ts">
import { ref, watch, onUnmounted } from 'vue'

const props = withDefaults(defineProps<{
  message: string
  type?: 'info' | 'success' | 'warning' | 'error'
  duration?: number
}>(), {
  type: 'info',
  duration: 4000,
})

const emit = defineEmits<{ close: [] }>()

const visible = ref(false)
let timer: ReturnType<typeof setTimeout> | null = null

function startTimer() {
  clear()
  if (props.duration > 0) {
    timer = setTimeout(() => { visible.value = false }, props.duration)
  }
}
function clear() {
  if (timer) { clearTimeout(timer); timer = null }
}

watch(() => props.message, (newVal) => {
  if (newVal) { visible.value = true; startTimer() }
}, { immediate: true })

watch(visible, (val) => {
  if (!val) { clear(); setTimeout(() => emit('close'), 240) }
})

onUnmounted(clear)
function closeToast() { visible.value = false }
</script>

<template>
  <Transition name="slide-fade-rtl">
    <div
      v-if="visible && message"
      class="toast"
      :class="'toast-' + type"
      @click="closeToast"
    >
      <span class="toast-icon" v-if="type !== 'info'" aria-hidden="true">
        <svg v-if="type === 'success'" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><polyline points="20 6 9 17 4 12"/></svg>
        <svg v-else-if="type === 'warning'" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><path d="M10.29 3.86 1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"/><line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/></svg>
        <svg v-else-if="type === 'error'" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><line x1="15" y1="9" x2="9" y2="15"/><line x1="9" y1="9" x2="15" y2="15"/></svg>
      </span>
      <span v-else class="toast-icon toast-icon-info" aria-hidden="true">
        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><line x1="12" y1="16" x2="12" y2="12"/><line x1="12" y1="8" x2="12.01" y2="8"/></svg>
      </span>
      <span class="toast-message">{{ message }}</span>
      <button class="toast-close" @click.stop="closeToast" aria-label="关闭">
        <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round"><line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/></svg>
      </button>
    </div>
  </Transition>
</template>

<style scoped>
.toast {
  position: fixed;
  top: 18px;
  right: 18px;
  z-index: 10000;
  display: flex;
  align-items: center;
  gap: 10px;
  padding: 11px 14px 11px 12px;
  border-radius: var(--radius-lg);
  font-size: 0.82rem;
  line-height: 1.45;
  max-width: 400px;
  cursor: pointer;
  border: 1px solid var(--color-border);
  background: var(--color-bg-primary);
  color: var(--color-text-primary);
  box-shadow: var(--shadow-lg);
  backdrop-filter: blur(6px);
  -webkit-backdrop-filter: blur(6px);
}

.toast-icon {
  flex-shrink: 0;
  width: 22px; height: 22px;
  border-radius: var(--radius-full);
  display: inline-flex; align-items: center; justify-content: center;
}
.toast-icon-info {
  color: var(--color-accent);
  background: var(--color-accent-light);
}
.toast-success .toast-icon { color: var(--color-success); background: var(--color-success-light); }
.toast-warning .toast-icon { color: var(--color-warning); background: var(--color-warning-light); }
.toast-error   .toast-icon { color: var(--color-error);   background: var(--color-error-light); }

.toast-message { flex: 1; min-width: 0; }

.toast-close {
  background: none; border: none;
  width: 22px; height: 22px;
  border-radius: var(--radius-sm);
  display: inline-flex; align-items: center; justify-content: center;
  color: var(--color-text-tertiary);
  cursor: pointer;
  transition: background var(--dur-fast) var(--ease), color var(--dur-fast) var(--ease);
}
.toast-close:hover { background: var(--color-bg-hover); color: var(--color-text-primary); }
</style>
