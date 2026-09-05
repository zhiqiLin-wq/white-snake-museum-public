<script setup lang="ts">
import { watch } from 'vue'

const props = withDefaults(defineProps<{
  visible: boolean
  title?: string
  width?: string
}>(), {
  width: '480px',
})

const emit = defineEmits<{
  'update:visible': [value: boolean]
  close: []
}>()

function closeModal() {
  emit('update:visible', false)
  emit('close')
}
function onOverlayClick(e: MouseEvent) {
  if ((e.target as HTMLElement).classList.contains('modal-overlay')) closeModal()
}
function onKeydown(e: KeyboardEvent) {
  if (e.key === 'Escape') closeModal()
}

watch(() => props.visible, (val) => {
  if (val) document.addEventListener('keydown', onKeydown)
  else document.removeEventListener('keydown', onKeydown)
})
</script>

<template>
  <Transition name="fade">
    <div v-if="visible" class="modal-overlay" @click="onOverlayClick">
      <Transition name="pop" appear>
        <div v-if="visible" class="modal-container" :style="{ maxWidth: width }">
          <div class="modal-header">
            <h3 v-if="title" class="modal-title">{{ title }}</h3>
            <span style="flex:1"></span>
            <button class="close-btn" @click="closeModal" aria-label="关闭">
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                <line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/>
              </svg>
            </button>
          </div>
          <div class="modal-body"><slot /></div>
          <div v-if="$slots.footer" class="modal-footer"><slot name="footer" /></div>
        </div>
      </Transition>
    </div>
  </Transition>
</template>

<style scoped>
.modal-overlay {
  position: fixed; inset: 0; z-index: 9000;
  display: flex; align-items: center; justify-content: center;
  background: rgba(20, 25, 34, 0.42);
  backdrop-filter: blur(2px);
  -webkit-backdrop-filter: blur(2px);
}

.modal-container {
  background: var(--color-bg-primary);
  border: 1px solid var(--color-border);
  border-radius: 12px;
  width: 90%;
  max-height: 82vh;
  display: flex;
  flex-direction: column;
  box-shadow: var(--shadow-lg);
  overflow: hidden;
}

.modal-header {
  display: flex; align-items: center;
  padding: 14px 18px;
  border-bottom: 1px solid var(--color-border-light);
  gap: 8px;
}
.modal-title {
  font-size: 0.95rem;
  font-weight: 600;
  color: var(--color-text-primary);
  letter-spacing: 0.01em;
  margin: 0;
}
.close-btn {
  width: 28px; height: 28px;
  display: inline-flex; align-items: center; justify-content: center;
  background: transparent;
  border: 1px solid transparent;
  color: var(--color-text-tertiary);
  border-radius: var(--radius-sm);
  cursor: pointer;
  transition: background var(--dur-fast) var(--ease), color var(--dur-fast) var(--ease), border-color var(--dur-fast) var(--ease);
}
.close-btn:hover {
  background: var(--color-bg-hover);
  color: var(--color-text-primary);
  border-color: var(--color-border);
}

.modal-body {
  padding: 18px;
  overflow-y: auto;
  color: var(--color-text-primary);
  font-size: 0.85rem;
  line-height: 1.6;
}

.modal-footer {
  padding: 12px 18px;
  border-top: 1px solid var(--color-border-light);
  display: flex;
  justify-content: flex-end;
  gap: 10px;
  background: var(--color-bg-secondary);
}
</style>
