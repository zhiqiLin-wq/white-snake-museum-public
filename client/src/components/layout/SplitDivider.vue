<script setup lang="ts">
import { ref } from 'vue'

const emit = defineEmits<{
  ratioChange: [ratio: number]
  doubleClick: []
}>()

const isDragging = ref(false)

function onMouseDown(event: MouseEvent) {
  isDragging.value = true
  event.preventDefault()

  const startX = event.clientX
  const container = (event.target as HTMLElement).parentElement
  const containerWidth = container?.clientWidth || window.innerWidth

  function onMouseMove(e: MouseEvent) {
    if (!isDragging.value) return
    const deltaX = e.clientX - startX
    const ratio = 0.5 + deltaX / containerWidth
    const clamped = Math.max(0.3, Math.min(0.7, ratio))
    emit('ratioChange', clamped)
  }

  function onMouseUp() {
    isDragging.value = false
    document.removeEventListener('mousemove', onMouseMove)
    document.removeEventListener('mouseup', onMouseUp)
  }

  document.addEventListener('mousemove', onMouseMove)
  document.addEventListener('mouseup', onMouseUp)
}

function onDoubleClick() {
  emit('ratioChange', 0.5)
  emit('doubleClick')
}
</script>

<template>
  <div
    class="split-divider"
    :class="{ dragging: isDragging }"
    @mousedown="onMouseDown"
    @dblclick="onDoubleClick"
  />
</template>

<style scoped>
/* 宽热区 + 1px 细线，hover/拖动时淡蓝加粗到 2px */
.split-divider {
  width: 7px;
  cursor: col-resize;
  background: transparent;
  position: relative;
  flex-shrink: 0;
  user-select: none;
}

.split-divider::before {
  content: '';
  position: absolute;
  top: 0;
  bottom: 0;
  left: 3px;
  width: 1px;
  background: var(--color-border, #E5E7EB);
  transition:
    background-color var(--dur-fast, 120ms) var(--ease, cubic-bezier(.22,.61,.36,1)),
    width var(--dur-fast, 120ms) var(--ease, cubic-bezier(.22,.61,.36,1));
}

.split-divider:hover::before,
.split-divider.dragging::before {
  left: 2.5px;
  width: 2px;
  background: var(--color-accent, #5B9BD5);
}
</style>
