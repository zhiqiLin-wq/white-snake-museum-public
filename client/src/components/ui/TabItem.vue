<script setup lang="ts">
defineProps<{
  title: string
  active: boolean
  dirty: boolean
  tabId?: string
}>()

const emit = defineEmits<{
  click: []
  close: []
  dragStart: [event: DragEvent, tabId: string]
}>()

function onDragStart(event: DragEvent) {
  const tabId = (event.currentTarget as HTMLElement).dataset.tabId
  if (tabId && event.dataTransfer) {
    event.dataTransfer.effectAllowed = 'move'
    event.dataTransfer.setData('text/plain', tabId)
    emit('dragStart', event, tabId)
  }
}
</script>

<template>
  <div
    class="tab-item"
    :class="{ active, dirty }"
    :data-tab-id="tabId"
    draggable="true"
    @click="emit('click')"
    @dragstart="onDragStart"
  >
    <span class="tab-title">{{ title.length > 20 ? title.slice(0, 20) + '...' : title }}</span>
    <span v-if="dirty && !active" class="dirty-dot" />
    <button
      v-if="true"
      class="tab-close"
      @click.stop="emit('close')"
    >&times;</button>
  </div>
</template>

<style scoped>
.tab-item {
  position: relative;
  display: inline-flex;
  align-items: center;
  gap: 4px;
  padding: 8px 12px 6px;
  cursor: pointer;
  color: var(--color-text-secondary);
  font-family: var(--font-sans);
  font-size: 0.8rem;
  white-space: nowrap;
  user-select: none;
  transition: color var(--dur) var(--ease), transform var(--dur) var(--ease);
  flex-shrink: 0;
}

.tab-item::after {
  content: '';
  position: absolute;
  left: 12px;
  right: 12px;
  bottom: 0;
  height: 2px;
  background: var(--color-accent);
  border-radius: 1px;
  transform: scaleX(0);
  transform-origin: center;
  transition: transform var(--dur) var(--ease);
}

.tab-item:hover {
  color: var(--color-text-primary);
  transform: translateY(-1px);
}

.tab-item.active {
  color: var(--color-text-primary);
}

.tab-item.active::after {
  transform: scaleX(1);
}

.tab-item.dirty .dirty-dot {
  width: 6px;
  height: 6px;
  border-radius: 50%;
  background: var(--color-warning);
  flex-shrink: 0;
}

.tab-close {
  background: none;
  border: none;
  color: var(--color-text-tertiary);
  cursor: pointer;
  font-size: 1rem;
  line-height: 1;
  padding: 0 2px;
  margin-left: 2px;
  border-radius: var(--radius-sm);
  transition: background var(--dur-fast) var(--ease), color var(--dur-fast) var(--ease);
}

.tab-close:hover {
  color: var(--color-error);
  background: var(--color-error-light);
}
</style>
