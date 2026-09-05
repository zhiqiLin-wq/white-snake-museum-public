<script setup lang="ts">
import { ref } from 'vue'

defineProps<{
  visible: boolean
  x: number
  y: number
}>()

const emit = defineEmits<{
  action: [action: string, payload?: Record<string, unknown>]
  close: []
}>()

const showCustomColor = ref(false)

const CATEGORIES = [
  { key: 'person', label: '人物', color: '#68A0C8' },
  { key: 'location', label: '地点', color: '#78B898' },
  { key: 'event', label: '事件', color: '#68B0B8' },
  { key: 'term', label: '术语', color: '#9888C0' },
  { key: 'motif', label: '母题', color: '#C898B0' },
  { key: 'custom', label: '自定义', color: '#B8B878' },
]

function annotateWithCategory(catKey: string) {
  emit('action', 'annotate_category', { category: catKey })
}
</script>

<template>
  <Teleport to="body">
    <div
      v-if="visible"
      class="selection-menu-backdrop"
      @click="emit('close')"
    />
    <div
      v-if="visible"
      class="selection-menu"
      :style="{ left: x + 'px', top: y + 'px' }"
    >
      <!-- Quick annotation by category -->
      <div class="sm-section-label">标注类别</div>
      <button
        v-for="cat in CATEGORIES"
        :key="cat.key"
        class="sm-item sm-cat-item"
        @click="annotateWithCategory(cat.key)"
      >
        <span class="sm-color-dot" :style="{ background: cat.color }" />
        {{ cat.label }}
      </button>

      <div class="sm-divider" />

      <button class="sm-item" @click="emit('action', 'annotate')">
        标注为...
      </button>
      <button class="sm-item" @click="emit('action', 'custom_color')">
        自定义颜色...
      </button>
      <button class="sm-item" @click="emit('action', 'write_marginalia')">
        写批注
      </button>

      <div class="sm-divider" />

      <button class="sm-item" @click="emit('action', 'send_agent')">
        发给 Agent
      </button>
      <button class="sm-item" @click="emit('action', 'copy')">
        复制
      </button>
    </div>
  </Teleport>
</template>

<style scoped>
.selection-menu-backdrop {
  position: fixed;
  inset: 0;
  z-index: 8000;
}

.selection-menu {
  position: fixed;
  z-index: 8001;
  background: var(--color-bg-primary);
  border: 1px solid var(--color-border);
  border-radius: var(--radius);
  box-shadow: var(--shadow-md);
  padding: 4px 0;
  min-width: 160px;
}

.sm-section-label {
  padding: 4px 14px 2px;
  font-size: 0.62rem;
  color: var(--color-text-tertiary);
  letter-spacing: 0.02em;
}

.sm-item {
  display: block;
  width: 100%;
  text-align: left;
  padding: 7px 14px;
  background: none;
  border: none;
  color: var(--color-text-primary);
  font-family: var(--font-sans);
  font-size: 0.78rem;
  cursor: pointer;
  transition: background-color var(--dur-fast) var(--ease), color var(--dur-fast) var(--ease);
}

.sm-item:hover {
  background: var(--color-accent-light);
  color: var(--color-accent);
}

.sm-cat-item {
  display: flex;
  align-items: center;
  gap: 8px;
}

.sm-color-dot {
  width: 8px;
  height: 8px;
  border-radius: 50%;
  flex-shrink: 0;
  box-shadow: 0 0 0 1px var(--color-border-light);
}

.sm-divider {
  height: 1px;
  background: var(--color-border-light);
  margin: 4px 8px;
}
</style>
