<script setup lang="ts">
import type { ContextTag } from '@/stores/agent'

defineProps<{
  tags: ContextTag[]
}>()

const emit = defineEmits<{
  'remove': [tagId: string]
}>()
</script>

<template>
  <div v-if="tags.length > 0" class="context-tags">
    <span
      v-for="tag in tags"
      :key="tag.id"
      class="ct-tag"
    >
      <span class="ct-icon">attach</span>
      <span class="ct-label">{{ tag.label }}</span>
      <button class="ct-remove" @click="emit('remove', tag.id)">&times;</button>
    </span>
  </div>
</template>

<style scoped>
.context-tags {
  display: flex;
  flex-wrap: wrap;
  gap: 4px;
  padding: 6px 0;
}

.ct-tag {
  display: inline-flex;
  align-items: center;
  gap: 4px;
  padding: 3px 8px;
  background: var(--color-accent-light);
  border: 1px solid var(--color-border);
  border-radius: var(--radius-sm);
  font-size: 0.72rem;
  color: var(--color-text-primary);
  transition: background-color var(--dur-fast) var(--ease), border-color var(--dur-fast) var(--ease);
}

.ct-tag:hover {
  background: var(--color-bg-tertiary);
  border-color: var(--color-border-dark);
}

.ct-icon {
  font-size: 0.65rem;
  color: var(--color-accent);
}

.ct-label {
  max-width: 180px;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.ct-remove {
  background: none;
  border: none;
  color: var(--color-text-tertiary);
  cursor: pointer;
  font-size: 0.85rem;
  padding: 0;
  line-height: 1;
  transition: color var(--dur-fast) var(--ease);
}

.ct-remove:hover {
  color: var(--color-verdict-refuted);
}
</style>
