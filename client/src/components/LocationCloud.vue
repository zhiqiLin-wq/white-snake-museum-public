<script setup lang="ts">
defineProps<{ locationNames: string[]; selectedLocation: string | null }>()
defineEmits<{ select: [name: string] }>()
</script>

<template>
  <div class="locations-cloud">
    <span
      v-for="name in locationNames" :key="name"
      class="loc-badge"
      :class="{ active: name === selectedLocation }"
      @click="$emit('select', name)"
    >{{ name }}</span>
  </div>
</template>

<style scoped>
.locations-cloud {
  padding: var(--space-2) var(--space-5);
  display: flex;
  flex-wrap: wrap;
  gap: var(--space-2);
  border-bottom: 1px solid var(--color-border-light);
}
.loc-badge {
  background: var(--color-bg-secondary);
  padding: 4px 12px;
  border-radius: var(--radius);
  font-size: 0.72rem;
  font-weight: 500;
  color: var(--color-text-secondary);
  cursor: pointer;
  transition:
    background var(--dur) var(--ease),
    border-color var(--dur) var(--ease),
    color var(--dur) var(--ease),
    transform var(--dur) var(--ease),
    box-shadow var(--dur) var(--ease);
  border: 1px solid var(--color-border);
}
.loc-badge:hover {
  background: var(--color-bg-hover);
  border-color: var(--color-accent);
  color: var(--color-accent);
  transform: translateY(-1px);
  box-shadow: var(--shadow-sm);
}
.loc-badge.active {
  background: var(--color-accent-light);
  color: var(--color-accent);
  border-color: var(--color-accent);
  font-weight: 600;
  box-shadow: var(--shadow-sm);
}
</style>
