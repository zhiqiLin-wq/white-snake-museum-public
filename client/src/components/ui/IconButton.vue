<script setup lang="ts">
withDefaults(defineProps<{
  icon: string
  label?: string
  size?: 'sm' | 'md'
  variant?: 'primary' | 'secondary' | 'ghost' | 'danger'
}>(), {
  variant: 'secondary',
  size: 'md',
})

const emit = defineEmits<{ click: [] }>()
</script>

<template>
  <button
    class="ib"
    :class="['ib-' + variant, size === 'sm' ? 'ib-sm' : '']"
    @click="emit('click')"
    :title="label || ''"
  >
    <span class="ib-icon">{{ icon }}</span>
    <span v-if="label" class="ib-label">{{ label }}</span>
  </button>
</template>

<style scoped>
.ib {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  gap: 6px;
  padding: 0 12px;
  height: 30px;
  border-radius: var(--radius);
  border: 1px solid var(--color-border);
  background: var(--color-bg-primary);
  color: var(--color-text-primary);
  cursor: pointer;
  font-family: inherit;
  font-size: 0.8rem;
  font-weight: 500;
  white-space: nowrap;
  transition:
    background-color var(--dur-fast) var(--ease),
    border-color var(--dur-fast) var(--ease),
    color var(--dur-fast) var(--ease),
    transform var(--dur-fast) var(--ease),
    box-shadow var(--dur-fast) var(--ease);
}
.ib-sm { height: 26px; padding: 0 9px; font-size: 0.75rem; border-radius: var(--radius-sm); }
.ib-icon { font-size: 0.95rem; line-height: 1; display: inline-flex; align-items: center; }
.ib-sm .ib-icon { font-size: 0.85rem; }

.ib:hover   { border-color: var(--color-border-dark); background: var(--color-bg-hover); }
.ib:active  { transform: translateY(0); }

/* --- variants --- */
.ib-primary {
  color: #fff;
  background: var(--color-accent);
  border-color: var(--color-accent);
}
.ib-primary:hover {
  background: var(--color-accent-hover);
  border-color: var(--color-accent-hover);
  transform: translateY(-1px);
  box-shadow: 0 4px 10px rgba(91,155,213,0.22);
}
.ib-primary:active { transform: translateY(0); }

.ib-ghost {
  background: transparent;
  border-color: transparent;
  color: var(--color-text-secondary);
}
.ib-ghost:hover { background: var(--color-bg-hover); color: var(--color-text-primary); }

.ib-danger {
  color: var(--color-error);
  border-color: rgba(255,77,79,.35);
  background: transparent;
}
.ib-danger:hover { background: var(--color-error-light); border-color: var(--color-error); }
</style>
