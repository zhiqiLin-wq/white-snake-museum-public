<script setup lang="ts">
import { ref } from 'vue'

const props = defineProps<{
  visible: boolean
  anchorX: number
  anchorY: number
  initialContent?: string
}>()

const emit = defineEmits<{
  confirm: [content: string]
  cancel: []
}>()

const content = ref(props.initialContent || '')

function confirm() {
  if (!content.value.trim()) {
    emit('cancel')
    return
  }
  emit('confirm', content.value)
}
</script>

<template>
  <Teleport to="body">
    <div
      v-if="visible"
      class="marginalia-form-backdrop"
      @click="emit('cancel')"
    />
    <div
      v-if="visible"
      class="marginalia-form"
      :style="{ left: anchorX + 'px', top: anchorY + 'px' }"
    >
      <div class="mf-header">批注</div>
      <textarea
        v-model="content"
        class="mf-textarea"
        placeholder="输入批注..."
        rows="4"
        autofocus
      />
      <div class="mf-actions">
        <button class="mf-btn-cancel" @click="emit('cancel')">取消</button>
        <button class="mf-btn-confirm" @click="confirm">保存</button>
      </div>
    </div>
  </Teleport>
</template>

<style scoped>
.marginalia-form-backdrop {
  position: fixed;
  inset: 0;
  z-index: 8000;
}

.marginalia-form {
  position: fixed;
  z-index: 8001;
  background: var(--color-bg-primary);
  border: 1px solid var(--color-border);
  border-radius: var(--radius-lg);
  padding: 12px;
  min-width: 240px;
  max-width: 320px;
  box-shadow: var(--shadow-lg);
}

.mf-header {
  font-size: 0.8rem;
  color: var(--color-text-primary);
  margin-bottom: 8px;
  font-weight: 500;
}

.mf-textarea {
  width: 100%;
  background: var(--color-bg-primary);
  border: 1px solid var(--color-border);
  border-radius: var(--radius);
  color: var(--color-text-primary);
  padding: 8px;
  font-size: 0.75rem;
  font-family: var(--font-sans);
  resize: vertical;
  box-sizing: border-box;
  outline: none;
  transition: border-color var(--dur-fast) var(--ease), box-shadow var(--dur-fast) var(--ease);
}

.mf-textarea:focus {
  border-color: var(--color-accent);
  box-shadow: var(--shadow-focus);
}

.mf-actions {
  display: flex;
  justify-content: flex-end;
  gap: 6px;
  margin-top: 8px;
}

.mf-btn-cancel {
  padding: 4px 12px;
  font-size: 0.7rem;
  background: var(--color-bg-tertiary);
  color: var(--color-text-primary);
  border: 1px solid var(--color-border);
  border-radius: var(--radius);
  cursor: pointer;
  transition: background-color var(--dur-fast) var(--ease), border-color var(--dur-fast) var(--ease);
}

.mf-btn-cancel:hover {
  background: var(--color-bg-hover);
  border-color: var(--color-border-dark);
}

.mf-btn-confirm {
  padding: 4px 12px;
  font-size: 0.7rem;
  background: var(--color-accent);
  color: #fff;
  border: none;
  border-radius: var(--radius);
  cursor: pointer;
  transition: background-color var(--dur-fast) var(--ease);
}

.mf-btn-confirm:hover {
  background: var(--color-accent-hover);
}
</style>
