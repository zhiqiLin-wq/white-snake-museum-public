<script setup lang="ts">
import { ref, onMounted, Teleport } from 'vue'
import { apiFetch } from '@/services/api'

const props = defineProps<{
  show: boolean
}>()

const emit = defineEmits<{
  imported: []
  skip: []
}>()

const chapterCount = ref(0)
const annotationCount = ref(0)
const marginaliaCount = ref(0)
const isImporting = ref(false)
const importError = ref<string | null>(null)

onMounted(() => {
  try {
    const raw = localStorage.getItem('white_snake_user_annotation_sets')
    if (raw) {
      const sets = JSON.parse(raw) as Record<string, { annotations?: Array<unknown>; marginalia?: Array<unknown> }>
      const chapterKeys = Object.keys(sets)
      chapterCount.value = chapterKeys.length
      for (const key of chapterKeys) {
        annotationCount.value += (sets[key].annotations || []).length
        marginaliaCount.value += (sets[key].marginalia || []).length
      }
    }
  } catch {
    chapterCount.value = 0
    annotationCount.value = 0
    marginaliaCount.value = 0
  }
})

async function importAll() {
  isImporting.value = true
  importError.value = null
  try {
    const raw = localStorage.getItem('white_snake_user_annotation_sets')
    if (raw) {
      const sets = JSON.parse(raw)
      const resp = await apiFetch('/api/annotations/import', {
        method: 'POST',
        body: JSON.stringify({ annotationSets: sets }),
      })
      if (resp.ok) {
        localStorage.removeItem('white_snake_user_annotation_sets')
        emit('imported')
        return
      }
      importError.value = 'Import failed with status ' + resp.status
    }
  } catch (err: unknown) {
    importError.value = err instanceof Error ? err.message : 'Import failed'
  } finally {
    isImporting.value = false
  }
}

function importConversationsOnly() {
  // Conversations are synced via syncConversationsFromServer in MuseumView
  emit('imported')
}

function skip() {
  emit('skip')
}
</script>

<template>
  <Teleport to="body">
    <div v-if="show" class="migration-overlay">
      <div class="migration-dialog">
        <h3 class="migration-title">Data Migration</h3>
        <p class="migration-text">
          Local annotation data detected ({{ chapterCount }} chapters, {{ annotationCount }} annotations, {{ marginaliaCount }} notes).
          Import to your account?
        </p>
        <p v-if="importError" class="migration-error">{{ importError }}</p>
        <div class="migration-actions">
          <button
            class="migration-btn primary"
            :disabled="isImporting"
            @click="importAll"
          >
            {{ isImporting ? 'Importing...' : 'Import All' }}
          </button>
          <button
            class="migration-btn"
            :disabled="isImporting"
            @click="importConversationsOnly"
          >
            Conversations Only
          </button>
          <button
            class="migration-btn secondary"
            :disabled="isImporting"
            @click="skip"
          >
            Skip
          </button>
        </div>
      </div>
    </div>
  </Teleport>
</template>

<style scoped>
.migration-overlay {
  position: fixed;
  inset: 0;
  z-index: 1000;
  display: flex;
  align-items: center;
  justify-content: center;
  background: rgba(0, 0, 0, 0.5);
}

.migration-dialog {
  background: var(--color-bg-primary, #fff);
  border-radius: 12px;
  padding: 32px;
  max-width: 480px;
  width: 90%;
  box-shadow: 0 8px 32px rgba(0, 0, 0, 0.2);
}

.migration-title {
  margin: 0 0 16px;
  font-family: var(--font-sans);
  font-size: 1.25rem;
  color: var(--color-text-primary);
}

.migration-text {
  margin: 0 0 24px;
  font-family: var(--font-sans);
  font-size: 0.95rem;
  line-height: 1.6;
  color: var(--color-text-secondary);
}

.migration-error {
  margin: 0 0 16px;
  padding: 8px 12px;
  background: #fef2f2;
  border: 1px solid #fecaca;
  border-radius: 6px;
  color: #dc2626;
  font-size: 0.85rem;
}

.migration-actions {
  display: flex;
  gap: 12px;
  flex-wrap: wrap;
}

.migration-btn {
  padding: 10px 20px;
  border: 1px solid var(--color-border);
  border-radius: 8px;
  background: var(--color-bg-secondary);
  color: var(--color-text-primary);
  font-family: var(--font-sans);
  font-size: 0.9rem;
  cursor: pointer;
  transition: opacity 0.15s;
}

.migration-btn:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}

.migration-btn.primary {
  background: var(--color-phase-active);
  color: #fff;
  border-color: var(--color-phase-active);
}

.migration-btn.secondary {
  color: var(--color-text-secondary);
}

.migration-btn:hover:not(:disabled) {
  opacity: 0.85;
}
</style>
