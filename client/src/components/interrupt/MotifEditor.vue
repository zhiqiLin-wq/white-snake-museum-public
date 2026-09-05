<script setup lang="ts">
import { ref } from 'vue'
import InterruptTimeoutOverlay from './InterruptTimeoutOverlay.vue'

export interface MotifItem {
  id: string
  name: string
  type: 'invariant' | 'variable'
  core: string
  isUserEdited?: boolean
}

export interface SuggestionItem {
  id: string
  name: string
  type: 'invariant' | 'variable'
  core: string
}

defineProps<{
  motifs: MotifItem[]
  suggestedAdditions: SuggestionItem[]
  trackingTarget: string
  newMotifName: string
  newMotifType: 'invariant' | 'variable'
  countdownRemaining: number
  isTimedOut: boolean
}>()

const emit = defineEmits<{
  'edit-type': [id: string, type: 'invariant' | 'variable']
  'edit-core': [id: string, core: string]
  'delete-motif': [id: string]
  'update:newMotifName': [value: string]
  'update:newMotifType': [value: 'invariant' | 'variable']
  'add-motif': []
  'add-suggested': [item: SuggestionItem]
  'confirm': []
  'restart': []
}>()

const editingMotifId = ref<string | null>(null)
const editingMotifCore = ref('')

function startEdit(id: string, core: string) {
  editingMotifId.value = id
  editingMotifCore.value = core
}

function cancelEdit() {
  editingMotifId.value = null
  editingMotifCore.value = ''
}

</script>

<template>
  <div class="ew-interrupt" style="position: relative">
    <InterruptTimeoutOverlay
      :visible="true"
      :countdown-remaining="countdownRemaining"
      :is-timed-out="isTimedOut"
      @restart="emit('restart')"
    />

    <template v-if="!isTimedOut">
      <h3>追踪目标: {{ trackingTarget }}</h3>
      <p class="ew-instruction">AI已完成母题分解，请审核并编辑:</p>

      <div class="ew-motif-list">
        <div v-for="m in motifs" :key="m.id" class="ew-motif-item">
          <div class="ew-motif-header">
            <span class="ew-motif-id">{{ m.id }}</span>
            <span class="ew-motif-name">{{ m.name }}</span>
            <select
              class="ew-motif-type-select"
              :value="m.type"
              @change="emit('edit-type', m.id, ($event.target as HTMLSelectElement).value as 'invariant' | 'variable')"
            >
              <option value="invariant">不变母题</option>
              <option value="variable">可变母题</option>
            </select>
            <button class="ew-motif-edit" @click="startEdit(m.id, m.core || '')">编辑</button>
            <button class="ew-motif-delete" @click="emit('delete-motif', m.id)">删除</button>
          </div>
          <div v-if="editingMotifId === m.id" class="ew-motif-edit-form">
            <textarea v-model="editingMotifCore" class="ew-motif-edit-textarea" rows="3" />
            <div class="ew-motif-edit-actions">
              <button class="ew-motif-save" @click="emit('edit-core', m.id, editingMotifCore); cancelEdit()">保存</button>
              <button class="ew-motif-cancel" @click="cancelEdit()">取消</button>
            </div>
          </div>
          <div v-else class="ew-motif-core">{{ m.core }}</div>
          <div v-if="m.isUserEdited" class="ew-motif-edited">已修改</div>
        </div>
      </div>

      <div v-if="suggestedAdditions.length > 0" class="ew-suggested">
        <h4>AI建议添加的母题</h4>
        <div v-for="s in suggestedAdditions" :key="s.id" class="ew-suggested-item">
          <span>{{ s.name }} ({{ s.type }})</span>
          <button @click="emit('add-suggested', s)">添加</button>
        </div>
      </div>

      <div class="ew-add-motif-form">
        <input
          :value="newMotifName"
          class="ew-add-motif-input"
          placeholder="添加新母题名称"
          @input="emit('update:newMotifName', ($event.target as HTMLInputElement).value)"
          @keydown.enter="emit('add-motif')"
        />
        <select
          :value="newMotifType"
          class="ew-add-motif-select"
          @change="emit('update:newMotifType', ($event.target as HTMLSelectElement).value as 'invariant' | 'variable')"
        >
          <option value="invariant">invariant</option>
          <option value="variable">variable</option>
        </select>
        <button class="ew-add-motif-btn" @click="emit('add-motif')">+ 添加母题</button>
      </div>

      <div class="ew-interrupt-actions">
        <button class="ew-confirm-btn" @click="emit('confirm')">确认母题，继续分析</button>
      </div>
    </template>
  </div>
</template>

<style scoped>
/* Reuse existing EvolutionWorkspaceView styles */
.ew-interrupt { padding: 12px; }
.ew-interrupt h3 {
  font-size: 0.85rem;
  margin: 0 0 4px;
  color: var(--color-text-primary);
  font-weight: 500;
}
.ew-instruction { font-size: 0.7rem; color: var(--color-text-tertiary); margin-bottom: 8px; }
.ew-motif-list { display: flex; flex-direction: column; gap: 6px; }
.ew-motif-item {
  padding: 6px 8px;
  border: 1px solid var(--color-border);
  border-radius: var(--radius);
  background: var(--color-bg-primary);
  transition: border-color var(--dur-fast) var(--ease), box-shadow var(--dur-fast) var(--ease);
}
.ew-motif-item:hover {
  border-color: var(--color-accent);
  box-shadow: var(--shadow-sm);
}
.ew-motif-header { display: flex; align-items: center; gap: 6px; font-size: 0.72rem; }
.ew-motif-id {
  font-family: var(--font-mono);
  font-size: 0.6rem;
  color: var(--color-text-tertiary);
}
.ew-motif-name { font-weight: 500; color: var(--color-text-primary); }
.ew-motif-type-select {
  font-size: 0.65rem;
  padding: 1px 4px;
  background: var(--color-bg-tertiary);
  color: var(--color-text-primary);
  border: 1px solid var(--color-border);
  border-radius: var(--radius-sm);
  cursor: pointer;
  transition: border-color var(--dur-fast) var(--ease);
}
.ew-motif-type-select:hover {
  border-color: var(--color-accent);
}
.ew-motif-edit, .ew-motif-delete {
  padding: 1px 6px;
  font-size: 0.6rem;
  background: var(--color-bg-tertiary);
  color: var(--color-text-secondary);
  border: 1px solid var(--color-border);
  border-radius: var(--radius-sm);
  cursor: pointer;
  transition: background-color var(--dur-fast) var(--ease), border-color var(--dur-fast) var(--ease), color var(--dur-fast) var(--ease);
}
.ew-motif-edit:hover {
  background: var(--color-accent-light);
  border-color: var(--color-accent);
  color: var(--color-accent);
}
.ew-motif-delete:hover {
  background: var(--color-bg-tertiary);
  border-color: var(--color-verdict-refuted);
  color: var(--color-verdict-refuted);
}
.ew-motif-core {
  font-size: 0.68rem;
  color: var(--color-text-secondary);
  margin-top: 4px;
  line-height: 1.5;
}
.ew-motif-edited {
  font-size: 0.6rem;
  color: var(--color-phase-interrupt);
  display: inline-block;
  padding: 1px 4px;
  background: var(--color-bg-tertiary);
  border-radius: var(--radius-sm);
  margin-top: 2px;
}
.ew-motif-edit-form { margin-top: 4px; }
.ew-motif-edit-textarea {
  width: 100%;
  font-size: 0.68rem;
  padding: 4px;
  background: var(--color-bg-primary);
  color: var(--color-text-primary);
  border: 1px solid var(--color-border);
  border-radius: var(--radius-sm);
  resize: vertical;
  font-family: var(--font-sans);
  outline: none;
  transition: border-color var(--dur-fast) var(--ease), box-shadow var(--dur-fast) var(--ease);
}
.ew-motif-edit-textarea:focus {
  border-color: var(--color-accent);
  box-shadow: var(--shadow-focus);
}
.ew-motif-edit-actions { display: flex; gap: 4px; margin-top: 4px; }
.ew-motif-save, .ew-motif-cancel {
  padding: 2px 8px;
  font-size: 0.62rem;
  border-radius: var(--radius-sm);
  cursor: pointer;
  transition: background-color var(--dur-fast) var(--ease), border-color var(--dur-fast) var(--ease);
}
.ew-motif-save {
  background: var(--color-verdict-supported);
  color: #fff;
  border: none;
}
.ew-motif-save:hover { background: #4f9770; }
.ew-motif-cancel {
  background: var(--color-bg-tertiary);
  color: var(--color-text-primary);
  border: 1px solid var(--color-border);
}
.ew-motif-cancel:hover {
  background: var(--color-bg-hover);
  border-color: var(--color-border-dark);
}
.ew-suggested {
  margin-top: 8px;
  padding: 8px;
  background: var(--color-bg-secondary);
  border: 1px solid var(--color-border-light);
  border-radius: var(--radius);
}
.ew-suggested h4 {
  font-size: 0.7rem;
  margin-bottom: 4px;
  color: var(--color-text-secondary);
  font-weight: 500;
}
.ew-suggested-item {
  display: flex;
  align-items: center;
  gap: 6px;
  font-size: 0.68rem;
  padding: 4px 0;
  color: var(--color-text-primary);
}
.ew-suggested-item button {
  padding: 1px 6px;
  font-size: 0.6rem;
  background: var(--color-accent);
  color: #fff;
  border: none;
  border-radius: var(--radius-sm);
  cursor: pointer;
  margin-left: auto;
  transition: background-color var(--dur-fast) var(--ease);
}
.ew-suggested-item button:hover {
  background: var(--color-accent-hover);
}
.ew-add-motif-form { display: flex; gap: 4px; margin-top: 8px; }
.ew-add-motif-input {
  flex: 1;
  font-size: 0.68rem;
  padding: 4px 6px;
  background: var(--color-bg-primary);
  color: var(--color-text-primary);
  border: 1px solid var(--color-border);
  border-radius: var(--radius-sm);
  outline: none;
  transition: border-color var(--dur-fast) var(--ease), box-shadow var(--dur-fast) var(--ease);
}
.ew-add-motif-input:focus {
  border-color: var(--color-accent);
  box-shadow: var(--shadow-focus);
}
.ew-add-motif-select {
  font-size: 0.65rem;
  padding: 2px 4px;
  background: var(--color-bg-tertiary);
  color: var(--color-text-primary);
  border: 1px solid var(--color-border);
  border-radius: var(--radius-sm);
  cursor: pointer;
}
.ew-add-motif-btn {
  padding: 3px 10px;
  font-size: 0.68rem;
  background: var(--color-accent);
  color: #fff;
  border: none;
  border-radius: var(--radius-sm);
  cursor: pointer;
  transition: background-color var(--dur-fast) var(--ease);
}
.ew-add-motif-btn:hover {
  background: var(--color-accent-hover);
}
.ew-interrupt-actions { margin-top: 12px; display: flex; justify-content: flex-end; }
.ew-confirm-btn {
  padding: 6px 16px;
  font-size: 0.75rem;
  background: var(--color-verdict-supported);
  color: #fff;
  border: none;
  border-radius: var(--radius);
  cursor: pointer;
  transition: background-color var(--dur-fast) var(--ease), box-shadow var(--dur-fast) var(--ease);
}
.ew-confirm-btn:hover {
  background: #4f9770;
  box-shadow: var(--shadow-sm);
}
</style>
