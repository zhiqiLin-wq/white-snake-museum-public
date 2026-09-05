<script setup lang="ts">
import { ref, computed } from 'vue'
import { useAgentStore } from '@/stores/agent'
import { useAnnotationStore } from '@/stores/annotations'
import type { AnnotationCategory } from '@/types/annotation'

const props = defineProps<{
  annotation: Record<string, unknown> | null
  passageKey?: string
  x: number
  y: number
  overlappingAnnotation?: Record<string, unknown> | null
}>()

const emit = defineEmits<{
  close: []
  edit: [id: string]
  delete: [id: string]
  adopt: [id: string]
  adoptAndEdit: [id: string]
  showOverlapping: [id: string]
}>()

const agentStore = useAgentStore()
const annotationStore = useAnnotationStore()

const source = computed(() => (props.annotation?.source as string) || 'agent')
const isUserAnnotation = computed(() => source.value === 'user')

const editMode = ref(false)
const editingColor = ref('')
const editingCategory = ref<AnnotationCategory>('custom')

const PRESET_COLORS: Record<string, string> = {
  person: '#68A0C8',
  location: '#78B898',
  event: '#68B0B8',
  term: '#9888C0',
  motif: '#C898B0',
  custom: '#B8B878',
}

function enterEditMode() {
  if (!props.annotation) return
  editingColor.value = (props.annotation.color as string) || ''
  editingCategory.value = (props.annotation.category as AnnotationCategory) || 'custom'
  editMode.value = true
}

function saveEdit() {
  if (!props.annotation || !props.passageKey) return
  const annId = props.annotation.id as string
  annotationStore.updateUserAnnotation(props.passageKey, annId, {
    color: editingColor.value,
    category: editingCategory.value,
  })
  editMode.value = false
}

function cancelEdit() {
  editMode.value = false
}

function askAbout() {
  if (!props.annotation) return
  agentStore.addContextTag({
    id: `ctx_${Date.now()}`,
    label: `${props.annotation.label || ''} - ${props.annotation.category || ''}`,
    data: { annotation: props.annotation },
    sourceType: 'annotation',
  })
  emit('close')
}

function onDelete() {
  console.log('[AnnotationPopup] onDelete called, annotation:', props.annotation)
  if (!props.annotation) return
  emit('delete', props.annotation.id as string)
}

function onAdopt() {
  console.log('[AnnotationPopup] onAdopt called')
  if (!props.annotation) return
  emit('adopt', props.annotation.id as string)
}

function onAdoptAndEdit() {
  if (!props.annotation) return
  emit('adoptAndEdit', props.annotation.id as string)
}

const categoryOptions: { key: AnnotationCategory; label: string }[] = [
  { key: 'person', label: '人物' },
  { key: 'location', label: '地点' },
  { key: 'event', label: '事件' },
  { key: 'term', label: '术语' },
  { key: 'motif', label: '母题' },
  { key: 'custom', label: '自定义' },
]
</script>

<template>
  <Teleport to="body">
    <div
      v-if="annotation"
      class="annotation-popup-backdrop"
      @click="emit('close')"
    />
    <div
      v-if="annotation"
      class="annotation-popup"
      :style="{ left: x + 'px', top: y + 'px' }"
    >
      <!-- Edit mode -->
      <template v-if="editMode">
        <div class="ap-edit-section">
          <div class="ap-edit-label">选择类别</div>
          <div class="ap-edit-categories">
            <label
              v-for="item in categoryOptions"
              :key="item.key"
              class="ap-edit-cat"
            >
              <input
                type="radio"
                :value="item.key"
                v-model="editingCategory"
              />
              {{ item.label }}
            </label>
          </div>
        </div>
        <div class="ap-edit-section">
          <div class="ap-edit-label">选择颜色</div>
          <div class="ap-edit-colors">
            <span
              v-for="(color, name) in PRESET_COLORS"
              :key="name"
              class="ap-color-swatch"
              :class="{ selected: editingColor === color }"
              :style="{ background: color }"
              @click="editingColor = color"
            />
            <input
              type="color"
              v-model="editingColor"
              class="ap-color-input"
            />
          </div>
        </div>
        <div class="ap-edit-actions">
          <button class="ap-btn-cancel" @click="cancelEdit">取消</button>
          <button class="ap-btn-save" @click="saveEdit">保存</button>
        </div>
      </template>

      <!-- View mode -->
      <template v-else>
        <div class="ap-header">
          <span class="ap-label">{{ annotation.label }}</span>
          <span class="ap-category">{{ annotation.category }}</span>
          <span class="ap-source-badge" :class="source">
            {{ isUserAnnotation ? '用户标注' : 'Agent标注' }}
          </span>
        </div>
        <div class="ap-explanation">{{ annotation.explanation }}</div>
        <div v-if="annotation.motifRef" class="ap-motif-ref">
          母题引用: {{ annotation.motifRef }}
        </div>

        <!-- Overlapping annotation indicator -->
        <div v-if="overlappingAnnotation && isUserAnnotation" class="ap-overlap">
          <div class="ap-overlap-line" />
          <span class="ap-overlap-text">下方有Agent标注重叠</span>
          <button
            class="ap-btn ap-btn-overlap"
            @click="emit('showOverlapping', overlappingAnnotation?.id as string)"
          >查看Agent标注</button>
        </div>

        <!-- User annotation actions -->
        <div v-if="isUserAnnotation" class="ap-actions">
          <button class="ap-btn" @click="enterEditMode">编辑</button>
          <button class="ap-btn" @click="onDelete">删除</button>
        </div>

        <!-- Agent annotation actions -->
        <div v-else class="ap-actions">
          <button class="ap-btn ap-btn-primary" @click="onAdopt">采纳</button>
          <button class="ap-btn" @click="onAdoptAndEdit">修改后采纳</button>
          <button class="ap-btn" @click="onDelete">删除</button>
          <button class="ap-btn" @click="askAbout">追问</button>
        </div>
      </template>
    </div>
  </Teleport>
</template>

<style scoped>
.annotation-popup-backdrop {
  position: fixed;
  inset: 0;
  z-index: 8000;
}

.annotation-popup {
  position: fixed;
  z-index: 8001;
  background: var(--color-bg-primary);
  border: 1px solid var(--color-border);
  border-radius: var(--radius-lg);
  padding: 10px 14px;
  min-width: 220px;
  max-width: 320px;
  box-shadow: var(--shadow-lg);
}

.ap-header {
  display: flex;
  align-items: center;
  gap: 6px;
  margin-bottom: 6px;
}

.ap-label {
  font-size: 0.85rem;
  color: var(--color-text-primary);
  font-weight: 500;
}

.ap-category {
  font-size: 0.65rem;
  background: var(--color-bg-tertiary);
  padding: 1px 6px;
  border-radius: var(--radius-sm);
  color: var(--color-text-secondary);
  border: 1px solid var(--color-border-light);
}

.ap-source-badge {
  font-size: 0.6rem;
  padding: 1px 5px;
  border-radius: var(--radius-sm);
  margin-left: auto;
  border: 1px solid transparent;
}

.ap-source-badge.user {
  background: var(--color-accent-light);
  color: var(--color-accent);
  border-color: var(--color-border);
}

.ap-source-badge.agent {
  background: var(--color-bg-tertiary);
  color: var(--color-annotation-term);
  border-color: var(--color-border);
}

.ap-explanation {
  font-size: 0.75rem;
  color: var(--color-text-secondary);
  line-height: 1.5;
}

.ap-motif-ref {
  margin-top: 4px;
  font-size: 0.7rem;
  color: var(--color-annotation-motif);
}

.ap-actions {
  display: flex;
  flex-wrap: wrap;
  gap: 4px;
  margin-top: 8px;
}

.ap-btn {
  padding: 3px 8px;
  font-size: 0.7rem;
  background: var(--color-bg-tertiary);
  color: var(--color-text-primary);
  border: 1px solid var(--color-border);
  border-radius: var(--radius-sm);
  cursor: pointer;
  transition: background-color var(--dur-fast) var(--ease), border-color var(--dur-fast) var(--ease), color var(--dur-fast) var(--ease);
}

.ap-btn:hover {
  background: var(--color-bg-hover);
  border-color: var(--color-border-dark);
}

.ap-btn-primary {
  background: var(--color-accent);
  color: #fff;
  border-color: var(--color-accent);
}

.ap-btn-primary:hover {
  background: var(--color-accent-hover);
  border-color: var(--color-accent-hover);
}

.ap-btn-overlap {
  background: var(--color-bg-tertiary);
  border-color: var(--color-border);
  color: var(--color-text-secondary);
}

.ap-btn-overlap:hover {
  background: var(--color-accent-light);
  border-color: var(--color-accent);
  color: var(--color-accent);
}

/* Overlapping indicator */
.ap-overlap {
  margin-top: 6px;
  padding-top: 6px;
  border-top: 1px solid var(--color-border-light);
}

.ap-overlap-line {
  display: none;
}

.ap-overlap-text {
  font-size: 0.65rem;
  color: var(--color-text-tertiary);
  display: inline-block;
  margin-right: 6px;
}

/* Edit mode */
.ap-edit-section {
  margin-bottom: 8px;
}

.ap-edit-label {
  font-size: 0.7rem;
  color: var(--color-text-secondary);
  margin-bottom: 4px;
}

.ap-edit-categories {
  display: flex;
  flex-wrap: wrap;
  gap: 6px;
  font-size: 0.7rem;
  color: var(--color-text-primary);
}

.ap-edit-cat {
  display: flex;
  align-items: center;
  gap: 3px;
  cursor: pointer;
}

.ap-edit-cat input {
  accent-color: var(--color-accent);
}

.ap-edit-colors {
  display: flex;
  align-items: center;
  gap: 4px;
}

.ap-color-swatch {
  width: 16px;
  height: 16px;
  border-radius: 50%;
  cursor: pointer;
  border: 2px solid transparent;
  transition: border-color var(--dur-fast) var(--ease), transform var(--dur-fast) var(--ease);
}

.ap-color-swatch:hover {
  transform: scale(1.1);
}

.ap-color-swatch.selected {
  border-color: var(--color-border-dark);
}

.ap-color-input {
  width: 20px;
  height: 20px;
  border: 1px solid var(--color-border);
  border-radius: var(--radius-sm);
  cursor: pointer;
  padding: 0;
  background: var(--color-bg-primary);
}

.ap-edit-actions {
  display: flex;
  gap: 4px;
  margin-top: 8px;
}

.ap-btn-cancel {
  padding: 3px 10px;
  font-size: 0.7rem;
  background: var(--color-bg-tertiary);
  color: var(--color-text-primary);
  border: 1px solid var(--color-border);
  border-radius: var(--radius-sm);
  cursor: pointer;
  transition: background-color var(--dur-fast) var(--ease), border-color var(--dur-fast) var(--ease);
}

.ap-btn-cancel:hover {
  background: var(--color-bg-hover);
  border-color: var(--color-border-dark);
}

.ap-btn-save {
  padding: 3px 10px;
  font-size: 0.7rem;
  background: var(--color-accent);
  color: #fff;
  border: none;
  border-radius: var(--radius-sm);
  cursor: pointer;
  transition: background-color var(--dur-fast) var(--ease);
}

.ap-btn-save:hover {
  background: var(--color-accent-hover);
}
</style>
