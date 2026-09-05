<script setup lang="ts">
import { ref, watch } from 'vue'
import type { AnnotationCategory } from '@/types/annotation'

const props = defineProps<{
  visible: boolean
  selectedText: string
  startChar?: number
  endChar?: number
  preselectedCategory?: AnnotationCategory | null
}>()

const emit = defineEmits<{
  confirm: [note: string, category: AnnotationCategory, color: string, span: { startChar: number; endChar: number }]
  cancel: []
}>()

const PRESET_COLORS: Record<string, string> = {
  person: '#68A0C8',
  location: '#78B898',
  event: '#68B0B8',
  term: '#9888C0',
  motif: '#C898B0',
  custom: '#B8B878',
}

const note = ref('')
const category = ref<AnnotationCategory>('person')
const selectedColor = ref(PRESET_COLORS.person)
const showCustomColorPicker = ref(false)

// CU-04: 
watch(() => props.preselectedCategory, (cat) => {
  if (cat) {
    category.value = cat
    if (PRESET_COLORS[cat]) {
      selectedColor.value = PRESET_COLORS[cat]
    }
  }
}, { immediate: true })

function onCategoryChange(cat: AnnotationCategory) {
  category.value = cat
  if (PRESET_COLORS[cat]) {
    selectedColor.value = PRESET_COLORS[cat]
  }
}

function confirm() {
  const span = {
    startChar: props.startChar ?? 0,
    endChar: props.endChar ?? props.selectedText.length,
  }
  emit('confirm', note.value, category.value, selectedColor.value, span)
}
</script>

<template>
  <Teleport to="body">
    <div
      v-if="visible"
      class="annotation-form-backdrop"
      @click="emit('cancel')"
    />
    <div v-if="visible" class="annotation-form">
      <div class="af-header">新建标注</div>
      <div class="af-text">"{{ selectedText }}"</div>

      <div class="af-category">
        <label
          v-for="c in (['person','location','event','term','motif','custom'] as AnnotationCategory[])"
          :key="c"
          class="af-cat-label"
        >
          <input
            type="radio"
            :value="c"
            :checked="category === c"
            @change="onCategoryChange(c)"
          />
          <span
            class="af-cat-dot"
            :style="{ background: PRESET_COLORS[c] }"
          />
          {{ ({person:'人物',location:'地点',event:'事件',term:'术语',motif:'母题',custom:'自定义'} as Record<string,string>)[c] }}
        </label>
      </div>

      <!-- Color preset swatches -->
      <div class="af-colors">
        <span class="af-colors-label">颜色:</span>
        <span
          v-for="(color, name) in PRESET_COLORS"
          :key="name"
          class="af-color-swatch"
          :class="{ selected: selectedColor === color }"
          :style="{ background: color }"
          @click="selectedColor = color"
        />
        <button
          class="af-custom-color-btn"
          @click="showCustomColorPicker = !showCustomColorPicker"
        >自定义...</button>
        <input
          v-if="showCustomColorPicker"
          v-model="selectedColor"
          type="color"
          class="af-color-input"
        />
      </div>

      <textarea
        v-model="note"
        class="af-textarea"
        placeholder="批注内容..."
        rows="3"
      />
      <div class="af-actions">
        <button class="af-cancel-btn" @click="emit('cancel')">取消</button>
        <button class="af-confirm-btn" @click="confirm">确认标注</button>
      </div>
    </div>
  </Teleport>
</template>

<style scoped>
.annotation-form-backdrop {
  position: fixed;
  inset: 0;
  z-index: 8000;
  background: rgba(0, 0, 0, 0.15);
}

.annotation-form {
  position: fixed;
  top: 50%;
  left: 50%;
  transform: translate(-50%, -50%);
  z-index: 8001;
  background: var(--color-bg-primary);
  border: 1px solid var(--color-border);
  border-radius: var(--radius-lg);
  padding: 16px;
  min-width: 340px;
  box-shadow: var(--shadow-lg);
}

.af-header {
  font-size: 0.9rem;
  color: var(--color-text-primary);
  margin-bottom: 8px;
  font-weight: 500;
}

.af-text {
  font-size: 0.8rem;
  color: var(--color-text-secondary);
  margin-bottom: 10px;
  padding: 6px 8px;
  background: var(--color-bg-secondary);
  border-left: 3px solid var(--color-accent);
  border-radius: var(--radius-sm);
  font-style: italic;
}

.af-category {
  display: flex;
  gap: 10px;
  margin-bottom: 10px;
  font-size: 0.75rem;
  color: var(--color-text-primary);
  flex-wrap: wrap;
}

.af-cat-label {
  display: flex;
  align-items: center;
  gap: 4px;
  cursor: pointer;
  padding: 2px 4px;
  border-radius: var(--radius-sm);
  transition: background-color var(--dur-fast) var(--ease);
}

.af-cat-label:hover {
  background: var(--color-bg-hover);
}

.af-cat-label input {
  accent-color: var(--color-accent);
}

.af-cat-dot {
  width: 8px;
  height: 8px;
  border-radius: 50%;
  display: inline-block;
  box-shadow: 0 0 0 1px var(--color-border-light);
}

.af-colors {
  display: flex;
  align-items: center;
  gap: 6px;
  margin-bottom: 10px;
}

.af-colors-label {
  font-size: 0.7rem;
  color: var(--color-text-secondary);
}

.af-color-swatch {
  width: 18px;
  height: 18px;
  border-radius: 50%;
  cursor: pointer;
  border: 2px solid transparent;
  transition: border-color var(--dur-fast) var(--ease), transform var(--dur-fast) var(--ease);
}

.af-color-swatch:hover {
  transform: scale(1.1);
}

.af-color-swatch.selected {
  border-color: var(--color-border-dark);
}

.af-custom-color-btn {
  background: var(--color-bg-tertiary);
  border: 1px solid var(--color-border);
  color: var(--color-text-secondary);
  border-radius: var(--radius-sm);
  cursor: pointer;
  font-size: 0.7rem;
  padding: 2px 6px;
  transition: background-color var(--dur-fast) var(--ease), border-color var(--dur-fast) var(--ease), color var(--dur-fast) var(--ease);
}

.af-custom-color-btn:hover {
  background: var(--color-accent-light);
  border-color: var(--color-accent);
  color: var(--color-accent);
}

.af-color-input {
  width: 24px;
  height: 24px;
  border: 1px solid var(--color-border);
  border-radius: var(--radius-sm);
  cursor: pointer;
  padding: 0;
  background: var(--color-bg-primary);
}

.af-textarea {
  width: 100%;
  background: var(--color-bg-primary);
  border: 1px solid var(--color-border);
  border-radius: var(--radius);
  color: var(--color-text-primary);
  padding: 8px;
  font-size: 0.78rem;
  resize: vertical;
  box-sizing: border-box;
  font-family: var(--font-sans);
  outline: none;
  transition: border-color var(--dur-fast) var(--ease), box-shadow var(--dur-fast) var(--ease);
}

.af-textarea:focus {
  border-color: var(--color-accent);
  box-shadow: var(--shadow-focus);
}

.af-actions {
  display: flex;
  justify-content: flex-end;
  gap: 8px;
  margin-top: 10px;
}

.af-cancel-btn {
  padding: 5px 14px;
  font-size: 0.75rem;
  background: var(--color-bg-tertiary);
  color: var(--color-text-primary);
  border: 1px solid var(--color-border);
  border-radius: var(--radius);
  cursor: pointer;
  transition: background-color var(--dur-fast) var(--ease), border-color var(--dur-fast) var(--ease);
}

.af-cancel-btn:hover {
  background: var(--color-bg-hover);
  border-color: var(--color-border-dark);
}

.af-confirm-btn {
  padding: 5px 14px;
  font-size: 0.75rem;
  background: var(--color-accent);
  color: #fff;
  border: none;
  border-radius: var(--radius);
  cursor: pointer;
  transition: background-color var(--dur-fast) var(--ease);
}

.af-confirm-btn:hover {
  background: var(--color-accent-hover);
}
</style>
