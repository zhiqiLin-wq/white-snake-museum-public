<script setup lang="ts">
import { ref } from 'vue'

export interface ChapterInfo {
  number: number
  title: string
  dynasty: string
  hasAnnotations: boolean
}

const props = defineProps<{
  visible: boolean
  chapterNumber: number
  allChapters?: ChapterInfo[]
  // U-09: Annotation counts for shortcut button labels
  userAnnotationCount?: number
  agentAnnotationCount?: number
}>()

const emit = defineEmits<{
  export: [config: { format: string; scope: string; includeUser: boolean; includeAgent: boolean; includeMarginalia: boolean; teiMode?: string; selectedChapters?: number[] }]
  cancel: []
}>()

const format = ref<'html' | 'pdf' | 'plaintext' | 'csv' | 'json' | 'tei'>('html')
const scope = ref<'current' | 'all' | 'custom'>('current')
const includeUser = ref(true)
const includeAgent = ref(false)
const includeMarginalia = ref(true)
const teiMode = ref<'standOff' | 'inline'>('standOff')
const selectedChapters = ref<number[]>([])

// U-09: Shortcut button handler
function setIncludeShortcut(user: boolean, agent: boolean) {
  includeUser.value = user
  includeAgent.value = agent
}

function toggleChapter(chNum: number) {
  const idx = selectedChapters.value.indexOf(chNum)
  if (idx >= 0) {
    selectedChapters.value.splice(idx, 1)
  } else {
    selectedChapters.value.push(chNum)
  }
}

function doExport() {
  const scopeChapters = scope.value === 'current'
    ? [props.chapterNumber]
    : scope.value === 'custom'
      ? selectedChapters.value
      : (props.allChapters || []).map(c => c.number)

  emit('export', {
    format: format.value,
    scope: scope.value,
    includeUser: includeUser.value,
    includeAgent: includeAgent.value,
    includeMarginalia: includeMarginalia.value,
    teiMode: teiMode.value,
    selectedChapters: scopeChapters,
  })
}
</script>

<template>
  <Teleport to="body">
    <div
      v-if="visible"
      class="export-panel-backdrop"
      @click="emit('cancel')"
    />
    <div v-if="visible" class="export-panel">
      <div class="ep-header">导出标注文献</div>

      <div class="ep-section">
        <div class="ep-label">导出格式</div>
        <div class="ep-options">
          <label class="ep-option">
            <input type="radio" v-model="format" value="html" /> HTML (保留高亮和批注)
          </label>
          <label class="ep-option">
            <input type="radio" v-model="format" value="pdf" /> PDF (打印优化)
          </label>
          <label class="ep-option">
            <input type="radio" v-model="format" value="plaintext" /> 纯文本 + 标注清单
          </label>
          <label class="ep-option">
            <input type="radio" v-model="format" value="csv" /> CSV
          </label>
          <label class="ep-option">
            <input type="radio" v-model="format" value="json" /> JSON
          </label>
          <label class="ep-option">
            <input type="radio" v-model="format" value="tei" /> TEI-XML
          </label>
        </div>
      </div>

      <div class="ep-section">
        <div class="ep-label">导出范围</div>
        <div class="ep-options">
          <label class="ep-option">
            <input type="radio" v-model="scope" value="current" /> 当前章节
          </label>
          <label class="ep-option">
            <input type="radio" v-model="scope" value="all" /> 所有已标注章节
          </label>
          <label class="ep-option">
            <input type="radio" v-model="scope" value="custom" /> 自定义选择
          </label>
        </div>
      </div>

      <!-- Custom chapter selection -->
      <div v-if="scope === 'custom' && allChapters && allChapters.length > 0" class="ep-section">
        <div class="ep-label">选择章节</div>
        <div class="ep-chapter-list">
          <label
            v-for="ch in allChapters"
            :key="ch.number"
            class="ep-option ep-chapter-option"
          >
            <input
              type="checkbox"
              :checked="selectedChapters.includes(ch.number)"
              @change="toggleChapter(ch.number)"
            />
            {{ ch.dynasty }} - {{ ch.title }}
            <span v-if="!ch.hasAnnotations" class="ep-no-annotations">(无标注)</span>
          </label>
        </div>
      </div>

      <div class="ep-section">
        <div class="ep-label">包含内容</div>
        <!-- U-09: Shortcut buttons -->
        <div class="ep-shortcuts">
          <button
            class="ep-shortcut-btn"
            :class="{ active: includeUser && !includeAgent }"
            @click="setIncludeShortcut(true, false)"
          >
            仅用户标注{{ userAnnotationCount !== undefined ? ` (${userAnnotationCount})` : '' }}
          </button>
          <button
            class="ep-shortcut-btn"
            :class="{ active: !includeUser && includeAgent }"
            @click="setIncludeShortcut(false, true)"
          >
            仅Agent标注{{ agentAnnotationCount !== undefined ? ` (${agentAnnotationCount})` : '' }}
          </button>
          <button
            class="ep-shortcut-btn"
            :class="{ active: includeUser && includeAgent }"
            @click="setIncludeShortcut(true, true)"
          >
            全部包含
          </button>
        </div>
        <div class="ep-checkboxes">
          <label class="ep-option">
            <input type="checkbox" v-model="includeUser" /> 用户标注
          </label>
          <label class="ep-option">
            <input type="checkbox" v-model="includeAgent" /> Agent标注
          </label>
          <label class="ep-option">
            <input type="checkbox" v-model="includeMarginalia" /> 批注
          </label>
        </div>
      </div>

      <!-- TEI mode selector -->
      <div v-if="format === 'tei'" class="ep-section">
        <div class="ep-label">TEI模式</div>
        <div class="ep-options">
          <label class="ep-option">
            <input type="radio" v-model="teiMode" value="standOff" /> Stand-off (标注与文本分离)
          </label>
          <label class="ep-option">
            <input type="radio" v-model="teiMode" value="inline" /> Inline (标注嵌入文本)
          </label>
        </div>
      </div>

      <!-- Format-specific preview info -->
      <div class="ep-section ep-preview-info">
        <div v-if="format === 'csv'" class="ep-info">
          CSV 格式: 17列数据含BOM头,兼容Excel打开
        </div>
        <div v-else-if="format === 'json'" class="ep-info">
          JSON 格式: 结构化标注数据,便于程序处理
        </div>
        <div v-else-if="format === 'tei'" class="ep-info">
          TEI P5 XML 格式: 数字人文标准TEI标注
        </div>
        <div v-else class="ep-info">
          {{ format === 'html' ? 'HTML 格式: 保留彩色高亮和批注气泡' : format === 'pdf' ? 'PDF 格式: 通过浏览器打印生成' : '' }}
        </div>
      </div>

      <div class="ep-actions">
        <button class="ep-btn-cancel" @click="emit('cancel')">取消</button>
        <button class="ep-btn-export" @click="doExport">导出</button>
      </div>
    </div>
  </Teleport>
</template>

<style scoped>
.export-panel-backdrop {
  position: fixed;
  inset: 0;
  z-index: 8000;
  background: rgba(0, 0, 0, 0.15);
}

.export-panel {
  position: fixed;
  top: 50%;
  left: 50%;
  transform: translate(-50%, -50%);
  z-index: 8001;
  background: var(--color-bg-primary);
  border: 1px solid var(--color-border);
  border-radius: var(--radius-lg);
  padding: 16px;
  min-width: 360px;
  max-width: 420px;
  box-shadow: var(--shadow-lg);
}

.ep-header {
  font-size: 0.9rem;
  color: var(--color-text-primary);
  margin-bottom: 12px;
  font-weight: 500;
}

.ep-section {
  margin-bottom: 10px;
}

.ep-label {
  font-size: 0.72rem;
  color: var(--color-text-secondary);
  margin-bottom: 4px;
}

.ep-shortcuts {
  display: flex;
  gap: 4px;
  margin-bottom: 6px;
}

.ep-shortcut-btn {
  padding: 2px 10px;
  font-size: 0.68rem;
  background: var(--color-bg-tertiary);
  color: var(--color-text-secondary);
  border: 1px solid var(--color-border);
  border-radius: var(--radius-sm);
  cursor: pointer;
  transition: background-color var(--dur-fast) var(--ease), border-color var(--dur-fast) var(--ease), color var(--dur-fast) var(--ease);
}

.ep-shortcut-btn:hover {
  background: var(--color-bg-hover);
  border-color: var(--color-border-dark);
  color: var(--color-text-primary);
}

.ep-shortcut-btn.active {
  background: var(--color-accent);
  color: #fff;
  border-color: var(--color-accent);
}

.ep-options, .ep-checkboxes {
  display: flex;
  flex-wrap: wrap;
  gap: 10px;
}

.ep-option {
  display: flex;
  align-items: center;
  gap: 4px;
  font-size: 0.78rem;
  color: var(--color-text-primary);
  cursor: pointer;
}

.ep-option input {
  accent-color: var(--color-accent);
}

.ep-chapter-list {
  max-height: 120px;
  overflow-y: auto;
  display: flex;
  flex-direction: column;
  gap: 2px;
  padding: 4px;
  border: 1px solid var(--color-border-light);
  border-radius: var(--radius);
  background: var(--color-bg-secondary);
}

.ep-chapter-option {
  padding: 2px 4px;
  border-radius: var(--radius-sm);
  transition: background-color var(--dur-fast) var(--ease);
}

.ep-chapter-option:hover {
  background: var(--color-bg-tertiary);
}

.ep-no-annotations {
  font-size: 0.65rem;
  color: var(--color-text-tertiary);
}

.ep-preview-info {
  padding: 6px 8px;
  background: var(--color-bg-secondary);
  border: 1px solid var(--color-border-light);
  border-radius: var(--radius);
}

.ep-info {
  font-size: 0.7rem;
  color: var(--color-text-secondary);
}

.ep-actions {
  display: flex;
  justify-content: flex-end;
  gap: 8px;
  margin-top: 14px;
}

.ep-btn-cancel {
  padding: 5px 14px;
  font-size: 0.75rem;
  background: var(--color-bg-tertiary);
  color: var(--color-text-primary);
  border: 1px solid var(--color-border);
  border-radius: var(--radius);
  cursor: pointer;
  transition: background-color var(--dur-fast) var(--ease), border-color var(--dur-fast) var(--ease);
}

.ep-btn-cancel:hover {
  background: var(--color-bg-hover);
  border-color: var(--color-border-dark);
}

.ep-btn-export {
  padding: 5px 14px;
  font-size: 0.75rem;
  background: var(--color-accent);
  color: #fff;
  border: none;
  border-radius: var(--radius);
  cursor: pointer;
  transition: background-color var(--dur-fast) var(--ease);
}

.ep-btn-export:hover {
  background: var(--color-accent-hover);
}
</style>
