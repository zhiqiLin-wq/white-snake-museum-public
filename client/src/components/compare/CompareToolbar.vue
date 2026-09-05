<script setup lang="ts">
import { computed } from 'vue'
import { useDynastyCompareStore } from '@/stores/dynastyCompare'

const props = defineProps<{
  // B1: 左右面板标题（朝代·章名）
  leftTitle?: string
  rightTitle?: string
  // B1: AI 是否已分析徽章显示
  leftAiReady?: boolean
  rightAiReady?: boolean
  // B3: 旁注展开/折叠（v-model 双向绑定）
  marginaliaExpanded?: boolean
}>()

const emit = defineEmits<{
  'toggle-sync': []
  'toggle-align': []
  'swap': []
  'analyze': []
  'export': []
  'update:marginaliaExpanded': [value: boolean]
}>()

const store = useDynastyCompareStore()

const categoryColors: Record<string, string> = {
  person: '#6DA8D6',
  location: '#7EC4A0',
  event: '#6FB9C4',
  term: '#9AA6D4',
  motif: '#D09EBC',
}

function lighten(hex: string, ratio = 0.88): string {
  // hex #RGB/RRGGBB → 混入白 ratio 比例
  const c = hex.replace('#', '')
  const full = c.length === 3 ? c.split('').map(x => x + x).join('') : c
  const r = parseInt(full.slice(0, 2), 16)
  const g = parseInt(full.slice(2, 4), 16)
  const b = parseInt(full.slice(4, 6), 16)
  const mr = Math.round(r + (255 - r) * ratio)
  const mg = Math.round(g + (255 - g) * ratio)
  const mb = Math.round(b + (255 - b) * ratio)
  return `rgb(${mr},${mg},${mb})`
}

const categories = [
  { key: 'person', label: '人物' },
  { key: 'location', label: '地点' },
  { key: 'motif', label: '母题' },
  { key: 'term', label: '术语' },
  { key: 'event', label: '事件' },
]

// U-14: Use compareState for button behavior
const isAnalyzing = computed(() => store.compareState === 'analyzing')
const canAnalyze = computed(() => store.compareState !== 'analyzing')

function isCategoryActive(key: string): boolean {
  return store.categoryFilter.has(key)
}

function toggleCategory(key: string) {
  store.toggleCategoryFilter(key)
}

function toggleMarginalia() {
  emit('update:marginaliaExpanded', !props.marginaliaExpanded)
}
</script>

<template>
  <div class="compare-toolbar">
    <!-- B1: 左侧栏 = 左章标题 + AI已分析徽记 + 工具按钮 -->
    <div class="ct-left">
      <div class="ct-title-block ct-title-block-left">
        <span class="ct-side-title" :title="leftTitle">{{ leftTitle || '左章' }}</span>
        <span v-if="leftAiReady" class="ct-ai-badge" title="AI 分析完成">AI已分析</span>
      </div>
      <div class="ct-sep" v-if="leftTitle"></div>
      <button
        class="ct-btn"
        :class="{ active: store.syncScroll }"
        title="同步滚动"
        @click="store.toggleSyncScroll(); emit('toggle-sync')"
      >{{ store.syncScroll ? '同步:开' : '同步:关' }}</button>
      <button
        class="ct-btn"
        :class="{ active: store.compareConfig?.alignParagraphs }"
        title="段落对齐"
        @click="emit('toggle-align')"
      >段落对齐</button>
      <button
        class="ct-btn"
        title="左右互换"
        @click="store.swapSides(); emit('swap')"
      >互换</button>
      <button
        class="ct-btn"
        :class="{ active: marginaliaExpanded }"
        title="显示/隐藏旁注气泡（关闭时以数字徽章折叠）"
        @click="toggleMarginalia"
      >旁注</button>
    </div>

    <div class="ct-categories">
      <button
        v-for="cat in categories"
        :key="cat.key"
        class="ct-cat-btn"
        :class="{ active: isCategoryActive(cat.key) }"
        :style="isCategoryActive(cat.key) ? {
          borderLeftColor: categoryColors[cat.key],
          background: lighten(categoryColors[cat.key], 0.9),
          color: 'var(--color-text-primary)',
        } : {}"
        @click="toggleCategory(cat.key)"
      >
        <span
          class="ct-cat-dot"
          :style="{ background: isCategoryActive(cat.key) ? categoryColors[cat.key] : 'var(--color-text-placeholder)' }"
        ></span>
        {{ cat.label }}
      </button>
    </div>

    <!-- B1: 右侧栏 = 工具按钮 + 右章 AI徽记 + 右章标题 -->
    <div class="ct-right">
      <button
        class="ct-btn ct-analyze-btn"
        :disabled="!canAnalyze"
        @click="emit('analyze')"
      >
        {{ isAnalyzing ? '分析中...' : 'AI 分析差异' }}
      </button>
      <button
        class="ct-btn ct-export-btn"
        @click="emit('export')"
      >导出</button>
      <div class="ct-sep" v-if="rightTitle"></div>
      <div class="ct-title-block ct-title-block-right">
        <span v-if="rightAiReady" class="ct-ai-badge" title="AI 分析完成">AI已分析</span>
        <span class="ct-side-title" :title="rightTitle">{{ rightTitle || '右章' }}</span>
      </div>
    </div>
  </div>
</template>

<style scoped>
.compare-toolbar {
  display: flex;
  align-items: center;
  /* 自适应高度 + 换行：宽度不足时分类按钮换行到第二行，不再与两侧标题/按钮重叠 */
  flex-wrap: wrap;
  row-gap: 6px;
  min-height: 40px;
  padding: 6px 12px;
  background: var(--color-bg-secondary, #F7F8FA);
  border-bottom: 1px solid var(--color-border, #E5E7EB);
  gap: 8px;
  flex-shrink: 0;
}

.ct-left {
  display: flex;
  align-items: center;
  gap: 4px;
  min-width: 0;
  flex: 0 1 auto;
  flex-wrap: wrap;
  row-gap: 4px;
}

.ct-categories {
  display: flex;
  flex-wrap: wrap;
  gap: 4px;
  row-gap: 4px;
  /* 分类放在中间，不挤占两侧标题空间 */
  margin: 0 auto;
}

.ct-right {
  display: flex;
  align-items: center;
  gap: 4px;
  min-width: 0;
  flex: 0 1 auto;
  flex-wrap: wrap;
  row-gap: 4px;
  margin-left: auto;
}

/* B1: 两端章标题块 */
.ct-title-block {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  max-width: 220px;
  min-width: 0;
  /* 空间不足时先收缩标题（ellipsis），而不是溢出压盖其他元素 */
  flex: 0 1 auto;
  height: 28px;
  padding: 0 8px;
  background: var(--color-bg-primary);
  border: 1px solid var(--color-border);
  border-radius: var(--radius-sm, 4px);
}
.ct-title-block-right {
  justify-content: flex-end;
}

.ct-side-title {
  font-size: 0.75rem;
  font-weight: 600;
  color: var(--color-text-primary);
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
  max-width: 160px;
  letter-spacing: 0.01em;
}

/* B1: AI 已分析徽章（从 dc-panel-header 移至此处） */
.ct-ai-badge {
  font-size: 0.6rem;
  padding: 1px 6px;
  background: var(--color-success-light);
  color: var(--color-verdict-supported);
  border: 1px solid var(--color-verdict-supported);
  border-radius: var(--radius-sm);
  flex-shrink: 0;
  font-weight: 600;
  letter-spacing: 0;
  line-height: 1.4;
}

/* 左右标题与按钮之间的细分割线 */
.ct-sep {
  width: 1px;
  height: 18px;
  background: var(--color-border);
  margin: 0 4px;
  flex-shrink: 0;
}

.ct-btn {
  padding: 3px 10px;
  font-size: 0.72rem;
  background: var(--color-bg-tertiary, #F0F2F5);
  color: var(--color-text-secondary, #4B5563);
  border: 1px solid var(--color-border, #E5E7EB);
  border-radius: var(--radius-sm, 4px);
  cursor: pointer;
  white-space: nowrap;
  flex-shrink: 0;
  transition: background var(--dur-fast, 120ms) var(--ease, cubic-bezier(.22,.61,.36,1)),
              color var(--dur-fast, 120ms) var(--ease, cubic-bezier(.22,.61,.36,1)),
              border-color var(--dur-fast, 120ms) var(--ease, cubic-bezier(.22,.61,.36,1));
}

.ct-btn:hover:not(:disabled) {
  background: var(--color-bg-hover, #F2F4F7);
  color: var(--color-text-primary, #1A1A1A);
  border-color: var(--color-border-dark, #D1D5DB);
}

.ct-btn.active {
  background: var(--color-accent, #5B9BD5);
  color: #FFFFFF;
  border-color: var(--color-accent, #5B9BD5);
}

.ct-btn.active:hover:not(:disabled) {
  background: var(--color-accent-hover, #4A8BC5);
  border-color: var(--color-accent-hover, #4A8BC5);
}

.ct-btn:disabled {
  opacity: 0.6;
  cursor: not-allowed;
}

.ct-cat-btn {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  padding: 3px 10px 3px 8px;
  height: 26px;
  font-size: 0.72rem;
  font-weight: 500;
  background: var(--color-bg-primary, #FFFFFF);
  color: var(--color-text-secondary, #4B5563);
  border: 1px solid var(--color-border, #E5E7EB);
  border-left-width: 2px;
  border-left-color: var(--color-border-dark, #D1D5DB);
  border-radius: var(--radius-sm, 4px);
  cursor: pointer;
  flex-shrink: 0;
  transition: background var(--dur-fast, 120ms) var(--ease, cubic-bezier(.22,.61,.36,1)),
              color var(--dur-fast, 120ms) var(--ease, cubic-bezier(.22,.61,.36,1)),
              border-color var(--dur-fast, 120ms) var(--ease, cubic-bezier(.22,.61,.36,1));
}
.ct-cat-dot {
  display: inline-block;
  width: 7px; height: 7px;
  border-radius: 999px;
  flex-shrink: 0;
  transition: background var(--dur-fast, 120ms) var(--ease, cubic-bezier(.22,.61,.36,1));
}

.ct-cat-btn:hover:not(.active) {
  background: var(--color-bg-hover, #F2F4F7);
  color: var(--color-text-primary, #1A1A1A);
  border-color: var(--color-border-dark, #D1D5DB);
}

.ct-cat-btn.active {
  border-left-width: 3px;
  box-shadow: 0 1px 2px rgba(0,0,0,0.04);
}

.ct-analyze-btn {
  background: var(--color-accent, #5B9BD5);
  color: #FFFFFF;
  border-color: var(--color-accent, #5B9BD5);
}

.ct-analyze-btn:hover:not(:disabled) {
  background: var(--color-accent-hover, #4A8BC5);
  border-color: var(--color-accent-hover, #4A8BC5);
  color: #FFFFFF;
}

.ct-export-btn {
  border-color: var(--color-border, #E5E7EB);
  color: var(--color-text-secondary, #4B5563);
}

.ct-export-btn:hover:not(:disabled) {
  border-color: var(--color-accent, #5B9BD5);
  color: var(--color-accent, #5B9BD5);
}

/* 工具栏窄屏自适应：标题隐藏，只保留按钮 */
@media (max-width: 1280px) {
  .ct-side-title { max-width: 120px; }
  .ct-title-block { max-width: 180px; }
}
@media (max-width: 1080px) {
  .ct-side-title { max-width: 92px; }
  .ct-title-block { max-width: 148px; }
  .ct-categories { margin: 0 4px; }
}
@media (max-width: 900px) {
  .ct-title-block { display: none; }
  .ct-sep { display: none; }
}
</style>
