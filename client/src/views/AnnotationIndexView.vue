<script setup lang="ts">
import { ref, computed } from 'vue'
import { useAnnotationStore } from '@/stores/annotations'
import { useSourceTreeStore } from '@/stores/sourceTree'
import { useWorkspaceStore } from '@/stores/workspace'
import type { AnnotationIndexItem } from '@/types/annotation'

const annotationStore = useAnnotationStore()
const sourceTreeStore = useSourceTreeStore()
const workspaceStore = useWorkspaceStore()

const searchQuery = ref('')
const searchDebounceTimer = ref<ReturnType<typeof setTimeout> | null>(null)
const dynastyFilter = ref<Set<string>>(new Set())
const categoryFilter = ref<Set<string>>(new Set())
const sourceFilter = ref<'all' | 'user' | 'agent'>('all')
const showWithNote = ref<boolean | null>(null)
const sortBy = ref<'chapter' | 'category' | 'created'>('chapter')
const expandedNoteId = ref<string | null>(null)
const contextMenuVisible = ref(false)
const contextMenuPos = ref({ x: 0, y: 0 })
const contextMenuItem = ref<AnnotationIndexItem | null>(null)

const dynasties = ['', '', '', '', '']
const categories = ['person', 'location', 'event', 'term', 'motif', 'custom']
const categoryLabels: Record<string, string> = {
  person: '', location: '', event: '', term: '', motif: '', custom: '',
}
const categoryColors: Record<string, string> = {
  person: '#6DA8D6', location: '#7EC4A0', event: '#6FB9C4', term: '#9AA6D4', motif: '#D09EBC', custom: '#B4B482',
}

const includeAgent = ref(true)
const allItems = computed(() => annotationStore.buildAnnotationIndex(includeAgent.value))

const filteredItems = computed(() => {
  let items = allItems.value

  if (searchQuery.value.trim()) {
    const q = searchQuery.value.trim().toLowerCase()
    items = items.filter(item =>
      item.label.toLowerCase().includes(q) ||
      item.text.toLowerCase().includes(q) ||
      item.contextSnippet.toLowerCase().includes(q) ||
      (item.note && item.note.toLowerCase().includes(q))
    )
  }

  if (dynastyFilter.value.size > 0) {
    items = items.filter(item => dynastyFilter.value.has(item.dynasty))
  }

  if (categoryFilter.value.size > 0) {
    items = items.filter(item => categoryFilter.value.has(item.category))
  }

  if (sourceFilter.value !== 'all') {
    items = items.filter(item => item.source === sourceFilter.value)
  }

  if (showWithNote.value === true) {
    items = items.filter(item => item.note || item.marginalia.length > 0)
  } else if (showWithNote.value === false) {
    items = items.filter(item => !item.note && item.marginalia.length === 0)
  }

  if (sortBy.value === 'chapter') {
    items = [...items].sort((a, b) => {
      if (a.chapterNumber !== b.chapterNumber) return a.chapterNumber - b.chapterNumber
      if (a.paragraphIndex !== b.paragraphIndex) return a.paragraphIndex - b.paragraphIndex
      return a.span.startChar - b.span.startChar
    })
  } else if (sortBy.value === 'category') {
    items = [...items].sort((a, b) => a.category.localeCompare(b.category))
  } else {
    items = [...items].sort((a, b) => b.createdAt - a.createdAt)
  }

  return items
})

const totalCount = computed(() => allItems.value.length)
const userCount = computed(() => allItems.value.filter(i => i.source === 'user').length)
const agentCount = computed(() => allItems.value.filter(i => i.source === 'agent').length)
const withNoteCount = computed(() => allItems.value.filter(i => i.note || i.marginalia.length > 0).length)
const coverageRate = computed(() => {
  const totalParas = sourceTreeStore.chapters.reduce((sum, ch) => sum + ch.paragraphs.length, 0)
  if (totalParas === 0) return 0
  const annotatedParas = new Set(allItems.value.map(i => `${i.chapterNumber}:${i.paragraphIndex}`)).size
  return Math.round((annotatedParas / totalParas) * 10000) / 100
})

function toggleDynasty(dynasty: string) {
  const s = new Set(dynastyFilter.value)
  if (s.has(dynasty)) s.delete(dynasty); else s.add(dynasty)
  dynastyFilter.value = s
}

function toggleCategory(cat: string) {
  const s = new Set(categoryFilter.value)
  if (s.has(cat)) s.delete(cat); else s.add(cat)
  categoryFilter.value = s
}

// 多彩标签淡底工具：hex → 混入白 ratio(0.9=极浅)
function lightenHex(hex: string, ratio = 0.9): string {
  const c = hex.replace('#', '')
  const full = c.length === 3 ? c.split('').map(x => x + x).join('') : c
  const r = parseInt(full.slice(0, 2), 16)
  const g = parseInt(full.slice(2, 4), 16)
  const b = parseInt(full.slice(4, 6), 16)
  return `rgb(${Math.round(r + (255 - r) * ratio)}, ${Math.round(g + (255 - g) * ratio)}, ${Math.round(b + (255 - b) * ratio)})`
}

// P2-15: Fix search debounce — actually apply the search
function onSearchInput() {
  if (searchDebounceTimer.value) clearTimeout(searchDebounceTimer.value)
  searchDebounceTimer.value = setTimeout(() => {
    // Trigger reactivity so filteredItems recomputes
    searchQuery.value = searchQuery.value
  }, 300)
}

function onRowClick(item: AnnotationIndexItem) {
  sourceTreeStore.setNavigationTarget({ chapterNumber: item.chapterNumber, paragraphIndex: item.paragraphIndex })
  workspaceStore.openTab('text-reader', `${item.dynasty} - ${item.chapterTitle}`, {
    chapterNumber: item.chapterNumber,
    paragraphIndex: item.paragraphIndex,
  })
}

// P1-15: Double-click — edit for user, adopt for agent
function onRowDoubleClick(item: AnnotationIndexItem) {
  if (item.source === 'user') {
    annotationStore.editingAnnotationId = item.id
  } else {
    // Agent: popup adopt confirmation
    if (confirm(` Agent ${item.label}?`)) {
      annotationStore.adoptAgentAnnotation(item.passageKey, item.id)
    }
  }
}

// P1-15: Right-click context menu
function onRowContextMenu(event: MouseEvent, item: AnnotationIndexItem) {
  event.preventDefault()
  contextMenuVisible.value = true
  contextMenuPos.value = { x: event.clientX, y: event.clientY }
  contextMenuItem.value = item
}

function closeContextMenu() {
  contextMenuVisible.value = false
  contextMenuItem.value = null
}

function contextMenuAction(action: string) {
  const item = contextMenuItem.value
  if (!item) return
  closeContextMenu()

  switch (action) {
    case 'detail':
      expandedNoteId.value = expandedNoteId.value === item.id ? null : item.id
      break
    case 'navigate':
      onRowClick(item)
      break
    case 'edit':
      if (item.source === 'user') {
        annotationStore.editingAnnotationId = item.id
      }
      break
    case 'delete':
      if (item.source === 'user') {
        if (confirm(`${item.label}?`)) {
          annotationStore.removeUserAnnotation(item.passageKey, item.id)
        }
      }
      break
  }
}

function exportCurrentResults(format: 'csv' | 'json') {
  const items = filteredItems.value
  if (format === 'csv') {
    const header = 'ID,Category,Label,Source,Passage,Chapter,Dynasty,Paragraph,Note'
    const rows = items.map(item => [
      item.id, item.categoryLabel, `"${item.label.replace(/"/g, '""')}"`, item.source,
      `"${item.passageKey}"`, item.chapterNumber, item.dynasty || '',
      item.paragraphIndex, `"${(item.note || '').replace(/"/g, '""')}"`
    ].join(','))
    const csv = '' + header + '\n' + rows.join('\n')
    downloadCSV(csv, 'annotation-index.csv')
  }
}

function downloadCSV(content: string, filename: string) {
  const blob = new Blob([content], { type: 'text/csv;charset=utf-8' })
  const url = URL.createObjectURL(blob)
  const a = document.createElement('a')
  a.href = url; a.download = filename; a.click()
  URL.revokeObjectURL(url)
}
</script>

<template>
  <div class="annotation-index-view" @click="closeContextMenu">
    <div class="ai-toolbar">
      <div class="ai-search">
        <input
          v-model="searchQuery"
          type="text"
          class="ai-search-input"
          placeholder="..."
          @input="onSearchInput"
        />
      </div>

      <div class="ai-filter-group">
        <span class="ai-filter-label">:</span>
        <button
          v-for="d in dynasties" :key="d"
          class="ai-filter-btn" :class="{ active: dynastyFilter.has(d) }"
          @click="toggleDynasty(d)"
        >{{ d }}</button>
      </div>

      <div class="ai-filter-group">
        <span class="ai-filter-label">分类:</span>
        <button
          v-for="cat in categories" :key="cat"
          class="ai-filter-btn ai-cat-btn" :class="{ active: categoryFilter.has(cat) }"
          :style="categoryFilter.has(cat) ? {
            borderLeftColor: categoryColors[cat],
            background: lightenHex(categoryColors[cat], 0.9),
          } : {}"
          @click="toggleCategory(cat)"
        >
          <span class="ai-cat-dot" :style="{ background: categoryFilter.has(cat) ? categoryColors[cat] : 'var(--color-text-placeholder)' }"></span>
          {{ categoryLabels[cat] }}
        </button>
      </div>

      <div class="ai-filter-group">
        <span class="ai-filter-label">:</span>
        <select v-model="sourceFilter" class="ai-select">
          <option value="all"></option>
          <option value="user"></option>
          <option value="agent">Agent</option>
        </select>
      </div>

      <div class="ai-filter-group">
        <label class="ai-checkbox">
          <input type="checkbox" v-model="showWithNote" :indeterminate="showWithNote === null" /> 
        </label>
      </div>

      <div class="ai-sort">
        <span class="ai-filter-label">:</span>
        <select v-model="sortBy" class="ai-select">
          <option value="chapter"></option>
          <option value="category"></option>
          <option value="created"></option>
        </select>
      </div>

      <div class="ai-export">
        <button class="ai-export-btn" @click="exportCurrentResults('csv')"> CSV</button>
        <button class="ai-export-btn" @click="exportCurrentResults('json')"> JSON</button>
      </div>
    </div>

    <div class="ai-summary-row">
       {{ totalCount }} /  {{ userCount }} / Agent {{ agentCount }} /  {{ withNoteCount }} /  {{ coverageRate }}% /  {{ filteredItems.length }}
    </div>

    <div class="ai-table-wrapper">
      <table class="ai-table" v-if="filteredItems.length > 0">
        <thead>
          <tr>
            <th class="ai-col-idx">#</th>
            <th class="ai-col-cat"></th>
            <th class="ai-col-label"></th>
            <th class="ai-col-text"></th>
            <th class="ai-col-dynasty"></th>
            <th class="ai-col-source"></th>
            <th class="ai-col-note"></th>
          </tr>
        </thead>
        <tbody>
          <tr
            v-for="(item, idx) in filteredItems"
            :key="item.id"
            class="ai-row"
            @click="onRowClick(item)"
            @dblclick="onRowDoubleClick(item)"
            @contextmenu="onRowContextMenu($event, item)"
          >
            <td class="ai-col-idx">{{ idx + 1 }}</td>
            <td class="ai-col-cat">
              <span class="ai-cat-dot" :style="{ background: item.color }"></span>
              {{ item.categoryLabel }}
            </td>
            <td class="ai-col-label">{{ item.label }}</td>
            <td class="ai-col-text" :title="item.contextSnippet">{{ item.contextSnippet.slice(0, 40) }}{{ item.contextSnippet.length > 40 ? '...' : '' }}</td>
            <td class="ai-col-dynasty">{{ item.dynasty }}</td>
            <td class="ai-col-source">
              <span class="ai-source-icon" :class="item.source === 'user' ? 'ai-source-user' : 'ai-source-agent'">
                {{ item.source === 'user' ? '' : '' }}
              </span>
            </td>
            <td class="ai-col-note">
              <span v-if="item.marginalia.length > 0" class="ai-has-note"></span>
              <span v-else-if="item.note" class="ai-has-note"></span>
              <span v-else class="ai-no-note">-</span>
            </td>
          </tr>
        </tbody>
      </table>
      <div v-else class="ai-empty">
        
        <button
          v-if="searchQuery || dynastyFilter.size > 0 || categoryFilter.size > 0 || sourceFilter !== 'all'"
          class="ai-clear-btn"
          @click="searchQuery = ''; dynastyFilter = new Set(); categoryFilter = new Set(); sourceFilter = 'all'; showWithNote = null"
        ></button>
      </div>
    </div>

    <!-- P1-15: Context menu -->
    <Teleport to="body">
      <div
        v-if="contextMenuVisible && contextMenuItem"
        class="ai-context-menu"
        :style="{ left: contextMenuPos.x + 'px', top: contextMenuPos.y + 'px' }"
        @click.stop
      >
        <div class="ai-cm-item" @click="contextMenuAction('detail')"></div>
        <div class="ai-cm-item" @click="contextMenuAction('navigate')"></div>
        <div v-if="contextMenuItem.source === 'user'" class="ai-cm-item" @click="contextMenuAction('edit')">/</div>
        <div v-if="contextMenuItem.source === 'user'" class="ai-cm-item ai-cm-danger" @click="contextMenuAction('delete')"></div>
      </div>
    </Teleport>
  </div>
</template>

<style scoped>
.annotation-index-view {
  display: flex;
  flex-direction: column;
  height: 100%;
  overflow: hidden;
  background: var(--color-bg-primary);
}

/* ---------- Toolbar ---------- */
.ai-toolbar {
  padding: var(--space-2) var(--space-4);
  border-bottom: 1px solid var(--color-border);
  background: var(--color-bg-secondary);
  display: flex;
  flex-wrap: wrap;
  gap: var(--space-2);
  align-items: center;
}

.ai-search-input {
  padding: var(--space-1) var(--space-2);
  border: 1px solid var(--color-border);
  border-radius: var(--radius-sm);
  background: var(--color-bg-primary);
  color: var(--color-text-primary);
  font-size: 0.78rem;
  width: 180px;
  transition: border-color var(--dur-fast) var(--ease),
              box-shadow var(--dur-fast) var(--ease);
}

.ai-search-input:focus {
  outline: none;
  border-color: var(--color-accent);
  box-shadow: var(--shadow-focus);
}

.ai-search-input::placeholder {
  color: var(--color-text-placeholder);
}

.ai-filter-group {
  display: flex;
  align-items: center;
  gap: var(--space-1);
}

.ai-filter-label {
  font-size: 0.68rem;
  color: var(--color-text-tertiary);
  font-weight: 500;
}

.ai-filter-btn {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  padding: 3px 10px 3px 8px;
  height: 26px;
  font-size: 0.72rem;
  font-weight: 500;
  background: var(--color-bg-primary);
  border: 1px solid var(--color-border);
  border-left-width: 2px;
  border-left-color: var(--color-border-dark);
  border-radius: var(--radius-sm);
  color: var(--color-text-secondary);
  cursor: pointer;
  white-space: nowrap;
  transition: background-color var(--dur-fast) var(--ease),
              color var(--dur-fast) var(--ease),
              border-color var(--dur-fast) var(--ease);
}

.ai-filter-btn:hover {
  background: var(--color-bg-hover);
  color: var(--color-text-primary);
  border-color: var(--color-border-dark);
}

/* 朝代/通用 Filter：激活用淡蓝点缀 */
.ai-filter-btn.active {
  background: var(--color-accent-light);
  color: var(--color-accent-active);
  border-color: var(--color-accent);
  border-left-width: 3px;
}

/* 分类 Filter：激活用分类自己的多彩淡底 + 左边条，不走 accent 蓝
   注意：实际背景/左边条色由模板内联 style 覆盖，以下为结构性样式 */
.ai-cat-btn.active {
  color: var(--color-text-primary);
  border-left-width: 3px;
  box-shadow: 0 1px 2px rgba(0,0,0,0.04);
}
.ai-cat-dot {
  display: inline-block;
  width: 7px; height: 7px;
  border-radius: 999px;
  flex-shrink: 0;
  transition: background var(--dur-fast) var(--ease);
}

.ai-select {
  padding: var(--space-1);
  font-size: 0.68rem;
  background: var(--color-bg-primary);
  border: 1px solid var(--color-border);
  border-radius: var(--radius-sm);
  color: var(--color-text-primary);
  cursor: pointer;
  transition: border-color var(--dur-fast) var(--ease);
}

.ai-select:focus {
  outline: none;
  border-color: var(--color-accent);
}

.ai-checkbox {
  font-size: 0.68rem;
  color: var(--color-text-secondary);
  display: flex;
  align-items: center;
  gap: var(--space-1);
  cursor: pointer;
}

.ai-export {
  margin-left: auto;
  display: flex;
  gap: var(--space-1);
}

.ai-export-btn {
  padding: var(--space-1) var(--space-3);
  font-size: 0.65rem;
  background: var(--color-bg-primary);
  color: var(--color-text-secondary);
  border: 1px solid var(--color-border);
  border-radius: var(--radius-sm);
  cursor: pointer;
  font-weight: 500;
  transition: background-color var(--dur-fast) var(--ease),
              color var(--dur-fast) var(--ease),
              border-color var(--dur-fast) var(--ease);
}

.ai-export-btn:hover {
  background: var(--color-accent-light);
  color: var(--color-accent);
  border-color: var(--color-accent);
}

/* ---------- Summary row ---------- */
.ai-summary-row {
  padding: var(--space-1) var(--space-4);
  font-size: 0.68rem;
  color: var(--color-text-tertiary);
  background: var(--color-bg-tertiary);
  border-bottom: 1px solid var(--color-border-light);
  font-family: var(--font-mono);
}

/* ---------- Table ---------- */
.ai-table-wrapper {
  flex: 1;
  overflow: auto;
}

.ai-table {
  width: 100%;
  border-collapse: separate;
  border-spacing: 0;
  font-size: 0.75rem;
}

.ai-table th {
  position: sticky;
  top: 0;
  z-index: 1;
  padding: var(--space-2);
  text-align: left;
  background: var(--color-bg-tertiary);
  color: var(--color-text-secondary);
  font-size: 0.68rem;
  border-bottom: 1px solid var(--color-border);
  white-space: nowrap;
  font-weight: 600;
  letter-spacing: 0.02em;
}

.ai-table td {
  padding: var(--space-2);
  border-bottom: 1px solid var(--color-border-light);
  color: var(--color-text-primary);
  vertical-align: middle;
}

.ai-row {
  cursor: pointer;
  transition: background-color var(--dur-fast) var(--ease);
}

.ai-row:hover {
  background: var(--color-accent-light);
}

.ai-col-idx { width: 36px; text-align: center; color: var(--color-text-tertiary); font-family: var(--font-mono); }
.ai-col-cat { width: 80px; white-space: nowrap; }
.ai-col-label { max-width: 120px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; font-weight: 500; }
.ai-col-text { max-width: 200px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; color: var(--color-text-secondary); }
.ai-col-dynasty { width: 50px; color: var(--color-text-secondary); }
.ai-col-source { width: 50px; text-align: center; }
.ai-col-note { width: 40px; text-align: center; }

.ai-cat-dot {
  display: inline-block;
  width: 8px;
  height: 8px;
  border-radius: var(--radius-full);
  margin-right: var(--space-1);
  vertical-align: middle;
  box-shadow: 0 0 0 1px rgba(0, 0, 0, 0.05);
}

.ai-source-icon {
  font-size: 0.85rem;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  width: 20px;
  height: 20px;
  border-radius: var(--radius-sm);
}

.ai-source-user {
  color: var(--color-accent);
  background: var(--color-accent-light);
}

.ai-source-agent {
  color: var(--color-text-secondary);
  background: var(--color-bg-tertiary);
}

.ai-has-note {
  font-size: 0.8rem;
  color: var(--color-accent);
  font-weight: 600;
}

.ai-no-note { color: var(--color-text-disabled); }

/* ---------- Empty state ---------- */
.ai-empty {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  height: 200px;
  color: var(--color-text-secondary);
  font-size: 0.85rem;
  gap: var(--space-2);
}

.ai-clear-btn {
  padding: var(--space-1) var(--space-3);
  font-size: 0.72rem;
  background: var(--color-bg-primary);
  color: var(--color-text-primary);
  border: 1px solid var(--color-border);
  border-radius: var(--radius-sm);
  cursor: pointer;
  font-weight: 500;
  transition: background-color var(--dur-fast) var(--ease),
              border-color var(--dur-fast) var(--ease),
              color var(--dur-fast) var(--ease);
}

.ai-clear-btn:hover {
  background: var(--color-accent-light);
  color: var(--color-accent);
  border-color: var(--color-accent);
}

/* ---------- Context menu ---------- */
.ai-context-menu {
  position: fixed;
  z-index: 10000;
  background: var(--color-bg-primary);
  border: 1px solid var(--color-border);
  border-radius: var(--radius-sm);
  box-shadow: var(--shadow-md);
  min-width: 160px;
  padding: var(--space-1) 0;
}

.ai-cm-item {
  padding: var(--space-2) var(--space-4);
  font-size: 0.75rem;
  color: var(--color-text-primary);
  cursor: pointer;
  transition: background-color var(--dur-fast) var(--ease),
              color var(--dur-fast) var(--ease);
}

.ai-cm-item:hover {
  background: var(--color-accent-light);
  color: var(--color-accent);
}

.ai-cm-danger {
  color: var(--color-error);
}

.ai-cm-danger:hover {
  background: var(--color-error-light);
  color: var(--color-error);
}
</style>
