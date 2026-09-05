<script setup lang="ts">
import { ref, watch, computed } from 'vue'
import { useSourceTreeStore } from '@/stores/sourceTree'
import { useWorkspaceStore } from '@/stores/workspace'
import { useAgentStore } from '@/stores/agent'
import ContextMenu from './ContextMenu.vue'

const sourceTreeStore = useSourceTreeStore()
const workspaceStore = useWorkspaceStore()
const agentStore = useAgentStore()

const searchFocused = ref(false)
const searchInput = ref('')
const searchDebounceTimer = ref<ReturnType<typeof setTimeout> | null>(null)
const showFilters = ref(false)
const dynastyFilter = ref<string>('')
const showSearchResults = ref(false)

const dynasties = ['唐', '宋', '元', '明', '清']

// Context menu state
const contextMenuVisible = ref(false)
const contextMenuX = ref(0)
const contextMenuY = ref(0)
const contextTarget = ref<{
  type: 'chapter' | 'paragraph'
  chapterNumber: number
  paragraphIndex?: number
} | null>(null)

const isExpanded = (chapterNumber: number): boolean =>
  sourceTreeStore.expandedChapters.has(chapterNumber)

const currentChapterNumber = computed(() => {
  return sourceTreeStore.selectedNode?.chapterNumber || null
})

const currentParagraphIndex = computed(() => {
  return sourceTreeStore.selectedNode?.paragraphIndex ?? null
})

function toggleChapter(chapterNumber: number) {
  sourceTreeStore.toggleChapter(chapterNumber)
}

function onChapterClick(chapterNumber: number, event?: MouseEvent) {
  sourceTreeStore.selectChapter(chapterNumber)
  const ch = sourceTreeStore.chapters.find(c => c.number === chapterNumber)
  const title = ch ? `${ch.dynasty} · ${ch.title}` : `章节${chapterNumber}`
  if (event?.ctrlKey) {
    // Ctrl+click: force new tab (bypass dedup)
    const tab = workspaceStore.openTab('text-reader', `${title} (${Date.now().toString(36)})`, {
      chapterNumber,
      target: `chapter_${chapterNumber}_${Date.now()}`,
    })
  } else {
    workspaceStore.openTab('text-reader', title, {
      chapterNumber,
    })
  }
}

function onParagraphClick(chapterNumber: number, paragraphIndex: number) {
  sourceTreeStore.selectParagraph(chapterNumber, paragraphIndex)
  sourceTreeStore.setNavigationTarget({ chapterNumber, paragraphIndex })
  const ch = sourceTreeStore.chapters.find(c => c.number === chapterNumber)
  const title = ch ? `${ch.dynasty} · ${ch.title}` : `章节${chapterNumber}`
  workspaceStore.openTab('text-reader', title, {
    chapterNumber,
    paragraphIndex,
  })
}

function onChapterContextMenu(event: MouseEvent, chapterNumber: number) {
  event.preventDefault()
  contextTarget.value = { type: 'chapter', chapterNumber }
  contextMenuX.value = event.clientX
  contextMenuY.value = event.clientY
  contextMenuVisible.value = true
}

function onParagraphContextMenu(event: MouseEvent, chapterNumber: number, paragraphIndex: number) {
  event.preventDefault()
  contextTarget.value = { type: 'paragraph', chapterNumber, paragraphIndex }
  contextMenuX.value = event.clientX
  contextMenuY.value = event.clientY
  contextMenuVisible.value = true
}

function closeContextMenu() {
  contextMenuVisible.value = false
  contextTarget.value = null
}

function handleMenuAction(action: string, payload?: Record<string, unknown>) {
  if (!contextTarget.value) return

  if (action === 'dynasty_compare') {
    const targetChapterNumber = payload?.targetChapterNumber as number
    if (!targetChapterNumber) return
    const currentCh = sourceTreeStore.chapters.find(c => c.number === contextTarget.value!.chapterNumber)
    const targetCh = sourceTreeStore.chapters.find(c => c.number === targetChapterNumber)
    if (!currentCh || !targetCh) return
    workspaceStore.openTab('dynasty-compare', `${currentCh.dynasty} vs ${targetCh.dynasty}`, {
      config: {
        left: { chapterNumber: currentCh.number, chapterTitle: currentCh.title, dynasty: currentCh.dynasty },
        right: { chapterNumber: targetCh.number, chapterTitle: targetCh.title, dynasty: targetCh.dynasty },
        syncScroll: true,
        alignParagraphs: false,
      },
    })
  } else if (action === 'send_chapter') {
    const ch = sourceTreeStore.chapters.find(c => c.number === contextTarget.value!.chapterNumber)
    agentStore.addContextTag({
      id: `ctx_${Date.now()}`,
      label: `${ch?.dynasty || ''} - ${ch?.title || ''} 全文`,
      data: { chapterNumber: contextTarget.value.chapterNumber },
      sourceType: 'chapter',
    })
  } else if (action === 'open_new_tab') {
    const ch = sourceTreeStore.chapters.find(c => c.number === contextTarget.value!.chapterNumber)
    workspaceStore.openTab('text-reader', `${ch?.dynasty || ''} - ${ch?.title || ''} (${Date.now().toString(36)})`, {
      chapterNumber: contextTarget.value.chapterNumber,
      target: `chapter_${contextTarget.value.chapterNumber}_${Date.now()}`,
    })
  } else if (action === 'split_compare') {
    const activeTab = workspaceStore.activeTab
    if (activeTab) {
      const activeChNum = (activeTab.data as Record<string, unknown>)?.chapterNumber as number | undefined
      if (activeChNum !== undefined && activeChNum !== contextTarget.value.chapterNumber) {
        // Open the target chapter in a new tab, then enter split
        const ch = sourceTreeStore.chapters.find(c => c.number === contextTarget.value!.chapterNumber)
        const newTab = workspaceStore.openTab('text-reader', `${ch?.dynasty || ''} - ${ch?.title || ''}`, {
          chapterNumber: contextTarget.value.chapterNumber,
        })
        const leftTab = workspaceStore.tabs.find(t => {
          const d = t.data as Record<string, unknown>
          return d.chapterNumber === activeChNum
        })
        if (leftTab) {
          workspaceStore.enterSplitMode(leftTab.id, newTab.id)
        }
      }
    }
  } else if (action === 'send_paragraph') {
    const para = sourceTreeStore.getParagraph(
      contextTarget.value!.chapterNumber,
      contextTarget.value!.paragraphIndex || 0
    )
    const preview = para ? para.text.slice(0, 30) : ''
    agentStore.addContextTag({
      id: `ctx_${Date.now()}`,
      label: ` "${preview}..."`,
      data: {
        chapterNumber: contextTarget.value.chapterNumber,
        paragraphIndex: contextTarget.value.paragraphIndex,
      },
      sourceType: 'text_selection',
    })
  } else if (action === 'copy_paragraph') {
    const para = sourceTreeStore.getParagraph(
      contextTarget.value!.chapterNumber,
      contextTarget.value!.paragraphIndex || 0
    )
    if (para) {
      navigator.clipboard.writeText(para.text)
    }
  } else if (action === 'add_to_compare') {
    if (contextTarget.value.type === 'paragraph') {
      sourceTreeStore.setPendingCompare(contextTarget.value.chapterNumber)
    }
  } else if (action === 'self_annotate') {
    onParagraphClick(
      contextTarget.value!.chapterNumber,
      contextTarget.value!.paragraphIndex || 0
    )
  }

  closeContextMenu()
}

// Search with debounce
function onSearchInput() {
  if (searchDebounceTimer.value) clearTimeout(searchDebounceTimer.value)
  if (searchInput.value.trim().length < 2) {
    showSearchResults.value = false
    sourceTreeStore.clearSearch()
    return
  }
  searchDebounceTimer.value = setTimeout(() => {
    sourceTreeStore.search(searchInput.value)
    showSearchResults.value = true
  }, 300)
}

function clearSearch() {
  searchInput.value = ''
  showSearchResults.value = false
  sourceTreeStore.clearSearch()
}

function onSearchFocus() {
  if (searchInput.value.trim().length >= 2 && sourceTreeStore.searchResults.length > 0) {
    showSearchResults.value = true
  }
}

function onSearchResultClick(chapterNumber: number, paragraphIndex: number) {
  sourceTreeStore.selectParagraph(chapterNumber, paragraphIndex)
  sourceTreeStore.setNavigationTarget({ chapterNumber, paragraphIndex })
  const ch = sourceTreeStore.chapters.find(c => c.number === chapterNumber)
  const title = ch ? `${ch.dynasty} · ${ch.title}` : `章节${chapterNumber}`
  workspaceStore.openTab('text-reader', title, {
    chapterNumber,
    paragraphIndex,
  })
  showSearchResults.value = false
}

function highlightExcerptInSourceTree(r: typeof sourceTreeStore.searchResults[number]): string {
  if (!r.matchPositionsInExcerpt || r.matchPositionsInExcerpt.length === 0) return escapeHtml(r.excerpt)
  let result = ''
  let lastEnd = 0
  const excerpt = r.excerpt
  for (const [start, end] of r.matchPositionsInExcerpt) {
    if (start > lastEnd) result += escapeHtml(excerpt.slice(lastEnd, start))
    result += `<mark>${escapeHtml(excerpt.slice(start, end))}</mark>`
    lastEnd = end
  }
  if (lastEnd < excerpt.length) result += escapeHtml(excerpt.slice(lastEnd))
  return result
}

function escapeHtml(text: string): string {
  return text.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
}

// Filter search results by dynasty
const filteredResults = computed(() => {
  if (dynastyFilter.value === '') return sourceTreeStore.searchResults
  return sourceTreeStore.searchResults.filter(r => r.dynasty === dynastyFilter.value)
})
</script>

<template>
  <div class="source-tree">
    <!-- Search box -->
    <div class="st-search" :class="{ focused: searchFocused }">
      <input
        ref="searchInputRef"
        v-model="searchInput"
        type="text"
        class="st-search-input"
        placeholder="搜索文献..."
        @focus="searchFocused = true; onSearchFocus()"
        @blur="searchFocused = false"
        @input="onSearchInput"
        @keydown.escape="clearSearch"
      />
      <button v-if="showFilters" class="st-filter-toggle" @click="showFilters = !showFilters">
        {{ dynastyFilter }}
      </button>
      <button v-if="searchInput" class="st-search-clear" @click="clearSearch">&times;</button>
    </div>

    <!-- Filter dropdown -->
    <div v-if="showFilters" class="st-filters">
      <button
        v-for="d in dynasties"
        :key="d"
        class="st-filter-btn"
        :class="{ active: dynastyFilter === d }"
        @click="dynastyFilter = d"
      >{{ d }}</button>
    </div>

    <!-- Search results overlay -->
    <div v-if="showSearchResults" class="st-search-results">
      <div class="st-results-header">
        搜索结果 ({{ filteredResults.length }})
      </div>
      <div v-if="filteredResults.length === 0" class="st-no-results">
        无匹配结果
      </div>
      <div
        v-for="r in filteredResults"
        :key="`${r.chapterNumber}:${r.paragraphIndex}`"
        class="st-result-item"
        @click="onSearchResultClick(r.chapterNumber, r.paragraphIndex)"
      >
        <div class="st-result-meta">{{ r.dynasty }} - {{ r.chapterTitle }} ¶{{ r.paragraphIndex }}</div>
        <div class="st-result-excerpt" v-html="highlightExcerptInSourceTree(r)" />
      </div>
    </div>

    <!-- Loading state -->
    <div v-if="sourceTreeStore.loading" class="st-skeleton">
      <div v-for="i in 3" :key="i" class="st-skeleton-row" />
    </div>

    <!-- Error state -->
    <div v-else-if="sourceTreeStore.error" class="st-error">
      加载失败
      <button class="st-retry" @click="sourceTreeStore.fetchChapters()">重试</button>
    </div>

    <!-- Empty state -->
    <div v-else-if="sourceTreeStore.chapters.length === 0" class="st-empty">
      暂无文献数据
    </div>

    <!-- Chapter tree -->
    <div v-else class="st-tree">
      <div
        v-for="ch in sourceTreeStore.chapters"
        :key="ch.number"
        class="st-chapter-group"
      >
        <!-- Chapter header -->
        <div
          class="st-chapter-header"
          :class="{
            active: currentChapterNumber === ch.number && currentParagraphIndex === null,
          }"
          @click="onChapterClick(ch.number, $event)"
          @contextmenu="onChapterContextMenu($event, ch.number)"
        >
          <button
            class="st-expand-toggle"
            @click.stop="toggleChapter(ch.number)"
          >
            <span :class="{ rotated: !isExpanded(ch.number) }">&#9660;</span>
          </button>
          <span class="st-chapter-label">{{ ch.dynasty }} - {{ ch.title }}</span>
          <span class="st-chapter-badge">{{ ch.paragraphs.length }}</span>
        </div>

        <!-- Paragraph list with expand animation -->
        <Transition name="expand">
          <div v-if="isExpanded(ch.number)" class="st-paragraph-list">
            <div
              v-for="para in ch.paragraphs"
              :key="para.index"
              class="st-paragraph-item"
              :class="{
                active: currentChapterNumber === ch.number && currentParagraphIndex === para.index,
              }"
              @click="onParagraphClick(ch.number, para.index)"
              @contextmenu="onParagraphContextMenu($event, ch.number, para.index)"
            >
              <span class="st-para-num">¶{{ para.index }}</span>
              <span class="st-para-preview">{{ para.text.slice(0, 30) }}{{ para.text.length > 30 ? '...' : '' }}</span>
            </div>
          </div>
        </Transition>
      </div>
    </div>

    <!-- Context menu -->
    <ContextMenu
      :visible="contextMenuVisible"
      :x="contextMenuX"
      :y="contextMenuY"
      :target-type="contextTarget?.type || null"
      :chapters="sourceTreeStore.chapters.map(c => ({ number: c.number, dynasty: c.dynasty, title: c.title }))"
      :current-chapter-number="contextTarget?.chapterNumber"
      @action="handleMenuAction"
      @close="closeContextMenu"
    />
  </div>
</template>

<style scoped>
.source-tree {
  display: flex;
  flex-direction: column;
  height: 100%;
  overflow: hidden;
}

/* Search */
.st-search {
  display: flex;
  align-items: center;
  margin: 6px 10px;
  padding: 4px 8px;
  border: 1px solid var(--color-border);
  border-radius: var(--radius);
  background: var(--color-bg-primary);
  transition: border-color var(--dur-fast) var(--ease), box-shadow var(--dur-fast) var(--ease);
}

.st-search.focused {
  border-color: var(--color-accent);
  box-shadow: var(--shadow-focus);
}

.st-search-input {
  flex: 1;
  border: none;
  outline: none;
  background: transparent;
  color: var(--color-text-primary);
  font-family: var(--font-sans);
  font-size: 0.78rem;
  padding: 3px 0;
}

.st-search-input::placeholder {
  color: var(--color-text-placeholder);
}

.st-filter-toggle {
  background: none;
  border: none;
  color: var(--color-text-secondary);
  cursor: pointer;
  font-size: 0.7rem;
  padding: 2px 6px;
  border-radius: var(--radius-sm);
  transition: background-color var(--dur-fast) var(--ease), color var(--dur-fast) var(--ease);
}

.st-filter-toggle:hover {
  background: var(--color-bg-tertiary);
  color: var(--color-accent);
}

.st-search-clear {
  background: none;
  border: none;
  color: var(--color-text-tertiary);
  cursor: pointer;
  font-size: 1rem;
  line-height: 1;
  padding: 0 2px;
  border-radius: var(--radius-sm);
  transition: background-color var(--dur-fast) var(--ease), color var(--dur-fast) var(--ease);
}

.st-search-clear:hover {
  background: var(--color-bg-tertiary);
  color: var(--color-text-primary);
}

/* Filters */
.st-filters {
  display: flex;
  gap: 4px;
  padding: 4px 10px;
  flex-wrap: wrap;
}

.st-filter-btn {
  background: var(--color-bg-tertiary);
  border: 1px solid var(--color-border);
  color: var(--color-text-secondary);
  padding: 2px 8px;
  border-radius: var(--radius-sm);
  font-size: 0.7rem;
  cursor: pointer;
  transition: background-color var(--dur-fast) var(--ease), border-color var(--dur-fast) var(--ease), color var(--dur-fast) var(--ease);
}

.st-filter-btn:hover {
  background: var(--color-bg-hover);
  border-color: var(--color-border-dark);
  color: var(--color-text-primary);
}

.st-filter-btn.active {
  background: var(--color-accent);
  color: #fff;
  border-color: var(--color-accent);
}

/* Search results */
.st-search-results {
  max-height: 200px;
  overflow-y: auto;
  border-bottom: 1px solid var(--color-border-light);
}

.st-results-header {
  padding: 4px 10px;
  font-size: 0.7rem;
  color: var(--color-text-secondary);
  background: var(--color-bg-secondary);
}

.st-no-results {
  padding: 10px;
  font-size: 0.78rem;
  color: var(--color-text-tertiary);
  text-align: center;
}

.st-result-item {
  padding: 6px 10px;
  cursor: pointer;
  border-bottom: 1px solid var(--color-border-light);
  transition: background-color var(--dur-fast) var(--ease);
}

.st-result-item:hover {
  background: var(--color-accent-light);
}

.st-result-meta {
  font-size: 0.68rem;
  color: var(--color-text-tertiary);
  font-family: var(--font-mono);
}

.st-result-excerpt {
  font-size: 0.75rem;
  color: var(--color-text-primary);
  margin-top: 2px;
  line-height: 1.4;
}

.st-result-excerpt :deep(mark) {
  background: var(--color-accent-light);
  color: var(--color-accent);
  border-radius: var(--radius-sm);
  padding: 0 2px;
  font-weight: 500;
}

/* Skeleton */
.st-skeleton {
  padding: 12px 10px;
}

.st-skeleton-row {
  height: 16px;
  background: var(--color-bg-tertiary);
  border-radius: var(--radius-sm);
  margin-bottom: 8px;
  animation: shimmer 1.5s var(--ease) infinite;
}

@keyframes shimmer {
  0%, 100% { opacity: 0.6; }
  50% { opacity: 1; }
}

.st-skeleton-row:nth-child(2) { width: 80%; }
.st-skeleton-row:nth-child(3) { width: 60%; }

.st-error, .st-empty {
  padding: 16px 10px;
  font-size: 0.8rem;
  color: var(--color-text-tertiary);
  text-align: center;
}

.st-error {
  color: var(--color-verdict-refuted);
}

.st-retry {
  margin-top: 6px;
  padding: 3px 10px;
  font-size: 0.72rem;
  background: var(--color-bg-tertiary);
  color: var(--color-text-primary);
  border: 1px solid var(--color-border);
  border-radius: var(--radius-sm);
  cursor: pointer;
  transition: background-color var(--dur-fast) var(--ease), border-color var(--dur-fast) var(--ease), color var(--dur-fast) var(--ease);
}

.st-retry:hover {
  background: var(--color-accent-light);
  border-color: var(--color-accent);
  color: var(--color-accent);
}

/* Tree */
.st-tree {
  flex: 1;
  overflow-y: auto;
}

.st-chapter-group {
  border-bottom: 1px solid var(--color-border-light);
}

.st-chapter-header {
  display: flex;
  align-items: center;
  padding: 7px 10px;
  cursor: pointer;
  gap: 6px;
  transition: background-color var(--dur-fast) var(--ease);
}

.st-chapter-header:hover {
  background: var(--color-bg-hover);
}

.st-chapter-header.active {
  background: var(--color-accent-light);
  border-left: 3px solid var(--color-accent);
}

.st-expand-toggle {
  background: none;
  border: none;
  color: var(--color-text-tertiary);
  cursor: pointer;
  font-size: 0.55rem;
  padding: 0;
  width: 14px;
  text-align: center;
  transition: transform var(--dur) var(--ease), color var(--dur-fast) var(--ease);
}

.st-expand-toggle:hover {
  color: var(--color-accent);
}

.st-expand-toggle .rotated {
  display: inline-block;
  transform: rotate(-90deg);
}

.st-chapter-label {
  flex: 1;
  font-family: var(--font-sans);
  font-size: 0.8rem;
  color: var(--color-text-primary);
  min-width: 0;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.st-chapter-badge {
  font-size: 0.62rem;
  color: var(--color-text-tertiary);
  background: var(--color-bg-secondary);
  padding: 1px 6px;
  border-radius: var(--radius-full);
  border: 1px solid var(--color-border-light);
}

/* Paragraph list */
.st-paragraph-list {
  overflow: hidden;
}

.st-paragraph-item {
  display: flex;
  align-items: flex-start;
  gap: 6px;
  padding: 5px 10px 5px 32px;
  cursor: pointer;
  transition: background-color var(--dur-fast) var(--ease);
  border-left: 3px solid transparent;
}

.st-paragraph-item:hover {
  background: var(--color-bg-hover);
}

.st-paragraph-item.active {
  background: var(--color-accent-light);
  border-left-color: var(--color-accent);
}

.st-para-num {
  font-family: var(--font-mono);
  font-size: 0.65rem;
  color: var(--color-text-tertiary);
  flex-shrink: 0;
  margin-top: 1px;
}

.st-para-preview {
  font-size: 0.73rem;
  color: var(--color-text-secondary);
  line-height: 1.4;
}

/* Expand animation — 与令牌一致 */
.expand-enter-active {
  transition: max-height var(--dur) var(--ease), opacity var(--dur-fast) var(--ease);
  max-height: 500px;
}

.expand-leave-active {
  transition: max-height var(--dur) var(--ease), opacity var(--dur-fast) var(--ease);
  max-height: 0;
}

.expand-enter-from {
  max-height: 0;
  opacity: 0;
}

.expand-leave-to {
  max-height: 0;
  opacity: 0;
}
</style>
