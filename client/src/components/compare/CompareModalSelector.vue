<script setup lang="ts">
import { ref, computed } from 'vue'
import { useSourceTreeStore } from '@/stores/sourceTree'
import { useWorkspaceStore } from '@/stores/workspace'
import { useDynastyCompareStore } from '@/stores/dynastyCompare'
import { useAgentStore } from '@/stores/agent'

const props = defineProps<{
  visible: boolean
}>()

const emit = defineEmits<{
  close: []
}>()

const sourceTreeStore = useSourceTreeStore()
const workspaceStore = useWorkspaceStore()
const dynastyCompareStore = useDynastyCompareStore()
const agentStore = useAgentStore()

const searchQuery = ref('')
const searchDebounceTimer = ref<ReturnType<typeof setTimeout> | null>(null)
const selectedLeftChapter = ref<number | null>(null)
const selectedRightChapter = ref<number | null>(null)

// Auto-select left chapter from active tab
const activeTab = computed(() => workspaceStore.activeTab)
const activeChapterNumber = computed(() => {
  const data = activeTab.value?.data as Record<string, unknown> | undefined
  if (activeTab.value?.type === 'text-reader') {
    return data?.chapterNumber as number | undefined
  }
  return undefined
})

// Initialize left side from active tab
if (activeChapterNumber.value && !selectedLeftChapter.value) {
  selectedLeftChapter.value = activeChapterNumber.value
}

const filteredChapters = computed(() => {
  let chs = sourceTreeStore.chapters
  if (searchQuery.value.trim()) {
    const q = searchQuery.value.trim().toLowerCase()
    chs = chs.filter(c =>
      c.title.toLowerCase().includes(q) ||
      c.dynasty.includes(q)
    )
  }
  // Exclude left selected chapter from right
  if (selectedLeftChapter.value) {
    chs = chs.filter(c => c.number !== selectedLeftChapter.value)
  }
  return chs
})

const leftChapter = computed(() => {
  if (!selectedLeftChapter.value) return null
  return sourceTreeStore.chapters.find(c => c.number === selectedLeftChapter.value) || null
})

const canStartCompare = computed(() =>
  selectedLeftChapter.value !== null && selectedRightChapter.value !== null
)

function onSearchInput() {
  if (searchDebounceTimer.value) clearTimeout(searchDebounceTimer.value)
  searchDebounceTimer.value = setTimeout(() => {}, 300)
}

function startCompare() {
  if (!canStartCompare.value) return
  const leftCh = leftChapter.value
  const rightCh = sourceTreeStore.chapters.find(c => c.number === selectedRightChapter.value)
  if (!leftCh || !rightCh) return

  const title = `${leftCh.dynasty} vs ${rightCh.dynasty}`
  workspaceStore.openTab('dynasty-compare', title, {
    config: {
      left: {
        chapterNumber: leftCh.number,
        chapterTitle: leftCh.title,
        dynasty: leftCh.dynasty,
      },
      right: {
        chapterNumber: rightCh.number,
        chapterTitle: rightCh.title,
        dynasty: rightCh.dynasty,
      },
      syncScroll: true,
      alignParagraphs: false,
    },
  })
  emit('close')

  // P1-04: Auto-trigger Agent analysis after opening compare tab
  dynastyCompareStore.setLoading(true)
  const prompt = `请对比分析 ${leftCh.dynasty}(${leftCh.title}) 和 ${rightCh.dynasty}(${rightCh.title}) 的差异`
  agentStore.sendMessage(prompt)
}
</script>

<template>
  <Teleport to="body">
    <div v-if="visible" class="compare-modal-backdrop" @click="emit('close')" />
    <div v-if="visible" class="compare-modal">
      <div class="cm-header">选择章节对比</div>

      <div class="cm-body">
        <div class="cm-side">
          <div class="cm-side-label">当前章节 (左栏)</div>
          <div v-if="leftChapter" class="cm-selected">
            {{ leftChapter.dynasty }} - {{ leftChapter.title }}
          </div>
          <div v-else class="cm-empty-side">
            请先在左侧文献目录中打开一个章节
          </div>
        </div>

        <div class="cm-divider">vs</div>

        <div class="cm-side">
          <div class="cm-side-label">选择对比章节 (右栏)</div>
          <div class="cm-search">
            <input
              v-model="searchQuery"
              type="text"
              class="cm-search-input"
              placeholder="搜索章节..."
              @input="onSearchInput"
            />
          </div>
          <div class="cm-chapter-list">
            <div
              v-for="ch in filteredChapters"
              :key="ch.number"
              class="cm-chapter-item"
              :class="{ selected: selectedRightChapter === ch.number }"
              @click="selectedRightChapter = ch.number"
            >
              <span class="cm-ch-dynasty">{{ ch.dynasty }}</span>
              <span class="cm-ch-title">{{ ch.title }}</span>
              <span class="cm-ch-count">{{ ch.paragraphs.length }}段</span>
            </div>

            <div v-if="filteredChapters.length === 0 && sourceTreeStore.chapters.length > 0" class="cm-no-results">
              无匹配章节
            </div>
          </div>
        </div>
      </div>

      <div class="cm-footer">
        <button class="cm-btn-cancel" @click="emit('close')">取消</button>
        <button
          class="cm-btn-start"
          :disabled="!canStartCompare"
          @click="startCompare"
        >开始对比</button>
      </div>
    </div>
  </Teleport>
</template>

<style scoped>
.compare-modal-backdrop {
  position: fixed;
  inset: 0;
  z-index: 9000;
  background: rgba(26, 26, 26, 0.35);
}

.compare-modal {
  position: fixed;
  top: 50%;
  left: 50%;
  transform: translate(-50%, -50%);
  z-index: 9001;
  background: var(--color-bg-secondary, #F7F8FA);
  border: 1px solid var(--color-border, #E5E7EB);
  border-radius: var(--radius, 8px);
  width: 600px;
  max-height: 80vh;
  display: flex;
  flex-direction: column;
  box-shadow: var(--shadow-md, 0 8px 24px rgba(0, 0, 0, 0.12));
}

.cm-header {
  font-size: 0.9rem;
  color: var(--color-text-primary, #1A1A1A);
  padding: 12px 16px;
  border-bottom: 1px solid var(--color-border, #E5E7EB);
  font-weight: 500;
}

.cm-body {
  display: flex;
  flex: 1;
  overflow: hidden;
  padding: 12px;
  gap: 12px;
}

.cm-side {
  flex: 1;
  min-width: 0;
  display: flex;
  flex-direction: column;
}

.cm-divider {
  display: flex;
  align-items: center;
  font-size: 0.8rem;
  color: var(--color-text-secondary, #4B5563);
  flex-shrink: 0;
}

.cm-side-label {
  font-size: 0.72rem;
  color: var(--color-text-secondary, #4B5563);
  margin-bottom: 8px;
}

.cm-selected {
  padding: 8px;
  background: var(--color-bg-tertiary, #F0F2F5);
  border: 1px solid var(--color-border, #E5E7EB);
  border-radius: var(--radius-sm, 4px);
  font-size: 0.78rem;
  color: var(--color-text-primary, #1A1A1A);
}

.cm-empty-side {
  padding: 8px;
  background: var(--color-bg-tertiary, #F0F2F5);
  border-radius: var(--radius-sm, 4px);
  font-size: 0.75rem;
  color: var(--color-text-secondary, #4B5563);
  text-align: center;
}

.cm-search {
  margin-bottom: 8px;
}

.cm-search-input {
  width: 100%;
  padding: 4px 8px;
  border: 1px solid var(--color-border, #E5E7EB);
  border-radius: var(--radius-sm, 4px);
  background: var(--color-bg-primary, #FFFFFF);
  color: var(--color-text-primary, #1A1A1A);
  font-size: 0.75rem;
  box-sizing: border-box;
  transition: border-color var(--dur-fast, 120ms) var(--ease, cubic-bezier(.22,.61,.36,1));
}

.cm-search-input:focus {
  outline: none;
  border-color: var(--color-accent, #5B9BD5);
  box-shadow: 0 0 0 3px rgba(91, 155, 213, .15);
}

.cm-chapter-list {
  flex: 1;
  overflow-y: auto;
  display: flex;
  flex-direction: column;
  gap: 2px;
}

.cm-chapter-item {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 6px 8px;
  cursor: pointer;
  border-radius: var(--radius-sm, 4px);
  border: 1px solid transparent;
  transition: background var(--dur-fast, 120ms) var(--ease, cubic-bezier(.22,.61,.36,1)),
              border-color var(--dur-fast, 120ms) var(--ease, cubic-bezier(.22,.61,.36,1));
}

.cm-chapter-item:hover {
  background: var(--color-bg-hover, #F2F4F7);
}

.cm-chapter-item.selected {
  background: var(--color-accent-light, #EBF4FC);
  border-color: var(--color-accent, #5B9BD5);
}

.cm-ch-dynasty {
  font-size: 0.7rem;
  color: var(--color-text-secondary, #4B5563);
  flex-shrink: 0;
}

.cm-ch-title {
  font-size: 0.78rem;
  color: var(--color-text-primary, #1A1A1A);
  flex: 1;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.cm-ch-count {
  font-size: 0.65rem;
  color: var(--color-text-tertiary, #6B7280);
  flex-shrink: 0;
}

.cm-no-results {
  padding: 16px;
  font-size: 0.75rem;
  color: var(--color-text-secondary, #4B5563);
  text-align: center;
}

.cm-footer {
  padding: 12px 16px;
  border-top: 1px solid var(--color-border, #E5E7EB);
  display: flex;
  justify-content: flex-end;
  gap: 8px;
}

.cm-btn-cancel {
  padding: 5px 14px;
  font-size: 0.75rem;
  background: var(--color-bg-tertiary, #F0F2F5);
  color: var(--color-text-primary, #1A1A1A);
  border: 1px solid var(--color-border, #E5E7EB);
  border-radius: var(--radius-sm, 4px);
  cursor: pointer;
  transition: background var(--dur-fast, 120ms) var(--ease, cubic-bezier(.22,.61,.36,1)),
              border-color var(--dur-fast, 120ms) var(--ease, cubic-bezier(.22,.61,.36,1));
}

.cm-btn-cancel:hover {
  background: var(--color-bg-hover, #F2F4F7);
  border-color: var(--color-border-dark, #D1D5DB);
}

.cm-btn-start {
  padding: 5px 14px;
  font-size: 0.75rem;
  background: var(--color-accent, #5B9BD5);
  color: #FFFFFF;
  border: 1px solid var(--color-accent, #5B9BD5);
  border-radius: var(--radius-sm, 4px);
  cursor: pointer;
  transition: background var(--dur-fast, 120ms) var(--ease, cubic-bezier(.22,.61,.36,1)),
              border-color var(--dur-fast, 120ms) var(--ease, cubic-bezier(.22,.61,.36,1));
}

.cm-btn-start:hover:not(:disabled) {
  background: var(--color-accent-hover, #4A8BC5);
  border-color: var(--color-accent-hover, #4A8BC5);
}

.cm-btn-start:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}
</style>
