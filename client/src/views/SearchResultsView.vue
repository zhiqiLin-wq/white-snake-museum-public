<script setup lang="ts">
import { ref, computed } from 'vue'
import { useWorkspaceStore } from '@/stores/workspace'
import { useSourceTreeStore } from '@/stores/sourceTree'
import type { SearchResultsEvent } from '@/types/sse-events'
import type { SearchResultItem } from '@/types/sse-events'
import type { SearchHighlight } from '@/types/literature'
import { computeJumpHighlights } from '@/utils/highlight'

const workspaceStore = useWorkspaceStore()
const sourceTreeStore = useSourceTreeStore()

const tabData = computed(() => {
  const tab = workspaceStore.activeTab
  return tab?.data as unknown as SearchResultsEvent | undefined
})

const query = computed(() => tabData.value?.query || '')
const totalMatches = computed(() => tabData.value?.totalMatches || 0)
const results = computed(() => tabData.value?.results || [])
const sortOptions = computed(() => tabData.value?.sortOptions || ['按相关度', '按朝代', '按章节顺序'])

const currentSort = ref('')
const currentPage = ref(1)
const pageSize = 20

const sortedResults = computed(() => {
  const sorted = [...results.value]
  if (currentSort.value === '按朝代') {
    sorted.sort((a, b) => a.chapterNumber.localeCompare(b.chapterNumber))
  } else if (currentSort.value === '按章节顺序') {
    sorted.sort((a, b) => Number(a.chapterNumber) - Number(b.chapterNumber))
  }
  return sorted
})

const pagedResults = computed(() => {
  const start = (currentPage.value - 1) * pageSize
  return sortedResults.value.slice(start, start + pageSize)
})

const totalPages = computed(() => Math.ceil(sortedResults.value.length / pageSize))

function navigateToSource(result: SearchResultItem) {
  const CHINESE_NUM: Record<string, number> = {
    '一': 1, '二': 2, '三': 3, '四': 4, '五': 5, '六': 6, '七': 7, '八': 8,
  }
  const chNum = typeof result.chapterNumber === 'number'
    ? result.chapterNumber
    : (Number(result.chapterNumber) || CHINESE_NUM[result.chapterNumber] || NaN)

  // 把目标章节/段落提到外层变量，供 computeJumpHighlights 使用
  const targetCh = sourceTreeStore.chapters.find(c => c.number === chNum)
  const targetPara = targetCh?.paragraphs.find(p => p.index === result.paragraphIndex)

  // B-162: 使用共享高亮计算函数（三层兜底：精确位置 → query匹配 → excerpt剥省略号 → 实体扫描）
  const ranges = computeJumpHighlights(query.value, result, targetPara)

  const highlights: SearchHighlight[] = ranges.map(({ start, end }) => ({
    query: query.value,
    chapterNumber: chNum,
    paragraphIndex: result.paragraphIndex,
    startChar: start,
    endChar: end,
    text: targetPara?.text?.slice(start, end) || query.value,
  }))

  sourceTreeStore.activeSearch = {
    query: query.value,
    highlights,
    source: 'agent',
    // 暂时性高亮：离开阅读页自动清除，重进同章节不再显示
    ephemeral: true,
    // B-162: chapters 可能尚未加载导致 highlights 为空；
    // TextReaderView 在 paragraphs 加载后用此字段重算
    pendingRecompute: highlights.length === 0 ? {
      excerpt: result.excerpt,
      paragraphMatchPositions: result.paragraphMatchPositions,
      paragraphIndex: result.paragraphIndex,
      chapterNumber: chNum,
    } : undefined,
  }

  // ===== 高亮链路证据：开发控制台可直接核对命中区间（可搜索 [跳转高亮] 关键字）=====
  // 生产模式 console.debug 自动不显示，仅 dev 模式可见。
  // 若点击跳转后控制台没打印这段，说明 navigateToSource 根本没被触发（按钮绑定/事件冒泡问题）。
  // 若打印了 highlights 长度为 0，说明所有兜底都没命中（查 chapterNumber 是否能在 sourceTreeStore.chapters 中找到）。
  // 若打印了有效区间但肉眼没看到蓝色高亮，去 TextReaderView.renderParagraphText 或 CSS 层排查。
  if (import.meta.env.DEV) {
    const evidence = highlights.map((h, i) => `  [${i}] para#${h.paragraphIndex} chars ${h.startChar}-${h.endChar} -> "${targetPara?.text?.slice(h.startChar, h.endChar) ?? ''}"`)
    console.debug(
      `[跳转高亮] ${query.value}\n` +
      `  chapter=${chNum}, 命中${highlights.length}处:\n` +
      evidence.join('\n') +
      (highlights.length === 0 ? '\n  !! 未命中任何区间：请检查 result.paragraphMatchPositions 与 query 长度' : '')
    )
  }

  // B-163: 复用演化分析综合论述的 C06 机制 —— 携带 citationHighlight 让
  // TextReaderView 用归一化匹配（NFC + 去标点 + 前缀兜底）在渲染段落中定位引文。
  // 为什么需要：出场轨迹结果的 paragraphMatchPositions 为空、paragraphIndex 是
  // chunk 级索引可能错位，excerpt 直接匹配原文可能因 chunk 清洗字符差异失败
  // （B-152n 结论），而 citationHighlight 的短前缀 + 归一化匹配实测成功率最高。
  // 策略与 EvolutionWorkspaceView.jumpToSourceFromModal 的 B-156 一致：
  // excerpt 剥省略号后取前 25 字，>=4 字才传。
  const excerptCore = (result.excerpt || '').replace(/^[…⋯]+|[…⋯]+$/g, '').trim()
  const citationText = excerptCore.slice(0, 25)

  const srCh = sourceTreeStore.chapters.find(c => c.number === chNum)
  const srTitle = srCh ? `${srCh.dynasty} · ${srCh.title}` : (result.chapterTitle || `章节${result.chapterNumber}`)
  workspaceStore.openTab('text-reader', srTitle, {
    chapterNumber: chNum,
    paragraphIndex: result.paragraphIndex,
    // 一次性闪烁标记：TextReaderView 消费后即删除，段落底色暂时高亮
    flashParagraph: Date.now(),
    // C06: 归一化引文高亮（金色 mark），且 mark 兜底滚动修正段落号错位
    citationHighlight: citationText.length >= 4 ? { text: citationText } : undefined,
  })
}

function highlightExcerpt(excerpt: string, ranges: [number, number][]): string {
  if (!ranges || ranges.length === 0) return excerpt
  let result = ''
  let lastEnd = 0
  for (const [start, end] of ranges) {
    if (start > lastEnd) result += excerpt.slice(lastEnd, start)
    result += `<mark>${excerpt.slice(start, end)}</mark>`
    lastEnd = end
  }
  if (lastEnd < excerpt.length) result += excerpt.slice(lastEnd)
  return result
}
</script>

<template>
  <div class="search-results-view">
    <div class="sr-header">
      <h2 class="sr-title">
        搜索结果: "{{ query }}"
        <span class="sr-count">{{ totalMatches }} 条匹配</span>
      </h2>

      <div v-if="sortOptions.length > 1" class="sr-sort">
        <label class="sr-sort-label">排序:</label>
        <select v-model="currentSort" class="sr-sort-select">
          <option v-for="opt in sortOptions" :key="opt" :value="opt">{{ opt }}</option>
        </select>
      </div>
    </div>

    <div v-if="pagedResults.length === 0" class="sr-empty">
      没有找到匹配结果
    </div>

    <div v-else class="sr-list">
      <div
        v-for="result in pagedResults"
        :key="`${result.chapterNumber}:${result.paragraphIndex}`"
        class="sr-item"
      >
        <div class="sr-item-header">
          <span class="sr-rank">#{{ result.rank }}</span>
          <span class="sr-source">{{ result.chapterTitle }} ¶{{ result.paragraphIndex }}</span>
          <span class="sr-score">{{ Math.round(result.relevanceScore * 100) }}%</span>
        </div>
        <div
          class="sr-excerpt"
          v-html="highlightExcerpt(result.excerpt, result.highlightRanges)"
        />
        <button
          v-if="result.chapterNumber"
          class="sr-navigate-btn"
          @click="navigateToSource(result)"
        >
          跳转到原文
        </button>
      </div>
    </div>

    <!-- Pagination -->
    <div v-if="totalPages > 1" class="sr-pagination">
      <button :disabled="currentPage <= 1" @click="currentPage--">上一页</button>
      <span>{{ currentPage }} / {{ totalPages }}</span>
      <button :disabled="currentPage >= totalPages" @click="currentPage++">下一页</button>
    </div>
  </div>
</template>

<style scoped>
.search-results-view {
  padding: var(--space-5);
  height: 100%;
  overflow-y: auto;
}

.sr-header {
  margin-bottom: var(--space-4);
}

.sr-title {
  font-family: var(--font-sans);
  font-size: 1rem;
  font-weight: 600;
  color: var(--color-text-primary);
  margin: 0;
}

.sr-count {
  font-size: 0.75rem;
  color: var(--color-text-tertiary);
  margin-left: var(--space-2);
  font-weight: 400;
}

.sr-sort {
  display: flex;
  align-items: center;
  gap: var(--space-2);
  margin-top: var(--space-2);
}

.sr-sort-label {
  font-size: 0.75rem;
  color: var(--color-text-secondary);
}

.sr-sort-select {
  padding: var(--space-1) var(--space-2);
  background: var(--color-bg-primary);
  border: 1px solid var(--color-border);
  color: var(--color-text-primary);
  border-radius: var(--radius-sm);
  font-size: 0.75rem;
  font-family: var(--font-sans);
  transition:
    border-color var(--dur-fast) var(--ease),
    box-shadow var(--dur-fast) var(--ease);
}

.sr-sort-select:hover {
  border-color: var(--color-border-dark);
}

.sr-sort-select:focus {
  outline: none;
  border-color: var(--color-accent);
  box-shadow: var(--shadow-focus);
}

.sr-empty {
  text-align: center;
  padding: var(--space-8);
  color: var(--color-text-tertiary);
  font-size: 0.85rem;
}

.sr-list {
  display: flex;
  flex-direction: column;
  gap: var(--space-3);
}

.sr-item {
  padding: var(--space-4);
  background: var(--color-bg-primary);
  border: 1px solid var(--color-border);
  border-radius: var(--radius-lg);
  transition:
    border-color var(--dur) var(--ease),
    box-shadow var(--dur) var(--ease),
    transform var(--dur) var(--ease);
}

.sr-item:hover {
  border-color: var(--color-border-dark);
  box-shadow: var(--shadow-md);
  transform: translateY(-1px);
}

.sr-item-header {
  display: flex;
  align-items: center;
  gap: var(--space-2);
  margin-bottom: var(--space-2);
}

.sr-rank {
  font-family: var(--font-mono);
  font-size: 0.7rem;
  color: var(--color-text-tertiary);
}

.sr-source {
  flex: 1;
  font-size: 0.78rem;
  color: var(--color-text-primary);
  font-weight: 500;
}

.sr-score {
  font-size: 0.7rem;
  color: var(--color-text-tertiary);
  font-family: var(--font-mono);
}

.sr-excerpt {
  font-size: 0.8rem;
  color: var(--color-text-primary);
  line-height: 1.6;
  margin-bottom: var(--space-2);
}

.sr-excerpt :deep(mark) {
  background: var(--color-accent-light);
  color: var(--color-text-primary);
  font-weight: 500;
  padding: 0 1px;
  border-radius: var(--radius-sm);
}

.sr-navigate-btn {
  padding: var(--space-1) var(--space-3);
  font-size: 0.72rem;
  background: var(--color-bg-primary);
  color: var(--color-accent);
  border: 1px solid var(--color-accent);
  border-radius: var(--radius-sm);
  cursor: pointer;
  font-family: var(--font-sans);
  font-weight: 500;
  transition:
    background-color var(--dur-fast) var(--ease),
    border-color var(--dur-fast) var(--ease),
    color var(--dur-fast) var(--ease),
    transform var(--dur) var(--ease),
    box-shadow var(--dur) var(--ease);
}

.sr-navigate-btn:hover {
  background: var(--color-accent);
  color: #fff;
  transform: translateY(-1px);
  box-shadow: 0 4px 10px rgba(91, 155, 213, 0.18);
}

.sr-navigate-btn:active {
  transform: translateY(0);
  box-shadow: none;
}

.sr-pagination {
  display: flex;
  align-items: center;
  justify-content: center;
  gap: var(--space-3);
  margin-top: var(--space-4);
  font-size: 0.78rem;
  color: var(--color-text-secondary);
}

.sr-pagination button {
  padding: var(--space-1) var(--space-3);
  background: var(--color-bg-primary);
  border: 1px solid var(--color-border);
  color: var(--color-text-primary);
  border-radius: var(--radius-sm);
  cursor: pointer;
  font-family: var(--font-sans);
  font-size: 0.75rem;
  transition:
    background-color var(--dur-fast) var(--ease),
    border-color var(--dur-fast) var(--ease),
    color var(--dur-fast) var(--ease);
}

.sr-pagination button:hover:not(:disabled) {
  background: var(--color-bg-hover);
  border-color: var(--color-border-dark);
}

.sr-pagination button:disabled {
  background: var(--color-bg-tertiary);
  color: var(--color-text-disabled);
  border-color: var(--color-border);
  cursor: not-allowed;
}
</style>
