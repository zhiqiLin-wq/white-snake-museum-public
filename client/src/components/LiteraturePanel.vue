<script setup lang="ts">
import { ref, computed, watch, nextTick } from 'vue'
import type { Chapter } from '../types'
import { escapeHtml } from '../utils/helpers'

const props = defineProps<{
  chapters: Chapter[]
  locationNames: string[]
  searchQuery: string
  queryWords: string[]
  loading: boolean
}>()

const emit = defineEmits<{
  search: [query: string]
  'clear-search': []
}>()

const localQuery = ref('')
const openChapters = ref<Set<number>>(new Set())

watch(() => props.searchQuery, val => { localQuery.value = val })

function highlightText(text: string): string {
  const words = props.queryWords
  if (!words.length) return escapeHtml(text)
  let escaped = escapeHtml(text)
  words.forEach(qw => {
    const escapedQw = escapeHtml(qw)
    if (!escapedQw.length) return
    const regex = new RegExp(`(${escapedQw.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')})`, 'gi')
    escaped = escaped.replace(regex, '<mark class="search-highlight">$1</mark>')
  })
  return escaped
}

const filteredChapters = computed(() => {
  const words = props.queryWords
  if (!words.length) return props.chapters.map(ch => ({ ...ch, hasMatch: false }))
  return props.chapters.map(ch => {
    const hasMatch = words.some(qw => {
      const eQw = escapeHtml(qw).replace(/[.*+?^${}()|[\]\\]/g, '\\$&')
      const regex = new RegExp(eQw, 'gi')
      return regex.test(escapeHtml(ch.title)) || regex.test(escapeHtml(ch.content))
    })
    return { ...ch, hasMatch }
  })
})

const matchStats = computed(() => {
  const words = props.queryWords
  if (!words.length) return null
  let totalMatches = 0, chapterCount = 0
  filteredChapters.value.forEach(ch => {
    if (ch.hasMatch) {
      chapterCount++
      totalMatches += (highlightText(ch.content).match(/<mark/g) || []).length
      totalMatches += (highlightText(ch.title).match(/<mark/g) || []).length
    }
  })
  return { chapterCount, totalMatches }
})

function toggleChapter(index: number) {
  openChapters.value.has(index) ? openChapters.value.delete(index) : openChapters.value.add(index)
}

function doSearch(query?: string) { emit('search', query ?? localQuery.value) }
function doClear() { localQuery.value = ''; emit('clear-search') }

watch(() => props.queryWords, async () => {
  if (props.queryWords.length > 0) {
    const matchIdx = filteredChapters.value.findIndex(ch => ch.hasMatch)
    if (matchIdx >= 0) { openChapters.value.add(matchIdx); await nextTick(); document.querySelector('.lit-card.search-match')?.scrollIntoView({ behavior: 'smooth', block: 'center' }) }
  } else { openChapters.value.clear() }
}, { flush: 'post' })
</script>

<template>
  <div class="chart-card" id="literatureCard">
    <div class="chart-title"><span class="bar-marker bar-gold"></span>文献检索 · 快速定位<span style="font-size:0.6rem;color:var(--color-text-tertiary);"> | 点击标题展开</span></div>
    <div class="search-bar-wrap">
      <input v-model="localQuery" type="text" class="search-input" placeholder="搜索文献标题或内容..." autocomplete="off"
        @keydown.enter="doSearch()" @input="doSearch(localQuery)" />
      <button class="search-btn" @click="doSearch()">搜索</button>
      <button class="search-clear-btn" @click="doClear">清除</button>
    </div>
    <div class="loc-search-tags">
      <span style="font-size:0.6rem;color:var(--color-text-tertiary);">快速定位:</span>
      <span v-for="name in locationNames" :key="name" class="loc-search-tag"
        :class="{ 'active-search-tag': searchQuery.includes(name) }"
        @click="localQuery = name; doSearch(name)">{{ name }}</span>
    </div>
    <div v-if="matchStats" class="search-info">"{{ searchQuery }}" — {{ matchStats.chapterCount }} 章 · {{ matchStats.totalMatches }} 处匹配</div>
    <div v-if="loading" class="loading">加载中...</div>
    <div v-else-if="!chapters.length" class="loading">暂无文献数据</div>
    <template v-else>
      <div v-for="(ch, idx) in filteredChapters" :key="idx"
        class="lit-card"
        :class="{
          'hidden-by-search': queryWords.length > 0 && !ch.hasMatch,
          'search-match': ch.hasMatch,
          'open': openChapters.has(idx) || ch.hasMatch
        }"
        @click="toggleChapter(idx)">
        <div class="lit-title"><span v-html="highlightText(ch.title)"></span><span class="lit-arrow" :class="{ 'lit-arrow-match': ch.hasMatch }">{{ openChapters.has(idx) ? '▼' : '▶' }}</span></div>
        <div class="lit-content" v-html="highlightText(ch.content)"></div>
      </div>
    </template>
  </div>
</template>

<style scoped>
/* ---------- 公共状态 ---------- */
.loading {
  color: var(--color-text-tertiary);
  text-align: center;
  padding: var(--space-5);
}

/* ---------- 搜索栏 ---------- */
.search-bar-wrap {
  display: flex;
  gap: var(--space-2);
  padding: var(--space-2);
  align-items: center;
  flex-wrap: wrap;
}
.search-input {
  flex: 1;
  min-width: 8rem;
  padding: 6px 12px;
  border-radius: var(--radius);
  border: 1px solid var(--color-border);
  background: var(--color-bg-primary);
  color: var(--color-text-primary);
  font-size: 0.72rem;
  outline: none;
  transition:
    border-color var(--dur) var(--ease),
    box-shadow var(--dur) var(--ease),
    background var(--dur) var(--ease);
}
.search-input:focus {
  border-color: var(--color-accent);
  box-shadow: var(--shadow-focus);
  background: var(--color-bg-primary);
}
.search-input::placeholder {
  color: var(--color-text-placeholder);
  font-style: italic;
}
.search-btn {
  padding: 6px 14px;
  border-radius: var(--radius);
  border: 1px solid var(--color-accent);
  background: var(--color-accent-light);
  color: var(--color-accent);
  font-size: 0.72rem;
  cursor: pointer;
  font-weight: 600;
  white-space: nowrap;
  transition:
    background var(--dur) var(--ease),
    color var(--dur) var(--ease),
    transform var(--dur) var(--ease),
    box-shadow var(--dur) var(--ease);
}
.search-btn:hover {
  background: var(--color-accent);
  color: #ffffff;
  transform: translateY(-1px);
  box-shadow: var(--shadow-sm);
}
.search-clear-btn {
  padding: 6px 10px;
  border-radius: var(--radius);
  border: 1px solid var(--color-border);
  background: var(--color-bg-secondary);
  color: var(--color-text-secondary);
  font-size: 0.72rem;
  cursor: pointer;
  transition:
    border-color var(--dur) var(--ease),
    color var(--dur) var(--ease),
    background var(--dur) var(--ease),
    transform var(--dur) var(--ease),
    box-shadow var(--dur) var(--ease);
}
.search-clear-btn:hover {
  border-color: var(--color-accent);
  color: var(--color-accent);
  background: var(--color-bg-hover);
  transform: translateY(-1px);
  box-shadow: var(--shadow-sm);
}

/* ---------- 搜索提示 ---------- */
.search-info {
  font-size: 0.58rem;
  color: var(--color-accent);
  padding: 2px var(--space-3) 4px;
  text-align: center;
}

/* ---------- 地点快速定位 Tag ---------- */
.loc-search-tags {
  display: flex;
  flex-wrap: wrap;
  gap: var(--space-1);
  padding: var(--space-1) var(--space-2) var(--space-2);
  align-items: center;
}
.loc-search-tag {
  font-size: 0.58rem;
  padding: 3px 8px;
  border-radius: var(--radius-sm);
  background: var(--color-bg-secondary);
  border: 1px solid var(--color-border);
  color: var(--color-text-secondary);
  cursor: pointer;
  transition:
    background var(--dur) var(--ease),
    border-color var(--dur) var(--ease),
    color var(--dur) var(--ease),
    transform var(--dur) var(--ease),
    box-shadow var(--dur) var(--ease);
}
.loc-search-tag:hover {
  background: var(--color-bg-hover);
  border-color: var(--color-accent);
  color: var(--color-accent);
  transform: translateY(-1px);
  box-shadow: var(--shadow-sm);
}
.loc-search-tag.active-search-tag {
  background: var(--color-accent-light);
  color: var(--color-accent);
  border-color: var(--color-accent);
  font-weight: 600;
}

/* ---------- 文献卡片 ---------- */
.lit-card {
  background: var(--color-bg-secondary);
  border-radius: var(--radius);
  padding: var(--space-2) var(--space-3);
  margin: var(--space-2) var(--space-4);
  border-left: 2px solid var(--color-border-dark);
  cursor: pointer;
  transition:
    background var(--dur) var(--ease),
    border-left-color var(--dur) var(--ease),
    transform var(--dur) var(--ease),
    box-shadow var(--dur) var(--ease);
}
.lit-card:hover {
  background: var(--color-bg-hover);
  transform: translateY(-1px);
  box-shadow: var(--shadow-sm);
}
.lit-card.hidden-by-search {
  display: none;
}
.lit-card.search-match {
  border-left-color: var(--color-accent);
  box-shadow: 0 0 0 2px var(--color-accent-ring);
  animation: pulseMatch 2s ease-in-out infinite;
}
@keyframes pulseMatch {
  0%, 100% { box-shadow: 0 0 0 2px var(--color-accent-ring); }
  50% { box-shadow: 0 0 0 4px var(--color-accent-ring); }
}
.lit-title {
  font-weight: 600;
  color: var(--color-text-primary);
  display: flex;
  justify-content: space-between;
  align-items: center;
  font-size: 0.76rem;
}
.lit-arrow {
  font-size: 0.7rem;
  color: var(--color-text-tertiary);
  transition: transform var(--dur-fast) var(--ease);
}
.lit-arrow-match {
  color: var(--color-accent);
}
.lit-content {
  font-size: 0.68rem;
  color: var(--color-text-secondary);
  margin-top: var(--space-2);
  line-height: 1.55;
  display: none;
  white-space: pre-wrap;
  max-height: 14rem;
  overflow-y: auto;
  padding-right: 4px;
}
.lit-card.open .lit-content {
  display: block;
}
.lit-content :deep(mark.search-highlight) {
  background: var(--color-accent-light);
  color: var(--color-accent);
  padding: 1px 4px;
  border-radius: var(--radius-sm);
  font-weight: 600;
}
</style>
