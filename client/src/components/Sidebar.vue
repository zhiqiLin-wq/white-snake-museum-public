<script setup lang="ts">
import { computed } from 'vue'
import type { LocationData, Chapter } from '../types'
import StatsGrid from './StatsGrid.vue'
import LocationCloud from './LocationCloud.vue'
import StackedBarChart from './charts/StackedBarChart.vue'
import LiteraturePanel from './LiteraturePanel.vue'
import SankeyChart from './charts/SankeyChart.vue'
import SunburstChart from './charts/SunburstChart.vue'
import HeatmapChart from './charts/HeatmapChart.vue'
import GraphChart from './charts/GraphChart.vue'
import LocationDetail from './LocationDetail.vue'

const props = defineProps<{
  locationsArray: LocationData[]
  selectedData: LocationData | null
  selectedLocation: string | null
  locationNames: string[]
  totalMatchSum: number
  totalWordsSum: number
  locationCount: number
  chapters: Chapter[]
  searchQuery: string
  queryWords: string[]
  litLoading: boolean
  agentStatus?: 'online' | 'offline' | 'starting' | 'unknown'
}>()

const emit = defineEmits<{
  'select-location': [name: string]
  'search': [query: string]
  'clear-search': []
  'open-agent': []
}>()

// 图表面板列表：配置式渲染，避免重复 DOM 结构
const chartList = computed(() => [
  { title: '朝代分布 · 柱状图',   comp: StackedBarChart, key: 'bar' },
  { title: '朝代流向 · 桑基图',   comp: SankeyChart,    key: 'sankey' },
  { title: '地点层级 · 旭日图',   comp: SunburstChart,  key: 'sun' },
  { title: '朝代对比 · 热力图',   comp: HeatmapChart,   key: 'heat' },
  { title: '关联网络 · 关系图',   comp: GraphChart,     key: 'graph' },
])

function onLiteratureTrace(name: string) {
  emit('search', name)
  setTimeout(() => {
    ;(document.getElementById('literatureCard') as HTMLElement | null)?.scrollIntoView({ behavior: 'smooth', block: 'center' })
  }, 100)
}
</script>

<template>
  <aside class="sidebar">
    <div class="sidebar-inner">
      <!-- Header -->
      <div class="museum-header">
        <div class="brand-row">
          <div class="brand-mark-sm" aria-hidden="true"></div>
          <div class="brand-wrap">
            <h1>白蛇传 · 文脉全息</h1>
            <div class="sub">11 处文本景观 &nbsp;|&nbsp; 宋 · 清 · 近现代</div>
          </div>
        </div>
      </div>

      <!-- Agent 卡片 -->
      <div class="agent-card" @click="emit('open-agent')">
        <div class="agent-icon">
          <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
            <path d="M12 2a4 4 0 0 1 4 4c0 1.5-.8 2.8-2 3.5V12c0 1.5.8 2.8 2 3.5a4 4 0 0 1-4 4 4 4 0 0 1-4-4c0-1.5.8-2.8 2-3.5V9.5A4 4 0 0 1 8 6a4 4 0 0 1 4-4z"/>
          </svg>
        </div>
        <div class="agent-body">
          <div class="agent-title">AI 研究助手</div>
          <div class="agent-sub">随时提问，协助考据</div>
        </div>
        <span
          class="agent-status"
          :class="{
            'is-online': agentStatus === 'online',
            'is-offline': agentStatus === 'offline',
            'is-starting': agentStatus === 'starting',
          }"
        ></span>
      </div>

      <StatsGrid
        :total-match-sum="totalMatchSum"
        :total-words-sum="totalWordsSum"
        :location-count="locationCount"
      />

      <LocationCloud
        :location-names="locationNames"
        :selected-location="selectedLocation"
        @select="name => emit('select-location', name)"
      />

      <section class="panel" v-for="(cfg, i) in chartList" :key="cfg.key">
        <div class="panel-head">
          <span class="panel-dot"></span>
          <span class="panel-title">{{ cfg.title }}</span>
        </div>
        <component :is="cfg.comp" :locations="props.locationsArray" />
      </section>

      <!-- LiteraturePanel：带 ID 支持滚动定位 -->
      <div id="literatureCard" class="panel">
        <div class="panel-head">
          <span class="panel-dot"></span>
          <span class="panel-title">文献检索 · 快速定位</span>
        </div>
        <LiteraturePanel
          :chapters="props.chapters"
          :location-names="props.locationNames"
          :search-query="props.searchQuery"
          :query-words="props.queryWords"
          :loading="props.litLoading"
          @search="(q: string) => emit('search', q)"
          @clear-search="() => emit('clear-search')"
        />
      </div>

      <!-- LocationDetail 放 litPanel 后，便于滚动定位联动 -->
      <LocationDetail
        :selected-data="props.selectedData"
        :selected-location="props.selectedLocation"
        @literature-trace="onLiteratureTrace"
      />

      <footer class="sidebar-foot">数据源：excel_data 语料库 &nbsp;|&nbsp; 仅供研究</footer>
    </div>
  </aside>
</template>

<style scoped>
/* ============================================================
   Sidebar · 极简黑白
============================================================ */
.sidebar {
  height: 100%;
  overflow-y: auto;
  overflow-x: hidden;
  background: var(--color-bg-primary);
  border-right: 1px solid var(--color-border);
}
.sidebar-inner {
  min-height: 100%;
  display: flex;
  flex-direction: column;
  padding: 0 0 var(--space-6);
}

/* ---------- Header ---------- */
.museum-header {
  padding: 20px 20px 16px;
  border-bottom: 1px solid var(--color-border-light);
  position: sticky;
  top: 0;
  background: rgba(255,255,255,.82);
  backdrop-filter: saturate(180%) blur(10px);
  -webkit-backdrop-filter: saturate(180%) blur(10px);
  z-index: 2;
}
.brand-row { display: flex; align-items: center; gap: 12px; }
.brand-mark-sm {
  width: 30px; height: 30px;
  border-radius: 8px;
  background: linear-gradient(135deg, var(--color-accent), #8E9FD0);
  box-shadow: 0 2px 6px rgba(91,155,213,.25);
  flex-shrink: 0;
  position: relative;
}
.brand-mark-sm::after {
  content: '';
  position: absolute;
  inset: 7px;
  border-radius: 50%;
  background: rgba(255,255,255,.85);
}
.brand-wrap { min-width: 0; flex: 1; }
.museum-header h1 {
  font-size: 1rem;
  font-weight: 600;
  color: var(--color-text-primary);
  letter-spacing: 0.01em;
  line-height: 1.2;
}
.sub {
  margin-top: 3px;
  font-size: 0.7rem;
  color: var(--color-text-tertiary);
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}

/* ---------- Agent Card ---------- */
.agent-card {
  margin: 14px 16px 0;
  padding: 12px 14px;
  border-radius: var(--radius-lg);
  background: var(--color-bg-secondary);
  border: 1px solid var(--color-border);
  display: flex;
  align-items: center;
  gap: 12px;
  cursor: pointer;
  transition:
    border-color var(--dur) var(--ease),
    background-color var(--dur) var(--ease),
    box-shadow var(--dur) var(--ease),
    transform var(--dur) var(--ease);
}
.agent-card:hover {
  border-color: var(--color-accent);
  background: #FFFFFF;
  box-shadow: var(--shadow-md);
  transform: translateY(-1px);
}
.agent-icon {
  width: 34px; height: 34px;
  display: flex; align-items: center; justify-content: center;
  border-radius: var(--radius);
  background: var(--color-accent-light);
  color: var(--color-accent);
  flex-shrink: 0;
}
.agent-body { flex: 1; min-width: 0; }
.agent-title {
  font-size: 0.85rem;
  font-weight: 600;
  color: var(--color-text-primary);
}
.agent-sub {
  font-size: 0.7rem;
  color: var(--color-text-tertiary);
  margin-top: 2px;
}
.agent-status {
  width: 8px; height: 8px; border-radius: 50%;
  background: #C8CED6;
  flex-shrink: 0;
}
.agent-status.is-online   { background: var(--color-success); }
.agent-status.is-offline  { background: var(--color-error); }
.agent-status.is-starting { background: var(--color-warning); animation: pulseDot 1.2s infinite; }
@keyframes pulseDot { 0%, 100% { opacity: 1; } 50% { opacity: 0.4; } }

/* ---------- 图表面板 ---------- */
.panel {
  margin: 14px 16px 0;
  border: 1px solid var(--color-border);
  border-radius: var(--radius-lg);
  background: var(--color-bg-primary);
  padding: 12px 12px 10px;
  transition: border-color var(--dur) var(--ease), box-shadow var(--dur) var(--ease);
}
.panel:hover {
  border-color: var(--color-border-dark);
  box-shadow: var(--shadow-sm);
}
.panel-head {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 0 4px 10px;
}
.panel-dot {
  width: 6px; height: 6px;
  border-radius: 999px;
  background: var(--color-accent);
  opacity: 0.75;
}
.panel-title {
  font-size: 0.78rem;
  font-weight: 600;
  color: var(--color-text-primary);
}

/* ---------- Literature 占位（LiteraturePanel 自己带样式，这里给外边距） ---------- */
.sidebar-inner > :deep(.literature-panel-root) {
  margin: 14px 16px 0;
}

/* ---------- Footer ---------- */
.sidebar-foot {
  padding: 18px 20px 6px;
  text-align: center;
  font-size: 0.62rem;
  color: var(--color-text-placeholder);
  margin-top: auto;
}
</style>
