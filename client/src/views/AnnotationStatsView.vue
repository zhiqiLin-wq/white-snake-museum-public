<script setup lang="ts">
import { ref, computed, watch, onMounted, onUnmounted } from 'vue'
import * as echarts from 'echarts'
import { useAnnotationStore } from '@/stores/annotations'
import { useSourceTreeStore } from '@/stores/sourceTree'
import { useWorkspaceStore } from '@/stores/workspace'

const annotationStore = useAnnotationStore()
const sourceTreeStore = useSourceTreeStore()
const workspaceStore = useWorkspaceStore()

const includeAgent = ref(true)
const pieChartEl = ref<HTMLDivElement | null>(null)
const vsPieChartEl = ref<HTMLDivElement | null>(null)
const vsStackChartEl = ref<HTMLDivElement | null>(null)
const barChartEl = ref<HTMLDivElement | null>(null)
const densityChartEl = ref<HTMLDivElement | null>(null)
let pieChart: echarts.ECharts | null = null
let vsPieChart: echarts.ECharts | null = null
let vsStackChart: echarts.ECharts | null = null
let barChart: echarts.ECharts | null = null
let densityChart: echarts.ECharts | null = null

const stats = computed(() => annotationStore.computeAnnotationStats(includeAgent.value))

function refreshStats() {
  stats.value
  renderCharts()
}

function formatNumber(n: number): string {
  if (n >= 10000) return (n / 10000).toFixed(1) + ''
  return n.toLocaleString('zh-CN')
}

function renderCharts() {
  const s = stats.value
  if (!s.total) return

  const fs = Math.round(parseFloat(getComputedStyle(document.documentElement).fontSize) * 0.65)

  // P1-16: Category distribution pie chart
  if (pieChartEl.value) {
    if (!pieChart) pieChart = echarts.init(pieChartEl.value)
    pieChart.setOption({
      tooltip: { trigger: 'item', formatter: '{b}: {c} ({d}%)' },
      series: [{
        type: 'pie',
        radius: ['40%', '70%'],
        center: ['50%', '50%'],
        data: s.categoryDistribution.map(c => ({
          name: c.categoryLabel,
          value: c.count,
          itemStyle: { color: c.color },
        })),
        label: { fontSize: fs },
        emphasis: { label: { fontSize: fs + 2, fontWeight: 'bold' } },
      }],
    })
  }

  // P1-16: User vs Agent pie chart (2 sectors)
  if (vsPieChartEl.value) {
    if (!vsPieChart) vsPieChart = echarts.init(vsPieChartEl.value)
    vsPieChart.setOption({
      title: { text: `: ${s.total}`, left: 'center', top: '42%', textStyle: { fontSize: fs + 2, color: '#7A6A58' } },
      tooltip: { trigger: 'item', formatter: '{b}: {c} ({d}%)' },
      series: [{
        type: 'pie',
        radius: ['40%', '70%'],
        center: ['50%', '50%'],
        data: [
          { name: '', value: s.userCount, itemStyle: { color: '#5B9BD5' } },
          { name: 'Agent', value: s.agentCount, itemStyle: { color: '#7EC4A0' } },
        ],
        label: { fontSize: fs, formatter: '{b}: {c} ({d}%)' },
      }],
    })
  }

  // P1-16: User vs Agent stacked bar (6 categories x 2 sources)
  if (vsStackChartEl.value) {
    if (!vsStackChart) vsStackChart = echarts.init(vsStackChartEl.value)
    vsStackChart.setOption({
      tooltip: { trigger: 'axis' },
      legend: { data: ['', 'Agent'], textStyle: { fontSize: fs } },
      xAxis: { type: 'category', data: s.userVsAgentByCategory.map(c => c.categoryLabel), axisLabel: { fontSize: fs } },
      yAxis: { type: 'value', axisLabel: { fontSize: fs } },
      series: [
        { name: '', type: 'bar', stack: 'total', data: s.userVsAgentByCategory.map(c => c.userCount), itemStyle: { color: '#5B9BD5' }, label: { show: true, position: 'inside', fontSize: fs - 2 } },
        { name: 'Agent', type: 'bar', stack: 'total', data: s.userVsAgentByCategory.map(c => c.agentCount), itemStyle: { color: '#7EC4A0' }, label: { show: true, position: 'inside', fontSize: fs - 2 } },
      ],
    })
  }

  // Dynasty distribution bar chart
  if (barChartEl.value) {
    if (!barChart) barChart = echarts.init(barChartEl.value)
    barChart.setOption({
      tooltip: { trigger: 'axis' },
      xAxis: { type: 'category', data: s.dynastyDistribution.map(d => d.dynasty), axisLabel: { fontSize: fs } },
      yAxis: { type: 'value', axisLabel: { fontSize: fs } },
      series: [{
        type: 'bar',
        data: s.dynastyDistribution.map(d => d.count),
        itemStyle: { color: '#5B9BD5' },
        label: { show: true, position: 'top', fontSize: fs },
      }],
    })
  }

  // P1-16: Chapter density horizontal stacked bar
  if (densityChartEl.value) {
    if (!densityChart) densityChart = echarts.init(densityChartEl.value)
    densityChart.setOption({
      tooltip: { trigger: 'axis' },
      legend: { data: ['', ''], textStyle: { fontSize: fs } },
      xAxis: { type: 'value', axisLabel: { fontSize: fs }, name: '' },
      yAxis: {
        type: 'category',
        data: s.chapterDensity.map(c => `${c.dynasty} - ${c.chapterTitle.slice(0, 8)}`),
        axisLabel: { fontSize: fs - 1 },
      },
      series: [
        {
          name: '', type: 'bar', stack: 'total',
          data: s.chapterDensity.map(c => c.annotatedParagraphs),
          itemStyle: { color: '#7EC4A0' },
          label: { show: true, formatter: (p: { value: number }) => p.value > 0 ? p.value.toString() : '', fontSize: fs - 2 },
        },
        {
          name: '', type: 'bar', stack: 'total',
          data: s.chapterDensity.map(c => c.totalParagraphs - c.annotatedParagraphs),
          itemStyle: { color: '#E4E7EB' },
        },
      ],
    })
  }
}

function exportStatsReport() {
  const csv = annotationStore.exportStatsCSV(includeAgent.value)
  const blob = new Blob([csv], { type: 'text/csv;charset=utf-8' })
  const url = URL.createObjectURL(blob)
  const a = document.createElement('a')
  a.href = url; a.download = 'annotation-stats.csv'; a.click()
  URL.revokeObjectURL(url)
}

function goToReading() {
  if (sourceTreeStore.chapters.length > 0) {
    const ch = sourceTreeStore.chapters[0]
    workspaceStore.openTab('text-reader', `${ch.dynasty} · ${ch.title}`, {
      chapterNumber: ch.number,
    })
  }
}

// P2-16: Navigate to passage on timeline click
function onTimelineItemClick(item: { passageKey: string; chapterNumber?: number; paragraphIndex?: number }) {
  const parts = item.passageKey.split(':')
  const chNum = item.chapterNumber || parseInt(parts[0], 10) || 1
  const paraIdx = item.paragraphIndex || parseInt(parts[1], 10) || 0
  sourceTreeStore.setNavigationTarget({ chapterNumber: chNum, paragraphIndex: paraIdx })
  const tch = sourceTreeStore.chapters.find(c => c.number === chNum)
  workspaceStore.openTab('text-reader', tch ? `${tch.dynasty} · ${tch.title}` : `章节${chNum}`, { chapterNumber: chNum, paragraphIndex: paraIdx })
}

watch(() => includeAgent.value, () => {
  setTimeout(renderCharts, 100)
})

onMounted(() => {
  setTimeout(renderCharts, 300)
})

onUnmounted(() => {
  pieChart?.dispose()
  vsPieChart?.dispose()
  vsStackChart?.dispose()
  barChart?.dispose()
  densityChart?.dispose()
})

// P2-16: Timeline color mapping
function timelineColor(action: string): string {
  if (action === 'user_annotate') return '#7EC4A0'
  if (action === 'agent_annotate') return '#5B9BD5'
  if (action.includes('delete')) return '#C89898'
  return '#B8B8B0'
}
</script>

<template>
  <div class="annotation-stats-view">
    <div class="as-toolbar">
      <div class="as-filter-group">
        <label class="as-checkbox">
          <input type="checkbox" v-model="includeAgent" @change="refreshStats" /> 包含Agent标注
        </label>
      </div>
      <div class="as-actions">
        <button class="as-action-btn" @click="refreshStats">刷新统计</button>
        <button class="as-action-btn" @click="exportStatsReport">导出报告</button>
      </div>
    </div>

    <div v-if="stats.total === 0" class="as-empty">
      <p>暂无标注数据</p>
      <button class="as-go-btn" @click="goToReading">前往文献阅读</button>
    </div>

    <div v-else class="as-content">
      <!-- P3-10: Overview cards with formatted numbers -->
      <div class="as-cards">
        <div class="as-card">
          <div class="as-card-value">{{ formatNumber(stats.total) }}</div>
          <div class="as-card-label">总标注数</div>
        </div>
        <div class="as-card">
          <div class="as-card-value">{{ formatNumber(stats.userCount) }}</div>
          <div class="as-card-label">用户标注</div>
        </div>
        <div class="as-card">
          <div class="as-card-value">{{ formatNumber(stats.agentCount) }}</div>
          <div class="as-card-label">Agent标注</div>
        </div>
        <div class="as-card">
          <div class="as-card-value">{{ formatNumber(stats.withMarginalia) }}</div>
          <div class="as-card-label">含批注</div>
        </div>
        <div class="as-card">
          <div class="as-card-value">{{ formatNumber(stats.paragraphsCovered) }}</div>
          <div class="as-card-label">覆盖段落</div>
        </div>
        <div class="as-card">
          <div class="as-card-value">{{ stats.coverageRate }}%</div>
          <div class="as-card-label">段落覆盖率</div>
        </div>
      </div>

      <!-- P1-16: Charts row — User vs Agent pie + stacked bar -->
      <div class="as-chart-row">
        <div class="as-chart-box">
          <div class="as-chart-title">用户 vs Agent 标注占比</div>
          <div ref="vsPieChartEl" class="as-chart" style="min-height: 300px;"></div>
        </div>
        <div class="as-chart-box">
          <div class="as-chart-title">分类分布 (6类 x 用户/Agent)</div>
          <div ref="vsStackChartEl" class="as-chart" style="min-height: 300px;"></div>
        </div>
      </div>

      <!-- Charts row 2 -->
      <div class="as-chart-row">
        <div class="as-chart-box">
          <div class="as-chart-title"></div>
          <div ref="pieChartEl" class="as-chart" style="min-height: 280px;"></div>
        </div>
        <div class="as-chart-box">
          <div class="as-chart-title"></div>
          <div ref="barChartEl" class="as-chart" style="min-height: 280px;"></div>
        </div>
      </div>

      <!-- P1-16: Chapter density horizontal stacked bar -->
      <div class="as-chart-row">
        <div class="as-chart-box as-chart-full">
          <div class="as-chart-title"> ( + )</div>
          <div ref="densityChartEl" class="as-chart" style="min-height: 320px;"></div>
        </div>
      </div>

      <!-- P2-16: Timeline with color dots and click-to-navigate -->
      <div class="as-timeline-section">
        <div class="as-chart-title"></div>
        <div class="as-timeline">
          <div
            v-for="item in stats.timeline"
            :key="item.annotationId + item.timestamp"
            class="as-timeline-item"
            @click="onTimelineItemClick(item)"
          >
            <span class="as-tl-dot" :style="{ background: timelineColor(item.action) }"></span>
            <span class="as-tl-time">{{ item.timestampLabel }}</span>
            <span class="as-tl-category">[{{ item.category }}]</span>
            <span class="as-tl-label">{{ item.label }}</span>
            <span class="as-tl-passage">{{ item.passageKey }}</span>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<style scoped>
.annotation-stats-view {
  display: flex;
  flex-direction: column;
  height: 100%;
  overflow: hidden;
  background: var(--color-bg-primary);
}

/* ---------- Toolbar ---------- */
.as-toolbar {
  padding: var(--space-2) var(--space-4);
  border-bottom: 1px solid var(--color-border);
  background: var(--color-bg-secondary);
  display: flex;
  align-items: center;
  gap: var(--space-3);
}

.as-filter-group {
  display: flex;
  align-items: center;
  gap: var(--space-2);
}

.as-checkbox {
  font-size: 0.75rem;
  color: var(--color-text-secondary);
  display: flex;
  align-items: center;
  gap: var(--space-1);
  cursor: pointer;
  font-weight: 500;
}

.as-actions {
  margin-left: auto;
  display: flex;
  gap: var(--space-2);
}

.as-action-btn {
  padding: var(--space-1) var(--space-3);
  font-size: 0.7rem;
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

.as-action-btn:hover {
  background: var(--color-accent-light);
  color: var(--color-accent);
  border-color: var(--color-accent);
}

/* ---------- Empty state ---------- */
.as-empty {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  flex: 1;
  color: var(--color-text-tertiary);
  font-size: 0.85rem;
  gap: var(--space-3);
  padding: var(--space-8);
}

.as-go-btn {
  padding: var(--space-2) var(--space-5);
  font-size: 0.78rem;
  background: var(--color-accent);
  color: #fff;
  border: 1px solid var(--color-accent);
  border-radius: var(--radius-sm);
  cursor: pointer;
  font-weight: 500;
  transition: background-color var(--dur) var(--ease),
              border-color var(--dur) var(--ease),
              box-shadow var(--dur) var(--ease);
}

.as-go-btn:hover {
  background: var(--color-accent-hover);
  border-color: var(--color-accent-hover);
  box-shadow: var(--shadow-focus);
}

/* ---------- Content ---------- */
.as-content {
  flex: 1;
  overflow-y: auto;
  padding: var(--space-4);
}

/* ---------- Overview cards ---------- */
.as-cards {
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: var(--space-3);
  margin-bottom: var(--space-5);
}

.as-card {
  padding: var(--space-3) var(--space-4);
  background: var(--color-bg-primary);
  border: 1px solid var(--color-border);
  border-radius: var(--radius);
  text-align: center;
  box-shadow: var(--shadow-sm);
  transition: box-shadow var(--dur-fast) var(--ease),
              transform var(--dur-fast) var(--ease);
}

.as-card:hover {
  box-shadow: var(--shadow);
}

.as-card:first-child {
  border: 1px solid var(--color-accent);
  background: var(--color-accent-light);
}

.as-card:first-child .as-card-value {
  color: var(--color-accent);
}

.as-card-value {
  font-size: 1.4rem;
  font-weight: 700;
  color: var(--color-text-primary);
  line-height: 1.2;
  font-family: var(--font-mono);
}

.as-card-label {
  font-size: 0.7rem;
  color: var(--color-text-tertiary);
  margin-top: var(--space-1);
  letter-spacing: 0.03em;
  font-weight: 500;
}

/* ---------- Chart rows ---------- */
.as-chart-row {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: var(--space-4);
  margin-bottom: var(--space-5);
}

.as-chart-full {
  grid-column: 1 / -1;
}

.as-chart-box {
  background: var(--color-bg-primary);
  border: 1px solid var(--color-border);
  border-radius: var(--radius);
  padding: var(--space-3);
  box-shadow: var(--shadow-sm);
  transition: box-shadow var(--dur-fast) var(--ease);
}

.as-chart-box:hover {
  box-shadow: var(--shadow);
}

.as-chart-title {
  font-size: 0.78rem;
  color: var(--color-text-primary);
  margin-bottom: var(--space-2);
  font-weight: 600;
  padding-bottom: var(--space-2);
  border-bottom: 1px solid var(--color-border-light);
  letter-spacing: 0.01em;
}

.as-chart {
  width: 100%;
}

/* ---------- Timeline ---------- */
.as-timeline-section {
  background: var(--color-bg-primary);
  border: 1px solid var(--color-border);
  border-radius: var(--radius);
  padding: var(--space-3);
  box-shadow: var(--shadow-sm);
}

.as-timeline {
  max-height: 260px;
  overflow-y: auto;
  display: flex;
  flex-direction: column;
  gap: var(--space-1);
}

.as-timeline-item {
  display: flex;
  gap: var(--space-2);
  align-items: baseline;
  padding: var(--space-2) var(--space-3);
  font-size: 0.7rem;
  border-bottom: 1px solid var(--color-border-light);
  cursor: pointer;
  border-radius: var(--radius-sm);
  transition: background-color var(--dur-fast) var(--ease);
}

.as-timeline-item:last-child {
  border-bottom: none;
}

.as-timeline-item:hover {
  background: var(--color-accent-light);
}

.as-tl-dot {
  width: 8px;
  height: 8px;
  border-radius: var(--radius-full);
  flex-shrink: 0;
  margin-top: 5px;
  box-shadow: 0 0 0 1px rgba(0, 0, 0, 0.05);
}

.as-tl-time {
  color: var(--color-text-tertiary);
  flex-shrink: 0;
  font-family: var(--font-mono);
  font-size: 0.65rem;
  min-width: 56px;
}

.as-tl-category {
  color: var(--color-text-secondary);
  flex-shrink: 0;
  font-weight: 500;
  padding: 0 var(--space-1);
  background: var(--color-bg-tertiary);
  border-radius: var(--radius-sm);
}

.as-tl-label {
  color: var(--color-text-primary);
  font-weight: 500;
  flex: 1;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.as-tl-passage {
  color: var(--color-text-tertiary);
  margin-left: auto;
  font-family: var(--font-mono);
  font-size: 0.65rem;
  flex-shrink: 0;
}
</style>
