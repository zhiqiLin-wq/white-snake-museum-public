<script setup lang="ts">
import { ref, computed, onMounted } from 'vue'
import { useWorkspaceStore } from '@/stores/workspace'
import StackedBarChart from '@/components/charts/StackedBarChart.vue'
import SankeyChart from '@/components/charts/SankeyChart.vue'
import SunburstChart from '@/components/charts/SunburstChart.vue'
import HeatmapChart from '@/components/charts/HeatmapChart.vue'
import GraphChart from '@/components/charts/GraphChart.vue'
import type { LocationData } from '@/types'

const workspaceStore = useWorkspaceStore()

const chartType = computed(() => {
  const tab = workspaceStore.activeTab
  if (!tab) return null
  return tab.type
})

const locations = ref<LocationData[]>([])
const loading = ref(false)
const errorMessage = ref<string | null>(null)

// 明确的页面标题（之前是空字符串 -> 用户打开不知道这页叫什么）
const chartLabelMap: Record<string, { title: string; subtitle: string }> = {
  'chart-stacked-bar': { title: '景观出现频次统计 · 堆叠柱状图', subtitle: '展示 11 处景观在白蛇传各文献中的文本字数分布（按字数区段堆叠）' },
  'chart-sankey':     { title: '朝代流向 · 桑基图',                  subtitle: '展示景观叙事在不同朝代之间的演变流向' },
  'chart-sunburst':   { title: '景观层级 · 旭日图',                  subtitle: '按朝代/景观/字数的层级分布展示累计规模' },
  'chart-chord':      { title: '共现关系 · 和弦图',                  subtitle: '展示景观之间的共现与关联强度' },
  'chart-heatmap':    { title: '朝代对比 · 热力图',                  subtitle: '每个景观在每个朝代/段落的文字热度对比' },
  'chart-graph':      { title: '关联网络 · 力导向图',                subtitle: '景观与叙事要素的关系网络可视化' },
}

const pageMeta = computed(() => chartType.value ? chartLabelMap[chartType.value] ?? { title: chartType.value, subtitle: '' } : { title: '', subtitle: '' })

// 数据摘要（雷峰塔信息、总字数、总景观数、总记录条数），用户一进来就看见有数据
const summary = computed(() => {
  const totalLocations = locations.value.length
  const totalRecords   = locations.value.reduce((s, l) => s + (l.records?.length || 0), 0)
  const totalWords     = locations.value.reduce((s, l) => s + (l.totalWords || 0), 0)
  const leifeng        = locations.value.find(l => l.name === '雷峰塔')
  return {
    totalLocations,
    totalRecords,
    totalWords,
    leifengRecords: leifeng?.records?.length ?? 0,
    leifengWords:   leifeng?.totalWords ?? 0,
  }
})

async function fetchLocations() {
  loading.value = true
  errorMessage.value = null
  try {
    const res = await fetch('/api/locations')
    if (!res.ok) throw new Error(`HTTP ${res.status} - ${res.statusText}`)
    const data = await res.json()
    if (!Array.isArray(data)) throw new Error('后端返回的数据不是数组格式')
    locations.value = data as LocationData[]
  } catch (err: unknown) {
    errorMessage.value = err instanceof Error ? err.message : String(err)
    // 出错了也在 console 打一份，DevTools 里能看到
    console.error('[ChartContainerView] fetchLocations failed:', err)
  } finally {
    loading.value = false
  }
}

onMounted(() => {
  fetchLocations()
})
</script>

<template>
  <div class="chart-container-view">
    <!-- Header：标题 + 副标题 -->
    <div class="cv-header">
      <h2>{{ pageMeta.title }}</h2>
      <p v-if="pageMeta.subtitle" class="cv-subtitle">{{ pageMeta.subtitle }}</p>
    </div>

    <!-- 加载中 -->
    <div v-if="loading" class="cv-loading">
      <div class="cv-spinner" />
      <p>正在从服务端加载景观频次统计数据...</p>
    </div>

    <!-- 错误态 -->
    <div v-else-if="errorMessage" class="cv-error">
      <p class="cve-title">加载失败：{{ errorMessage }}</p>
      <p class="cve-hint">提示：请确认 Fastify 后端已启动（端口 3000）。你可以在终端执行 <code>cd server && npm run dev</code>。</p>
      <button @click="fetchLocations">↻ 重试</button>
    </div>

    <!-- 空态：0 条数据但也没报错（后端返回空数组） -->
    <div v-else-if="locations.length === 0" class="cv-empty">
      <div class="cve-icon">📭</div>
      <div class="cve-title">后端返回了 0 条景观数据</div>
      <div class="cve-hint">excel_data 目录下可能没有 xlsx 文件，或文件解析后全部为空。请检查：</div>
      <ul class="cve-list">
        <li>excel_data 目录存在，且有 雷峰塔的结果数据.xlsx / 金山寺分词结果.xlsx 等文件</li>
        <li>各 xlsx 文件的首行 header 必须包含「切词数 / 匹配词数 / 字数 / 文本内容」列（或 金山寺专用格式）</li>
        <li>Network 面板查看 /api/locations 的响应体（应为非空 JSON 数组，元素含 name/records/文本字数）</li>
      </ul>
      <button @click="fetchLocations">↻ 重试加载</button>
    </div>

    <!-- 正常态：数据摘要 + 图表 -->
    <div v-else class="cv-chart-area">
      <!-- 摘要行：用户一眼看见"雷峰塔的 X 条数据已经拿到了"，不是白屏 -->
      <div class="cv-summary">
        <div class="cv-s-cell">
          <span class="cv-s-label">景观数</span>
          <span class="cv-s-value">{{ summary.totalLocations }}</span>
        </div>
        <div class="cv-s-cell">
          <span class="cv-s-label">累计段落</span>
          <span class="cv-s-value">{{ summary.totalRecords.toLocaleString() }}</span>
        </div>
        <div class="cv-s-cell">
          <span class="cv-s-label">累计字数</span>
          <span class="cv-s-value">{{ summary.totalWords.toLocaleString() }}</span>
        </div>
        <div class="cv-s-cell cv-s-highlight">
          <span class="cv-s-label">雷峰塔</span>
          <span class="cv-s-value">{{ summary.leifengRecords }} 段 · {{ summary.leifengWords.toLocaleString() }} 字</span>
        </div>
      </div>

      <div class="cv-chart-wrap">
        <StackedBarChart
          v-if="chartType === 'chart-stacked-bar'"
          :locations="locations"
        />
        <SankeyChart
          v-else-if="chartType === 'chart-sankey'"
          :locations="locations"
        />
        <SunburstChart
          v-else-if="chartType === 'chart-sunburst'"
          :locations="locations"
        />
        <HeatmapChart
          v-else-if="chartType === 'chart-heatmap'"
          :locations="locations"
        />
        <GraphChart
          v-else-if="chartType === 'chart-graph'"
          :locations="locations"
        />
      </div>
    </div>
  </div>
</template>

<style scoped>
.chart-container-view {
  display: flex;
  flex-direction: column;
  height: 100%;
  background: var(--color-bg-primary);
}

/* ========= Header ========= */
.cv-header {
  padding: var(--space-3) var(--space-5);
  border-bottom: 1px solid var(--color-border);
  background: var(--color-bg-secondary);
}

.cv-header h2 {
  font-size: 1rem;
  font-weight: 700;
  color: var(--color-text-primary);
  margin: 0 0 var(--space-1) 0;
  letter-spacing: 0.02em;
}

.cv-subtitle {
  font-size: 0.75rem;
  color: var(--color-text-tertiary);
  margin: 0;
  line-height: 1.5;
}

/* ========= Loading ========= */
.cv-loading {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  flex: 1;
  color: var(--color-text-secondary);
  font-size: 0.85rem;
  gap: var(--space-4);
}

.cv-spinner {
  width: 28px;
  height: 28px;
  border-radius: var(--radius-full);
  border: 1px solid var(--color-border);
  border-top: 1px solid var(--color-accent);
  box-shadow: inset 0 0 0 1px var(--color-accent-ring);
  animation: cv-spin 0.8s linear infinite;
}

@keyframes cv-spin {
  to { transform: rotate(360deg); }
}

/* ========= Error & Empty ========= */
.cv-error,
.cv-empty {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: flex-start;
  flex: 1;
  padding: var(--space-8) var(--space-6);
  gap: var(--space-2);
  text-align: center;
  color: var(--color-text-secondary);
}

.cv-error .cve-title {
  color: var(--color-error);
  font-weight: 700;
}

.cve-icon {
  font-size: 46px;
  margin-bottom: var(--space-1);
  filter: grayscale(0.2);
}

.cve-title {
  font-size: 1rem;
  color: var(--color-text-primary);
  font-weight: 600;
  margin: 0;
}

.cve-hint {
  font-size: 0.82rem;
  color: var(--color-text-tertiary);
  margin: var(--space-1) 0 0 0;
  line-height: 1.5;
}

.cve-list {
  text-align: left;
  margin: 0;
  padding: 0 0 0 var(--space-5);
  font-size: 0.8rem;
  color: var(--color-text-secondary);
  line-height: 1.9;
  max-width: 520px;
}

.cve-list code {
  background: var(--color-bg-tertiary);
  padding: 1px var(--space-2);
  border-radius: var(--radius-sm);
  font-size: 0.78rem;
  color: var(--color-text-primary);
  font-family: var(--font-mono);
  border: 1px solid var(--color-border);
}

.cv-error button,
.cv-empty button {
  padding: var(--space-2) var(--space-5);
  background: var(--color-bg-primary);
  color: var(--color-text-primary);
  border: 1px solid var(--color-border);
  border-radius: var(--radius-sm);
  cursor: pointer;
  font-size: 0.85rem;
  margin-top: var(--space-3);
  font-weight: 500;
  transition: background-color var(--dur-fast) var(--ease),
              color var(--dur-fast) var(--ease),
              border-color var(--dur-fast) var(--ease),
              box-shadow var(--dur-fast) var(--ease);
}

.cv-error button:hover,
.cv-empty button:hover {
  background: var(--color-accent-light);
  color: var(--color-accent);
  border-color: var(--color-accent);
  box-shadow: var(--shadow-focus);
}

/* ========= Summary row ========= */
.cv-summary {
  width: calc(100% - 40px);
  max-width: 1100px;
  margin: var(--space-4) auto var(--space-2);
  display: grid;
  grid-template-columns: repeat(4, 1fr);
  gap: var(--space-3);
}

.cv-s-cell {
  background: var(--color-bg-primary);
  border: 1px solid var(--color-border);
  border-radius: var(--radius-lg);
  padding: var(--space-3) var(--space-4);
  display: flex;
  flex-direction: column;
  gap: var(--space-1);
  box-shadow: var(--shadow-sm);
  transition: box-shadow var(--dur-fast) var(--ease);
}

.cv-s-cell:hover {
  box-shadow: var(--shadow);
}

.cv-s-highlight {
  background: var(--color-accent-light);
  border-color: var(--color-accent);
}

.cv-s-highlight .cv-s-label {
  color: var(--color-accent);
  font-weight: 600;
}

.cv-s-highlight .cv-s-value {
  color: var(--color-accent);
}

.cv-s-label {
  font-size: 0.7rem;
  color: var(--color-text-tertiary);
  letter-spacing: 0.03em;
  font-weight: 500;
}

.cv-s-value {
  font-size: 1.05rem;
  font-weight: 700;
  color: var(--color-text-primary);
  font-family: var(--font-mono);
  line-height: 1.2;
}

/* ========= Chart wrap ========= */
.cv-chart-area {
  flex: 1;
  overflow-y: auto;
  overflow-x: hidden;
  display: flex;
  flex-direction: column;
  align-items: stretch;
  padding: var(--space-4) 0;
}

.cv-chart-wrap {
  width: calc(100% - 40px);
  max-width: 1100px;
  margin: 0 auto;
  min-height: 540px;
  flex-shrink: 0;
  background: var(--color-bg-primary);
  border: 1px solid var(--color-border);
  border-radius: var(--radius);
  box-shadow: var(--shadow-sm);
  padding: var(--space-3);
}

.cv-chart-area::-webkit-scrollbar { width: 6px; }
.cv-chart-area::-webkit-scrollbar-track { background: transparent; }
.cv-chart-area::-webkit-scrollbar-thumb {
  background: var(--color-border);
  border-radius: var(--radius-full);
}
.cv-chart-area::-webkit-scrollbar-thumb:hover { background: var(--color-text-secondary); }

@media (max-width: 768px) {
  .cv-summary { grid-template-columns: repeat(2, 1fr); }
}
</style>
