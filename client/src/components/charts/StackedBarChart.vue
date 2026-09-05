<script setup lang="ts">
import { ref, watch, onMounted, onUnmounted, nextTick, computed } from 'vue'
import * as echarts from 'echarts'
import type { LocationData } from '../../types'

const props = defineProps<{ locations: LocationData[] }>()
const chartEl = ref<HTMLDivElement | null>(null)
let chart: echarts.ECharts | null = null  // 组件级闭包变量（每个实例独立，模块级会被多 tab 覆盖）
let resizeObs: ResizeObserver | null = null

const hasData = computed(() => {
  if (!props.locations.length) return false
  // 至少有一条记录有 文本字数 > 0，否则画图就是一排零高度柱子 = 视觉空白
  return props.locations.some(loc => (loc.records?.reduce((s, r) => s + (r['文本字数'] || 0), 0) || 0) > 0)
})

function render() {
  if (!chartEl.value) return
  if (!props.locations.length) return

  if (!chart) {
    chart = echarts.init(chartEl.value)
    // 监听容器尺寸变化自动 resize（Flex 布局下容器尺寸晚于 DOM 出现时也能刷新）
    resizeObs = new ResizeObserver(() => chart?.resize())
    resizeObs.observe(chartEl.value)
  }

  const names = props.locations.map(l => l.name)
  const low: number[] = [], mid: number[] = [], high: number[] = []
  props.locations.forEach(loc => {
    let l = 0, m = 0, h = 0
    loc.records.forEach(r => {
      const t = r['文本字数'] || 0
      if (t <= 5) l += t
      else if (t <= 10) m += t
      else h += t
    })
    low.push(l); mid.push(m); high.push(h)
  })
  const fs = Math.max(10, Math.round(parseFloat(getComputedStyle(document.documentElement).fontSize) * 0.65))
  const fontFamily = "'Inter','Noto Sans SC','PingFang SC'"
  const palette = ['#5B9BD5','#7EC4A0','#9AA6D4','#D09EBC','#B4B482','#6FB9C4','#C4AE62','#9BA8B5']
  chart.setOption({
    color: palette,
    title: {
      text: '各景观文本片段字数分布（按字数区段堆叠）',
      left: 'center', top: 4,
      textStyle: { color: '#6B7280', fontSize: fs, fontWeight: 600, fontFamily },
    },
    tooltip: {
      trigger: 'axis',
      axisPointer: { type: 'shadow' },
      backgroundColor: '#FFFFFF',
      borderColor: '#E5E7EB',
      borderWidth: 1,
      textStyle: { color: '#1A1A1A', fontSize: 12, fontFamily },
      extraCssText: 'box-shadow: none;',
    },
    legend: {
      data: ['≤5字/段', '6-10字/段', '>10字/段'],
      top: 28,
      itemWidth: 12,
      itemHeight: 8,
      itemGap: 16,
      textStyle: { color: '#6B7280', fontSize: 11, fontFamily },
    },
    grid: { top: 72, left: 52, right: 20, bottom: 70, backgroundColor: 'transparent' },
    xAxis: {
      type: 'category', data: names,
      name: '景观名', nameLocation: 'middle', nameGap: 46,
      nameTextStyle: { color: '#6B7280', fontSize: fs, fontFamily },
      axisLine: { lineStyle: { color: '#D1D5DB' } },
      axisTick: { show: false },
      axisLabel: { rotate: 35, color: '#6B7280', fontSize: 11, fontFamily },
      splitLine: { lineStyle: { color: '#F0F2F5', type: 'dashed', width: 1 } },
    },
    yAxis: {
      type: 'value',
      name: '累计字数',
      nameTextStyle: { color: '#6B7280', fontSize: fs, fontFamily },
      axisLine: { lineStyle: { color: '#D1D5DB' } },
      axisTick: { show: false },
      axisLabel: { color: '#6B7280', fontSize: 11, fontFamily },
      splitLine: { lineStyle: { color: '#F0F2F5', type: 'dashed', width: 1 } },
    },
    series: [
      { name: '≤5字/段', type: 'bar', stack: 'total', data: low, itemStyle: { color: palette[0] }, label: { show: true, position: 'inside', color: '#1A1A1A', fontSize: fs - 2, fontFamily } },
      { name: '6-10字/段', type: 'bar', stack: 'total', data: mid, itemStyle: { color: palette[1] }, label: { show: true, position: 'inside', color: '#1A1A1A', fontSize: fs - 2, fontFamily } },
      { name: '>10字/段', type: 'bar', stack: 'total', data: high, itemStyle: { color: palette[2] }, label: { show: true, position: 'insideTop', color: '#1A1A1A', fontSize: fs - 2, fontFamily } },
    ],
  }, true)
  chart.resize()   // 保证 init 后立刻根据真实尺寸重算（抗容器晚布局）
}

// 数据变更时重画
watch(
  () => props.locations,
  async () => {
    await nextTick()  // 等 DOM 更新后再读元素尺寸
    render()
  },
  { deep: true }
)

onMounted(async () => {
  // 等 DOM 和外层 flex 布局稳定后再 init（比 setTimeout 可靠得多）
  await nextTick()
  requestAnimationFrame(() => render())
})

onUnmounted(() => {
  resizeObs?.disconnect()
  resizeObs = null
  chart?.dispose()
  chart = null
})
</script>

<template>
  <div class="stacked-wrapper">
    <!-- 无数据提示：比"一片空白"友好得多 -->
    <div v-if="!hasData" class="empty-hint">
      <div class="eh-icon">📊</div>
      <div class="eh-title">暂无统计数据</div>
      <div class="eh-sub">后端未返回有效的景观分段字数记录，请检查：</div>
      <ul class="eh-list">
        <li>后端 Fastify 服务是否启动（端口 3000）</li>
        <li>excel_data 目录下是否有 雷峰塔 / 金山寺 等分词结果 xlsx 文件</li>
        <li>打开浏览器 DevTools Network 面板，观察 /api/locations 的响应体是否为非空 JSON 数组</li>
      </ul>
    </div>
    <div ref="chartEl" id="stackedBarChart" class="chart-box"></div>
  </div>
</template>

<style scoped>
.stacked-wrapper {
  width: 100%;
  min-height: 520px;
  position: relative;
}
.chart-box {
  width: 100%;
  min-height: 520px;
  height: 520px;
}
.empty-hint {
  position: absolute;
  inset: 0;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  gap: 10px;
  padding: 40px 28px;
  background: rgba(248, 245, 238, 0.4);
  border: 1.5px dashed rgba(180, 170, 155, 0.45);
  border-radius: 14px;
  z-index: 1;
}
.eh-icon { font-size: 44px; }
.eh-title { font-size: 16px; font-weight: 600; color: #5A4A38; }
.eh-sub { font-size: 13px; color: #7A6A58; }
.eh-list {
  margin: 0;
  padding: 0 0 0 18px;
  font-size: 12.5px;
  color: #7A6A58;
  line-height: 1.8;
}
</style>
