<script setup lang="ts">
import { ref, watch, onMounted, onUnmounted } from 'vue'
import * as echarts from 'echarts'
import type { LocationData } from '../../types'

const props = defineProps<{ locations: LocationData[] }>()
const chartEl = ref<HTMLDivElement | null>(null)
const chartContainer = ref<HTMLElement | null>(null)
let chart: echarts.ECharts | null = null

const palette = ['#5B9BD5','#7EC4A0','#9AA6D4','#D09EBC','#B4B482','#6FB9C4','#C4AE62','#9BA8B5']
const fontFamily = "'Inter','Noto Sans SC','PingFang SC'"

function build(locs: LocationData[]) {
  const siteNames = new Set(locs.map(l => l.name))
  const nodes: { name: string; itemStyle?: { color: string } }[] = []
  const links: { source: string; target: string; value: number; lineStyle?: { color: string; opacity: number } }[] = []
  const usedWords = new Set<string>()
  let colorIdx = 0
  locs.forEach(loc => {
    if (!loc.keywords?.length) return
    const siteColor = palette[colorIdx % palette.length]
    colorIdx++
    nodes.push({ name: loc.name, itemStyle: { color: siteColor } })
    loc.keywords.forEach(kw => {
      const uw = siteNames.has(kw.word) ? kw.word + '.' : kw.word
      if (!usedWords.has(uw)) { nodes.push({ name: uw, itemStyle: { color: palette[colorIdx % palette.length] } }); colorIdx++; usedWords.add(uw) }
      links.push({ source: loc.name, target: uw, value: kw.weight, lineStyle: { color: siteColor, opacity: 0.15 } })
    })
  })
  const nodeMap = new Map<string, typeof nodes[0]>(); nodes.forEach(n => nodeMap.set(n.name, n))
  return { nodes: Array.from(nodeMap.values()), links }
}

function computeChartHeight(nodeCount: number): number {
  const minHeight = 500
  const perNodeHeight = 30
  return Math.max(minHeight, nodeCount * perNodeHeight)
}

function render() {
  if (!chartEl.value || !props.locations.length) return
  if (!chart) chart = echarts.init(chartEl.value)
  const { nodes, links } = build(props.locations)
  if (!nodes.length) return
  const fs = Math.round(parseFloat(getComputedStyle(document.documentElement).fontSize) * 0.65)
  if (chartContainer.value) {
    const height = computeChartHeight(nodes.length)
    chartContainer.value.style.height = height + 'px'
  }
  chart.resize()
  chart.setOption({
    color: palette,
    tooltip: {
      trigger: 'item',
      triggerOn: 'mousemove',
      backgroundColor: '#FFFFFF',
      borderColor: '#E5E7EB',
      borderWidth: 1,
      textStyle: { color: '#1A1A1A', fontSize: 12, fontFamily },
      extraCssText: 'box-shadow: none;',
    },
    series: [{ type: 'sankey', layout: 'none', emphasis: { focus: 'adjacency' }, nodeAlign: 'left',
      data: nodes, links,
      label: { show: true, fontSize: fs, color: '#1A1A1A', fontFamily },
      lineStyle: { curveness: 0.5, opacity: 0.15 },
    }],
  })
}

watch(() => props.locations, () => { chart?.dispose(); chart = null; setTimeout(render, 150) }, { deep: true })
onMounted(() => setTimeout(render, 200))
onUnmounted(() => chart?.dispose())
</script>
<template><div ref="chartContainer" class="chart-box" style="min-height: 500px; width: 100%; height: 500px;"><div ref="chartEl" id="sankeyChart" style="width: 100%; height: 100%; min-height: 500px;"></div></div></template>
<style scoped>.chart-box { width: 100%; min-height: 500px; }</style>
