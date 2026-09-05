<script setup lang="ts">
import { ref, watch, onMounted, onUnmounted } from 'vue'
import * as echarts from 'echarts'
import type { LocationData } from '../../types'

const props = defineProps<{ locations: LocationData[] }>()
const chartEl = ref<HTMLDivElement | null>(null)
let chart: echarts.ECharts | null = null

function build(locs: LocationData[]) {
  const siteNames = new Set(locs.map(l => l.name))
  const nameSet = new Set<string>()
  const nodes: { name: string; category: number; symbolSize: number }[] = []
  const links: { source: string; target: string; value: number }[] = []
  const baseSize = parseFloat(getComputedStyle(document.documentElement).fontSize)
  locs.forEach(loc => {
    if (!nameSet.has(loc.name)) { nodes.push({ name: loc.name, category: 0, symbolSize: Math.round(baseSize * 2.5) }); nameSet.add(loc.name) }
    (loc.keywords || []).forEach(kw => {
      const uw = siteNames.has(kw.word) ? kw.word + '·' : kw.word
      if (!nameSet.has(uw)) { nodes.push({ name: uw, category: 1, symbolSize: Math.round(baseSize * 1.2) }); nameSet.add(uw) }
      links.push({ source: loc.name, target: uw, value: kw.weight })
    })
  })
  return { nodes, links }
}

function render() {
  if (!chartEl.value || !props.locations.length) return
  if (!chart) chart = echarts.init(chartEl.value)
  const graph = build(props.locations)
  if (!graph.nodes.length) return
  const fs = Math.round(parseFloat(getComputedStyle(document.documentElement).fontSize) * 0.65)
  const fontFamily = "'Inter','Noto Sans SC','PingFang SC'"
  const palette = ['#5B9BD5','#7EC4A0','#9AA6D4','#D09EBC','#B4B482','#6FB9C4','#C4AE62','#9BA8B5']
  chart.setOption({
    color: palette,
    tooltip: {
      show: true,
      backgroundColor: '#FFFFFF',
      borderColor: '#E5E7EB',
      borderWidth: 1,
      textStyle: { color: '#1A1A1A', fontSize: 12, fontFamily },
      extraCssText: 'box-shadow: none;',
    },
    legend: {
      data: ['', ''],
      itemWidth: 12,
      itemHeight: 8,
      itemGap: 16,
      textStyle: { color: '#6B7280', fontSize: 11, fontFamily },
    },
    series: [{
      type: 'graph', layout: 'force', force: { repulsion: 200, edgeLength: 100 }, roam: true,
      data: graph.nodes, links: graph.links,
      categories: [{ name: '', itemStyle: { color: palette[0] } }, { name: '', itemStyle: { color: palette[1] } }],
      label: { show: true, fontSize: fs, color: '#1A1A1A', fontFamily },
      edgeSymbol: ['none', 'none'], lineStyle: { color: '#D8DDE3', opacity: 0.8 },
    }],
  })
}

watch(() => props.locations, () => { chart?.dispose(); chart = null; setTimeout(render, 150) }, { deep: true })
onMounted(() => setTimeout(render, 200))
onUnmounted(() => chart?.dispose())
</script>
<template><div ref="chartEl" id="graphChart" class="chart-box" style="min-height: 500px; width: 100%;"></div></template>
<style scoped>.chart-box { width: 100%; min-height: 500px; }</style>
