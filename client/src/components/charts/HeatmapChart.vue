<script setup lang="ts">
import { ref, watch, onMounted, onUnmounted } from 'vue'
import * as echarts from 'echarts'
import type { LocationData } from '../../types'

const props = defineProps<{ locations: LocationData[] }>()
const chartEl = ref<HTMLDivElement | null>(null)
let chart: echarts.ECharts | null = null

function build(locs: LocationData[]) {
  const sites = locs.map(l => l.name)
  const wordSet = new Set<string>()
  const data: [string, string, number][] = []
  locs.forEach(loc => { (loc.keywords || []).forEach(kw => { wordSet.add(kw.word); data.push([loc.name, kw.word, kw.weight]) }) })
  return { xAxis: sites, yAxis: Array.from(wordSet), data }
}

function render() {
  if (!chartEl.value || !props.locations.length) return
  if (!chart) chart = echarts.init(chartEl.value)
  const chord = build(props.locations)
  if (!chord.data.length) return
  chart.resize()
  const fs = Math.round(parseFloat(getComputedStyle(document.documentElement).fontSize) * 0.65)
  const maxV = Math.max(...chord.data.map(d => d[2]), 1)
  const fontFamily = "'Inter','Noto Sans SC','PingFang SC'"
  const palette = ['#5B9BD5','#7EC4A0','#9AA6D4','#D09EBC','#B4B482','#6FB9C4','#C4AE62','#9BA8B5']
  chart.setOption({
    color: palette,
    tooltip: {
      position: 'top',
      backgroundColor: '#FFFFFF',
      borderColor: '#E5E7EB',
      borderWidth: 1,
      textStyle: { color: '#1A1A1A', fontSize: 12, fontFamily },
      extraCssText: 'box-shadow: none;',
    },
    grid: { left: '15%', bottom: '15%', containLabel: true, backgroundColor: 'transparent' },
    xAxis: {
      type: 'category',
      data: chord.xAxis,
      axisLine: { lineStyle: { color: '#D1D5DB' } },
      axisTick: { show: false },
      axisLabel: { rotate: 45, fontSize: 11, color: '#6B7280', fontFamily },
      splitLine: { lineStyle: { color: '#F0F2F5', type: 'dashed', width: 1 } },
    },
    yAxis: {
      type: 'category',
      data: chord.yAxis,
      axisLine: { lineStyle: { color: '#D1D5DB' } },
      axisTick: { show: false },
      axisLabel: { fontSize: 11, color: '#6B7280', fontFamily },
      splitLine: { lineStyle: { color: '#F0F2F5', type: 'dashed', width: 1 } },
    },
    visualMap: { min: 0, max: maxV, calculable: true, orient: 'horizontal', left: 'center', bottom: '0%',
      textStyle: { color: '#6B7280', fontSize: 11, fontFamily },
      inRange: { color: ['#EBF4FC', '#5B9BD5'] },
    },
    series: [{ type: 'heatmap', data: chord.data, label: { show: false }, emphasis: { itemStyle: { shadowBlur: 0, borderColor: '#6B7280', borderWidth: 1 } } }],
  })
}

watch(() => props.locations, () => { chart?.dispose(); chart = null; setTimeout(render, 150) }, { deep: true })
onMounted(() => setTimeout(render, 200))
onUnmounted(() => chart?.dispose())
</script>
<template><div ref="chartEl" id="heatmapChart" class="chart-box" style="min-height: 500px; width: 100%; height: 500px;"></div></template>
<style scoped>.chart-box { width: 100%; min-height: 500px; }</style>
