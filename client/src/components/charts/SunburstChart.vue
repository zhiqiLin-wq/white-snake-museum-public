<script setup lang="ts">
import { ref, watch, onMounted, onUnmounted } from 'vue'
import * as echarts from 'echarts'
import type { LocationData } from '../../types'

const props = defineProps<{ locations: LocationData[] }>()
const chartEl = ref<HTMLDivElement | null>(null)
let chart: echarts.ECharts | null = null

const palette = ['#5B9BD5','#7EC4A0','#9AA6D4','#D09EBC','#B4B482','#6FB9C4','#C4AE62','#9BA8B5']
const fontFamily = "'Inter','Noto Sans SC','PingFang SC'"

function build(locs: LocationData[]) {
  const siteNames = new Set(locs.map(l => l.name))
  let colorIdx = 0
  return locs.filter(loc => loc.keywords?.length).map(loc => {
    const siteColor = palette[colorIdx % palette.length]
    colorIdx++
    return {
      name: loc.name,
      itemStyle: { color: siteColor },
      children: loc.keywords.map(kw => ({
        name: siteNames.has(kw.word) ? kw.word + '·' : kw.word,
        value: kw.weight,
        itemStyle: { color: palette[colorIdx % palette.length] + 'CC' },
      })),
    }
  })
}

function render() {
  if (!chartEl.value || !props.locations.length) return
  if (!chart) chart = echarts.init(chartEl.value)
  const data = build(props.locations)
  if (!data.length) return
  const fs = Math.round(parseFloat(getComputedStyle(document.documentElement).fontSize) * 0.65)
  chart.setOption({
    color: palette,
    tooltip: {
      trigger: 'item',
      backgroundColor: '#FFFFFF',
      borderColor: '#E5E7EB',
      borderWidth: 1,
      textStyle: { color: '#1A1A1A', fontSize: 12, fontFamily },
      extraCssText: 'box-shadow: none;',
    },
    series: [{ type: 'sunburst', data, radius: [0, '90%'], label: { rotate: 'radial', fontSize: fs, color: '#1A1A1A', fontFamily }, itemStyle: { borderRadius: 4, borderWidth: 1, borderColor: '#FFFFFF' } }],
  })
}

watch(() => props.locations, () => { chart?.dispose(); chart = null; setTimeout(render, 150) }, { deep: true })
onMounted(() => setTimeout(render, 200))
onUnmounted(() => chart?.dispose())
</script>
<template><div ref="chartEl" id="sunburstChart" class="chart-box" style="min-height: 500px; width: 100%;"></div></template>
<style scoped>.chart-box { width: 100%; min-height: 500px; }</style>
