<script setup lang="ts">
import { computed } from 'vue'
import type { LocationData } from '../types'
import { escapeHtml } from '../utils/helpers'

const props = defineProps<{ selectedData: LocationData | null; selectedLocation: string | null }>()
defineEmits<{ 'literature-trace': [name: string] }>()

const truncated = computed(() =>
  props.selectedData?.records.map(r => ({
    ...r,
    摘要: (r.摘要 || '').length > 140 ? r.摘要.substring(0, 140) + '...' : (r.摘要 || ''),
  })) ?? []
)
</script>

<template>
  <div class="detail-panel">
    <div class="detail-header">
      <div class="detail-title">{{ selectedData ? `${selectedData.name} · ` : '' }}</div>
      <div v-if="selectedData" class="stat-badge">
        <span class="stat-label">匹配</span> {{ selectedData.totalMatch }} &nbsp;<span class="stat-sep">|</span>&nbsp; <span class="stat-label">字数</span> {{ selectedData.totalWords }} &nbsp;<span class="stat-sep">|</span>&nbsp; <span class="stat-label">记录</span> {{ selectedData.records.length }}
      </div>
      <button v-if="selectedData" class="lit-trace-btn" @click="$emit('literature-trace', selectedData.name)">
        文献溯源 · {{ selectedData.name }}
      </button>
    </div>
    <div class="data-table-wrapper">
      <div v-if="!selectedData" class="relation-note">请在地图上点击地点查看详情</div>
      <table v-else class="data-table">
        <thead><tr><th>切词数</th><th>匹配词数</th><th>文本字数</th><th>摘要</th></tr></thead>
        <tbody>
          <tr v-for="(r, i) in truncated" :key="i">
            <td>{{ r.切词数 || '-' }}</td><td>{{ r.匹配词数 || 1 }}</td><td>{{ r.文本字数 || '-' }}</td>
            <td class="text-preview" v-html="escapeHtml(r.摘要)"></td>
          </tr>
        </tbody>
      </table>
    </div>
  </div>
</template>

<style scoped>
.detail-panel {
  margin: var(--space-2) var(--space-4) var(--space-4);
  background: var(--color-bg-primary);
  border-radius: var(--radius);
  border: 1px solid var(--color-border);
  box-shadow: var(--shadow-sm);
  display: flex;
  flex-direction: column;
  overflow: hidden;
  flex: 1;
  min-height: 14rem;
}
.detail-header {
  padding: var(--space-3) var(--space-4);
  background: var(--color-bg-secondary);
  border-bottom: 1px solid var(--color-border-light);
}
.detail-title {
  font-size: 1rem;
  font-weight: 700;
  color: var(--color-text-primary);
}
.stat-badge {
  display: flex;
  gap: var(--space-3);
  font-size: 0.65rem;
  color: var(--color-text-tertiary);
  margin-top: var(--space-1);
}
.stat-label {
  display: inline-block;
  font-size: 0.62rem;
  font-weight: 600;
  color: var(--color-accent);
  border-left: 2px solid var(--color-accent);
  padding-left: var(--space-2);
}
.stat-sep {
  color: var(--color-border-dark);
}
.lit-trace-btn {
  display: inline-block;
  margin-top: var(--space-2);
  padding: 4px 12px;
  border-radius: var(--radius);
  border: 1px solid var(--color-accent);
  background: var(--color-accent-light);
  color: var(--color-accent);
  font-size: 0.68rem;
  font-weight: 500;
  cursor: pointer;
  transition:
    background var(--dur) var(--ease),
    color var(--dur) var(--ease),
    transform var(--dur) var(--ease),
    box-shadow var(--dur) var(--ease);
}
.lit-trace-btn:hover {
  background: var(--color-accent);
  color: #ffffff;
  transform: translateY(-1px);
  box-shadow: var(--shadow-sm);
}
.data-table-wrapper {
  padding: var(--space-2) var(--space-4) var(--space-4);
  overflow: auto;
  flex: 1;
  max-height: 16rem;
}
.data-table {
  width: 100%;
  border-collapse: collapse;
  font-size: 0.68rem;
  background: var(--color-bg-primary);
  border-radius: var(--radius);
  overflow: hidden;
}
.data-table th {
  background: var(--color-bg-tertiary);
  color: var(--color-text-primary);
  padding: 6px 4px;
  position: sticky;
  top: 0;
  font-weight: 600;
  font-size: 0.64rem;
  text-align: left;
  border-bottom: 1px solid var(--color-border);
}
.data-table td {
  border-bottom: 1px solid var(--color-border-light);
  padding: 5px 4px;
  color: var(--color-text-secondary);
}
.text-preview {
  max-width: 12rem;
  white-space: normal;
  line-height: 1.45;
}
.relation-note {
  font-size: 0.62rem;
  margin: var(--space-2) var(--space-4);
  padding: var(--space-2) var(--space-3);
  background: var(--color-bg-secondary);
  border: 1px solid var(--color-border-light);
  border-radius: var(--radius);
  color: var(--color-text-tertiary);
  text-align: center;
}
</style>
