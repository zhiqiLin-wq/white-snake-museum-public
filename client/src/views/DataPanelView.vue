<script setup lang="ts">
import { computed } from 'vue'
import { useWorkspaceStore } from '@/stores/workspace'
import { useAgentStore } from '@/stores/agent'
import type { StatsDataEvent } from '@/types/sse-events'

const workspaceStore = useWorkspaceStore()
const agentStore = useAgentStore()

const tabData = computed(() => {
  const tab = workspaceStore.activeTab
  return tab?.data as unknown as StatsDataEvent | undefined
})

const title = computed(() => tabData.value?.title || '')
const columns = computed(() => tabData.value?.columns || [])
const rows = computed(() => tabData.value?.rows || [])
const aiSummary = computed(() => tabData.value?.aiSummary || '')

function formatCell(value: unknown, type: string): string {
  if (value === null || value === undefined) return '-'
  switch (type) {
    case 'integer':
      return Number(value).toLocaleString()
    case 'percent':
      return `${Math.round(Number(value) * 100)}%`
    case 'float':
      return Number(value).toFixed(2)
    default:
      return String(value)
  }
}

function getCellAlign(type: string): string {
  return type === 'string' ? 'left' : 'right'
}

function askAbout(row: Record<string, unknown>) {
  agentStore.addContextTag({
    id: `ctx_${Date.now()}`,
    label: `${title.value} 数据点`,
    data: { type: 'data_point', label: title.value, payload: row },
    sourceType: 'data_point',
  })
}

function exportCsv() {
  if (rows.value.length === 0) return
  const bom = '﻿'
  const headers = columns.value.map(c => c.label).join(',')
  const body = rows.value.map(row =>
    columns.value.map(c => {
      const val = row[c.key]
      return val === null || val === undefined ? '' : `"${String(val).replace(/"/g, '""')}"`
    }).join(',')
  ).join('\n')
  const csv = bom + headers + '\n' + body
  const blob = new Blob([csv], { type: 'text/csv;charset=utf-8' })
  const url = URL.createObjectURL(blob)
  const a = document.createElement('a')
  a.href = url
  a.download = `${title.value}_${new Date().toISOString().slice(0, 10)}.csv`
  a.click()
  URL.revokeObjectURL(url)
}
</script>

<template>
  <div class="data-panel-view">
    <div class="dp-header">
      <h2 class="dp-title">{{ title }}</h2>
      <button class="dp-export-btn" @click="exportCsv">导出 CSV</button>
    </div>

    <div v-if="rows.length === 0" class="dp-empty">
      暂无数据
    </div>

    <div v-else class="dp-table-container">
      <table class="dp-table">
        <thead>
          <tr>
            <th v-for="col in columns" :key="col.key" :style="({ textAlign: getCellAlign(col.type) } as Record<string, string>)">
              {{ col.label }}
            </th>
            <th class="dp-action-col">操作</th>
          </tr>
        </thead>
        <tbody>
          <tr v-for="(row, i) in rows" :key="i" class="dp-row">
            <td
              v-for="col in columns"
              :key="col.key"
              :style="({ textAlign: getCellAlign(col.type) } as Record<string, string>)"
            >
              {{ formatCell(row[col.key], col.type) }}
            </td>
            <td class="dp-action-col">
              <button class="dp-ask-btn" @click="askAbout(row)">追问</button>
            </td>
          </tr>
        </tbody>
      </table>
    </div>

    <div v-if="aiSummary" class="dp-summary">
      <span class="dp-summary-label">AI </span>
      <p>{{ aiSummary }}</p>
    </div>
  </div>
</template>

<style scoped>
.data-panel-view {
  padding: var(--space-5);
  height: 100%;
  overflow-y: auto;
}

.dp-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin-bottom: var(--space-4);
}

.dp-title {
  font-family: var(--font-sans);
  font-size: 1.05rem;
  font-weight: 600;
  color: var(--color-text-primary);
  margin: 0;
}

.dp-export-btn {
  padding: var(--space-1) var(--space-3);
  font-size: 0.75rem;
  background: var(--color-bg-primary);
  color: var(--color-text-primary);
  border: 1px solid var(--color-border);
  border-radius: var(--radius);
  cursor: pointer;
  font-family: var(--font-sans);
  font-weight: 500;
  transition:
    background-color var(--dur-fast) var(--ease),
    border-color var(--dur-fast) var(--ease),
    transform var(--dur) var(--ease),
    box-shadow var(--dur) var(--ease);
}

.dp-export-btn:hover {
  background: var(--color-bg-hover);
  border-color: var(--color-border-dark);
  transform: translateY(-1px);
  box-shadow: var(--shadow-sm);
}

.dp-export-btn:active {
  transform: translateY(0);
  box-shadow: none;
}

.dp-empty {
  text-align: center;
  padding: var(--space-8);
  color: var(--color-text-tertiary);
  font-size: 0.85rem;
}

.dp-table-container {
  overflow-x: auto;
  margin-bottom: var(--space-4);
  background: var(--color-bg-primary);
  border: 1px solid var(--color-border);
  border-radius: var(--radius-lg);
}

.dp-table {
  width: 100%;
  border-collapse: separate;
  border-spacing: 0;
  font-size: 0.8rem;
}

.dp-table th {
  padding: var(--space-2) var(--space-3);
  background: var(--color-bg-secondary);
  color: var(--color-text-secondary);
  font-weight: 600;
  border-bottom: 1px solid var(--color-border);
  position: sticky;
  top: 0;
  text-align: inherit;
}

.dp-table th:first-child {
  border-top-left-radius: var(--radius-lg);
}

.dp-table th:last-child {
  border-top-right-radius: var(--radius-lg);
}

.dp-table td {
  padding: var(--space-2) var(--space-3);
  border-bottom: 1px solid var(--color-border-light);
  color: var(--color-text-primary);
}

.dp-table tr:last-child td {
  border-bottom: none;
}

.dp-row {
  transition: background-color var(--dur-fast) var(--ease);
}

.dp-row:nth-child(even) {
  background: var(--color-bg-secondary);
}

.dp-row:hover {
  background: var(--color-bg-hover);
}

.dp-action-col {
  width: 60px;
  text-align: center;
}

.dp-ask-btn {
  padding: 1px var(--space-2);
  font-size: 0.68rem;
  background: var(--color-bg-primary);
  color: var(--color-accent);
  border: 1px solid var(--color-border);
  border-radius: var(--radius-sm);
  cursor: pointer;
  font-family: var(--font-sans);
  font-weight: 500;
  transition:
    background-color var(--dur-fast) var(--ease),
    border-color var(--dur-fast) var(--ease),
    color var(--dur-fast) var(--ease);
}

.dp-ask-btn:hover {
  background: var(--color-accent-light);
  border-color: var(--color-accent);
  color: var(--color-accent);
}

.dp-summary {
  padding: var(--space-4);
  background: var(--color-bg-secondary);
  border: 1px solid var(--color-border);
  border-radius: var(--radius-lg);
  font-size: 0.82rem;
  color: var(--color-text-primary);
  line-height: 1.6;
}

.dp-summary-label {
  display: block;
  font-weight: 600;
  color: var(--color-text-secondary);
  margin-bottom: var(--space-2);
}
</style>
