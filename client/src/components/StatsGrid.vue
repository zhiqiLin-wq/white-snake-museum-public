<script setup lang="ts">
defineProps<{
  totalMatchSum: number
  totalWordsSum: number
  locationCount: number
}>()
</script>

<template>
  <div class="viz-grid">
    <div class="viz-card">
      <div class="viz-title"><span class="bar-marker bar-sage"></span>总匹配数</div>
      <div class="stat-number">{{ totalMatchSum || '--' }}</div>
    </div>
    <div class="viz-card">
      <div class="viz-title"><span class="bar-marker bar-plum"></span>总文本字数</div>
      <div class="stat-number">{{ totalWordsSum ? totalWordsSum.toLocaleString() : '--' }}</div>
    </div>
    <div class="viz-card">
      <div class="viz-title"><span class="dot-marker dot-gold"></span>地点数量</div>
      <div class="stat-number">{{ locationCount || '--' }}</div>
    </div>
  </div>
</template>

<style scoped>
.viz-grid {
  display: flex;
  gap: var(--space-3);
  padding: var(--space-3) var(--space-5);
  flex-wrap: wrap;
}
.viz-card {
  flex: 1;
  background: var(--color-bg-secondary);
  border-radius: var(--radius);
  padding: var(--space-2);
  text-align: center;
  border: 1px solid var(--color-border);
  transition:
    transform var(--dur) var(--ease),
    box-shadow var(--dur) var(--ease),
    border-color var(--dur) var(--ease);
}
.viz-card:hover {
  transform: translateY(-1px);
  box-shadow: var(--shadow-sm);
  border-color: var(--color-border-dark);
}
.viz-title {
  font-size: 0.62rem;
  color: var(--color-text-tertiary);
}
.bar-marker,
.dot-marker {
  display: inline-block;
  vertical-align: middle;
  margin-right: 4px;
}
.stat-number {
  font-size: 1.3rem;
  font-weight: 700;
  color: var(--color-text-primary);
  margin-top: 2px;
}
</style>
