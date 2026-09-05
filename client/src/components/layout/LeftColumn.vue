<script setup lang="ts">
import { onMounted } from 'vue'
import { useSourceTreeStore } from '@/stores/sourceTree'
import { useWorkspaceStore } from '@/stores/workspace'
import SourceTree from '@/components/sourcetree/SourceTree.vue'

const sourceTreeStore = useSourceTreeStore()
const workspaceStore = useWorkspaceStore()

function openMap() {
  workspaceStore.openTab('map', '地图')
}

onMounted(() => {
  sourceTreeStore.fetchChapters()
})
</script>

<template>
  <div class="left-column">
    <div class="lc-header">
      <h1 class="lc-logo">白蛇传 文脉全息</h1>
    </div>

    <div class="lc-sourcetree">
      <div class="lc-section-label">文献目录</div>
      <SourceTree />
    </div>

    <!-- 可视化图表入口 -->
    <div class="lc-charts-section">
      <div class="lc-section-label">可视化图表</div>
      <div class="lc-chart-icons">
        <button
          class="lc-chart-btn"
          title="堆叠柱状图 - 朝代/段落对比"
          @click="workspaceStore.openTab('chart-stacked-bar', '堆叠柱状图')"
        >柱状图</button>
        <button
          class="lc-chart-btn"
          title="桑基图 - 实体流向"
          @click="workspaceStore.openTab('chart-sankey', '桑基图')"
        >桑基图</button>
        <button
          class="lc-chart-btn"
          title="旭日图 - 层级分布"
          @click="workspaceStore.openTab('chart-sunburst', '旭日图')"
        >旭日图</button>
        <button
          class="lc-chart-btn"
          title="热力图 - 朝代x实体"
          @click="workspaceStore.openTab('chart-heatmap', '热力图')"
        >热力图</button>
        <button
          class="lc-chart-btn"
          title="力导向图 - 实体关系网络"
          @click="workspaceStore.openTab('chart-graph', '力导向图')"
        >关系图</button>
      </div>
    </div>

    <div class="lc-footer">
      <button class="lc-map-btn" title="地图视图" @click="openMap"> 地图 </button>
      <div class="lc-stats">
        <span>{{ sourceTreeStore.chapterCount }} 章</span>
      </div>
    </div>
  </div>
</template>

<style scoped>
.left-column {
  display: flex;
  flex-direction: column;
  height: 100%;
  overflow: hidden;
  background: var(--color-bg-primary, #FFFFFF);
}

.lc-header {
  padding: 14px 16px;
  border-bottom: 1px solid var(--color-border, #E5E7EB);
}

.lc-logo {
  font-family: var(--font-sans);
  font-size: 1.1rem;
  color: var(--color-text-primary, #1A1A1A);
  margin: 0;
  font-weight: 600;
  letter-spacing: 0.05em;
}

.lc-sourcetree {
  flex: 1;
  overflow-y: auto;
  padding: 8px 0;
}

.lc-section-label {
  font-size: 0.7rem;
  text-transform: uppercase;
  color: var(--color-text-tertiary, #6B7280);
  padding: 6px 16px;
  letter-spacing: 0.08em;
}

.lc-footer {
  padding: 10px 16px;
  border-top: 1px solid var(--color-border, #E5E7EB);
  display: flex;
  align-items: center;
  justify-content: space-between;
}

.lc-map-btn {
  background: var(--color-bg-tertiary, #F0F2F5);
  border: 1px solid var(--color-border, #E5E7EB);
  color: var(--color-text-primary, #1A1A1A);
  padding: 5px 12px;
  border-radius: var(--radius-sm, 4px);
  font-size: 0.78rem;
  cursor: pointer;
  transition: background var(--dur-fast, 120ms) var(--ease, cubic-bezier(.22,.61,.36,1)),
              border-color var(--dur-fast, 120ms) var(--ease, cubic-bezier(.22,.61,.36,1)),
              color var(--dur-fast, 120ms) var(--ease, cubic-bezier(.22,.61,.36,1));
}

.lc-map-btn:hover {
  background: var(--color-accent, #5B9BD5);
  border-color: var(--color-accent, #5B9BD5);
  color: #FFFFFF;
}

.lc-stats {
  font-size: 0.72rem;
  color: var(--color-text-tertiary, #6B7280);
}

/* A-09:  */
.lc-charts-section {
  padding: 6px 0;
  border-top: 1px solid var(--color-border, #E5E7EB);
}

.lc-chart-icons {
  display: flex;
  flex-wrap: wrap;
  gap: 4px;
  padding: 4px 10px;
}

.lc-chart-btn {
  padding: 3px 8px;
  font-size: 0.65rem;
  background: var(--color-bg-tertiary, #F0F2F5);
  color: var(--color-text-secondary, #4B5563);
  border: 1px solid var(--color-border, #E5E7EB);
  border-radius: var(--radius-sm, 4px);
  cursor: pointer;
  transition: border-color var(--dur-fast, 120ms) var(--ease, cubic-bezier(.22,.61,.36,1)),
              color var(--dur-fast, 120ms) var(--ease, cubic-bezier(.22,.61,.36,1)),
              background var(--dur-fast, 120ms) var(--ease, cubic-bezier(.22,.61,.36,1));
}

.lc-chart-btn:hover {
  border-color: var(--color-accent, #5B9BD5);
  color: var(--color-accent, #5B9BD5);
  background: var(--color-accent-light, #EBF4FC);
}
</style>
