<script setup lang="ts">
import { computed, defineAsyncComponent, ref } from 'vue'
import { useWorkspaceStore } from '@/stores/workspace'
import TabBar from '@/components/ui/TabBar.vue'
import SplitDivider from '@/components/layout/SplitDivider.vue'
import TextReaderView from '@/views/TextReaderView.vue'

const workspaceStore = useWorkspaceStore()

// A11: Compare modal selector visibility
const compareModalVisible = ref(false)
const CompareModalSelector = defineAsyncComponent(() => import('@/components/compare/CompareModalSelector.vue'))

function handleOpenCompareSelector() {
  compareModalVisible.value = true
}

const activeViewComponent = computed(() => {
  const tab = workspaceStore.activeTab
  if (!tab) return null

  const viewComponentMap: Record<string, ReturnType<typeof defineAsyncComponent>> = {
    'text-reader': defineAsyncComponent(() => import('@/views/TextReaderView.vue')),
    'compare-view': defineAsyncComponent(() => import('@/views/CompareView.vue')),
    'data-panel': defineAsyncComponent(() => import('@/views/DataPanelView.vue')),
    'search-results': defineAsyncComponent(() => import('@/views/SearchResultsView.vue')),
    'evolution-workspace': defineAsyncComponent(() => import('@/views/EvolutionWorkspaceView.vue')),
    'split-view': defineAsyncComponent(() => import('@/views/SplitView.vue')),
    'map': defineAsyncComponent(() => import('@/views/MapView.vue')),
    'chart-stacked-bar': defineAsyncComponent(() => import('@/views/ChartContainerView.vue')),
    'chart-sankey': defineAsyncComponent(() => import('@/views/ChartContainerView.vue')),
    'chart-sunburst': defineAsyncComponent(() => import('@/views/ChartContainerView.vue')),
    'chart-chord': defineAsyncComponent(() => import('@/views/ChartContainerView.vue')),
    'chart-heatmap': defineAsyncComponent(() => import('@/views/ChartContainerView.vue')),
    'chart-graph': defineAsyncComponent(() => import('@/views/ChartContainerView.vue')),
    'dynasty-compare': defineAsyncComponent(() => import('@/views/DynastyCompareView.vue')),
    'annotation-index': defineAsyncComponent(() => import('@/views/AnnotationIndexView.vue')),
    'annotation-stats': defineAsyncComponent(() => import('@/views/AnnotationStatsView.vue')),
    'report-library': defineAsyncComponent(() => import('@/views/ReportLibraryView.vue')),
  }

  return viewComponentMap[tab.type] || null
})

const leftTab = computed(() =>
  workspaceStore.tabs.find(t => t.id === workspaceStore.split.leftTabId) || null
)
const rightTab = computed(() =>
  workspaceStore.tabs.find(t => t.id === workspaceStore.split.rightTabId) || null
)

function handleSelectTab(tabId: string) {
  workspaceStore.setActiveTab(tabId)
}

function handleCloseTab(tabId: string) {
  workspaceStore.closeTab(tabId)
}

function handleSplitDrop(tabId: string, side: 'left' | 'right') {
  const activeId = workspaceStore.activeTabId
  if (!activeId) return
  if (side === 'left') {
    workspaceStore.enterSplitMode(tabId, activeId)
  } else {
    workspaceStore.enterSplitMode(activeId, tabId)
  }
}

function handleDragBack(tabId: string) {
  workspaceStore.exitSplitMode()
}

function handleReorderTabs(fromIndex: number, toIndex: number) {
  workspaceStore.reorderTabs(fromIndex, toIndex)
}
</script>

<template>
  <div class="main-workspace">
    <TabBar
      :tabs="workspaceStore.tabs"
      :active-tab-id="workspaceStore.activeTabId"
      @select-tab="handleSelectTab"
      @close-tab="handleCloseTab"
      @split-drop="handleSplitDrop"
      @drag-back="handleDragBack"
      @reorder-tabs="handleReorderTabs"
      @open-compare-selector="handleOpenCompareSelector"
    />

    <!-- A11: Compare modal selector -->
    <CompareModalSelector
      v-if="compareModalVisible"
      :visible="compareModalVisible"
      @close="compareModalVisible = false"
    />

    <!-- Split layout -->
    <div v-if="workspaceStore.split.enabled && (leftTab || rightTab)" class="workspace-split">
      <div class="split-panel" :style="{ flex: `0 0 ${workspaceStore.split.splitRatio * 100}%` }">
        <TextReaderView
          v-if="leftTab"
          :tab-id="leftTab.id"
          split-side="left"
          :key="leftTab.id + '-left'"
        />
      </div>
      <SplitDivider @ratio-change="(r) => workspaceStore.setSplitRatio(r)" />
      <div class="split-panel" :style="{ flex: `0 0 ${(1 - workspaceStore.split.splitRatio) * 100}%` }">
        <TextReaderView
          v-if="rightTab"
          :tab-id="rightTab.id"
          split-side="right"
          :key="rightTab.id + '-right'"
        />
      </div>
    </div>

    <!-- Single panel layout -->
    <div v-else class="workspace-content">
      <template v-if="activeViewComponent">
        <component
          :is="activeViewComponent"
          :key="workspaceStore.activeTab?.id"
          :tab-id="workspaceStore.activeTab?.id"
        />
      </template>
      <div v-else class="workspace-empty">
        <p> 点击左侧文献目录开始阅读，或向 AI 助手提问 </p>
      </div>
    </div>
  </div>
</template>

<style scoped>
.main-workspace {
  display: flex;
  flex-direction: column;
  height: 100%;
  overflow: hidden;
  background: var(--color-bg-primary, #FFFFFF);
}

.workspace-content {
  flex: 1;
  overflow-y: auto;
  position: relative;
}

.workspace-split {
  display: flex;
  flex: 1;
  overflow: hidden;
}

.split-panel {
  min-width: 0;
  overflow: hidden;
  display: flex;
  flex-direction: column;
}

.workspace-empty {
  display: flex;
  align-items: center;
  justify-content: center;
  height: 100%;
  color: var(--color-text-tertiary, #6B7280);
  font-family: var(--font-sans);
  font-size: 0.9rem;
  background: var(--color-bg-primary, #FFFFFF);
}
</style>
