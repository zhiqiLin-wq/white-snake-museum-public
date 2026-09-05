<script setup lang="ts">
import { ref, onMounted, onUnmounted, computed } from 'vue'
import LeftColumn from '@/components/layout/LeftColumn.vue'
import MainWorkspace from '@/components/layout/MainWorkspace.vue'
import RightColumn from '@/components/layout/RightColumn.vue'
import { SSEEventBus, registerGlobalRoutes } from '@/services/sseEventBus'
import { useAgentStore } from '@/stores/agent'
import { useAuthStore } from '@/stores/auth'
import { useEvolutionStore } from '@/stores/evolution'
import { useAnnotationStore } from '@/stores/annotations'
import { useSourceTreeStore } from '@/stores/sourceTree'
import { useWorkspaceStore } from '@/stores/workspace'
import { useMapStore } from '@/stores/map'
import { useDynastyCompareStore } from '@/stores/dynastyCompare'
import DataMigrationPrompt from '@/components/DataMigrationPrompt.vue'
import { apiFetch } from '@/services/api'

const SIDEBAR_MIN_WIDTH = 160
const SIDEBAR_MAX_WIDTH = 400
const SIDEBAR_DEFAULT_WIDTH = 220
const SIDEBAR_STORAGE_KEY = 'museum_sidebar_width'

function loadSidebarWidth(): number {
  try {
    const raw = localStorage.getItem(SIDEBAR_STORAGE_KEY)
    if (raw) {
      const val = parseInt(raw, 10)
      if (!isNaN(val) && val >= SIDEBAR_MIN_WIDTH && val <= SIDEBAR_MAX_WIDTH) {
        return val
      }
    }
  } catch { /* localStorage not available */ }
  return SIDEBAR_DEFAULT_WIDTH
}

function saveSidebarWidth(width: number): void {
  try {
    localStorage.setItem(SIDEBAR_STORAGE_KEY, String(width))
  } catch { /* localStorage not available */ }
}

// Chat (agent) panel draggable width
const CHAT_MIN_WIDTH = 280
const CHAT_MAX_WIDTH = 560
const CHAT_DEFAULT_WIDTH = 360
const CHAT_STORAGE_KEY = 'museum_chat_width'

function loadChatWidth(): number {
  try {
    const raw = localStorage.getItem(CHAT_STORAGE_KEY)
    if (raw) {
      const val = parseInt(raw, 10)
      if (!isNaN(val) && val >= CHAT_MIN_WIDTH && val <= CHAT_MAX_WIDTH) {
        return val
      }
    }
  } catch { /* localStorage not available */ }
  return CHAT_DEFAULT_WIDTH
}

function saveChatWidth(width: number): void {
  try {
    localStorage.setItem(CHAT_STORAGE_KEY, String(width))
  } catch { /* localStorage not available */ }
}

const viewportWidth = ref(window.innerWidth)
const rightPanelOverlay = ref(false)
const workspaceStore = useWorkspaceStore()

// Sidebar draggable width
const sidebarWidth = ref(loadSidebarWidth())
const isDraggingSidebar = ref(false)

function onSidebarDragStart(e: MouseEvent) {
  isDraggingSidebar.value = true
  e.preventDefault()

  const startX = e.clientX
  const startWidth = sidebarWidth.value

  function onMouseMove(me: MouseEvent) {
    if (!isDraggingSidebar.value) return
    const deltaX = me.clientX - startX
    const newWidth = Math.max(SIDEBAR_MIN_WIDTH, Math.min(SIDEBAR_MAX_WIDTH, startWidth + deltaX))
    sidebarWidth.value = newWidth
  }

  function onMouseUp() {
    isDraggingSidebar.value = false
    saveSidebarWidth(sidebarWidth.value)
    document.removeEventListener('mousemove', onMouseMove)
    document.removeEventListener('mouseup', onMouseUp)
  }

  document.addEventListener('mousemove', onMouseMove)
  document.addEventListener('mouseup', onMouseUp)
}

function onSidebarDblClick() {
  sidebarWidth.value = SIDEBAR_DEFAULT_WIDTH
  saveSidebarWidth(SIDEBAR_DEFAULT_WIDTH)
}

// Chat (agent) panel drag state
const chatWidth = ref(loadChatWidth())
const isDraggingChat = ref(false)

function onChatDragStart(e: MouseEvent) {
  isDraggingChat.value = true
  e.preventDefault()

  const startX = e.clientX
  const startWidth = chatWidth.value

  function onMouseMove(me: MouseEvent) {
    if (!isDraggingChat.value) return
    // Dragging left widens the chat panel (divider moves left)
    const deltaX = startX - me.clientX
    const newWidth = Math.max(CHAT_MIN_WIDTH, Math.min(CHAT_MAX_WIDTH, startWidth + deltaX))
    chatWidth.value = newWidth
  }

  function onMouseUp() {
    isDraggingChat.value = false
    saveChatWidth(chatWidth.value)
    document.removeEventListener('mousemove', onMouseMove)
    document.removeEventListener('mouseup', onMouseUp)
  }

  document.addEventListener('mousemove', onMouseMove)
  document.addEventListener('mouseup', onMouseUp)
}

function onChatDblClick() {
  chatWidth.value = CHAT_DEFAULT_WIDTH
  saveChatWidth(CHAT_DEFAULT_WIDTH)
}

// D4-6: Migration prompt state
const showMigrationPrompt = ref(false)
const allChapterNumbers = ref<number[]>([])

const isCompact = computed(() => viewportWidth.value < 1024)
const isMobile = computed(() => viewportWidth.value < 768)
const isSplitEnabled = computed(() => workspaceStore.split.enabled)

function onResize() {
  viewportWidth.value = window.innerWidth
}

function toggleRightPanel() {
  rightPanelOverlay.value = !rightPanelOverlay.value
}

onMounted(async () => {
  window.addEventListener('resize', onResize)

  const annotationStore = useAnnotationStore()
  const agentStore = useAgentStore()
  const authStore = useAuthStore()

  registerGlobalRoutes({
    agentStore,
    evolutionStore: useEvolutionStore(),
    annotationStore,
    sourceTreeStore: useSourceTreeStore(),
    workspaceStore: useWorkspaceStore(),
    mapStore: useMapStore(),
    dynastyCompareStore: useDynastyCompareStore(),
  })

  //  store
  //
  useEvolutionStore().restoreAllCompletedAnalyses()

  // D4-4: Load data from server after login (SQLite — single source of truth)
  if (authStore.isAuthenticated) {
    await agentStore.syncConversationsFromServer()
    // sync 完成后设定活跃对话：有历史则选最新的，没有则创建一个
    if (agentStore.conversations.length > 0) {
      agentStore.setActiveConversation(agentStore.conversations[0].id)
    } else {
      agentStore.createConversation()
    }
    await annotationStore.initFromServer()

    // D4-6: Detect migration need (one-time localStorage → SQLite migration)
    try {
      const statsResp = await apiFetch('/api/annotations/stats')
      const stats = await statsResp.json() as { totalAnnotations: number; byChapter: Record<string, number> }
      const localRaw = localStorage.getItem('white_snake_user_annotation_sets')

      allChapterNumbers.value = Object.keys(stats.byChapter).map(Number)

      if (stats.totalAnnotations === 0 && localRaw) {
        showMigrationPrompt.value = true
      }
    } catch { /* Migration check is best-effort */ }
  }
})

async function onMigrationComplete() {
  showMigrationPrompt.value = false
  const annotationStore = useAnnotationStore()
  try {
    await annotationStore.reloadFromServer(allChapterNumbers.value)
  } catch (err) {
    console.error('[MuseumView] Failed to reload after migration:', err)
  }
}

onUnmounted(() => {
  window.removeEventListener('resize', onResize)
})
</script>

<template>
  <div class="museum-layout" :class="{ compact: isCompact, mobile: isMobile, 'split-active': isSplitEnabled }">
    <LeftColumn v-if="!isCompact" class="layout-left" :style="{ width: sidebarWidth + 'px' }" />
    <LeftColumn v-else class="layout-left layout-left-icons" />

    <!-- Sidebar resize divider -->
    <div
      v-if="!isCompact"
      class="sidebar-divider"
      :class="{ dragging: isDraggingSidebar }"
      @mousedown="onSidebarDragStart"
      @dblclick="onSidebarDblClick"
    />

    <MainWorkspace class="layout-main" />

    <!-- Chat panel resize divider -->
    <div
      v-if="!isMobile"
      class="chat-divider"
      :class="{ dragging: isDraggingChat }"
      @mousedown="onChatDragStart"
      @dblclick="onChatDblClick"
    />

    <RightColumn v-if="!isMobile" class="layout-right" :style="{ width: chatWidth + 'px' }" />

    <!-- Floating toggle button for mobile right panel -->
    <button
      v-if="isMobile"
      class="right-panel-toggle"
      @click="toggleRightPanel"
    >
      AI
    </button>

    <!-- Overlay right panel for mobile -->
    <Transition name="overlay-slide">
      <div v-if="isMobile && rightPanelOverlay" class="right-panel-overlay">
        <div class="overlay-backdrop" @click="toggleRightPanel" />
        <RightColumn class="overlay-content" />
      </div>
    </Transition>

    <!-- D4-6: Migration prompt -->
    <DataMigrationPrompt
      v-if="showMigrationPrompt"
      :show="showMigrationPrompt"
      @imported="onMigrationComplete"
      @skip="showMigrationPrompt = false"
    />
  </div>
</template>

<style scoped>
.museum-layout {
  display: flex;
  height: 100vh;
  overflow: hidden;
}

.layout-left {
  flex-shrink: 0;
  background: var(--color-bg-secondary);
  overflow: hidden;
}

.layout-left-icons {
  width: 48px;
}

/* Sidebar / chat resize dividers — 宽热区 + 1px 细线，hover/拖动时淡蓝加粗到 2px */
.sidebar-divider,
.chat-divider {
  width: 7px;
  cursor: col-resize;
  background: transparent;
  position: relative;
  flex-shrink: 0;
  user-select: none;
}

.sidebar-divider::before,
.chat-divider::before {
  content: '';
  position: absolute;
  top: 0;
  bottom: 0;
  left: 3px;
  width: 1px;
  background: var(--color-border);
  transition:
    background-color var(--dur-fast) var(--ease),
    width var(--dur-fast) var(--ease);
}

.sidebar-divider:hover::before,
.sidebar-divider.dragging::before,
.chat-divider:hover::before,
.chat-divider.dragging::before {
  left: 2.5px;
  width: 2px;
  background: var(--color-accent);
}

.layout-main {
  flex: 1;
  min-width: 0;
  background: var(--color-bg-primary);
}

.layout-right {
  flex-shrink: 0;
  min-width: 0;
  background: var(--color-bg-secondary);
}

/* Floating toggle for mobile */
.right-panel-toggle {
  position: fixed;
  bottom: var(--space-5);
  right: var(--space-5);
  z-index: 100;
  width: 44px;
  height: 44px;
  border-radius: var(--radius-full);
  border: 1px solid var(--color-border);
  background: var(--color-bg-primary);
  color: var(--color-text-primary);
  font-family: var(--font-sans);
  font-size: 0.75rem;
  font-weight: 600;
  cursor: pointer;
  box-shadow: var(--shadow-md);
  transition:
    background-color var(--dur-fast) var(--ease),
    border-color var(--dur-fast) var(--ease),
    box-shadow var(--dur) var(--ease),
    transform var(--dur) var(--ease);
}

.right-panel-toggle:hover {
  background: var(--color-bg-hover);
  border-color: var(--color-border-dark);
  box-shadow: var(--shadow-lg);
  transform: translateY(-1px);
}

.right-panel-toggle:active {
  transform: translateY(0);
}

/* Overlay for mobile */
.right-panel-overlay {
  position: fixed;
  inset: 0;
  z-index: 200;
  display: flex;
}

.overlay-backdrop {
  position: absolute;
  inset: 0;
  background: rgba(0, 0, 0, 0.4);
}

.overlay-content {
  position: relative;
  margin-left: auto;
  width: var(--agent-panel-width);
  max-width: 90vw;
  height: 100%;
  z-index: 1;
  border-left: 1px solid var(--color-border);
}

.overlay-slide-enter-active,
.overlay-slide-leave-active {
  transition: opacity var(--dur) var(--ease);
}

.overlay-slide-enter-active .overlay-content,
.overlay-slide-leave-active .overlay-content {
  transition: transform var(--dur) var(--ease-out);
}

.overlay-slide-enter-from {
  opacity: 0;
}

.overlay-slide-enter-from .overlay-content {
  transform: translateX(100%);
}

.overlay-slide-leave-to {
  opacity: 0;
}

.overlay-slide-leave-to .overlay-content {
  transform: translateX(100%);
}
</style>
