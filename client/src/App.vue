<script setup lang="ts">
import { watch, ref, onMounted, onUnmounted } from 'vue'
import { useRoute } from 'vue-router'
import { useAgentStore } from '@/stores/agent'
import { useWorkspaceStore } from '@/stores/workspace'
import { useAuthStore } from '@/stores/auth'
import GlobalErrorBoundary from '@/components/GlobalErrorBoundary.vue'
import Toast from '@/components/ui/Toast.vue'

const agentStore = useAgentStore()
const workspaceStore = useWorkspaceStore()
const authStore = useAuthStore()
const route = useRoute()
const authReady = ref(false)

// Dynamic page title
watch(
  () => agentStore.isStreaming,
  (streaming) => {
    if (streaming) {
      document.title = '思考中… — 白蛇传文脉全息'
    } else {
      document.title = '白蛇传 · 文脉全息博物馆'
    }
  }
)

// Global Toast for SSE errors
const toastMessage = ref('')
const toastType = ref<'info' | 'success' | 'warning' | 'error'>('info')

watch(
  () => agentStore.errorMessage,
  (msg) => {
    if (msg) {
      toastMessage.value = msg
      toastType.value = 'error'
    }
  }
)

// Global keyboard navigation
function onKeydown(e: KeyboardEvent) {
  // Ctrl+K: focus agent input
  if (e.ctrlKey && e.key === 'k') {
    e.preventDefault()
    const textarea = document.querySelector('.ac-textarea') as HTMLTextAreaElement | null
    textarea?.focus()
  }

  // Ctrl+1-6: switch to tab by index
  const numMatch = e.key.match(/^(\d)$/)
  if (e.ctrlKey && numMatch) {
    const idx = parseInt(numMatch[1]) - 1
    if (idx < workspaceStore.tabs.length) {
      e.preventDefault()
      workspaceStore.setActiveTab(workspaceStore.tabs[idx].id)
    }
  }

  // Ctrl+W: close active tab
  if (e.ctrlKey && e.key === 'w' && workspaceStore.activeTabId) {
    e.preventDefault()
    workspaceStore.closeTab(workspaceStore.activeTabId)
  }

  // Ctrl+Shift+E: switch to EvolutionWorkspace
  if (e.ctrlKey && e.shiftKey && e.key === 'E') {
    e.preventDefault()
    const evoTab = workspaceStore.tabs.find(t => t.type === 'evolution-workspace')
    if (evoTab) workspaceStore.setActiveTab(evoTab.id)
  }
}

onMounted(async () => {
  window.addEventListener('keydown', onKeydown)
  await authStore.fetchMe()
  authReady.value = true
})

onUnmounted(() => {
  window.removeEventListener('keydown', onKeydown)
})
</script>

<template>
  <GlobalErrorBoundary>
    <Toast
      :message="toastMessage"
      :type="toastType"
      @close="toastMessage = ''"
    />
    <div v-if="!authReady" class="auth-loader">
      <div class="auth-spinner" aria-label="加载中"></div>
    </div>
    <RouterView v-if="authReady" />
  </GlobalErrorBoundary>
</template>

<style>
/* ============================================================
   GLOBAL RESET · 极简黑白风格
============================================================ */
*, *::before, *::after { box-sizing: border-box; }
* { margin: 0; padding: 0; }

html {
  font-size: clamp(14px, 1.2vw, 16px);
  -webkit-text-size-adjust: 100%;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
  text-rendering: optimizeLegibility;
}

html, body, #app {
  height: 100%;
  width: 100%;
}

body {
  font-family: var(--font-sans);
  background: var(--color-bg-primary);
  color: var(--color-text-primary);
  font-size: 0.9rem;
  line-height: 1.6;
  overflow: hidden;
  -webkit-tap-highlight-color: transparent;
}

/* 选区文字：淡蓝高亮（点缀色的一处使用） */
::selection {
  background: var(--color-accent-light);
  color: var(--color-text-primary);
}

/* ============================================================
   SCROLLBAR · 极细、低饱和、hover 丝滑变深
============================================================ */
::-webkit-scrollbar {
  width: 6px;
  height: 6px;
}
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb {
  background: #D8DDE3;
  border-radius: 999px;
  transition: background var(--dur-fast) var(--ease);
}
::-webkit-scrollbar-thumb:hover { background: #B8BFC8; }
::-webkit-scrollbar-corner { background: transparent; }

/* ============================================================
   FOCUS RING · 统一 accessibility
============================================================ */
:focus-visible {
  outline: none;
  box-shadow: var(--shadow-focus);
  border-radius: var(--radius-sm);
}

/* ============================================================
   TRANSITION PRESETS · Vue <Transition name=""> 直接用
============================================================ */
/* --- fade（透明度） --- */
.fade-enter-from, .fade-leave-to { opacity: 0; }
.fade-enter-active, .fade-leave-active {
  transition: opacity var(--dur) var(--ease);
}

/* --- slide-fade（抽屉：位移 + 透明） --- */
.slide-fade-ltr-enter-from, .slide-fade-ltr-leave-to {
  opacity: 0; transform: translateX(-16px);
}
.slide-fade-ltr-enter-active, .slide-fade-ltr-leave-active {
  transition: opacity var(--dur) var(--ease), transform var(--dur) var(--ease-out);
}

.slide-fade-rtl-enter-from, .slide-fade-rtl-leave-to {
  opacity: 0; transform: translateX(16px);
}
.slide-fade-rtl-enter-active, .slide-fade-rtl-leave-active {
  transition: opacity var(--dur) var(--ease), transform var(--dur) var(--ease-out);
}

.slide-fade-ttb-enter-from, .slide-fade-ttb-leave-to {
  opacity: 0; transform: translateY(-12px);
}
.slide-fade-ttb-enter-active, .slide-fade-ttb-leave-active {
  transition: opacity var(--dur) var(--ease), transform var(--dur) var(--ease-out);
}

/* --- pop（弹窗/卡片缩放） --- */
.pop-enter-from, .pop-leave-to {
  opacity: 0; transform: scale(0.96);
}
.pop-enter-active, .pop-leave-active {
  transition: opacity var(--dur) var(--ease), transform var(--dur-slow) var(--ease-out);
}

/* ============================================================
   UTILITY COMPONENT CLASSES · 全站复用（按钮/输入/卡片/Badge）
============================================================ */

/* ---------- 按钮 ---------- */
.btn {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  gap: 6px;
  height: 32px;
  padding: 0 14px;
  font-size: 0.82rem;
  font-weight: 500;
  font-family: inherit;
  color: var(--color-text-primary);
  background: var(--color-bg-primary);
  border: 1px solid var(--color-border);
  border-radius: var(--radius);
  cursor: pointer;
  user-select: none;
  white-space: nowrap;
  transition:
    background-color var(--dur-fast) var(--ease),
    border-color var(--dur-fast) var(--ease),
    color var(--dur-fast) var(--ease),
    transform var(--dur-fast) var(--ease),
    box-shadow var(--dur-fast) var(--ease);
}
.btn:hover { background: var(--color-bg-hover); border-color: var(--color-border-dark); }
.btn:active { background: var(--color-bg-active); transform: translateY(0); }
.btn:disabled {
  background: var(--color-bg-tertiary);
  color: var(--color-text-disabled);
  border-color: var(--color-border);
  cursor: not-allowed;
  transform: none !important;
}

/* 主按钮（唯一大量用点缀色的地方） */
.btn-primary {
  color: #fff;
  background: var(--color-accent);
  border-color: var(--color-accent);
}
.btn-primary:hover {
  background: var(--color-accent-hover);
  border-color: var(--color-accent-hover);
  transform: translateY(-1px);
  box-shadow: 0 4px 10px rgba(91, 155, 213, 0.18);
}
.btn-primary:active { background: var(--color-accent-active); transform: translateY(0); }

/* ghost 按钮：无边框背景，hover 才出底 */
.btn-ghost {
  background: transparent;
  border-color: transparent;
  color: var(--color-text-secondary);
}
.btn-ghost:hover { background: var(--color-bg-hover); color: var(--color-text-primary); }

.btn-sm { height: 26px; padding: 0 10px; font-size: 0.75rem; border-radius: var(--radius-sm); }
.btn-lg { height: 40px; padding: 0 18px; font-size: 0.9rem; }

/* ---------- 输入框 / 文本域 ---------- */
.input, .textarea {
  width: 100%;
  font-family: inherit;
  font-size: 0.85rem;
  color: var(--color-text-primary);
  background: var(--color-bg-primary);
  border: 1px solid var(--color-border);
  border-radius: var(--radius);
  padding: 8px 12px;
  transition:
    border-color var(--dur-fast) var(--ease),
    box-shadow var(--dur-fast) var(--ease),
    background-color var(--dur-fast) var(--ease);
}
.textarea { resize: vertical; min-height: 80px; line-height: 1.6; }
.input::placeholder, .textarea::placeholder { color: var(--color-text-placeholder); }
.input:hover, .textarea:hover { border-color: var(--color-border-dark); }
.input:focus, .textarea:focus {
  outline: none;
  border-color: var(--color-accent);
  box-shadow: var(--shadow-focus);
}
.input:disabled, .textarea:disabled {
  background: var(--color-bg-tertiary);
  color: var(--color-text-disabled);
  cursor: not-allowed;
}

/* ---------- 卡片 ---------- */
.card {
  background: var(--color-bg-primary);
  border: 1px solid var(--color-border);
  border-radius: var(--radius-lg);
  padding: 16px;
  transition:
    border-color var(--dur) var(--ease),
    box-shadow var(--dur) var(--ease),
    transform var(--dur) var(--ease);
}
.card-hover:hover {
  border-color: var(--color-border-dark);
  box-shadow: var(--shadow-md);
  transform: translateY(-1px);
}
.card-flat {
  border-radius: var(--radius);
  padding: 12px 14px;
}

/* ---------- Badge / 标签 ---------- */
.badge {
  display: inline-flex;
  align-items: center;
  gap: 4px;
  height: 20px;
  padding: 0 8px;
  font-size: 0.7rem;
  font-weight: 500;
  line-height: 1;
  color: var(--color-text-tertiary);
  background: var(--color-bg-secondary);
  border: 1px solid var(--color-border);
  border-radius: var(--radius-full);
  white-space: nowrap;
  transition:
    background-color var(--dur-fast) var(--ease),
    color var(--dur-fast) var(--ease),
    border-color var(--dur-fast) var(--ease);
}
.badge-accent {
  color: var(--color-accent);
  background: var(--color-accent-light);
  border-color: transparent;
}
.badge-success { color: #389E0D; background: var(--color-success-light); border-color: transparent; }
.badge-warning { color: #D48806; background: var(--color-warning-light); border-color: transparent; }
.badge-error   { color: #CF1322; background: var(--color-error-light);   border-color: transparent; }

/* ---------- 分割线 ---------- */
.divider {
  height: 1px;
  background: var(--color-border);
  margin: 12px 0;
  border: 0;
}

/* ---------- 文本等级 ---------- */
.t-h1 { font-size: 1.6rem; font-weight: 600; letter-spacing: -0.02em; color: var(--color-text-primary); }
.t-h2 { font-size: 1.25rem; font-weight: 600; color: var(--color-text-primary); }
.t-h3 { font-size: 1rem; font-weight: 600; color: var(--color-text-primary); }
.t-sub { color: var(--color-text-tertiary); font-size: 0.8rem; }
.t-muted { color: var(--color-text-placeholder); font-size: 0.75rem; }
.t-link {
  color: var(--color-accent);
  text-decoration: none;
  cursor: pointer;
  transition: color var(--dur-fast) var(--ease);
}
.t-link:hover { color: var(--color-accent-hover); text-decoration: underline; }

/* ---------- 标签页 Tab 激活下划线（组件内配合使用） ---------- */
.tab-indicator {
  position: relative;
}
.tab-indicator::after {
  content: '';
  position: absolute;
  left: 12px; right: 12px; bottom: 0;
  height: 2px;
  background: var(--color-accent);
  border-radius: 2px;
  transform: scaleX(0);
  transform-origin: center;
  transition: transform var(--dur) var(--ease-out);
}
.tab-indicator.is-active::after { transform: scaleX(1); }

/* ============================================================
   AUTH LOADER
============================================================ */
.auth-loader {
  display: flex;
  align-items: center;
  justify-content: center;
  height: 100vh;
  background: var(--color-bg-primary);
}
.auth-spinner {
  width: 28px;
  height: 28px;
  border: 2px solid var(--color-border);
  border-top-color: var(--color-accent);
  border-radius: 50%;
  animation: spin 0.7s linear infinite;
}
@keyframes spin {
  to { transform: rotate(360deg); }
}

/* ============================================================
   LEAFLET MAP · 淡染底图 + 极简控件（全局，两处地图实例共用）
============================================================ */
/* 底图淡处理：轻压饱和 + 提亮，清亮简约、街道不艳、无灰感 */
.leaflet-tile-pane {
  filter: saturate(0.75) brightness(1.1) contrast(0.97);
}

/* 缩放控件 */
.leaflet-control-zoom {
  border: none !important;
  border-radius: var(--radius) !important;
  overflow: hidden;
  box-shadow: var(--shadow-md) !important;
}
.leaflet-control-zoom a {
  width: 30px !important;
  height: 30px !important;
  line-height: 30px !important;
  background: var(--color-bg-primary) !important;
  color: var(--color-text-secondary) !important;
  border-bottom: 1px solid var(--color-border-light) !important;
  font-size: 15px !important;
  transition: background-color var(--dur-fast) var(--ease), color var(--dur-fast) var(--ease);
}
.leaflet-control-zoom a:last-child { border-bottom: none !important; }
.leaflet-control-zoom a:hover {
  background: var(--color-bg-hover) !important;
  color: var(--color-text-primary) !important;
}

/* Tooltip */
.leaflet-tooltip {
  background: var(--color-bg-primary);
  border: 1px solid var(--color-border);
  border-radius: var(--radius-sm);
  color: var(--color-text-primary);
  font-family: var(--font-sans);
  font-size: 0.72rem;
  line-height: 1.55;
  padding: 6px 10px;
  box-shadow: var(--shadow-md);
}
.leaflet-tooltip b { font-weight: 600; }
.leaflet-tooltip-top:before { border-top-color: var(--color-bg-primary); }
.leaflet-tooltip-bottom:before { border-bottom-color: var(--color-bg-primary); }
.leaflet-tooltip-left:before { border-left-color: var(--color-bg-primary); }
.leaflet-tooltip-right:before { border-right-color: var(--color-bg-primary); }

/* 时间脉冲标记 */
.mv-pulse-marker {
  animation: map-pulse 1.6s var(--ease) infinite;
  transform-origin: center;
  transform-box: fill-box;
}
@keyframes map-pulse {
  0%, 100% { opacity: 1; }
  50% { opacity: 0.45; }
}
</style>
