<script setup lang="ts">
import { computed, ref, watch, onUnmounted, nextTick } from 'vue'
import { useAgentStore } from '@/stores/agent'
import { SSEEventBus } from '@/services/sseEventBus'

const agentStore = useAgentStore()

const props = defineProps<{
  chapterNumber?: number
}>()

const emit = defineEmits<{ stop: [] }>()

// ---- 状态 ----
const p = computed(() => agentStore.annotationProgress)
// 仅展示当前章节的标注进度（避免在已切换到其他章节的阅读器里残留旧面板）
const chapterMatched = computed(() =>
  props.chapterNumber === undefined || p.value.chapterNumber === null
  || p.value.chapterNumber === props.chapterNumber
)
const visible = computed(() =>
  p.value.status !== 'idle' && !dismissed.value && chapterMatched.value
)
const running = computed(() => p.value.active && chapterMatched.value)

const dismissed = ref(false)
watch(() => p.value.status, (status, old) => {
  if (status === 'running' && old !== 'running') {
    // 新一轮开始：复位关闭态，并清除上一轮残留的自动隐藏计时
    dismissed.value = false
    scheduleAutoHide(true)
  } else if (status === 'done' || status === 'cancelled') {
    scheduleAutoHide(true, 8000)
  } else if (status === 'error') {
    scheduleAutoHide(true, 15000)
  }
})

let hideTimer: ReturnType<typeof setTimeout> | null = null
function scheduleAutoHide(cancelExisting: boolean, ms?: number) {
  if (cancelExisting && hideTimer) {
    clearTimeout(hideTimer)
    hideTimer = null
  }
  if (ms) {
    hideTimer = setTimeout(() => { dismissed.value = true }, ms)
  }
}
onUnmounted(() => { if (hideTimer) clearTimeout(hideTimer) })

function dismiss() {
  dismissed.value = true
}

// ---- 停止 ----
function onStop() {
  SSEEventBus.getInstance().abort()
  agentStore.cancelAnnotationProgress()
  agentStore.setAnnotationLoading(false)
  // 与聊天区停止按钮一致：结束流式态
  agentStore.finishStreaming()
  emit('stop')
}

// ---- 计时器 ----
const now = ref(Date.now())
let ticker: ReturnType<typeof setInterval> | null = null
watch(running, (isRunning) => {
  if (isRunning) {
    ticker = setInterval(() => { now.value = Date.now() }, 500)
  } else if (ticker) {
    clearInterval(ticker)
    ticker = null
  }
}, { immediate: true })
onUnmounted(() => { if (ticker) clearInterval(ticker) })

const elapsedSeconds = computed(() => {
  const start = p.value.startedAt
  if (!start) return 0
  const end = running.value ? now.value : start + (frozenElapsed.value || 0)
  return Math.max(0, Math.floor((end - start) / 1000))
})
const frozenElapsed = ref(0)
watch(() => p.value.status, (s) => {
  if (s !== 'running' && p.value.startedAt) {
    frozenElapsed.value = Date.now() - p.value.startedAt
  }
})

// ---- 阶段 stepper ----
interface Step { key: string; label: string }
const steps = computed<Step[]>(() => {
  const base: Step[] = [
    { key: 'prepare', label: '准备' },
    { key: 'discovery', label: '发现实体' },
    { key: 'resolution', label: '精确定位' },
  ]
  if (p.value.marginaliaTotal > 0 || p.value.phase === 'marginalia') {
    base.push({ key: 'marginalia', label: '旁注' })
  }
  base.push({ key: 'saving', label: '保存' })
  return base
})

const PHASE_ORDER = ['prepare', 'discovery', 'resolution', 'marginalia', 'saving', 'done']
function phaseIndex(phase: string): number {
  return PHASE_ORDER.indexOf(phase)
}

function stepState(stepKey: string): 'done' | 'running' | 'pending' | 'error' {
  if (p.value.status === 'done') return 'done'
  const cur = phaseIndex(p.value.phase)
  const mine = phaseIndex(stepKey)
  if (p.value.status === 'error') {
    if (mine < cur) return 'done'
    if (mine === cur) return 'error'
    return 'pending'
  }
  if (cur < 0) return 'pending'
  if (mine < cur) return 'done'
  if (mine === cur) return 'running'
  return 'pending'
}

// ---- 进度条百分比 ----
const progressPercent = computed(() => {
  switch (p.value.phase) {
    case 'prepare': return 4
    case 'discovery': {
      const frac = p.value.windowsTotal > 0 ? p.value.windowsDone / p.value.windowsTotal : 0
      return Math.round(6 + frac * 20)
    }
    case 'resolution': {
      const frac = p.value.totalParagraphs > 0
        ? Math.min(1, p.value.paragraphsDone / p.value.totalParagraphs) : 0
      return Math.round(28 + frac * 50)
    }
    case 'marginalia': {
      const frac = p.value.marginaliaTotal > 0 ? p.value.marginaliaDone / p.value.marginaliaTotal : 0
      return Math.round(80 + frac * 12)
    }
    case 'saving': return 95
    case 'done': return 100
    case 'error': return Math.min(96, progressBeforeError.value)
    default: return 0
  }
})
const progressBeforeError = ref(0)
watch(() => p.value.phase, (phase) => {
  if (phase !== 'error') progressBeforeError.value = progressPercent.value
})

// ---- 实时统计 ----
const stats = computed(() => p.value.finalStats ?? {
  entities: p.value.entities,
  paragraphs: p.value.paragraphsDone,
  marginalia: p.value.marginaliaDone,
})

const CATEGORY_LABELS: Record<string, string> = {
  person: '人物', location: '地点', event: '事件', term: '术语', motif: '母题', custom: '其他',
}
const categoryChips = computed(() => {
  const counts = p.value.finalStats?.categoryCounts ?? p.value.categoryCounts
  return Object.entries(counts || {})
    .filter(([, n]) => n > 0)
    .map(([key, n]) => ({ key, label: CATEGORY_LABELS[key] || key, n }))
})

// ---- 子步骤消息流 ----
const recentMessages = computed(() => p.value.recent.slice(-8))
const recentRef = ref<HTMLElement | null>(null)
watch(() => recentMessages.value.length, () => {
  void nextTick(() => {
    if (recentRef.value) recentRef.value.scrollTop = recentRef.value.scrollHeight
  })
})

const statusText = computed(() => {
  switch (p.value.status) {
    case 'done': return '标注完成'
    case 'error': return '标注失败'
    case 'cancelled': return '已停止'
    default: return '智能标注进行中'
  }
})
</script>

<template>
  <Transition name="ap-fade">
    <div v-if="visible" class="ap-card" :class="`ap-${p.status}`">
      <!-- 顶部进度条 -->
      <div class="ap-track">
        <div class="ap-fill" :style="{ width: progressPercent + '%' }" />
      </div>

      <!-- 头部 -->
      <div class="ap-header">
        <div class="ap-title-group">
          <span class="ap-title-dot" :class="{ pulsing: running }" />
          <span class="ap-title">{{ statusText }}</span>
          <span v-if="p.chapterTitle" class="ap-chapter">{{ p.chapterTitle }}</span>
        </div>
        <div class="ap-header-right">
          <span class="ap-timer">{{ elapsedSeconds }}s</span>
          <button
            v-if="running"
            class="ap-stop-btn"
            title="停止本次标注"
            @click="onStop"
          >停止</button>
          <button v-else class="ap-close-btn" title="关闭" @click="dismiss">×</button>
        </div>
      </div>

      <!-- 阶段 stepper -->
      <div class="ap-stepper">
        <template v-for="(step, idx) in steps" :key="step.key">
          <div class="ap-step" :class="stepState(step.key)">
            <div class="ap-marker">
              <svg v-if="stepState(step.key) === 'done'" class="ap-check" viewBox="0 0 16 16" fill="none">
                <path d="M3.5 8.5L6.5 11.5L12.5 4.5" stroke="currentColor" stroke-width="2"
                  stroke-linecap="round" stroke-linejoin="round" />
              </svg>
              <span v-else-if="stepState(step.key) === 'error'" class="ap-x">!</span>
              <div v-else-if="stepState(step.key) === 'running'" class="ap-pulse" />
              <span v-else class="ap-num">{{ idx + 1 }}</span>
            </div>
            <span class="ap-step-label">{{ step.label }}</span>
          </div>
          <div
            v-if="idx < steps.length - 1"
            class="ap-connector"
            :class="{ filled: phaseIndex(steps[idx + 1].key) <= phaseIndex(p.phase) || p.status === 'done' }"
          />
        </template>
      </div>

      <!-- 实时数字 -->
      <div class="ap-stats">
        <div class="ap-stat">
          <span class="ap-stat-num">{{ stats.entities }}</span>
          <span class="ap-stat-label">实体</span>
        </div>
        <div class="ap-stat">
          <span class="ap-stat-num">{{ stats.paragraphs }}<span class="ap-stat-total">/{{ p.totalParagraphs || '—' }}</span></span>
          <span class="ap-stat-label">高亮段落</span>
        </div>
        <div v-if="p.marginaliaTotal > 0 || stats.marginalia > 0" class="ap-stat">
          <span class="ap-stat-num">{{ stats.marginalia }}<span class="ap-stat-total">/{{ p.marginaliaTotal || '—' }}</span></span>
          <span class="ap-stat-label">旁注</span>
        </div>
        <div v-if="p.phase === 'discovery'" class="ap-stat">
          <span class="ap-stat-num">{{ p.candidates }}</span>
          <span class="ap-stat-label">候选</span>
        </div>
      </div>

      <!-- 分类计数 -->
      <div v-if="categoryChips.length > 0" class="ap-chips">
        <span v-for="chip in categoryChips" :key="chip.key" class="ap-chip">
          {{ chip.label }} {{ chip.n }}
        </span>
      </div>

      <!-- 当前消息 + 滚动日志 -->
      <div class="ap-log" ref="recentRef">
        <div
          v-for="(msg, i) in recentMessages"
          :key="msg.ts + '-' + i"
          class="ap-log-line"
          :class="{ current: i === recentMessages.length - 1 && running }"
        >
          <span class="ap-log-bar" />
          <span class="ap-log-text">{{ msg.message }}</span>
        </div>
      </div>

      <!-- 错误信息 -->
      <div v-if="p.status === 'error' && p.errorMessage" class="ap-error-msg">
        {{ p.errorMessage }}
      </div>
    </div>
  </Transition>
</template>

<style scoped>
.ap-card {
  position: sticky;
  top: 8px;
  z-index: 20;
  background: var(--color-bg-primary);
  border: 1px solid var(--color-border);
  border-radius: var(--radius-lg);
  box-shadow: var(--shadow-md);
  padding: var(--space-3) var(--space-4) var(--space-3);
  margin-bottom: var(--space-4);
  font-size: 0.78rem;
}

/* 顶部进度条 */
.ap-track {
  position: absolute;
  top: 0; left: 0; right: 0;
  height: 3px;
  background: var(--color-bg-tertiary);
  border-radius: var(--radius-lg) var(--radius-lg) 0 0;
  overflow: hidden;
}
.ap-fill {
  height: 100%;
  background: var(--color-accent);
  border-radius: var(--radius-full);
  transition: width 0.4s var(--ease);
}
.ap-error .ap-fill { background: var(--color-error); }

/* 头部 */
.ap-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: var(--space-3);
}
.ap-title-group {
  display: flex;
  align-items: center;
  gap: var(--space-2);
  min-width: 0;
}
.ap-title-dot {
  width: 8px; height: 8px;
  border-radius: 50%;
  background: var(--color-accent);
  flex-shrink: 0;
}
.ap-title-dot.pulsing {
  animation: ap-pulse-dot 1.4s var(--ease) infinite;
}
@keyframes ap-pulse-dot {
  0%, 100% { box-shadow: 0 0 0 0 rgba(91, 155, 213, 0.35); }
  50% { box-shadow: 0 0 0 5px rgba(91, 155, 213, 0); }
}
.ap-title {
  font-weight: 600;
  color: var(--color-text-primary);
  font-size: 0.8rem;
}
.ap-chapter {
  color: var(--color-text-tertiary);
  font-size: 0.72rem;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}
.ap-header-right {
  display: flex;
  align-items: center;
  gap: var(--space-2);
  flex-shrink: 0;
}
.ap-timer {
  font-family: var(--font-mono);
  font-size: 0.7rem;
  color: var(--color-text-tertiary);
}
.ap-stop-btn {
  padding: 2px 10px;
  font-size: 0.7rem;
  background: transparent;
  color: var(--color-text-secondary);
  border: 1px solid var(--color-border);
  border-radius: var(--radius-sm);
  cursor: pointer;
  transition: all var(--dur-fast) var(--ease);
}
.ap-stop-btn:hover {
  border-color: var(--color-error);
  color: var(--color-error);
  background: var(--color-error-light);
}
.ap-close-btn {
  width: 22px; height: 22px;
  display: flex;
  align-items: center;
  justify-content: center;
  background: transparent;
  border: none;
  color: var(--color-text-tertiary);
  font-size: 1rem;
  line-height: 1;
  cursor: pointer;
  border-radius: var(--radius-sm);
}
.ap-close-btn:hover { color: var(--color-text-primary); background: var(--color-bg-hover); }

/* stepper */
.ap-stepper {
  display: flex;
  align-items: flex-start;
  margin-bottom: var(--space-3);
}
.ap-step {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 4px;
  flex-shrink: 0;
  width: 52px;
}
.ap-marker {
  width: 22px; height: 22px;
  display: flex;
  align-items: center;
  justify-content: center;
}
.ap-check {
  width: 18px; height: 18px;
  color: var(--color-accent);
}
.ap-num {
  width: 18px; height: 18px;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 0.62rem;
  font-family: var(--font-mono);
  color: var(--color-text-tertiary);
  border: 1px solid var(--color-border);
  border-radius: 50%;
  background: var(--color-bg-primary);
}
.ap-pulse {
  width: 10px; height: 10px;
  border-radius: 50%;
  background: var(--color-accent);
  border: 2px solid var(--color-bg-primary);
  position: relative;
}
.ap-pulse::before {
  content: '';
  position: absolute;
  inset: -4px;
  border-radius: 50%;
  border: 1.5px solid var(--color-accent);
  animation: ap-ring 1.5s var(--ease) infinite;
}
@keyframes ap-ring {
  0% { transform: scale(0.85); opacity: 0.8; }
  100% { transform: scale(1.55); opacity: 0; }
}
.ap-x {
  width: 18px; height: 18px;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 0.66rem;
  font-weight: 700;
  color: var(--color-error);
  border: 1px solid var(--color-error);
  border-radius: 50%;
}
.ap-step-label {
  font-size: 0.66rem;
  color: var(--color-text-tertiary);
  white-space: nowrap;
}
.ap-step.done .ap-step-label { color: var(--color-text-secondary); }
.ap-step.running .ap-step-label { color: var(--color-accent); font-weight: 600; }
.ap-step.error .ap-step-label { color: var(--color-error); font-weight: 600; }
.ap-connector {
  flex: 1;
  height: 1px;
  background: var(--color-border);
  margin-top: 11px;
  min-width: 8px;
  transition: background var(--dur-slow) var(--ease);
}
.ap-connector.filled { background: var(--color-accent); opacity: 0.5; }

/* 统计数字 */
.ap-stats {
  display: flex;
  gap: var(--space-4);
  margin-bottom: var(--space-2);
}
.ap-stat {
  display: flex;
  align-items: baseline;
  gap: 4px;
}
.ap-stat-num {
  font-size: 1.05rem;
  font-weight: 700;
  font-family: var(--font-mono);
  color: var(--color-text-primary);
}
.ap-stat-total {
  font-size: 0.72rem;
  font-weight: 400;
  color: var(--color-text-tertiary);
}
.ap-stat-label {
  font-size: 0.68rem;
  color: var(--color-text-tertiary);
}

/* 分类 chips */
.ap-chips {
  display: flex;
  flex-wrap: wrap;
  gap: 6px;
  margin-bottom: var(--space-2);
}
.ap-chip {
  font-size: 0.66rem;
  color: var(--color-text-secondary);
  background: var(--color-accent-light);
  border-radius: var(--radius-full);
  padding: 1px 8px;
  line-height: 1.6;
}

/* 日志 */
.ap-log {
  max-height: 96px;
  overflow-y: auto;
  background: var(--color-bg-secondary);
  border-radius: var(--radius-sm);
  padding: 6px 10px;
  display: flex;
  flex-direction: column;
  gap: 1px;
  scrollbar-width: thin;
  scrollbar-color: var(--color-border) transparent;
}
.ap-log::-webkit-scrollbar { width: 4px; }
.ap-log::-webkit-scrollbar-thumb { background: var(--color-border); border-radius: var(--radius-full); }
.ap-log-line {
  display: flex;
  align-items: flex-start;
  gap: 7px;
  font-size: 0.7rem;
  line-height: 1.55;
  color: var(--color-text-tertiary);
}
.ap-log-line.current .ap-log-text {
  color: var(--color-text-secondary);
}
.ap-log-bar {
  flex-shrink: 0;
  width: 2px;
  align-self: stretch;
  min-height: 12px;
  background: var(--color-border);
  border-radius: var(--radius-full);
}
.ap-log-line.current .ap-log-bar {
  background: var(--color-accent);
  animation: ap-bar-breathe 1.4s var(--ease) infinite;
}
@keyframes ap-bar-breathe {
  0%, 100% { opacity: 0.35; }
  50% { opacity: 0.9; }
}
.ap-log-text {
  flex: 1;
  min-width: 0;
  word-break: break-word;
}

/* 错误态 */
.ap-error-msg {
  margin-top: var(--space-2);
  font-size: 0.72rem;
  color: var(--color-error);
  line-height: 1.5;
}

/* 过渡 */
.ap-fade-enter-active {
  transition: all var(--dur-slow) var(--ease-out);
}
.ap-fade-leave-active {
  transition: all var(--dur-fast) var(--ease);
}
.ap-fade-enter-from {
  opacity: 0;
  transform: translateY(-8px);
}
.ap-fade-leave-to {
  opacity: 0;
  transform: translateY(-8px);
}
</style>
