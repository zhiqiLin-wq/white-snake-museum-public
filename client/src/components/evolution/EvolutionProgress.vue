<script setup lang="ts">
import { computed, ref, watch, onUnmounted, nextTick } from 'vue'
import type { EvolutionPhase } from '@/types/evolution'
import type { StatusStep } from '@/types/sse-events'
import { useAgentStore } from '@/stores/agent'

const props = defineProps<{
  phase: EvolutionPhase
  statusSteps: StatusStep[]
  isLive: boolean
}>()

const emit = defineEmits<{
  cancel: []
}>()

const agentStore = useAgentStore()

// ---- 步骤定义 ----
interface EvoStep {
  code: string
  label: string
  desc: string
  phase: EvolutionPhase
}

const ALL_STEPS: EvoStep[] = [
  { code: 'E1', label: '分解母题', desc: '将追踪目标分解为叙事母题', phase: 'E1_decompose' },
  { code: 'E2', label: '扫描矩阵', desc: '扫描各朝代文本中的母题出现', phase: 'E2_scan' },
  { code: 'E3', label: '分类变化', desc: '识别每个母题的变化类型', phase: 'E3_classify' },
  { code: 'E4', label: '推断谱系', desc: '建模文本间的传承关系', phase: 'E4_phylogeny' },
  { code: 'E5', label: '外部关联', desc: '关联历史事件和社会背景', phase: 'E5_correlate' },
  { code: 'E6', label: '提出假设', desc: '基于模式提出演化假设', phase: 'E6_hypothesize' },
  { code: 'E7', label: '验证假设', desc: '逐条验证假设中的论断', phase: 'E7_test' },
  { code: 'E8', label: '综合呈现', desc: '生成最终分析报告', phase: 'E8_synthesize' },
]

// 中断态 → 对应步骤
const interruptPhaseMap: Record<string, EvolutionPhase> = {
  E1_interrupt: 'E1_decompose',
  E6_interrupt: 'E6_hypothesize',
  E7_interrupt: 'E7_test',
}

function stepStatus(step: EvoStep): 'done' | 'running' | 'pending' {
  const currentPhase = props.phase
  if (currentPhase === 'idle' || currentPhase === 'error') return 'pending'

  const resolvedPhase = interruptPhaseMap[currentPhase] || currentPhase
  const stepIdx = ALL_STEPS.indexOf(step)
  const currentIdx = ALL_STEPS.findIndex(s => s.phase === resolvedPhase)

  if (currentIdx === -1) return 'pending'
  if (stepIdx < currentIdx) return 'done'
  if (stepIdx === currentIdx) return 'running'
  return 'pending'
}

const completedCount = computed(() =>
  ALL_STEPS.filter(s => stepStatus(s) === 'done').length
)
const progressPercent = computed(() =>
  Math.round((completedCount.value / ALL_STEPS.length) * 100)
)

// ---- 计时器 ----
const elapsedSeconds = ref(0)
let timer: ReturnType<typeof setInterval> | null = null

const startTime = computed(() => agentStore.thinkingStartedAt)

watch(startTime, (val) => {
  if (val) {
    elapsedSeconds.value = 0
    if (timer) clearInterval(timer)
    timer = setInterval(() => {
      elapsedSeconds.value = Math.floor((Date.now() - val) / 1000)
    }, 250)
  } else {
    if (timer) { clearInterval(timer); timer = null }
  }
}, { immediate: true })

onUnmounted(() => {
  if (timer) clearInterval(timer)
})

// ---- 子步骤消息（当前运行阶段的所有可见步骤：thinking_step/status/tool_call/tool_result）----
// 显示最近 40 条以保证细粒度反馈不丢失，容器内滚动避免面板过高
const subSteps = computed(() => {
  if (!props.isLive) return []
  return props.statusSteps.filter(s =>
    s.stepType === 'thinking_step'
    || s.stepType === 'status'
    || s.stepType === 'tool_call'
    || s.stepType === 'tool_result'
  ).slice(-40)
})

// 自动滚动到最新子步骤
const substepsRef = ref<HTMLElement | null>(null)
watch(() => subSteps.value.length, () => {
  void nextTick(() => {
    if (substepsRef.value) {
      substepsRef.value.scrollTop = substepsRef.value.scrollHeight
    }
  })
})

// ---- 当前阶段的标题 ----
const currentStep = computed(() =>
  ALL_STEPS.find(s => stepStatus(s) === 'running')
)
</script>

<template>
  <div class="evo-progress">
    <!-- 顶部进度条 -->
    <div class="evo-progress-track">
      <div class="evo-progress-fill" :style="{ width: progressPercent + '%' }" />
    </div>

    <!-- 头部 -->
    <div class="evo-progress-header">
      <span class="evo-progress-title">演化分析进度</span>
      <span class="evo-progress-meta">
        <span class="evo-meta-count">{{ completedCount }}/{{ ALL_STEPS.length }}</span>
        <span class="evo-meta-sep">·</span>
        <span class="evo-meta-time">{{ elapsedSeconds }}s</span>
      </span>
    </div>

    <!-- 时间线 -->
    <div class="evo-timeline">
      <div
        v-for="(step, idx) in ALL_STEPS"
        :key="step.code"
        class="evo-step"
        :class="stepStatus(step)"
      >
        <!-- 左侧标记 -->
        <div class="evo-step-marker">
          <template v-if="stepStatus(step) === 'done'">
            <svg class="evo-check-svg" viewBox="0 0 16 16" fill="none">
              <path d="M3.5 8.5L6.5 11.5L12.5 4.5" stroke="currentColor" stroke-width="2"
                stroke-linecap="round" stroke-linejoin="round" />
            </svg>
          </template>
          <div v-else-if="stepStatus(step) === 'running'" class="evo-pulse-ring" />
          <span v-else class="evo-step-num">{{ idx + 1 }}</span>
        </div>

        <!-- 连接线 -->
        <div
          v-if="idx < ALL_STEPS.length - 1"
          class="evo-step-line"
          :class="{
            filled: stepStatus(step) === 'done',
            active: stepStatus(step) === 'done' && stepStatus(ALL_STEPS[idx + 1]) === 'running',
          }"
        />

        <!-- 右侧内容 -->
        <div class="evo-step-body">
          <div class="evo-step-label">
            <span class="evo-step-code">{{ step.code }}</span>
            <span class="evo-step-name">{{ step.label }}</span>
          </div>
          <div class="evo-step-desc">{{ step.desc }}</div>

          <!-- 运行中步骤的子消息 -->
          <div
            v-if="stepStatus(step) === 'running' && subSteps.length > 0"
            class="evo-substeps-scroll"
            ref="substepsRef"
          >
            <TransitionGroup
              name="evo-sub"
              tag="div"
              class="evo-substeps"
            >
              <div
                v-for="msg in subSteps"
                :key="msg.ts"
                class="evo-substep"
                :class="{ done: msg.done, ['evo-sub-' + (msg.stepType || 'status')]: true }"
              >
                <span class="evo-substep-bar" />
                <span class="evo-substep-text">{{ msg.label || msg.message }}</span>
                <span v-if="msg.durationMs" class="evo-substep-dur">
                  {{ (msg.durationMs / 1000).toFixed(1) }}s
                </span>
              </div>
            </TransitionGroup>
          </div>
        </div>
      </div>
    </div>

    <!-- 底部操作 -->
    <div class="evo-progress-footer">
      <div v-if="currentStep" class="evo-current-label">
        正在执行: {{ currentStep.code }} {{ currentStep.label }}
      </div>
      <button class="evo-cancel-btn" @click="emit('cancel')">取消分析</button>
    </div>
  </div>
</template>

<style scoped>
.evo-progress {
  padding: var(--space-5) var(--space-6);
  background: var(--color-bg-primary);
  border: 1px solid var(--color-border);
  border-radius: var(--radius);
  box-shadow: var(--shadow-sm);
}

/* ---- 顶部进度条 ---- */
.evo-progress-track {
  height: 3px;
  background: var(--color-bg-tertiary);
  border-radius: var(--radius-full);
  overflow: hidden;
  margin-bottom: var(--space-4);
}

.evo-progress-fill {
  height: 100%;
  background: var(--color-text-tertiary);
  border-radius: var(--radius-full);
  transition: width var(--dur-slow) var(--ease);
  position: relative;
}

.evo-progress-fill::after {
  content: '';
  position: absolute;
  top: 0; right: 0; bottom: 0;
  width: 24px;
  background: linear-gradient(90deg, transparent, rgba(255,255,255,0.5));
  animation: evo-shimmer 1.8s var(--ease) infinite;
}

@keyframes evo-shimmer {
  0%, 100% { opacity: 0; }
  50% { opacity: 1; }
}

/* ---- 头部 ---- */
.evo-progress-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: var(--space-4);
}

.evo-progress-title {
  font-size: 0.82rem;
  font-weight: 600;
  color: var(--color-text-primary);
}

.evo-progress-meta {
  font-size: 0.72rem;
  font-family: var(--font-mono);
  color: var(--color-text-secondary);
}

.evo-meta-sep {
  margin: 0 4px;
  color: var(--color-text-tertiary);
}

/* ---- 时间线（由外层 .ew-scroll-area 统一滚动，此处不再嵌套滚动）---- */
.evo-timeline {
  position: relative;
}

.evo-step {
  display: flex;
  align-items: flex-start;
  gap: 0;
  position: relative;
  padding-bottom: var(--space-3);
}

.evo-step:last-child {
  padding-bottom: 0;
}

/* ---- 左侧标记 ---- */
.evo-step-marker {
  width: 24px;
  height: 24px;
  flex-shrink: 0;
  display: flex;
  align-items: center;
  justify-content: center;
  position: relative;
  z-index: 1;
}

.evo-check-svg {
  width: 16px;
  height: 16px;
  color: var(--color-text-tertiary);
}

.evo-pulse-ring {
  width: 10px;
  height: 10px;
  border-radius: 50%;
  background: var(--color-bg-primary);
  border: 2px solid var(--color-accent);
  position: relative;
}

.evo-pulse-ring::before {
  content: '';
  position: absolute;
  inset: -4px;
  border-radius: 50%;
  border: 1.5px solid var(--color-accent);
  animation: evo-pulse 1.6s var(--ease) infinite;
}

@keyframes evo-pulse {
  0% { transform: scale(0.9); opacity: 0.8; }
  100% { transform: scale(1.6); opacity: 0; }
}

.evo-step-num {
  font-size: 0.62rem;
  font-family: var(--font-mono);
  color: var(--color-text-tertiary);
  width: 18px;
  height: 18px;
  display: flex;
  align-items: center;
  justify-content: center;
  border: 1px solid var(--color-border);
  border-radius: 50%;
}

/* ---- 连接线 ---- */
.evo-step-line {
  position: absolute;
  left: 11px;
  top: 24px;
  width: 2px;
  bottom: -4px;
  background: var(--color-border);
  z-index: 0;
}

.evo-step-line.filled {
  background: var(--color-accent);
  opacity: 0.4;
}

.evo-step-line.active {
  background: linear-gradient(180deg, var(--color-accent), var(--color-border));
  opacity: 0.6;
}

/* ---- 右侧内容 ---- */
.evo-step-body {
  flex: 1;
  padding-left: var(--space-3);
  min-width: 0;
}

.evo-step-label {
  display: flex;
  align-items: center;
  gap: 6px;
}

.evo-step-code {
  font-size: 0.62rem;
  font-family: var(--font-mono);
  color: var(--color-text-tertiary);
}

.evo-step-name {
  font-size: 0.8rem;
  font-weight: 500;
  color: var(--color-text-secondary);
  transition: color var(--dur-fast) var(--ease);
}

.evo-step-desc {
  font-size: 0.7rem;
  color: var(--color-text-tertiary);
  line-height: 1.45;
  margin-top: 2px;
}

/* ---- 状态变体 ---- */
.evo-step.done .evo-step-name {
  color: var(--color-text-primary);
}

.evo-step.running .evo-step-name {
  color: var(--color-accent);
  font-weight: 600;
}

.evo-step.running .evo-step-code {
  color: var(--color-accent);
}

.evo-step.running .evo-step-desc {
  color: var(--color-text-secondary);
}

/* ---- 子步骤（细粒度事件流）---- */
.evo-substeps-scroll {
  margin-top: var(--space-3);
  max-height: 260px;
  overflow-y: auto;
  /* 细滚动条，简洁不抢眼 */
  scrollbar-width: thin;
  scrollbar-color: var(--color-border) transparent;
}
.evo-substeps-scroll::-webkit-scrollbar {
  width: 4px;
}
.evo-substeps-scroll::-webkit-scrollbar-thumb {
  background: var(--color-border);
  border-radius: var(--radius-full);
}
.evo-substeps-scroll::-webkit-scrollbar-track {
  background: transparent;
}
.evo-substeps {
  padding: var(--space-2) var(--space-3);
  background: var(--color-bg-secondary);
  border-radius: var(--radius-sm);
  display: flex;
  flex-direction: column;
  gap: 2px;
  position: relative;
}

.evo-substep {
  display: flex;
  align-items: flex-start;
  gap: 8px;
  font-size: 0.7rem;
  color: var(--color-text-secondary);
  line-height: 1.5;
  padding: 3px 0;
}

/* 左侧细条标记，替代 emoji 图标 */
.evo-substep-bar {
  flex-shrink: 0;
  width: 2px;
  align-self: stretch;
  min-height: 14px;
  background: var(--color-accent);
  opacity: 0.55;
  border-radius: var(--radius-full);
  transition: opacity var(--dur) var(--ease);
}

/* thinking_step 进行中：细条呼吸 */
.evo-substep.evo-sub-thinking_step:not(.done) .evo-substep-bar {
  animation: evo-bar-breathe 1.4s var(--ease) infinite;
}

.evo-substep.done .evo-substep-bar {
  background: var(--color-border-dark);
  opacity: 0.5;
}

.evo-substep.done .evo-substep-text {
  color: var(--color-text-tertiary);
}

@keyframes evo-bar-breathe {
  0%, 100% { opacity: 0.3; }
  50% { opacity: 0.9; }
}

.evo-substep-text {
  flex: 1;
  min-width: 0;
  word-break: break-word;
}

.evo-substep-dur {
  font-family: var(--font-mono);
  font-size: 0.6rem;
  color: var(--color-text-tertiary);
  flex-shrink: 0;
  align-self: flex-start;
  margin-top: 1px;
}

/* ---- 底部 ---- */
.evo-progress-footer {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-top: var(--space-4);
  padding-top: var(--space-3);
  border-top: 1px solid var(--color-border-light);
}

.evo-current-label {
  font-size: 0.72rem;
  color: var(--color-accent);
  font-weight: 500;
}

.evo-cancel-btn {
  padding: 4px 12px;
  font-size: 0.72rem;
  background: transparent;
  color: var(--color-text-secondary);
  border: 1px solid var(--color-border);
  border-radius: var(--radius-sm);
  cursor: pointer;
  transition: all var(--dur-fast) var(--ease);
}

.evo-cancel-btn:hover {
  border-color: var(--color-error);
  color: var(--color-error);
  background: var(--color-error-light);
}

/* ---- 子步骤过渡 ---- */
.evo-sub-enter-active {
  transition: all var(--dur) var(--ease);
}

.evo-sub-enter-from {
  opacity: 0;
  transform: translateY(-4px);
}

.evo-sub-leave-active {
  transition: all var(--dur-fast) var(--ease);
  position: absolute;
}

.evo-sub-leave-to {
  opacity: 0;
}
</style>
