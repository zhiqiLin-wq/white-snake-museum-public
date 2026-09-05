<script setup lang="ts">
import { computed, ref, watch, onUnmounted } from 'vue'
import { useAgentStore } from '@/stores/agent'
import type { StatusStep } from '@/types/sse-events'

const props = defineProps<{
  steps: StatusStep[]
  isStreaming: boolean
}>()

const emit = defineEmits<{
  'retry-step': [toolName: string]
  'skip-step': [toolName: string]
}>()

const agentStore = useAgentStore()

const isExpanded = ref(true)
const expandedStepTs = ref<number | null>(null)
// U-04: Elapsed timer
const elapsedSeconds = ref(0)
const autoCollapseTimer = ref<ReturnType<typeof setTimeout> | null>(null)
let elapsedInterval: ReturnType<typeof setInterval> | null = null

const hasSteps = computed(() => props.steps.length > 0)
const thinkingPlan = computed(() => agentStore.thinkingPlan)

// U-04: Start/stop elapsed timer based on thinking_start (via thinkingStartedAt)
watch(
  () => agentStore.thinkingStartedAt,
  (startedAt) => {
    if (startedAt) {
      elapsedSeconds.value = 0
      if (elapsedInterval) clearInterval(elapsedInterval)
      elapsedInterval = setInterval(() => {
        elapsedSeconds.value = Math.floor((Date.now() - startedAt) / 1000)
      }, 250)
    } else {
      if (elapsedInterval) {
        clearInterval(elapsedInterval)
        elapsedInterval = null
      }
    }
  },
  { immediate: true }
)

// U-04: Stop timer when streaming ends
watch(
  () => props.isStreaming,
  (streaming) => {
    if (!streaming) {
      if (elapsedInterval) {
        clearInterval(elapsedInterval)
        elapsedInterval = null
      }
    }
  }
)

// U-06: Auto-collapse 3s after streaming stops
watch(
  () => props.isStreaming,
  (streaming, wasStreaming) => {
    if (!streaming && wasStreaming) {
      autoCollapseTimer.value = setTimeout(() => {
        isExpanded.value = false
      }, 3000)
    }
    if (streaming) {
      if (autoCollapseTimer.value) {
        clearTimeout(autoCollapseTimer.value)
        autoCollapseTimer.value = null
      }
    }
  }
)

function onUserInteract() {
  // U-06: Cancel auto-collapse on user interaction
  if (autoCollapseTimer.value) {
    clearTimeout(autoCollapseTimer.value)
    autoCollapseTimer.value = null
  }
}

function toggleExpanded() {
  onUserInteract()
  isExpanded.value = !isExpanded.value
}

function toggleStepDetail(ts: number) {
  onUserInteract()
  expandedStepTs.value = expandedStepTs.value === ts ? null : ts
}

// U-03: Step count display
const completedCount = computed(() => props.steps.filter(s => s.done).length)
const toolSteps = computed(() => props.steps.filter(s => s.stepType === 'tool_result' || s.stepType === 'tool_call'))

// Track completed plan step names from the agent store
const completedPlanStepNames = computed(() => agentStore.completedPlanStepNames)

// Latest thinking_step message for inline display below plan skeleton
const currentThinkingMessage = computed(() => {
  const thinkingSteps = props.steps.filter(s => s.stepType === 'thinking_step')
  if (thinkingSteps.length === 0) return null
  return thinkingSteps[thinkingSteps.length - 1].message
})

// 展示用步骤：thinking_step 只保留最后一条（避免累积后多行堆叠），
// 其余类型（tool_call/tool_result/status）全部展示
const displaySteps = computed(() => {
  const lastThinkIdx = (() => {
    for (let i = props.steps.length - 1; i >= 0; i--) {
      if (props.steps[i].stepType === 'thinking_step') return i
    }
    return -1
  })()
  return props.steps.filter((s, i) =>
    s.stepType !== 'thinking_step' || i === lastThinkIdx
  )
})

function isPlanStepRunning(idx: number, stepName: string): boolean {
  if (!props.isStreaming) return false
  if (completedPlanStepNames.value.has(stepName)) return false
  // The first uncompleted step is "running"
  if (idx === 0) return completedPlanStepNames.value.size === 0
  const planSteps = thinkingPlan.value?.steps || []
  const prevStep = planSteps[idx - 1]
  return prevStep ? completedPlanStepNames.value.has(prevStep.name) : false
}

// U-04: Collapsed summary
const collapsedSummary = computed(() => {
  const totalToolSteps = toolSteps.value.length
  if (totalToolSteps === 0) return ''
  return `${completedCount.value}/${totalToolSteps} 已完成 - ${elapsedSeconds.value}s`
})

onUnmounted(() => {
  if (elapsedInterval) clearInterval(elapsedInterval)
  if (autoCollapseTimer.value) clearTimeout(autoCollapseTimer.value)
})
</script>

<template>
  <div v-if="hasSteps" class="thinking-steps">
    <!-- U-04: Summary bar with intent + timer -->
    <div v-if="isExpanded" class="ts-summary-bar">
      <span class="ts-summary-intent">意图: {{ agentStore.currentIntent || '分析中...' }}</span>
      <span class="ts-summary-timer">耗时: {{ elapsedSeconds }}s</span>
    </div>

    <button class="ts-header" @click="toggleExpanded">
      <span class="ts-header-icon">{{ isStreaming ? '思考中...' : '已完成' }}</span>
      <span class="ts-header-text">{{ isExpanded ? '思考过程' : collapsedSummary }}</span>
      <span class="ts-toggle">{{ isExpanded ? '收起' : '展开' }}</span>
    </button>

    <Transition name="ts-collapse">
      <div v-if="isExpanded" class="ts-list" @click="onUserInteract">
        <!-- U-02: Thinking plan skeleton -->
        <div v-if="thinkingPlan && thinkingPlan.steps.length > 0" class="ts-plan-skeleton">
          <div
            v-for="(planStep, idx) in thinkingPlan.steps"
            :key="planStep.name"
            class="ts-plan-step"
            :class="{
              'plan-done': completedPlanStepNames.has(planStep.name),
              'plan-running': isPlanStepRunning(idx, planStep.name),
            }"
          >
            <span class="ts-plan-icon">
              {{ completedPlanStepNames.has(planStep.name) ? 'OK' : (isPlanStepRunning(idx, planStep.name) ? '...' : 'wait') }}
            </span>
            <span class="ts-plan-label">{{ planStep.label }}</span>
          </div>
          <div v-if="currentThinkingMessage" class="ts-plan-message">{{ currentThinkingMessage }}</div>
        </div>

        <!-- U-03: Enhanced step cards -->
        <template v-for="(step, idx) in displaySteps" :key="step.ts + '_' + idx">
          <!-- U-17: Thinking step as lightweight animation -->
          <div
            v-if="step.stepType === 'thinking_step'"
            class="ts-thinking-animation"
          >
            <span class="ts-thinking-text">{{ step.message }}</span>
          </div>

          <!-- Tool call / Tool result steps -->
          <div
            v-else-if="step.stepType === 'tool_call' || step.stepType === 'tool_result'"
            class="ts-step-card"
            :class="{
              done: step.done,
              running: !step.done,
              error: step.status === 'error',
            }"
            @click="toggleStepDetail(step.ts)"
          >
            <div class="ts-step-card-header">
              <span class="ts-step-icon">
                <span v-if="step.status === 'error'">ERR</span>
                <span v-else-if="step.done">OK</span>
                <span v-else class="ts-spinner">...</span>
              </span>
              <span class="ts-step-badge">{{ step.tool || '' }}</span>
              <span class="ts-step-label">{{ step.label || step.message }}</span>
              <span class="ts-step-count">
                {{ thinkingPlan ? `${(step.stepIndex || 0) + 1}/${thinkingPlan.totalSteps}` : `#${(step.stepIndex || 0) + 1}` }}
              </span>
              <span v-if="step.durationMs" class="ts-step-duration">{{ (step.durationMs / 1000).toFixed(1) }}s</span>
            </div>

            <!-- Expanded detail -->
            <div v-if="expandedStepTs === step.ts && step.done" class="ts-step-detail">
              <div v-if="step.summary" class="ts-step-summary">{{ step.summary }}</div>
            </div>

            <!-- U-05: Error state with retry/skip buttons -->
            <div v-if="step.status === 'error'" class="ts-step-error">
              <span class="ts-error-msg">{{ step.errorMsg || '' }}</span>
              <button class="ts-retry-btn" @click.stop="emit('retry-step', step.tool || '')">重试</button>
              <button class="ts-skip-btn" @click.stop="emit('skip-step', step.tool || '')">跳过</button>
            </div>
          </div>

          <!-- Generic status step -->
          <div
            v-else
            class="ts-step"
            :class="{ done: step.done, running: !step.done }"
          >
            <span class="ts-step-icon">{{ step.done ? 'OK' : '...' }}</span>
            <span class="ts-step-text">{{ step.message }}</span>
          </div>
        </template>
      </div>
    </Transition>
  </div>
</template>

<style scoped>
.thinking-steps {
  margin: 6px 0;
  border: 1px solid var(--color-border);
  border-radius: var(--radius);
  overflow: hidden;
}

/* U-04: Summary bar */
.ts-summary-bar {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 4px 12px;
  font-size: 0.68rem;
  color: var(--color-text-secondary);
  background: var(--color-bg-secondary);
  border-bottom: 1px solid var(--color-border);
}

.ts-summary-intent {
  color: var(--color-accent);
}

.ts-summary-timer {
  font-family: var(--font-mono);
  font-size: 0.65rem;
}

/* Header */
.ts-header {
  display: flex;
  align-items: center;
  gap: 6px;
  width: 100%;
  padding: 7px 10px;
  background: var(--color-bg-secondary);
  border: none;
  color: var(--color-text-secondary);
  cursor: pointer;
  font-family: var(--font-sans);
  font-size: 0.78rem;
  transition: background-color var(--dur-fast) var(--ease), color var(--dur-fast) var(--ease);
}

.ts-header:hover {
  background: var(--color-bg-tertiary);
  color: var(--color-text-primary);
}

.ts-header-icon {
  color: var(--color-accent);
}

.ts-header-text {
  flex: 1;
  text-align: left;
}

.ts-toggle {
  font-size: 0.7rem;
  color: var(--color-text-tertiary);
}

.ts-list {
  padding: 4px 0;
}

/* U-02: Plan skeleton */
.ts-plan-skeleton {
  padding: 4px 12px 8px;
  display: flex;
  flex-wrap: wrap;
  gap: 4px;
}

.ts-plan-step {
  display: flex;
  align-items: center;
  gap: 3px;
  padding: 2px 8px;
  border-radius: var(--radius-sm);
  font-size: 0.65rem;
  background: var(--color-bg-tertiary);
  color: var(--color-text-secondary);
  border: 1px solid var(--color-border);
  transition: border-color var(--dur-fast) var(--ease), color var(--dur-fast) var(--ease);
}

.ts-plan-step.plan-running {
  border-color: var(--color-accent);
  color: var(--color-accent);
  background: var(--color-accent-light);
}

.ts-plan-step.plan-done {
  border-color: var(--color-verdict-supported);
  color: var(--color-verdict-supported);
}

.ts-plan-icon {
  font-family: var(--font-mono);
  font-size: 0.55rem;
}

.ts-plan-message {
  padding: 2px 8px 4px;
  font-size: 0.68rem;
  color: var(--color-accent);
  font-style: italic;
  animation: think-breathe 1.5s var(--ease) infinite;
}

/* U-17: Thinking step animation */
.ts-thinking-animation {
  padding: 2px 12px;
  font-size: 0.68rem;
  color: var(--color-text-secondary);
  animation: think-breathe 1.5s var(--ease) infinite;
}

@keyframes think-breathe {
  0%, 100% { opacity: 0.4; }
  50% { opacity: 1; }
}

.ts-thinking-text {
  font-style: italic;
}

/* U-03: Enhanced step cards */
.ts-step-card {
  margin: 2px 8px;
  padding: 6px 10px;
  border: 1px solid var(--color-border);
  border-radius: var(--radius-sm);
  cursor: pointer;
  transition: border-color var(--dur-fast) var(--ease), background-color var(--dur-fast) var(--ease);
}

.ts-step-card:hover {
  border-color: var(--color-accent);
  background: var(--color-bg-secondary);
}

.ts-step-card.running {
  border-left: 2px solid var(--color-accent);
}

.ts-step-card.done {
  border-left: 2px solid var(--color-verdict-supported);
}

.ts-step-card.error {
  border-left: 2px solid var(--color-verdict-refuted);
  background: var(--color-bg-tertiary);
}

.ts-step-card-header {
  display: flex;
  align-items: center;
  gap: 6px;
  font-size: 0.72rem;
}

.ts-step-badge {
  font-family: var(--font-mono);
  font-size: 0.6rem;
  padding: 1px 5px;
  background: var(--color-bg-tertiary);
  border-radius: var(--radius-sm);
  color: var(--color-text-secondary);
  white-space: nowrap;
  border: 1px solid var(--color-border-light);
}

.ts-step-label {
  flex: 1;
  color: var(--color-text-primary);
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.ts-step-count {
  font-family: var(--font-mono);
  font-size: 0.6rem;
  color: var(--color-text-tertiary);
}

.ts-step-duration {
  font-family: var(--font-mono);
  font-size: 0.6rem;
  color: var(--color-text-tertiary);
}

/* Spinner */
.ts-spinner {
  animation: ts-spin 1s linear infinite;
  display: inline-block;
  color: var(--color-accent);
}

@keyframes ts-spin {
  from { transform: rotate(0deg); }
  to { transform: rotate(360deg); }
}

/* Expanded detail */
.ts-step-detail {
  margin-top: 4px;
  padding: 6px 8px;
  background: var(--color-bg-tertiary);
  border-radius: var(--radius-sm);
}

.ts-step-summary {
  font-size: 0.68rem;
  color: var(--color-text-secondary);
  line-height: 1.4;
}

/* U-05: Error + retry */
.ts-step-error {
  display: flex;
  align-items: center;
  gap: 6px;
  margin-top: 4px;
  font-size: 0.68rem;
}

.ts-error-msg {
  flex: 1;
  color: var(--color-verdict-refuted);
}

.ts-retry-btn,
.ts-skip-btn {
  padding: 1px 8px;
  font-size: 0.62rem;
  border: 1px solid var(--color-border);
  border-radius: var(--radius-sm);
  cursor: pointer;
  background: var(--color-bg-tertiary);
  color: var(--color-text-primary);
  transition: background-color var(--dur-fast) var(--ease), border-color var(--dur-fast) var(--ease), color var(--dur-fast) var(--ease);
}

.ts-retry-btn:hover {
  background: var(--color-accent);
  color: #fff;
  border-color: var(--color-accent);
}

.ts-skip-btn:hover {
  background: var(--color-text-secondary);
  color: #fff;
  border-color: var(--color-text-secondary);
}

/* Generic step */
.ts-step {
  display: flex;
  align-items: flex-start;
  gap: 6px;
  padding: 4px 12px;
  font-size: 0.75rem;
}

.ts-step.done {
  color: var(--color-verdict-supported);
}

.ts-step.running {
  color: var(--color-accent);
}

.ts-step-icon {
  flex-shrink: 0;
  font-family: var(--font-mono);
  font-size: 0.65rem;
  margin-top: 1px;
}

/* Collapse transition — 与令牌一致的时长/缓动 */
.ts-collapse-enter-active,
.ts-collapse-leave-active {
  transition: max-height var(--dur) var(--ease), opacity var(--dur-fast) var(--ease);
}

.ts-collapse-enter-from,
.ts-collapse-leave-to {
  max-height: 0;
  opacity: 0;
}
</style>
