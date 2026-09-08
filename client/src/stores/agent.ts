import { defineStore } from 'pinia'
import { ref, computed } from 'vue'
import type { SourceCitation, ToolCallRecord, StatusStep, ThinkingPlanEvent } from '@/types/sse-events'
import { SSEEventBus } from '@/services/sseEventBus'
import { apiFetch } from '@/services/api'
// U-20: Pinia setup store imports -- lazy resolution, no circular dependency risk
import { useWorkspaceStore } from '@/stores/workspace'
import { useAnnotationStore } from '@/stores/annotations'
import { useSourceTreeStore } from '@/stores/sourceTree'

export interface EvolutionCard {
  trackingTarget: string
  summary: string
  narrative: string
  threadId: string
  workflow?: string  // "evolution" | "dynasty_compare"
}

// v19: 长报告卡片（完整回答分段写入文件，随消息持久化）
export interface ReportCard {
  reportId: string
  title: string
  topic?: string
  totalChars: number
  sectionCount: number
  sections: string[]
  createdAt?: string
  content: string  // 完整 Markdown 正文（用于下载 .md）
}

export interface AgentMessage {
  role: 'user' | 'assistant' | 'system'
  content: string
  sources?: SourceCitation[]
  toolCalls?: ToolCallRecord[]
  timestamp: number
  card?: EvolutionCard
  report?: ReportCard  // v19: 完整长报告下载卡片
  reasoning?: string  // v17: 思考过程全文，历史消息可展开查看
}

export interface ContextTag {
  id: string
  label: string
  data: Record<string, unknown>
  sourceType: 'text_selection' | 'chapter' | 'data_point' | 'comparison' | 'map_location' | 'annotation'
}

export interface Conversation {
  id: string
  title: string
  messages: AgentMessage[]
  contextTags: ContextTag[]
  createdAt: number
  updatedAt: number
}

export interface SkillSummary {
  name: string
  displayName: string
  description: string
  category: string
  icon: string
  triggers: string[]
}

function genId(): string {
  return crypto.randomUUID()
}

// ===== 智能标注实时进度状态机 =====
// 由后端 annotation_progress 事件驱动（prepare → discovery → resolution →
// marginalia → saving → done/error），前端进度面板据此渲染。
export type AnnotationPhase =
  | 'idle' | 'prepare' | 'discovery' | 'resolution'
  | 'marginalia' | 'saving' | 'done' | 'error'
export type AnnotationStatus = 'idle' | 'running' | 'done' | 'error' | 'cancelled'

export interface AnnotationFinalStats {
  entities: number
  paragraphs: number
  marginalia: number
  categoryCounts: Record<string, number>
}

export interface AnnotationProgressState {
  active: boolean
  status: AnnotationStatus
  phase: AnnotationPhase
  chapterNumber: number | null
  chapterTitle: string
  message: string
  totalParagraphs: number
  windowsDone: number
  windowsTotal: number
  candidates: number
  entities: number
  paragraphsDone: number
  marginaliaDone: number
  marginaliaTotal: number
  currentEntity: string | null
  lastParagraphIndex: number | null
  categoryCounts: Record<string, number>
  startedAt: number | null
  recent: Array<{ ts: number; message: string }>
  finalStats: AnnotationFinalStats | null
  errorMessage: string
}

const ANNOTATION_RECENT_MAX = 40

function createIdleAnnotationProgress(): AnnotationProgressState {
  return {
    active: false,
    status: 'idle',
    phase: 'idle',
    chapterNumber: null,
    chapterTitle: '',
    message: '',
    totalParagraphs: 0,
    windowsDone: 0,
    windowsTotal: 0,
    candidates: 0,
    entities: 0,
    paragraphsDone: 0,
    marginaliaDone: 0,
    marginaliaTotal: 0,
    currentEntity: null,
    lastParagraphIndex: null,
    categoryCounts: {},
    startedAt: null,
    recent: [],
    finalStats: null,
    errorMessage: '',
  }
}

export const useAgentStore = defineStore('agent', () => {
  // ===== Session management =====
  // Conversations are loaded from server (SQLite); runtime state only — no localStorage
  const conversations = ref<Conversation[]>([])
  const activeId = ref<string | null>(null)

  const activeConversation = computed(() =>
    conversations.value.find(c => c.id === activeId.value) || null
  )

  const messages = computed(() => activeConversation.value?.messages || [])

  // ===== SSE streaming state =====
  const isStreaming = ref(false)
  const isAnnotationLoading = ref(false)
  // 智能标注实时进度（annotation_progress 事件驱动）
  const annotationProgress = ref<AnnotationProgressState>(createIdleAnnotationProgress())
  const currentChunk = ref('')
  const currentSources = ref<SourceCitation[]>([])
  const currentToolCalls = ref<ToolCallRecord[]>([])
  const currentReport = ref<ReportCard | null>(null)  // v19: 长报告卡片
  const statusSteps = ref<StatusStep[]>([])
  const currentIntent = ref<string | null>(null)
  const errorMessage = ref<string | null>(null)
  // U-02: Thinking plan skeleton
  const thinkingPlan = ref<ThinkingPlanEvent | null>(null)
  // U-04: Thinking start timestamp for elapsed timer
  const thinkingStartedAt = ref<number | null>(null)
  // Track which plan steps have been completed (by step name from THINKING_STEP events)
  const completedPlanStepNames = ref<Set<string>>(new Set())
  // v17: ReAct 决策轮思考内容实时透传（thinking_delta 事件累积）
  const reasoningDelta = ref('')
  const reasoningActive = ref(false)

  // ===== Skills =====
  const skills = ref<SkillSummary[]>([])
  const selectedSkill = ref<string | null>(null)
  const skillsLoading = ref(false)

  // ===== Agent status =====
  const agentStatus = ref<'online' | 'offline' | 'starting' | 'unknown'>('unknown')

  // ===== Batch delete =====
  const batchMode = ref(false)
  const selectedIds = ref<Set<string>>(new Set())

  // ===== D4-2: Server sync (SQLite — single source of truth) =====

  // Track locally deleted IDs to prevent sync from re-adding them
  const deletedIds = ref<Set<string>>(new Set())
  let _initialSyncDone = false

  async function syncConversationsFromServer(): Promise<void> {
    const resp = await apiFetch('/api/conversations')
    if (!resp.ok) {
      throw new Error(`Failed to load conversations from server: HTTP ${resp.status}`)
    }
    const data = await resp.json() as { conversations: Array<{ id: string; title: string; createdAt: number; updatedAt: number }> }
    const serverConvs = data.conversations || []

    // First sync: if no server data, upload local conversations
    if (serverConvs.length === 0 && !_initialSyncDone) {
      for (const conv of conversations.value) {
        apiFetch('/api/conversations', {
          method: 'POST',
          body: JSON.stringify({ id: conv.id, title: conv.title }),
        }).catch(err => console.error('[agent] Failed to upload conversation:', err))
      }
      _initialSyncDone = true
      return
    }
    _initialSyncDone = true

    const serverIds = new Set(serverConvs.map(c => c.id))
    const localIds = new Set(conversations.value.map(c => c.id))

    // (a) Server has, local doesn't → add to local (but skip deleted ones)
    for (const sc of serverConvs) {
      if (!localIds.has(sc.id) && !deletedIds.value.has(sc.id)) {
        const conv: Conversation = {
          id: sc.id,
          title: sc.title,
          messages: [],
          contextTags: [],
          createdAt: sc.createdAt,
          updatedAt: sc.updatedAt,
        }
        conversations.value.push(conv)
        // Load messages for this conversation from server
        loadMessages(sc.id).catch(err => console.error('[agent] Failed to load messages for', sc.id, err))
      }
    }

    // (b) Local has, server doesn't → POST to server (only if not deleted)
    for (const conv of conversations.value) {
      if (!serverIds.has(conv.id) && !deletedIds.value.has(conv.id)) {
        apiFetch('/api/conversations', {
          method: 'POST',
          body: JSON.stringify({ id: conv.id, title: conv.title }),
        }).catch(err => console.error('[agent] Failed to upload conversation:', err))
      }
    }

    // (c) Both have → keep local messages, update title/updatedAt from server
    for (const sc of serverConvs) {
      const local = conversations.value.find(c => c.id === sc.id)
      if (local && local.messages.length > 0) {
        local.title = sc.title || local.title
        local.updatedAt = sc.updatedAt
      } else if (local && local.messages.length === 0) {
        // Local has no messages — try loading from server
        local.title = sc.title || local.title
        local.updatedAt = sc.updatedAt
        loadMessages(sc.id).catch(err => console.error('[agent] Failed to load messages for', sc.id, err))
      }
    }
  }

  // Load messages for a conversation from server
  async function loadMessages(conversationId: string): Promise<void> {
    const resp = await apiFetch(`/api/conversations/${conversationId}/messages`)
    if (!resp.ok) {
      if (resp.status === 404) return // No messages yet
      throw new Error(`Failed to load messages: HTTP ${resp.status}`)
    }
    const data = await resp.json() as { messages: Array<{
      id: string; conversationId: string; role: 'user' | 'assistant' | 'system'
      content: string; sources?: unknown; toolCalls?: unknown
      evolutionCard?: unknown; reportCard?: unknown; reasoning?: string | null; timestamp: number
    }> }
    const msgs = data.messages || []

    const local = conversations.value.find(c => c.id === conversationId)
    if (!local) return

    // Only load if local has no messages (server is source of truth after refresh)
    if (local.messages.length === 0 && msgs.length > 0) {
      local.messages = msgs.map(m => ({
        role: m.role,
        content: m.content,
        sources: m.sources as AgentMessage['sources'],
        toolCalls: m.toolCalls as AgentMessage['toolCalls'],
        timestamp: m.timestamp,
        card: m.evolutionCard as EvolutionCard | undefined,
        report: m.reportCard as ReportCard | undefined,
        reasoning: m.reasoning || undefined,
      }))
      local.updatedAt = msgs.length > 0 ? msgs[msgs.length - 1].timestamp : local.updatedAt
    }
  }

  // Persist a single message to server
  async function persistMessage(convId: string, msg: AgentMessage): Promise<void> {
    const msgId = genId()
    apiFetch(`/api/conversations/${convId}/messages`, {
      method: 'POST',
      body: JSON.stringify({
        messageId: msgId,
        role: msg.role,
        content: msg.content,
        sources: msg.sources ? JSON.stringify(msg.sources) : undefined,
        toolCalls: msg.toolCalls ? JSON.stringify(msg.toolCalls) : undefined,
        evolutionCard: msg.card ? JSON.stringify(msg.card) : undefined,
        reportCard: msg.report ? JSON.stringify(msg.report) : undefined,
        reasoning: msg.reasoning,
      }),
    }).catch(err => console.error('[agent] Failed to persist message:', err))
  }

  // ===== Conversation CRUD =====

  function ensureActiveConversation() {
    if (!activeId.value || !conversations.value.find(c => c.id === activeId.value)) {
      const conv: Conversation = {
        id: genId(),
        title: '',
        messages: [],
        contextTags: [],
        createdAt: Date.now(),
        updatedAt: Date.now(),
      }
      conversations.value.unshift(conv)
      activeId.value = conv.id
    }
  }

  function createConversation(): Conversation {
    const conv: Conversation = {
      id: genId(),
      title: '',
      messages: [],
      contextTags: [],
      createdAt: Date.now(),
      updatedAt: Date.now(),
    }
    conversations.value.unshift(conv)
    activeId.value = conv.id
    resetStreamState()
    // D4-2: Sync to server (SQLite)
    apiFetch('/api/conversations', { method: 'POST', body: JSON.stringify({ id: conv.id, title: conv.title }) }).catch(err => {
      console.error('[agent] Failed to create conversation on server:', err)
    })
    return conv
  }

  function deleteConversation(id: string) {
    const idx = conversations.value.findIndex(c => c.id === id)
    if (idx === -1) return
    const removed = conversations.value.splice(idx, 1)[0]
    if (activeId.value === id) {
      activeId.value = conversations.value[0]?.id || null
    }
    selectedIds.value.delete(id)
    // Track deleted ID to prevent sync from re-adding it
    const next = new Set(deletedIds.value)
    next.add(id)
    deletedIds.value = next
    // Sync delete to server (SQLite), rollback on failure
    apiFetch('/api/conversations/' + id, { method: 'DELETE' }).then(resp => {
      if (!resp.ok) {
        // Restore conversation on failure
        conversations.value.splice(idx, 0, removed)
        if (!activeId.value) activeId.value = id
        const restored = new Set(deletedIds.value)
        restored.delete(id)
        deletedIds.value = restored
        console.error('[agent] Failed to delete conversation on server: HTTP', resp.status)
      }
    }).catch(err => {
      // Restore conversation on network failure
      conversations.value.splice(idx, 0, removed)
      if (!activeId.value) activeId.value = id
      const restored = new Set(deletedIds.value)
      restored.delete(id)
      deletedIds.value = restored
      console.error('[agent] Failed to delete conversation on server:', err)
    })
  }

  function setActiveConversation(id: string) {
    activeId.value = id
    resetStreamState()
  }

  function toggleBatchMode() {
    batchMode.value = !batchMode.value
    selectedIds.value.clear()
  }

  function toggleSelectConversation(id: string) {
    if (selectedIds.value.has(id)) {
      selectedIds.value.delete(id)
    } else {
      selectedIds.value.add(id)
    }
  }

  function batchDelete() {
    if (selectedIds.value.size === 0) return
    const toDelete = new Set(selectedIds.value)
    conversations.value = conversations.value.filter(c => !toDelete.has(c.id))
    if (activeId.value && toDelete.has(activeId.value)) {
      activeId.value = conversations.value[0]?.id || null
    }
    // Track all deleted IDs to prevent sync from re-adding
    const next = new Set(deletedIds.value)
    for (const id of toDelete) {
      next.add(id)
      // Fire delete to server for each
      apiFetch('/api/conversations/' + id, { method: 'DELETE' }).catch(err => {
        console.error('[agent] Failed to delete conversation on server:', err)
      })
    }
    deletedIds.value = next
    selectedIds.value.clear()
    batchMode.value = false
  }

  // ===== Message management =====

  function addMessage(msg: AgentMessage) {
    const conv = activeConversation.value
    if (!conv) return
    conv.messages.push(msg)
    conv.updatedAt = Date.now()
    // Persist message to server (fire-and-forget)
    persistMessage(conv.id, msg)
  }

  // B-140:  ->
  function addEvolutionCard(card: EvolutionCard) {
    const conv = activeConversation.value
    if (!conv) return
    const label = card.workflow === 'dynasty_compare' ? '跨朝代对比' : '演化分析'
    const msg: AgentMessage = {
      role: 'assistant',
      content: `${label}${card.trackingTarget}\n\n${card.summary}`,
      timestamp: Date.now(),
      card,
    }
    conv.messages.push(msg)
    conv.updatedAt = Date.now()
    // Persist message to server (fire-and-forget)
    persistMessage(conv.id, msg)
  }

  // v19: 长报告就绪 —— 挂到当前流式 assistant 消息（finishStreaming 时落库）
  function setReportReady(report: ReportCard) {
    currentReport.value = report
    addStatusStep(`完整报告已生成：${report.title}（${report.sectionCount} 节）`)
  }

  // ===== SSE write methods =====

  function resetStreamState() {
    isStreaming.value = false
    currentChunk.value = ''
    currentSources.value = []
    currentToolCalls.value = []
    statusSteps.value = []
    currentIntent.value = null
    errorMessage.value = null
    thinkingPlan.value = null
    thinkingStartedAt.value = null
    completedPlanStepNames.value = new Set()
    reasoningDelta.value = ''
    reasoningActive.value = false
  }

  function setAnnotationLoading(loading: boolean) {
    isAnnotationLoading.value = loading
  }

  // ===== 智能标注实时进度状态机 actions =====

  function _pushAnnotationRecent(message: string) {
    const p = annotationProgress.value
    p.recent.push({ ts: Date.now(), message })
    if (p.recent.length > ANNOTATION_RECENT_MAX) {
      // 保留最新的 N 条（滚动窗口）
      p.recent.splice(0, p.recent.length - ANNOTATION_RECENT_MAX)
    }
  }

  /** 新一轮标注开始（prepare 事件） */
  function beginAnnotationProgress(
    chapterNumber: number,
    chapterTitle: string,
    totalParagraphs: number,
    message?: string,
  ) {
    const fresh = createIdleAnnotationProgress()
    annotationProgress.value = {
      ...fresh,
      active: true,
      status: 'running',
      phase: 'prepare',
      chapterNumber,
      chapterTitle: chapterTitle || `第${chapterNumber}章`,
      totalParagraphs: totalParagraphs || 0,
      startedAt: Date.now(),
      message: message || `第${chapterNumber}章 · 共 ${totalParagraphs} 段，准备标注流水线…`,
    }
    _pushAnnotationRecent(annotationProgress.value.message)
    isAnnotationLoading.value = true
  }

  /** 标注进行中任意阶段的增量更新（discovery/resolution/marginalia/saving） */
  function updateAnnotationProgress(
    data: Partial<Pick<AnnotationProgressState,
      'phase' | 'message' | 'totalParagraphs' | 'windowsDone' | 'windowsTotal'
      | 'candidates' | 'entities' | 'paragraphsDone' | 'marginaliaDone'
      | 'marginaliaTotal' | 'currentEntity' | 'lastParagraphIndex'
      | 'categoryCounts'>>,
  ) {
    const p = annotationProgress.value
    // 已结束的进度（done/error/cancelled）不再被迟到的 running 帧覆盖
    if (!p.active && data.phase && data.phase !== 'done' && data.phase !== 'error') return
    if (data.phase) p.phase = data.phase
    if (typeof data.totalParagraphs === 'number') p.totalParagraphs = data.totalParagraphs
    if (typeof data.windowsDone === 'number') p.windowsDone = data.windowsDone
    if (typeof data.windowsTotal === 'number') p.windowsTotal = data.windowsTotal
    if (typeof data.candidates === 'number') p.candidates = data.candidates
    if (typeof data.entities === 'number') p.entities = data.entities
    if (typeof data.paragraphsDone === 'number') p.paragraphsDone = data.paragraphsDone
    if (typeof data.marginaliaDone === 'number') p.marginaliaDone = data.marginaliaDone
    if (typeof data.marginaliaTotal === 'number') p.marginaliaTotal = data.marginaliaTotal
    if (data.currentEntity !== undefined) p.currentEntity = data.currentEntity
    if (data.lastParagraphIndex !== undefined) p.lastParagraphIndex = data.lastParagraphIndex
    if (data.categoryCounts) p.categoryCounts = { ...data.categoryCounts }
    if (data.message) {
      p.message = data.message
      _pushAnnotationRecent(data.message)
    }
  }

  /** 标注正常完成（done 事件）—— 封存真实统计供结果摘要展示 */
  function finishAnnotationProgress(extra?: {
    marginalia?: number
    finalStats?: AnnotationFinalStats | null
  }) {
    const p = annotationProgress.value
    const marginalia = extra?.marginalia ?? p.marginaliaDone
    p.finalStats = extra?.finalStats ?? {
      entities: p.entities,
      paragraphs: p.paragraphsDone,
      marginalia,
      categoryCounts: { ...p.categoryCounts },
    }
    p.status = 'done'
    p.phase = 'done'
    p.active = false
    p.marginaliaDone = marginalia
    p.message = p.message || '标注完成'
    isAnnotationLoading.value = false
  }

  /** 标注失败（error 事件） */
  function failAnnotationProgress(message: string) {
    const p = annotationProgress.value
    p.status = 'error'
    p.phase = 'error'
    p.active = false
    p.errorMessage = message
    p.message = message
    _pushAnnotationRecent(message)
    isAnnotationLoading.value = false
  }

  /** 用户主动停止（点击停止按钮 / abort） */
  function cancelAnnotationProgress() {
    const p = annotationProgress.value
    if (!p.active) return
    p.status = 'cancelled'
    p.active = false
    p.message = '已停止本次标注'
    _pushAnnotationRecent(p.message)
    isAnnotationLoading.value = false
  }

  function resetAnnotationProgress() {
    annotationProgress.value = createIdleAnnotationProgress()
  }

  function startStreaming() {
    isStreaming.value = true
    currentChunk.value = ''
    currentSources.value = []
    currentToolCalls.value = []
    currentReport.value = null
    statusSteps.value = []
    currentIntent.value = null
    errorMessage.value = null
    workflowCard.value = null
    thinkingPlan.value = null
    thinkingStartedAt.value = Date.now()
    completedPlanStepNames.value = new Set()
    reasoningDelta.value = ''
    reasoningActive.value = false
    // 新一轮对话开始：清空上一轮标注进度残留
    resetAnnotationProgress()
  }

  // v17: ReAct 决策轮思考增量累积（thinking_delta）
  function appendReasoning(delta: string) {
    if (!delta) return
    reasoningDelta.value += delta
    reasoningActive.value = true
  }

  // 其他内容事件到达（chunk/tool_call）→ 结束当前思考段并折叠
  function pauseReasoning() {
    if (reasoningActive.value) reasoningActive.value = false
  }

  // U-02: Set thinking plan skeleton
  function setThinkingPlan(plan: ThinkingPlanEvent) {
    thinkingPlan.value = plan
  }

  function appendChunk(text: string) {
    pauseReasoning()
    currentChunk.value += text
  }

  function addSource(source: SourceCitation) {
    currentSources.value.push(source)
  }

  function addToolCall(toolCall: ToolCallRecord) {
    pauseReasoning()
    currentToolCalls.value.push({ ...toolCall, status: 'running' })
    // U-03: Write enhanced status step for ThinkingSteps
    const stepIndex = statusSteps.value.length
    statusSteps.value.push({
      message: toolCall.tool as string,
      done: false,
      ts: Date.now(),
      tool: toolCall.tool as string,
      label: (toolCall.input as Record<string, unknown>)?.label as string || toolCall.tool as string,
      stepIndex,
      totalSteps: thinkingPlan.value?.totalSteps,
      stepType: 'tool_call',
      status: 'running',
    })
  }

  function updateToolResult(tool: string, resultPreview: string) {
    const existing = currentToolCalls.value.find(t => t.tool === tool)
    if (existing) {
      existing.status = 'completed'
      existing.resultPreview = resultPreview
    }
    // U-03: Update corresponding status step
    const step = statusSteps.value.find(s => s.tool === tool && s.stepType === 'tool_call')
    if (step) {
      step.done = true
      step.durationMs = Date.now() - step.ts
      step.summary = resultPreview.slice(0, 120)
      step.stepType = 'tool_result'
      step.status = 'completed'
    }
  }

  function addStatusStep(message: string) {
    const last = statusSteps.value[statusSteps.value.length - 1]
    if (last && last.message.startsWith(message.split(':')[0])) {
      last.message = message
      last.ts = Date.now()
    } else {
      statusSteps.value.push({
        message,
        done: false,
        ts: Date.now(),
        stepType: 'status',
      })
    }
  }

  function addThinkingStep(stepName: string, displayMessage?: string) {
    // 累积最近 thinking_step（保留滚动窗口 6 条），让用户在进度面板看到连续变化
    // 旧实现"替换最后一条"导致长时间无可见变化
    const display = displayMessage || stepName
    // 上一条若未完成且消息相同则刷新时间戳，避免重复堆叠相同消息
    const lastIdx = statusSteps.value.length - 1
    if (lastIdx >= 0 && statusSteps.value[lastIdx].stepType === 'thinking_step'
        && statusSteps.value[lastIdx].message === display) {
      statusSteps.value[lastIdx].ts = Date.now()
    } else {
      statusSteps.value.push({
        message: display,
        done: false,
        ts: Date.now(),
        stepType: 'thinking_step',
      })
    }
    // 保留最近 6 条 thinking_step，更早的裁剪（避免无限增长）
    const thinkIdxs: number[] = []
    statusSteps.value.forEach((s, i) => {
      if (s.stepType === 'thinking_step') thinkIdxs.push(i)
    })
    if (thinkIdxs.length > 6) {
      const removeCount = thinkIdxs.length - 6
      const toRemove = new Set(thinkIdxs.slice(0, removeCount))
      statusSteps.value = statusSteps.value.filter((_, i) => !toRemove.has(i))
    }
    // Mark the plan step as completed so the plan skeleton updates
    if (stepName) {
      const next = new Set(completedPlanStepNames.value)
      next.add(stepName)
      completedPlanStepNames.value = next
    }
  }

  function setIntent(intent: string) {
    currentIntent.value = intent
  }

  const workflowCard = ref<{ summary: string; resultRefs?: { label: string; type: string; data: Record<string, unknown> }[] } | null>(null)

  function setWorkflowCard(summary: string) {
    workflowCard.value = { summary }
  }

  function setWorkflowCardFull(card: { summary: string; resultRefs?: { label: string; type: string; data: Record<string, unknown> }[] }) {
    workflowCard.value = card
  }

  function finishStreaming() {
    // v17: 思考过程全文随消息保存（在 resetStreamState 清空前取出）
    const reasoningSnapshot = reasoningDelta.value.trim() || undefined
    const reportSnapshot = currentReport.value || undefined  // v19: 长报告卡片
    if (currentChunk.value || reportSnapshot) {
      const conv = activeConversation.value
      if (conv) {
        const assistantMsg: AgentMessage = {
          role: 'assistant',
          content: currentChunk.value,
          sources: [...currentSources.value],
          toolCalls: [...currentToolCalls.value],
          timestamp: Date.now(),
          reasoning: reasoningSnapshot,
          report: reportSnapshot,
        }
        conv.messages.push(assistantMsg)
        conv.updatedAt = Date.now()
        // Persist assistant message to server (fire-and-forget)
        persistMessage(conv.id, assistantMsg)
      }
      currentChunk.value = ''
      currentSources.value = []
      currentToolCalls.value = []
      currentReport.value = null
    }
    statusSteps.value.forEach(s => (s.done = true))
    isStreaming.value = false

    // D4-3: Sync conversation title to server after first round completes
    const conv = activeConversation.value
    if (conv && conv.title) {
      apiFetch('/api/conversations', {
        method: 'POST',
        body: JSON.stringify({ id: conv.id, title: conv.title }),
      }).catch(() => { /* fire-and-forget */ })
    }

    // State updated in-memory; server sync handled by individual API calls
  }

  function setError(message: string) {
    errorMessage.value = message
    isStreaming.value = false
  }

  // ===== Context tags =====

  function addContextTag(tag: ContextTag) {
    const conv = activeConversation.value
    if (!conv) return
    // Remove duplicate by sourceType + label
    const existingIdx = conv.contextTags.findIndex(
      t => t.sourceType === tag.sourceType && t.label === tag.label
    )
    if (existingIdx !== -1) {
      conv.contextTags.splice(existingIdx, 1)
    }
    conv.contextTags.push(tag)
    conv.updatedAt = Date.now()
    // State updated in-memory; server sync handled by individual API calls
  }

  function removeContextTag(tagId: string) {
    const conv = activeConversation.value
    if (!conv) return
    conv.contextTags = conv.contextTags.filter(t => t.id !== tagId)
    conv.updatedAt = Date.now()
    // State updated in-memory; server sync handled by individual API calls
  }

  function clearContextTags() {
    const conv = activeConversation.value
    if (!conv) return
    conv.contextTags = []
    conv.updatedAt = Date.now()
    // State updated in-memory; server sync handled by individual API calls
  }

  // ===== Skill management =====

  function selectSkill(name: string | null) {
    selectedSkill.value = selectedSkill.value === name ? null : name
  }

  async function fetchSkills() {
    skillsLoading.value = true
    try {
      const resp = await apiFetch('/api/agent/skills')
      if (resp.ok) {
        const data = await resp.json()
        // B-153: 后端返回 snake_case(display_name)，前端接口是 camelCase(displayName)。
        // 不映射会导致技能 chip 回退显示原始名（如 "leifeng_culture"）而非中文名。
        skills.value = (data.skills || []).map((s: Record<string, unknown>) => ({
          ...s,
          displayName: s.displayName ?? s.display_name ?? '',
        })) as SkillSummary[]
      }
    } catch {
      // Silent failure
    } finally {
      skillsLoading.value = false
    }
  }

  async function checkHealth() {
    try {
      const resp = await apiFetch('/api/agent/health')
      if (resp.ok) {
        const data = await resp.json()
        agentStatus.value = data.status === 'ok' ? 'online'
          : data.status === 'starting' ? 'starting'
          : 'online'
      }
    } catch {
      agentStatus.value = 'offline'
    }
  }

  // ===== Send message (delegates to SSEEventBus) =====

  async function sendMessage(query: string): Promise<void> {
    if (!query.trim() || isStreaming.value) return

    ensureActiveConversation()
    const conv = activeConversation.value
    if (!conv) return

    // Update title based on first message
    if (conv.messages.length === 0) {
      conv.title = query.slice(0, 30) + (query.length > 30 ? '...' : '')
    }

    // Add user message
    const userMsg: AgentMessage = {
      role: 'user',
      content: query,
      timestamp: Date.now(),
    }
    conv.messages.push(userMsg)
    conv.updatedAt = Date.now()

    // Persist user message to server (fire-and-forget, non-blocking)
    if (activeId.value) {
      persistMessage(activeId.value, userMsg)
    }

    startStreaming()

    const body: Record<string, unknown> = {
      query,
      stream: true,
      // Use raw UUID v4 as conversation_id so Python agent uses the correct thread_id
      conversation_id: activeId.value || 'stream',
    }

    if (selectedSkill.value) {
      body.skill = selectedSkill.value
      // B-153: 技能单次生效 — 发送后自动清除，避免"雷峰塔技能"一直挂着
      // 污染后续无关提问（用户感知为"发什么内容都跟这个 chip 无关"）。
      selectedSkill.value = null
    }

    // CU-1.3: Token-aware history passing — send full history trimmed by token budget
    const filteredHistory = conv.messages
      .slice(0, -1)
      .filter(m => m.role !== 'system')

    function estimateTokens(content: string): number {
      const chineseChars = (content.match(/[-]/g) || []).length
      const otherChars = content.length - chineseChars
      return Math.ceil(chineseChars / 1.5 + otherChars / 4)
    }

    const TOKEN_BUDGET = 60000
    const priorityMsgs = filteredHistory.filter(m => m.toolCalls && m.toolCalls.length > 0)
    const normalMsgs = filteredHistory.filter(m => !m.toolCalls || m.toolCalls.length === 0)
    const selected: AgentMessage[] = []
    let totalTokens = 0

    // Phase 1: always include priority messages (with tool calls)
    for (const pm of priorityMsgs) {
      selected.push(pm)
      totalTokens += estimateTokens(pm.content)
    }

    // Phase 2: add normal messages from most recent to oldest, stop when over budget
    for (let i = normalMsgs.length - 1; i >= 0; i--) {
      const msg = normalMsgs[i]
      const msgTokens = estimateTokens(msg.content)
      if (totalTokens + msgTokens > TOKEN_BUDGET) break
      selected.push(msg)
      totalTokens += msgTokens
    }

    selected.sort((a, b) => a.timestamp - b.timestamp)
    const recentHistory = selected.map(m => ({ role: m.role, content: m.content }))
    if (recentHistory.length > 0) {
      body.history = recentHistory
    }

    // Add context tags (conversation-level + auto-detected workspace chapter)
    const workspaceStore = useWorkspaceStore()
    const autoContextTags: Record<string, unknown>[] = []

    // CU-annotation: Auto-inject active workspace chapter as context
    // so the Agent knows which document the user is viewing when they say ""
    const activeTab = workspaceStore.activeTab
    const activeChapterNumber = (activeTab?.data as Record<string, unknown>)?.chapterNumber as number | undefined
    const activeChapterTitle = (activeTab?.data as Record<string, unknown>)?.chapterTitle as string | undefined
    const activeParagraphIndex = (activeTab?.data as Record<string, unknown>)?.paragraphIndex as number | undefined
    if (activeChapterNumber !== undefined) {
      const tagData: Record<string, unknown> = {
        chapterNumber: activeChapterNumber,
        chapterTitle: activeChapterTitle,
      }
      // CU-08:  Agent ""
      if (activeParagraphIndex !== undefined) {
        tagData.paragraphIndex = activeParagraphIndex
      }
      autoContextTags.push({
        source_type: 'chapter',
        label: activeChapterTitle || `${activeChapterNumber}`,
        data: tagData,
      })
    }

    const allContextTags = [
      ...conv.contextTags.map(t => ({
        source_type: t.sourceType,
        label: t.label,
        data: t.data,
      })),
      ...autoContextTags,
    ]
    if (allContextTags.length > 0) {
      body.context_tags = allContextTags
    }

    // Pass split side when in split mode
    if (workspaceStore.split.enabled) {
      body.split_side = workspaceStore.activeSplitSide || 'left'
    }

    // 阶段 2 — P2-05: 标注类请求附带段落列表
    if (activeChapterNumber !== undefined) {
      const sourceTreeStore = useSourceTreeStore()
      const paragraphs = sourceTreeStore.getParagraphList(activeChapterNumber)
      if (paragraphs && paragraphs.length > 0) {
        body.paragraphs = paragraphs
      }
    }

    // Pass existing user annotation IDs to avoid duplicate annotations
    const annotationStore = useAnnotationStore()
    if (activeTab) {
      const chapterNumber = (activeTab.data as Record<string, unknown>)?.chapterNumber as number | undefined
      if (chapterNumber !== undefined) {
        const existingUserIds: string[] = []
        const chapterData = annotationStore.getAnnotationsByChapter(chapterNumber)
        for (const item of chapterData) {
          for (const ann of item.annotations) {
            existingUserIds.push(ann.id)
          }
        }
        if (existingUserIds.length > 0) {
          body.existing_user_annotations = existingUserIds
        }

        // U-08: Pass existing agent annotations as full objects to avoid re-annotating
        const existingAgentAnns: { id: string; span: { startChar: number; endChar: number }; category: string; color: string }[] = []
        const prefix = `${chapterNumber}:`
        for (const [key, anns] of Object.entries(annotationStore.agentAnnotations)) {
          if (key.startsWith(prefix)) {
            for (const ann of anns) {
              existingAgentAnns.push({
                id: ann.id,
                span: ann.span,
                category: ann.category,
                color: ann.color,
              })
            }
          }
        }
        if (existingAgentAnns.length > 0) {
          body.existing_agent_annotations = existingAgentAnns
        }
      }
    }

    try {
      const eventBus = SSEEventBus.getInstance()
      await eventBus.startConnection('/api/agent/chat', body)
    } catch (err: unknown) {
      const message = err instanceof Error ? err.message : ''
      setError(message)
      const errorMsg: AgentMessage = {
        role: 'system',
        content: `AI 服务异常: ${message}`,
        timestamp: Date.now(),
      }
      if (conv) {
        conv.messages.push(errorMsg)
        conv.updatedAt = Date.now()
      }
      // State updated in-memory; server sync handled by individual API calls
    }
  }

  async function resumeEvolution(body: Record<string, unknown>): Promise<void> {
    startStreaming()
    try {
      const eventBus = SSEEventBus.getInstance()
      await eventBus.startConnection('/api/agent/chat/resume', body)
    } catch (err: unknown) {
      const message = err instanceof Error ? err.message : ''
      if (message.includes('410')) {
        setError('')
        const conv = activeConversation.value
        if (conv) {
          conv.messages.push({
            role: 'system',
            content: '该中断已超时(超过10分钟),请重新发起演化分析。',
            timestamp: Date.now(),
          })
          conv.updatedAt = Date.now()
          // State updated in-memory; server sync handled by individual API calls
        }
      } else if (message.includes('404')) {
        setError('')
      } else {
        setError(message)
      }
    }
  }

  function clearChat() {
    const conv = activeConversation.value
    if (conv) {
      conv.messages = []
      conv.contextTags = []
      conv.updatedAt = Date.now()
    }
    resetStreamState()
    // State updated in-memory; server sync handled by individual API calls
  }

  return {
    // State
    conversations,
    activeId,
    activeConversation,
    messages,
    isStreaming,
    isAnnotationLoading,
    annotationProgress,
    currentChunk,
    currentSources,
    currentToolCalls,
    statusSteps,
    thinkingPlan,
    thinkingStartedAt,
    completedPlanStepNames,
    currentIntent,
    errorMessage,
    reasoningDelta,
    reasoningActive,
    skills,
    selectedSkill,
    skillsLoading,
    agentStatus,
    batchMode,
    selectedIds,

    // Session CRUD
    createConversation,
    deleteConversation,
    setActiveConversation,
    ensureActiveConversation,
    toggleBatchMode,
    toggleSelectConversation,
    batchDelete,

    // Messages
    addMessage,
    addEvolutionCard,
    setReportReady,

    // SSE write
    startStreaming,
    setAnnotationLoading,
    beginAnnotationProgress,
    updateAnnotationProgress,
    finishAnnotationProgress,
    failAnnotationProgress,
    cancelAnnotationProgress,
    resetAnnotationProgress,
    appendChunk,
    appendReasoning,
    addSource,
    addToolCall,
    updateToolResult,
    addStatusStep,
    addThinkingStep,
    setThinkingPlan,
    setIntent,
    workflowCard,
    setWorkflowCard,
    setWorkflowCardFull,
    finishStreaming,
    setError,
    resetStreamState,

    // Context tags
    addContextTag,
    removeContextTag,
    clearContextTags,

    // Skills
    selectSkill,
    fetchSkills,
    checkHealth,

    // Messaging
    sendMessage,
    resumeEvolution,
    clearChat,

    // D4-2: Server sync (SQLite — single source of truth)
    syncConversationsFromServer,
    loadMessages,
    persistMessage,
  }
})
