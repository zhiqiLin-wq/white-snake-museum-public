import { defineStore } from 'pinia'
import { ref, computed } from 'vue'
import { apiFetch } from '@/services/api'
import type {
  EvolutionPhase,
  InterruptType,
  Motif,
  CandidateHypothesis,
  ClaimVerdict,
  PhylogenyNode,
  PhylogenyEdge,
  MotifMatrixCell,
  DecisionOption,
  ResumePayload,
  EvidenceRefData,
} from '@/types/evolution'

/*
 * State transition rules:
 *   idle -> E1_decompose: initEvolution() called
 *   E1_decompose -> E1_interrupt: SSE evolution_interrupt (confirm_motifs)
 *   E1_interrupt -> E2_scan: submitResume() succeeds + SSE stream resumes
 *   E2_scan -> E3_classify: auto
 *   E3_classify -> E4_phylogeny: auto
 *   E4_phylogeny -> E5_correlate: auto
 *   E5_correlate -> E6_hypothesize: auto
 *   E6_hypothesize -> E6_interrupt: SSE evolution_interrupt (select_hypothesis)
 *   E6_interrupt -> E7_test: submitResume() succeeds
 *   E7_test -> E7_interrupt: SSE evolution_interrupt (decide_action)
 *   E7_interrupt -> E8_synthesize: submitResume() succeeds
 *   E8_synthesize -> complete: auto
 *   Any phase -> error: SSE error event
 */

// keyed by threadId
interface CompletedAnalysis {
  threadId: string
  trackingTarget: string
  phase: 'complete' | 'E8_synthesize'
  narrative: string
  evidenceRefs: Record<string, EvidenceRefData>
  motifMatrix: { rows: string[]; columns: string[]; cells: MotifMatrixCell[] }
  phylogeny: { nodes: PhylogenyNode[]; edges: PhylogenyEdge[] }
  claimVerdicts: ClaimVerdict[]
  motifs: Motif[]
  savedAt: number
}

export const useEvolutionStore = defineStore('evolution', () => {
  // ===== Core state =====
  const phase = ref<EvolutionPhase>('idle')
  const threadId = ref('')
  const interruptId = ref('')
  const interruptInstance = ref(0)
  const interruptType = ref<InterruptType | null>(null)
  const trackingTarget = ref('')

  // ===== Motif data =====
  const motifs = ref<Motif[]>([])
  const suggestedAdditions = ref<{ id: string; name: string; type: 'invariant' | 'variable'; core: string }[]>([])

  // ===== Hypothesis data =====
  const candidateHypotheses = ref<CandidateHypothesis[]>([])
  const allowMultiSelect = ref(false)

  // ===== Evidence data =====
  const claimVerdicts = ref<ClaimVerdict[]>([])

  // ===== Visualization data =====
  const phylogeny = ref<{ nodes: PhylogenyNode[]; edges: PhylogenyEdge[] }>({ nodes: [], edges: [] })
  const motifMatrix = ref<{ rows: string[]; columns: string[]; cells: MotifMatrixCell[] }>({
    rows: [],
    columns: [],
    cells: [],
  })
  const narrative = ref('')
  const evidenceRefs = ref<Record<string, EvidenceRefData>>({})

  // ===== Decision data =====
  const decisionOptions = ref<DecisionOption[]>([])
  const decisionSummary = ref('')
  const totalClaims = ref(0)
  const insufficientCount = ref(0)

  // ===== Multi-analysis support =====
  const completedAnalyses = ref<Record<string, CompletedAnalysis>>({})

  // ===== Getters =====

  const isAwaitingUserInput = computed(() =>
    phase.value === 'E1_interrupt' || phase.value === 'E6_interrupt' || phase.value === 'E7_interrupt'
  )

  const currentInterruptType = computed(() => interruptType.value)

  const canResume = computed(() => {
    if (!isAwaitingUserInput.value) return false
    if (interruptType.value === 'confirm_motifs') {
      // Can resume if motifs have been confirmed (any edit or explicit confirm)
      return true
    }
    if (interruptType.value === 'select_hypothesis') {
      return selectedHypotheses.value.length > 0
    }
    if (interruptType.value === 'decide_next_action') {
      return selectedDecisionOption.value !== null
    }
    return false
  })

  const selectedHypotheses = computed(() =>
    candidateHypotheses.value.filter(h => (h as unknown as { selected?: boolean }).selected)
  )

  const selectedDecisionOption = ref<string | null>(null)

  // ===== Actions: Initialization =====

  function initEvolution(tId: string, target: string) {
    threadId.value = tId
    trackingTarget.value = target
    phase.value = 'E1_decompose'
    interruptType.value = null
    motifs.value = []
    suggestedAdditions.value = []
    candidateHypotheses.value = []
    claimVerdicts.value = []
    phylogeny.value = { nodes: [], edges: [] }
    motifMatrix.value = { rows: [], columns: [], cells: [] }
    narrative.value = ''
    evidenceRefs.value = {}
    decisionOptions.value = []
    selectedDecisionOption.value = null
    clearPersistedState()
    // 只清旧版单例存档 key。多分析存档（PERSIST_COMPLETED_KEY）必须保留：
    // 否则每次开始新分析都会清空全部历史快照，旧分析卡片打开后内容为空
    try {
      localStorage.removeItem(PERSIST_COMPLETE_KEY)
    } catch { /* localStorage 不可用时忽略 */ }
  }

  function setPhase(newPhase: EvolutionPhase) {
    phase.value = newPhase
    if (newPhase === 'complete' || newPhase === 'E8_synthesize') {
      snapshotCompletedAnalysis()
      persistAllCompletedAnalyses()
    }
  }

  function resetLiveState() {
    phase.value = 'idle'
    threadId.value = ''
    interruptId.value = ''
    interruptInstance.value = 0
    interruptType.value = null
    trackingTarget.value = ''
    motifs.value = []
    suggestedAdditions.value = []
    candidateHypotheses.value = []
    allowMultiSelect.value = false
    claimVerdicts.value = []
    phylogeny.value = { nodes: [], edges: [] }
    motifMatrix.value = { rows: [], columns: [], cells: [] }
    narrative.value = ''
    evidenceRefs.value = {}
    decisionOptions.value = []
    decisionSummary.value = ''
    totalClaims.value = 0
    insufficientCount.value = 0
    selectedDecisionOption.value = null
    clearPersistedState()
  }

  /** 取消分析时调用：只停止运行态标记，保留已分析出的矩阵/叙述/证据数据 */
  function stopLiveState() {
    interruptId.value = ''
    interruptInstance.value = 0
    interruptType.value = null
    // isAwaitingUserInput 是 computed，清掉 interruptType 后会自动变为 false
    clearPersistedState()
  }

  function reset() {
    resetLiveState()
    completedAnalyses.value = {}
    clearCompleteState()
  }

  // ===== Interrupt state persistence (for page refresh recovery) =====

  const PERSIST_KEY = 'evolution_interrupt_state'
  const PERSIST_COMPLETE_KEY = 'evolution_complete_state'

  interface PersistedState {
    threadId: string
    interruptId: string
    interruptType: InterruptType
    trackingTarget: string
    phase: EvolutionPhase
    motifs: Motif[]
    suggestedAdditions: { id: string; name: string; type: 'invariant' | 'variable'; core: string }[]
    candidateHypotheses: CandidateHypothesis[]
    claimVerdicts: ClaimVerdict[]
    decisionOptions: DecisionOption[]
    decisionSummary: string
    totalClaims: number
    insufficientCount: number
    savedAt: number
  }

  function persistInterruptState() {
    if (!isAwaitingUserInput.value) return
    const state: PersistedState = {
      threadId: threadId.value,
      interruptId: interruptId.value,
      interruptType: interruptType.value!,
      trackingTarget: trackingTarget.value,
      phase: phase.value,
      motifs: JSON.parse(JSON.stringify(motifs.value)),
      suggestedAdditions: JSON.parse(JSON.stringify(suggestedAdditions.value)),
      candidateHypotheses: JSON.parse(JSON.stringify(candidateHypotheses.value)),
      claimVerdicts: JSON.parse(JSON.stringify(claimVerdicts.value)),
      decisionOptions: JSON.parse(JSON.stringify(decisionOptions.value)),
      decisionSummary: decisionSummary.value,
      totalClaims: totalClaims.value,
      insufficientCount: insufficientCount.value,
      savedAt: Date.now(),
    }
    try {
      localStorage.setItem(PERSIST_KEY, JSON.stringify(state))
    } catch { /* localStorage  */ }
  }

  function restoreInterruptState(): PersistedState | null {
    try {
      const raw = localStorage.getItem(PERSIST_KEY)
      if (!raw) return null
      const state = JSON.parse(raw) as PersistedState
      if (Date.now() - state.savedAt > 2 * 60 * 60 * 1000) {
        localStorage.removeItem(PERSIST_KEY)
        return null
      }
      return state
    } catch {
      localStorage.removeItem(PERSIST_KEY)
      return null
    }
  }

  function applyPersistedState(state: PersistedState) {
    threadId.value = state.threadId
    interruptId.value = state.interruptId
    interruptType.value = state.interruptType
    trackingTarget.value = state.trackingTarget
    phase.value = state.phase
    motifs.value = state.motifs
    suggestedAdditions.value = state.suggestedAdditions
    candidateHypotheses.value = state.candidateHypotheses
    claimVerdicts.value = state.claimVerdicts
    decisionOptions.value = state.decisionOptions
    decisionSummary.value = state.decisionSummary
    totalClaims.value = state.totalClaims
    insufficientCount.value = state.insufficientCount
    interruptInstance.value++
  }

  function clearPersistedState() {
    try {
      localStorage.removeItem(PERSIST_KEY)
    } catch { /*  */ }
  }

  // ===== Multi-analysis: snapshot, query, persistence =====

  function snapshotCompletedAnalysis() {
    const tid = threadId.value
    if (!tid) return
    if (phase.value !== 'complete' && phase.value !== 'E8_synthesize') return

    // C12: 去除 excerptFull 以减少 localStorage 占用
    const cleanedRefs: Record<string, EvidenceRefData> = {}
    for (const [key, ref] of Object.entries(evidenceRefs.value)) {
      const { excerptFull, ...rest } = ref
      cleanedRefs[key] = rest as EvidenceRefData
    }

    const snapshot: CompletedAnalysis = {
      threadId: tid,
      trackingTarget: trackingTarget.value,
      phase: phase.value as 'complete' | 'E8_synthesize',
      narrative: narrative.value,
      evidenceRefs: cleanedRefs,
      motifMatrix: JSON.parse(JSON.stringify(motifMatrix.value)),
      phylogeny: JSON.parse(JSON.stringify(phylogeny.value)),
      claimVerdicts: JSON.parse(JSON.stringify(claimVerdicts.value)),
      motifs: JSON.parse(JSON.stringify(motifs.value)),
      savedAt: Date.now(),
    }
    completedAnalyses.value = { ...completedAnalyses.value, [tid]: snapshot }
  }

  function getAnalysis(tid: string): CompletedAnalysis | null {
    if (!tid) return null
    //  ref 
    if (tid === threadId.value && (phase.value === 'complete' || phase.value === 'E8_synthesize')) {
      return {
        threadId: threadId.value,
        trackingTarget: trackingTarget.value,
        phase: phase.value as 'complete' | 'E8_synthesize',
        narrative: narrative.value,
        evidenceRefs: JSON.parse(JSON.stringify(evidenceRefs.value)),
        motifMatrix: JSON.parse(JSON.stringify(motifMatrix.value)),
        phylogeny: JSON.parse(JSON.stringify(phylogeny.value)),
        claimVerdicts: JSON.parse(JSON.stringify(claimVerdicts.value)),
        motifs: JSON.parse(JSON.stringify(motifs.value)),
        savedAt: Date.now(),
      }
    }
    return completedAnalyses.value[tid] || null
  }

  function isActiveAnalysis(tid: string): boolean {
    return tid === threadId.value && threadId.value !== ''
      && phase.value !== 'complete' && phase.value !== 'E8_synthesize'
  }

  function removeCompletedAnalysis(tid: string) {
    if (!tid) return
    const updated = { ...completedAnalyses.value }
    delete updated[tid]
    completedAnalyses.value = updated
    persistAllCompletedAnalyses()
  }

  const PERSIST_COMPLETED_KEY = 'evolution_completed_analyses'

  function persistAllCompletedAnalyses() {
    // 
    if (phase.value === 'complete' || phase.value === 'E8_synthesize') {
      snapshotCompletedAnalysis()
    }
    const entries = Object.values(completedAnalyses.value)
    if (entries.length === 0) return
    try {
      localStorage.setItem(PERSIST_COMPLETED_KEY, JSON.stringify(entries))
    } catch { /*  */ }
  }

  function restoreAllCompletedAnalyses(): CompletedAnalysis[] {
    try {
      const raw = localStorage.getItem(PERSIST_COMPLETED_KEY)
      if (!raw) {
        //  evolution_complete_state
        const oldRaw = localStorage.getItem(PERSIST_COMPLETE_KEY)
        if (oldRaw) {
          const oldState = JSON.parse(oldRaw)
          if (Date.now() - oldState.savedAt <= 24 * 60 * 60 * 1000) {
            const restored: CompletedAnalysis = {
              threadId: oldState.threadId || oldState.trackingTarget || 'legacy',
              trackingTarget: oldState.trackingTarget || '',
              phase: oldState.phase || 'complete',
              narrative: oldState.narrative || '',
              evidenceRefs: oldState.evidenceRefs || {},
              motifMatrix: oldState.motifMatrix || { rows: [], columns: [], cells: [] },
              phylogeny: oldState.phylogeny || { nodes: [], edges: [] },
              claimVerdicts: oldState.claimVerdicts || [],
              motifs: oldState.motifs || [],
              savedAt: oldState.savedAt,
            }
            localStorage.removeItem(PERSIST_COMPLETE_KEY)
            return [restored]
          }
          localStorage.removeItem(PERSIST_COMPLETE_KEY)
        }
        return []
      }
      const entries = JSON.parse(raw) as CompletedAnalysis[]
      const now = Date.now()
      // 保留 7 天内的快照，最多 30 条（超出丢弃最旧的）
      const RETENTION_MS = 7 * 24 * 60 * 60 * 1000
      const MAX_ENTRIES = 30
      const valid = entries
        .filter(e => now - e.savedAt <= RETENTION_MS)
        .sort((a, b) => b.savedAt - a.savedAt)
        .slice(0, MAX_ENTRIES)
      completedAnalyses.value = {}
      for (const entry of valid) {
        completedAnalyses.value[entry.threadId] = entry
      }
      return valid
    } catch {
      localStorage.removeItem(PERSIST_COMPLETED_KEY)
      return []
    }
  }

  //  API 
  function persistCompleteState() {
    persistAllCompletedAnalyses()
  }

  // ===== Server-side recovery: 从 SQLite checkpoint 恢复历史分析 =====

  /** 正在/已尝试从服务端恢复的 threadId，避免重复请求 */
  const serverRestoreAttempts = new Set<string>()

  /** 本地快照缺失时，从后端 SQLite checkpoint 拉取历史分析并写入 completedAnalyses */
  async function loadAnalysisFromServer(tid: string): Promise<CompletedAnalysis | null> {
    if (!tid) return null
    if (completedAnalyses.value[tid]) return completedAnalyses.value[tid]
    if (serverRestoreAttempts.has(tid)) return null
    serverRestoreAttempts.add(tid)
    try {
      const resp = await apiFetch(`/api/agent/chat/evolution-state?thread_id=${encodeURIComponent(tid)}`)
      if (!resp.ok) {
        console.warn(`[evolution] 服务端无历史分析 thread=${tid} (${resp.status})`)
        return null
      }
      const data = await resp.json() as Record<string, unknown>
      // 中断态 phase 统一按完成态展示（历史卡片只做展示）
      const phaseRaw = (data.currentPhase as string) || 'complete'
      const phase: CompletedAnalysis['phase'] =
        phaseRaw === 'E8_synthesize' || phaseRaw === 'complete' ? phaseRaw : 'complete'
      const analysis: CompletedAnalysis = {
        threadId: (data.threadId as string) || tid,
        trackingTarget: (data.trackingTarget as string) || '',
        phase,
        narrative: (data.narrative as string) || '',
        evidenceRefs: (data.evidenceRefs as CompletedAnalysis['evidenceRefs']) || {},
        motifMatrix: (data.motifMatrix as CompletedAnalysis['motifMatrix']) || { rows: [], columns: [], cells: [] },
        phylogeny: (data.phylogeny as CompletedAnalysis['phylogeny']) || { nodes: [], edges: [] },
        claimVerdicts: (data.claimVerdicts as CompletedAnalysis['claimVerdicts']) || [],
        motifs: [],
        savedAt: Date.now(),
      }
      completedAnalyses.value = { ...completedAnalyses.value, [tid]: analysis }
      persistAllCompletedAnalyses()
      console.log(`[evolution] 从服务端恢复历史分析 thread=${tid} cells=${analysis.motifMatrix.cells.length}`)
      return analysis
    } catch (err) {
      console.warn(`[evolution] 服务端恢复失败 thread=${tid}:`, err)
      return null
    }
  }

  function restoreCompleteState(): boolean {
    const list = restoreAllCompletedAnalyses()
    if (list.length === 0) return false
    //  ref
    const latest = list.reduce((a, b) => a.savedAt > b.savedAt ? a : b)
    trackingTarget.value = latest.trackingTarget
    phase.value = latest.phase
    narrative.value = latest.narrative
    evidenceRefs.value = latest.evidenceRefs
    motifMatrix.value = latest.motifMatrix
    phylogeny.value = latest.phylogeny
    claimVerdicts.value = latest.claimVerdicts
    motifs.value = latest.motifs
    threadId.value = latest.threadId
    return true
  }

  function clearCompleteState() {
    try {
      localStorage.removeItem(PERSIST_COMPLETE_KEY)
      localStorage.removeItem(PERSIST_COMPLETED_KEY)
    } catch { /*  */ }
  }

  // ===== Actions: SSE event handlers =====

  function handleEvolutionInterrupt(payload: {
    threadId: string
    interruptId: string
    interruptType: InterruptType
    data: Record<string, unknown>
  }) {
    threadId.value = payload.threadId
    interruptId.value = payload.interruptId
    interruptType.value = payload.interruptType

    switch (payload.interruptType) {
      case 'confirm_motifs': {
        const data = payload.data as Record<string, unknown>
        trackingTarget.value = (data.trackingTarget as string) || ''
        motifs.value = (data.motifs || []) as Motif[]
        suggestedAdditions.value = (data.suggestedAdditions || []) as {
          id: string; name: string; type: 'invariant' | 'variable'; core: string
        }[]
        phase.value = 'E1_interrupt'
        break
      }
      case 'select_hypothesis': {
        const data = payload.data as Record<string, unknown>
        candidateHypotheses.value = (data.candidateHypotheses || []) as CandidateHypothesis[]
        allowMultiSelect.value = (data.allowMultiSelect as boolean) || false
        phase.value = 'E6_interrupt'
        break
      }
      case 'decide_next_action': {
        const data = payload.data as Record<string, unknown>
        const status = data.status as Record<string, number> | undefined
        decisionOptions.value = (data.options || []) as DecisionOption[]
        decisionSummary.value = (data.summary as string) || ''
        totalClaims.value = (status?.total as number) || 0
        insufficientCount.value = (status?.insufficient as number) || 0
        phase.value = 'E7_interrupt'
        break
      }
    }

    //  phase  watcher  isAwaitingUserInput
    interruptInstance.value++

    // 
    persistInterruptState()
  }

  function handleMotifMatrix(data: { rows: string[]; columns: string[]; cells: MotifMatrixCell[] }) {
    motifMatrix.value = {
      rows: data.rows || [],
      columns: data.columns || [],
      cells: (data.cells || []) as unknown as MotifMatrixCell[],
    }
  }

  function handlePhylogenyData(data: { nodes: PhylogenyNode[]; edges: PhylogenyEdge[] }) {
    phylogeny.value = data
  }

  function handleEvidenceData(data: { claims: ClaimVerdict[] }) {
    claimVerdicts.value = data.claims || []
  }

  function handleEvolutionNarrative(markdown: string, refs?: Record<string, EvidenceRefData>) {
    if (refs) {
      evidenceRefs.value = refs
    }
    narrative.value = markdown
  }

  // ===== Actions: User operations on motifs =====

  function editMotif(id: string, changes: Partial<Motif>) {
    const idx = motifs.value.findIndex(m => m.id === id)
    if (idx === -1) return
    const current = motifs.value[idx]
    motifs.value[idx] = { ...current, ...changes, isUserEdited: true }
  }

  function addMotif(name: string, type: 'invariant' | 'variable', core: string) {
    const newMotif: Motif = {
      id: `user_m${Date.now()}`,
      name,
      type,
      core,
      isUserEdited: true,
    }
    motifs.value.push(newMotif)
  }

  function deleteMotif(id: string) {
    motifs.value = motifs.value.filter(m => m.id !== id)
  }

  function confirmMotifs() {
    // User has reviewed/edited motifs, ready to resume
  }

  // ===== Actions: User operations on hypotheses =====

  function toggleHypothesisSelection(id: string) {
    const hyp = candidateHypotheses.value.find(h => h.id === id) as
      (CandidateHypothesis & { selected?: boolean }) | undefined
    if (!hyp) return

    if (allowMultiSelect.value) {
      hyp.selected = !hyp.selected
    } else {
      // Single select: deselect all others, select this one
      candidateHypotheses.value.forEach(h => {
        (h as CandidateHypothesis & { selected?: boolean }).selected = h.id === id
      })
    }
  }

  function confirmHypothesisSelection() {
    // User has selected hypotheses, ready to resume
  }

  // ===== Actions: User operations on decisions =====

  function selectDecisionOption(optionId: string) {
    selectedDecisionOption.value = optionId
  }

  // ===== Actions: Resume =====

  function submitResume(): ResumePayload {
    clearPersistedState()
    const resume: Record<string, unknown> = {}

    if (interruptType.value === 'confirm_motifs') {
      resume.action = 'confirm'
      resume.edited_motifs = motifs.value
      phase.value = 'E2_scan'
    } else if (interruptType.value === 'select_hypothesis') {
      resume.action = 'select'
      resume.selected_hypothesis_ids = candidateHypotheses.value
        .filter(h => (h as CandidateHypothesis & { selected?: boolean }).selected)
        .map(h => h.id)
      phase.value = 'E7_test'
    } else if (interruptType.value === 'decide_next_action') {
      resume.action = 'decide'
      resume.decision = selectedDecisionOption.value
      const decision = selectedDecisionOption.value
      if (decision === 'reexplore') {
        phase.value = 'E2_scan'
      } else if (decision === 'change_hypothesis') {
        phase.value = 'E6_hypothesize'
      } else {
        phase.value = 'E7_test'
      }
    }

    return {
      threadId: threadId.value,
      interruptId: interruptId.value,
      resume,
    }
  }

  function exportEvolutionReport(): void {
    const html = [
      '<!DOCTYPE html>',
      '<html lang="zh-CN">',
      '<head><meta charset="UTF-8">',
      '<title>' + trackingTarget.value + ' - </title>',
      '<style>',
      'body{font-family:sans-serif;max-width:900px;margin:0 auto;padding:20px;}',
      'table{border-collapse:collapse;width:100%;}',
      'th,td{border:1px solid #ddd;padding:8px;text-align:left;}',
      '.verdict-supported{color:green}',
      '.verdict-insufficient{color:orange}',
      '.verdict-refuted{color:red}',
      '</style></head>',
      '<body>',
      '<h1>' + trackingTarget.value + ' - </h1>',
      narrative.value || '',
      '</body></html>',
    ].join('\n')

    const blob = new Blob([html], { type: 'text/html;charset=utf-8' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = trackingTarget.value + '--' + new Date().toISOString().slice(0, 10) + '.html'
    a.click()
    URL.revokeObjectURL(url)
  }

  return {
    // State
    phase,
    threadId,
    interruptId,
    interruptInstance,
    interruptType,
    trackingTarget,
    motifs,
    suggestedAdditions,
    candidateHypotheses,
    allowMultiSelect,
    claimVerdicts,
    phylogeny,
    motifMatrix,
    narrative,
    evidenceRefs,
    decisionOptions,
    decisionSummary,
    totalClaims,
    insufficientCount,
    selectedDecisionOption,

    // Getters
    isAwaitingUserInput,
    currentInterruptType,
    canResume,
    selectedHypotheses,

    // Actions: init
    initEvolution,
    setPhase,
    reset,
    resetLiveState,
    stopLiveState,

    // Actions: SSE handlers
    handleEvolutionInterrupt,
    handleMotifMatrix,
    handlePhylogenyData,
    handleEvidenceData,
    handleEvolutionNarrative,

    // Actions: user operations
    editMotif,
    addMotif,
    deleteMotif,
    confirmMotifs,
    toggleHypothesisSelection,
    confirmHypothesisSelection,
    selectDecisionOption,

    // Actions: resume
    submitResume,

    // Multi-analysis
    completedAnalyses,
    getAnalysis,
    isActiveAnalysis,
    snapshotCompletedAnalysis,
    persistAllCompletedAnalyses,
    restoreAllCompletedAnalyses,
    removeCompletedAnalysis,
    loadAnalysisFromServer,

    // Actions: state persistence
    restoreInterruptState,
    applyPersistedState,
    clearPersistedState,
    persistCompleteState,
    restoreCompleteState,
    clearCompleteState,

    // Actions: export
    exportEvolutionReport,
  }
})
