export interface EvidenceRef {
  chunkId: string
  chapterNumber: number
  chapterTitle: string
  paragraphIndex: number
  excerpt: string
}

/** 服务端发送的完整证据引用数据（含朝代和文献类型元数据）。
 *  excerpt: 截断版本（max 500 chars），用于弹窗展示。
 *  excerptFull: 完整 chunk 文本（max 2000 chars），仅服务端使用。
 *  label: 锚点ID (如 "唐-ch1:3")，前端用此键直接查找证据。 */
export interface EvidenceRefData {
  chunkId?: string
  chapterNumber: string | number
  chapterTitle: string
  paragraphIndex: number
  excerpt: string
  excerptFull?: string
  charStart?: number
  charEnd?: number
  dynasty?: string
  sourceType?: 'primary_literature' | 'research_literature'
  /** 锚点ID标签 (如 "唐-ch1:3")，前端 citation badge 点击时用此键精确查找 */
  label?: string
}

export type EvolutionPhase =
  | 'idle'
  | 'E1_decompose'
  | 'E1_interrupt'
  | 'E2_scan'
  | 'E3_classify'
  | 'E4_phylogeny'
  | 'E5_correlate'
  | 'E6_hypothesize'
  | 'E6_interrupt'
  | 'E7_test'
  | 'E7_interrupt'
  | 'E8_synthesize'
  | 'complete'
  | 'error'

export type InterruptType = 'confirm_motifs' | 'select_hypothesis' | 'decide_next_action'

export interface Motif {
  id: string
  name: string
  type: 'invariant' | 'variable'
  core: string
  variations?: Record<string, string>
  isUserEdited: boolean
}

export interface CandidateHypothesis {
  id: string
  statement: string
  confidence: number
  reasoning: string
}

export interface ClaimVerdict {
  claimId: string
  statement: string
  verdict: 'supported' | 'insufficient' | 'refuted'
  confidence: 'high' | 'medium' | 'low'
  evidencePro: EvidenceRef[]
  evidenceCon: EvidenceRef[]
  explanation: string
  gapDescription?: string
}

export interface PhylogenyNode {
  id: string
  label: string
  era: string
  type: 'extant' | 'missing' | 'external_influence'
  chapterNumber?: string
}

export interface PhylogenyEdge {
  from: string
  to: string
  type: 'vertical' | 'horizontal'
  style: 'solid' | 'dashed' | 'wavy'
  label?: string
}

export interface MotifMatrixCell {
  era: string
  motifId: string
  status: 'absent' | 'stable' | 'changed'
  value: string | null
  tooltip?: string | null
  occurrenceCount: number
  evidence: EvidenceRef[]
}

export interface DecisionOption {
  id: string
  type: 'deep_investigate' | 're_explore' | 'accept_uncertainty' | 'change_hypothesis'
  label: string
  description: string
}

export interface DecisionStatus {
  totalClaims: number
  insufficientCount: number
  options: DecisionOption[]
  summary: string
}

export interface EvolutionState {
  phase: EvolutionPhase
  threadId: string
  interruptId: string
  interruptType: InterruptType | null
  trackingTarget: string
  motifs: Motif[]
  suggestedAdditions: { id: string; name: string; type: 'invariant' | 'variable'; core: string }[]
  candidateHypotheses: CandidateHypothesis[]
  allowMultiSelect: boolean
  claimVerdicts: ClaimVerdict[]
  phylogeny: { nodes: PhylogenyNode[]; edges: PhylogenyEdge[] }
  motifMatrix: { rows: string[]; columns: string[]; cells: MotifMatrixCell[] }
  narrative: string
  decisionStatus: DecisionStatus
  isAwaitingUserInput: boolean
}

export interface ResumePayload {
  threadId: string
  interruptId: string
  resume: Record<string, unknown>
}
