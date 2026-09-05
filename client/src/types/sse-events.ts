import type { Annotation, Insight, AnnotationSuggestion, AnnotationCategory } from './annotation'
import type { Motif, PhylogenyNode, PhylogenyEdge, MotifMatrixCell, CandidateHypothesis, ClaimVerdict, DecisionOption, EvidenceRef, EvidenceRefData } from './evolution'
import type { DynastyCompareEvent } from './dynasty-compare'
import type { SourceCitation } from './literature'

export type { DynastyCompareEvent, SourceCitation }

// SSE event type enum — must match backend events.py SSEEventType exactly
export enum SSEEventType {
  // Lifecycle
  THINKING_START = 'thinking_start',
  DONE = 'done',
  ERROR = 'error',
  // Progress
  STATUS = 'status',
  INTENT = 'intent',
  THINKING_STEP = 'thinking_step',
  THINKING_PLAN = 'thinking_plan',
  EXPORT_READY = 'export_ready',
  WORKFLOW_DONE = 'workflow_done',
  // Content
  CHUNK = 'chunk',
  SOURCE = 'source',
  // Tools
  TOOL_CALL = 'tool_call',
  TOOL_RESULT = 'tool_result',
  // Workspace drivers
  ANNOTATION_DATA = 'annotation_data',
  ANNOTATION_SUGGEST = 'annotation_suggest',
  PARAGRAPH_ALIGNMENT = 'paragraph_alignment',
  STATS_DATA = 'stats_data',
  COMPARISON_DATA = 'comparison_data',
  SEARCH_RESULTS = 'search_results',
  TEXT_NAVIGATE = 'text_navigate',
  MAP_HIGHLIGHT = 'map_highlight',
  MAP_LAYER_UPDATE = 'map_layer_update',  // v11: Agent 
  // Evolution
  MOTIF_MATRIX = 'motif_matrix',
  PHYLOGENY_DATA = 'phylogeny_data',
  EVIDENCE_DATA = 'evidence_data',
  EVOLUTION_NARRATIVE = 'evolution_narrative',
  // Human-AI collaboration
  EVOLUTION_INTERRUPT = 'evolution_interrupt',
  INTERRUPT_RESUMED = 'interrupt_resumed',
  // v11: Agent 
  OPEN_SPLIT_VIEW = 'open_split_view',
}

// SSE event payload interfaces (corresponding to design doc section 10.2 JSON Schema)

export interface ToolCallRecord {
  tool: string
  input: Record<string, unknown>
  status: string
  resultPreview?: string
}

export interface ThinkingPlanStep {
  name: string
  label: string
}

export interface ThinkingPlanEvent {
  totalSteps: number
  steps: ThinkingPlanStep[]
}

export interface StatusStep {
  message: string
  done: boolean
  ts: number
  // U-03: Enhanced step card fields
  tool?: string
  label?: string
  stepIndex?: number
  totalSteps?: number
  durationMs?: number
  summary?: string
  stepType?: 'thinking_step' | 'tool_call' | 'tool_result' | 'status'
  status?: 'running' | 'completed' | 'error' | 'skipped'
  errorMsg?: string
}

// CU-02:  — 
export interface PassageAnnotationGroup {
  passageKey: string
  paragraphIndex: number
  annotations: Annotation[]
  insights: Insight[]
}

export interface AnnotationDataEvent {
  chapterNumber: number
  chapterTitle: string
  paragraphRange: [number, number]
  passageAnnotations: PassageAnnotationGroup[]
  splitSide?: 'left' | 'right'
}

export interface AnnotationSuggestEvent {
  target: string
  suggestions: AnnotationSuggestion[]
  timestamp: number
}

export interface ParagraphAlignment {
  leftParagraphIndex: number
  rightParagraphIndex: number
  confidence: number
}

export interface ParagraphAlignmentEvent {
  leftChapter: number
  rightChapter: number
  alignments: ParagraphAlignment[]
}

export interface StatsColumn {
  key: string
  label: string
  type: 'string' | 'integer' | 'percent' | 'float'
}

export interface StatsDataEvent {
  title: string
  columns: StatsColumn[]
  rows: Record<string, unknown>[]
  aiSummary: string
}

export interface ComparisonHighlight {
  span: [number, number]
  sentiment: 'positive' | 'negative' | 'neutral'
  label: string
}

export interface ComparisonSide {
  label: string
  chapterNumber: string
  paragraphIndex: number
  text: string
  highlights: ComparisonHighlight[]
}

export interface DiffChange {
  type: 'emergence' | 'disappearance' | 'substitution' | 'amplification' | 'diminution' | 'transvaluation' | 'transplantation'
  description: string
}

export interface ComparisonDataEvent {
  left: ComparisonSide
  right: ComparisonSide
  diffSummary: {
    changes: DiffChange[]
    narrative: string
  }
  source: 'agent' | 'split-view'
}

export interface SearchResultItem {
  rank: number
  chunkId: string
  chapterNumber: string
  chapterTitle: string
  paragraphIndex: number
  excerpt: string
  highlightRanges: [number, number][]
  paragraphMatchPositions: [number, number][]
  relevanceScore: number
  /** 研究文献条目（无朝代归属、前端无对应详情页）——为 true 时不显示跳转按钮 */
  isResearch?: boolean
  /** Python 版 search_paragraphs 附带的原始 metadata */
  metadata?: { dynasty?: string }
}

export interface SearchResultsEvent {
  query: string
  totalMatches: number
  results: SearchResultItem[]
  sortOptions: string[]
}

export interface TextNavigateEvent {
  chapterNumber: number
  chapterTitle: string
  paragraphIndex: number
}

export interface MapHighlightEvent {
  name: string
  lat: number
  lng: number
  action: 'fly_to' | 'highlight' | 'add_marker'
  era?: string
}

export interface MapLayerUpdateEvent {
  dynasty: string | string[]  // ''  ['', ''] 
  action: 'set' | 'toggle' | 'reset'
}

// Evolution event payloads (use types from evolution.ts)

export interface MotifMatrixEvent {
  rows: string[]
  columns: string[]
  cells: MotifMatrixCell[]
}

export interface PhylogenyDataEvent {
  nodes: PhylogenyNode[]
  edges: PhylogenyEdge[]
}

export interface EvidenceDataEvent {
  selectedHypothesisId?: string
  claims: ClaimVerdict[]
}

export interface EvolutionNarrativeEvent {
  markdown: string
  evidenceRefs?: Record<string, EvidenceRefData>
}

export interface EvolutionInterruptEvent {
  threadId: string
  interruptId: string
  interruptType: 'confirm_motifs' | 'select_hypothesis' | 'decide_next_action'
  data: Record<string, unknown>
}

export interface InterruptResumedEvent {
  threadId: string
  interruptId: string
  newPhase: string
}

export interface SuggestionItem {
  id: string
  name: string
  type: 'invariant' | 'variable'
  core: string
}

// Map traceability SSE event types

export interface LocationTraceabilityEvent {
  type: 'location_traceability'
  data: Record<string, {
    literaryRecords: Array<{
      dynasty: string; dynastyOrder: number
      chapterNumber: number; chapterTitle: string
      paragraphIndex: number; excerpt: string
      span: { startChar: number; endChar: number }
      descriptionStyle: string
    }>
    totalMentions: Record<string, number>
  }>
}

export interface MutualConstructionEvent {
  type: 'mutual_construction'
  locationName: string
  mutualConstruction: {
    landscapeToText: string
    textToLandscape: string
  }
  descriptionStyles: Record<string, string>
  relatedMotifs: string[]
}
