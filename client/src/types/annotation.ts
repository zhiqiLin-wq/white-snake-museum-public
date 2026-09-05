export type AnnotationCategory = 'person' | 'location' | 'event' | 'term' | 'motif' | 'custom'

export type PassageKey = string

/** 阶段 0 — TYP-01: 前端传递给 Agent 的段落信息 */
export interface ParagraphInfo {
  index: number       // 0-based 段落索引
  text: string        // 段落文本（已 trim，与 TextReaderView 渲染文本完全一致）
}

/** 阶段 0 — P2-04: 标注覆盖率信息 */
export interface CoverageInfo {
  paragraphsTotal: number
  paragraphsAnnotated: number
  coverageRate: number
  totalEntities: number
  quality: string
}

export interface AnnotationSpan {
  startChar: number
  endChar: number
}

export interface Annotation {
  id: string
  category: AnnotationCategory
  label: string
  span: AnnotationSpan
  explanation: string
  confidence: number
  motifRef?: string
  color: string
  source: 'user' | 'agent' | 'rule'
  is_retry?: boolean
  chapterNumber?: number
  paragraphIndex?: number
  text?: string
  createdAt: number
  updatedAt: number
}

export interface UserAnnotation {
  id: string
  category: AnnotationCategory
  label?: string
  span: AnnotationSpan
  note: string
  createdAt: number
  color: string
}

export interface Insight {
  id: string
  text: string
  motifRef?: string
  category: string
  label: string
}

export interface Marginalia {
  id: string
  annotationId?: string
  chapterNumber: number
  paragraphIndex: number
  anchorCharOffset: number
  anchorStartChar?: number
  anchorEndChar?: number
  content: string
  color?: string
  source: 'user' | 'agent'
  createdAt: number
  updatedAt: number
}

export interface AnnotationSuggestion {
  id: string
  span: AnnotationSpan
  text: string
  category: AnnotationCategory
  color: string
  reason: string
}

export interface AnnotationSet {
  chapterNumber: number
  annotations: Annotation[]
  marginalia: Marginalia[]
  savedAt: number
}

export interface AnnotationVisibility {
  user: {
    person: boolean
    location: boolean
    event: boolean
    term: boolean
    motif: boolean
    custom: boolean
  }
  agent: {
    person: boolean
    location: boolean
    event: boolean
    term: boolean
    motif: boolean
    custom: boolean
  }
}

// C03: New types for annotation index, stats, and export

export interface AnnotationIndexItem {
  id: string
  category: AnnotationCategory
  categoryLabel: string
  label: string
  text: string
  span: AnnotationSpan
  color: string
  source: 'user' | 'agent'
  passageKey: string
  chapterNumber: number
  dynasty: string
  chapterTitle: string
  paragraphIndex: number
  contextSnippet: string
  note?: string
  confidence?: number
  createdAt: number
  marginalia: { id: string; content: string }[]
}

export interface AnnotationStats {
  total: number
  userCount: number
  agentCount: number
  withMarginalia: number
  paragraphsCovered: number
  totalParagraphs: number
  coverageRate: number
  categoryDistribution: { category: string; categoryLabel: string; count: number; userCount: number; agentCount: number; percentage: number; color: string }[]
  dynastyDistribution: { dynasty: string; dynastyOrder: number; count: number; percentage: number }[]
  chapterDensity: { chapterNumber: number; chapterTitle: string; dynasty: string; annotationCount: number; totalParagraphs: number; annotatedParagraphs: number; densityPercent: number }[]
  userVsAgentByCategory: { category: string; categoryLabel: string; userCount: number; agentCount: number }[]
  timeline: { action: string; annotationId: string; passageKey: string; category: string; label: string; timestamp: number; timestampLabel: string }[]
}

export interface ExportJSON {
  exportMeta: { exportedAt: string; platform: string; version: string; formatVersion: string }
  annotationSets: Record<string, {
    chapterNumber: number
    chapterTitle: string
    dynasty: string
    annotations: {
      id: string
      category: string
      categoryLabel: string
      label: string
      text: string
      span: AnnotationSpan
      color: string
      source: 'user' | 'agent'
      passageKey: string
      paragraphIndex: number
      note?: string
      agentAnnotationId?: string
      motifRef?: string
      confidence?: number
      createdAt: number
      updatedAt: number
      marginalia: { id: string; content: string }[]
    }[]
    marginalia?: { id: string; content: string }[]
    paragraphTexts?: { paragraphIndex: number; text: string }[]
    savedAt: string
  }>
}
