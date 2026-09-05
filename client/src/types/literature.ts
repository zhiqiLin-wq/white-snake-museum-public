export interface Paragraph {
  index: number
  text: string
}

export interface Chapter {
  number: number
  title: string
  dynasty: string
  paragraphs: Paragraph[]
}

export interface SearchResult {
  chapterNumber: number
  chapterTitle: string
  dynasty: string
  paragraphIndex: number
  excerpt: string
  excerptOffset: number
  matchPositions: [number, number][]
  matchPositionsInExcerpt: [number, number][]
  score: number
}

export interface SourceCitation {
  chunkId: string
  chapterNumber: number
  chapterTitle: string
  excerpt: string
  relevanceScore: number
}

/**  —  TextReaderView  */
export interface SearchHighlight {
  chunkId?: string
  query: string
  chapterNumber: number
  paragraphIndex: number
  startChar: number
  endChar: number
  text: string
}

/**  */
export interface ActiveSearch {
  query: string
  highlights: SearchHighlight[]
  source: 'user' | 'agent'
  /** 暂时性高亮（搜索结果跳转）：离开阅读页即自动清除，重进不保留 */
  ephemeral?: boolean
  /** B-162: 跳转时 chapters 可能未加载导致 highlights 为空；
   * TextReaderView 在 paragraphs 加载后用此字段重算高亮 */
  pendingRecompute?: {
    excerpt?: string
    paragraphMatchPositions?: [number, number][]
    paragraphIndex: number
    chapterNumber: number
  }
}
