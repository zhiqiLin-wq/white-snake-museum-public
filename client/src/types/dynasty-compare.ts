// A.4: Dynasty compare types

export interface DynastyCompareConfig {
  left: {
    chapterNumber: number
    chapterTitle: string
    dynasty: string
  }
  right: {
    chapterNumber: number
    chapterTitle: string
    dynasty: string
  }
  /**  true */
  syncScroll: boolean
  /**  Agent align_paragraphs  true */
  alignParagraphs: boolean
}

export interface ElementOccurrence {
  chapterNumber: number
  paragraphIndex: number
  /** 0 [startChar, endChar) */
  span: {
    startChar: number
    endChar: number
  }
  /**  */
  text: string
  /** Agent 15-30 */
  contextNote: string
}

export interface ElementEvolution {
  elementId: string
  elementName: string
  category: 'person' | 'location' | 'motif' | 'term' | 'event'
  color: string
  leftOccurrences: ElementOccurrence[]
  rightOccurrences: ElementOccurrence[]
  evolution: {
    changeType: 'emergence' | 'disappearance' | 'amplification' | 'diminution' | 'transvaluation' | 'continuity' | 'transplantation'
    summary: string
    confidence: number
  }
}

export interface LocationEvolution {
  locationName: string
  leftStatus: 'present' | 'absent'
  leftMentions: number
  rightStatus: 'present' | 'absent'
  rightMentions: number
  changeType: 'newly_appeared' | 'disappeared' | 'persistent' | 'increased' | 'decreased'
  coordinates?: { lat: number; lng: number }
}

export interface DynastyCompareEvent {
  type: 'dynasty_compare'
  config: {
    left: { chapterNumber: number; chapterTitle: string; dynasty: string }
    right: { chapterNumber: number; chapterTitle: string; dynasty: string }
  }
  elements: ElementEvolution[]
  locationChanges: LocationEvolution[]
  summary: { overview: string; keyFindings: string[] }
  timestamp: string
}
