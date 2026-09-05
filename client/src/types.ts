export interface Chapter {
  number: string
  title: string
  content: string
}

export interface ExcelRecord {
  切词数: number
  匹配词数: number
  文本字数: number
  摘要: string
}

export interface Keyword {
  word: string
  weight: number
}

export interface LocationData {
  name: string
  records: ExcelRecord[]
  totalMatch: number
  totalWords: number
  keywords: Keyword[]
}

export interface StorySegment {
  from: string
  to: string
}

