// B.2: Map traceability and layer types

export interface PhysicalLandscapeInfo {
  name: string
  historicalName: string
  currentName: string
  builtYear: number | null
  destroyedYear: number | null
  rebuiltYear: number | null
  historicalChanges: string
  panoramaUrl: string
  poi: {
    address: string | null
    openTime: string | null
    ticketPrice: string | null
  } | null
}

export interface LiteraryRecord {
  dynasty: string
  dynastyOrder: number
  chapterNumber: number
  chapterTitle: string
  paragraphIndex: number
  excerpt: string
  span: { startChar: number; endChar: number }
  descriptionStyle: string
}

export interface MutualConstruction {
  landscapeToText: string
  textToLandscape: string
}

export interface LocationTraceabilityCard {
  locationName: string
  physicalLandscape: PhysicalLandscapeInfo
  literaryRecords: LiteraryRecord[]
  totalMentions: Record<string, number>
  mutualConstruction: MutualConstruction | null
  descriptionStyles: Record<string, string>
  loadedAt: number
}

export interface MapMarkerStyle {
  fillColor: string
  radius: number
  opacity: number
  dashed: boolean
  pulse: boolean
}
