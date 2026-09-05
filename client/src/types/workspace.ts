export type TabType =
  | 'text-reader'
  | 'compare-view'
  | 'data-panel'
  | 'search-results'
  | 'evolution-workspace'
  | 'split-view'
  | 'map'
  | 'chart-stacked-bar'
  | 'chart-sankey'
  | 'chart-sunburst'
  | 'chart-chord'
  | 'chart-heatmap'
  | 'chart-graph'
  | 'dynasty-compare'
  | 'annotation-index'
  | 'annotation-stats'
  | 'report-library'

export interface WorkspaceTab {
  id: string
  type: TabType
  title: string
  data: Record<string, unknown>
  dirty: boolean
  createdAt: number
}

export interface SplitState {
  enabled: boolean
  leftTabId: string | null
  rightTabId: string | null
  splitRatio: number
  syncScroll: boolean
  alignParagraphs: boolean
}
