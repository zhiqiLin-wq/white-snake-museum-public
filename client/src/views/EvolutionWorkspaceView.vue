<script setup lang="ts">
import { computed, ref, watch, onMounted, onUnmounted, nextTick } from 'vue'
import { useEvolutionStore } from '@/stores/evolution'
import type { EvidenceRefData } from '@/types/evolution'
import { useWorkspaceStore } from '@/stores/workspace'
import { useAgentStore } from '@/stores/agent'
import { useAnnotationStore } from '@/stores/annotations'
import { useSourceTreeStore } from '@/stores/sourceTree'
import Modal from '@/components/ui/Modal.vue'
import MotifEditor from '@/components/interrupt/MotifEditor.vue'
import HypothesisSelector from '@/components/interrupt/HypothesisSelector.vue'
import ActionDecider from '@/components/interrupt/ActionDecider.vue'
import EvolutionProgress from '@/components/evolution/EvolutionProgress.vue'
import { SSEEventBus } from '@/services/sseEventBus'
import * as echarts from 'echarts'

import { marked } from 'marked'

const evolutionStore = useEvolutionStore()
const workspaceStore = useWorkspaceStore()
const agentStore = useAgentStore()
const sourceTreeStore = useSourceTreeStore()

const props = defineProps<{
  tabId?: string
}>()

interface SelectedHypothesis extends Record<string, unknown> {
  id: string
  statement: string
  confidence: number
  reasoning: string
  selected?: boolean
}

//  threadId
const analysisThreadId = computed(() => {
  if (!props.tabId) return ''
  const tab = workspaceStore.tabs.find(t => t.id === props.tabId)
  if (!tab) return ''
  return (tab.data as Record<string, unknown>).threadId as string || ''
})

const isLive = computed(() => evolutionStore.isActiveAnalysis(analysisThreadId.value))

//  ref 
const displayPhase = computed(() => {
  const tid = analysisThreadId.value
  if (!tid || isLive.value) return evolutionStore.phase
  const analysis = evolutionStore.getAnalysis(tid)
  return analysis?.phase || 'idle'
})

const displayMotifMatrix = computed(() => {
  const tid = analysisThreadId.value
  if (!tid || isLive.value) return evolutionStore.motifMatrix
  const analysis = evolutionStore.getAnalysis(tid)
  return analysis?.motifMatrix || { rows: [], columns: [], cells: [] }
})

const displayPhylogeny = computed(() => {
  const tid = analysisThreadId.value
  if (!tid || isLive.value) return evolutionStore.phylogeny
  const analysis = evolutionStore.getAnalysis(tid)
  return analysis?.phylogeny || { nodes: [], edges: [] }
})

const displayClaimVerdicts = computed(() => {
  const tid = analysisThreadId.value
  if (!tid || isLive.value) return evolutionStore.claimVerdicts
  const analysis = evolutionStore.getAnalysis(tid)
  return analysis?.claimVerdicts || []
})

const displayNarrative = computed(() => {
  const tid = analysisThreadId.value
  if (!tid || isLive.value) return evolutionStore.narrative
  const analysis = evolutionStore.getAnalysis(tid)
  return analysis?.narrative || ''
})

const displayEvidenceRefs = computed(() => {
  const tid = analysisThreadId.value
  if (!tid || isLive.value) return evolutionStore.evidenceRefs
  const analysis = evolutionStore.getAnalysis(tid)
  return analysis?.evidenceRefs || {}
})

const displayTrackingTarget = computed(() => {
  const tid = analysisThreadId.value
  if (!tid || isLive.value) return evolutionStore.trackingTarget
  const analysis = evolutionStore.getAnalysis(tid)
  return analysis?.trackingTarget || ''
})

const phase = computed(() => evolutionStore.phase)
const activeSubTab = ref<'matrix' | 'phylogeny' | 'evidence' | 'narrative'>('matrix')
const showCloseConfirm = ref(false)

// Evidence popup modal state
const evidenceModalVisible = ref(false)
const evidenceModalTitle = ref('')
const evidenceModalExcerpt = ref('')
const evidenceModalExcerptHtml = ref('')
const evidenceModalChapterNumber = ref<number>(0)
const evidenceModalChapterTitle = ref('')
const evidenceModalParagraphIndex = ref<number>(0)
const evidenceModalDynasty = ref('')
const evidenceModalSourceType = ref('')
const evidenceModalQuotedText = ref('')  // B-150: 证据 excerpt，用于原文跳转时兜底高亮匹配
const evidenceModalQuote = ref('')  // B-152n: LLM 引文（data-quote），跳转原文高亮首选

// S-02: Interrupt timeout countdown (10 minutes)
const INTERRUPT_TIMEOUT_SECONDS = 600
const interruptStartTime = ref<number>(0)
const countdownRemaining = ref(0)
const isTimedOut = ref(false)
let countdownTimer: ReturnType<typeof setInterval> | null = null

function startCountdown() {
  interruptStartTime.value = Date.now()
  isTimedOut.value = false
  countdownRemaining.value = INTERRUPT_TIMEOUT_SECONDS
  stopCountdown()
  countdownTimer = setInterval(() => {
    const elapsed = Math.floor((Date.now() - interruptStartTime.value) / 1000)
    countdownRemaining.value = Math.max(0, INTERRUPT_TIMEOUT_SECONDS - elapsed)
    if (countdownRemaining.value <= 0) {
      isTimedOut.value = true
      stopCountdown()
    }
  }, 1000)
}

function stopCountdown() {
  if (countdownTimer) {
    clearInterval(countdownTimer)
    countdownTimer = null
  }
}

function formatCountdown(seconds: number): string {
  const m = Math.floor(seconds / 60)
  const s = seconds % 60
  return `${m}:${s.toString().padStart(2, '0')}`
}

function handleRestartEvolution() {
  isTimedOut.value = false
  stopCountdown()
  // B-151: 超时后重新开始——只停止运行态标记，保留已分析出的数据
  evolutionStore.stopLiveState()
  evolutionStore.setPhase('idle')
  // 同时重置 agentStore 的 streaming 状态
  agentStore.resetStreamState()
}

// interruptInstance 
watch(
  () => evolutionStore.interruptInstance,
  (instance) => {
    if (instance > 0 && evolutionStore.isAwaitingUserInput) {
      startCountdown()
    }
  },
  { immediate: true }
)

// 
watch(
  () => evolutionStore.isAwaitingUserInput,
  (isAwaiting) => {
    if (!isAwaiting) {
      stopCountdown()
      isTimedOut.value = false
    }
  }
)

const phaseLabels: Record<string, string> = {
  E1_decompose: 'E1 分解母题 — AI自动将追踪目标分解为叙事母题...',
  E2_scan: 'E2 扫描矩阵 — 扫描四个朝代文本中的母题出现情况...',
  E3_classify: 'E3 分类变化 — 识别每个母题的变化类型...',
  E4_phylogeny: 'E4 推断谱系 — 建模文本间的传承关系...',
  E5_correlate: 'E5 外部关联 — 关联历史事件和社会背景...',
  E6_hypothesize: 'E6 提出假设 — 基于模式提出演化假设...',
  E7_test: 'E7 验证假设 — 逐条验证假设中的论断...',
  E8_synthesize: 'E8 综合呈现 — 生成最终分析报告...',
  complete: '演化分析完成',
}

const phaseLabel = computed(() => phaseLabels[displayPhase.value] || '')

// 顶部步进器：判断某个阶段(E1..E8)是否已完成
function isPhaseDone(p: string): boolean {
  const cur = displayPhase.value
  if (cur === 'complete') return true
  if (cur === 'idle' || cur === 'error') return false
  const curNum = parseInt(cur.replace(/^E(\d).*/, '$1'), 10)
  const myNum = parseInt(p.slice(1), 10)
  if (isNaN(curNum) || isNaN(myNum)) return false
  // 当前阶段号 > 该阶段号 → 已完成；同号(含 interrupt) → 仍在该阶段，未完成
  return curNum > myNum
}

//  Agent Panel 
const latestStatusMessage = computed(() => {
  const steps = agentStore.statusSteps
  if (steps.length === 0) return ''
  return steps[steps.length - 1].message || ''
})

// Matrix cell helper functions
interface MatrixCell {
  motifId: string
  era: string
  status: 'absent' | 'stable' | 'changed'
  value: string | null
  tooltip?: string | null
  occurrenceCount: number
  evidence: Array<{ chapterNumber: string; chapterTitle: string; paragraphIndex: number; excerpt: string }>
}

function findCell(dynasty: string, motifCol: string): MatrixCell | undefined {
  const cells = displayMotifMatrix.value.cells as unknown as MatrixCell[]
  if (!cells || cells.length === 0) return undefined
  // motifCol format: "M1: " → extract "M1"
  const motifId = motifCol.split(':')[0].trim()
  return cells.find(c => c.motifId === motifId && c.era === dynasty)
}

function getCellClass(dynasty: string, motifCol: string): Record<string, boolean> {
  const cell = findCell(dynasty, motifCol)
  if (!cell) return { absent: true, stable: false, changed: false }
  return {
    absent: cell.status === 'absent',
    stable: cell.status === 'stable',
    changed: cell.status === 'changed',
  }
}

function getCellValue(dynasty: string, motifCol: string): string {
  const cell = findCell(dynasty, motifCol)
  if (!cell) return '—'
  if (cell.status === 'absent') return '—'
  return cell.value || `${cell.occurrenceCount}`
}

function getCellTooltip(dynasty: string, motifCol: string): string {
  const cell = findCell(dynasty, motifCol)
  if (!cell) return `${dynasty} — `
  if (cell.status === 'absent') return `${dynasty} — `
  const evidenceCount = cell.evidence?.length || 0
  return `${dynasty} —  ${cell.occurrenceCount} ${evidenceCount} `
}

function showCellEvidence(dynasty: string, motifCol: string) {
  const cell = findCell(dynasty, motifCol)
  if (!cell || !cell.evidence || cell.evidence.length === 0) return
  // Build combined evidence text showing ALL evidence items
  const parts = cell.evidence.map((ev, i) => {
    const chTitle = ev.chapterTitle || ''
    const pIdx = ev.paragraphIndex
    const excerpt = ev.excerpt || ''
    return `${i + 1}${chTitle}  ${pIdx}\n${excerpt}`
  })
  const combinedExcerpt = parts.join('\n\n---\n\n')
  const first = cell.evidence[0]
  showEvidencePopup(
    combinedExcerpt,
    `${dynasty} · ${first.chapterTitle || ''}`,
    typeof first.chapterNumber === 'string' ? parseInt(first.chapterNumber, 10) : first.chapterNumber,
    first.paragraphIndex,
  )
}

// Phylogeny ECharts graph
const phyloChartRef = ref<HTMLDivElement | null>(null)
let phyloChart: echarts.ECharts | null = null

function buildPhyloGraph() {
  const data = displayPhylogeny.value
  if (!data.nodes || data.nodes.length === 0) return

  const fs = Math.round(parseFloat(getComputedStyle(document.documentElement).fontSize) * 0.65)

  const categories = [
    { name: '现存文本', itemStyle: { color: '#58A8D0' }, symbol: 'circle' },
    { name: '缺失文本', itemStyle: { color: '#A8B8A8' }, symbol: 'diamond' },
    { name: '外部影响', itemStyle: { color: '#E8C870' }, symbol: 'triangle' },
  ]

  const nodeStyleMap: Record<string, number> = { extant: 0, missing: 1, external_influence: 2 }

  // Build label->id lookup; edges reference labels, nodes need unique ids in ECharts
  const labelToId = new Map<string, string>()
  for (const n of data.nodes) {
    if (n.label && n.id) {
      labelToId.set(n.label, n.id)
    }
  }

  const graphNodes = data.nodes.map((n, i) => ({
    name: n.id || n.label || `node_${i}`,
    category: nodeStyleMap[n.type] ?? 0,
    symbolSize: n.type === 'missing' ? 12 : 18,
    itemStyle: n.type === 'missing'
      ? { borderColor: '#A8B8A8', borderWidth: 2, borderType: 'dashed' as const, color: 'transparent' }
      : undefined,
    label: { show: true, formatter: n.label || n.id },
  }))

  const edgeStyleMap: Record<string, { color: string; type: string }> = {
    solid: { color: '#58A8D0', type: 'solid' },
    dashed: { color: '#A8B8A8', type: 'dashed' },
    wavy: { color: '#E8C870', type: 'dotted' },
  }

  const graphLinks = data.edges.map(e => {
    const sourceId = labelToId.get(e.from) || e.from
    const targetId = labelToId.get(e.to) || e.to
    return {
      source: sourceId,
      target: targetId,
      lineStyle: edgeStyleMap[e.style] || edgeStyleMap.solid,
    }
  })

  if (!phyloChart && phyloChartRef.value) {
    phyloChart = echarts.init(phyloChartRef.value)
  }
  if (!phyloChart) return

  phyloChart.setOption({
    tooltip: { show: true, formatter: '{b}' },
    legend: {
      data: categories.map(c => c.name),
      textStyle: { color: 'var(--color-text-secondary)' as never, fontSize: fs },
    },
    series: [{
      type: 'graph',
      layout: 'force',
      force: { repulsion: 300, edgeLength: [80, 200], gravity: 0.1 },
      roam: true,
      draggable: true,
      data: graphNodes,
      links: graphLinks,
      categories,
      label: { show: true, fontSize: fs, color: 'var(--color-text-primary)' as never },
      edgeSymbol: ['none', 'arrow'],
      edgeSymbolSize: 8,
      emphasis: { focus: 'adjacency' },
    }],
  }, true)
}

//  tab  ECharts 
// v-if  DOM ECharts 
watch(
  () => activeSubTab.value,
  (newTab, oldTab) => {
    if (oldTab === 'phylogeny') {
      phyloChart?.dispose()
      phyloChart = null
    }
    if (newTab === 'phylogeny') {
      void nextTick(() => {
        setTimeout(() => buildPhyloGraph(), 100)
      })
    }
  }
)

//  tab
watch(
  () => displayPhylogeny.value.nodes,
  () => {
    if (activeSubTab.value === 'phylogeny' && displayPhylogeny.value.nodes.length > 0) {
      void nextTick(() => {
        setTimeout(() => buildPhyloGraph(), 100)
      })
    }
  },
  { deep: true }
)

onMounted(() => {
  //  MuseumView  store
  // 
  if (!analysisThreadId.value) {
    tryRestoreInterruptState()
  } else if (
    !evolutionStore.isActiveAnalysis(analysisThreadId.value) &&
    !evolutionStore.getAnalysis(analysisThreadId.value)
  ) {
    // 本地快照缺失（清缓存/过期/被误删）：从后端 SQLite checkpoint 恢复
    void evolutionStore.loadAnalysisFromServer(analysisThreadId.value)
  }
  if (activeSubTab.value === 'phylogeny') {
    setTimeout(() => buildPhyloGraph(), 200)
  }
})

onUnmounted(() => {
  phyloChart?.dispose()
  phyloChart = null
  stopCountdown()
})

// Motif inline editing state
const editingMotifId = ref<string | null>(null)
const editingMotifCore = ref('')
const newMotifName = ref('')
const newMotifType = ref<'invariant' | 'variable'>('variable')

function startEditMotif(motifId: string, currentCore: string) {
  editingMotifId.value = motifId
  editingMotifCore.value = currentCore
}

function saveEditMotif(motifId: string) {
  evolutionStore.editMotif(motifId, { core: editingMotifCore.value })
  editingMotifId.value = null
}

function cancelEditMotif() {
  editingMotifId.value = null
}

function addNewMotif() {
  const name = newMotifName.value.trim()
  if (!name) return
  evolutionStore.addMotif(name, newMotifType.value, '')
  newMotifName.value = ''
}

function handleCloseClick() {
  if (evolutionStore.isAwaitingUserInput) {
    showCloseConfirm.value = true
  } else {
    closeCurrentTab()
  }
}

function confirmClose() {
  showCloseConfirm.value = false
  closeCurrentTab()
}

function closeCurrentTab() {
  const tid = analysisThreadId.value
  if (tid && !isLive.value) {
    evolutionStore.removeCompletedAnalysis(tid)
  }
  workspaceStore.closeTab(workspaceStore.activeTabId!)
}

function submitResume() {
  const payload = evolutionStore.submitResume()
  console.log('[EvolutionWorkspace] submitResume payload:', JSON.stringify(payload, null, 2))
  agentStore.resumeEvolution({
    thread_id: payload.threadId,
    interrupt_id: payload.interruptId,
    resume: payload.resume,
  }).then(() => {
    console.log('[EvolutionWorkspace] resumeEvolution succeeded')
  }).catch((err: unknown) => {
    console.error('[EvolutionWorkspace] resumeEvolution failed:', err)
  })
}

async function handleCancel() {
  const bus = SSEEventBus.getInstance()
  bus.abort()
  evolutionStore.clearPersistedState()
  try {
    await fetch('/api/agent/chat/cancel', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ thread_id: evolutionStore.threadId })
    })
  } catch {
    // ignore network errors, frontend already aborted
  }
  // B-151: 取消后停止运行态 + 置为 idle，让 UI 不再显示 spinner
  evolutionStore.stopLiveState()
  // 不调用 resetLiveState()——保留已分析出的矩阵/叙述/证据数据不被清空
  // 但 phase 需要变为 idle 以触发 UI 切换（stopLiveState 不改变 phase）
  evolutionStore.setPhase('idle')
  // 同时重置 agentStore，清除旧的状态步骤
  agentStore.resetStreamState()
}

interface EvidenceRef {
  chapterNumber: number
  chapterTitle: string
  paragraphIndex: number
  excerpt: string
  excerptFull?: string
  chunkId?: string
  dynasty?: string
  sourceType?: string
}

function navigateToSource(evidence: EvidenceRef) {
  // chapterNumber 可能无效（研究文献无章节号），但证据文本仍需展示
  // → 弹窗始终打开，仅"跳转原文"按钮在 jumpToSourceFromModal 内按 chapterNumber 守卫禁用
  const rawCh = evidence.chapterNumber
  const parsedCh = typeof rawCh === 'string' ? parseInt(rawCh, 10) : (rawCh as number)
  const safeCh = (typeof parsedCh === 'number' && !isNaN(parsedCh) && parsedCh >= 0) ? parsedCh : -1
  const rawPara = evidence.paragraphIndex
  const safePara = (typeof rawPara === 'number' && !isNaN(rawPara) && rawPara >= 0) ? rawPara : 0
  // 弹窗展示完整内容（excerptFull 不截断，fallback 到 excerpt）
  const fullText = evidence.excerptFull || evidence.excerpt || ''
  showEvidencePopup(
    fullText,
    evidence.chapterTitle || '',
    safeCh,
    safePara,
    evidence.dynasty || '',
    evidence.sourceType || '',
  )
}

function showEvidencePopup(excerpt: string, chapterTitle: string, chapterNumber: number, paragraphIndex: number, dynasty?: string, sourceType?: string) {
  evidenceModalExcerpt.value = excerpt
  evidenceModalExcerptHtml.value = escapeHtml(excerpt)
  evidenceModalChapterTitle.value = chapterTitle
  evidenceModalChapterNumber.value = chapterNumber
  evidenceModalParagraphIndex.value = paragraphIndex
  evidenceModalDynasty.value = dynasty || ''
  evidenceModalSourceType.value = sourceType || ''
  const srcLabel = sourceType === 'research_literature' ? '[研究文献]' : ''
  const dynastyLabel = dynasty ? `[${dynasty}代]` : ''
  evidenceModalTitle.value = chapterTitle
    ? `${dynastyLabel} ${srcLabel} ${chapterTitle} 段落${paragraphIndex}`
    : `来源段落${paragraphIndex}`
  evidenceModalVisible.value = true
}

function jumpToSourceFromModal() {
  if (evidenceModalChapterNumber.value <= 0 || isNaN(evidenceModalChapterNumber.value)) return
  const evCh = sourceTreeStore.chapters.find(c => c.number === evidenceModalChapterNumber.value)
  const evTitle = evidenceModalChapterTitle.value
    ? (evCh ? `${evCh.dynasty} · ${evidenceModalChapterTitle.value}` : evidenceModalChapterTitle.value)
    : `章节${evidenceModalChapterNumber.value}`

  // B-150: 仅当 excerpt 足够长才尝试原文高亮匹配，短文本跳了也匹配不到
  // B-152f: 阈值 20 → 8 → 4 — 与 TextReaderView 的 normQuote.length >= 4 一致
  // B-152n: 优先用 LLM 引文（10-60字短句，与原文匹配成功率高）；excerpt 全文
  // 匹配必败——chunk 清洗文本与原文渲染存在字符差异，且前缀兜底会命中结构标题
  // B-156: LLM 引文被后端删除时，回退到 excerpt 前 25 字（而非全文 500 字），
  // 短前缀更可能落在同一段落内，匹配器 12 字前缀兜底成功率更高
  const _llmQuote = evidenceModalQuote.value || ''
  const _excerptHead = (evidenceModalQuotedText.value || '').slice(0, 25)
  const highlightText = _llmQuote || (_excerptHead.length >= 4 ? _excerptHead : '') || ''
  const shouldHighlight = highlightText.length >= 4

  workspaceStore.openTab('text-reader', evTitle, {
    chapterNumber: evidenceModalChapterNumber.value,
    paragraphIndex: evidenceModalParagraphIndex.value,
    citationHighlight: shouldHighlight
      ? { text: highlightText }
      : undefined,
  })
  evidenceModalVisible.value = false
}

const narrativeHtml = computed(() => {
  const text = displayNarrative.value
  if (!text) return ''
  const refs = displayEvidenceRefs.value

  // B-150: 锚点ID + 前端引文提取。
  // B-152e: 单遍替换 — 可选引文组与锚点同时匹配，引文直接写入该次出现的 data-quote。
  // 旧实现先存全局 quoteMap[label] 再生成 badge，同一标签出现两次时后者覆盖前者（高亮串扰）。
  // 引文按"出现位置"隔离：每个徽章携带自己前面的引文，互不影响。

  // B-152h: 引号后允许任意字符(汉字/标点/空白)再跟标签 — LLM 实际输出
  // "引文"的围观场景[标签]，旧正则只允许标点+空白导致引文丢失→高亮全失败
  // B-152o: 引号必须用 \u 转义！此前字面写法被写成了 ASCII 直引号(0x22)，
  // 而 LLM 输出的是标准弯引号 \u201C/\u201D → 引文提取全灭(withQuote=0)的最终元凶
  const citeRegex = /(?:[\u201c"]([^\u201c\u201d"]{4,200})[\u201d"][^[]{0,60})?\[([\w\u4e00-\u9fff]{1,4}-ch\d+:\d+)\](?!\()/g

  let processedText = text.replace(
    citeRegex,
    (_match: string, quoted: string | undefined, label: string) => {
      const ref = (refs && refs[label]) ? refs[label] : null
      let displayLabel: string
      if (label.startsWith('src-')) {
        // 研究文献: 优先用实际朝代，否则显示"研"
        displayLabel = ref?.dynasty || '研'  // 研
      } else {
        displayLabel = label.split('-ch')[0]
      }

      if (!ref) {
        return `<span class="ew-citation-badge ew-citation-missing" title="未找到证据: ${escapeAttr(label)}">${escapeAttr(displayLabel)}</span>`
      }

      const quote = (quoted || '').trim()
      const quoteAttr = quote ? ` data-quote="${escapeAttr(quote)}"` : ''
      const quotePrefix = quote ? `"${escapeHtml(quote)}"` : ''
      return `${quotePrefix}<span class="ew-citation-badge" data-ref-label="${escapeAttr(label)}"${quoteAttr} title="${escapeAttr(ref.chapterTitle || '')} 段落${ref.paragraphIndex ?? 0}">${escapeAttr(displayLabel)}</span>`
    }
  )

  // Step 3: 漏网旧格式短ID [明3][宋2][唐1][清5] → 灰色不可点击
  // B-151: LLM 偶尔仍按旧习惯写短ID，不在 evidence_refs 中，只能灰显
  processedText = processedText.replace(
    /\[([\w一-鿿]{1,4}\d+)\]/g,
    (_match, shortId) => {
      // 跳过已处理的锚点标签（已被 Step 2 替换为 HTML）
      return `<span class="ew-citation-badge ew-citation-missing" title="旧格式引用: ${escapeAttr(shortId)}（证据未收录）">${escapeAttr(shortId)}</span>`
    }
  )

  // Markdown 渲染
  let html = ''
  try {
    html = marked.parse(processedText, { gfm: false, breaks: true }) as string
  } catch {
    html = processedText.replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/\n/g, '<br>')
  }

  return html
})

function escapeAttr(text: string): string {
  return text
    .replace(/&/g, '&amp;')
    .replace(/"/g, '&quot;')
    .replace(/“/g, '&#8220;')
    .replace(/”/g, '&#8221;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/\n/g, ' ')
    .replace(/\r/g, '')
}

function escapeHtml(text: string): string {
  return text
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
}

function handleCitationClick(event: Event) {
  const el = event.target instanceof Element ? event.target : (event.target as Node).parentElement
  if (!el) return
  const target = el.closest('.ew-citation-badge') as HTMLElement | null
  if (!target) return

  if (target.classList.contains('ew-citation-missing')) {
    return
  }

  const refs = displayEvidenceRefs.value
  let ref: EvidenceRefData | null = null

  // 按锚点标签精确查找
  const refLabel = target.getAttribute('data-ref-label')
  if (refLabel && refs) {
    ref = refs[refLabel] || null
  }

  if (!ref) {
    console.warn(`[EvolutionWorkspace] Citation lookup failed. label=${refLabel}`, Object.keys(refs || {}).slice(0, 10))
    return
  }

  // B-151: 尝试在 excerpt 中高亮 LLM 写的引文
  // 弹窗展示用 excerptFull（完整不截断），fallback 到 excerpt
  const fullText = ref.excerptFull || ref.excerpt
  const quoteFromBadge = target.getAttribute('data-quote') || ''
  let highlightedHtml = escapeHtml(fullText)
  if (quoteFromBadge && quoteFromBadge.length >= 4) {
    // 策略1: 精确匹配
    let idx = fullText.indexOf(quoteFromBadge)
    // 策略2: 去空白匹配（与 TextReaderView 一致）
    if (idx < 0) {
      const normExcerpt = fullText.replace(/\s+/g, '')
      const normQuote = quoteFromBadge.replace(/\s+/g, '')
      const normIdx = normExcerpt.indexOf(normQuote)
      if (normIdx >= 0) {
        // 将归一化位置映射回原始文本
        let origPos = 0, normPos = 0
        while (normPos < normIdx && origPos < fullText.length) {
          if (/\s/.test(fullText[origPos])) { origPos++; continue }
          normPos++; origPos++
        }
        idx = origPos
      }
    }
    // B-152f: 策略3: 前缀匹配兜底 — LLM 引文稍有改写/省略号时，用前10字定位起点。
    // 宁可不高亮，绝不高亮错：只在命中时包 mark，匹配失败静默降级为不高亮
    if (idx < 0) {
      const normExcerpt = fullText.replace(/\s+/g, '')
      const normQuote = quoteFromBadge.replace(/\s+/g, '')
      const head = normQuote.slice(0, 10)
      if (head) {
        const normIdx = normExcerpt.indexOf(head)
        if (normIdx >= 0) {
          let origPos = 0, normPos = 0
          while (normPos < normIdx && origPos < fullText.length) {
            if (/\s/.test(fullText[origPos])) { origPos++; continue }
            normPos++; origPos++
          }
          idx = origPos
        }
      }
    }
    if (idx >= 0) {
      const matchLen = Math.min(quoteFromBadge.length, fullText.length - idx)
      const before = escapeHtml(fullText.slice(0, idx))
      const match = escapeHtml(fullText.slice(idx, idx + matchLen))
      const after = escapeHtml(fullText.slice(idx + matchLen))
      highlightedHtml = before + '<mark class="ew-citation-highlight">' + match + '</mark>' + after
    }
  }

  evidenceModalExcerpt.value = fullText
  evidenceModalExcerptHtml.value = highlightedHtml
  evidenceModalChapterTitle.value = ref.chapterTitle
  evidenceModalChapterNumber.value = typeof ref.chapterNumber === 'string'
    ? parseInt(ref.chapterNumber, 10)
    : ref.chapterNumber
  evidenceModalParagraphIndex.value = ref.paragraphIndex
  evidenceModalDynasty.value = ref.dynasty || ''
  evidenceModalSourceType.value = ref.sourceType || ''
  // 存储 excerpt 用于跳转到原文；B-152n: 同时保存 LLM 引文供跳转高亮首选
  evidenceModalQuotedText.value = fullText
  evidenceModalQuote.value = quoteFromBadge
  const srcLabel = ref.sourceType === 'research_literature' ? '[研究文献]' : ''
  const dynastyLabel = ref.dynasty ? `[${ref.dynasty}代]` : ''
  evidenceModalTitle.value = ref.chapterTitle
    ? `${dynastyLabel} ${srcLabel} ${ref.chapterTitle} 段落${ref.paragraphIndex}`
    : `来源段落${ref.paragraphIndex}`
  evidenceModalVisible.value = true
}

// 
function tryRestoreInterruptState() {
  const persisted = evolutionStore.restoreInterruptState()
  if (!persisted) return
  evolutionStore.applyPersistedState(persisted)

  // 
  const tabLabelMap: Record<string, string> = {
    confirm_motifs: '确认母题分解',
    select_hypothesis: '选择演化假设',
    decide_next_action: '决定后续方向',
  }
  const tabLabel = tabLabelMap[persisted.interruptType] || '演化分析'
  workspaceStore.openTab('evolution-workspace', tabLabel, {
    trackingTarget: persisted.trackingTarget,
    interruptType: persisted.interruptType,
  })
}

function handleStartEvolution() {
  // Triggered when user sends a tracking question from AgentChat
}
</script>

<template>
  <div class="evolution-workspace">
    <!-- Phase progress · 极简横向步进器 -->
    <div class="ew-progress">
      <div class="ew-stepper">
        <template v-for="(p, idx) in ['E1','E2','E3','E4','E5','E6','E7','E8']" :key="p">
          <div
            class="ew-phase-dot"
            :class="{
              active: displayPhase.startsWith(p),
              done: isPhaseDone(p),
              interrupt: displayPhase === `${p}_interrupt`.toLowerCase(),
            }"
            :title="phaseLabels[`${p.toLowerCase()}_decompose`] || p"
          >
            <svg v-if="isPhaseDone(p) && !displayPhase.startsWith(p)" class="ew-phase-check" viewBox="0 0 16 16" fill="none">
              <path d="M3.5 8.5L6.5 11.5L12.5 4.5" stroke="currentColor" stroke-width="2.2"
                stroke-linecap="round" stroke-linejoin="round" />
            </svg>
            <span v-else class="ew-phase-label">{{ p }}</span>
          </div>
          <div
            v-if="idx < 7"
            class="ew-phase-connector"
            :class="{ filled: isPhaseDone(p) && (isPhaseDone(['E1','E2','E3','E4','E5','E6','E7','E8'][idx + 1]) || displayPhase.startsWith(['E1','E2','E3','E4','E5','E6','E7','E8'][idx + 1])) }"
          />
        </template>
      </div>
      <button class="ew-close-btn" @click="handleCloseClick" aria-label="关闭">
        <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.4" stroke-linecap="round">
          <line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/>
        </svg>
      </button>
    </div>

    <div class="ew-status-text">
      <span class="ew-status-dot" :class="{ live: isLive }" />
      <span class="ew-status-label">{{ phaseLabel }}</span>
    </div>

    <!-- 可滚动内容区：步进器/状态固定置顶，下方内容可滚动，底部不被遮挡 -->
    <div class="ew-scroll-area">
    <!-- Idle state -->
    <div v-if="displayPhase === 'idle'" class="ew-idle">
      在右侧Agent Panel中输入演化追踪问题开始分析
    </div>

    <!-- Running states (E1-E8 except interrupt) -->
    <EvolutionProgress
      v-if="displayPhase !== 'idle' && displayPhase !== 'error' && !evolutionStore.isAwaitingUserInput && displayPhase !== 'complete'"
      :phase="displayPhase"
      :status-steps="agentStore.statusSteps"
      :is-live="isLive"
      @cancel="handleCancel"
    />

    <!-- U-12: Interrupt #1 - Motif editor component -->
    <MotifEditor
      v-if="displayPhase === 'E1_interrupt'"
      :motifs="evolutionStore.motifs"
      :suggested-additions="evolutionStore.suggestedAdditions"
      :tracking-target="evolutionStore.trackingTarget"
      :new-motif-name="newMotifName"
      :new-motif-type="newMotifType"
      :countdown-remaining="countdownRemaining"
      :is-timed-out="isTimedOut"
      @edit-type="(id, type) => evolutionStore.editMotif(id, { type })"
      @edit-core="(id, core) => evolutionStore.editMotif(id, { core })"
      @delete-motif="evolutionStore.deleteMotif"
      @update:newMotifName="newMotifName = $event"
      @update:newMotifType="newMotifType = $event"
      @add-motif="addNewMotif"
      @add-suggested="(s) => evolutionStore.addMotif(s.name, s.type, s.core)"
      @confirm="submitResume"
      @restart="handleRestartEvolution"
    />

    <!-- U-12: Interrupt #2 - Hypothesis selector component -->
    <HypothesisSelector
      v-if="displayPhase === 'E6_interrupt'"
      :hypotheses="(evolutionStore.candidateHypotheses as any[]).map(h => ({ ...h, selected: (h as SelectedHypothesis).selected }))"
      :allow-multi-select="evolutionStore.allowMultiSelect"
      :countdown-remaining="countdownRemaining"
      :is-timed-out="isTimedOut"
      @toggle="evolutionStore.toggleHypothesisSelection"
      @confirm="evolutionStore.confirmHypothesisSelection(); submitResume()"
      @restart="handleRestartEvolution"
    />

    <!-- U-12: Interrupt #3 - Action decider component -->
    <ActionDecider
      v-if="displayPhase === 'E7_interrupt'"
      :options="evolutionStore.decisionOptions"
      :summary="evolutionStore.decisionSummary"
      :total-claims="evolutionStore.claimVerdicts.length"
      :insufficient-count="evolutionStore.claimVerdicts.filter(v => v.verdict === 'insufficient').length"
      :selected-option="evolutionStore.selectedDecisionOption"
      :countdown-remaining="countdownRemaining"
      :is-timed-out="isTimedOut"
      @select="evolutionStore.selectDecisionOption"
      @confirm="submitResume"
      @restart="handleRestartEvolution"
    />

    <!-- Complete: E8 / final presentation -->
    <div v-if="displayPhase === 'complete' || displayPhase === 'E8_synthesize'" class="ew-complete">
      <!-- Sub-tab bar -->
      <div class="ew-subtabs">
        <button
          :class="{ active: activeSubTab === 'matrix' }"
          @click="activeSubTab = 'matrix'"
        >母题矩阵</button>
        <button
          :class="{ active: activeSubTab === 'phylogeny' }"
          @click="activeSubTab = 'phylogeny'"
        >谱系图</button>
        <button
          :class="{ active: activeSubTab === 'evidence' }"
          @click="activeSubTab = 'evidence'"
        >证据卡片</button>
        <button
          :class="{ active: activeSubTab === 'narrative' }"
          @click="activeSubTab = 'narrative'"
        >综合叙述</button>
      </div>

      <!-- Motif matrix -->
      <div v-if="activeSubTab === 'matrix'" class="ew-subpanel">
        <h4 v-if="!displayMotifMatrix.rows || displayMotifMatrix.rows.length === 0">
          暂无母题矩阵数据
          <span style="font-weight:normal;color:var(--color-text-secondary);font-size:0.7rem">
            (rows={{ String(displayMotifMatrix.rows) }},
             cols={{ (displayMotifMatrix.columns || []).length }},
             cells={{ (displayMotifMatrix.cells || []).length }},
             nonEmpty={{ (displayMotifMatrix.cells || []).filter(c => c.occurrenceCount > 0).length }})
          </span>
        </h4>
        <div
          v-if="displayMotifMatrix.rows && displayMotifMatrix.rows.length > 0
            && displayMotifMatrix.cells && displayMotifMatrix.cells.length > 0
            && displayMotifMatrix.cells.every(c => c.status === 'absent')"
          class="ew-matrix-empty-warn"
        >
          所有{{ displayMotifMatrix.cells.length }}个单元格均为空
           (occurrenceCount=0)
        </div>
        <table v-else class="ew-matrix-table">
          <thead>
            <tr>
              <th>朝代</th>
              <th v-for="col in displayMotifMatrix.columns" :key="col">{{ col }}</th>
            </tr>
          </thead>
          <tbody>
            <tr v-for="row in displayMotifMatrix.rows" :key="row">
              <td class="ew-matrix-row-header">{{ row }}</td>
              <td
                v-for="col in displayMotifMatrix.columns"
                :key="col"
                class="ew-matrix-cell"
                :class="getCellClass(row, col)"
                :title="getCellTooltip(row, col)"
                @click="showCellEvidence(row, col)"
              >
                {{ getCellValue(row, col) }}
              </td>
            </tr>
          </tbody>
        </table>
      </div>

      <!-- Phylogeny graph -->
      <div v-if="activeSubTab === 'phylogeny'" class="ew-subpanel">
        <div v-if="!displayPhylogeny.nodes || displayPhylogeny.nodes.length === 0">暂无谱系数据</div>
        <div v-else ref="phyloChartRef" class="ew-phylo-chart"></div>
      </div>

      <!-- Evidence cards -->
      <div v-if="activeSubTab === 'evidence'" class="ew-subpanel">
        <div v-if="displayClaimVerdicts.length === 0">暂无证据数据</div>
        <div
          v-for="verdict in displayClaimVerdicts"
          :key="verdict.claimId"
          class="ew-evidence-card"
          :class="'verdict-' + verdict.verdict"
        >
          <div class="ew-verdict-badge">{{ verdict.verdict === 'supported' ? '已证实' : verdict.verdict === 'refuted' ? '已否定' : verdict.verdict === 'insufficient' ? '证据不足' : '待验证' }}</div>
          <div class="ew-claim-statement">{{ verdict.statement }}</div>
          <div class="ew-claim-explanation">{{ verdict.explanation }}</div>
          <div v-if="verdict.evidencePro && verdict.evidencePro.length > 0" class="ew-evidence-sources">
            <span class="ew-evidence-label">支持证据:</span>
            <a
              v-for="(ev, i) in verdict.evidencePro"
              :key="'pro-' + i"
              href="#"
              class="ew-evidence-link"
              @click.prevent="navigateToSource(ev as EvidenceRef)"
            >
              <span class="ew-evidence-link-title">{{ ev.chapterTitle || '未命名章节' }}</span>
              <span v-if="ev.paragraphIndex !== undefined && ev.paragraphIndex >= 0" class="ew-evidence-link-para">段落 {{ ev.paragraphIndex }}</span>
              <span class="ew-evidence-link-action">查看原文</span>
            </a>
          </div>
          <div v-if="verdict.evidenceCon && verdict.evidenceCon.length > 0" class="ew-evidence-sources">
            <span class="ew-evidence-label">反对证据:</span>
            <a
              v-for="(ev, i) in verdict.evidenceCon"
              :key="'con-' + i"
              href="#"
              class="ew-evidence-link"
              @click.prevent="navigateToSource(ev as EvidenceRef)"
            >
              <span class="ew-evidence-link-title">{{ ev.chapterTitle || '未命名章节' }}</span>
              <span v-if="ev.paragraphIndex !== undefined && ev.paragraphIndex >= 0" class="ew-evidence-link-para">段落 {{ ev.paragraphIndex }}</span>
              <span class="ew-evidence-link-action">查看原文</span>
            </a>
          </div>
        </div>
      </div>

      <!-- Narrative -->
      <div v-if="activeSubTab === 'narrative'" class="ew-subpanel">
        <div v-if="!displayNarrative">暂无综合叙述</div>
        <div v-else-if="!displayEvidenceRefs || Object.keys(displayEvidenceRefs).length === 0" class="ew-no-refs-warning">
          未加载到证据引用数据。叙述中的引用标记将无法点击查看原文。
        </div>
        <div v-else class="ew-narrative-text" v-html="narrativeHtml" @click="handleCitationClick" />
      </div>
    </div>

    <!-- Error state -->
    <div v-if="displayPhase === 'error'" class="ew-error">
      演化分析出错
      <button @click="evolutionStore.resetLiveState()">重新开始</button>
    </div>
    </div><!-- /ew-scroll-area -->

    <!-- Close confirmation modal -->
    <Modal
      :visible="showCloseConfirm"
      title="确认关闭"
      @update:visible="showCloseConfirm = false"
    >
      <p>演化分析正在进行中，确定要关闭吗？中断后数据将丢失。</p>
      <template #footer>
        <button class="ew-cancel-btn" @click="showCloseConfirm = false">取消</button>
        <button class="ew-confirm-close-btn" @click="confirmClose">确认关闭</button>
      </template>
    </Modal>

    <!-- Evidence excerpt popup modal -->
    <Modal
      :visible="evidenceModalVisible"
      :title="evidenceModalTitle"
      width="680px"
      @update:visible="evidenceModalVisible = false"
    >
      <div class="ew-evidence-modal-body">
        <div v-if="evidenceModalDynasty || evidenceModalSourceType" class="ew-evidence-meta">
          <span v-if="evidenceModalDynasty" class="ew-evidence-tag ew-evi-dynasty">[{{ evidenceModalDynasty }}代]</span>
          <span v-if="evidenceModalSourceType === 'research_literature'" class="ew-evidence-tag ew-evi-research">研究文献</span>
          <span v-else class="ew-evidence-tag ew-evi-primary">主文献</span>
        </div>
        <blockquote class="ew-evidence-excerpt" v-html="evidenceModalExcerptHtml" />
        <p class="ew-evidence-source">
          {{ evidenceModalChapterTitle }}
          <template v-if="evidenceModalParagraphIndex !== undefined"> · 段落 {{ evidenceModalParagraphIndex }}</template>
        </p>
      </div>
      <template #footer>
        <button class="ew-cancel-btn" @click="evidenceModalVisible = false">关闭</button>
        <button v-if="evidenceModalChapterNumber > 0" class="ew-jump-btn" @click="jumpToSourceFromModal">跳转到原文</button>
      </template>
    </Modal>
  </div>
</template>

<style scoped>
.evolution-workspace {
  display: flex;
  flex-direction: column;
  height: 100%;
  overflow: hidden;
  padding: var(--space-4);
  background: var(--color-bg-primary);
}

/* 可滚动内容区：步进器/状态固定置顶，下方内容滚动，底部不被遮挡 */
.ew-scroll-area {
  flex: 1;
  min-height: 0;
  overflow-y: auto;
  padding-right: var(--space-1);
  scrollbar-width: thin;
  scrollbar-color: var(--color-border-dark) transparent;
}
.ew-scroll-area::-webkit-scrollbar {
  width: 6px;
}
.ew-scroll-area::-webkit-scrollbar-thumb {
  background: var(--color-border-dark);
  border-radius: var(--radius-full);
}
.ew-scroll-area::-webkit-scrollbar-track {
  background: transparent;
}

/* ---------- Phase progress · 横向步进器 ---------- */
.ew-progress {
  display: flex;
  align-items: center;
  gap: var(--space-3);
  margin-bottom: var(--space-2);
  padding: var(--space-2) var(--space-3);
  background: var(--color-bg-secondary);
  border: 1px solid var(--color-border-light);
  border-radius: var(--radius);
}

.ew-stepper {
  display: flex;
  align-items: center;
  gap: 0;
  flex: 1;
}

.ew-phase-dot {
  width: 26px;
  height: 26px;
  flex-shrink: 0;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 0.66rem;
  font-weight: 600;
  border-radius: 50%;
  background: var(--color-bg-primary);
  color: var(--color-text-tertiary);
  border: 1.5px solid var(--color-border);
  position: relative;
  z-index: 1;
  transition: background-color var(--dur) var(--ease),
              color var(--dur) var(--ease),
              border-color var(--dur) var(--ease),
              transform var(--dur) var(--ease);
}

.ew-phase-label {
  font-family: var(--font-mono);
  letter-spacing: 0.02em;
}

.ew-phase-check {
  width: 14px;
  height: 14px;
  color: var(--color-accent);
}

.ew-phase-dot.done {
  background: var(--color-bg-primary);
  color: var(--color-accent);
  border-color: var(--color-accent);
}

.ew-phase-dot.active {
  background: var(--color-bg-primary);
  color: var(--color-accent);
  border-color: var(--color-accent);
  box-shadow: 0 0 0 4px var(--color-accent-ring);
  animation: ew-dot-pulse 1.8s var(--ease) infinite;
}

@keyframes ew-dot-pulse {
  0%, 100% { box-shadow: 0 0 0 4px var(--color-accent-ring); }
  50% { box-shadow: 0 0 0 6px rgba(91, 155, 213, 0.05); }
}

.ew-phase-dot.interrupt {
  background: var(--color-warning);
  color: #fff;
  border-color: var(--color-warning);
  animation: ew-dot-blink 1.2s var(--ease) infinite;
}

@keyframes ew-dot-blink {
  0%, 100% { opacity: 1; }
  50% { opacity: 0.55; }
}

.ew-phase-connector {
  flex: 1;
  height: 2px;
  min-width: 8px;
  background: var(--color-border);
  border-radius: var(--radius-full);
  transition: background-color var(--dur) var(--ease);
}

.ew-phase-connector.filled {
  background: var(--color-accent);
}

.ew-close-btn {
  flex-shrink: 0;
  background: transparent;
  border: 1px solid transparent;
  color: var(--color-text-tertiary);
  cursor: pointer;
  width: 26px;
  height: 26px;
  border-radius: 50%;
  display: flex;
  align-items: center;
  justify-content: center;
  transition: background-color var(--dur-fast) var(--ease),
              color var(--dur-fast) var(--ease),
              border-color var(--dur-fast) var(--ease);
}

.ew-close-btn:hover {
  background: var(--color-error-light);
  color: var(--color-error);
  border-color: var(--color-error);
}

.ew-status-text {
  display: flex;
  align-items: center;
  gap: var(--space-2);
  font-size: 0.76rem;
  color: var(--color-text-secondary);
  margin-bottom: var(--space-3);
  padding-left: var(--space-2);
}

.ew-status-dot {
  width: 7px;
  height: 7px;
  border-radius: 50%;
  background: var(--color-text-tertiary);
  flex-shrink: 0;
}

.ew-status-dot.live {
  background: var(--color-accent);
  animation: ew-dot-pulse 1.8s var(--ease) infinite;
}

.ew-status-label {
  line-height: 1.4;
}

/* ---------- Idle / running states ---------- */
.ew-idle {
  text-align: center;
  padding: var(--space-8);
  color: var(--color-text-tertiary);
  font-size: 0.85rem;
  background: var(--color-bg-secondary);
  border: 1px dashed var(--color-border);
  border-radius: var(--radius);
  margin: var(--space-4) 0;
}

/* ew-cancel-btn, ew-matrix-empty-warn 等仍被使用 */

/* ---------- Interrupt states (delegated to components, but layout wrappers here) ---------- */
.ew-interrupt {
  flex: 1;
  overflow-y: auto;
  position: relative;
}

/* ---------- Complete state ---------- */
.ew-complete {
  flex: 1;
  display: flex;
  flex-direction: column;
  overflow: hidden;
}

.ew-subtabs {
  display: flex;
  gap: var(--space-1);
  margin-bottom: var(--space-3);
  border-bottom: 1px solid var(--color-border);
  padding: 0 var(--space-1);
}

.ew-subtabs button {
  padding: var(--space-2) var(--space-4);
  font-size: 0.75rem;
  background: none;
  border: none;
  border-bottom: 2px solid transparent;
  color: var(--color-text-secondary);
  cursor: pointer;
  font-weight: 500;
  margin-bottom: -1px;
  transition: color var(--dur-fast) var(--ease),
              border-color var(--dur-fast) var(--ease),
              background-color var(--dur-fast) var(--ease);
  border-radius: var(--radius-sm) var(--radius-sm) 0 0;
}

.ew-subtabs button:hover {
  color: var(--color-text-primary);
  background: var(--color-bg-hover);
}

.ew-subtabs button.active {
  color: var(--color-accent);
  border-bottom-color: var(--color-accent);
  background: var(--color-accent-light);
}

.ew-subpanel {
  flex: 1;
  overflow-y: auto;
  padding-right: var(--space-1);
}

/* ---------- Matrix table ---------- */
.ew-matrix-empty-warn {
  color: var(--color-warning);
  font-size: 0.78rem;
  margin-bottom: var(--space-2);
  padding: var(--space-2) var(--space-3);
  background: var(--color-warning-light);
  border: 1px solid var(--color-warning);
  border-radius: var(--radius-sm);
  font-weight: 500;
}

.ew-matrix-table {
  width: 100%;
  border-collapse: separate;
  border-spacing: 0;
  font-size: 0.75rem;
  border: 1px solid var(--color-border);
  border-radius: var(--radius-sm);
  overflow: hidden;
  box-shadow: var(--shadow-sm);
}

.ew-matrix-table th,
.ew-matrix-table td {
  padding: var(--space-2);
  border-bottom: 1px solid var(--color-border-light);
  border-right: 1px solid var(--color-border-light);
}

.ew-matrix-table th:last-child,
.ew-matrix-table td:last-child {
  border-right: none;
}

.ew-matrix-table tbody tr:last-child td {
  border-bottom: none;
}

.ew-matrix-table th {
  background: var(--color-bg-tertiary);
  color: var(--color-text-secondary);
  font-weight: 500;
  max-width: 160px;
  min-width: 80px;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: normal;
  word-break: keep-all;
  line-height: 1.3;
  vertical-align: top;
}

.ew-matrix-row-header {
  font-weight: 500;
  color: var(--color-text-primary);
  background: var(--color-bg-tertiary);
  width: 72px;
}

.ew-matrix-cell {
  cursor: pointer;
  text-align: center;
  font-family: var(--font-mono);
  font-size: 0.7rem;
  padding: var(--space-2) var(--space-1) !important;
  max-width: 160px;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: normal;
  word-break: break-all;
  line-height: 1.3;
  vertical-align: top;
  background: var(--color-bg-primary);
  transition: background-color var(--dur-fast) var(--ease),
              color var(--dur-fast) var(--ease);
}

.ew-matrix-cell:hover {
  background: var(--color-accent-light);
}

.ew-matrix-cell.absent {
  color: var(--color-text-disabled);
  background: var(--color-bg-secondary);
  opacity: 0.7;
}

.ew-matrix-cell.absent:hover {
  background: var(--color-bg-hover);
}

.ew-matrix-cell.stable {
  color: #7FB0DC;
  font-weight: 500;
  background: var(--color-bg-primary);
  box-shadow: inset 2px 0 0 rgba(91, 155, 213, 0.4);
}

.ew-matrix-cell.stable:hover {
  background: var(--color-accent-light);
}

.ew-matrix-cell.changed {
  color: #C4AE62;
  font-weight: 500;
  background: var(--color-bg-primary);
  box-shadow: inset 2px 0 0 rgba(196, 174, 98, 0.4);
}

.ew-matrix-cell.changed:hover {
  background: var(--color-warning-light);
}

/* ---------- Evidence cards ---------- */
.ew-evidence-card {
  padding: var(--space-3) var(--space-4);
  margin-bottom: var(--space-2);
  border-radius: var(--radius-sm);
  border-left: 3px solid var(--color-border);
  background: var(--color-bg-primary);
  border-top: 1px solid var(--color-border-light);
  border-right: 1px solid var(--color-border-light);
  border-bottom: 1px solid var(--color-border-light);
  box-shadow: var(--shadow-sm);
  transition: box-shadow var(--dur-fast) var(--ease),
              transform var(--dur-fast) var(--ease);
}

.ew-evidence-card:hover {
  box-shadow: var(--shadow);
  transform: translateY(-1px);
}

.ew-evidence-card.verdict-supported {
  border-left-color: var(--color-verdict-supported);
}

.ew-evidence-card.verdict-insufficient {
  border-left-color: var(--color-verdict-insufficient);
}

.ew-evidence-card.verdict-refuted {
  border-left-color: var(--color-verdict-refuted);
}

.ew-verdict-badge {
  font-size: 0.68rem;
  margin-bottom: var(--space-2);
  display: inline-block;
  padding: 1px var(--space-2);
  border-radius: var(--radius-full);
  font-weight: 600;
  letter-spacing: 0.02em;
}

.ew-evidence-card.verdict-supported .ew-verdict-badge {
  background: var(--color-success-light);
  color: var(--color-verdict-supported);
}

.ew-evidence-card.verdict-insufficient .ew-verdict-badge {
  background: var(--color-warning-light);
  color: var(--color-verdict-insufficient);
}

.ew-evidence-card.verdict-refuted .ew-verdict-badge {
  background: var(--color-error-light);
  color: var(--color-verdict-refuted);
}

.ew-claim-statement {
  font-size: 0.82rem;
  color: var(--color-text-primary);
  margin-bottom: var(--space-1);
  font-weight: 600;
  line-height: 1.55;
}

.ew-claim-explanation {
  font-size: 0.76rem;
  color: var(--color-text-secondary);
  line-height: 1.65;
  margin-bottom: 0;
}

.ew-evidence-sources {
  margin-top: var(--space-2);
  padding-top: var(--space-2);
  border-top: 1px solid var(--color-border-light);
}

.ew-evidence-label {
  font-size: 0.7rem;
  color: var(--color-text-tertiary);
  margin-right: var(--space-1);
  font-weight: 500;
}

.ew-evidence-link {
  display: flex;
  align-items: center;
  gap: var(--space-2);
  font-size: 0.72rem;
  color: var(--color-text-secondary);
  text-decoration: none;
  margin: var(--space-1) 0;
  padding: var(--space-1) var(--space-2);
  background: var(--color-bg-secondary);
  border: 1px solid var(--color-border-light);
  border-radius: var(--radius-sm);
  transition: background-color var(--dur-fast) var(--ease),
              border-color var(--dur-fast) var(--ease),
              transform var(--dur-fast) var(--ease);
}

.ew-evidence-link:hover {
  background: var(--color-accent-light);
  border-color: var(--color-accent);
  transform: translateX(2px);
}

.ew-evidence-link-title {
  color: var(--color-accent);
  font-weight: 500;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
  max-width: 220px;
  flex-shrink: 1;
  min-width: 0;
}

.ew-evidence-link-para {
  color: var(--color-text-tertiary);
  font-size: 0.68rem;
  flex-shrink: 0;
  white-space: nowrap;
}

.ew-evidence-link-action {
  margin-left: auto;
  color: var(--color-accent);
  font-size: 0.66rem;
  flex-shrink: 0;
  opacity: 0.7;
  transition: opacity var(--dur-fast) var(--ease);
}

.ew-evidence-link:hover .ew-evidence-link-action {
  opacity: 1;
}

/* ---------- Narrative ---------- */
.ew-narrative-text {
  font-size: 0.85rem;
  line-height: 1.85;
  color: var(--color-text-primary);
  padding: var(--space-5) var(--space-6);
  background: var(--color-bg-primary);
  border: 1px solid var(--color-border);
  border-radius: var(--radius);
  box-shadow: var(--shadow-sm);
}

/* 综合报告 Markdown 排版 */
.ew-narrative-text :deep(h1),
.ew-narrative-text :deep(h2),
.ew-narrative-text :deep(h3) {
  font-weight: 600;
  color: var(--color-text-primary);
  line-height: 1.4;
  margin-top: 1.5em;
  margin-bottom: 0.6em;
}

.ew-narrative-text :deep(h1) {
  font-size: 1.15rem;
  padding-bottom: 0.4em;
  border-bottom: 1px solid var(--color-border);
}

.ew-narrative-text :deep(h2) {
  font-size: 1rem;
  padding-bottom: 0.3em;
  border-bottom: 1px solid var(--color-border-light);
}

.ew-narrative-text :deep(h3) {
  font-size: 0.9rem;
}

.ew-narrative-text :deep(p) {
  margin: 0 0 0.8em 0;
}

.ew-narrative-text :deep(ul),
.ew-narrative-text :deep(ol) {
  margin: 0.4em 0 0.8em 1.5em;
  padding-left: 0.5em;
}

.ew-narrative-text :deep(li) {
  margin-bottom: 0.3em;
}

.ew-narrative-text :deep(blockquote) {
  margin: 0.8em 0;
  padding: 0.5em 1em;
  border-left: 3px solid var(--color-accent);
  background: var(--color-bg-secondary);
  border-radius: 0 var(--radius-sm) var(--radius-sm) 0;
  color: var(--color-text-secondary);
  font-size: 0.82rem;
}

.ew-narrative-text :deep(strong) {
  font-weight: 600;
  color: var(--color-text-primary);
}

.ew-narrative-text :deep(table) {
  width: 100%;
  border-collapse: collapse;
  margin: 0.8em 0;
  font-size: 0.8rem;
}

.ew-narrative-text :deep(th),
.ew-narrative-text :deep(td) {
  padding: 0.4em 0.6em;
  border: 1px solid var(--color-border-light);
  text-align: left;
}

.ew-narrative-text :deep(th) {
  background: var(--color-bg-secondary);
  font-weight: 600;
}

.ew-narrative-text :deep(hr) {
  border: none;
  border-top: 1px solid var(--color-border-light);
  margin: 1.2em 0;
}

.ew-no-refs-warning {
  padding: var(--space-3) var(--space-4);
  font-size: 0.78rem;
  color: var(--color-warning);
  background: var(--color-warning-light);
  border: 1px solid var(--color-warning);
  border-radius: var(--radius-sm);
  margin-bottom: var(--space-3);
  font-weight: 500;
}

.ew-phylogeny {
  display: flex;
  flex-wrap: wrap;
  gap: var(--space-2);
}

.ew-phylo-node {
  padding: var(--space-2) var(--space-3);
  background: var(--color-bg-tertiary);
  border: 1px solid var(--color-border);
  border-radius: var(--radius-sm);
  font-size: 0.75rem;
  color: var(--color-text-primary);
}

.ew-phylo-chart {
  width: 100%;
  height: 400px;
  background: var(--color-bg-primary);
  border: 1px solid var(--color-border);
  border-radius: var(--radius);
  box-shadow: var(--shadow-sm);
}

/* ---------- Error state ---------- */
.ew-error {
  text-align: center;
  padding: var(--space-8);
  color: var(--color-error);
  font-size: 0.85rem;
  background: var(--color-error-light);
  border: 1px solid var(--color-error);
  border-radius: var(--radius);
  margin: var(--space-4) 0;
  display: flex;
  align-items: center;
  justify-content: center;
  gap: var(--space-2);
}

.ew-error button {
  margin-left: var(--space-2);
}

/* ---------- Shared buttons ---------- */
.ew-cancel-btn,
.ew-confirm-close-btn {
  padding: var(--space-2) var(--space-4);
  font-size: 0.75rem;
  border-radius: var(--radius-sm);
  cursor: pointer;
  font-weight: 500;
  transition: background-color var(--dur-fast) var(--ease),
              color var(--dur-fast) var(--ease),
              border-color var(--dur-fast) var(--ease),
              box-shadow var(--dur-fast) var(--ease);
}

.ew-cancel-btn {
  background: var(--color-bg-primary);
  color: var(--color-text-primary);
  border: 1px solid var(--color-border);
}

.ew-cancel-btn:hover {
  background: var(--color-bg-hover);
  border-color: var(--color-border-dark);
}

.ew-confirm-close-btn {
  background: var(--color-error);
  color: #fff;
  border: 1px solid var(--color-error);
}

.ew-confirm-close-btn:hover {
  filter: brightness(1.05);
  box-shadow: var(--shadow-focus);
}

/* S-02: Countdown and timeout */
.ew-countdown {
  padding: var(--space-2) var(--space-3);
  margin-bottom: var(--space-3);
  font-size: 0.72rem;
  color: var(--color-text-secondary);
  background: var(--color-bg-secondary);
  border: 1px solid var(--color-border);
  border-radius: var(--radius-sm);
  font-family: var(--font-mono);
  display: flex;
  align-items: center;
  gap: var(--space-2);
}

.ew-countdown-warn {
  color: var(--color-error);
  font-weight: 600;
  margin-left: var(--space-2);
}

.ew-timeout-overlay {
  position: absolute;
  inset: 0;
  display: flex;
  align-items: center;
  justify-content: center;
  background: rgba(26, 26, 26, 0.6);
  backdrop-filter: blur(2px);
  z-index: 10;
  border-radius: var(--radius);
}

.ew-timeout-card {
  text-align: center;
  padding: var(--space-6);
  background: var(--color-bg-primary);
  border: 1px solid var(--color-error);
  border-radius: var(--radius);
  max-width: 340px;
  box-shadow: var(--shadow-lg);
}

.ew-timeout-card h3 {
  font-size: 0.95rem;
  color: var(--color-error);
  margin: 0 0 var(--space-2);
  font-weight: 600;
}

.ew-timeout-card p {
  font-size: 0.78rem;
  color: var(--color-text-secondary);
  margin: 0 0 var(--space-4);
  line-height: 1.5;
}

.ew-restart-btn {
  padding: var(--space-2) var(--space-5);
  font-size: 0.78rem;
  background: var(--color-accent);
  color: #fff;
  border: 1px solid var(--color-accent);
  border-radius: var(--radius-sm);
  cursor: pointer;
  font-weight: 500;
  transition: background-color var(--dur) var(--ease),
              box-shadow var(--dur) var(--ease);
}

.ew-restart-btn:hover {
  background: var(--color-accent-hover);
  box-shadow: var(--shadow-focus);
}

/* ---------- Citation badges (narrative, via :deep) ---------- */
.ew-narrative-text :deep(.ew-citation-badge) {
  display: inline-block;
  color: var(--color-accent);
  background: var(--color-accent-light);
  cursor: pointer;
  font-size: 0.7rem;
  font-weight: 600;
  margin: 0 2px;
  padding: 0 var(--space-2);
  border-radius: var(--radius-sm);
  text-decoration: none;
  border: 1px solid var(--color-accent);
  transition: background-color var(--dur-fast) var(--ease),
              color var(--dur-fast) var(--ease),
              border-color var(--dur-fast) var(--ease);
  white-space: nowrap;
  line-height: 1.5;
  vertical-align: 1px;
}

.ew-narrative-text :deep(.ew-citation-badge:hover) {
  background: var(--color-accent);
  color: #fff;
  border-color: var(--color-accent);
  text-decoration: none;
}

.ew-narrative-text :deep(.ew-citation-missing) {
  color: var(--color-text-tertiary);
  background: var(--color-bg-tertiary);
  border: 1px dashed var(--color-border);
  cursor: not-allowed;
  font-weight: 400;
}

.ew-narrative-text :deep(.ew-citation-missing:hover) {
  background: var(--color-bg-tertiary);
  color: var(--color-text-tertiary);
  border-color: var(--color-border);
}

/* ---------- Evidence popup metadata tags ---------- */
.ew-evidence-meta {
  display: flex;
  gap: var(--space-2);
  margin-bottom: var(--space-2);
  flex-wrap: wrap;
}

.ew-evidence-tag {
  font-size: 0.68rem;
  padding: 1px var(--space-2);
  border-radius: var(--radius-sm);
  font-weight: 500;
  border: 1px solid var(--color-border);
}

.ew-evi-dynasty {
  background: var(--color-accent-light);
  color: var(--color-accent);
  border-color: var(--color-accent);
}

.ew-evi-primary {
  background: var(--color-bg-tertiary);
  color: var(--color-text-secondary);
  border-color: var(--color-border);
}

.ew-evi-research {
  background: var(--color-bg-tertiary);
  color: var(--color-text-secondary);
  border-style: dashed;
}

/* ---------- Citation quote highlight in evidence popup ---------- */
.ew-evidence-excerpt :deep(mark.ew-citation-highlight) {
  background: var(--color-accent-light);
  color: var(--color-accent);
  padding: 1px 3px;
  border-radius: 2px;
  font-weight: 600;
  border-bottom: 1px solid var(--color-accent);
}

/* ---------- Evidence popup modal body ---------- */
.ew-evidence-modal-body {
  font-size: 0.82rem;
  color: var(--color-text-primary);
}

.ew-evidence-excerpt {
  margin: 0 0 var(--space-3);
  padding: var(--space-3) var(--space-4);
  background: var(--color-bg-secondary);
  border-left: 3px solid var(--color-accent);
  border-top: 1px solid var(--color-border-light);
  border-right: 1px solid var(--color-border-light);
  border-bottom: 1px solid var(--color-border-light);
  border-radius: 0 var(--radius-sm) var(--radius-sm) 0;
  font-size: 0.82rem;
  line-height: 1.7;
  color: var(--color-text-primary);
  white-space: pre-wrap;
  word-break: break-word;
}

.ew-evidence-source {
  font-size: 0.72rem;
  color: var(--color-text-tertiary);
  margin: 0;
  padding-top: var(--space-2);
  border-top: 1px solid var(--color-border-light);
}

.ew-jump-btn {
  padding: var(--space-2) var(--space-4);
  font-size: 0.75rem;
  background: var(--color-accent);
  color: #fff;
  border: 1px solid var(--color-accent);
  border-radius: var(--radius-sm);
  cursor: pointer;
  font-weight: 500;
  transition: background-color var(--dur) var(--ease),
              border-color var(--dur) var(--ease),
              box-shadow var(--dur) var(--ease);
}

.ew-jump-btn:hover {
  background: var(--color-accent-hover);
  border-color: var(--color-accent-hover);
  box-shadow: var(--shadow-focus);
}
</style>
