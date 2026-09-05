<script setup lang="ts">
import { ref, computed, watch, onMounted, onUnmounted, nextTick } from 'vue'
import { useSourceTreeStore } from '@/stores/sourceTree'
import { useAnnotationStore } from '@/stores/annotations'
import { useWorkspaceStore } from '@/stores/workspace'
import { useAgentStore } from '@/stores/agent'
import AnnotationPopup from '@/components/textreader/AnnotationPopup.vue'
import SelectionMenu from '@/components/textreader/SelectionMenu.vue'
import UserAnnotationForm from '@/components/textreader/UserAnnotationForm.vue'
import MarginaliaBubble from '@/components/textreader/MarginaliaBubble.vue'
import MarginaliaForm from '@/components/textreader/MarginaliaForm.vue'
import LayerPanel from '@/components/textreader/LayerPanel.vue'
import ExportPanel from '@/components/textreader/ExportPanel.vue'
import type { AnnotationCategory, AnnotationVisibility, Marginalia } from '@/types/annotation'
import type { ActiveSearch, SearchHighlight } from '@/types/literature'
import { computeJumpHighlights } from '@/utils/highlight'

const props = defineProps<{
  tabId?: string
  splitSide?: 'left' | 'right'
  // U-15: Hover linkage highlight ID from dynasty compare
  linkedHighlightId?: string | null
  // 对比页控制：旁注是否展开显示（false 时折叠为 hover 徽章）
  marginaliaExpanded?: boolean
  // 对比页控制：隐藏自身工具栏（标题/徽记/控制），因为标题已在对比工具栏展示
  hideToolbar?: boolean
}>()
const marginaliaExpandedInner = computed(() => props.marginaliaExpanded ?? true)

const sourceTreeStore = useSourceTreeStore()
const annotationStore = useAnnotationStore()
const workspaceStore = useWorkspaceStore()
const agentStore = useAgentStore()

// DEBUG: watch agent annotations
watch(() => annotationStore.agentAnnotations, (anns) => {
  const chNum = (workspaceStore.activeTab?.data as Record<string, unknown>)?.chapterNumber as number | undefined
  if (chNum) {
    const keys = Object.keys(anns).filter(k => k.startsWith(chNum + ':'))
    if (keys.length > 0) {
      const total = keys.reduce((s: number, k: string) => s + (anns[k]?.length || 0), 0)
      console.log('[TextReader] agentAnnotations updated: chapter', chNum, '-', keys.length, 'keys,', total, 'annotations')
    }
  }
}, { deep: true })

const currentTab = computed(() => {
  if (props.tabId) {
    return workspaceStore.tabs.find(t => t.id === props.tabId) || workspaceStore.activeTab
  }
  return workspaceStore.activeTab
})
const chapterNumber = computed(() => (currentTab.value?.data as Record<string, unknown>)?.chapterNumber as number | undefined)
const paragraphIndex = computed(() => (currentTab.value?.data as Record<string, unknown>)?.paragraphIndex as number | undefined)

// C06: 从综合叙述跳转时携带的引文高亮数据
interface CitationHighlight {
  text: string
}
const citationHighlight = computed<CitationHighlight | null>(() => {
  const cd = currentTab.value?.data as Record<string, unknown> | undefined
  return (cd?.citationHighlight as CitationHighlight) || null
})

function buildPassageKey(chNum: number, paraIdx: number): string {
  const base = `${chNum}:${paraIdx}`
  if (props.splitSide) {
    return `${base}:${props.splitSide}`
  }
  return base
}

const showAiAnnotations = computed({
  get: () => {
    const v = annotationStore.visibility.agent
    return v.person || v.location || v.event || v.term || v.motif || v.custom
  },
  set: (val: boolean) => {
    const t = { person: val, location: val, event: val, term: val, motif: val, custom: val }
    annotationStore.visibility = { user: { ...annotationStore.visibility.user }, agent: { ...t } }
    localStorage.setItem('white_snake_visibility_v2', JSON.stringify(annotationStore.visibility))
  },
})
const showAiInsights = ref(localStorage.getItem('white_snake_show_ai_insights') !== 'false')
watch(showAiInsights, (val) => {
  localStorage.setItem('white_snake_show_ai_insights', String(val))
})
// 跟随 agentStore.isAnnotationLoading：标注流式推送期间亮起，isComplete 时关闭
const isAnnotationLoading = computed(() => agentStore.isAnnotationLoading)
const annotationTimeout = ref<ReturnType<typeof setTimeout> | null>(null)

// Popup state
const popupVisible = ref(false)
const popupAnnotation = ref<Record<string, unknown> | null>(null)
const popupPassageKey = ref('')
const popupX = ref(0)
const popupY = ref(0)

// Selection menu state
const selectionMenuVisible = ref(false)
const selectionMenuX = ref(0)
const selectionMenuY = ref(0)
const selectedText = ref('')
const selectionStartChar = ref(0)
const selectionEndChar = ref(0)
const selectionPassageKey = ref('')

// User annotation form
const userAnnotationFormVisible = ref(false)
const preselectedCategory = ref<AnnotationCategory | null>(null)

// Marginalia form
const marginaliaFormVisible = ref(false)
const marginaliaFormX = ref(0)
const marginaliaFormY = ref(0)
const marginaliaPassageKey = ref('')
const marginaliaParagraphIndex = ref(0)

// Layer panel
const layerPanelVisible = ref(false)

// Export panel
const exportPanelVisible = ref(false)

// 工具栏展开/折叠：默认折叠（图例+控制按钮收起），点「≡」展开
const toolbarExpanded = ref(false)
// 窄面板（对比页）旁注折叠为徽章：是否显示气泡（hover/click 时展开）
const hoveredMargPara = ref<number | null>(null)
const pinnedMargPara = ref<number | null>(null)

// Annotation context menu (right-click on annotation tags)
const annCtxMenuVisible = ref(false)
const annCtxMenuX = ref(0)
const annCtxMenuY = ref(0)
const annCtxMenuSource = ref<'user' | 'agent'>('agent')
const annCtxMenuId = ref('')
const annCtxMenuPassageKey = ref('')

// Paragraphs for current chapter
const paragraphs = computed(() => {
  if (chapterNumber.value === undefined || chapterNumber.value === null) return []
  return sourceTreeStore.getChapterParagraphs(chapterNumber.value)
})

const chapterTitle = computed(() => {
  if (chapterNumber.value === undefined || chapterNumber.value === null) return ''
  const ch = sourceTreeStore.chapters.find(c => c.number === chapterNumber.value)
  return ch ? `${ch.dynasty} - ${ch.title}` : ''
})

const currentPassageKey = computed(() => {
  if (chapterNumber.value === undefined || chapterNumber.value === null || paragraphIndex.value === undefined) return ''
  return buildPassageKey(chapterNumber.value, paragraphIndex.value)
})

// Search highlight state (from sourceTreeStore, shared by user search and agent search)
const activeSearch = computed<ActiveSearch | null>(() => {
  return sourceTreeStore.activeSearch
})

function getParaSearchHighlights(paraIndex: number): SearchHighlight[] {
  if (!activeSearch.value || chapterNumber.value === undefined || chapterNumber.value === null) return []
  return activeSearch.value.highlights.filter(
    h => h.chapterNumber === chapterNumber.value && h.paragraphIndex === paraIndex
  )
}

// Active paragraph for highlight
const activeParagraphIdx = ref<number | null>(null)

// 搜索结果跳转的暂时性高亮：只在本次浏览生效，离开/切 tab 即清除，重进不再出现
const ephemeralHighlightKey = ref<string>('') // 非空时，mark.search-highlight 叠加临时闪烁样式
watch(() => (currentTab.value?.data as Record<string, unknown> | undefined)?.flashParagraph, (token) => {
  if (token && paragraphIndex.value !== undefined && paragraphIndex.value !== null) {
    // 生成一次性随机 key：只在这段 tab 生命周期内生效
    ephemeralHighlightKey.value = `eph_${Date.now()}`
    const tab = currentTab.value
    if (tab) delete (tab.data as Record<string, unknown>).flashParagraph
  }
}, { immediate: true })

// B-162: 延迟重算 — navigateToSource 在 SearchResultsView 里执行时，chapters 可能
// 还没加载完（targetPara=undefined → 三层兜底全跳过 → highlights 为空）。
// 这里在 paragraphs 加载后补算：当 activeSearch.ephemeral=true、highlights 为空、
// 且 pendingRecompute 有数据时，用已加载的段落文本重算高亮。
watch(
  () => [paragraphs.value, activeSearch.value] as const,
  ([paras, search]) => {
    if (!search?.ephemeral || !search.pendingRecompute || search.highlights.length > 0) return
    if (paras.length === 0) return
    const pr = search.pendingRecompute
    const targetPara = paras.find(p => p.index === pr.paragraphIndex)
    if (!targetPara) return
    const ranges = computeJumpHighlights(search.query, {
      chapterNumber: pr.chapterNumber,
      paragraphIndex: pr.paragraphIndex,
      excerpt: pr.excerpt,
      paragraphMatchPositions: pr.paragraphMatchPositions,
    }, targetPara)
    if (ranges.length === 0) return
    // 写回 activeSearch.highlights，清 pendingRecompute（一次性）
    const newHighlights: SearchHighlight[] = ranges.map(({ start, end }) => ({
      query: search.query,
      chapterNumber: pr.chapterNumber,
      paragraphIndex: pr.paragraphIndex,
      startChar: start,
      endChar: end,
      text: targetPara.text.slice(start, end),
    }))
    sourceTreeStore.activeSearch = {
      ...search,
      highlights: newHighlights,
      pendingRecompute: undefined,
    }
    if (import.meta.env.DEV) {
      console.debug(
        `[TR延迟重算] ch#${pr.chapterNumber} para#${pr.paragraphIndex} ` +
        `补算 ${newHighlights.length} 处高亮（chapters 加载滞后导致首次为空）`
      )
    }
  },
  { immediate: true },
)

// Annotation counts
const userAnnCount = computed(() => chapterNumber.value ? annotationStore.userAnnotationCount(chapterNumber.value) : 0)
const agentAnnCount = computed(() => chapterNumber.value ? annotationStore.agentAnnotationCount(chapterNumber.value) : 0)

// F02: Helper functions for marginalia/insights presence
function hasMarginalia(paraIndex: number): boolean {
  if (chapterNumber.value === undefined || chapterNumber.value === null) return false
  return annotationStore.marginaliaForKey(buildPassageKey(chapterNumber.value, paraIndex)).length > 0
}

function hasInsights(paraIndex: number): boolean {
  if (chapterNumber.value === undefined || chapterNumber.value === null) return false
  return annotationStore.insightsForKey(buildPassageKey(chapterNumber.value, paraIndex)).length > 0
}

function hasMarginaliaOrInsights(paraIndex: number): boolean {
  return hasMarginalia(paraIndex) || (showAiInsights.value && hasInsights(paraIndex))
}

// ---- Marginalia positioning: align bubbles with anchor text ----
// Map: paraIndex → { margId → topOffsetPx }
const marginaliaOffsets = ref<Record<number, Record<string, number>>>({})
// Map: paraIndex → paragraph text height in px
const paraTextHeights = ref<Record<number, number>>({})

// Estimated height of a collapsed marginalia bubble (px), used for collision avoidance
const EST_MARG_BUBBLE_HEIGHT = 80
const MARG_BUBBLE_GAP = 8

function calculateMarginaliaPositions() {
  if (chapterNumber.value === undefined) return
  const chNum = chapterNumber.value
  const newOffsets: Record<number, Record<string, number>> = {}
  const newHeights: Record<number, number> = {}

  for (const para of paragraphs.value) {
    const key = buildPassageKey(chNum, para.index)
    const margs = annotationStore.marginaliaForKey(key)
      .filter(m => m.source !== 'agent' || showAiInsights.value)
    if (margs.length === 0) continue

    // Target the specific paragraph (may have multiple instances via split view)
    const paraEls = document.querySelectorAll(`[data-passage-key="${key}"]`)
    if (paraEls.length === 0) continue

    const textEl = paraEls[0] as HTMLElement
    const textRect = textEl.getBoundingClientRect()
    if (textRect.height === 0) continue

    newHeights[para.index] = textRect.height
    const rawOffsets: { id: string; top: number }[] = []

    for (const marg of margs) {
      const rangeEl = textEl.querySelector(`.marginalia-range[data-marg-id="${marg.id}"]`) as HTMLElement | null
      if (rangeEl) {
        const rangeRect = rangeEl.getBoundingClientRect()
        // offset from top of paragraph text (clamp to text height)
        const topOffset = Math.max(0, Math.min(rangeRect.top - textRect.top, textRect.height - 20))
        rawOffsets.push({ id: marg.id, top: Math.round(topOffset) })
      }
    }

    if (rawOffsets.length === 0) continue

    // Sort by anchor position (top to bottom in the text)
    rawOffsets.sort((a, b) => a.top - b.top)

    // Collision avoidance: push down bubbles that would overlap the one above
    const resolved: Record<string, number> = {}
    for (let i = 0; i < rawOffsets.length; i++) {
      let top = rawOffsets[i].top
      if (i > 0) {
        const prevTop = rawOffsets[i - 1].top
        const minTop = prevTop + EST_MARG_BUBBLE_HEIGHT + MARG_BUBBLE_GAP
        if (top < minTop) {
          top = minTop
        }
      }
      // Clamp to text height (allow slight overflow rather than hiding)
      rawOffsets[i].top = Math.min(top, textRect.height + 40)
      resolved[rawOffsets[i].id] = rawOffsets[i].top
    }

    newOffsets[para.index] = resolved
  }

  marginaliaOffsets.value = newOffsets
  paraTextHeights.value = newHeights
}

function getMargTop(margId: string, paraIndex: number): number {
  return marginaliaOffsets.value[paraIndex]?.[margId] ?? 0
}

function getParaTextHeight(paraIndex: number): number {
  return paraTextHeights.value[paraIndex] ?? 0
}

// Schedule reposition after DOM updates
let margPositionTimer: ReturnType<typeof setTimeout> | null = null
function scheduleMarginaliaReposition() {
  if (margPositionTimer) clearTimeout(margPositionTimer)
  margPositionTimer = setTimeout(() => {
    void nextTick(() => {
      requestAnimationFrame(() => {
        calculateMarginaliaPositions()
      })
    })
  }, 50)
}

// Reposition on annotation/marginalia changes
watch(
  () => [
    annotationStore.agentAnnotations,
    annotationStore.userAnnotations,
    annotationStore.userMarginalia,
    annotationStore.agentMarginalia,
  ] as const,
  () => { scheduleMarginaliaReposition() },
  { deep: true }
)

// C06+: 段落定位 helper —— 支持异步段落加载重试
// 修复：原 watch 无 immediate，首次打开 tab 时 paragraphIndex 已有值却不触发；
// 且段落 DOM 可能尚未渲染，getElementById 返回 null 后无重试 → 跳转后停在顶部
function scrollToParagraphIndex(idx: number | null | undefined) {
  if (idx === undefined || idx === null) {
    activeParagraphIdx.value = null
    return
  }
  activeParagraphIdx.value = idx
  const tryScroll = (attempt: number) => {
    void nextTick(() => {
      const el = document.getElementById(`para-${idx}`)
      if (el) {
        el.scrollIntoView({ behavior: 'smooth', block: 'center' })
        // C06: 如果有 citationHighlight，进一步滚动到 <mark> 位置
        // B-156: mark 可能不在目标段落内（段落号错位），在章节容器内兜底搜索
        if (citationHighlight.value?.text) {
          setTimeout(() => {
            const contentEl = el.closest('.text-reader-content')
            const mark = (contentEl?.querySelector('mark.citation-highlight') as HTMLElement | null)
              || (el.querySelector('mark.citation-highlight') as HTMLElement | null)
            if (mark) {
              mark.scrollIntoView({ behavior: 'smooth', block: 'center' })
            }
          }, 400)
        }
      } else if (attempt < 12) {
        // 段落可能仍在异步加载，每 80ms 重试一次（共 ~960ms）
        setTimeout(() => tryScroll(attempt + 1), 80)
      }
    })
  }
  tryScroll(0)
}

// Reposition on chapter change or paragraphs update；段落异步加载完成后兜底定位
watch(() => paragraphs.value, () => {
  scheduleMarginaliaReposition()
  // C06+: 章节段落刚加载完成时，若 paragraphIndex 已设值，补一次定位（兜底重试耗尽场景）
  if (paragraphIndex.value !== undefined && paragraphIndex.value !== null) {
    void nextTick(() => {
      const el = document.getElementById(`para-${paragraphIndex.value}`)
      // 元素已存在且大致在视口内则不重复滚动；否则补滚
      if (!el) {
        scrollToParagraphIndex(paragraphIndex.value)
      }
    })
  }
})

// immediate: true —— 首次打开 tab 时 paragraphIndex 已有值也要触发定位
watch(() => paragraphIndex.value, (idx) => {
  scrollToParagraphIndex(idx)
}, { immediate: true })

// Render paragraph text with annotation highlights, marginalia ranges, and search highlights
// Uses sweep-line algorithm: splits text into non-overlapping segments,
// nests tags so that search_highlight (bottom) < marginalia (middle) < annotation (top)
function renderParagraphText(
  text: string,
  annotations: { category: string; label?: string; span: { startChar: number; endChar: number }; id?: string; color?: string; source?: string }[],
  marginalia?: { id: string; anchorCharOffset: number; anchorStartChar?: number; anchorEndChar?: number; content: string; color?: string }[],
  searchHighlights?: SearchHighlight[],
  citationHighlight?: { text: string } | null,
  /** 开发排查：传 paraIndex + chapterNum 方便精确定位哪段没高亮 */
  _dbgMeta?: { paraIndex: number; chapterNumber: number }
): { html: string } {
  interface TextEvent {
    pos: number
    endPos: number
    type: 'search_highlight' | 'marginalia' | 'annotation' | 'citation_highlight'
    color: string
    source?: string
    annotationId?: string
    margId?: string
    margContent?: string
    query?: string
    ephemeral?: boolean
  }

  const typeOrder: Record<string, number> = { search_highlight: 0, marginalia: 1, annotation: 2, citation_highlight: 3 }

  // 暂时性高亮（搜索跳转 ephemeral）直接跟随 activeSearch.ephemeral 标记。
  // "暂时"由组件卸载时清除 activeSearch 保证，不依赖 tab 内临时 key，避免"命中旧 tab / 组件复用"导致 key 不更新而不高亮。
  const isEphemeral = !!activeSearch.value?.ephemeral

  // ---- 高亮链路第三段证据：只要 activeSearch.ephemeral=true，此段有 search 高亮就打印 [TR渲染] 证据 ----
  // （仅 DEV 模式，生产不打印）
  if (import.meta.env.DEV && searchHighlights && searchHighlights.length > 0 && isEphemeral) {
    console.debug(
      `[TR渲染] ch#${_dbgMeta?.chapterNumber} para#${_dbgMeta?.paraIndex}\n` +
      `  ephemeral=Y，要渲染 ${searchHighlights.length} 处 search-highlight-ephemeral 标记:\n` +
      searchHighlights.map((h, i) => `    [${i}] chars ${h.startChar}-${h.endChar} -> "${text.slice(h.startChar, Math.min(h.endChar, h.startChar + 20))}${h.endChar > h.startChar + 20 ? '…' : ''}"`).join('\n')
    )
  }

  const events: TextEvent[] = []
  for (const sh of (searchHighlights || [])) {
    const start = Math.max(0, sh.startChar)
    const end = Math.min(sh.endChar, text.length)
    if (end > start) {
      events.push({
        pos: start,
        endPos: end,
        type: 'search_highlight',
        color: 'rgba(255, 230, 0, 0.35)',
        query: sh.query,
        ephemeral: isEphemeral,
      })
    }
  }
  for (const m of (marginalia || [])) {
    const start = m.anchorStartChar !== undefined && m.anchorStartChar >= 0 ? m.anchorStartChar : m.anchorCharOffset
    const end = m.anchorEndChar !== undefined && m.anchorEndChar > start ? m.anchorEndChar : Math.min(start + 1, text.length)
    if (start >= 0 && end > start && end <= text.length) {
      events.push({ pos: start, endPos: end, type: 'marginalia', color: m.color || '#F5A623', margId: m.id, margContent: m.content })
    }
  }
  for (const ann of annotations) {
    const start = Math.max(0, ann.span.startChar)
    const end = Math.min(ann.span.endChar, text.length)
    if (end > start) {
      events.push({ pos: start, endPos: end, type: 'annotation', color: ann.color || '#F1C40F', source: ann.source, annotationId: ann.id })
    }
  }

  // C06: Citation highlight from evolution narrative
  // 归一化策略：NFC 归一化 + 仅保留字母/数字/CJK，移除所有标点与空白
  // B-156: 增加 NFC 归一化 — 统一 Unicode 组合字符序列，防止"字还是那个字"
  // 但码位不同（如全角/半角变体、组合记号）导致逐字比较失败、高亮中间断
  if (citationHighlight && citationHighlight.text) {
    const isContentChar = (ch: string) => /[\p{L}\p{N}]/u.test(ch)
    // 构建归一化原文 + 归一化字符→原始下标映射
    let normText = ''
    const textOrigPos: number[] = []
    for (let i = 0; i < text.length; i++) {
      const ch = text[i].normalize('NFC')
      if (isContentChar(ch)) {
        normText += ch
        textOrigPos.push(i)
      }
    }
    let normQuote = ''
    for (let i = 0; i < citationHighlight.text.length; i++) {
      const ch = citationHighlight.text[i].normalize('NFC')
      if (isContentChar(ch)) normQuote += ch
    }
    if (normQuote.length >= 4) {
      // 1) 整串精确匹配
      let startN = normText.indexOf(normQuote)
      // 2) 前缀兜底：LLM 引文首尾有改写时，用前 12 字定位起点
      if (startN < 0) {
        const head = normQuote.slice(0, 12)
        if (head.length >= 4) startN = normText.indexOf(head)
      }
      if (startN >= 0 && startN < textOrigPos.length) {
        // 从起点逐字符比较，找连续匹配长度——高亮仅覆盖真正连续匹配的部分，
        // 即便后半段因 LLM 改写而分叉，也只高亮前缀匹配段，不再"断在错位置"
        let matchLenN = 0
        while (matchLenN < normQuote.length
               && startN + matchLenN < normText.length
               && normText[startN + matchLenN] === normQuote[matchLenN]) {
          matchLenN++
        }
        if (matchLenN > 0) {
          const origIdx = textOrigPos[startN]
          const origEnd = textOrigPos[startN + matchLenN - 1] + 1
          if (origEnd > origIdx) {
            events.push({
              pos: origIdx, endPos: origEnd,
              type: 'citation_highlight',
              color: '#f0c040',
            })
          }
        }
      }
    }
  }

  if (!events.length) {
    return { html: escapeHtml(text) }
  }

  // Collect all segment boundaries
  const boundaries = new Set<number>()
  boundaries.add(0)
  boundaries.add(text.length)
  for (const evt of events) {
    boundaries.add(evt.pos)
    boundaries.add(evt.endPos)
  }
  const sortedBounds = [...boundaries].sort((a, b) => a - b)

  // Build segments
  const htmlParts: string[] = []
  for (let i = 0; i < sortedBounds.length - 1; i++) {
    const segStart = sortedBounds[i]
    const segEnd = sortedBounds[i + 1]
    if (segStart >= segEnd) continue

    const segText = escapeHtml(text.slice(segStart, segEnd))

    // Find events covering this segment
    const covering = events.filter(evt => evt.pos <= segStart && evt.endPos >= segEnd)
    if (covering.length === 0) {
      htmlParts.push(segText)
      continue
    }

    // Sort: outer (low typeOrder) first so they wrap inner (high typeOrder)
    covering.sort((a, b) => (typeOrder[a.type] || 0) - (typeOrder[b.type] || 0))

    // Nest tags: outer → inner → text → close inner → close outer
    let result = segText
    for (let j = covering.length - 1; j >= 0; j--) {
      const evt = covering[j]
      if (evt.type === 'search_highlight') {
        const extraClass = evt.ephemeral ? ' search-highlight-ephemeral' : ''
        result = `<mark class="search-highlight${extraClass}" data-query="${escapeHtml(evt.query || '')}">${result}</mark>`
      } else if (evt.type === 'annotation') {
        const borderStyle = evt.source === 'user' ? 'solid' : 'dashed'
        const suffix = evt.source === 'user' ? '' : ''
        const linkedClass = (props.linkedHighlightId && evt.annotationId && evt.annotationId.startsWith(props.linkedHighlightId))
          ? ' linked-blink' : ''
        const suffixHtml = segEnd === evt.endPos ? `<sup class="ann-suffix">${suffix}</sup>` : ''
        result = `<span class="annotation-tag${linkedClass}" style="border-bottom: 2px ${borderStyle} ${evt.color}; background: ${evt.color}20;" data-annotation-id="${evt.annotationId || ''}" data-source="${evt.source || ''}">${result}${suffixHtml}</span>`
      } else if (evt.type === 'citation_highlight') {
        result = `<mark class="citation-highlight">${result}</mark>`
      } else {
        const title = escapeHtml((evt.margContent || '').slice(0, 80))
        result = `<span class="marginalia-range" data-marg-id="${evt.margId || ''}" title="${title}" style="text-decoration: underline wavy ${evt.color}; text-underline-offset: 4px; background: ${evt.color}15; padding: 1px 0; border-radius: 2px;">${result}</span>`
      }
    }
    htmlParts.push(result)
  }

  return { html: htmlParts.join('') }
}

function escapeHtml(text: string): string {
  return text
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
}

// Annotation click handler
const overlappingAnnotation = ref<Record<string, unknown> | null>(null)

function onAnnotationClick(event: MouseEvent, passageKey: string) {
  const target = event.target as HTMLElement
  const annotationId = target.getAttribute('data-annotation-id')
  if (!annotationId) return

  overlappingAnnotation.value = null

  const annotations = annotationStore.annotationsForKey(passageKey)
  const ann = annotations.find(a => a.id === annotationId)
  if (ann) {
    popupAnnotation.value = ann as unknown as Record<string, unknown>
    popupPassageKey.value = passageKey
    popupX.value = event.clientX
    popupY.value = event.clientY + 16
    popupVisible.value = true
    return
  }

  // Also check user annotations
  const userAnns = annotationStore.userAnnotationsForKey(passageKey)
  const userAnn = userAnns.find(a => a.id === annotationId)
  if (userAnn) {
    popupAnnotation.value = {
      ...userAnn,
      explanation: userAnn.note,
      source: 'user',
      color: userAnn.color,
      id: userAnn.id,
      label: userAnn.label || '',
      category: userAnn.category,
    } as unknown as Record<string, unknown>
    popupPassageKey.value = passageKey
    popupX.value = event.clientX
    popupY.value = event.clientY + 16
    popupVisible.value = true

    // Check for overlapping agent annotations
    const pos = userAnn.span
    for (const agentAnn of annotations) {
      if (
        agentAnn.id !== userAnn.id &&
        agentAnn.span.startChar < pos.endChar &&
        agentAnn.span.endChar > pos.startChar
      ) {
        overlappingAnnotation.value = agentAnn as unknown as Record<string, unknown>
        break
      }
    }
  }
}

function handleShowOverlapping(id: string) {
  const passageKey = popupPassageKey.value
  if (!passageKey) return
  const anns = annotationStore.annotationsForKey(passageKey)
  const ann = anns.find(a => a.id === id)
  if (ann) {
    popupAnnotation.value = ann as unknown as Record<string, unknown>
    popupVisible.value = true
  }
}

// Calculate real char offsets in paragraph text
function calculateCharOffsets(): { startChar: number; endChar: number } | null {
  const selection = window.getSelection()
  if (!selection || selection.isCollapsed) return null

  const range = selection.getRangeAt(0)
  const container = range.commonAncestorContainer

  // Find the parent paragraph text element
  const paraEl = container instanceof Element
    ? container.closest('.tr-para-text')
    : (container.parentElement?.closest('.tr-para-text') || null)
  if (!paraEl) return null

  // Get the passage key and original paragraph text
  const passageKey = paraEl.getAttribute('data-passage-key') || ''
  const parts = passageKey.split(':')
  const chNum = parseInt(parts[0], 10)
  const paraIdx = parseInt(parts[1], 10)
  if (isNaN(chNum) || isNaN(paraIdx)) return null

  const para = sourceTreeStore.getParagraph(chNum, paraIdx)
  if (!para) return null

  // Walk text nodes, skipping annotation suffix emojis
  // Each annotation tag renders as: <span class="annotation-tag">ENTITY<sup class="ann-suffix">EMOJI</sup></span>
  // The emoji in sup.ann-suffix is NOT part of the original text and must be excluded from offset counting
  const walker = document.createTreeWalker(paraEl, NodeFilter.SHOW_TEXT, null)
  let offset = 0
  const startContainer = range.startContainer
  const startOffset = range.startOffset
  const endContainer = range.endContainer
  const endOffset = range.endOffset
  let startChar = -1
  let endChar = -1

  while (true) {
    const node = walker.nextNode() as Text | null
    if (!node) break

    const isSuffix = (node.parentElement as HTMLElement | null)?.classList.contains('ann-suffix') ?? false

    if (node === startContainer) {
      startChar = offset + startOffset
    }
    if (node === endContainer) {
      endChar = offset + endOffset
      break
    }
    // Only count text that belongs to the original paragraph (exclude emoji suffixes)
    if (!isSuffix) {
      offset += (node.textContent || '').length
    }
  }

  if (startChar >= 0 && endChar >= 0 && startChar < endChar) {
    return { startChar, endChar }
  }
  return null
}

// Selection handler
function onParagraphContextMenu(event: MouseEvent, passageKey: string) {
  const selection = window.getSelection()
  if (!selection || selection.isCollapsed) return

  const text = selection.toString().trim()
  if (!text) return

  if (props.splitSide) {
    workspaceStore.setActiveSplitSide(props.splitSide)
  }

  event.preventDefault()
  const offsets = calculateCharOffsets()
  selectedText.value = text
  selectionStartChar.value = offsets?.startChar ?? 0
  selectionEndChar.value = offsets?.endChar ?? text.length
  selectionPassageKey.value = passageKey
  selectionMenuX.value = event.clientX
  selectionMenuY.value = event.clientY
  selectionMenuVisible.value = true
}

// Annotation right-click context menu handler
function onAnnotationContextMenu(event: MouseEvent, passageKey: string) {
  const target = event.target as HTMLElement
  const annTag = target.closest('.annotation-tag') as HTMLElement | null
  if (!annTag) return

  event.preventDefault()
  const annId = annTag.getAttribute('data-annotation-id') || ''
  const source = annTag.getAttribute('data-source') as 'user' | 'agent' | null
  if (!annId) return

  annCtxMenuId.value = annId
  annCtxMenuPassageKey.value = passageKey
  annCtxMenuSource.value = source === 'user' ? 'user' : 'agent'
  annCtxMenuX.value = event.clientX
  annCtxMenuY.value = event.clientY
  annCtxMenuVisible.value = true
}

function closeAnnCtxMenu() {
  annCtxMenuVisible.value = false
}

function handleAnnCtxMenuAction(action: string) {
  const passageKey = annCtxMenuPassageKey.value
  const annId = annCtxMenuId.value
  if (!passageKey || !annId) {
    closeAnnCtxMenu()
    return
  }

  switch (action) {
    case 'edit': {
      closePopup()
      const userAnns = annotationStore.userAnnotationsForKey(passageKey)
      const ann = userAnns.find(a => a.id === annId)
      if (ann) {
        popupAnnotation.value = { ...ann, source: 'user', explanation: ann.note, label: ann.label || '', id: ann.id }
      }
      popupPassageKey.value = passageKey
      popupX.value = annCtxMenuX.value
      popupY.value = annCtxMenuY.value
      popupVisible.value = true
      break
    }
    case 'write_marginalia': {
      const parts = passageKey.split(':')
      marginaliaPassageKey.value = passageKey
      marginaliaParagraphIndex.value = parseInt(parts[1], 10) || 0
      marginaliaFormX.value = annCtxMenuX.value
      marginaliaFormY.value = annCtxMenuY.value
      marginaliaFormVisible.value = true
      break
    }
    case 'ask_agent': {
      const tagEl = document.querySelector(`[data-annotation-id="${annId}"]`)
      const text = tagEl?.textContent?.trim() || ''
      agentStore.addContextTag({
        id: `ctx_${Date.now()}`,
        label: `: ${text.slice(0, 30)}${text.length > 30 ? '...' : ''}`,
        data: { annotationId: annId, passageKey, text },
        sourceType: 'annotation',
      })
      break
    }
    case 'copy': {
      const tagEl = document.querySelector(`[data-annotation-id="${annId}"]`)
      const text = tagEl?.textContent?.trim() || ''
      navigator.clipboard.writeText(text)
      break
    }
    case 'delete':
      handlePopupDelete(annId)
      break
    case 'adopt':
      handlePopupAdopt(annId)
      break
    case 'adopt_edit':
      handlePopupAdoptAndEdit(annId)
      break
  }
  closeAnnCtxMenu()
}

function handleSelectionAction(action: string, payload?: Record<string, unknown>) {
  if (action === 'send_agent' && selectedText.value) {
    agentStore.addContextTag({
      id: `ctx_${Date.now()}`,
      label: `"${selectedText.value.slice(0, 30)}${selectedText.value.length > 30 ? '...' : ''}"`,
      data: { text: selectedText.value },
      sourceType: 'text_selection',
    })
  } else if (action === 'annotate_category' && payload?.category) {
    preselectedCategory.value = payload.category as AnnotationCategory
    userAnnotationFormVisible.value = true
  } else if (action === 'annotate') {
    preselectedCategory.value = null
    userAnnotationFormVisible.value = true
  } else if (action === 'write_marginalia') {
    const passageKey = selectionPassageKey.value || currentPassageKey.value
    if (!passageKey) return
    marginaliaPassageKey.value = passageKey
    const parts = passageKey.split(':')
    marginaliaParagraphIndex.value = parseInt(parts[1], 10) || 0
    marginaliaFormX.value = selectionMenuX.value
    marginaliaFormY.value = selectionMenuY.value
    marginaliaFormVisible.value = true
  } else if (action === 'custom_color') {
    preselectedCategory.value = null
    userAnnotationFormVisible.value = true
  } else if (action === 'copy') {
    navigator.clipboard.writeText(selectedText.value)
  }
  selectionMenuVisible.value = false
}

function handleUserAnnotation(note: string, category: AnnotationCategory, color: string, span: { startChar: number; endChar: number }) {
  const passageKey = selectionPassageKey.value || currentPassageKey.value
  if (!passageKey) return
  annotationStore.addUserAnnotation(passageKey, {
    category,
    span,
    note,
    color,
  })
  userAnnotationFormVisible.value = false
  selectedText.value = ''
}

function handleMarginaliaConfirm(content: string) {
  if (!marginaliaPassageKey.value) return
  if (annotationStore.editingMarginaliaId) {
    // Edit existing marginalia
    annotationStore.updateMarginalia(
      marginaliaPassageKey.value,
      annotationStore.editingMarginaliaId,
      { content }
    )
    annotationStore.editingMarginaliaId = null
  } else {
    // Create new marginalia
    annotationStore.addMarginalia(marginaliaPassageKey.value, {
      chapterNumber: chapterNumber.value || 0,
      paragraphIndex: marginaliaParagraphIndex.value,
      anchorCharOffset: selectionEndChar.value,
      anchorStartChar: selectionStartChar.value > 0 ? selectionStartChar.value : undefined,
      anchorEndChar: selectionEndChar.value > 0 ? selectionEndChar.value : undefined,
      content,
    })
  }
  marginaliaFormVisible.value = false
}

// ===== CU-08: Search highlight keep/dismiss actions =====

function removeHighlightsFromActiveSearch(highlights: SearchHighlight[]) {
  if (!sourceTreeStore.activeSearch) return
  const removeSet = new Set(highlights.map(h => `${h.chapterNumber}:${h.paragraphIndex}:${h.startChar}:${h.endChar}`))
  const remaining = sourceTreeStore.activeSearch.highlights.filter(
    h => !removeSet.has(`${h.chapterNumber}:${h.paragraphIndex}:${h.startChar}:${h.endChar}`)
  )
  if (remaining.length === 0) {
    sourceTreeStore.activeSearch = null
  } else {
    sourceTreeStore.activeSearch = { ...sourceTreeStore.activeSearch, highlights: remaining }
  }
}

function keepAllHighlightsInPara(paraIndex: number) {
  if (chapterNumber.value == null) return
  const passageKey = buildPassageKey(chapterNumber.value, paraIndex)
  const paraHighlights = getParaSearchHighlights(paraIndex)

  for (const h of paraHighlights) {
    annotationStore.addUserAnnotation(passageKey, {
      category: 'custom',
      span: { startChar: h.startChar, endChar: h.endChar },
      note: ` "${h.query}" `,
      color: '#F5A623',
    })
  }

  removeHighlightsFromActiveSearch(paraHighlights)
}

function dismissAllHighlightsInPara(paraIndex: number) {
  const paraHighlights = getParaSearchHighlights(paraIndex)
  removeHighlightsFromActiveSearch(paraHighlights)
}

// Paragraph click for marginalia
function onParagraphClick(event: MouseEvent, passageKey: string) {
  if (props.splitSide) {
    workspaceStore.setActiveSplitSide(props.splitSide)
  }

  const target = event.target as HTMLElement
  // Only open marginalia form if clicking on whitespace (not on an annotation tag)
  if (target.classList.contains('annotation-tag')) return
  if (target.closest('.annotation-tag')) return

  const rect = target.getBoundingClientRect()
  const clickX = event.clientX - rect.left
  const elWidth = rect.width

  // If clicked near the right edge of the paragraph, open marginalia
  if (clickX > elWidth * 0.7) {
    marginaliaPassageKey.value = passageKey
    const parts = passageKey.split(':')
    marginaliaParagraphIndex.value = parseInt(parts[1], 10) || 0
    marginaliaFormX.value = event.clientX
    marginaliaFormY.value = event.clientY
    marginaliaFormVisible.value = true
  }
}

function closePopup() {
  popupVisible.value = false
  popupAnnotation.value = null
  popupPassageKey.value = ''
  overlappingAnnotation.value = null
}

// Popup actions
function handlePopupEdit(id: string) {
  // Handled inside AnnotationPopup
}

function handlePopupDelete(id: string) {
  if (!popupPassageKey.value) {
    return
  }
  const source = popupAnnotation.value?.source as string
  if (source === 'user') {
    annotationStore.removeUserAnnotation(popupPassageKey.value, id)
  } else {
    annotationStore.removeAgentAnnotation(popupPassageKey.value, id)
  }
  closePopup()
}

function handlePopupAdopt(id: string) {
  if (!popupPassageKey.value) return
  annotationStore.adoptAgentAnnotation(popupPassageKey.value, id)
  closePopup()
}

function handlePopupAdoptAndEdit(id: string) {
  if (!popupPassageKey.value) return
  annotationStore.adoptAgentAnnotation(popupPassageKey.value, id)
  // Show the user annotation form to edit
  preselectedCategory.value = null
  userAnnotationFormVisible.value = true
  closePopup()
}

// Suggestion actions
function acceptSuggestion(passageKey: string, suggestionId: string) {
  annotationStore.acceptSuggestion(passageKey, suggestionId)
}

function dismissSuggestion(passageKey: string, suggestionId: string) {
  annotationStore.dismissSuggestion(passageKey, suggestionId)
}

// Visibility toggle (old API for legend clicking — now uses new layered API)
function toggleLegacyVisibility(layer: 'user' | 'agent', category: string) {
  annotationStore.toggleVisibility(layer, category as keyof AnnotationVisibility['user'])
}

function isCategoryVisible(layer: 'user' | 'agent', category: string): boolean {
  return annotationStore.visibility[layer][category as keyof AnnotationVisibility['user']] ?? true
}

// Export
function handleExport(config: { format: string; scope: string; includeUser: boolean; includeAgent: boolean; includeMarginalia: boolean; teiMode?: string; selectedChapters?: number[] }) {
  const chapters = config.selectedChapters && config.selectedChapters.length > 0
    ? config.selectedChapters
    : [chapterNumber.value || 0].filter(Boolean)
  if (chapters.length === 0) return

  const paraTexts = paragraphs.value.map(p => ({ index: p.index, text: p.text }))
  const chTitle = chapterTitle.value || 'export'

  if (config.format === 'html') {
    let html = ''
    for (const chNum of chapters) {
      html += annotationStore.exportAnnotationsHTML(chNum, config.includeAgent, config.includeUser, config.includeMarginalia, paraTexts)
    }
    downloadFile(html, `${chTitle}.html`, 'text/html;charset=utf-8')
  } else if (config.format === 'pdf') {
    let html = ''
    for (const chNum of chapters) {
      html += annotationStore.exportAnnotationsHTML(chNum, config.includeAgent, config.includeUser, config.includeMarginalia, paraTexts)
    }
    openPrintWindow(html)
  } else if (config.format === 'plaintext') {
    let text = ''
    for (const chNum of chapters) {
      text += annotationStore.exportAnnotationsPlainText(chNum, config.includeAgent, config.includeUser, config.includeMarginalia)
    }
    downloadFile(text, `${chTitle}.txt`, 'text/plain;charset=utf-8')
  } else if (config.format === 'csv') {
    const csv = annotationStore.exportAnnotationsCSV(chapters, config.includeAgent, config.includeUser, config.includeMarginalia)
    downloadFile(csv, `${chTitle}.csv`, 'text/csv;charset=utf-8')
  } else if (config.format === 'json') {
    const json = annotationStore.exportAnnotationsJSON(chapters, config.includeAgent, config.includeUser, config.includeMarginalia, false, undefined)
    downloadFile(json, `${chTitle}.json`, 'application/json;charset=utf-8')
  } else if (config.format === 'tei') {
    const teiMode = (config.teiMode || 'standOff') as 'standOff' | 'inline'
    const tei = annotationStore.exportAnnotationsTEI(chapters, config.includeAgent, config.includeUser, config.includeMarginalia, teiMode, undefined)
    downloadFile(tei, `${chTitle}.xml`, 'application/xml;charset=utf-8')
  }
  exportPanelVisible.value = false
}

function downloadFile(content: string, filename: string, mimeType: string) {
  const blob = new Blob([content], { type: mimeType })
  const url = URL.createObjectURL(blob)
  const a = document.createElement('a')
  a.href = url
  a.download = filename
  a.click()
  URL.revokeObjectURL(url)
}

function openPrintWindow(html: string) {
  const printWindow = window.open('', '_blank')
  if (!printWindow) return
  printWindow.document.write(html)
  printWindow.document.close()
  printWindow.onload = () => {
    printWindow.print()
    printWindow.close()
  }
  setTimeout(() => {
    if (!printWindow.closed) {
      printWindow.print()
    }
  }, 500)
}

// Keyboard shortcuts
function onKeydown(e: KeyboardEvent) {
  if (e.ctrlKey && e.shiftKey && e.key === 'A') {
    e.preventDefault()
    showAiAnnotations.value = !showAiAnnotations.value
  }
  if (e.ctrlKey && e.shiftKey && e.key === 'N') {
    e.preventDefault()
    showAiInsights.value = !showAiInsights.value
  }
  if (e.key === 'Escape') {
    closePopup()
    selectionMenuVisible.value = false
    userAnnotationFormVisible.value = false
    marginaliaFormVisible.value = false
    layerPanelVisible.value = false
    exportPanelVisible.value = false
    closeAnnCtxMenu()
  }
}

// D03: Watch navigationTarget for same-chapter paragraph jumps
watch(
  () => sourceTreeStore.navigationTarget,
  (target) => {
    if (!target) return
    if (target.chapterNumber !== chapterNumber.value) return
    void nextTick(() => {
      const el = document.getElementById(`para-${target.paragraphIndex}`)
      if (!el) return
      // Clear old highlight
      const container = el.closest('.text-reader-content')
      if (container) {
        container.querySelectorAll('.tr-paragraph.para-active').forEach(oldEl => {
          oldEl.classList.remove('para-active')
        })
      }
      // Set new highlight
      el.classList.add('para-active')
      activeParagraphIdx.value = target.paragraphIndex
      el.scrollIntoView({ behavior: 'smooth', block: 'center' })
      // C06: 如果有 citationHighlight，进一步滚动到 <mark> 位置
      // B-156: paragraph_index 可能与前端段落号错位（chunker 清洗文本 vs 原文 raw 分段
      // 过滤空段差异），mark 可能在"非目标段落"中——在整个章节内容里搜索 mark
      if (citationHighlight.value?.text) {
        setTimeout(() => {
          const contentEl = el.closest('.text-reader-content')
          const mark = (contentEl?.querySelector('mark.citation-highlight') as HTMLElement | null)
            || (el.querySelector('mark.citation-highlight') as HTMLElement | null)
          if (mark) {
            mark.scrollIntoView({ behavior: 'smooth', block: 'center' })
          }
        }, 400)
      }
    })
  },
  { deep: true }
)

// Watch chapterNumber changes to clear activeParagraphIdx
watch(() => chapterNumber.value, () => {
  activeParagraphIdx.value = null
})

// C-15: Watch editingMarginaliaId to open MarginaliaForm for editing
watch(() => annotationStore.editingMarginaliaId, (margId) => {
  if (!margId) return
  const stores = [annotationStore.userMarginalia, annotationStore.agentMarginalia]
  for (const store of stores) {
    for (const [key, margs] of Object.entries(store)) {
      const found = (margs as Marginalia[]).find(m => m.id === margId)
      if (found) {
        const parts = key.split(':')
        marginaliaPassageKey.value = key
        marginaliaParagraphIndex.value = parseInt(parts[1], 10) || 0
        marginaliaFormX.value = 200
        marginaliaFormY.value = 200
        marginaliaFormVisible.value = true
        return
      }
    }
  }
})

// ====== 旁注面板 Drawer：章节级统一展示（替代原先段落旁悬浮弹出卡片）======
const notesDrawerVisible = ref(false)
// 当前 Drawer 中聚焦的旁注 / 洞察 ID（用于点击锚点后滚动+高亮对应条目）
const notesFocusId = ref<string | null>(null)

function openNotesDrawer(focusId?: string) {
  notesFocusId.value = focusId || null
  notesDrawerVisible.value = true
  // 打开后下一帧滚动到焦点条目
  if (focusId) {
    void nextTick(() => {
      const el = document.querySelector(`[data-note-id="${CSS.escape(focusId)}"]`) as HTMLElement | null
      if (el) {
        el.scrollIntoView({ behavior: 'smooth', block: 'center' })
      }
    })
  }
}
function closeNotesDrawer() {
  notesDrawerVisible.value = false
  notesFocusId.value = null
}
// 点击段落旁注锚点（波浪下划线文字）时打开 Drawer 并定位该条旁注
function jumpToMarginaliaInDrawer(margId: string, evt: MouseEvent) {
  evt.stopPropagation()
  openNotesDrawer(margId)
}

// 汇总当前章节所有旁注+洞察，按段落索引分组（Drawer 内容来源）
const chapterNotesByPara = computed(() => {
  if (chapterNumber.value == null) return []
  const chNum = chapterNumber.value
  const items: {
    paraIndex: number
    paraExcerpt: string
    marginalia: ReturnType<typeof annotationStore.marginaliaForKey>
    insights: ReturnType<typeof annotationStore.insightsForKey>
  }[] = []
  for (const para of paragraphs.value) {
    const key = buildPassageKey(chNum, para.index)
    const margs = annotationStore.marginaliaForKey(key).filter(m => m.source !== 'agent' || showAiInsights.value)
    const insights = showAiInsights.value ? annotationStore.insightsForKey(key) : []
    if (margs.length === 0 && insights.length === 0) continue
    items.push({
      paraIndex: para.index,
      paraExcerpt: para.text.slice(0, 40) + (para.text.length > 40 ? '…' : ''),
      marginalia: margs,
      insights,
    })
  }
  return items
})
// 统计汇总（供 Drawer 头部展示）
const chapterNoteCounts = computed(() => {
  let m = 0, i = 0
  for (const g of chapterNotesByPara.value) {
    m += g.marginalia.length
    i += g.insights.length
  }
  return { marginalia: m, insights: i }
})

// 从 Drawer 里点击段落头 → 滚动到正文对应段落（使用全局 document，避开 Vue 模板内 TS 检查报错）
function scrollToPara(paraIndex: number) {
  const paraEl = globalThis.document?.getElementById('para-' + paraIndex) as HTMLElement | null
  if (paraEl) paraEl.scrollIntoView({ behavior: 'smooth', block: 'center' })
}

onMounted(() => {
  window.addEventListener('keydown', onKeydown)
  scheduleMarginaliaReposition()
})
onUnmounted(() => {
  window.removeEventListener('keydown', onKeydown)
  if (annotationTimeout.value) clearTimeout(annotationTimeout.value)
  if (margPositionTimer) clearTimeout(margPositionTimer)
  // 搜索结果跳转的暂时性高亮：离开阅读页即清除，重进同章节不再显示
  if (sourceTreeStore.activeSearch?.ephemeral) {
    sourceTreeStore.activeSearch = null
  }
  // D04: Clear all para-active on unmount
  const containers = document.querySelectorAll('.text-reader-content')
  containers.forEach(container => {
    container.querySelectorAll('.tr-paragraph.para-active').forEach(el => {
      el.classList.remove('para-active')
    })
  })
})
</script>

<template>
  <div class="text-reader-view" :class="{ 'tr-hide-toolbar': hideToolbar }">
    <!-- Toolbar (紧凑单行 + 折叠区) -->
    <div class="tr-toolbar" :class="{ 'tr-toolbar-collapsed': !toolbarExpanded }">
      <div class="tr-toolbar-main">
        <div class="tr-toolbar-left">
          <h2 v-if="chapterTitle" class="tr-title">{{ chapterTitle }}</h2>
          <div class="tr-badges">
            <span class="tr-badge tr-badge-user">
              我的标注 <span class="tr-badge-num">{{ userAnnCount }}</span>
            </span>
            <span class="tr-badge tr-badge-agent">
              AI标注 <span class="tr-badge-num">{{ agentAnnCount }}</span>
            </span>
          </div>
        </div>
        <div class="tr-toolbar-right">
          <button class="tr-quick-btn" @click="showAiAnnotations = !showAiAnnotations"
            :class="{ active: showAiAnnotations }" title="显示/隐藏 AI 标注">AI标</button>
          <button class="tr-quick-btn" @click="showAiInsights = !showAiInsights"
            :class="{ active: showAiInsights }" title="显示/隐藏 AI 洞察">AI洞</button>
          <button class="tr-quick-btn" @click="annotationStore.saveToServer(chapterNumber ?? 0)" title="保存标注到服务端">保存</button>
          <button
            class="tr-expand-btn"
            @click="toolbarExpanded = !toolbarExpanded"
            :title="toolbarExpanded ? '收起工具栏' : '展开工具栏（图例、图层、导出等）'"
          >{{ toolbarExpanded ? '×' : '≡' }}</button>
        </div>
      </div>

      <!-- 折叠区：图例 + 更多控制按钮 -->
      <transition name="tr-toolbar-fold">
        <div v-show="toolbarExpanded" class="tr-toolbar-folded">
          <div class="tr-legend">
            <span
              class="tr-legend-item"
              :class="{ off: !isCategoryVisible('user', 'person') }"
              @click="toggleLegacyVisibility('user', 'person')"
            >
              <span class="tr-dot" style="background: var(--color-annotation-person)" />人物
            </span>
            <span
              class="tr-legend-item"
              :class="{ off: !isCategoryVisible('user', 'location') }"
              @click="toggleLegacyVisibility('user', 'location')"
            >
              <span class="tr-dot" style="background: var(--color-annotation-location)" />地点
            </span>
            <span
              class="tr-legend-item"
              :class="{ off: !isCategoryVisible('user', 'event') }"
              @click="toggleLegacyVisibility('user', 'event')"
            >
              <span class="tr-dot" style="background: var(--color-annotation-event)" />事件
            </span>
            <span
              class="tr-legend-item"
              :class="{ off: !isCategoryVisible('user', 'term') }"
              @click="toggleLegacyVisibility('user', 'term')"
            >
              <span class="tr-dot" style="background: var(--color-annotation-term)" />术语
            </span>
            <span
              class="tr-legend-item"
              :class="{ off: !isCategoryVisible('user', 'motif') }"
              @click="toggleLegacyVisibility('user', 'motif')"
            >
              <span class="tr-dot" style="background: var(--color-annotation-motif)" />母题
            </span>
            <span
              class="tr-legend-item"
              :class="{ off: !isCategoryVisible('user', 'custom') }"
              @click="toggleLegacyVisibility('user', 'custom')"
            >
              <span class="tr-dot" style="background: var(--color-annotation-custom)" />自定义
            </span>
          </div>

          <div class="tr-controls">
            <button
              class="tr-control-btn"
              :class="{ active: showAiAnnotations }"
              @click="showAiAnnotations = !showAiAnnotations"
            >AI 标注</button>
            <button
              class="tr-control-btn"
              :class="{ active: showAiInsights }"
              @click="showAiInsights = !showAiInsights"
            >AI 洞察</button>
            <button class="tr-control-btn" @click="layerPanelVisible = !layerPanelVisible">
              图层
            </button>
            <button class="tr-control-btn" @click="annotationStore.saveToServer(chapterNumber ?? 0)">
              保存
            </button>
            <button
              class="tr-control-btn"
              @click="annotationStore.clearAgentAnnotationsForChapter(chapterNumber ?? 0)"
            >清除Agent标注</button>
            <button class="tr-control-btn" @click="exportPanelVisible = true">
              导出...
            </button>
          </div>
        </div>
      </transition>
    </div>

    <!-- Layer panel overlay -->
    <div v-if="layerPanelVisible" class="tr-overlay-panel">
      <LayerPanel
        :chapter-number="chapterNumber || undefined"
        :agent-annotation-count="agentAnnCount"
      />
      <button class="tr-close-overlay" @click="layerPanelVisible = false">&times;</button>
    </div>

    <!-- Content -->
    <div class="text-reader-content">
      <div v-if="paragraphs.length === 0" class="tr-empty">
        <p>请在左侧文献目录中选择章节</p>
      </div>

      <div
        v-for="para in paragraphs"
        :id="`para-${para.index}`"
        :key="para.index"
        class="tr-paragraph"
        :class="{
          'para-active': activeParagraphIdx === para.index,
          'tr-has-float-marg': hasMarginaliaOrInsights(para.index) && marginaliaExpandedInner,
        }"
      >
        <span class="tr-para-num">{{ para.index }}</span>
        <div class="tr-para-body">
          <div
            class="tr-para-text"
            :data-passage-key="buildPassageKey(chapterNumber!, para.index)"
            v-html="renderParagraphText(
              para.text,
              annotationStore.visibleAnnotations(buildPassageKey(chapterNumber!, para.index)),
              annotationStore.marginaliaForKey(buildPassageKey(chapterNumber!, para.index)).filter(m => m.source !== 'agent' || showAiInsights),
              getParaSearchHighlights(para.index),
              citationHighlight,
              { paraIndex: para.index, chapterNumber: chapterNumber ?? -1 }
            ).html"
            @click="onAnnotationClick($event, buildPassageKey(chapterNumber!, para.index)); onParagraphClick($event, buildPassageKey(chapterNumber!, para.index))"
            @contextmenu="(e: MouseEvent) => {
              const target = e.target as HTMLElement
              if (target.closest('.annotation-tag')) {
                onAnnotationContextMenu(e, buildPassageKey(chapterNumber!, para.index))
              } else {
                onParagraphContextMenu(e, buildPassageKey(chapterNumber!, para.index))
              }
            }"
          />

          <!-- Suggested annotations -->
          <div
            v-if="annotationStore.suggestedAnnotations[buildPassageKey(chapterNumber!, para.index)]?.length"
            class="tr-suggestions"
          >
            <div
              v-for="sug in annotationStore.suggestedAnnotations[buildPassageKey(chapterNumber!, para.index)]"
              :key="sug.id"
              class="tr-suggestion-item"
            >
              <span class="tr-suggestion-label">[{{ sug.category }}] {{ sug.text }}</span>
              <span class="tr-suggestion-reason">{{ sug.reason }}</span>
              <div class="tr-suggestion-actions">
                <button
                  class="tr-sug-btn tr-sug-accept"
                  @click="acceptSuggestion(buildPassageKey(chapterNumber!, para.index), sug.id)"
                >采纳</button>
                <button
                  class="tr-sug-btn tr-sug-dismiss"
                  @click="dismissSuggestion(buildPassageKey(chapterNumber!, para.index), sug.id)"
                >忽略</button>
              </div>
            </div>
          </div>

          <!-- Loading indicator -->
          <div v-if="isAnnotationLoading && activeParagraphIdx === para.index" class="tr-loading-annotation">
            ...
          </div>

          <!-- Search highlight action bar (CU-08) -->
          <div
            v-if="getParaSearchHighlights(para.index).length > 0"
            class="tr-search-actions"
          >
            <span class="tr-search-actions-label">
              搜索 "{{ activeSearch?.query }}" ({{ getParaSearchHighlights(para.index).length }} 处匹配)
              <span v-if="activeSearch?.source === 'agent'" class="tr-search-source-tag">Agent</span>
              <span v-else class="tr-search-source-tag">用户</span>
            </span>
            <div class="tr-search-actions-btns">
              <button class="tr-search-btn tr-search-keep" @click="keepAllHighlightsInPara(para.index)">
                保留
              </button>
              <button class="tr-search-btn tr-search-dismiss" @click="dismissAllHighlightsInPara(para.index)">
                清除
              </button>
            </div>
          </div>
        </div>

      <!-- Marginalia + AI Insights 锚点标记（第 3 版 Drawer 方案）
           - 段号右侧一个极小的彩色"▌"标记，仅旁注/洞察存在时显示
           - 点击标记：打开章节级右侧批注抽屉（Drawer），按段落分组展示全部内容
           - 正文中的波浪下划线（marginalia-range）也可点击，直接定位到 Drawer 中对应条目 -->
      <template v-if="hasMarginaliaOrInsights(para.index)">
        <button
          class="tr-note-anchor"
          :class="{ 'has-insight': hasInsights(para.index) }"
          :title="`查看本段旁注 (${annotationStore.marginaliaForKey(buildPassageKey(chapterNumber!, para.index)).length}) / AI洞察 (${showAiInsights ? annotationStore.insightsForKey(buildPassageKey(chapterNumber!, para.index)).length : 0})`"
          @click.stop="openNotesDrawer()"
        ><span class="tr-note-anchor-bar"></span></button>
      </template>
      </div><!-- /tr-paragraph -->
    </div><!-- /text-reader-content -->

    <!-- ============ 章节批注 Drawer（章节级统一侧栏） ============ -->
    <transition name="tr-drawer-mask">
      <div v-if="notesDrawerVisible" class="tr-drawer-mask" @click="closeNotesDrawer"></div>
    </transition>
    <transition name="tr-drawer">
      <aside v-if="notesDrawerVisible" class="tr-notes-drawer" role="dialog" aria-label="旁注与AI洞察">
        <header class="tr-drawer-head">
          <div class="tr-drawer-head-title">
            <span class="tr-drawer-title-icon" aria-hidden="true"></span>
            <span class="tr-drawer-title-text">章节批注</span>
          </div>
          <div class="tr-drawer-counts">
            <span class="tr-drawer-chip">
              <span class="tr-drawer-chip-dot" style="background:var(--color-accent)"></span>
              旁注 {{ chapterNoteCounts.marginalia }}
            </span>
            <span class="tr-drawer-chip" v-if="showAiInsights">
              <span class="tr-drawer-chip-dot" style="background:#7EC4A0"></span>
              洞察 {{ chapterNoteCounts.insights }}
            </span>
          </div>
          <button class="tr-drawer-close" @click="closeNotesDrawer" aria-label="关闭">×</button>
        </header>

        <div v-if="chapterNotesByPara.length === 0" class="tr-drawer-empty">
          当前章节还没有旁注或 AI 洞察。<br>
          选中文本后右键可添加旁注。
        </div>

        <div v-else class="tr-drawer-body">
          <section
            v-for="group in chapterNotesByPara"
            :key="group.paraIndex"
            class="tr-drawer-group"
          >
            <header
              class="tr-drawer-group-head"
              @click="() => scrollToPara(group.paraIndex)"
              title="跳转到对应段落"
            >
              <span class="tr-drawer-group-idx">¶ {{ group.paraIndex }}</span>
              <span class="tr-drawer-group-excerpt">{{ group.paraExcerpt }}</span>
              <span class="tr-drawer-group-arrow">→</span>
            </header>

            <!-- 旁注条目 -->
            <article
              v-for="marg in group.marginalia"
              :key="marg.id"
              :data-note-id="marg.id"
              class="tr-drawer-item"
              :class="{
                focused: notesFocusId === marg.id,
                'is-ai': marg.source === 'agent',
              }"
              :style="marg.color ? { borderLeftColor: marg.color } : {}"
            >
              <div class="tr-drawer-item-head">
                <span class="tr-drawer-item-tag" :class="marg.source === 'agent' ? 'tag-ai' : 'tag-user'">
                  {{ marg.source === 'agent' ? 'AI 旁注' : '我的旁注' }}
                </span>
                <span class="tr-drawer-item-time">
                  {{ new Date(marg.updatedAt).toLocaleString('zh-CN', { month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' }) }}
                </span>
                <div
                  v-if="!marg.source || marg.source === 'user'"
                  class="tr-drawer-item-actions"
                >
                  <button
                    class="tr-drawer-btn"
                    @click.stop="annotationStore.editingMarginaliaId = marg.id"
                  >编辑</button>
                  <button
                    class="tr-drawer-btn tr-drawer-btn-del"
                    @click.stop="annotationStore.removeMarginalia(buildPassageKey(chapterNumber!, group.paraIndex), marg.id)"
                  >删除</button>
                </div>
              </div>
              <div class="tr-drawer-item-body">{{ marg.content }}</div>
            </article>

            <!-- AI 洞察条目 -->
            <article
              v-for="insight in group.insights"
              :key="insight.id"
              :data-note-id="insight.id"
              class="tr-drawer-item tr-drawer-item-insight"
              :class="{ focused: notesFocusId === insight.id }"
            >
              <div class="tr-drawer-item-head">
                <span class="tr-drawer-item-tag tag-insight">AI 洞察 · {{ insight.label }}</span>
              </div>
              <div class="tr-drawer-item-body">
                {{ insight.text }}
                <span v-if="insight.motifRef" class="tr-drawer-item-motif">· 关联母题：{{ insight.motifRef }}</span>
              </div>
            </article>
          </section>
        </div>
      </aside>
    </transition>

    <!-- Sub-components -->
    <AnnotationPopup
      :annotation="popupAnnotation"
      :passage-key="popupPassageKey"
      :x="popupX"
      :y="popupY"
      :overlapping-annotation="overlappingAnnotation"
      @close="closePopup()"
      @edit="handlePopupEdit"
      @delete="handlePopupDelete"
      @adopt="handlePopupAdopt"
      @adopt-and-edit="handlePopupAdoptAndEdit"
      @show-overlapping="handleShowOverlapping"
    />

    <SelectionMenu
      :visible="selectionMenuVisible"
      :x="selectionMenuX"
      :y="selectionMenuY"
      @action="handleSelectionAction"
      @close="selectionMenuVisible = false"
    />

    <UserAnnotationForm
      :visible="userAnnotationFormVisible"
      :selected-text="selectedText"
      :start-char="selectionStartChar"
      :end-char="selectionEndChar"
      :preselected-category="preselectedCategory"
      @confirm="handleUserAnnotation"
      @cancel="userAnnotationFormVisible = false"
    />

    <MarginaliaForm
      :visible="marginaliaFormVisible"
      :anchor-x="marginaliaFormX"
      :anchor-y="marginaliaFormY"
      :initial-content="annotationStore.editingMarginaliaId
        ? (Object.values(annotationStore.userMarginalia).flat().find(m => m.id === annotationStore.editingMarginaliaId)?.content || '')
        : ''"
      @confirm="handleMarginaliaConfirm"
      @cancel="marginaliaFormVisible = false; annotationStore.editingMarginaliaId = null"
    />

    <ExportPanel
      :visible="exportPanelVisible"
      :chapter-number="chapterNumber || 0"
      :all-chapters="sourceTreeStore.chapters.map(c => ({
        number: c.number,
        title: c.title,
        dynasty: c.dynasty,
        hasAnnotations: annotationStore.hasUserAnnotations(c.number) || annotationStore.agentAnnotationCount(c.number) > 0,
      }))"
      :user-annotation-count="userAnnCount"
      :agent-annotation-count="agentAnnCount"
      @export="handleExport"
      @cancel="exportPanelVisible = false"
    />

    <!-- Annotation right-click context menu -->
    <Teleport to="body">
      <div
        v-if="annCtxMenuVisible"
        class="ann-ctx-backdrop"
        @click="closeAnnCtxMenu"
      />
      <div
        v-if="annCtxMenuVisible"
        class="ann-ctx-menu"
        :style="{ left: annCtxMenuX + 'px', top: annCtxMenuY + 'px' }"
      >
        <template v-if="annCtxMenuSource === 'user'">
          <button class="ann-ctx-item" @click="handleAnnCtxMenuAction('edit')">
            编辑类别/颜色
          </button>
          <button class="ann-ctx-item" @click="handleAnnCtxMenuAction('write_marginalia')">
            写/编辑批注
          </button>
          <button class="ann-ctx-item" @click="handleAnnCtxMenuAction('ask_agent')">
            就此追问 Agent
          </button>
          <button class="ann-ctx-item" @click="handleAnnCtxMenuAction('copy')">
            复制标注文字
          </button>
          <div class="ann-ctx-divider" />
          <button class="ann-ctx-item ann-ctx-danger" @click="handleAnnCtxMenuAction('delete')">
            删除标注
          </button>
        </template>
        <template v-else>
          <button class="ann-ctx-item ann-ctx-primary" @click="handleAnnCtxMenuAction('adopt')">
            采纳为我的标注
          </button>
          <button class="ann-ctx-item" @click="handleAnnCtxMenuAction('adopt_edit')">
            修改后采纳
          </button>
          <button class="ann-ctx-item" @click="handleAnnCtxMenuAction('ask_agent')">
            就此追问 Agent
          </button>
          <button class="ann-ctx-item" @click="handleAnnCtxMenuAction('copy')">
            复制标注文字
          </button>
          <div class="ann-ctx-divider" />
          <button class="ann-ctx-item ann-ctx-danger" @click="handleAnnCtxMenuAction('delete')">
            删除此 Agent 标注
          </button>
        </template>
      </div>
    </Teleport>
  </div>
</template>

<style scoped>
.text-reader-view {
  display: flex;
  flex-direction: column;
  height: 100%;
  overflow: hidden;
}
.text-reader-view.tr-hide-toolbar .tr-toolbar { display: none; }

/* A2 紧凑单行工具栏 */
.tr-toolbar {
  padding: 8px 20px;
  border-bottom: 1px solid var(--color-border);
  background: var(--color-bg-secondary);
  transition: box-shadow var(--dur-fast) var(--ease);
}
.tr-toolbar.tr-toolbar-collapsed {
  box-shadow: 0 1px 0 var(--color-border);
}
.tr-toolbar-main {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: var(--space-3);
}
.tr-toolbar-left {
  display: flex;
  align-items: center;
  gap: var(--space-3);
  min-width: 0;
}
.tr-toolbar-right {
  display: flex;
  align-items: center;
  gap: var(--space-1);
}
.tr-title {
  font-family: var(--font-sans);
  font-size: 0.92rem;
  font-weight: 600;
  color: var(--color-text-primary);
  margin: 0;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
  max-width: 320px;
}
.tr-badges {
  display: flex;
  gap: 6px;
  flex-shrink: 0;
}
.tr-badge {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  gap: 4px;
  font-size: 0.66rem;
  /* 统一宽度：不管数字是 0 / 12 / 999 都不跳 */
  min-width: 78px;
  height: 22px;
  padding: 0 8px;
  border-radius: var(--radius-full);
  border: 1px solid transparent;
  color: var(--color-text-secondary);
  background: var(--color-bg-tertiary);
  line-height: 1;
  font-weight: 500;
  white-space: nowrap;
}
.tr-badge .tr-badge-num {
  /* 数字等宽：1/2/3 位数水平占位一致 */
  font-variant-numeric: tabular-nums;
  font-family: var(--font-mono);
  font-weight: 600;
  min-width: 1.2em;
  text-align: center;
}
.tr-badge-user {
  background: var(--color-accent-light);
  color: var(--color-accent);
  border-color: color-mix(in srgb, var(--color-accent) 22%, transparent);
}
.tr-badge-agent {
  background: var(--color-bg-tertiary);
  color: var(--color-text-secondary);
  border-color: var(--color-border);
}
.tr-quick-btn,
.tr-expand-btn {
  font-family: var(--font-sans);
  padding: 2px 10px;
  font-size: 0.68rem;
  height: 24px;
  line-height: 1;
  background: var(--color-bg-primary);
  color: var(--color-text-secondary);
  border: 1px solid var(--color-border);
  border-radius: var(--radius-sm);
  cursor: pointer;
  font-weight: 500;
  transition: all var(--dur-fast) var(--ease);
  white-space: nowrap;
}
.tr-quick-btn:hover,
.tr-expand-btn:hover {
  background: var(--color-bg-hover);
  border-color: var(--color-border-dark);
  color: var(--color-text-primary);
}
.tr-quick-btn.active {
  background: var(--color-accent-light);
  color: var(--color-accent);
  border-color: var(--color-accent);
}
.tr-expand-btn {
  padding: 2px 8px;
  width: 24px;
  text-align: center;
  font-size: 0.85rem;
}
.tr-toolbar-folded {
  margin-top: 10px;
  padding-top: 10px;
  border-top: 1px dashed var(--color-border);
  display: flex;
  flex-direction: column;
  gap: var(--space-3);
}
.tr-toolbar-fold-enter-active,
.tr-toolbar-fold-leave-active {
  transition: all 0.22s var(--ease);
  overflow: hidden;
  opacity: 1;
}
.tr-toolbar-fold-enter-from,
.tr-toolbar-fold-leave-to {
  opacity: 0;
  max-height: 0;
  margin-top: 0;
  padding-top: 0;
}

.tr-legend {
  display: flex;
  flex-wrap: wrap;
  gap: var(--space-3);
}
.tr-legend-item {
  display: flex;
  align-items: center;
  gap: 4px;
  font-size: 0.7rem;
  color: var(--color-text-secondary);
  cursor: pointer;
  user-select: none;
  transition: all var(--dur-fast) var(--ease);
}
.tr-legend-item:hover { color: var(--color-text-primary); }
.tr-legend-item.off { opacity: 0.35; }
.tr-dot {
  width: 8px; height: 8px;
  border-radius: var(--radius-full);
  display: inline-block;
}
.tr-controls {
  display: flex;
  gap: var(--space-2);
  flex-wrap: wrap;
}
.tr-control-btn {
  padding: 2px 8px;
  font-size: 0.7rem;
  background: var(--color-bg-primary);
  color: var(--color-text-secondary);
  border: 1px solid var(--color-border);
  border-radius: var(--radius-sm);
  cursor: pointer;
  font-family: var(--font-sans);
  font-weight: 500;
  height: 24px;
  line-height: 1;
  transition: all var(--dur-fast) var(--ease);
}
.tr-control-btn:hover {
  background: var(--color-bg-hover);
  border-color: var(--color-border-dark);
  color: var(--color-text-primary);
  transform: translateY(-1px);
  box-shadow: var(--shadow-sm);
}
.tr-control-btn:active {
  background: var(--color-bg-active);
  transform: translateY(0);
  box-shadow: none;
}
.tr-control-btn.active {
  background: var(--color-accent-light);
  color: var(--color-accent);
  border-color: var(--color-accent);
}

/* Overlay panels */
.tr-overlay-panel {
  position: absolute;
  top: 48px;
  right: var(--space-4);
  z-index: 500;
}
.tr-close-overlay {
  position: absolute;
  top: var(--space-1);
  right: var(--space-1);
  background: none;
  border: none;
  color: var(--color-text-secondary);
  cursor: pointer;
  font-size: 1rem;
  transition: color var(--dur-fast) var(--ease);
}
.tr-close-overlay:hover {
  color: var(--color-text-primary);
}

/* A3 内容区 (现代化排版 古籍阅读感) */
.text-reader-content {
  flex: 1;
  overflow-y: auto;
  /* 书籍内页感：左右留白（收窄左侧空白，避免过宽） */
  padding: 28px 40px 100px 32px;
  /* 列宽约束，正文舒适宽度 */
  max-width: 820px;
  margin: 0 auto;
  width: 100%;
  box-sizing: border-box;
  position: relative;
}
/* 窄面板: 对比页/SplitView 内嵌时去掉外边距 */
.sv-panel .text-reader-content,
.dc-panel-body .text-reader-content {
  padding: 20px 18px 60px 16px;
  max-width: 100%;
}
.dc-panel-body .tr-title {
  max-width: 200px;
}

.tr-empty {
  display: flex;
  align-items: center;
  justify-content: center;
  height: 200px;
  color: var(--color-text-tertiary);
  font-size: 0.85rem;
}

/* 段落 grid：2 列 × 2 行（行1=段号+正文；行2=旁注弹出触发器用的绝对定位叠层区）
   正文仍然是第2列第1行，不压缩正文宽度。旁注触发器 & 浮层用 position absolute */
.tr-paragraph {
  display: grid;
  grid-template-columns: 30px 1fr;
  grid-template-rows: auto;
  gap: 0 10px;
  margin-bottom: 28px;
  padding: 6px 0;
  border: none;
  border-radius: 0;
  position: relative;
}
.tr-paragraph.tr-has-float-marg {
  padding-right: 0;
}
/* 段号行1、正文行1；旁注trigger / 浮层 absolute 挂在 .tr-paragraph 上，不占正文位置 */

/* ======== 旁注第 3 版：段号旁小锚点 + 章节级右侧 Drawer（参考 Google Docs / Apple Books） ======== */

/* 段落旁的小锚点：仅在有旁注/洞察时显示，位于段号列右边（正文左上角外侧）
   常态一条细蓝条，极轻量；hover / 有洞察时加粗变色 */
.tr-note-anchor {
  position: absolute;
  top: 6px;
  left: -6px;
  width: 4px;
  height: 24px;
  padding: 0;
  border: none;
  background: transparent;
  cursor: pointer;
  border-radius: 2px;
  z-index: 3;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  transition: all 140ms var(--ease);
}
.tr-note-anchor-bar {
  display: block;
  width: 3px;
  height: 100%;
  background: var(--color-accent, #5B9BD5);
  opacity: 0.55;
  border-radius: 2px;
  transition: all 140ms var(--ease);
}
.tr-note-anchor:hover .tr-note-anchor-bar,
.tr-paragraph:hover .tr-note-anchor .tr-note-anchor-bar {
  width: 4px;
  opacity: 1;
  background: var(--color-accent, #5B9BD5);
}
.tr-note-anchor.has-insight .tr-note-anchor-bar {
  /* 同时存在 AI 洞察 → 用渐变条，区分纯旁注段落 */
  background: linear-gradient(180deg,
      var(--color-accent, #5B9BD5) 0%,
      #7EC4A0 100%);
}

/* ======== Drawer 外层遮罩 & 滑入动画 ======== */
.tr-drawer-mask {
  position: fixed;
  inset: 0;
  background: rgba(15, 23, 42, 0.32);
  backdrop-filter: blur(1px);
  z-index: 80;
}
.tr-drawer-mask-enter-active,
.tr-drawer-mask-leave-active {
  transition: opacity 180ms var(--ease);
}
.tr-drawer-mask-enter-from,
.tr-drawer-mask-leave-to {
  opacity: 0;
}

/* Drawer 主体：贴阅读页右边缘（fixed right），高度=视口，干净的白底卡片风 */
.tr-notes-drawer {
  position: fixed;
  top: 0;
  right: 0;
  height: 100vh;
  width: min(420px, 92vw);
  background: var(--color-bg-primary, #fff);
  border-left: 1px solid var(--color-border, #E5E7EB);
  box-shadow: -8px 0 24px rgba(15, 23, 42, 0.08);
  z-index: 90;
  display: flex;
  flex-direction: column;
}
.tr-drawer-enter-active,
.tr-drawer-leave-active {
  transition: transform 220ms cubic-bezier(.22,.61,.36,1);
}
.tr-drawer-enter-from,
.tr-drawer-leave-to {
  transform: translateX(100%);
}

/* Drawer 头部 */
.tr-drawer-head {
  display: flex;
  align-items: center;
  gap: 10px;
  padding: 14px 16px 12px;
  border-bottom: 1px solid var(--color-border);
  background: linear-gradient(180deg,
      color-mix(in srgb, var(--color-accent-light) 50%, transparent) 0%,
      var(--color-bg-primary) 100%);
}
.tr-drawer-head-title {
  display: flex;
  align-items: center;
  gap: 8px;
}
/* 标题图标：纯 CSS 绘制"批注小卡片"——圆角矩形 + 右上角折叠三角形，不用 emoji */
.tr-drawer-title-icon {
  position: relative;
  width: 18px;
  height: 18px;
  flex-shrink: 0;
  background: color-mix(in srgb, var(--color-accent) 75%, transparent);
  border-radius: 3px;
  line-height: 0;
  display: inline-block;
}
.tr-drawer-title-icon::after {
  /* 右上角小折角 */
  content: '';
  position: absolute;
  top: 0;
  right: 0;
  width: 0;
  height: 0;
  border-top: 5px solid var(--color-bg-primary);
  border-right: 5px solid transparent;
  border-bottom: 5px solid transparent;
  border-left: 5px solid color-mix(in srgb, var(--color-accent) 55%, transparent);
  border-radius: 0 0 0 3px;
  background-clip: padding-box;
}
.tr-drawer-title-icon::before {
  /* 卡片上三条横线，模拟批注文字 */
  content: '';
  position: absolute;
  left: 3px;
  right: 3px;
  top: 4px;
  height: 7px;
  background-image:
    linear-gradient(var(--color-bg-primary) 1px, transparent 1px);
  background-size: 100% 3px;
  background-repeat: repeat-y;
}
.tr-drawer-title-text {
  font-size: 0.9rem;
  font-weight: 700;
  color: var(--color-text-primary);
  letter-spacing: 0.02em;
}
.tr-drawer-counts {
  display: inline-flex;
  gap: 6px;
  margin-left: auto;
}
.tr-drawer-chip {
  display: inline-flex;
  align-items: center;
  gap: 5px;
  padding: 2px 9px;
  font-size: 0.68rem;
  color: var(--color-text-secondary);
  background: var(--color-bg-tertiary);
  border: 1px solid var(--color-border);
  border-radius: 999px;
  font-weight: 600;
}
.tr-drawer-chip-dot {
  width: 6px; height: 6px;
  border-radius: 999px;
  display: inline-block;
}
.tr-drawer-close {
  width: 26px; height: 26px;
  border: 1px solid var(--color-border);
  background: var(--color-bg-primary);
  color: var(--color-text-secondary);
  border-radius: 6px;
  cursor: pointer;
  font-size: 1rem;
  line-height: 1;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  transition: all 120ms var(--ease);
}
.tr-drawer-close:hover {
  background: var(--color-bg-hover);
  color: var(--color-text-primary);
  border-color: var(--color-border-dark);
}

/* Drawer 空态 */
.tr-drawer-empty {
  padding: 60px 28px;
  text-align: center;
  color: var(--color-text-tertiary);
  font-size: 0.82rem;
  line-height: 1.8;
}

/* Drawer 主体滚动区 */
.tr-drawer-body {
  flex: 1;
  overflow-y: auto;
  padding: 14px 16px 32px;
  display: flex;
  flex-direction: column;
  gap: 16px;
}

/* 按段落分组 */
.tr-drawer-group {
  display: flex;
  flex-direction: column;
  gap: 8px;
}
.tr-drawer-group-head {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 6px 8px;
  border-radius: 6px;
  background: color-mix(in srgb, var(--color-bg-secondary) 60%, transparent);
  cursor: pointer;
  transition: background-color 120ms var(--ease);
}
.tr-drawer-group-head:hover {
  background: var(--color-bg-secondary);
}
.tr-drawer-group-idx {
  font-size: 0.72rem;
  font-weight: 700;
  color: var(--color-accent);
  font-family: var(--font-mono);
  flex-shrink: 0;
}
.tr-drawer-group-excerpt {
  flex: 1;
  font-size: 0.74rem;
  color: var(--color-text-secondary);
  line-height: 1.45;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}
.tr-drawer-group-arrow {
  font-size: 0.7rem;
  color: var(--color-text-tertiary);
  flex-shrink: 0;
}

/* 单条旁注/洞察条目 */
.tr-drawer-item {
  margin-left: 8px;
  border-left: 2.5px solid var(--color-border-dark);
  padding: 10px 12px;
  background: var(--color-bg-secondary);
  border-radius: 0 8px 8px 0;
  display: flex;
  flex-direction: column;
  gap: 8px;
  transition: all 140ms var(--ease);
}
.tr-drawer-item:hover {
  background: color-mix(in srgb, var(--color-accent-light) 25%, var(--color-bg-secondary));
}
.tr-drawer-item.focused {
  box-shadow: 0 0 0 2px color-mix(in srgb, var(--color-accent) 55%, transparent),
              0 4px 12px rgba(91, 155, 213, 0.18);
  border-left-width: 3px;
}
.tr-drawer-item.is-ai {
  border-left-color: #7EC4A0;
}
.tr-drawer-item-insight {
  border-left-color: color-mix(in srgb, var(--color-accent) 60%, transparent);
  background: color-mix(in srgb, var(--color-accent-light) 50%, transparent);
}
.tr-drawer-item-insight:hover {
  background: color-mix(in srgb, var(--color-accent-light) 85%, transparent);
}

/* 条目头：标签 + 时间 + 操作 */
.tr-drawer-item-head {
  display: flex;
  align-items: center;
  gap: 8px;
}
.tr-drawer-item-tag {
  display: inline-block;
  font-size: 0.62rem;
  font-weight: 700;
  padding: 2px 8px;
  border-radius: 999px;
  letter-spacing: 0.02em;
}
.tr-drawer-item-tag.tag-user {
  background: var(--color-accent-light);
  color: var(--color-accent);
}
.tr-drawer-item-tag.tag-ai {
  background: color-mix(in srgb, #7EC4A0 22%, transparent);
  color: #3D8A64;
}
.tr-drawer-item-tag.tag-insight {
  background: color-mix(in srgb, var(--color-accent) 18%, transparent);
  color: var(--color-accent);
}
.tr-drawer-item-time {
  font-size: 0.62rem;
  color: var(--color-text-tertiary);
  margin-right: auto;
}
.tr-drawer-item-actions {
  display: inline-flex;
  gap: 6px;
}
.tr-drawer-btn {
  padding: 1px 8px;
  font-size: 0.62rem;
  background: var(--color-bg-primary);
  color: var(--color-text-secondary);
  border: 1px solid var(--color-border);
  border-radius: 5px;
  cursor: pointer;
  transition: all 120ms var(--ease);
}
.tr-drawer-btn:hover {
  color: var(--color-accent);
  border-color: var(--color-accent);
  background: var(--color-accent-light);
}
.tr-drawer-btn-del:hover {
  color: var(--color-verdict-refuted);
  border-color: var(--color-verdict-refuted);
  background: var(--color-error-light);
}

/* 条目正文 */
.tr-drawer-item-body {
  font-size: 0.8rem;
  line-height: 1.65;
  color: var(--color-text-primary);
  white-space: pre-wrap;
  word-break: break-word;
}
.tr-drawer-item-motif {
  display: inline;
  font-size: 0.72rem;
  color: var(--color-accent);
  margin-left: 6px;
}

/* 旧悬浮卡片样式全部退役；避免未使用选择器警告（下方占位注释） */
/* Legacy popover classes deprecated → Drawer 方案见上 */

/* Split / DynastyCompare 窄面板：锚点标记向内偏移几 px，避免被分屏分隔条挡住 */
.sv-panel .tr-note-anchor,
.dc-panel-body .tr-note-anchor {
  left: 2px;
}
.tr-paragraph.para-active {
  background: transparent;
  border-color: transparent;
  /* 跳转定位蓝竖线：2px 细线条 + 低透明度，避免视觉过重 */
  box-shadow: inset 2px 0 0 color-mix(in srgb, var(--color-accent) 70%, transparent);
  animation: tr-para-pulse 1.8s var(--ease) infinite;
}
@keyframes tr-para-pulse {
  0%, 100% { box-shadow: inset 2px 0 0 color-mix(in srgb, var(--color-accent) 70%, transparent); }
  50% { box-shadow: inset 2px 0 0 color-mix(in srgb, var(--color-accent) 25%, transparent); }
}
/* 搜索结果跳转的暂时性 chunk 级高亮：仅具体命中词/短语，不是整段
   - 淡蓝底 + 细描边 + 轻柔呼吸闪烁（低饱和，避免视觉过重）
   - 离开阅读页即清除，重进不保留 */
.tr-para-text :deep(mark.search-highlight-ephemeral) {
  background: color-mix(in srgb, var(--color-accent, #5B9BD5) 13%, transparent);
  color: var(--color-text-primary);
  font-weight: 500;
  padding: 0 2px;
  border-radius: 3px;
  border: 1px solid color-mix(in srgb, var(--color-accent, #5B9BD5) 45%, transparent);
  box-shadow: 0 0 0 1px color-mix(in srgb, var(--color-accent, #5B9BD5) 18%, transparent);
  animation: tr-search-eph-pulse 1.4s var(--ease, ease) infinite;
}
@keyframes tr-search-eph-pulse {
  0%, 100% { box-shadow: 0 0 0 1px color-mix(in srgb, var(--color-accent, #5B9BD5) 18%, transparent); }
  50% { box-shadow: 0 0 4px 1px color-mix(in srgb, var(--color-accent, #5B9BD5) 32%, transparent); }
}
.tr-para-num {
  font-family: var(--font-mono);
  font-size: 0.72rem;
  color: var(--color-text-tertiary);
  grid-column: 1;
  grid-row: 1;
  min-width: 28px;
  padding-top: 6px;
  text-align: right;
  opacity: 0.65;
}
.tr-para-body {
  grid-column: 2;
  grid-row: 1;
  min-width: 0;
}
/* A3 正文排版: 字号加大、行距2、段首缩进2em、两端对齐 */
.tr-para-text {
  font-size: 1rem;
  line-height: 2;
  color: var(--color-text-primary);
  font-family: var(--font-sans);
  user-select: text;
  cursor: text;
  text-align: justify;
  text-indent: 2em;
}

.tr-para-text :deep(.annotation-tag) {
  cursor: pointer;
  padding: 0 1px;
  border-radius: var(--radius-sm);
  transition:
    background-color var(--dur-fast) var(--ease),
    box-shadow var(--dur-fast) var(--ease);
}

.tr-para-text :deep(.annotation-tag:hover) {
  box-shadow: 0 0 0 1px var(--color-border-dark);
}

.tr-para-text :deep(.marginalia-range) {
  cursor: pointer;
  transition:
    background-color var(--dur-fast) var(--ease),
    opacity var(--dur-fast) var(--ease);
}

.tr-para-text :deep(.marginalia-range:hover) {
  opacity: 0.85;
}

.tr-para-text :deep(.ann-suffix) {
  font-size: 0.58rem;
  opacity: 0.6;
  vertical-align: super;
  margin: 0 1px;
}

/* Search highlight mark */
.tr-para-text :deep(mark.search-highlight) {
  background: var(--color-accent-light);
  color: inherit;
  border-radius: var(--radius-sm);
  padding: 0 1px;
}

/* C06: Citation highlight from evolution narrative */
.tr-para-text :deep(mark.citation-highlight) {
  background: var(--color-accent-light);
  color: var(--color-text-primary);
  padding: 0 2px;
  border-radius: var(--radius-sm);
  font-weight: 500;
  border: 1px solid color-mix(in srgb, var(--color-accent) 45%, transparent);
}

/* Search action bar */
.tr-search-actions {
  margin-top: var(--space-2);
  padding: var(--space-2) var(--space-3);
  background: var(--color-accent-light);
  border: 1px solid var(--color-accent);
  border-radius: var(--radius-sm);
  display: flex;
  align-items: center;
  gap: var(--space-3);
  font-size: 0.72rem;
}

.tr-search-actions-label {
  color: var(--color-text-secondary);
  flex: 1;
}

.tr-search-source-tag {
  padding: 1px var(--space-2);
  border-radius: var(--radius-sm);
  font-size: 0.65rem;
  background: var(--color-bg-primary);
  color: var(--color-text-tertiary);
  border: 1px solid var(--color-border);
  margin-left: var(--space-1);
}

.tr-search-actions-btns {
  display: flex;
  gap: var(--space-1);
}

.tr-search-btn {
  padding: var(--space-1) var(--space-2);
  font-size: 0.68rem;
  border: 1px solid var(--color-border);
  border-radius: var(--radius-sm);
  cursor: pointer;
  background: var(--color-bg-primary);
  color: var(--color-text-primary);
  font-family: var(--font-sans);
  transition:
    background-color var(--dur-fast) var(--ease),
    border-color var(--dur-fast) var(--ease),
    color var(--dur-fast) var(--ease);
}

.tr-search-keep:hover {
  border-color: var(--color-accent);
  background: var(--color-accent-light);
  color: var(--color-accent);
}

.tr-search-dismiss:hover {
  border-color: var(--color-error);
  background: var(--color-error-light);
  color: var(--color-error);
}

/* Suggested annotations */
.tr-suggestions {
  margin-top: var(--space-2);
  padding: var(--space-2);
  background: var(--color-bg-secondary);
  border: 1px dashed var(--color-border);
  border-radius: var(--radius-sm);
}

.tr-suggestion-item {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: var(--space-2);
  padding: var(--space-1) 0;
  font-size: 0.72rem;
}

.tr-suggestion-label {
  color: var(--color-text-primary);
  font-weight: 500;
}

.tr-suggestion-reason {
  color: var(--color-text-tertiary);
  font-size: 0.68rem;
  font-style: italic;
}

.tr-suggestion-actions {
  display: flex;
  gap: var(--space-1);
  margin-left: auto;
}

.tr-sug-btn {
  padding: 1px var(--space-2);
  font-size: 0.65rem;
  background: var(--color-bg-primary);
  border: 1px solid var(--color-border);
  border-radius: var(--radius-sm);
  cursor: pointer;
  font-family: var(--font-sans);
  transition:
    background-color var(--dur-fast) var(--ease),
    border-color var(--dur-fast) var(--ease),
    color var(--dur-fast) var(--ease);
}

.tr-sug-accept:hover {
  border-color: var(--color-accent);
  background: var(--color-accent-light);
  color: var(--color-accent);
}

.tr-sug-dismiss:hover {
  border-color: var(--color-error);
  background: var(--color-error-light);
  color: var(--color-error);
}

.tr-loading-annotation {
  font-size: 0.72rem;
  color: var(--color-text-tertiary);
  margin-top: var(--space-2);
  font-style: italic;
}

/* A1: 旁注徽章 — 窄面板/对比页折叠时显示数量，hover/点击再展开气泡 */
.tr-marg-badge {
  position: absolute;
  right: 2px;
  top: 4px;
  width: 20px;
  height: 20px;
  line-height: 20px;
  text-align: center;
  border-radius: var(--radius-full);
  background: var(--color-accent);
  color: #fff;
  font-size: 0.62rem;
  font-weight: 700;
  cursor: pointer;
  box-shadow: 0 1px 4px rgba(0, 0, 0, 0.15);
  z-index: 30;
  transition:
    transform 150ms var(--ease),
    background-color 150ms var(--ease);
}
.tr-marg-badge:hover {
  transform: scale(1.15);
  background: var(--color-accent-hover);
}
.tr-marg-badge.pinned {
  background: var(--color-verdict-supported);
}

/* 以下为旧悬浮气泡外壳兼容占位（外壳已废弃，侧注改为段落下方 inline，见 .tr-inline-notes） */

/* 旁注列（用户旁注） */
/* 旧悬浮气泡外壳不再使用 → 侧注改为段落下方 inline（见 .tr-inline-notes）
   以下保留兼容占位：避免未引用的选择器警告，以及 Split/Compare 面板样式 */

/* SplitView / DynastyCompare 窄面板：侧注保持 inline（段落下方），无需调整宽度 */
.sv-panel .tr-paragraph.tr-has-float-marg,
.dc-panel-body .tr-paragraph.tr-has-float-marg {
  padding-right: 0;
}
/* split 面板内，badge 收起时 inline-notes 自动切换紧凑模式 */
.sv-panel .tr-inline-notes,
.dc-panel-body .tr-inline-notes {
  padding-left: 2px;
  margin-left: 6px;
  gap: 4px;
}
.sv-panel .tr-note-item,
.dc-panel-body .tr-note-item {
  padding: 6px 8px;
}
.sv-panel .tr-note-content,
.dc-panel-body .tr-note-content {
  font-size: 0.72rem;
}

/* 中等宽度 & 窄屏：inline-notes 已是段落下方流式，不需要额外响应式覆盖 */
@media (max-width: 1180px) {
  .tr-inline-notes {
    margin-left: 4px;
  }
}
@media (max-width: 900px) {
  .tr-inline-notes {
    margin-left: 0;
    padding-left: 2px;
    border-left: none;
  }
  .tr-note-item {
    padding: 6px 8px;
  }
}

/* Annotation right-click context menu */
.ann-ctx-backdrop {
  position: fixed;
  inset: 0;
  z-index: 8000;
}

.ann-ctx-menu {
  position: fixed;
  z-index: 8001;
  background: var(--color-bg-primary);
  border: 1px solid var(--color-border);
  border-radius: var(--radius-sm);
  box-shadow: var(--shadow-md);
  min-width: 170px;
  padding: var(--space-1) 0;
}

.ann-ctx-item {
  display: block;
  width: 100%;
  text-align: left;
  padding: var(--space-2) var(--space-4);
  background: none;
  border: none;
  color: var(--color-text-primary);
  font-family: var(--font-sans);
  font-size: 0.78rem;
  cursor: pointer;
  transition:
    background-color var(--dur-fast) var(--ease),
    color var(--dur-fast) var(--ease);
}

.ann-ctx-item:hover {
  background: var(--color-bg-hover);
}

.ann-ctx-primary {
  color: var(--color-accent);
  font-weight: 500;
}

.ann-ctx-danger {
  color: var(--color-error);
  font-weight: 500;
}

.ann-ctx-divider {
  height: 1px;
  background: var(--color-border-light);
  margin: var(--space-1) var(--space-2);
}
</style>

<!-- U-15: Non-scoped style for dynamically rendered linked-blink class -->
<style>
.annotation-tag.linked-blink {
  animation: linked-blink 1s ease infinite;
}

@keyframes linked-blink {
  0%, 100% { opacity: 1; }
  50% { opacity: 0.3; }
}
</style>
