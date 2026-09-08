import { defineStore } from 'pinia'
import { ref, computed } from 'vue'
// U-20: Pinia setup store import -- lazy resolution, no circular dependency risk
import { useSourceTreeStore } from '@/stores/sourceTree'
import { apiFetch } from '@/services/api'
import type {
  Annotation,
  UserAnnotation,
  AnnotationVisibility,
  PassageKey,
  Insight,
  Marginalia,
  AnnotationSuggestion,
  AnnotationSet,
  AnnotationCategory,
  AnnotationIndexItem,
  AnnotationStats,
  ExportJSON,
  AnnotationSpan,
} from '@/types/annotation'

function genId(): string {
  return Date.now().toString(36) + Math.random().toString(36).slice(2, 8)
}

const CATEGORY_COLORS: Record<string, string> = {
  person: '#6DA8D6',
  location: '#7EC4A0',
  event: '#6FB9C4',
  term: '#9AA6D4',
  motif: '#D09EBC',
  custom: '#B4B482',
}

function defaultVisibility(): AnnotationVisibility {
  const t = {
    person: true,
    location: true,
    event: true,
    term: true,
    motif: true,
    custom: true,
  }
  return { user: { ...t }, agent: { ...t } }
}

function loadVisibility(): AnnotationVisibility {
  return defaultVisibility()
}

async function saveVisibilityToServer(vis: AnnotationVisibility) {
  const resp = await apiFetch('/api/annotations/visibility', {
    method: 'PUT',
    body: JSON.stringify({ visibility: vis }),
  })
  if (!resp.ok) {
    console.error('[annotations] Failed to save visibility to server: HTTP ' + resp.status)
  }
}

function deriveColor(category: AnnotationCategory): string {
  return CATEGORY_COLORS[category] || CATEGORY_COLORS.custom
}

export const useAnnotationStore = defineStore('annotations', () => {
  // ===== State =====

  const agentAnnotations = ref<Record<PassageKey, Annotation[]>>({})
  const aiInsights = ref<Record<PassageKey, Insight[]>>({})
  const userAnnotations = ref<Record<PassageKey, UserAnnotation[]>>({})
  const userMarginalia = ref<Record<PassageKey, Marginalia[]>>({})
  const agentMarginalia = ref<Record<PassageKey, Marginalia[]>>({})
  const suggestedAnnotations = ref<Record<PassageKey, AnnotationSuggestion[]>>({})
  const visibility = ref<AnnotationVisibility>(loadVisibility())
  const agentConversationId = ref<string | null>(null)
  const editingAnnotationId = ref<string | null>(null)
  const editingMarginaliaId = ref<string | null>(null)
  // 阶段 2 — P2-04: 标注覆盖率状态
  const chapterCoverage = ref<Record<number, import('@/types/annotation').CoverageInfo>>({})

  // ===== Getters =====

  /**
   * 分栏（对比阅读）模式下 TextReaderView 以 `ch:p:left/right` 查询，
   * 而服务端加载与 SSE 流式写入的 agent 数据统一以裸 key `ch:p` 为准。
   * 带后缀 key 查不到时回退裸 key，保证对比视图中 AI 高亮/旁注/洞察正常显示。
   */
  function resolveAgentKey(record: Record<string, unknown[]>, key: string): string {
    if (record[key] && record[key]!.length > 0) return key
    const m = /^(\d+:\d+):(?:left|right)$/.exec(key)
    if (m && record[m[1]] && record[m[1]]!.length > 0) return m[1]
    return key
  }

  function visibleAnnotations(
    passageKey: PassageKey,
    splitSide?: 'left' | 'right'
  ): (Annotation | UserAnnotation)[] {
    const agentKey = resolveAgentKey(agentAnnotations.value as Record<string, unknown[]>, passageKey)
    const agent = (agentAnnotations.value[agentKey] || []).filter(
      a => visibility.value.agent[a.category]
    )
    const user = (userAnnotations.value[passageKey] || []).filter(
      a => visibility.value.user[a.category]
    )

    const combined: (Annotation | UserAnnotation)[] = [
      ...agent.map(a => ({ ...a, source: 'agent' as const })),
      ...user.map(a => ({ ...a, source: 'user' as const })),
    ]
    combined.sort((a, b) => a.span.startChar - b.span.startChar)
    return combined
  }

  function annotationsForKey(passageKey: PassageKey): Annotation[] {
    const key = resolveAgentKey(agentAnnotations.value as Record<string, unknown[]>, passageKey)
    return agentAnnotations.value[key] || []
  }

  function userAnnotationsForKey(passageKey: PassageKey): UserAnnotation[] {
    return userAnnotations.value[passageKey] || []
  }

  function insightsForKey(passageKey: PassageKey): Insight[] {
    const key = resolveAgentKey(aiInsights.value as Record<string, unknown[]>, passageKey)
    return aiInsights.value[key] || []
  }

  function marginaliaForKey(passageKey: PassageKey): Marginalia[] {
    const user = userMarginalia.value[passageKey] || []
    const agentKey = resolveAgentKey(agentMarginalia.value as Record<string, unknown[]>, passageKey)
    const agent = agentMarginalia.value[agentKey] || []
    return [...user, ...agent].sort((a, b) => a.createdAt - b.createdAt)
  }

  // 阶段 2 — P2-04: Coverage 状态管理
  function updateCoverage(chapterNumber: number, coverage: import('@/types/annotation').CoverageInfo) {
    chapterCoverage.value = {
      ...chapterCoverage.value,
      [chapterNumber]: {
        ...coverage,
        coverageRate: Math.max(0, Math.min(1, coverage.coverageRate)),
      },
    }
  }

  function coverageForChapter(chapterNumber: number): import('@/types/annotation').CoverageInfo | null {
    return chapterCoverage.value[chapterNumber] || null
  }

  function userAnnotationCount(chapterNumber: number): number {
    let count = 0
    for (const [key, anns] of Object.entries(userAnnotations.value)) {
      if (key.startsWith(`${chapterNumber}:`)) {
        count += anns.length
      }
    }
    return count
  }

  function agentAnnotationCount(chapterNumber: number): number {
    let count = 0
    for (const [key, anns] of Object.entries(agentAnnotations.value)) {
      if (key.startsWith(`${chapterNumber}:`)) {
        count += anns.length
      }
    }
    return count
  }

  function hasUserAnnotations(chapterNumber: number): boolean {
    for (const key of Object.keys(userAnnotations.value)) {
      if (key.startsWith(`${chapterNumber}:`)) return true
    }
    return false
  }

  function getAnnotationsByChapter(chapterNumber: number): {
    passageKey: PassageKey
    annotations: UserAnnotation[]
    marginalia: Marginalia[]
  }[] {
    const result: {
      passageKey: PassageKey
      annotations: UserAnnotation[]
      marginalia: Marginalia[]
    }[] = []
    for (const key of Object.keys(userAnnotations.value)) {
      if (key.startsWith(`${chapterNumber}:`)) {
        result.push({
          passageKey: key,
          annotations: userAnnotations.value[key] || [],
          marginalia: (userMarginalia.value[key] || []).concat(agentMarginalia.value[key] || []),
        })
      }
    }
    return result
  }

  function escapeHtmlExport(text: string): string {
    return text
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;')
  }

  function exportAnnotationsHTML(chapterNumber: number, includeAgent: boolean, includeUser: boolean = true, includeMarginalia: boolean = true, paragraphs?: { index: number; text: string }[]): string {
    const chapterData = getAnnotationsByChapter(chapterNumber)
    let html = `<!DOCTYPE html><html><head><meta charset="utf-8"><title>Chapter ${chapterNumber} Annotations</title><style>
      body { font-family: "Noto Serif SC", serif; line-height: 1.8; max-width: 860px; margin: 0 auto; padding: 20px; color: #333; }
      .ann-user { border: 2px solid; padding: 0 3px; border-radius: 3px; }
      .ann-agent { border: 2px dashed; padding: 0 3px; border-radius: 3px; }
      .marginalia { border-left: 3px solid #ccc; padding: 4px 8px; margin: 4px 0; font-size: 0.8rem; color: #666; }
      .legend { margin-bottom: 16px; padding: 8px; background: #f5f5f5; border-radius: 4px; font-size: 0.75rem; }
      .line-num { color: #999; font-size: 0.7rem; margin-right: 12px; user-select: none; display: inline-block; width: 28px; text-align: right; vertical-align: top; }
      .para-text { display: inline; }
      .para-block { margin-bottom: 18px; line-height: 1.9; }
      .ann-suffix { font-size: 0.6rem; opacity: 0.6; margin-left: 2px; }
      .export-title { font-size: 1.1rem; margin-bottom: 16px; border-bottom: 2px solid #ccc; padding-bottom: 8px; }
      @media print { body { print-color-adjust: exact; -webkit-print-color-adjust: exact; } }
    </style></head><body>`

    const chapterTitle = paragraphs && paragraphs.length > 0
      ? ` ${chapterNumber} `
      : `Chapter ${chapterNumber} Annotations`
    html += `<h1 class="export-title">${chapterTitle}</h1>`

    // Legend
    html += '<div class="legend">'
    for (const [cat, color] of Object.entries(CATEGORY_COLORS)) {
      html += `<span style="margin-right:14px"><span style="display:inline-block;width:10px;height:10px;background:${color};border-radius:50%;margin-right:4px"></span>${cat}</span>`
    }
    html += '<span style="margin-left:8px;font-size:0.7rem;color:#888">= · =Agent</span>'
    html += '</div>'

    // If paragraphs are provided, render full text with inline highlights
    if (paragraphs && paragraphs.length > 0) {
      for (const para of paragraphs) {
        const key = `${chapterNumber}:${para.index}`
        const userAnns = includeUser ? (userAnnotations.value[key] || []) : []
        const agentAnns = includeAgent ? (agentAnnotations.value[key] || []) : []
        const margs = includeMarginalia ? (userMarginalia.value[key] || []).concat(agentMarginalia.value[key] || []) : []

        // Merge all annotations for this paragraph
        const allAnns: { startChar: number; endChar: number; color: string; source: string; annotationId: string }[] = [
          ...userAnns.map(a => ({ startChar: a.span.startChar, endChar: a.span.endChar, color: a.color, source: 'user', annotationId: a.id })),
          ...agentAnns.map(a => ({ startChar: a.span.startChar, endChar: a.span.endChar, color: a.color, source: 'agent', annotationId: a.id })),
        ].sort((a, b) => a.startChar - b.startChar)

        html += `<div class="para-block">`
        html += `<span class="line-num">${para.index}</span>`

        let lastEnd = 0
        for (const ann of allAnns) {
          const start = Math.max(lastEnd, ann.startChar)
          if (start > lastEnd) {
            html += `<span class="para-text">${escapeHtmlExport(para.text.slice(lastEnd, start))}</span>`
          }
          const end = Math.min(ann.endChar, para.text.length)
          if (start < end) {
            const cls = ann.source === 'user' ? 'ann-user' : 'ann-agent'
            const suffix = ann.source === 'user' ? '' : ''
            html += `<span class="${cls}" style="border-color:${ann.color};background:${ann.color}20">${escapeHtmlExport(para.text.slice(start, end))}<sup class="ann-suffix">${suffix}</sup></span>`
          }
          lastEnd = Math.max(lastEnd, ann.endChar)
        }
        if (lastEnd < para.text.length) {
          html += `<span class="para-text">${escapeHtmlExport(para.text.slice(lastEnd))}</span>`
        }

        // Marginalia for this paragraph
        for (const marg of margs) {
          html += `<div class="marginalia" style="margin-left:40px">[] ${escapeHtmlExport(marg.content)}</div>`
        }

        html += '</div>'
      }
    } else {
      // Fallback: plain annotation listing
      if (includeUser) {
        for (const item of chapterData) {
          html += `<p><strong>Passage ${item.passageKey}</strong></p>`
          for (const ann of item.annotations) {
            html += `<p><span class="ann-user" style="border-color:${ann.color};background:${ann.color}20">${ann.label || '...'}</span> ${ann.note ? `- ${ann.note}` : ''}</p>`
          }
          if (includeMarginalia) {
            for (const marg of item.marginalia) {
              html += `<div class="marginalia">${escapeHtmlExport(marg.content)}</div>`
            }
          }
        }
      }

      if (includeAgent) {
        for (const [key, anns] of Object.entries(agentAnnotations.value)) {
          if (key.startsWith(`${chapterNumber}:`)) {
            html += `<p><strong>Passage ${key} (Agent)</strong></p>`
            for (const ann of anns) {
              html += `<p><span class="ann-agent" style="border-color:${ann.color};background:${ann.color}20">${ann.label}</span> ${ann.explanation}</p>`
            }
          }
        }
      }
    }

    html += '</body></html>'
    return html
  }

  function exportAnnotationsPlainText(chapterNumber: number, includeAgent: boolean, includeUser: boolean = true, includeMarginalia: boolean = true): string {
    const chapterData = includeUser ? getAnnotationsByChapter(chapterNumber) : []
    let text = `Chapter ${chapterNumber} Annotations\n${'='.repeat(30)}\n\n`

    for (const item of chapterData) {
      text += `Passage ${item.passageKey}:\n`
      for (const ann of item.annotations) {
        text += `  [${ann.category}] ${ann.label || '...'}`
        if (ann.note) text += ` - ${ann.note}`
        text += '\n'
      }
      if (includeMarginalia) {
        for (const marg of item.marginalia) {
          text += `  [Marginalia] ${marg.content}\n`
        }
      }
      text += '\n'
    }

    if (includeAgent) {
      text += `\nAgent Annotations\n${'='.repeat(30)}\n\n`
      for (const [key, anns] of Object.entries(agentAnnotations.value)) {
        if (key.startsWith(`${chapterNumber}:`)) {
          text += `Passage ${key}:\n`
          for (const ann of anns) {
            text += `  [${ann.category}] ${ann.label} - ${ann.explanation}\n`
          }
          text += '\n'
        }
      }
    }

    text += '\n--- End ---'
    return text
  }

  // ===== Actions =====

  function loadAgentAnnotations(
    annotations: Annotation[],
    passageKey: PassageKey,
    conversationId?: string,
    insights?: Insight[]
  ) {
    if (conversationId && agentConversationId.value !== conversationId) {
      clearAgentAnnotations()
      agentConversationId.value = conversationId
    }
    agentAnnotations.value = { ...agentAnnotations.value, [passageKey]: annotations }
    if (insights !== undefined) {
      aiInsights.value = { ...aiInsights.value, [passageKey]: insights }
    }
    const chNum = parseInt(passageKey.split(':')[0], 10)
    if (!isNaN(chNum) && chNum >= 1) {
      saveAgentToServer(chNum).catch(err => console.error('[annotations] Failed to save agent annotations:', err))
    }
  }

  // Alias for backward compatibility
  function loadAiAnnotations(
    annotations: Annotation[],
    passageKey: PassageKey,
    insights?: Insight[]
  ) {
    loadAgentAnnotations(annotations, passageKey, undefined, insights)
  }

  function addUserAnnotation(
    passageKey: PassageKey,
    annotation: Omit<UserAnnotation, 'id' | 'createdAt' | 'color'> & { color?: string }
  ) {
    if (!userAnnotations.value[passageKey]) {
      userAnnotations.value[passageKey] = []
    }
    const color = annotation.color || deriveColor(annotation.category)
    const newAnn: UserAnnotation = {
      id: genId(),
      ...annotation,
      color,
      createdAt: Date.now(),
    }
    userAnnotations.value[passageKey].push(newAnn)
    const chNum = parseInt(passageKey.split(':')[0], 10)
    saveToServer(chNum).catch(err => console.error('[annotations] Failed to save user annotation:', err))
  }

  function updateUserAnnotation(
    passageKey: PassageKey,
    annotationId: string,
    changes: Partial<Pick<UserAnnotation, 'note' | 'label' | 'category' | 'color' | 'span'>>
  ) {
    const anns = userAnnotations.value[passageKey]
    if (!anns) return
    const ann = anns.find(a => a.id === annotationId)
    if (!ann) return
    Object.assign(ann, changes)
    const chNum = parseInt(passageKey.split(':')[0], 10)
    saveToServer(chNum).catch(err => console.error('[annotations] Failed to save annotation update:', err))
  }

  function removeUserAnnotation(passageKey: PassageKey, annotationId: string) {
    if (!userAnnotations.value[passageKey]) return
    userAnnotations.value[passageKey] = userAnnotations.value[passageKey].filter(
      a => a.id !== annotationId
    )
    // Clean up orphaned marginalia that reference this annotation
    if (userMarginalia.value[passageKey]) {
      userMarginalia.value[passageKey] = userMarginalia.value[passageKey].filter(
        m => m.annotationId !== annotationId
      )
      if (userMarginalia.value[passageKey].length === 0) {
        delete userMarginalia.value[passageKey]
      }
    }
    const chNum = parseInt(passageKey.split(':')[0], 10)
    saveToServer(chNum).catch(err => console.error('[annotations] Failed to save after remove:', err))
  }

  function removeAgentAnnotation(passageKey: PassageKey, annotationId: string) {
    const anns = agentAnnotations.value[passageKey]
    if (!anns) return
    agentAnnotations.value[passageKey] = anns.filter(a => a.id !== annotationId)
    if (agentAnnotations.value[passageKey].length === 0) {
      delete agentAnnotations.value[passageKey]
    }
  }

  // Alias for backward compatibility
  function editUserAnnotation(
    passageKey: PassageKey,
    annotationId: string,
    changes: Partial<Pick<UserAnnotation, 'note' | 'label' | 'category'>>
  ) {
    updateUserAnnotation(passageKey, annotationId, changes)
  }

  // ===== Marginalia actions =====

  function addMarginalia(
    passageKey: PassageKey,
    data: Omit<Marginalia, 'id' | 'createdAt' | 'updatedAt' | 'color' | 'source'> & { color?: string; source?: 'user' | 'agent' }
  ) {
    if (!userMarginalia.value[passageKey]) {
      userMarginalia.value[passageKey] = []
    }
    const now = Date.now()
    const margColors = ['#E09A68', '#D08AA8', '#72BD8A', '#5B9BD5', '#9AA6D4']
    const existingCount = userMarginalia.value[passageKey].length
    const newMarg: Marginalia = {
      id: genId(),
      ...data,
      color: data.color || margColors[existingCount % margColors.length],
      source: data.source || 'user',
      createdAt: now,
      updatedAt: now,
    }
    userMarginalia.value[passageKey].push(newMarg)
    const chNum = parseInt(passageKey.split(':')[0], 10)
    if (!isNaN(chNum) && chNum >= 1) {
      saveToServer(chNum).catch(err => console.error('[annotations] Failed to save marginalia:', err))
    }
  }

  function _findMarginaliaStore(passageKey: PassageKey, marginaliaId: string): 'user' | 'agent' | null {
    const userMargs = userMarginalia.value[passageKey]
    if (userMargs && userMargs.some(m => m.id === marginaliaId)) return 'user'
    const agentMargs = agentMarginalia.value[passageKey]
    if (agentMargs && agentMargs.some(m => m.id === marginaliaId)) return 'agent'
    return null
  }

  function updateMarginalia(
    passageKey: PassageKey,
    marginaliaId: string,
    changes: Partial<Pick<Marginalia, 'content'>>
  ) {
    const store = _findMarginaliaStore(passageKey, marginaliaId)
    if (!store) return
    const margs = store === 'user'
      ? userMarginalia.value[passageKey]
      : agentMarginalia.value[passageKey]
    const marg = margs!.find(m => m.id === marginaliaId)
    if (!marg) return
    Object.assign(marg, changes, { updatedAt: Date.now() })
    const chNum = parseInt(passageKey.split(':')[0], 10)
    if (!isNaN(chNum) && chNum >= 1) {
      saveToServer(chNum).catch(err => console.error('[annotations] Failed to save marginalia update:', err))
    }
  }

  function removeMarginalia(passageKey: PassageKey, marginaliaId: string) {
    const store = _findMarginaliaStore(passageKey, marginaliaId)
    if (!store) return
    if (store === 'user') {
      userMarginalia.value[passageKey] = userMarginalia.value[passageKey].filter(
        m => m.id !== marginaliaId
      )
      if (userMarginalia.value[passageKey].length === 0) {
        delete userMarginalia.value[passageKey]
      }
    } else {
      agentMarginalia.value[passageKey] = agentMarginalia.value[passageKey].filter(
        m => m.id !== marginaliaId
      )
      if (agentMarginalia.value[passageKey].length === 0) {
        delete agentMarginalia.value[passageKey]
      }
    }
    const chNum = parseInt(passageKey.split(':')[0], 10)
    if (!isNaN(chNum) && chNum >= 1) {
      saveToServer(chNum).catch(err => console.error('[annotations] Failed to save after marginalia remove:', err))
    }
  }

  function clearAgentMarginaliaForChapter(chapterNumber: number) {
    const prefix = `${chapterNumber}:`
    for (const key of Object.keys(agentMarginalia.value)) {
      if (key.startsWith(prefix)) {
        delete agentMarginalia.value[key]
      }
    }
  }

  function clearAgentAnnotationsForChapter(chapterNumber: number) {
    const prefix = `${chapterNumber}:`
    for (const key of Object.keys(agentAnnotations.value)) {
      if (key.startsWith(prefix)) {
        delete agentAnnotations.value[key]
      }
    }
    for (const key of Object.keys(aiInsights.value)) {
      if (key.startsWith(prefix)) {
        delete aiInsights.value[key]
      }
    }
    for (const key of Object.keys(agentMarginalia.value)) {
      if (key.startsWith(prefix)) {
        delete agentMarginalia.value[key]
      }
    }
    // Delete agent annotations from server
    apiFetch('/api/annotations', {
      method: 'DELETE',
      body: JSON.stringify({ chapterNumber, cleanAgent: true, deleteAll: true }),
    }).catch(err => console.error('[annotations] Failed to clear agent annotations on server:', err))
  }

  function loadAgentMarginalia(
    passageKey: PassageKey,
    marginaliaList: Marginalia[]
  ) {
    if (!agentMarginalia.value[passageKey]) {
      agentMarginalia.value[passageKey] = []
    }
    const existing = agentMarginalia.value[passageKey]
    for (const m of marginaliaList) {
      const idx = existing.findIndex(e => e.id === m.id)
      if (idx === -1) {
        existing.push(m)
      } else {
        // 同 id 旁注（重跑标注）：内容就地更新，避免旧文案残留
        existing[idx] = m
      }
    }
    const chNum = parseInt(passageKey.split(':')[0], 10)
    if (!isNaN(chNum) && chNum >= 1) {
      saveAgentToServer(chNum).catch(err => console.error('[annotations] Failed to save agent marginalia:', err))
    }
  }

  // ===== Suggested annotations =====

  function loadSuggestedAnnotations(
    passageKey: PassageKey,
    suggestions: AnnotationSuggestion[]
  ) {
    suggestedAnnotations.value[passageKey] = suggestions
  }

  function acceptSuggestion(passageKey: PassageKey, suggestionId: string) {
    const suggestions = suggestedAnnotations.value[passageKey]
    if (!suggestions) return
    const sug = suggestions.find(s => s.id === suggestionId)
    if (!sug) return
    addUserAnnotation(passageKey, {
      category: sug.category,
      span: sug.span,
      label: sug.text,
      note: '',
      color: sug.color,
    })
    suggestedAnnotations.value[passageKey] = suggestions.filter(s => s.id !== suggestionId)
  }

  function acceptAllSuggestions(passageKey: PassageKey) {
    const suggestions = suggestedAnnotations.value[passageKey]
    if (!suggestions) return
    for (const sug of [...suggestions]) {
      acceptSuggestion(passageKey, sug.id)
    }
    suggestedAnnotations.value[passageKey] = []
  }

  function dismissSuggestion(passageKey: PassageKey, suggestionId: string) {
    const suggestions = suggestedAnnotations.value[passageKey]
    if (!suggestions) return
    suggestedAnnotations.value[passageKey] = suggestions.filter(s => s.id !== suggestionId)
  }

  function dismissAllSuggestions(passageKey: PassageKey) {
    delete suggestedAnnotations.value[passageKey]
  }

  // ===== Adopt agent annotations =====

  function adoptAgentAnnotation(passageKey: PassageKey, annotationId: string) {
    const agentAnns = agentAnnotations.value[passageKey]
    if (!agentAnns) return
    const ann = agentAnns.find(a => a.id === annotationId)
    if (!ann) return
    addUserAnnotation(passageKey, {
      category: ann.category,
      span: ann.span,
      label: ann.label,
      note: ann.explanation,
      color: ann.color,
    })
    // U-11: Remove the original agent annotation to prevent double rendering
    agentAnnotations.value[passageKey] = agentAnns.filter(a => a.id !== annotationId)
  }

  function adoptAllAgentAnnotations(passageKey: PassageKey) {
    const agentAnns = agentAnnotations.value[passageKey]
    if (!agentAnns) return
    for (const ann of [...agentAnns]) {
      adoptAgentAnnotation(passageKey, ann.id)
    }
    // U-11: Clear any remaining agent annotations after adoption
    agentAnnotations.value[passageKey] = []
  }

  // U-10: Save and adopt all agent annotations for a chapter (adopt + persist to server)
  function saveAndAdoptAllAgentAnnotations(chapterNumber: number) {
    const prefix = `${chapterNumber}:`
    const keysToProcess: PassageKey[] = []
    for (const key of Object.keys(agentAnnotations.value)) {
      if (key.startsWith(prefix) && agentAnnotations.value[key].length > 0) {
        keysToProcess.push(key)
      }
    }
    if (keysToProcess.length === 0) return
    for (const key of keysToProcess) {
      const agentAnns = agentAnnotations.value[key]
      if (!agentAnns) continue
      for (const ann of [...agentAnns]) {
        addUserAnnotation(key, {
          category: ann.category,
          span: ann.span,
          label: ann.label,
          note: ann.explanation,
          color: ann.color,
        })
      }
      // Clear agent annotations after adoption
      agentAnnotations.value[key] = []
    }
    // Persist to server
    saveToServer(chapterNumber).catch(err => console.error('[annotations] Failed to save after adopt-all:', err))
  }

  // ===== Visibility =====

  function toggleVisibility(layer: 'user' | 'agent', category: keyof AnnotationVisibility['user']) {
    visibility.value[layer][category] = !visibility.value[layer][category]
    saveVisibilityToServer(visibility.value)
  }

  function showAll() {
    const t = { person: true, location: true, event: true, term: true, motif: true, custom: true }
    visibility.value = { user: { ...t }, agent: { ...t } }
    saveVisibilityToServer(visibility.value)
  }

  function hideAll() {
    const f = { person: false, location: false, event: false, term: false, motif: false, custom: false }
    visibility.value = { user: { ...f }, agent: { ...f } }
    saveVisibilityToServer(visibility.value)
  }

  function showUserOnly() {
    const t = { person: true, location: true, event: true, term: true, motif: true, custom: true }
    const f = { person: false, location: false, event: false, term: false, motif: false, custom: false }
    visibility.value = { user: { ...t }, agent: { ...f } }
    saveVisibilityToServer(visibility.value)
  }

  function showAgentOnly() {
    const f = { person: false, location: false, event: false, term: false, motif: false, custom: false }
    const t = { person: true, location: true, event: true, term: true, motif: true, custom: true }
    visibility.value = { user: { ...f }, agent: { ...t } }
    saveVisibilityToServer(visibility.value)
  }

  // ===== Agent annotations lifecycle =====

  function clearAgentAnnotations(passageKey?: PassageKey) {
    if (passageKey) {
      delete agentAnnotations.value[passageKey]
      delete aiInsights.value[passageKey]
      delete agentMarginalia.value[passageKey]
    } else {
      agentAnnotations.value = {}
      aiInsights.value = {}
      agentMarginalia.value = {}
      agentConversationId.value = null
    }
  }

  function clearUserAnnotationsForKey(passageKey: PassageKey) {
    delete userAnnotations.value[passageKey]
    delete userMarginalia.value[passageKey]
  }

  // ===== Server persistence (SQLite — single source of truth) =====

  // C04: CSV export (17 columns per design doc C.1.1, BOM header, special char escaping)
  function exportAnnotationsCSV(chapterNumbers: number[], includeAgent: boolean, includeUser: boolean = true, includeMarginalia: boolean = true): string {
    const rows: string[] = []
    // Column order per design doc C.1.1: id, passageKey, chapterNumber, dynasty, chapterTitle,
    // paragraphIndex, category, categoryLabel, label, text, startChar, endChar, color, source,
    // note, marginaliaCount, createdAt
    const header = [
      'id', 'passageKey', 'chapterNumber', 'dynasty', 'chapterTitle',
      'paragraphIndex', 'category', 'categoryLabel', 'label', 'text',
      'startChar', 'endChar', 'color', 'source', 'note', 'marginaliaCount', 'createdAt'
    ]
    rows.push(header.join(','))

    const CATEGORY_LABELS: Record<string, string> = {
      person: '人物', location: '地点', event: '事件', term: '术语', motif: '母题', custom: '自定义',
    }

    const chs = chapters()

    for (const chNum of chapterNumbers) {
      const prefix = `${chNum}:`
      const ch = chs.find(c => c.number === chNum)
      const dynasty = ch?.dynasty || ''
      const chapterTitle = ch?.title || ''

      // User annotations
      if (includeUser) {
        for (const [key, anns] of Object.entries(userAnnotations.value)) {
          if (!key.startsWith(prefix)) continue
          for (const ann of anns) {
            const parts = key.split(':')
            const paraIdx = parseInt(parts[1], 10) || 0
            const margs = includeMarginalia ? (userMarginalia.value[key] || []).concat(agentMarginalia.value[key] || []).length : 0
            rows.push([
              `"${ann.id}"`,
              `"${key}"`,
              chNum,
              `"${dynasty}"`,
              `"${chapterTitle.replace(/"/g, '""')}"`,
              paraIdx,
              ann.category,
              `"${CATEGORY_LABELS[ann.category] || ann.category}"`,
              `"${(ann.label || '').replace(/"/g, '""')}"`,
              `"${(ann.label || '').replace(/"/g, '""')}"`, // text = label (UserAnnotation has no text field)
              ann.span.startChar,
              ann.span.endChar,
              `"${ann.color}"`,
              'user',
              `"${(ann.note || '').replace(/"/g, '""')}"`,
              margs,
              ann.createdAt,
            ].join(','))
          }
        }
      }
      // Agent annotations
      if (includeAgent) {
        for (const [key, anns] of Object.entries(agentAnnotations.value)) {
          if (!key.startsWith(prefix)) continue
          for (const ann of anns) {
            const parts = key.split(':')
            const paraIdx = parseInt(parts[1], 10) || 0
            const margs = includeMarginalia ? (userMarginalia.value[key] || []).concat(agentMarginalia.value[key] || []).length : 0
            rows.push([
              `"${ann.id}"`,
              `"${key}"`,
              chNum,
              `"${dynasty}"`,
              `"${chapterTitle.replace(/"/g, '""')}"`,
              paraIdx,
              ann.category,
              `"${CATEGORY_LABELS[ann.category] || ann.category}"`,
              `"${ann.label.replace(/"/g, '""')}"`,
              `"${(ann.text || '').replace(/"/g, '""')}"`,
              ann.span.startChar,
              ann.span.endChar,
              `"${ann.color}"`,
              'agent',
              `"${(ann.explanation || '').replace(/"/g, '""')}"`,
              margs,
              ann.createdAt,
            ].join(','))
          }
        }
      }
    }

    return '' + rows.join('\n')
  }

  // C04: JSON export per design doc C.1.1 ExportJSON schema
  function exportAnnotationsJSON(chapterNumbers: number[], includeAgent: boolean, includeUser: boolean = true, includeMarginalia: boolean = true, includeParagraphTexts: boolean = false, paragraphs?: Record<number, { index: number; text: string }[]>): string {
    const CATEGORY_LABELS: Record<string, string> = {
      person: '人物', location: '地点', event: '事件', term: '术语', motif: '母题', custom: '自定义',
    }
    const exportData: ExportJSON = {
      exportMeta: {
        exportedAt: new Date().toISOString(),
        platform: '白蛇传·文脉全息博物馆',
        version: 'v10',
        formatVersion: '1.0',
      },
      annotationSets: {},
    }

    for (const chNum of chapterNumbers) {
      const prefix = `${chNum}:`
      const items: ExportJSON['annotationSets'][string]['annotations'] = []

      if (includeUser) {
        for (const [key, anns] of Object.entries(userAnnotations.value)) {
          if (!key.startsWith(prefix)) continue
          for (const ann of anns) {
            const paraIdx = parseInt(key.split(':')[1], 10) || 0
            items.push({
              id: ann.id,
              category: ann.category,
              categoryLabel: CATEGORY_LABELS[ann.category] || ann.category,
              label: ann.label || '',
              text: ann.label || '',
              span: ann.span,
              color: ann.color,
              source: 'user' as const,
              passageKey: key,
              paragraphIndex: paraIdx,
              note: ann.note,
              createdAt: ann.createdAt,
              updatedAt: ann.createdAt,
              marginalia: includeMarginalia ? (userMarginalia.value[key] || []).concat(agentMarginalia.value[key] || []).map(m => ({ id: m.id, content: m.content })) : [],
            })
          }
        }
      }

      if (includeAgent) {
        for (const [key, anns] of Object.entries(agentAnnotations.value)) {
          if (!key.startsWith(prefix)) continue
          for (const ann of anns) {
            const paraIdx = parseInt(key.split(':')[1], 10) || 0
            items.push({
              id: ann.id,
              category: ann.category,
              categoryLabel: CATEGORY_LABELS[ann.category] || ann.category,
              label: ann.label,
              text: ann.text || '',
              span: ann.span,
              color: ann.color,
              source: 'agent' as const,
              passageKey: key,
              paragraphIndex: paraIdx,
              note: ann.explanation,
              confidence: ann.confidence,
              agentAnnotationId: ann.id,
              motifRef: ann.motifRef,
              createdAt: ann.createdAt,
              updatedAt: ann.updatedAt,
              marginalia: [],
            })
          }
        }
      }

      const ch = chapters().find(c => c.number === chNum)
      exportData.annotationSets[String(chNum)] = {
        chapterNumber: chNum,
        chapterTitle: ch?.title || '',
        dynasty: ch?.dynasty || '',
        annotations: items,
        marginalia: [],
        paragraphTexts: paragraphs && paragraphs[chNum]
          ? paragraphs[chNum].map(p => ({ paragraphIndex: p.index, text: p.text }))
          : undefined,
        savedAt: new Date().toISOString(),
      }

      // Collect marginalia entries for this chapter (user + agent)
      if (includeMarginalia) {
        const chapterMarginalia: { id: string; content: string }[] = []
        const margPrefix = `${chNum}:`
        for (const [key, margs] of Object.entries(userMarginalia.value)) {
          if (key.startsWith(margPrefix)) {
            chapterMarginalia.push(...margs.map(m => ({ id: m.id, content: m.content })))
          }
        }
        for (const [key, margs] of Object.entries(agentMarginalia.value)) {
          if (key.startsWith(margPrefix)) {
            chapterMarginalia.push(...margs.map(m => ({ id: m.id, content: m.content })))
          }
        }
        if (chapterMarginalia.length > 0) {
          exportData.annotationSets[String(chNum)].marginalia = chapterMarginalia
        }
      }
    }

    return JSON.stringify(exportData, null, 2)
  }

  // C04: TEI P5 XML export
  function exportAnnotationsTEI(chapterNumbers: number[], includeAgent: boolean, includeUser: boolean = true, includeMarginalia: boolean = true, mode: 'standOff' | 'inline' = 'standOff', paragraphs?: Record<number, { index: number; text: string }[]>): string {
    const CATEGORY_LABELS: Record<string, string> = {
      person: '人物', location: '地点', event: '事件', term: '术语', motif: '母题', custom: '自定义',
    }
    let xml = '<?xml version="1.0" encoding="UTF-8"?>\n'
    xml += '<TEI xmlns="http://www.tei-c.org/ns/1.0">\n'
    xml += '  <teiHeader>\n'
    xml += '    <fileDesc>\n'
    xml += '      <titleStmt>\n'
    xml += '        <title>White Snake Museum - Annotations Export</title>\n'
    xml += '      </titleStmt>\n'
    xml += '      <publicationStmt><p>Generated by White Snake Museum</p></publicationStmt>\n'
    xml += `      <sourceDesc><p>Export date: ${new Date().toISOString()}</p></sourceDesc>\n`
    xml += '    </fileDesc>\n'
    xml += '    <encodingDesc>\n'
    xml += '      <classDecl>\n'
    xml += '        <taxonomy xml:id="annotation-categories">\n'
    for (const [cat, label] of Object.entries(CATEGORY_LABELS)) {
      xml += `          <category xml:id="cat-${cat}"><catDesc>${label}</catDesc></category>\n`
    }
    xml += '        </taxonomy>\n'
    xml += '      </classDecl>\n'
    xml += '    </encodingDesc>\n'
    xml += '  </teiHeader>\n'
    xml += '  <text>\n'

    for (const chNum of chapterNumbers) {
      xml += `    <div type="chapter" n="${chNum}">\n`

      if (mode === 'standOff') {
        xml += '      <body>\n'
        if (paragraphs && paragraphs[chNum]) {
          for (const para of paragraphs[chNum]) {
            xml += `        <p n="${para.index}">${para.text}</p>\n`
          }
        }
        xml += '      </body>\n'
        xml += '      <standOff>\n'
        const prefix = `${chNum}:`
        const pushAnns = (anns: (UserAnnotation | Annotation)[], source: string) => {
          for (const ann of anns) {
            xml += `        <fs type="annotation" xml:id="ann-${ann.id}" source="${source}">\n`
            xml += `          <f name="category"><symbol value="${ann.category}"/></f>\n`
            xml += `          <f name="label"><string>${ann.label || ''}</string></f>\n`
            xml += `          <f name="span"><ptr target="#char-${ann.span.startChar}-${ann.span.endChar}"/></f>\n`
            if ('note' in ann && ann.note) {
              xml += `          <f name="note"><string>${ann.note}</string></f>\n`
            }
            if ('explanation' in ann && ann.explanation) {
              xml += `          <f name="explanation"><string>${(ann as Annotation).explanation}</string></f>\n`
            }
            if ('confidence' in ann && (ann as Annotation).confidence) {
              xml += `          <f name="confidence"><numeric value="${(ann as Annotation).confidence}"/></f>\n`
            }
            xml += '        </fs>\n'
          }
        }
        if (includeUser) {
          for (const [key, anns] of Object.entries(userAnnotations.value)) {
            if (key.startsWith(prefix)) pushAnns(anns, 'user')
          }
        }
        if (includeAgent) {
          for (const [key, anns] of Object.entries(agentAnnotations.value)) {
            if (key.startsWith(prefix)) pushAnns(anns, 'agent')
          }
        }
        xml += '      </standOff>\n'
      } else {
        // inline mode
        if (paragraphs && paragraphs[chNum]) {
          for (const para of paragraphs[chNum]) {
            const key = `${chNum}:${para.index}`
            const userAnns = includeUser ? (userAnnotations.value[key] || []) : []
            const agentAnns = includeAgent ? (agentAnnotations.value[key] || []) : []
            const allAnns = [
              ...userAnns.map(a => ({ ...a, source: 'user' as const })),
              ...agentAnns.map(a => ({ ...a, source: 'agent' as const })),
            ].sort((a, b) => a.span.startChar - b.span.startChar)

            let paraXml = ''
            let lastEnd = 0
            for (const ann of allAnns) {
              const start = Math.max(lastEnd, ann.span.startChar)
              if (start > lastEnd) {
                paraXml += para.text.slice(lastEnd, start)
              }
              const end = Math.min(ann.span.endChar, para.text.length)
              if (start < end) {
                paraXml += `<rs type="${ann.category}" ana="#ann-${ann.id}">${para.text.slice(start, end)}</rs>`
              }
              lastEnd = Math.max(lastEnd, ann.span.endChar)
            }
            if (lastEnd < para.text.length) {
              paraXml += para.text.slice(lastEnd)
            }
            xml += `      <p n="${para.index}">${paraXml}</p>\n`
          }
        }
      }

      xml += `    </div>\n`
    }

    xml += '  </text>\n</TEI>'
    return xml
  }

  // C04: Agent-only export shortcuts (convenience wrappers)
  function exportAgentOnlyCSV(chapterNumbers: number[]): string {
    return exportAnnotationsCSV(chapterNumbers, true, false, true)
  }

  function exportAgentOnlyJSON(chapterNumbers: number[]): string {
    return exportAnnotationsJSON(chapterNumbers, true, false, true, false, undefined)
  }

  function exportAgentOnlyHTML(chapterNumber: number, includeMarginalia?: boolean, paragraphs?: { index: number; text: string }[]): string {
    return exportAnnotationsHTML(chapterNumber, true, false, includeMarginalia ?? true, paragraphs)
  }

  function exportAgentOnlyPlainText(chapterNumber: number): string {
    return exportAnnotationsPlainText(chapterNumber, true, false, true)
  }

  function exportAgentOnlyTEI(chapterNumbers: number[], mode?: 'standOff' | 'inline', paragraphs?: Record<number, { index: number; text: string }[]>): string {
    return exportAnnotationsTEI(chapterNumbers, true, false, true, mode, paragraphs)
  }

  // C05: Build flat annotation index
  function buildAnnotationIndex(includeAgent: boolean): AnnotationIndexItem[] {
    const items: AnnotationIndexItem[] = []
    const CATEGORY_LABELS: Record<string, string> = {
      person: '人物', location: '地点', event: '事件', term: '术语', motif: '母题', custom: '自定义',
    }
    const DYN_ORDER: Record<string, number> = { '唐': 1, '宋': 2, '元': 3, '明': 4, '清': 5 }

    const processAnnotations = (
      record: Record<PassageKey, UserAnnotation[] | Annotation[]>,
      source: 'user' | 'agent'
    ) => {
      for (const [key, anns] of Object.entries(record)) {
        const parts = key.split(':')
        const chNum = parseInt(parts[0], 10)
        const paraIdx = parseInt(parts[1], 10) || 0
        const ch = chapters().find(c => c.number === chNum)
        const dynasty = ch?.dynasty || ''
        const title = ch?.title || ''

        for (const ann of anns) {
          const note = source === 'user' ? (ann as UserAnnotation).note : (ann as Annotation).explanation
          const confidence = source === 'agent' ? (ann as Annotation).confidence : undefined
          const text = source === 'agent' ? (ann as Annotation).text : undefined
          const para = ch?.paragraphs.find(p => p.index === paraIdx)
          const snippet = para ? para.text.slice(0, 60) : ''

          items.push({
            id: ann.id,
            category: ann.category,
            categoryLabel: CATEGORY_LABELS[ann.category] || ann.category,
            label: ann.label || '',
            text: text || '',
            span: ann.span,
            color: ann.color,
            source,
            passageKey: key,
            chapterNumber: chNum,
            dynasty,
            chapterTitle: title,
            paragraphIndex: paraIdx,
            contextSnippet: snippet,
            note: note || undefined,
            confidence,
            createdAt: ann.createdAt,
            marginalia: (userMarginalia.value[key] || []).concat(agentMarginalia.value[key] || []).map(m => ({ id: m.id, content: m.content })),
          })
        }
      }
    }

    processAnnotations(userAnnotations.value, 'user')
    if (includeAgent) {
      processAnnotations(agentAnnotations.value, 'agent')
    }

    // Sort by chapterNumber, then paragraphIndex, then startChar
    items.sort((a, b) => {
      if (a.chapterNumber !== b.chapterNumber) return a.chapterNumber - b.chapterNumber
      if (a.paragraphIndex !== b.paragraphIndex) return a.paragraphIndex - b.paragraphIndex
      return a.span.startChar - b.span.startChar
    })

    return items
  }

  // C05: Compute annotation statistics
  function computeAnnotationStats(includeAgent: boolean): AnnotationStats {
    const CATEGORY_LABELS: Record<string, string> = {
      person: '人物', location: '地点', event: '事件', term: '术语', motif: '母题', custom: '自定义',
    }
    const DYN_ORDER: Record<string, number> = { '唐': 1, '宋': 2, '元': 3, '明': 4, '清': 5 }

    let total = 0, userCount = 0, agentCount = 0, withMarginalia = 0
    let paragraphsCovered = 0, totalParagraphs = 0

    const categoryMap: Record<string, { count: number; userCount: number; agentCount: number; color: string }> = {}
    const dynastyMap: Record<string, number> = {}
    const chapterDensityMap: Record<number, { annotated: Set<number>; total: number }> = {}
    const userVsAgentMap: Record<string, { userCount: number; agentCount: number }> = {}
    const timeline: AnnotationStats['timeline'] = []

    const chs = chapters()
    for (const ch of chs) {
      totalParagraphs += ch.paragraphs.length
      chapterDensityMap[ch.number] = { annotated: new Set(), total: ch.paragraphs.length }
    }

    const processAnnotations = (
      record: Record<PassageKey, UserAnnotation[] | Annotation[]>,
      source: 'user' | 'agent'
    ) => {
      for (const [key, anns] of Object.entries(record)) {
        const parts = key.split(':')
        const chNum = parseInt(parts[0], 10)
        const paraIdx = parseInt(parts[1], 10) || 0
        const ch = chs.find(c => c.number === chNum)
        const dynasty = ch?.dynasty || ''

        if (chapterDensityMap[chNum]) {
          chapterDensityMap[chNum].annotated.add(paraIdx)
        }

        for (const ann of anns) {
          total++
          if (source === 'user') userCount++; else agentCount++

          // Category distribution
          if (!categoryMap[ann.category]) {
            categoryMap[ann.category] = { count: 0, userCount: 0, agentCount: 0, color: ann.color || CATEGORY_COLORS[ann.category] || '' }
          }
          categoryMap[ann.category].count++
          if (source === 'user') categoryMap[ann.category].userCount++
          else categoryMap[ann.category].agentCount++

          // Dynasty distribution
          if (dynasty) {
            dynastyMap[dynasty] = (dynastyMap[dynasty] || 0) + 1
          }

          // User vs agent by category
          if (!userVsAgentMap[ann.category]) {
            userVsAgentMap[ann.category] = { userCount: 0, agentCount: 0 }
          }
          if (source === 'user') userVsAgentMap[ann.category].userCount++
          else userVsAgentMap[ann.category].agentCount++

          // Timeline
          // P1-14: Check marginalia from both user and agent stores, not just note field
          const note = source === 'user' ? (ann as UserAnnotation).note : undefined
          const hasMarginaliaFromStore = (userMarginalia.value[key] || []).some(
            m => m.annotationId === ann.id
          ) || (agentMarginalia.value[key] || []).some(
            m => m.annotationId === ann.id
          )
          if (note || hasMarginaliaFromStore) withMarginalia++
          timeline.push({
            action: source === 'user' ? 'user_annotate' : 'agent_annotate',
            annotationId: ann.id,
            passageKey: key,
            category: ann.category,
            label: ann.label || '',
            timestamp: ann.createdAt,
            timestampLabel: new Date(ann.createdAt).toLocaleString('zh-CN'),
          })
        }
      }
    }

    processAnnotations(userAnnotations.value, 'user')
    if (includeAgent) {
      processAnnotations(agentAnnotations.value, 'agent')
    }

    // Calculate paragraphs covered
    for (const data of Object.values(chapterDensityMap)) {
      paragraphsCovered += data.annotated.size
    }

    // Build result
    const categoryDistribution = Object.entries(categoryMap).map(([cat, data]) => ({
      category: cat,
      categoryLabel: CATEGORY_LABELS[cat] || cat,
      ...data,
      percentage: total > 0 ? Math.round((data.count / total) * 10000) / 100 : 0,
    }))

    const dynastyDistribution = Object.entries(dynastyMap)
      .map(([dynasty, count]) => ({
        dynasty,
        dynastyOrder: DYN_ORDER[dynasty] || 0,
        count,
        percentage: total > 0 ? Math.round((count / total) * 10000) / 100 : 0,
      }))
      .sort((a, b) => a.dynastyOrder - b.dynastyOrder)

    const chapterDensity = Object.entries(chapterDensityMap).map(([chNum, data]) => {
      const ch = chs.find(c => c.number === parseInt(chNum, 10))
      return {
        chapterNumber: parseInt(chNum, 10),
        chapterTitle: ch?.title || '',
        dynasty: ch?.dynasty || '',
        annotationCount: 0, // Will be computed below
        totalParagraphs: data.total,
        annotatedParagraphs: data.annotated.size,
        densityPercent: data.total > 0 ? Math.round((data.annotated.size / data.total) * 10000) / 100 : 0,
      }
    }).sort((a, b) => a.chapterNumber - b.chapterNumber)

    // Compute annotation counts per chapter
    for (const item of chapterDensity) {
      let count = 0
      const prefix = `${item.chapterNumber}:`
      for (const [key, anns] of Object.entries(userAnnotations.value)) {
        if (key.startsWith(prefix)) count += anns.length
      }
      if (includeAgent) {
        for (const [key, anns] of Object.entries(agentAnnotations.value)) {
          if (key.startsWith(prefix)) count += anns.length
        }
      }
      item.annotationCount = count
    }

    const userVsAgentByCategory = Object.entries(userVsAgentMap).map(([cat, data]) => ({
      category: cat,
      categoryLabel: CATEGORY_LABELS[cat] || cat,
      ...data,
    }))

    // Sort timeline by timestamp
    timeline.sort((a, b) => b.timestamp - a.timestamp)
    const recentTimeline = timeline.slice(0, 50)

    return {
      total,
      userCount,
      agentCount,
      withMarginalia,
      paragraphsCovered,
      totalParagraphs,
      coverageRate: totalParagraphs > 0 ? Math.round((paragraphsCovered / totalParagraphs) * 10000) / 100 : 0,
      categoryDistribution,
      dynastyDistribution,
      chapterDensity,
      userVsAgentByCategory,
      timeline: recentTimeline,
    }
  }

  // C05: Export stats as CSV
  function exportStatsCSV(includeAgent: boolean): string {
    const stats = computeAnnotationStats(includeAgent)
    const rows: string[] = ['指标,值']

    rows.push(`总标注数,${stats.total}`)
    rows.push(`用户标注,${stats.userCount}`)
    rows.push(`Agent标注,${stats.agentCount}`)
    rows.push(`含批注,${stats.withMarginalia}`)
    rows.push(`覆盖段落,${stats.paragraphsCovered}`)
    rows.push(`总段落,${stats.totalParagraphs}`)
    rows.push(`覆盖率(%),${stats.coverageRate}`)
    rows.push('')

    rows.push('类别,总数,用户,Agent,占比(%)')
    for (const cat of stats.categoryDistribution) {
      rows.push(`${cat.categoryLabel},${cat.count},${cat.userCount},${cat.agentCount},${cat.percentage}`)
    }
    rows.push('')

    rows.push('朝代,标注数,占比(%)')
    for (const dyn of stats.dynastyDistribution) {
      rows.push(`${dyn.dynasty},${dyn.count},${dyn.percentage}`)
    }
    rows.push('')

    rows.push('章节,朝代,标注数,总段落,已标注段落,密度(%)')
    for (const ch of stats.chapterDensity) {
      rows.push(`${ch.chapterTitle},${ch.dynasty},${ch.annotationCount},${ch.totalParagraphs},${ch.annotatedParagraphs},${ch.densityPercent}`)
    }

    return rows.join('\n')
  }

  // Helper: get chapters from sourceTreeStore
  function chapters() {
    try {
      const store = useSourceTreeStore()
      return store.chapters || []
    } catch {
      return []
    }
  }

  // ===== D4-1: Server sync (SQLite — single source of truth) =====

  async function initFromServer(): Promise<void> {
    const statsResp = await apiFetch('/api/annotations/stats')
    if (!statsResp.ok) {
      throw new Error(`Failed to load annotation stats from server: HTTP ${statsResp.status}`)
    }
    const stats = await statsResp.json() as { totalAnnotations: number; byChapter: Record<string, number> }

    const chapterNumbers = Object.keys(stats.byChapter).map(Number).filter(n => stats.byChapter[String(n)] > 0)

    for (const chapterNumber of chapterNumbers) {
      const resp = await apiFetch('/api/annotations/' + chapterNumber)
      if (!resp.ok) {
        throw new Error(`Failed to load annotations for chapter ${chapterNumber}: HTTP ${resp.status}`)
      }
      const data = await resp.json() as {
        annotations: Record<string, Array<Record<string, unknown>>>
        marginalia: Record<string, Array<Record<string, unknown>>>
        visibility: AnnotationVisibility | null
      }

      if (data.annotations) {
        for (const [key, anns] of Object.entries(data.annotations)) {
          if (anns.length > 0) {
            const userAnns: UserAnnotation[] = []
            const agentAnns: Annotation[] = []
            for (const a of anns) {
              const source = (a.source as string) || 'agent'
              if (source === 'agent') {
                agentAnns.push({
                  id: a.id as string,
                  category: a.category as AnnotationCategory,
                  label: a.label as string,
                  span: a.span as AnnotationSpan,
                  explanation: a.explanation as string || '',
                  confidence: (a.confidence as unknown as number) ?? 0.5,
                  source: 'agent' as const,
                  createdAt: a.createdAt as number,
                  updatedAt: (a.updatedAt as number) || (a.createdAt as number) || Date.now(),
                  color: a.color as string,
                })
              } else {
                userAnns.push({
                  id: a.id as string,
                  category: a.category as AnnotationCategory,
                  label: a.label as string,
                  span: a.span as AnnotationSpan,
                  note: a.note as string,
                  createdAt: a.createdAt as number,
                  color: a.color as string,
                })
              }
            }
            if (userAnns.length > 0) userAnnotations.value[key] = userAnns
            if (agentAnns.length > 0) agentAnnotations.value[key] = agentAnns
          }
        }
      }

      if (data.marginalia) {
        for (const [key, margs] of Object.entries(data.marginalia)) {
          if (margs.length > 0) {
            const userMarg: Marginalia[] = []
            const agentMarg: Marginalia[] = []
            for (const m of margs) {
              const msource = (m.source as string) || 'user'
              const marg = {
                id: m.id as string,
                annotationId: m.annotationId as string,
                chapterNumber: m.chapterNumber as number,
                paragraphIndex: m.paragraphIndex as number,
                anchorCharOffset: m.anchorCharOffset as number,
                content: m.content as string,
                source: msource as 'user' | 'agent',
                createdAt: m.createdAt as number,
                updatedAt: m.updatedAt as number,
              }
              if (msource === 'agent') {
                agentMarg.push(marg)
              } else {
                userMarg.push(marg)
              }
            }
            if (userMarg.length > 0) userMarginalia.value[key] = userMarg
            if (agentMarg.length > 0) agentMarginalia.value[key] = agentMarg
          }
        }
      }

      if (data.visibility) {
        visibility.value = data.visibility
      }
    }

    // If no chapters have data yet, still load visibility from dedicated endpoint
    if (chapterNumbers.length === 0) {
      const visResp = await apiFetch('/api/annotations/visibility')
      if (!visResp.ok) {
        throw new Error(`Failed to load visibility: HTTP ${visResp.status}`)
      }
      const visData = await visResp.json() as { visibility: AnnotationVisibility | null }
      if (visData.visibility) {
        visibility.value = visData.visibility
      }
    }
  }

  async function saveToServer(chapterNumber: number): Promise<void> {
    const prefix = `${chapterNumber}:`
    const annotationsForChapter: Array<Record<string, unknown>> = []
    const marginaliaForChapter: Array<Record<string, unknown>> = []

    for (const [key, anns] of Object.entries(userAnnotations.value)) {
      if (key.startsWith(prefix)) {
        for (const ann of anns) {
          annotationsForChapter.push({
            id: ann.id,
            label: ann.label,
            category: ann.category,
            span: ann.span,
            color: ann.color,
            note: ann.note,
            createdAt: ann.createdAt,
            updatedAt: ann.createdAt,
            passageKey: key,
            paragraphIndex: parseInt(key.split(':')[1], 10) || 0,
            source: 'user',
          })
        }
      }
    }

    for (const [key, margs] of Object.entries(userMarginalia.value)) {
      if (key.startsWith(prefix)) {
        for (const m of margs) {
          marginaliaForChapter.push({
            id: m.id,
            annotationId: m.annotationId,
            chapterNumber: m.chapterNumber,
            paragraphIndex: m.paragraphIndex,
            anchorCharOffset: m.anchorCharOffset,
            content: m.content,
            source: m.source,
            createdAt: m.createdAt,
            updatedAt: m.updatedAt,
          })
        }
      }
    }

    const resp = await apiFetch('/api/annotations/' + chapterNumber, {
      method: 'POST',
      body: JSON.stringify({
        annotations: annotationsForChapter,
        marginalia: marginaliaForChapter,
        visibility: visibility.value,
      }),
    })
    if (!resp.ok) {
      throw new Error(`Failed to save annotations for chapter ${chapterNumber}: HTTP ${resp.status}`)
    }
  }

  // 防抖 timer：合并逐段推送触发的多次保存请求为最后一次，避免并发保存风暴
  let _saveAgentTimer: ReturnType<typeof setTimeout> | null = null

  async function saveAgentToServer(chapterNumber: number): Promise<void> {
    // 防抖 500ms：新路径 Pass 2 逐段推送时每段都触发一次调用，但每次都是全章
    // 全量保存（遍历所有该章段落打包），合并为最后一次即可包含全部已加载段落，
    // 避免 N 次并发请求与后端 supervisor 的保存重复竞争。
    if (_saveAgentTimer) {
      clearTimeout(_saveAgentTimer)
    }
    return new Promise<void>((resolve) => {
      _saveAgentTimer = setTimeout(async () => {
        _saveAgentTimer = null
        try {
          await _doSaveAgentToServer(chapterNumber)
        } catch (e) {
          console.error('[annotations] debounced saveAgentToServer failed:', e)
        } finally {
          resolve()
        }
      }, 500)
    })
  }

  async function _doSaveAgentToServer(chapterNumber: number): Promise<void> {
    const prefix = `${chapterNumber}:`
    const annotations: Array<Record<string, unknown>> = []
    const marginalia: Array<Record<string, unknown>> = []

    for (const [key, anns] of Object.entries(agentAnnotations.value)) {
      if (key.startsWith(prefix)) {
        for (const ann of anns) {
          annotations.push({
            id: ann.id,
            chapter_number: chapterNumber,
            paragraph_index: parseInt(key.split(':')[1], 10) || 0,
            category: ann.category,
            entity: ann.label,
            start_char: ann.span.startChar,
            end_char: ann.span.endChar,
            quote: ann.label,
            explanation: ann.explanation,
            confidence: ann.confidence,
            color: ann.color,
            source: 'agent',
            created_at: new Date(ann.createdAt).toISOString(),
            updated_at: new Date(ann.updatedAt || ann.createdAt).toISOString(),
          })
        }
      }
    }

    for (const [key, margs] of Object.entries(agentMarginalia.value)) {
      if (key.startsWith(prefix)) {
        for (const m of margs) {
          if (!m.content || !m.content.trim()) continue
          marginalia.push({
            id: m.id,
            annotation_id: m.annotationId,
            chapter_number: m.chapterNumber,
            paragraph_index: m.paragraphIndex,
            anchor_char_offset: m.anchorCharOffset,
            content: m.content,
            source: 'agent',
            created_at: new Date(m.createdAt).toISOString(),
            updated_at: new Date(m.updatedAt).toISOString(),
          })
        }
      }
    }

    if (annotations.length === 0 && marginalia.length === 0) return

    const resp = await apiFetch('/api/annotations/save', {
      method: 'POST',
      body: JSON.stringify({ chapterNumber, annotations, marginalia }),
    })
    if (!resp.ok) {
      console.error(`[annotations] Failed to save agent annotations for chapter ${chapterNumber}: HTTP ${resp.status}`)
    }
  }

  async function reloadFromServer(chapterNumbers: number[]): Promise<void> {
    for (const chapterNumber of chapterNumbers) {
      const resp = await apiFetch('/api/annotations/' + chapterNumber)
      if (!resp.ok) {
        throw new Error(`Failed to reload annotations for chapter ${chapterNumber}: HTTP ${resp.status}`)
      }
      const data = await resp.json() as {
        annotations: Record<string, Array<Record<string, unknown>>>
        marginalia: Record<string, Array<Record<string, unknown>>>
        visibility: AnnotationVisibility | null
      }
      if (data.annotations) {
        for (const [key, anns] of Object.entries(data.annotations)) {
          if (anns.length > 0) {
            const userAnns: UserAnnotation[] = []
            const agentAnns: Annotation[] = []
            for (const a of anns) {
              const source = (a.source as string) || 'agent'
              if (source === 'agent') {
                agentAnns.push({
                  id: a.id as string,
                  category: a.category as AnnotationCategory,
                  label: a.label as string,
                  span: a.span as AnnotationSpan,
                  explanation: a.explanation as string || '',
                  confidence: (a.confidence as unknown as number) ?? 0.5,
                  source: 'agent' as const,
                  createdAt: a.createdAt as number,
                  updatedAt: (a.updatedAt as number) || (a.createdAt as number) || Date.now(),
                  color: a.color as string,
                })
              } else {
                userAnns.push({
                  id: a.id as string,
                  category: a.category as AnnotationCategory,
                  label: a.label as string,
                  span: a.span as AnnotationSpan,
                  note: a.note as string,
                  createdAt: a.createdAt as number,
                  color: a.color as string,
                })
              }
            }
            if (userAnns.length > 0) userAnnotations.value[key] = userAnns
            if (agentAnns.length > 0) agentAnnotations.value[key] = agentAnns
          }
        }
      }
      if (data.marginalia) {
        for (const [key, margs] of Object.entries(data.marginalia)) {
          if (margs.length > 0) {
            const userMarg: Marginalia[] = []
            const agentMarg: Marginalia[] = []
            for (const m of margs) {
              const source = (m.source as string) || 'agent'
              if (source === 'agent') {
                agentMarg.push(m as unknown as Marginalia)
              } else {
                userMarg.push(m as unknown as Marginalia)
              }
            }
            if (userMarg.length > 0) userMarginalia.value[key] = userMarg
            if (agentMarg.length > 0) agentMarginalia.value[key] = agentMarg
          }
        }
      }
      if (data.visibility) {
        visibility.value = data.visibility
      }
    }
  }

  return {
    // State
    agentAnnotations,
    aiAnnotations: agentAnnotations, // backward-compat alias
    aiInsights,
    userAnnotations,
    userMarginalia,
    agentMarginalia,
    suggestedAnnotations,
    visibility,
    agentConversationId,
    editingAnnotationId,
    editingMarginaliaId,
    chapterCoverage,
    updateCoverage,
    coverageForChapter,

    // Getters
    visibleAnnotations,
    annotationsForKey,
    userAnnotationsForKey,
    insightsForKey,
    marginaliaForKey,
    userAnnotationCount,
    agentAnnotationCount,
    hasUserAnnotations,
    getAnnotationsByChapter,
    exportAnnotationsHTML,
    exportAnnotationsPlainText,
    exportAnnotationsCSV,
    exportAnnotationsJSON,
    exportAnnotationsTEI,
    exportAgentOnlyCSV,
    exportAgentOnlyJSON,
    exportAgentOnlyHTML,
    exportAgentOnlyPlainText,
    exportAgentOnlyTEI,
    buildAnnotationIndex,
    computeAnnotationStats,
    exportStatsCSV,

    // Agent annotation actions
    loadAgentAnnotations,
    loadAiAnnotations,

    // User annotation actions
    addUserAnnotation,
    updateUserAnnotation,
    removeUserAnnotation,
    removeAgentAnnotation,
    editUserAnnotation,

    // Marginalia actions
    addMarginalia,
    updateMarginalia,
    removeMarginalia,
    loadAgentMarginalia,
    clearAgentMarginaliaForChapter,

    // Suggested annotations
    loadSuggestedAnnotations,
    acceptSuggestion,
    acceptAllSuggestions,
    dismissSuggestion,
    dismissAllSuggestions,

    // Adopt agent annotations
    adoptAgentAnnotation,
    adoptAllAgentAnnotations,
    saveAndAdoptAllAgentAnnotations,

    // Visibility
    toggleVisibility,
    showAll,
    hideAll,
    showUserOnly,
    showAgentOnly,

    // Lifecycle
    clearAgentAnnotations,
    clearAgentAnnotationsForChapter,
    clearUserAnnotationsForKey,
    // D4-1: Server sync (SQLite — single source of truth)
    initFromServer,
    saveToServer,
    saveAgentToServer,
    reloadFromServer,
  }
})
