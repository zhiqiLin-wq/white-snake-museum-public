import { useAuthStore } from '@/stores/auth'

export interface ParsedSSEEvent {
  type: string
  data: Record<string, unknown>
  raw: string
}

export type EventHandler = (event: ParsedSSEEvent) => void

export class SSEEventBus {
  private static instance: SSEEventBus | null = null
  private handlers = new Map<string, EventHandler[]>()
  private eventBuffer: ParsedSSEEvent[] = []
  private isInterrupted = false
  private reconnectCount = 0
  private maxReconnectCount = 3
  private abortController: AbortController | null = null
  private lastEventTime = 0
  private heartbeatTimer: ReturnType<typeof setInterval> | null = null
  private heartbeatTimedOut = false
  private timeoutMs = 600000  // 600 seconds (10 min, for slow LLM calls in evolution)

  static getInstance(): SSEEventBus {
    if (!SSEEventBus.instance) {
      SSEEventBus.instance = new SSEEventBus()
    }
    return SSEEventBus.instance
  }

  get isReconnecting(): boolean {
    return this.reconnectCount > 0 && this.reconnectCount <= this.maxReconnectCount
  }

  private startHeartbeat() {
    this.lastEventTime = Date.now()
    this.stopHeartbeat()
    this.heartbeatTimer = setInterval(() => {
      const elapsed = Date.now() - this.lastEventTime
      if (elapsed > this.timeoutMs) {
        console.warn(`[SSEEventBus] SSE connection timeout after ${Math.round(elapsed / 1000)}s, reconnecting...`)
        this.stopHeartbeat()
        this.heartbeatTimedOut = true
        this.abortController?.abort()
      }
    }, 30000) as unknown as ReturnType<typeof setInterval>
  }

  private stopHeartbeat() {
    if (this.heartbeatTimer) {
      clearInterval(this.heartbeatTimer)
      this.heartbeatTimer = null
    }
  }

  on(eventType: string, handler: EventHandler): void {
    if (!this.handlers.has(eventType)) {
      this.handlers.set(eventType, [])
    }
    this.handlers.get(eventType)!.push(handler)
  }

  off(eventType: string, handler: EventHandler): void {
    const handlers = this.handlers.get(eventType)
    if (!handlers) return
    const idx = handlers.indexOf(handler)
    if (idx !== -1) handlers.splice(idx, 1)
    if (handlers.length === 0) this.handlers.delete(eventType)
  }

  dispatch(rawLine: string, sseEventType?: string): void {
    let eventType = sseEventType || 'message'
    let dataStr = ''

    if (rawLine.startsWith('event: ')) {
      return // Handled by caller
    }

    if (rawLine.startsWith('data: ')) {
      dataStr = rawLine.slice(6)
    } else {
      return
    }

    let data: Record<string, unknown>
    try {
      data = JSON.parse(dataStr)
    } catch {
      data = { text: dataStr }
    }

    // Prefer type from JSON data, fall back to SSE event: line
    const resolvedType = (data.type as string) || eventType

    const event: ParsedSSEEvent = {
      type: resolvedType,
      data,
      raw: rawLine,
    }

    if (this.isInterrupted && event.type !== 'interrupt_resumed') {
      this.eventBuffer.push(event)
      return
    }

    this.routeEvent(event)
  }

  private routeEvent(event: ParsedSSEEvent): void {
    this.lastEventTime = Date.now()
    const handlers = this.handlers.get(event.type)
    if (handlers) {
      for (const handler of handlers) {
        try {
          handler(event)
        } catch (err) {
          console.error(`[SSEEventBus] Handler error for ${event.type}:`, err)
        }
      }
    }

    // Also fire wildcard handlers
    const wildcardHandlers = this.handlers.get('*')
    if (wildcardHandlers) {
      for (const handler of wildcardHandlers) {
        try {
          handler(event)
        } catch (err) {
          console.error('[SSEEventBus] Wildcard handler error:', err)
        }
      }
    }
  }

  enterInterrupt(): void {
    this.isInterrupted = true
    this.eventBuffer = []
  }

  exitInterrupt(): void {
    this.isInterrupted = false
    this.flushBuffer()
  }

  flushBuffer(): void {
    const events = [...this.eventBuffer]
    this.eventBuffer = []
    for (const event of events) {
      this.routeEvent(event)
    }
  }

  async startConnection(url: string, body: Record<string, unknown>): Promise<void> {
    this.abortController = new AbortController()
    //  SSE  interrupt 
    this.isInterrupted = false
    this.eventBuffer = []
    this.startHeartbeat()

    try {
      const authStore = useAuthStore()
      const headers: Record<string, string> = { 'Content-Type': 'application/json' }
      if (authStore.token) {
        headers['Authorization'] = `Bearer ${authStore.token}`
      }
      const response = await fetch(url, {
        method: 'POST',
        headers,
        body: JSON.stringify(body),
        signal: this.abortController.signal,
      })

      if (!response.ok) {
        const err = new Error(`HTTP ${response.status}`)
        ;(err as Error & { statusCode: number }).statusCode = response.status
        throw err
      }

      const reader = response.body?.getReader()
      if (!reader) throw new Error(' SSE ')

      const decoder = new TextDecoder()
      let buffer = ''
      let currentEvent = ''

      while (true) {
        const { done, value } = await reader.read()
        if (done) break

        buffer += decoder.decode(value, { stream: true })
        const lines = buffer.split('\n')
        buffer = lines.pop() || ''

        for (const line of lines) {
          const cleanLine = line.replace(/\r$/, '')
          if (cleanLine.startsWith('event: ')) {
            currentEvent = cleanLine.slice(7).trim()
          } else if (cleanLine.startsWith('data: ')) {
            this.dispatch(cleanLine, currentEvent)
            currentEvent = ''
          }
        }
      }

      this.reconnectCount = 0
    } catch (err: unknown) {
      if (err instanceof Error && err.name === 'AbortError') {
        if (this.heartbeatTimedOut) {
          this.heartbeatTimedOut = false
          // 心跳超时断流：派发本地 error 事件让 store 结束流式状态并提示用户，而非静默中断
          this.routeEvent({
            type: 'error',
            data: { message: '连接超时：服务器长时间无响应，请重试' },
            raw: '',
          })
        }
        return
      }
      // O-06: Do not retry on 4xx errors (client errors like 404, 410)
      const statusCode = (err as Error & { statusCode?: number }).statusCode
      if (statusCode && statusCode >= 400 && statusCode < 500) {
        throw err
      }
      if (this.reconnectCount < this.maxReconnectCount) {
        this.reconnectCount++
        const delay = Math.pow(2, this.reconnectCount - 1) * 1000
        await new Promise(resolve => setTimeout(resolve, delay))
        await this.startConnection(url, body)
      } else {
        throw err
      }
    } finally {
      // 流结束（正常完成或断开）后清理看门狗，防止旧流的超时 abort 干扰后续会话
      this.stopHeartbeat()
      this.heartbeatTimedOut = false
    }
  }

  async reconnect(url: string, body: Record<string, unknown>): Promise<void> {
    this.reconnectCount = 0
    await this.startConnection(url, body)
  }

  abort(): void {
    this.stopHeartbeat()
    if (this.abortController) {
      this.abortController.abort()
      this.abortController = null
    }
  }

  clearAllHandlers(): void {
    this.handlers.clear()
  }
}

export interface StoreRefs {
  agentStore: ReturnType<typeof import('@/stores/agent').useAgentStore>
  evolutionStore: ReturnType<typeof import('@/stores/evolution').useEvolutionStore>
  annotationStore: ReturnType<typeof import('@/stores/annotations').useAnnotationStore>
  sourceTreeStore: ReturnType<typeof import('@/stores/sourceTree').useSourceTreeStore>
  workspaceStore: ReturnType<typeof import('@/stores/workspace').useWorkspaceStore>
  mapStore: ReturnType<typeof import('@/stores/map').useMapStore>
  dynastyCompareStore: ReturnType<typeof import('@/stores/dynastyCompare').useDynastyCompareStore>
}

/*
 * Global SSE event routing table.
 * Maps every SSE event type to the corresponding Pinia Store handler.
 * Called once in MuseumView.vue onMounted.
 * Stores are passed as parameters to avoid circular imports.
 */
export function registerGlobalRoutes(stores: StoreRefs): void {
  const bus = SSEEventBus.getInstance()
  const { agentStore, evolutionStore, annotationStore, sourceTreeStore, workspaceStore, mapStore, dynastyCompareStore } = stores

  // --- Lifecycle events ---
  bus.on('thinking_start', () => {
    agentStore.startStreaming()
    // /
    evolutionStore.resetLiveState()
  })

  bus.on('done', () => {
    agentStore.finishStreaming()
  })

  bus.on('error', (event) => {
    const msg = typeof event.data === 'string' ? event.data : (event.data as Record<string, unknown>).message as string || ''
    agentStore.setError(msg)
  })

  // --- Progress events ---
  bus.on('status', (event) => {
    const msg = typeof event.data === 'string' ? event.data : (event.data as Record<string, unknown>).message as string || ''
    agentStore.addStatusStep(msg)
  })

  bus.on('intent', (event) => {
    const data = event.data as Record<string, unknown>
    agentStore.setIntent((data.intent || data.intentLabel) as string || '')
    // v12: 路由理由透出 — 轻量系统提示行（步骤条灰字），让用户知道为何走这条路径
    const routeReason = (data.routeReason || data.route_reason) as string | undefined
    if (routeReason) {
      agentStore.addStatusStep(`路径选择：${routeReason}`)
    }
  })

  bus.on('thinking_plan', (event) => {
    const data = event.data as Record<string, unknown>
    agentStore.setThinkingPlan({
      totalSteps: data.totalSteps as number || 0,
      steps: (data.steps || []) as { name: string; label: string }[],
    })
  })

  // Step name -> Evolution phase mapping for step-progress-driven phase tracking
  const STEP_TO_PHASE: Record<string, string> = {
    'decompose_motifs': 'E1_decompose',
    'scan_motif_matrix': 'E2_scan',
    'classify_changes': 'E3_classify',
    'infer_phylogeny': 'E4_phylogeny',
    'correlate_external': 'E5_correlate',
    'formulate_hypothesis': 'E6_hypothesize',
    'test_claims': 'E7_test',
    'synthesize': 'E8_synthesize',
  }

  bus.on('thinking_step', (event) => {
    const data = event.data as Record<string, unknown>
    const stepName = (data.step as string) || ''
    const stepMessage = (data.message as string) || stepName
    agentStore.addThinkingStep(stepName, stepMessage)
    // Drive evolution phase from THINKING_STEP events
    if (stepName && STEP_TO_PHASE[stepName]) {
      const phase = STEP_TO_PHASE[stepName]
      // Do not overwrite interrupt phases (E1_interrupt, E6_interrupt, E7_interrupt)
      if (!evolutionStore.isAwaitingUserInput) {
        evolutionStore.setPhase(phase as Parameters<typeof evolutionStore.setPhase>[0])
      }
    }
  })

  bus.on('workflow_done', (event) => {
    const data = event.data
    if ((data.workflow as string) === 'evolution_workbench') {
      //  completeinterrupt  done  SSE 
      if (!evolutionStore.isAwaitingUserInput) {
        evolutionStore.setPhase('complete')
      }
    }
    agentStore.setWorkflowCardFull({
      summary: (data.summary || '') as string,
      resultRefs: data.resultRefs as { label: string; type: string; data: Record<string, unknown> }[] | undefined,
    })
  })

  // B-140:  -> 
  bus.on('evolution_chat_card', (event) => {
    const data = event.data as Record<string, unknown>
    agentStore.addEvolutionCard({
      trackingTarget: (data.trackingTarget || '') as string,
      summary: (data.summary || '') as string,
      narrative: (data.narrative || '') as string,
      threadId: (data.threadId || '') as string,
      workflow: (data.workflow || 'evolution') as string,
    })
  })

  // v19: 长报告就绪 → 挂到当前 assistant 消息（可下载完整 Markdown）
  bus.on('report_ready', (event) => {
    const data = event.data as Record<string, unknown>
    agentStore.setReportReady({
      reportId: (data.reportId || '') as string,
      title: (data.title || '研究报告') as string,
      topic: (data.topic || '') as string,
      totalChars: (data.totalChars || 0) as number,
      sectionCount: (data.sectionCount || 0) as number,
      sections: (data.sections || []) as string[],
      createdAt: data.createdAt as string | undefined,
      content: (data.content || '') as string,
    })
  })

  // --- Content events ---
  bus.on('chunk', (event) => {
    agentStore.appendChunk(event.data.text as string || '')
  })

  // v17: ReAct 决策轮思考增量 → 气泡内折叠"思考中..."区域
  bus.on('thinking_delta', (event) => {
    agentStore.appendReasoning((event.data.delta as string) || '')
  })

  bus.on('source', (event) => {
    const raw = (event.data || {}) as Record<string, unknown>
    // SSE 事件字段是 snake_case (Python) → 前端 SourceCitation 用 camelCase
    const src = {
      chunkId: String(raw.chunk_id ?? raw.chunkId ?? ''),
      chapterNumber: raw.chapter_number ?? raw.chapterNumber,
      chapterTitle: String(raw.chapter_title ?? raw.chapterTitle ?? ''),
      excerpt: String(raw.excerpt ?? ''),
      relevanceScore: Number(raw.relevance_score ?? raw.relevanceScore ?? 0),
    } as unknown as Parameters<typeof agentStore.addSource>[0]
    agentStore.addSource(src)
  })

  // --- Tool events ---
  bus.on('tool_call', (event) => {
    agentStore.addToolCall(event.data as unknown as Parameters<typeof agentStore.addToolCall>[0])
  })

  bus.on('tool_result', (event) => {
    agentStore.updateToolResult(
      event.data.tool as string || '',
      event.data.result_preview as string || ''
    )
  })

  // CU-9.3: Step retry/skip from backend
  bus.on('retry_step', (event) => {
    const tool = (event.data as Record<string, unknown>).tool as string || ''
    const step = agentStore.statusSteps.find(s => s.tool === tool)
    if (step) {
      step.status = 'running'
      step.done = false
      step.errorMsg = undefined
    }
  })

  bus.on('skip_step', (event) => {
    const tool = (event.data as Record<string, unknown>).tool as string || ''
    const step = agentStore.statusSteps.find(s => s.tool === tool)
    if (step) {
      step.status = 'skipped'
      step.done = true
      step.errorMsg = undefined
    }
  })

  // --- Workspace driver events ---
  bus.on('annotation_data', (event) => {
    const data = event.data as Record<string, unknown>
    const splitSide = (data.splitSide) as 'left' | 'right' | undefined

    // 阶段 2 — P2-04: 格式检测 — 新格式有顶层的 paragraphIndex (number) 和 annotations 数组
    const isNewFormat = typeof data.paragraphIndex === 'number' && Array.isArray(data.annotations)

    if (isNewFormat) {
      // === 新格式（逐段 annotation_data） ===
      const passageKey = data.passageKey as string
      if (passageKey) {
        annotationStore.loadAiAnnotations(
          (data.annotations || []) as Parameters<typeof annotationStore.loadAiAnnotations>[0],
          passageKey,
          (data.insights || []) as Parameters<typeof annotationStore.loadAiAnnotations>[2]
        )
      }
      const coverage = data.coverage as import('@/types/annotation').CoverageInfo | undefined
      const chapterNumber = data.chapterNumber as number
      if (coverage && chapterNumber) {
        annotationStore.updateCoverage(chapterNumber, coverage)
      }
      if (data.isComplete === true) {
        agentStore.setAnnotationLoading(false)
      }
      // Note: isComplete 事件不打开新 tab，由调用方管理
      if (!data.isComplete && passageKey) {
        // 第一个非完成段到达 → 标注已开始，打开 loading 指示器
        // （isComplete:true 时上面已关闭；防 false→false 无意义调用，
        //  仅在确实有逐段流式推送时才亮起）
        agentStore.setAnnotationLoading(true)
        // Open or activate text-reader tab at the first annotated paragraph
        const chapterTitle = (data.chapterTitle || '') as string
        workspaceStore.openTab('text-reader', chapterTitle, {
          chapterNumber,
          paragraphIndex: data.paragraphIndex as number,
          annotationData: data,
          splitSide,
        })
      }
    } else {
      // === 旧格式（passageAnnotations 数组） ===
      const passageAnnotations = (data.passageAnnotations || []) as Array<{
        passageKey: string
        paragraphIndex: number
        annotations: Parameters<typeof annotationStore.loadAiAnnotations>[0]
        insights: Parameters<typeof annotationStore.loadAiAnnotations>[2]
      }>

      console.log('[SSE] annotation_data received:', passageAnnotations.length, 'passages, total annotations:', passageAnnotations.reduce((s, g) => s + (g.annotations?.length || 0), 0))

      const deleteAll = data.deleteAll === true
      const deleteCategories = (data.deleteCategories || data.categories) as string[] | undefined

      for (const group of passageAnnotations) {
        const pKey = group.passageKey || ''
        if (!pKey) continue

        annotationStore.loadAiAnnotations(
          group.annotations || [],
          pKey,
          group.insights || undefined
        )

        if (deleteAll) {
          if (deleteCategories && deleteCategories.length > 0) {
            const userAnns = annotationStore.userAnnotationsForKey(pKey)
            if (userAnns.length > 0) {
              const kept = userAnns.filter(a => !deleteCategories.includes(a.category))
              if (kept.length === 0) {
                annotationStore.clearUserAnnotationsForKey(pKey)
              } else {
                annotationStore.clearUserAnnotationsForKey(pKey)
                for (const ann of kept) {
                  annotationStore.addUserAnnotation(pKey, {
                    category: ann.category,
                    span: ann.span,
                    label: ann.label,
                    note: ann.note,
                    color: ann.color,
                  })
                }
              }
            }
          } else {
            annotationStore.clearUserAnnotationsForKey(pKey)
          }
        }
      }

      // Open or activate text-reader tab at the first annotated paragraph
      const chapterTitle = (data.chapterTitle || '') as string
      const chapterNumber = data.chapterNumber as number | undefined
      const firstParaIdx = passageAnnotations.length > 0
        ? (passageAnnotations[0].paragraphIndex as number)
        : (data.paragraphRange as [number, number] | undefined)?.[0]

      workspaceStore.openTab('text-reader', chapterTitle, {
        chapterNumber,
        paragraphIndex: firstParaIdx,
        annotationData: data,
        splitSide,
      })
    }  // end else (旧格式)
  })

  bus.on('marginalia_data', (event) => {
    const data = event.data as Record<string, unknown>
    const marginaliaList = (data.marginalia || []) as Array<{
      chapterNumber: number
      paragraphIndex: number
      content: string
      id?: string
      color?: string
      anchorCharOffset?: number
    }>

    if (!marginaliaList || marginaliaList.length === 0) return

    // Track which paragraphs were affected so we only clear those
    const affectedParagraphs = new Set<string>()
    for (const m of marginaliaList) {
      if (!m.chapterNumber || m.paragraphIndex === undefined || !m.content) continue
      const passageKey = `${m.chapterNumber}:${m.paragraphIndex}`
      affectedParagraphs.add(passageKey)
      // Clear existing agent marginalia for this specific paragraph before adding new ones
      if (annotationStore.agentMarginalia[passageKey]) {
        annotationStore.agentMarginalia[passageKey] = []
      }
      annotationStore.loadAgentMarginalia(passageKey, [{
        id: m.id || `agent_marg_${Date.now()}_${Math.random().toString(36).slice(2, 8)}`,
        chapterNumber: m.chapterNumber,
        paragraphIndex: m.paragraphIndex,
        anchorCharOffset: m.anchorCharOffset || 0,
        content: m.content,
        color: m.color || '#F5A623',
        source: 'agent' as const,
        createdAt: Date.now(),
        updatedAt: Date.now(),
      }])
    }
  })

  bus.on('annotation_suggest', (event) => {
    const data = event.data as Record<string, unknown>
    const passageAnnotations = (data.passageAnnotations || []) as Array<{
      passageKey: string
      paragraphIndex: number
      suggestions: Parameters<typeof annotationStore.loadSuggestedAnnotations>[1]
    }>
    for (const group of passageAnnotations) {
      annotationStore.loadSuggestedAnnotations(group.passageKey, group.suggestions || [])
    }
  })

  bus.on('paragraph_alignment', (event) => {
    const data = event.data
    // Store alignment data on both relevant tabs
    const leftChapter = data.leftChapter as number
    const rightChapter = data.rightChapter as number
    const alignments = data.alignments
    const relevantTab = workspaceStore.tabs.find(t => {
      const d = t.data as Record<string, unknown>
      return d.chapterNumber === leftChapter || d.chapterNumber === rightChapter
    })
    if (relevantTab) {
      workspaceStore.updateTabData(relevantTab.id, {
        paragraphAlignments: alignments as Record<string, unknown>,
        leftChapter,
        rightChapter,
      })
    }
  })

  bus.on('stats_data', (event) => {
    const data = event.data
    const title = (data.title || '') as string
    workspaceStore.openTab('data-panel', title, data as Record<string, unknown>)
  })

  bus.on('comparison_data', (event) => {
    const data = event.data
    const title = (data.title || '') as string
    workspaceStore.openTab('compare-view', title, data as Record<string, unknown>)
  })

  bus.on('search_results', (event) => {
    const data = event.data
    const query = (data.query || '') as string
    workspaceStore.openTab('search-results', `搜索结果: ${query}`, data as Record<string, unknown>)
  })

  bus.on('text_navigate', (event) => {
    const data = event.data
    const chNum = Number(data.chapterNumber)
    if (isNaN(chNum)) return
    workspaceStore.openTab('text-reader', data.chapterTitle as string || '', {
      chapterNumber: chNum,
      paragraphIndex: data.paragraphIndex as number | undefined,
    })
  })

  bus.on('map_highlight', (event) => {
    mapStore.addHighlight(event.data as unknown as Parameters<typeof mapStore.addHighlight>[0])
  })

  bus.on('map_layer_update', (event) => {
    const data = event.data as Record<string, unknown>
    const dynasty = data.dynasty as string | string[]
    const action = (data.action as string) || 'set'

    if (action === 'reset') {
      mapStore.resetLayers()
    } else if (Array.isArray(dynasty)) {
      if (dynasty.length === 0) {
        mapStore.resetLayers()
      } else {
        mapStore.setTimeLayer(dynasty[0])
        for (let i = 1; i < dynasty.length; i++) {
          mapStore.toggleTimeLayer(dynasty[i])
        }
      }
    } else if (action === 'toggle') {
      mapStore.toggleTimeLayer(dynasty as string)
    } else {
      mapStore.setTimeLayer(dynasty as string)
    }

    // Auto-activate or open map tab so the user sees the result immediately
    const existingMapTab = workspaceStore.tabs.find(t => t.type === 'map')
    if (existingMapTab) {
      workspaceStore.setActiveTab(existingMapTab.id)
    } else {
      workspaceStore.openTab('map', '地图')
    }
  })

  // --- Evolution events ---
  bus.on('motif_matrix', (event) => {
    const data = event.data as Record<string, unknown>
    console.log('[SSE] motif_matrix received:', JSON.stringify({ rows: data.rows, columns_count: (data.columns as any[])?.length, cells_count: (data.cells as any[])?.length }))
    evolutionStore.handleMotifMatrix(data as unknown as Parameters<typeof evolutionStore.handleMotifMatrix>[0])
    // Phase is now driven by THINKING_STEP events (scan_motif_matrix sets E2_scan)
  })

  bus.on('phylogeny_data', (event) => {
    evolutionStore.handlePhylogenyData(event.data as unknown as Parameters<typeof evolutionStore.handlePhylogenyData>[0])
    const data = event.data as Record<string, unknown>
    if (!data.isFinal) {
      evolutionStore.setPhase('E4_phylogeny')
    }
  })

  bus.on('evidence_data', (event) => {
    evolutionStore.handleEvidenceData(event.data as unknown as Parameters<typeof evolutionStore.handleEvidenceData>[0])
    const data = event.data as Record<string, unknown>
    if (!data.isFinal) {
      evolutionStore.setPhase('E7_test')
    }
  })

  bus.on('evolution_narrative', (event) => {
    const data = event.data as Record<string, unknown>
    evolutionStore.handleEvolutionNarrative(
      (data.narrative || '') as string,
      data.evidenceRefs as Record<string, import('@/types/evolution').EvidenceRefData> | undefined
    )
    evolutionStore.setPhase('E8_synthesize')
  })

  // --- Human-AI collaboration events ---
  bus.on('evolution_interrupt', (event) => {
    const data = event.data as Record<string, unknown>
    console.log('[SSE] evolution_interrupt received:', JSON.stringify(data, null, 2))
    console.log('[SSE] interruptId:', data.interruptId, 'interruptType:', data.interruptType, 'threadId:', data.threadId)
    evolutionStore.handleEvolutionInterrupt(data as unknown as Parameters<typeof evolutionStore.handleEvolutionInterrupt>[0])

    // J-03: Auto-open evolution-workspace tab when interrupt arrives
    const interruptType = data.interruptType as string || ''
    const trackingTarget = (data.data as Record<string, unknown>)?.trackingTarget as string || ''
    const evoThreadId = (data.threadId as string) || ''
    const tabLabelMap: Record<string, string> = {
      confirm_motifs: '确认母题分解',
      select_hypothesis: '选择演化假设',
      decide_next_action: '决定后续方向',
    }
    const tabLabel = tabLabelMap[interruptType] || '演化分析'
    workspaceStore.openTab('evolution-workspace', tabLabel, {
      trackingTarget,
      interruptType,
      threadId: evoThreadId,
    })
  })

  bus.on('interrupt_resumed', (event) => {
    const data = event.data
    const message = (data.message || '演化分析已继续...') as string
    agentStore.addStatusStep(message)
    agentStore.setError('')
  })

  // U-16: Export ready — generate file client-side from annotationStore and trigger download
  bus.on('export_ready', (event) => {
    const data = event.data as Record<string, unknown>
    const chapterNumbers = (data.chapterNumbers || []) as number[]
    const format = (data.format || 'csv') as string
    const source = (data.source || 'all') as string

    if (!chapterNumbers || chapterNumbers.length === 0) {
      console.warn('[SSE] export_ready: no chapterNumbers, skipping download')
      return
    }

    const includeAgent = source === 'agent' || source === 'all'
    const includeUser = source === 'user' || source === 'all'

    let content: string
    let mimeType: string
    let extension: string

    switch (format) {
      case 'json':
        content = annotationStore.exportAnnotationsJSON(chapterNumbers, includeAgent, includeUser, true, false)
        mimeType = 'application/json'
        extension = 'json'
        break
      case 'html':
        content = annotationStore.exportAnnotationsHTML(chapterNumbers[0], includeAgent, includeUser, true)
        mimeType = 'text/html'
        extension = 'html'
        break
      case 'csv':
      default:
        content = annotationStore.exportAnnotationsCSV(chapterNumbers, includeAgent, includeUser, true)
        mimeType = 'text/csv'
        extension = 'csv'
        break
    }

    const blob = new Blob(['' + content], { type: `${mimeType};charset=utf-8` })
    const url = URL.createObjectURL(blob)
    const chDesc = chapterNumbers.length === 1 ? `chapter${chapterNumbers[0]}` : `chapters${chapterNumbers.join('-')}`
    const fileName = `annotations-${chDesc}-${source}.${extension}`

    const a = document.createElement('a')
    a.href = url
    a.download = fileName
    document.body.appendChild(a)
    a.click()
    document.body.removeChild(a)
    URL.revokeObjectURL(url)
  })

  // A13: Dynasty compare SSE event
  bus.on('dynasty_compare', (event) => {
    const data = event.data as Record<string, unknown>
    dynastyCompareStore.loadCompareData({
      config: data.config as Parameters<typeof dynastyCompareStore.loadCompareData>[0]['config'],
      elements: (data.elements || []) as Parameters<typeof dynastyCompareStore.loadCompareData>[0]['elements'],
      locationChanges: (data.locationChanges || []) as Parameters<typeof dynastyCompareStore.loadCompareData>[0]['locationChanges'],
      summary: data.summary as Parameters<typeof dynastyCompareStore.loadCompareData>[0]['summary'],
    })
    // Only open a new tab if one doesn't already exist for this compare
    const cfg = data.config as Record<string, unknown>
    const left = cfg?.left as Record<string, unknown> | undefined
    const right = cfg?.right as Record<string, unknown> | undefined
    const leftNum = left?.chapterNumber
    const rightNum = right?.chapterNumber
    const existingTab = workspaceStore.tabs.find(t => {
      if (t.type !== 'dynasty-compare') return false
      const d = t.data as Record<string, unknown>
      const c = d?.config as Record<string, unknown> | undefined
      const l = c?.left as Record<string, unknown> | undefined
      const r = c?.right as Record<string, unknown> | undefined
      return l?.chapterNumber === leftNum && r?.chapterNumber === rightNum
    })
    if (!existingTab) {
      const lDynasty = (left?.dynasty as string)
        || sourceTreeStore.chapters.find(c => c.number === leftNum)?.dynasty
        || '?'
      const rDynasty = (right?.dynasty as string)
        || sourceTreeStore.chapters.find(c => c.number === rightNum)?.dynasty
        || '?'
      const tabTitle = `${lDynasty} vs ${rDynasty}`
      workspaceStore.openTab('dynasty-compare', tabTitle, { config: data.config as Record<string, unknown> })
    }
  })

  // A13: Location traceability SSE event
  bus.on('location_traceability', (event) => {
    const data = event.data as Record<string, unknown>
    const traceData = data.data as Record<string, {
      literaryRecords: {
        dynasty: string; dynastyOrder: number
        chapterNumber: number; chapterTitle: string
        paragraphIndex: number; excerpt: string
        span: { startChar: number; endChar: number }
        descriptionStyle: string
      }[]
      totalMentions: Record<string, number>
    }>
    if (traceData) {
      for (const [locationName, locData] of Object.entries(traceData)) {
        mapStore.cacheTraceabilityData(locationName, locData)
      }
    }
  })

  // A13: Mutual construction SSE event
  bus.on('mutual_construction', (event) => {
    const data = event.data as Record<string, unknown>
    mapStore.cacheMutualConstruction(
      data.locationName as string,
      {
        mutualConstruction: data.mutualConstruction as { landscapeToText: string; textToLandscape: string },
        descriptionStyles: data.descriptionStyles as Record<string, string>,
      }
    )
  })

  // v11: Agent  Split View
  bus.on('open_split_view', (event) => {
    const data = event.data as Record<string, unknown>
    const leftChNum = data.leftChapterNumber as number
    const rightChNum = data.rightChapterNumber as number

    if (!leftChNum || !rightChNum) return
    if (leftChNum === rightChNum) return

    const leftCh = sourceTreeStore.chapters.find(c => c.number === leftChNum)
    const rightCh = sourceTreeStore.chapters.find(c => c.number === rightChNum)

    const leftTitle = leftCh
      ? `${leftCh.dynasty} - ${leftCh.title}`
      : `${leftChNum}`
    const rightTitle = rightCh
      ? `${rightCh.dynasty} - ${rightCh.title}`
      : `${rightChNum}`

    const leftTab = workspaceStore.openTab('text-reader', leftTitle, {
      chapterNumber: leftChNum,
    })
    const rightTab = workspaceStore.openTab('text-reader', rightTitle, {
      chapterNumber: rightChNum,
    })

    workspaceStore.enterSplitMode(leftTab.id, rightTab.id)
  })
}
