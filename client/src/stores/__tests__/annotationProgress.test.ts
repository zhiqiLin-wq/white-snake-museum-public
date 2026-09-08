import { describe, it, expect, beforeEach } from 'vitest'
import { setActivePinia, createPinia } from 'pinia'
import { useAgentStore } from '@/stores/agent'

describe('agent store — 智能标注实时进度状态机', () => {
  beforeEach(() => {
    setActivePinia(createPinia())
    localStorage.clear()
  })

  it('beginAnnotationProgress 初始化：进入 prepare 阶段并记录章节与段落总数', () => {
    const store = useAgentStore()
    store.beginAnnotationProgress(1, '第一章 标题', 21)

    const p = store.annotationProgress
    expect(p.active).toBe(true)
    expect(p.status).toBe('running')
    expect(p.phase).toBe('prepare')
    expect(p.chapterNumber).toBe(1)
    expect(p.chapterTitle).toBe('第一章 标题')
    expect(p.totalParagraphs).toBe(21)
    expect(p.startedAt).not.toBeNull()
    expect(p.candidates).toBe(0)
    expect(p.entities).toBe(0)
    expect(p.paragraphsDone).toBe(0)
  })

  it('discovery 阶段更新：窗口进度与候选数累积，消息进入 recent 日志', () => {
    const store = useAgentStore()
    store.beginAnnotationProgress(1, 'ch1', 21)
    store.updateAnnotationProgress({
      phase: 'discovery',
      windowsDone: 2,
      windowsTotal: 7,
      candidates: 11,
      message: '扫描窗口 2/7，已发现 11 个候选实体',
    })

    const p = store.annotationProgress
    expect(p.phase).toBe('discovery')
    expect(p.windowsDone).toBe(2)
    expect(p.windowsTotal).toBe(7)
    expect(p.candidates).toBe(11)
    expect(p.recent.length).toBeGreaterThanOrEqual(1)
    expect(p.recent[p.recent.length - 1].message).toContain('2/7')
  })

  it('resolution 阶段更新：已定位实体数、完成段落数、分类计数单调累积', () => {
    const store = useAgentStore()
    store.beginAnnotationProgress(1, 'ch1', 21)
    store.updateAnnotationProgress({
      phase: 'resolution',
      entities: 5,
      paragraphsDone: 1,
      categoryCounts: { person: 3, location: 2 },
      lastParagraphIndex: 0,
      message: '段落 1/21 完成，定位 5 个实体',
    })
    store.updateAnnotationProgress({
      phase: 'resolution',
      entities: 12,
      paragraphsDone: 2,
      categoryCounts: { person: 7, location: 3, term: 2 },
      lastParagraphIndex: 2,
      message: '段落 3/21 完成，定位 7 个实体',
    })

    const p = store.annotationProgress
    expect(p.phase).toBe('resolution')
    expect(p.entities).toBe(12)
    expect(p.paragraphsDone).toBe(2)
    expect(p.categoryCounts.person).toBe(7)
    expect(p.categoryCounts.term).toBe(2)
    expect(p.lastParagraphIndex).toBe(2)
  })

  it('marginalia 阶段更新：旁注进度', () => {
    const store = useAgentStore()
    store.beginAnnotationProgress(1, 'ch1', 21)
    store.updateAnnotationProgress({
      phase: 'marginalia',
      marginaliaTotal: 10,
      marginaliaDone: 4,
      message: '旁注 4/10：「白素贞」✓',
    })

    const p = store.annotationProgress
    expect(p.phase).toBe('marginalia')
    expect(p.marginaliaTotal).toBe(10)
    expect(p.marginaliaDone).toBe(4)
  })

  it('finishAnnotationProgress：进入 done 态，封存真实统计，active 关闭但结果保留展示', () => {
    const store = useAgentStore()
    store.beginAnnotationProgress(1, 'ch1', 21)
    store.updateAnnotationProgress({
      phase: 'resolution',
      entities: 48,
      paragraphsDone: 17,
      categoryCounts: { person: 20, location: 12, event: 6, term: 8, motif: 2 },
    })
    store.updateAnnotationProgress({ phase: 'saving', message: '正在保存标注...' })
    store.finishAnnotationProgress({ marginalia: 5 })

    const p = store.annotationProgress
    expect(p.status).toBe('done')
    expect(p.phase).toBe('done')
    expect(p.active).toBe(false)
    expect(p.finalStats).not.toBeNull()
    expect(p.finalStats!.entities).toBe(48)
    expect(p.finalStats!.paragraphs).toBe(17)
    expect(p.finalStats!.marginalia).toBe(5)
    expect(p.finalStats!.categoryCounts.person).toBe(20)
  })

  it('failAnnotationProgress：进入 error 态并保留错误信息', () => {
    const store = useAgentStore()
    store.beginAnnotationProgress(1, 'ch1', 21)
    store.failAnnotationProgress('实体发现阶段失败: 模型超时')

    const p = store.annotationProgress
    expect(p.status).toBe('error')
    expect(p.phase).toBe('error')
    expect(p.active).toBe(false)
    expect(p.errorMessage).toContain('模型超时')
  })

  it('cancelAnnotationProgress：用户主动停止，状态为 cancelled', () => {
    const store = useAgentStore()
    store.beginAnnotationProgress(1, 'ch1', 21)
    store.updateAnnotationProgress({ phase: 'discovery', windowsDone: 1, windowsTotal: 7, candidates: 3 })
    store.cancelAnnotationProgress()

    const p = store.annotationProgress
    expect(p.status).toBe('cancelled')
    expect(p.active).toBe(false)
  })

  it('startStreaming 重置上一轮标注进度，避免跨会话残留', () => {
    const store = useAgentStore()
    store.beginAnnotationProgress(1, 'ch1', 21)
    store.finishAnnotationProgress({ marginalia: 0 })
    expect(store.annotationProgress.status).toBe('done')

    store.startStreaming()
    expect(store.annotationProgress.active).toBe(false)
    expect(store.annotationProgress.status).toBe('idle')
    expect(store.annotationProgress.phase).toBe('idle')
    expect(store.annotationProgress.finalStats).toBeNull()
  })

  it('recent 日志有上限，防止长时间运行无限增长', () => {
    const store = useAgentStore()
    store.beginAnnotationProgress(1, 'ch1', 100)
    for (let i = 0; i < 60; i++) {
      store.updateAnnotationProgress({ phase: 'resolution', message: `段落 ${i} 完成` })
    }
    expect(store.annotationProgress.recent.length).toBeLessThanOrEqual(40)
  })
})
