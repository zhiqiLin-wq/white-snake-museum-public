import { describe, it, expect, beforeEach } from 'vitest'
import { setActivePinia, createPinia } from 'pinia'
import { useEvolutionStore } from '@/stores/evolution'

describe('useEvolutionStore', () => {
  beforeEach(() => {
    setActivePinia(createPinia())
  })

  it('should initialize evolution in E1_decompose phase', () => {
    const store = useEvolutionStore()
    store.initEvolution('thread_001', '雷峰塔')
    expect(store.phase).toBe('E1_decompose')
    expect(store.trackingTarget).toBe('雷峰塔')
  })

  it('should transition to E1_interrupt on confirm_motifs interrupt', () => {
    const store = useEvolutionStore()
    store.initEvolution('thread_001', '雷峰塔')
    store.handleEvolutionInterrupt({
      threadId: 'thread_001',
      interruptId: 'int_001',
      interruptType: 'confirm_motifs',
      data: {
        motifs: [
          { id: 'M1', name: '镇压者身份', type: 'invariant', core: '镇压妖怪的人物身份', isUserEdited: false },
        ],
        suggested_additions: [],
      },
    })
    expect(store.phase).toBe('E1_interrupt')
    expect(store.isAwaitingUserInput).toBe(true)
    expect(store.motifs.length).toBe(1)
  })

  it('should edit motif and mark as user edited', () => {
    const store = useEvolutionStore()
    store.initEvolution('thread_001', '雷峰塔')
    store.handleEvolutionInterrupt({
      threadId: 'thread_001',
      interruptId: 'int_001',
      interruptType: 'confirm_motifs',
      data: {
        motifs: [
          { id: 'M1', name: '镇压者身份', type: 'invariant', core: '镇压妖怪的人物身份', isUserEdited: false },
        ],
        suggested_additions: [],
      },
    })
    store.editMotif('M1', { name: '镇压者角色' })
    expect(store.motifs[0].name).toBe('镇压者角色')
    expect(store.motifs[0].isUserEdited).toBe(true)
  })

  it('should build resume payload for confirm_motifs', () => {
    const store = useEvolutionStore()
    store.initEvolution('thread_001', '雷峰塔')
    store.handleEvolutionInterrupt({
      threadId: 'thread_001',
      interruptId: 'int_001',
      interruptType: 'confirm_motifs',
      data: { motifs: [], suggested_additions: [] },
    })
    store.confirmMotifs()
    const payload = store.submitResume()
    expect(payload.threadId).toBe('thread_001')
    expect(payload.interruptId).toBe('int_001')
  })

  it('should reset to idle', () => {
    const store = useEvolutionStore()
    store.initEvolution('thread_001', '雷峰塔')
    store.reset()
    expect(store.phase).toBe('idle')
    expect(store.motifs.length).toBe(0)
  })

  // B-150: 验证锚点ID模式的 evidence refs 存储
  it('should store evidence refs with label keys from evolution_narrative event', () => {
    const store = useEvolutionStore()
    store.initEvolution('thread_001', '雷峰塔')
    store.handleEvolutionNarrative(
      '白蛇化人在唐代已出现。[唐-ch1:3]',
      {
        '唐-ch1:3': {
          chapterNumber: 1,
          chapterTitle: '太平广记·李黄',
          paragraphIndex: 3,
          excerpt: '白蛇化为人形，与李黄结为夫妇...',
          dynasty: '唐',
          sourceType: 'primary_literature',
          label: '唐-ch1:3',
        },
      }
    )
    expect(store.narrative).toBe('白蛇化人在唐代已出现。[唐-ch1:3]')
    expect(store.evidenceRefs).toHaveProperty('唐-ch1:3')
    expect(store.evidenceRefs['唐-ch1:3'].label).toBe('唐-ch1:3')
    expect(store.evidenceRefs['唐-ch1:3'].dynasty).toBe('唐')
    // B-150: 确认已删除的字段不存在
    const ref = store.evidenceRefs['唐-ch1:3'] as unknown as Record<string, unknown>
    expect(ref).not.toHaveProperty('shortId')
    expect(ref).not.toHaveProperty('quotedText')
    expect(ref).not.toHaveProperty('modalQuote')
  })
})
