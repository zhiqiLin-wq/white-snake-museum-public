import { describe, it, expect, beforeEach } from 'vitest'
import { setActivePinia, createPinia } from 'pinia'
import { useAnnotationStore } from '@/stores/annotations'

function makeAnnotation(overrides: Record<string, unknown> = {}) {
  const now = Date.now()
  return {
    id: 'a1',
    category: 'person' as const,
    label: '许宣',
    span: { startChar: 0, endChar: 2 },
    explanation: '主人公',
    confidence: 0.95,
    color: '#4A90D9',
    source: 'agent' as const,
    createdAt: now,
    updatedAt: now,
    ...overrides,
  }
}

describe('useAnnotationStore', () => {
  beforeEach(() => {
    setActivePinia(createPinia())
    localStorage.clear()
  })

  it('should load agent annotations', () => {
    const store = useAnnotationStore()
    store.loadAiAnnotations([
      makeAnnotation({ id: 'a1' }),
      makeAnnotation({ id: 'a2', category: 'location', label: '雷峰塔', span: { startChar: 9, endChar: 12 } }),
    ], '4:12')
    const anns = store.annotationsForKey('4:12')
    expect(anns.length).toBe(2)
  })

  it('should add and remove user annotations with auto-derived color', () => {
    const store = useAnnotationStore()
    store.addUserAnnotation('4:12', {
      category: 'person',
      span: { startChar: 0, endChar: 2 },
      note: '这个是主人公',
    })
    const anns = store.userAnnotations['4:12']
    expect(anns.length).toBe(1)
    expect(anns[0].color).toBe('#4A90D9')
    store.removeUserAnnotation('4:12', anns[0].id)
    expect(store.userAnnotations['4:12'].length).toBe(0)
  })

  it('should add user annotation with explicit color', () => {
    const store = useAnnotationStore()
    store.addUserAnnotation('3:5', {
      category: 'custom',
      span: { startChar: 5, endChar: 10 },
      note: 'custom note',
      color: '#FF0000',
    })
    expect(store.userAnnotations['3:5'][0].color).toBe('#FF0000')
  })

  // J-01: New actions

  it('should add, update, and remove marginalia', () => {
    const store = useAnnotationStore()
    store.addMarginalia('3:0', {
      chapterNumber: 3,
      paragraphIndex: 0,
      anchorCharOffset: 10,
      content: '这是一条批注',
    })
    const margs = store.marginaliaForKey('3:0')
    expect(margs.length).toBe(1)
    expect(margs[0].content).toBe('这是一条批注')

    store.updateMarginalia('3:0', margs[0].id, { content: '修改后的批注' })
    expect(store.marginaliaForKey('3:0')[0].content).toBe('修改后的批注')

    store.removeMarginalia('3:0', margs[0].id)
    expect(store.marginaliaForKey('3:0').length).toBe(0)
  })

  it('should load, accept, and dismiss suggestions', () => {
    const store = useAnnotationStore()
    store.loadSuggestedAnnotations('3:0', [
      {
        id: 's1',
        span: { startChar: 0, endChar: 3 },
        text: '白素贞',
        category: 'person',
        color: '#4A90D9',
        reason: '主要人物',
      },
    ])
    expect(store.suggestedAnnotations['3:0'].length).toBe(1)

    store.acceptSuggestion('3:0', 's1')
    expect(store.suggestedAnnotations['3:0'].length).toBe(0)
    expect(store.userAnnotations['3:0'].length).toBe(1)
  })

  it('should accept all suggestions', () => {
    const store = useAnnotationStore()
    store.loadSuggestedAnnotations('3:0', [
      { id: 's1', span: { startChar: 0, endChar: 2 }, text: 'a', category: 'person', color: '#4A90D9', reason: 'r1' },
      { id: 's2', span: { startChar: 3, endChar: 5 }, text: 'b', category: 'location', color: '#50C878', reason: 'r2' },
    ])
    store.acceptAllSuggestions('3:0')
    expect(store.suggestedAnnotations['3:0'].length).toBe(0)
    expect(store.userAnnotations['3:0'].length).toBe(2)
  })

  it('should dismiss suggestion and dismiss all', () => {
    const store = useAnnotationStore()
    store.loadSuggestedAnnotations('3:0', [
      { id: 's1', span: { startChar: 0, endChar: 2 }, text: 'a', category: 'person', color: '#4A90D9', reason: 'r1' },
    ])
    store.dismissSuggestion('3:0', 's1')
    expect(store.suggestedAnnotations['3:0'].length).toBe(0)

    store.loadSuggestedAnnotations('3:0', [
      { id: 's2', span: { startChar: 0, endChar: 2 }, text: 'b', category: 'person', color: '#4A90D9', reason: 'r2' },
    ])
    store.dismissAllSuggestions('3:0')
    expect(store.suggestedAnnotations['3:0']).toBeUndefined()
  })

  it('should adopt agent annotation', () => {
    const store = useAnnotationStore()
    const now = Date.now()
    store.loadAiAnnotations([{
      id: 'ag1',
      category: 'event',
      label: '水漫金山',
      span: { startChar: 0, endChar: 4 },
      explanation: '重要事件',
      confidence: 0.9,
      color: '#F5A623',
      source: 'agent' as const,
      createdAt: now,
      updatedAt: now,
    }], '2:3')
    store.adoptAgentAnnotation('2:3', 'ag1')
    expect(store.userAnnotations['2:3'].length).toBe(1)
    expect(store.userAnnotations['2:3'][0].category).toBe('event')
  })

  it('should adopt all agent annotations', () => {
    const store = useAnnotationStore()
    const now = Date.now()
    store.loadAiAnnotations([
      { id: 'ag1', category: 'person', label: 'a', span: { startChar: 0, endChar: 1 }, explanation: 'e1', confidence: 0.9, color: '#4A90D9', source: 'agent' as const, createdAt: now, updatedAt: now },
      { id: 'ag2', category: 'location', label: 'b', span: { startChar: 2, endChar: 3 }, explanation: 'e2', confidence: 0.8, color: '#50C878', source: 'agent' as const, createdAt: now, updatedAt: now },
    ], '2:3')
    store.adoptAllAgentAnnotations('2:3')
    expect(store.userAnnotations['2:3'].length).toBe(2)
  })

  // J-02: Layered visibility

  it('should have layered visibility structure', () => {
    const store = useAnnotationStore()
    expect(store.visibility.user).toBeDefined()
    expect(store.visibility.agent).toBeDefined()
    expect(store.visibility.user.person).toBe(true)
    expect(store.visibility.user.custom).toBe(true)
  })

  it('should toggle visibility by layer and category', () => {
    const store = useAnnotationStore()
    store.toggleVisibility('user', 'person')
    expect(store.visibility.user.person).toBe(false)
    store.toggleVisibility('user', 'person')
    expect(store.visibility.user.person).toBe(true)
  })

  it('should show all, hide all, user only, agent only', () => {
    const store = useAnnotationStore()

    store.hideAll()
    expect(store.visibility.user.person).toBe(false)
    expect(store.visibility.agent.person).toBe(false)

    store.showAll()
    expect(store.visibility.user.person).toBe(true)
    expect(store.visibility.agent.person).toBe(true)

    store.showUserOnly()
    expect(store.visibility.user.person).toBe(true)
    expect(store.visibility.agent.person).toBe(false)

    store.showAgentOnly()
    expect(store.visibility.user.person).toBe(false)
    expect(store.visibility.agent.person).toBe(true)
  })

  it('should filter visible annotations by layered visibility', () => {
    const store = useAnnotationStore()
    const now = Date.now()
    store.loadAiAnnotations([{
      id: 'ag1', category: 'person', label: 'a', span: { startChar: 0, endChar: 1 },
      explanation: 'e1', confidence: 0.9, color: '#4A90D9', source: 'agent' as const,
      createdAt: now, updatedAt: now,
    }], '1:0')
    store.addUserAnnotation('1:0', {
      category: 'location', span: { startChar: 2, endChar: 3 }, note: 'n1',
    })
    // Hide agent person, should only see user location
    store.toggleVisibility('agent', 'person')
    const visible = store.visibleAnnotations('1:0')
    // Agent person is hidden, user location should be visible
    const locationItems = visible.filter(v => v.category === 'location')
    expect(locationItems.length).toBe(1)
  })

  // Getters

  it('should count annotations by chapter', () => {
    const store = useAnnotationStore()
    store.addUserAnnotation('3:0', { category: 'person', span: { startChar: 0, endChar: 2 }, note: 'n1' })
    store.addUserAnnotation('3:1', { category: 'location', span: { startChar: 0, endChar: 2 }, note: 'n2' })
    store.addUserAnnotation('4:0', { category: 'event', span: { startChar: 0, endChar: 2 }, note: 'n3' })
    expect(store.userAnnotationCount(3)).toBe(2)
    expect(store.userAnnotationCount(4)).toBe(1)
    expect(store.hasUserAnnotations(3)).toBe(true)
    expect(store.hasUserAnnotations(5)).toBe(false)
  })

  it('should get annotations by chapter', () => {
    const store = useAnnotationStore()
    store.addUserAnnotation('3:0', { category: 'person', span: { startChar: 0, endChar: 2 }, note: 'n1' })
    const result = store.getAnnotationsByChapter(3)
    expect(result.length).toBe(1)
    expect(result[0].passageKey).toBe('3:0')
  })

  it('should export HTML and plain text', () => {
    const store = useAnnotationStore()
    store.addUserAnnotation('3:0', { category: 'person', span: { startChar: 0, endChar: 2 }, note: 'n1' })
    store.addMarginalia('3:0', {
      chapterNumber: 3, paragraphIndex: 0, anchorCharOffset: 5, content: '批注内容',
    })
    const html = store.exportAnnotationsHTML(3, false)
    expect(html).toContain('Chapter 3')
    expect(html).toContain('ann-user')
    expect(html).toContain('批注内容')

    const text = store.exportAnnotationsPlainText(3, false)
    expect(text).toContain('Chapter 3')
    expect(text).toContain('[person]')
    expect(text).toContain('[Marginalia]')
  })

  // Persistence (SQLite server — tested at integration level)

  it('should add annotation and marginalia to in-memory state', () => {
    const store = useAnnotationStore()
    store.addUserAnnotation('3:0', { category: 'person', span: { startChar: 0, endChar: 2 }, note: 'n1' })
    store.addMarginalia('3:0', {
      chapterNumber: 3, paragraphIndex: 0, anchorCharOffset: 5, content: '批注',
    })

    // State is updated immediately in-memory (server persist is tested via integration)
    expect(store.userAnnotations['3:0']).toBeDefined()
    expect(store.userAnnotations['3:0'].length).toBe(1)
    expect(store.marginaliaForKey('3:0').length).toBe(1)
  })

  it('should clear agent annotations', () => {
    const store = useAnnotationStore()
    const now = Date.now()
    store.loadAiAnnotations([{
      id: 'ag1', category: 'person', label: 'a', span: { startChar: 0, endChar: 1 },
      explanation: 'e1', confidence: 0.9, color: '#4A90D9', source: 'agent' as const,
      createdAt: now, updatedAt: now,
    }], '1:0')
    store.clearAgentAnnotations()
    expect(store.annotationsForKey('1:0').length).toBe(0)
  })

  it('should support visibleAnnotations with splitSide parameter', () => {
    const store = useAnnotationStore()
    store.addUserAnnotation('2:1', { category: 'person', span: { startChar: 0, endChar: 2 }, note: 'n1' })
    const visible = store.visibleAnnotations('2:1', 'left')
    expect(visible.length).toBe(1)
  })
})
