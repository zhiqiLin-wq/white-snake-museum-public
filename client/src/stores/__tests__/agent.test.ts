import { describe, it, expect, beforeEach } from 'vitest'
import { setActivePinia, createPinia } from 'pinia'
import { useAgentStore } from '@/stores/agent'

describe('useAgentStore', () => {
  beforeEach(() => {
    setActivePinia(createPinia())
    localStorage.clear()
  })

  it('should create a conversation', () => {
    const store = useAgentStore()
    store.createConversation()
    expect(store.conversations.length).toBe(1)
    expect(store.activeConversation).not.toBeNull()
  })

  it('should switch conversations', () => {
    const store = useAgentStore()
    const conv1 = store.createConversation()
    const conv2 = store.createConversation()
    store.setActiveConversation(conv1.id)
    expect(store.activeId).toBe(conv1.id)
  })

  it('should add and remove context tags', () => {
    const store = useAgentStore()
    store.createConversation()
    store.addContextTag({
      id: 'tag1',
      label: 'Test tag',
      data: {},
      sourceType: 'text_selection',
    })
    const conv = store.activeConversation
    expect(conv).not.toBeNull()
    expect(conv!.contextTags.length).toBe(1)
    store.removeContextTag('tag1')
    expect(conv!.contextTags.length).toBe(0)
  })

  it('should manage streaming state', () => {
    const store = useAgentStore()
    expect(store.isStreaming).toBe(false)
    store.startStreaming()
    expect(store.isStreaming).toBe(true)
    store.appendChunk('Hello')
    expect(store.currentChunk).toBe('Hello')
    store.finishStreaming()
    expect(store.isStreaming).toBe(false)
  })
})
