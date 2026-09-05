import { describe, it, expect, beforeEach } from 'vitest'
import { SSEEventBus } from '@/services/sseEventBus'
import type { ParsedSSEEvent } from '@/services/sseEventBus'

describe('SSEEventBus', () => {
  let bus: SSEEventBus

  beforeEach(() => {
    bus = SSEEventBus.getInstance()
    bus.clearAllHandlers()
  })

  it('should parse SSE data line and dispatch to registered handler', () => {
    const received: ParsedSSEEvent[] = []
    bus.on('chunk', (event) => {
      received.push(event)
    })

    bus.dispatch('data: {"type":"chunk","text":"Hello World"}')

    expect(received.length).toBe(1)
    expect(received[0].type).toBe('chunk')
    expect(received[0].data.text).toBe('Hello World')
  })

  it('should handle non-JSON data as text', () => {
    const received: ParsedSSEEvent[] = []
    bus.on('chunk', (event) => {
      received.push(event)
    })

    bus.dispatch('data: plain text content', 'chunk')

    expect(received.length).toBe(1)
    expect(received[0].data.text).toBe('plain text content')
  })

  it('should buffer events during interrupt', () => {
    const received: ParsedSSEEvent[] = []
    bus.on('chunk', (event) => {
      received.push(event)
    })

    bus.enterInterrupt()

    bus.dispatch('data: {"type":"chunk","text":"buffered 1"}')
    bus.dispatch('data: {"type":"chunk","text":"buffered 2"}')

    expect(received.length).toBe(0)

    bus.exitInterrupt()

    expect(received.length).toBe(2)
    expect(received[0].data.text).toBe('buffered 1')
    expect(received[1].data.text).toBe('buffered 2')
  })

  it('should allow interrupt_resumed through interrupt barrier', () => {
    const received: ParsedSSEEvent[] = []
    bus.on('interrupt_resumed', (event) => {
      received.push(event)
    })
    bus.on('chunk', (event) => {
      received.push(event)
    })

    bus.enterInterrupt()

    bus.dispatch('data: {"type":"interrupt_resumed","threadId":"t1"}')
    bus.dispatch('data: {"type":"chunk","text":"after resume"}')

    expect(received.length).toBe(1)
    expect(received[0].type).toBe('interrupt_resumed')

    bus.exitInterrupt()
    expect(received.length).toBe(2)
    expect(received[1].type).toBe('chunk')
  })

  it('should unregister handlers', () => {
    const received: ParsedSSEEvent[] = []
    const handler = (event: ParsedSSEEvent) => received.push(event)

    bus.on('test', handler)
    bus.dispatch('data: {"type":"test","value":1}')
    expect(received.length).toBe(1)

    bus.off('test', handler)
    bus.dispatch('data: {"type":"test","value":2}')
    expect(received.length).toBe(1)
  })

  it('should fire wildcard handlers for all events', () => {
    const received: ParsedSSEEvent[] = []
    bus.on('*', (event) => {
      received.push(event)
    })

    bus.dispatch('data: {"type":"chunk","text":"a"}')
    bus.dispatch('data: {"type":"tool_call","tool":"search"}')

    expect(received.length).toBe(2)
  })

  it('should handle handler errors gracefully', () => {
    bus.on('bad', () => {
      throw new Error('Handler error')
    })

    const received: ParsedSSEEvent[] = []
    bus.on('bad', (event) => {
      received.push(event)
    })

    expect(() => bus.dispatch('data: {"type":"bad"}')).not.toThrow()
    expect(received.length).toBe(1)
  })

  it('should be a singleton', () => {
    const bus2 = SSEEventBus.getInstance()
    expect(bus2).toBe(bus)
  })

  // J-07: New event routes

  it('should route annotation_suggest events', () => {
    const received: ParsedSSEEvent[] = []
    bus.on('annotation_suggest', (event) => {
      received.push(event)
    })

    bus.dispatch('data: {"type":"annotation_suggest","target":"3:0","suggestions":[{"id":"s1","span":{"startChar":0,"endChar":2},"text":"a","category":"person","color":"#4A90D9","reason":"test"}]}')

    expect(received.length).toBe(1)
    expect(received[0].type).toBe('annotation_suggest')
    expect(received[0].data.target).toBe('3:0')
    expect(received[0].data.suggestions).toHaveLength(1)
  })

  it('should route paragraph_alignment events', () => {
    const received: ParsedSSEEvent[] = []
    bus.on('paragraph_alignment', (event) => {
      received.push(event)
    })

    bus.dispatch('data: {"type":"paragraph_alignment","leftChapter":1,"rightChapter":2,"alignments":[{"leftParagraphIndex":0,"rightParagraphIndex":0,"confidence":0.95}]}')

    expect(received.length).toBe(1)
    expect(received[0].type).toBe('paragraph_alignment')
    expect(received[0].data.leftChapter).toBe(1)
    expect(received[0].data.rightChapter).toBe(2)
  })

  it('should route annotation_data with splitSide', () => {
    const received: ParsedSSEEvent[] = []
    bus.on('annotation_data', (event) => {
      received.push(event)
    })

    bus.dispatch('data: {"type":"annotation_data","passageKey":"1:3","annotations":[],"insights":[],"splitSide":"left"}')

    expect(received.length).toBe(1)
    expect(received[0].data.splitSide).toBe('left')
  })
})
