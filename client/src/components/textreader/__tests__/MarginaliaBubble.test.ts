import { describe, it, expect, beforeEach } from 'vitest'
import { mount } from '@vue/test-utils'
import MarginaliaBubble from '@/components/textreader/MarginaliaBubble.vue'

describe('MarginaliaBubble', () => {
  const mockMarginalia = {
    id: 'm1',
    annotationId: 'a1',
    chapterNumber: 3,
    paragraphIndex: 0,
    anchorCharOffset: 10,
    content: '这是一条测试批注',
    source: 'user' as const,
    createdAt: Date.now(),
    updatedAt: Date.now(),
  }

  it('should render marginalia content', () => {
    const wrapper = mount(MarginaliaBubble, {
      props: {
        marginalia: mockMarginalia,
        editing: false,
      },
    })
    expect(wrapper.text()).toContain('这是一条测试批注')
  })

  it('should show preview when collapsed (first 2 lines)', () => {
    const multiLineMarginalia = {
      ...mockMarginalia,
      content: '第一行\n第二行\n第三行\n第四行',
    }
    const wrapper = mount(MarginaliaBubble, {
      props: {
        marginalia: multiLineMarginalia,
        editing: false,
      },
    })
    // Should show first 2 lines
    expect(wrapper.text()).toContain('第一行')
    expect(wrapper.text()).toContain('第二行')
    expect(wrapper.text()).toContain('...')
  })

  it('should expand on click', async () => {
    const multiLineMarginalia = {
      ...mockMarginalia,
      content: '第一行\n第二行\n第三行',
    }
    const wrapper = mount(MarginaliaBubble, {
      props: {
        marginalia: multiLineMarginalia,
        editing: false,
      },
    })
    await wrapper.trigger('click')
    expect(wrapper.text()).toContain('第三行')
  })

  it('should emit edit event', async () => {
    const wrapper = mount(MarginaliaBubble, {
      props: {
        marginalia: mockMarginalia,
        editing: false,
      },
    })
    const editBtn = wrapper.find('.mb-btn')
    await editBtn.trigger('click')
    expect(wrapper.emitted('edit')).toBeTruthy()
    expect(wrapper.emitted('edit')![0]).toEqual(['m1'])
  })

  it('should emit delete event', async () => {
    const wrapper = mount(MarginaliaBubble, {
      props: {
        marginalia: mockMarginalia,
        editing: false,
      },
    })
    const deleteBtn = wrapper.find('.mb-btn-del')
    await deleteBtn.trigger('click')
    expect(wrapper.emitted('delete')).toBeTruthy()
    expect(wrapper.emitted('delete')![0]).toEqual(['m1'])
  })

  it('should apply editing class when editing prop is true', () => {
    const wrapper = mount(MarginaliaBubble, {
      props: {
        marginalia: mockMarginalia,
        editing: true,
      },
    })
    expect(wrapper.classes()).toContain('editing')
  })
})
