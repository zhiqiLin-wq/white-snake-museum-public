import { describe, it, expect, beforeEach } from 'vitest'
import { mount } from '@vue/test-utils'
import { setActivePinia, createPinia } from 'pinia'
import { createRouter, createWebHistory } from 'vue-router'
import { h } from 'vue'

// Simple component tests for layout structure
import TabBar from '@/components/ui/TabBar.vue'
import TabItem from '@/components/ui/TabItem.vue'
import IconButton from '@/components/ui/IconButton.vue'

describe('TabBar', () => {
  it('should render tab items', () => {
    const wrapper = mount(TabBar, {
      props: {
        tabs: [
          { id: '1', type: 'text-reader' as const, title: 'Tab 1', data: {}, dirty: false, createdAt: Date.now() },
          { id: '2', type: 'text-reader' as const, title: 'Tab 2', data: {}, dirty: true, createdAt: Date.now() },
        ],
        activeTabId: '1',
      },
      global: {
        stubs: { TabItem: true },
      },
    })
    expect(wrapper.find('.tab-bar').exists()).toBe(true)
    expect(wrapper.findAllComponents({ name: 'TabItem' }).length).toBe(2)
  })
})

describe('TabItem', () => {
  it('should display title and emit click and close', async () => {
    const wrapper = mount(TabItem, {
      props: {
        title: 'Test Tab',
        active: false,
        dirty: false,
      },
    })

    expect(wrapper.text()).toContain('Test Tab')

    await wrapper.trigger('click')
    expect(wrapper.emitted('click')).toBeTruthy()

    const closeBtn = wrapper.find('.tab-close')
    await closeBtn.trigger('click')
    expect(wrapper.emitted('close')).toBeTruthy()
  })

  it('should truncate long titles', () => {
    const wrapper = mount(TabItem, {
      props: {
        title: 'A very long tab title that exceeds twenty characters',
        active: false,
        dirty: false,
      },
    })
    const displayed = wrapper.find('.tab-title').text()
    expect(displayed.length).toBeLessThanOrEqual(23)
    expect(displayed.endsWith('...')).toBe(true)
  })
})

describe('IconButton', () => {
  it('should render icon and label', () => {
    const wrapper = mount(IconButton, {
      props: {
        icon: 'X',
        label: 'Close',
        variant: 'danger',
      },
    })
    expect(wrapper.text()).toContain('Close')
    expect(wrapper.find('.btn-icon').text()).toBe('X')
    expect(wrapper.classes()).toContain('variant-danger')
  })

  it('should emit click', async () => {
    const wrapper = mount(IconButton, {
      props: {
        icon: '+',
        variant: 'primary',
      },
    })
    await wrapper.trigger('click')
    expect(wrapper.emitted('click')).toBeTruthy()
  })
})
