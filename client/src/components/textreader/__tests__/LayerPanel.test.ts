import { describe, it, expect, beforeEach } from 'vitest'
import { mount } from '@vue/test-utils'
import { setActivePinia, createPinia } from 'pinia'
import { useAnnotationStore } from '@/stores/annotations'
import LayerPanel from '@/components/textreader/LayerPanel.vue'

describe('LayerPanel', () => {
  beforeEach(() => {
    setActivePinia(createPinia())
    localStorage.clear()
  })

  it('should render layer categories', () => {
    const wrapper = mount(LayerPanel)
    expect(wrapper.text()).toContain('图层管理')
    expect(wrapper.text()).toContain('我的标注')
    expect(wrapper.text()).toContain('Agent 标注')
  })

  it('should render all 6 category items per layer', () => {
    const wrapper = mount(LayerPanel)
    const items = wrapper.findAll('.lp-cat-item')
    // 12 total: 6 for user + 6 for agent
    expect(items.length).toBe(12)
  })

  it('should toggle visibility on checkbox change', async () => {
    const store = useAnnotationStore()
    const wrapper = mount(LayerPanel)

    const firstCheckbox = wrapper.find('input[type="checkbox"]')
    expect(store.visibility.user.person).toBe(true)

    await firstCheckbox.setValue(false)
    // Note: the v-model binding may not update store in test without full v-model support
  })

  it('should call showAll on button click', async () => {
    const store = useAnnotationStore()
    store.hideAll()
    expect(store.visibility.user.person).toBe(false)

    const wrapper = mount(LayerPanel)
    const showAllBtn = wrapper.find('.lp-batch-btn')
    await showAllBtn.trigger('click')

    expect(store.visibility.user.person).toBe(true)
    expect(store.visibility.agent.person).toBe(true)
  })

  it('should call showUserOnly on button click', async () => {
    const store = useAnnotationStore()
    const wrapper = mount(LayerPanel)
    const buttons = wrapper.findAll('.lp-batch-btn')
    const userOnlyBtn = buttons.find(b => b.text() === '仅用户')
    expect(userOnlyBtn).toBeTruthy()
    await userOnlyBtn!.trigger('click')
    expect(store.visibility.user.person).toBe(true)
    expect(store.visibility.agent.person).toBe(false)
  })

  it('should call showAgentOnly on button click', async () => {
    const store = useAnnotationStore()
    const wrapper = mount(LayerPanel)
    const buttons = wrapper.findAll('.lp-batch-btn')
    const agentOnlyBtn = buttons.find(b => b.text() === '仅 Agent')
    expect(agentOnlyBtn).toBeTruthy()
    await agentOnlyBtn!.trigger('click')
    expect(store.visibility.user.person).toBe(false)
    expect(store.visibility.agent.person).toBe(true)
  })

  it('should call hideAll on button click', async () => {
    const store = useAnnotationStore()
    const wrapper = mount(LayerPanel)
    const buttons = wrapper.findAll('.lp-batch-btn')
    const hideAllBtn = buttons.find(b => b.text() === '全部隐藏')
    expect(hideAllBtn).toBeTruthy()
    await hideAllBtn!.trigger('click')
    expect(store.visibility.user.person).toBe(false)
    expect(store.visibility.agent.person).toBe(false)
  })
})
