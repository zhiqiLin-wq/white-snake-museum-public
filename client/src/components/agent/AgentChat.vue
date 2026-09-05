<script setup lang="ts">
import { ref, computed, watch, nextTick, onMounted } from 'vue'
import { useAgentStore } from '@/stores/agent'
import { useWorkspaceStore } from '@/stores/workspace'
import { useSourceTreeStore } from '@/stores/sourceTree'
import type { ContextTag, EvolutionCard } from '@/stores/agent'
import { SSEEventBus } from '@/services/sseEventBus'
import { marked } from 'marked'
import ThinkingSteps from './ThinkingSteps.vue'
import WorkflowCard from './WorkflowCard.vue'
import ContextTags from './ContextTags.vue'

//  markedGFM
marked.setOptions({
  gfm: true,
  breaks: true,
})

const agentStore = useAgentStore()
const workspaceStore = useWorkspaceStore()
const sourceTreeStore = useSourceTreeStore()

const inputText = ref('')
const textareaRef = ref<HTMLTextAreaElement | null>(null)
const messageContainer = ref<HTMLElement | null>(null)
// v17: 思考折叠区/思考内容区引用——思考中自动贴底显示最新思考
const thinkingBoxRef = ref<HTMLElement | null>(null)
const thinkingContentRef = ref<HTMLElement | null>(null)
const showSessionList = ref(false)
// 用户是否手动上滑——上滑时暂停自动滚动，滑回底部恢复
const userScrolledUp = ref(false)
// 用户鼠标在滚轮上互动：最后一次wheel 后 N 毫秒内完全禁用自动滚动。
// 解决思考阶段高频 reasoningDelta 驱动的 rAF 自动贴底与用户抢 scrollTop 的卡动感
let _interacting = false
let _interactionTimer: ReturnType<typeof setTimeout> | null = null
function _markInteracting(durationMs = 250) {
  _interacting = true
  if (_interactionTimer !== null) clearTimeout(_interactionTimer)
  _interactionTimer = setTimeout(() => {
    _interactionTimer = null
    _interacting = false
    // 用户活跃结束时如果在底部附近，自动恢复跟随滚动
    const el = messageContainer.value
    if (el) {
      const distFromBottom = el.scrollHeight - el.scrollTop - el.clientHeight
      if (distFromBottom <= 10) {
        userScrolledUp.value = false
      }
    }
  }, durationMs)
}
// 节流标记——避免高频 reasoningDelta 触发过多 DOM 操作
let _scrollTimer: ReturnType<typeof requestAnimationFrame> | null = null

const conversations = computed(() => agentStore.conversations)
const activeConv = computed(() => agentStore.activeConversation)
const messages = computed(() => agentStore.messages)
const isStreaming = computed(() => agentStore.isStreaming)
const statusSteps = computed(() => agentStore.statusSteps)
const errorMessage = computed(() => agentStore.errorMessage)
const agentStatus = computed(() => agentStore.agentStatus)

// v17: ReAct 决策轮思考折叠区（thinking_delta 累积内容）
const reasoningText = computed(() => agentStore.reasoningDelta.trim())
const reasoningActive = computed(() => agentStore.reasoningActive)

const showWelcome = computed(() =>
  !isStreaming.value &&
  (!activeConv.value || activeConv.value.messages.length === 0)
)

const contextTags = computed<ContextTag[]>(() =>
  activeConv.value?.contextTags || []
)

function scrollToBottom() {
  if (userScrolledUp.value || _interacting) return
  // 节流：用 rAF 合并同一帧内的多次调用
  if (_scrollTimer !== null) return
  _scrollTimer = requestAnimationFrame(() => {
    _scrollTimer = null
    const el = messageContainer.value
    if (!el) return
    el.scrollTop = el.scrollHeight
  })
}

// 思考进行中：让最新思考内容始终可见。
// 1) 内层 .ac-thinking-content（max-height 滚动区）自动滚到底
// 2) 回答还没开始输出时，外层容器把思考区底部贴到视口底部
function scrollToThinking() {
  if (userScrolledUp.value || _interacting) return
  const contentEl = thinkingContentRef.value
  if (contentEl) contentEl.scrollTop = contentEl.scrollHeight
  const answerStreaming = !!agentStore.currentChunk
  if (answerStreaming || !thinkingBoxRef.value) {
    scrollToBottom()
    return
  }
  if (_scrollTimer !== null) return
  _scrollTimer = requestAnimationFrame(() => {
    _scrollTimer = null
    const el = messageContainer.value
    const box = thinkingBoxRef.value
    if (!el || !box) return
    const elRect = el.getBoundingClientRect()
    const boxRect = box.getBoundingClientRect()
    const boxBottomInContent = boxRect.bottom - elRect.top + el.scrollTop
    el.scrollTop = Math.max(boxBottomInContent - el.clientHeight + 24, 0)
  })
}

// 用户滚轮任意方向 → 立即置 userScrolledUp，并用互斥锁暂停自动滚动 250ms。
// （用户向下滑时不想被 reasoningDelta 往回拉到底部；向上滑同样需要暂停贴底。）
function onContainerWheel(e: WheelEvent) {
  if (e.deltaY !== 0) {
    if (e.deltaY < 0) userScrolledUp.value = true
    _markInteracting(250)
  }
}

// 监听容器滚动——只负责"滑回底部附近时恢复自动滚动"
function onContainerScroll() {
  const el = messageContainer.value
  if (!el) return
  const distFromBottom = el.scrollHeight - el.scrollTop - el.clientHeight
  if (distFromBottom <= 10 && userScrolledUp.value) {
    userScrolledUp.value = false
  }
}

watch(messages, scrollToBottom)
watch(() => agentStore.currentChunk, (chunk, old) => {
  // 回答从无到有的瞬间重置节流计时，保证首字立即滚到底
  if (chunk && !old) {
    if (_scrollTimer !== null) {
      cancelAnimationFrame(_scrollTimer)
      _scrollTimer = null
    }
    userScrolledUp.value = false
  }
  scrollToBottom()
})
// 流式思考内容：贴底显示最新思考（内层+外层），rAF 节流
watch(() => agentStore.reasoningDelta, scrollToThinking)
// 只 watch 数组长度变化，不用 deep: true 避免昂贵遍历
watch(() => agentStore.statusSteps.length, scrollToBottom)

function sendMessage() {
  const text = inputText.value.trim()
  if (!text) return
  if (isStreaming.value) {
    if (!confirm('Agent 正在回复中，确定要中断吗？')) return
    onStopStreaming()
    agentStore.addMessage({
      role: 'system',
      content: '用户中断了回复',
      timestamp: Date.now(),
    })
  }
  inputText.value = ''
  agentStore.sendMessage(text)
}

// 欢迎屏建议问题：覆盖核心研究场景（前 6 个基础场景 + 后 6 个工具专属场景）
const suggestedQuestions = [
  { tag: '版本源流', text: '白蛇传故事的起源和主要版本有哪些？' },
  { tag: '文脉演化', text: '白蛇的形象从唐代到现当代是如何演变的？' },
  { tag: '跨朝对读', text: '对比清代和现当代版本中许仙的描写差异' },
  { tag: '空间叙事', text: '雷峰塔在白蛇传各版本中有哪些故事？它象征着什么？' },
  { tag: '事件时间线', text: '整理白蛇传故事的关键事件时间线' },
  { tag: '学术研究', text: '有哪些关于白蛇传的学术研究论文？' },
  { tag: '逐字对勘', text: '把明代和清代版本中的"游湖借伞"情节逐字对勘，找出增删改写之处' },
  { tag: '韵文诗词', text: '白蛇传各版本中有哪些韵文诗词？' },
  { tag: '出场轨迹', text: '梳理小青在各朝代版本中的出场轨迹' },
  { tag: '多实体共现', text: '许仙与法海在哪些段落同时出现？' },
  { tag: '旅程规划', text: '规划一条杭州白蛇传主题游览路线' },
  { tag: '文本标注', text: '帮我在第一回标注关键的人物和地点' },
]

function sendSuggested(text: string) {
  if (isStreaming.value) return
  inputText.value = ''
  agentStore.sendMessage(text)
}

// v19: 下载完整长报告为 Markdown 文件
function downloadReport(report: { title: string; content: string }) {
  const safeTitle = (report.title || '研究报告').replace(/[\\/:*?"<>|]/g, '_').slice(0, 60)
  const blob = new Blob([report.content], { type: 'text/markdown;charset=utf-8' })
  const url = URL.createObjectURL(blob)
  const a = document.createElement('a')
  a.href = url
  a.download = `${safeTitle}.md`
  document.body.appendChild(a)
  a.click()
  document.body.removeChild(a)
  URL.revokeObjectURL(url)
}

// v19.2: 平台内预览报告 —— 打开主工作区"报告文件库"标签页并自动展开该报告
function previewReport(report: { reportId: string }) {
  workspaceStore.openTab('report-library', '报告文件库', { reportId: report.reportId })
}

function onKeydown(e: KeyboardEvent) {
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault()
    sendMessage()
  }
}

function onStopStreaming() {
  SSEEventBus.getInstance().abort()
  agentStore.addMessage({
    role: 'system',
    content: '用户停止了回复',
    timestamp: Date.now(),
  })
  agentStore.finishStreaming()
}

function removeTag(tagId: string) {
  agentStore.removeContextTag(tagId)
}

function createNewSession() {
  agentStore.createConversation()
  showSessionList.value = false
}

function switchSession(id: string) {
  agentStore.setActiveConversation(id)
  showSessionList.value = false
}

const _ZH_CN_MAP: Record<string, number> = {
  一: 1, 二: 2, 三: 3, 四: 4, 五: 5, 六: 6, 七: 7, 八: 8, 九: 9, 十: 10,
  十一: 11, 十二: 12, 十三: 13, 十四: 14, 十五: 15, 十六: 16, 十七: 17, 十八: 18, 十九: 19, 二十: 20,
}

// chunkId 前缀判断来源类型：primary_* = 主文献（7 个版本），research_* = 研究文献（61 块研究文章）
function _chunkType(s: { chunkId?: string; chapterNumber?: unknown; chapterTitle?: string }): 'primary' | 'research' {
  if (s.chunkId) {
    if (s.chunkId.startsWith('primary_')) return 'primary'
    if (s.chunkId.startsWith('research_')) return 'research'
  }
  // 兜底（chunkId 缺失的旧消息）：能在章节目录中找到对应章节号就算 primary，否则 research
  let chNum = Number(s.chapterNumber)
  if (isNaN(chNum)) {
    const m = String(s.chapterNumber ?? '').match(/[一二三四五六七八九十]{1,3}/)
    if (m) chNum = _ZH_CN_MAP[m[0]] || NaN
  }
  if (!isNaN(chNum) && chNum > 0 && sourceTreeStore.chapters.some(c => c.number === chNum)) return 'primary'
  if (s.chapterTitle && sourceTreeStore.chapters.some(c => {
    const t = c.title ?? ''
    const st = s.chapterTitle!
    return t === st || t.includes(st) || st.includes(t)
  })) return 'primary'
  return 'research'
}

function primarySources(sources: { chunkId?: string; chapterNumber?: unknown; chapterTitle?: string }[]) {
  return (sources || [])
    .filter(s => _chunkType(s) === 'primary')
    .map(s => ({ ...s, chapterNumber: s.chapterNumber as string | number | undefined, excerpt: undefined as string | undefined }))
}
function researchSources(sources: { chunkId?: string; chapterNumber?: unknown; chapterTitle?: string }[]) {
  return (sources || [])
    .filter(s => _chunkType(s) === 'research')
    .map(s => ({ ...s, chapterNumber: s.chapterNumber as string | number | undefined, excerpt: undefined as string | undefined }))
}

function navigateToSource(source: { chapter_number?: number | string; chapterNumber?: number | string; chapter_title?: string; chapterTitle?: string; excerpt?: string; chunkId?: string }) {
  const chapterNumber = source.chapterNumber ?? source.chapter_number
  const chapterTitle = source.chapterTitle ?? source.chapter_title

  // 1) 数字章节号
  let chNum = Number(chapterNumber)
  if (!chNum || isNaN(chNum)) {
    // 2) 中文章节号（"三" / "第X回" 等）
    const rawStr = String(chapterNumber ?? '')
    const chMatch = rawStr.match(/[一二三四五六七八九十]{1,3}/)
    if (chMatch) chNum = _ZH_CN_MAP[chMatch[0]] || NaN
  }

  if (!isNaN(chNum) && chNum > 0) {
    const srCh = sourceTreeStore.chapters.find(c => c.number === chNum)
    const srTitle = srCh ? `${srCh.dynasty} · ${srCh.title}` : (chapterTitle || `章节${chNum}`)
    workspaceStore.openTab('text-reader', srTitle, { chapterNumber: chNum })
    return
  }

  // 3) 非正文章节（研究文献/章节号为 '研究'）→ 按标题在章节目录里模糊匹配
  if (chapterTitle) {
    const hit = sourceTreeStore.chapters.find(c =>
      c.title === chapterTitle || c.title.includes(chapterTitle) ||
      String(chapterTitle).includes(c.title)
    )
    if (hit) {
      workspaceStore.openTab('text-reader', `${hit.dynasty} · ${hit.title}`, { chapterNumber: hit.number })
      return
    }
  }
  // 兜底：无法匹配的来源就不跳转（静默，因为就是一个引用参考的标签，不是点击都有落地页）
}

function onRetryStep(toolName: string) {
  // Retry the failed step by re-sending the last user message
  const conv = activeConv.value
  if (!conv) return
  const lastUserMsg = [...conv.messages].reverse().find(m => m.role === 'user')
  if (lastUserMsg) {
    agentStore.sendMessage(lastUserMsg.content)
  }
  void toolName // reserved for future per-step retry SSE event
}

function onSkipStep(toolName: string) {
  // Mark the step as skipped and dismiss the error state
  const step = agentStore.statusSteps.find(s => s.tool === toolName && s.status === 'error')
  if (step) {
    step.status = 'skipped'
    step.done = true
    step.errorMsg = undefined
  }
  // If all error steps resolved, clear error message
  const hasActiveErrors = agentStore.statusSteps.some(s => s.status === 'error')
  if (!hasActiveErrors) {
    agentStore.setError('')
  }
}

// Health polling
let healthTimer: ReturnType<typeof setInterval> | null = null
watch(agentStatus, (status) => {
  if (status === 'offline' && !healthTimer) {
    healthTimer = setInterval(() => agentStore.checkHealth(), 30000)
  } else if (status === 'online' && healthTimer) {
    clearInterval(healthTimer)
    healthTimer = null
  }
}, { immediate: true })

//  marked  Markdown
function openEvolutionCard(card: EvolutionCard) {
  if (card.workflow === 'dynasty_compare') {
    //  dynasty-compare 
    const existingTab = workspaceStore.tabs.find(t => t.type === 'dynasty-compare')
    if (existingTab) {
      workspaceStore.setActiveTab(existingTab.id)
    }
    return
  }
  // 
  workspaceStore.openTab('evolution-workspace', card.trackingTarget, {
    trackingTarget: card.trackingTarget,
    threadId: card.threadId,
  })
}

function renderSimpleMarkdown(text: string): string {
  if (!text) return ''
  try {
    // 归一化全角标点：LLM 在中文内容里常输出全角星号 ＊、下划线 ＿、
    // 反引号 ｀、波浪号 ～，marked 只识别半角，否则 **text** 会原样显示
    // 为带星号的纯文本（用户看到星号没加粗）。
    let normalized = text
      .replace(/\uFF0A/g, '*')   // ＊ → *
      .replace(/\uFF3F/g, '_')   // ＿ → _
      .replace(/\uFF40/g, '`')   // ｀ → `
      .replace(/\uFF5E/g, '~')   // ～ → ~
    // CommonMark emphasis 的 flanking 规则：** 紧贴 CJK 标点（「」『』（）等）
    // 且外侧是汉字时无法配对（如 汉字**「X」**、句中**「X」**句中），
    // marked 会原样输出星号甚至把 ** 错位配对。预把 **text** 模式直接转为
    // <strong>（marked 原生支持 inline HTML），仅匹配单行、无嵌套星号的片段。
    normalized = normalized.replace(/\*\*([^*\n]{1,120}?)\*\*/g, '<strong>$1</strong>')
    const raw = marked.parse(normalized) as string
    // marked  \n  chunk
    //
    return raw
  } catch {
    // marked
    return text.replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/\n/g, '<br>')
  }
}

onMounted(() => {
  agentStore.checkHealth()
  // ensureActiveConversation 移至 MuseumView syncConversationsFromServer 之后调用
  // 避免在服务器数据加载前创建空对话
})
</script>

<template>
  <div class="agent-chat">
    <!-- Status bar -->
    <div class="ac-status">
      <span class="ac-status-dot" :class="agentStatus" />
      <span v-if="agentStatus === 'online'">已连接</span>
      <span v-else-if="agentStatus === 'offline'">已断开</span>
      <span v-else-if="agentStatus === 'starting'">启动中...</span>
      <span v-else>检查中...</span>

      <!-- Session switcher -->
      <button class="ac-session-btn" @click="showSessionList = !showSessionList">
        {{ activeConv?.title || '对话记录' }}
      </button>

      <button class="ac-new-btn" @click="createNewSession">+</button>
    </div>

    <!-- Session list dropdown -->
    <div v-if="showSessionList" class="ac-session-list">
      <div class="ac-session-toolbar">
        <button
          class="ac-batch-toggle"
          :class="{ active: agentStore.batchMode }"
          @click="agentStore.toggleBatchMode()"
        >{{ agentStore.batchMode ? '取消' : '批量管理' }}</button>
        <template v-if="agentStore.batchMode">
          <button class="ac-select-all-btn" @click="agentStore.selectedIds.size === conversations.length ? agentStore.selectedIds.clear() : conversations.forEach(c => agentStore.toggleSelectConversation(c.id))">
            {{ agentStore.selectedIds.size === conversations.length ? '取消全选' : '全选' }}
          </button>
          <button
            class="ac-batch-delete-btn"
            :disabled="agentStore.selectedIds.size === 0"
            @click="agentStore.batchDelete()"
          >删除 ({{ agentStore.selectedIds.size }})</button>
        </template>
      </div>
      <div
        v-for="conv in conversations"
        :key="conv.id"
        class="ac-session-item"
        :class="{ active: conv.id === agentStore.activeId, selected: agentStore.selectedIds.has(conv.id) }"
      >
        <input
          v-if="agentStore.batchMode"
          type="checkbox"
          class="ac-session-checkbox"
          :checked="agentStore.selectedIds.has(conv.id)"
          @click.stop="agentStore.toggleSelectConversation(conv.id)"
        />
        <div class="ac-session-avatar">{{ (conv.title || '新对话').charAt(0) }}</div>
        <div class="ac-session-body" @click="switchSession(conv.id)">
          <span class="ac-session-title">{{ conv.title || '新对话' }}</span>
          <span class="ac-session-msgcount">{{ conv.messages.length }} 条消息</span>
        </div>
        <span class="ac-session-time">{{ new Date(conv.updatedAt).toLocaleString('zh-CN', { month:'short', day:'numeric', hour:'2-digit', minute:'2-digit' }) }}</span>
        <button
          v-if="!agentStore.batchMode && conversations.length > 1"
          class="ac-session-delete"
          title="删除会话"
          @click.stop="agentStore.deleteConversation(conv.id)"
        >&times;</button>
      </div>
      <div v-if="conversations.length === 0" class="ac-no-sessions">
        暂无会话
      </div>
    </div>

    <!-- Message area -->
    <div ref="messageContainer" class="ac-messages" @scroll="onContainerScroll" @wheel="onContainerWheel">
      <!-- Welcome screen -->
      <div v-if="showWelcome" class="ac-welcome">
        <div class="ac-welcome-logo">白蛇传 AI</div>
        <div class="ac-welcome-sub">文脉全息 Agent</div>
        <div class="ac-welcome-hints">
          <div class="ac-welcome-hints-title">试试这些研究问题：</div>
          <div class="ac-suggest-grid">
            <button
              v-for="q in suggestedQuestions"
              :key="q.text"
              class="ac-suggest-card"
              @click="sendSuggested(q.text)"
            >
              <span class="ac-suggest-tag">{{ q.tag }}</span>
              <span class="ac-suggest-text">{{ q.text }}</span>
            </button>
          </div>
        </div>
      </div>

      <!-- Messages -->
      <template v-for="msg in messages" :key="msg.timestamp">
        <!-- B-140:  /  -->
        <div
          v-if="msg.card"
          class="ac-message assistant"
        >
          <div class="ac-evolution-card" @click="openEvolutionCard(msg.card)">
            <div class="ac-evolution-card-header">
              <span class="ac-evolution-card-icon"></span>
              <span class="ac-evolution-card-title">{{ msg.card.workflow === 'dynasty_compare' ? '跨朝代对比' : '演化分析' }}{{ msg.card.trackingTarget }}</span>
              <span class="ac-evolution-card-expand">点击展开完整报告</span>
            </div>
            <div class="ac-evolution-card-summary">
              {{ msg.card.summary }}
            </div>
          </div>
        </div>
        <div
          v-else-if="msg.role !== 'system'"
          class="ac-message"
          :class="msg.role"
        >
          <!-- v17: 历史消息保留思考过程，可展开查看 -->
          <details v-if="msg.reasoning" class="ac-thinking ac-thinking-history">
            <summary>
              <span class="ac-thinking-dot" />
              <span>已思考</span>
            </summary>
            <div class="ac-thinking-content">{{ msg.reasoning }}</div>
          </details>
          <div class="ac-bubble" v-html="renderSimpleMarkdown(msg.content)" />
          <div v-if="msg.sources && msg.sources.length > 0" class="ac-sources">
            <!-- 主文献 chunkId 以 primary_ 开头 → 绿色可点击跳转卡 -->
            <span
              v-for="s in primarySources(msg.sources)"
              :key="'p-' + (s.chunkId || s.excerpt)"
              class="ac-source-link clickable"
              @click="navigateToSource(s)"
              :title="'跳转到章节原文：' + (s.chapterTitle || '')"
            >
              {{ s.chapterTitle }}
            </span>
            <!-- 研究文献 chunkId 以 research_ 开头 / 无法匹配主文献章节 → 灰色标签，不可点击 -->
            <span
              v-for="s in researchSources(msg.sources)"
              :key="'r-' + (s.chunkId || s.excerpt)"
              class="ac-source-tag"
              :title="'引用自研究文献：' + (s.chapterTitle || '')"
            >
              引用：{{ s.chapterTitle || '研究文献' }}
            </span>
          </div>
          <!-- v19: 完整长报告下载卡片 -->
          <div v-if="msg.report" class="ac-report-card">
            <div class="ac-report-head">
              <span class="ac-report-icon">MD</span>
              <span class="ac-report-title">{{ msg.report.title }}</span>
            </div>
            <div class="ac-report-meta">
              完整报告 · {{ msg.report.sectionCount }} 个小节 · 约 {{ Math.round(msg.report.totalChars / 1000) }} 千字
            </div>
            <div class="ac-report-sections" v-if="msg.report.sections && msg.report.sections.length">
              <span v-for="(s, i) in msg.report.sections" :key="i" class="ac-report-section">{{ s }}</span>
            </div>
            <div class="ac-report-buttons">
              <button class="ac-report-download" @click="previewReport(msg.report)">
                平台内预览
              </button>
              <button class="ac-report-download secondary" @click="downloadReport(msg.report)">
                下载完整报告 (.md)
              </button>
            </div>
          </div>
        </div>
        <div v-else class="ac-system-msg">
          {{ msg.content }}
        </div>
      </template>

      <!-- Streaming bubble -->
      <div v-if="isStreaming" class="ac-message assistant">
        <!-- v17: 决策轮思考内容折叠区 — 思考中呼吸圆点，结束后折叠为"已思考" -->
        <details
          v-if="reasoningText"
          ref="thinkingBoxRef"
          class="ac-thinking"
          :open="reasoningActive || undefined"
        >
          <summary>
            <span class="ac-thinking-dot" :class="{ active: reasoningActive }" />
            <span>{{ reasoningActive ? '思考中…' : '已思考' }}</span>
          </summary>
          <div ref="thinkingContentRef" class="ac-thinking-content">{{ reasoningText }}</div>
        </details>
        <div class="ac-bubble streaming">
          <span v-html="renderSimpleMarkdown(agentStore.currentChunk)" /><span class="cursor-blink">|</span>
        </div>

        <!-- Thinking steps during streaming -->
        <ThinkingSteps
          :steps="statusSteps"
          :is-streaming="true"
          @retry-step="onRetryStep"
          @skip-step="onSkipStep"
        />
      </div>

      <!-- Error card -->
      <div v-if="errorMessage && !isStreaming" class="ac-error">
        {{ errorMessage }}
      </div>

      <!-- Reconnecting indicator -->
      <div v-if="SSEEventBus.getInstance().isReconnecting && isStreaming" class="ac-reconnecting">
        ...
      </div>

      <!-- Workflow done card -->
      <WorkflowCard
        v-if="agentStore.workflowCard"
        :summary="agentStore.workflowCard.summary"
        :result-refs="agentStore.workflowCard.resultRefs"
        @navigate="(type, data) => workspaceStore.openTab(type as never, '', data)"
      />
    </div>

    <!-- Input area -->
    <div class="ac-input-area">
      <ContextTags
        :tags="contextTags"
        @remove="removeTag"
      />

      <div class="ac-input-row">
        <textarea
          ref="textareaRef"
          v-model="inputText"
          class="ac-textarea"
          placeholder="输入问题或指令..."
          rows="1"
          @keydown="onKeydown"
        />

        <button
          class="ac-send-btn"
          :disabled="!inputText.trim()"
          @click="sendMessage"
        >
          {{ isStreaming ? '中断' : '发送' }}
        </button>
        <button
          v-if="isStreaming"
          class="ac-stop-btn"
          @click="onStopStreaming"
        >
          停止
        </button>
      </div>
    </div>
  </div>
</template>

<style scoped>
.agent-chat {
  display: flex;
  flex-direction: column;
  height: 100%;
  overflow: hidden;
}

/* Status bar */
.ac-status {
  position: relative;
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 9px 14px;
  border-bottom: 1px solid var(--color-border);
  background: var(--color-bg-primary);
  font-size: 0.72rem;
  color: var(--color-text-tertiary);
  font-weight: 500;
  flex-shrink: 0;
}

.ac-status-dot {
  width: 7px;
  height: 7px;
  border-radius: 50%;
  flex-shrink: 0;
}

.ac-status-dot.online {
  background: var(--color-accent);
  box-shadow: 0 0 0 3px rgba(91, 155, 213, 0.14);
}

.ac-status-dot.offline {
  background: var(--color-error);
  box-shadow: 0 0 0 3px rgba(200, 146, 146, 0.14);
}

.ac-status-dot.starting {
  background: var(--color-warning);
  box-shadow: 0 0 0 3px rgba(196, 174, 98, 0.16);
  animation: pulse 1.2s var(--ease) infinite;
}

@keyframes pulse {
  0%, 100% { opacity: 1; }
  50% { opacity: 0.45; }
}

.ac-session-btn {
  margin-left: auto;
  display: inline-flex;
  align-items: center;
  gap: 5px;
  max-width: 200px;
  padding: 4px 10px 4px 8px;
  background: var(--color-bg-tertiary);
  border: 1px solid var(--color-border);
  border-radius: var(--radius-full);
  color: var(--color-text-primary);
  cursor: pointer;
  font-size: 0.72rem;
  font-weight: 500;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
  transition: background-color var(--dur-fast) var(--ease), border-color var(--dur-fast) var(--ease);
}
.ac-session-btn::before {
  content: '';
  width: 5px;
  height: 5px;
  border-radius: 50%;
  background: var(--color-accent);
  flex-shrink: 0;
}
.ac-session-btn:hover {
  background: var(--color-bg-hover);
  border-color: var(--color-border-dark);
}

.ac-new-btn {
  background: var(--color-bg-tertiary);
  border: 1px solid var(--color-border);
  color: var(--color-text-secondary);
  cursor: pointer;
  width: 24px;
  height: 24px;
  padding: 0;
  border-radius: 50%;
  font-size: 1.05rem;
  line-height: 1;
  display: flex;
  align-items: center;
  justify-content: center;
  flex-shrink: 0;
  transition: background-color var(--dur-fast) var(--ease), border-color var(--dur-fast) var(--ease), color var(--dur-fast) var(--ease), transform var(--dur-fast) var(--ease);
}
.ac-new-btn:hover {
  background: var(--color-accent);
  border-color: var(--color-accent);
  color: #fff;
  transform: rotate(90deg);
}

/* Session list */
.ac-session-list {
  max-height: 240px;
  overflow-y: auto;
  border-bottom: 1px solid var(--color-border);
  background: var(--color-bg-primary);
  padding: 6px 0;
  box-shadow: inset 0 1px 0 var(--color-border-light);
}

.ac-session-toolbar {
  display: flex;
  align-items: center;
  gap: 4px;
  padding: 6px 10px;
  border-bottom: 1px solid var(--color-border);
  flex-wrap: wrap;
}

.ac-batch-toggle {
  padding: 2px 8px;
  font-size: 0.65rem;
  background: var(--color-bg-tertiary);
  border: 1px solid var(--color-border);
  border-radius: var(--radius-sm);
  color: var(--color-text-secondary);
  cursor: pointer;
  transition: background-color var(--dur-fast) var(--ease), border-color var(--dur-fast) var(--ease), color var(--dur-fast) var(--ease);
}

.ac-batch-toggle.active {
  background: var(--color-accent);
  color: #fff;
  border-color: var(--color-accent);
}

.ac-select-all-btn {
  padding: 2px 8px;
  font-size: 0.65rem;
  background: var(--color-bg-tertiary);
  border: 1px solid var(--color-border);
  border-radius: var(--radius-sm);
  color: var(--color-text-secondary);
  cursor: pointer;
  transition: background-color var(--dur-fast) var(--ease), border-color var(--dur-fast) var(--ease);
}

.ac-select-all-btn:hover {
  background: var(--color-bg-hover);
  border-color: var(--color-border-dark);
}

.ac-batch-delete-btn {
  padding: 2px 8px;
  font-size: 0.65rem;
  background: var(--color-error);
  color: #fff;
  border: none;
  border-radius: var(--radius-sm);
  cursor: pointer;
  margin-left: auto;
  transition: opacity var(--dur-fast) var(--ease);
}

.ac-batch-delete-btn:disabled {
  opacity: 0.4;
  cursor: not-allowed;
}

.ac-session-item {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 8px 12px;
  cursor: pointer;
  font-size: 0.75rem;
  border-left: 2px solid transparent;
  transition: background-color var(--dur-fast) var(--ease), border-color var(--dur-fast) var(--ease);
}
.ac-session-item:not(:last-child) {
  border-bottom: 1px solid var(--color-border-light);
}

.ac-session-item:hover {
  background: var(--color-bg-hover);
}

.ac-session-item.active {
  background: var(--color-accent-light);
  border-left-color: var(--color-accent);
}

.ac-session-item.selected {
  background: var(--color-bg-tertiary);
}

.ac-session-checkbox {
  flex-shrink: 0;
  cursor: pointer;
  accent-color: var(--color-accent);
}

.ac-session-avatar {
  width: 24px;
  height: 24px;
  border-radius: 50%;
  background: var(--color-bg-tertiary);
  color: var(--color-text-secondary);
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 0.7rem;
  font-weight: 600;
  flex-shrink: 0;
  transition: background-color var(--dur-fast) var(--ease), color var(--dur-fast) var(--ease);
}
.ac-session-item.active .ac-session-avatar {
  background: var(--color-accent);
  color: #fff;
}

.ac-session-body {
  flex: 1;
  min-width: 0;
  cursor: pointer;
  display: flex;
  flex-direction: column;
  gap: 1px;
}

.ac-session-title {
  display: block;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
  color: var(--color-text-primary);
  font-weight: 500;
}

.ac-session-msgcount {
  font-size: 0.62rem;
  color: var(--color-text-tertiary);
}

.ac-session-time {
  font-size: 0.62rem;
  color: var(--color-text-tertiary);
  flex-shrink: 0;
  font-variant-numeric: tabular-nums;
}

.ac-session-delete {
  flex-shrink: 0;
  background: none;
  border: none;
  color: var(--color-text-tertiary);
  cursor: pointer;
  font-size: 1rem;
  line-height: 1;
  padding: 2px 4px;
  border-radius: var(--radius-sm);
  opacity: 0;
  transition: background-color var(--dur-fast) var(--ease), color var(--dur-fast) var(--ease), opacity var(--dur-fast) var(--ease);
}
.ac-session-item:hover .ac-session-delete {
  opacity: 1;
}
.ac-session-delete:hover {
  color: var(--color-error);
  background: var(--color-error-light);
}

.ac-no-sessions {
  padding: 16px;
  font-size: 0.72rem;
  color: var(--color-text-tertiary);
  text-align: center;
}

/* Messages */
.ac-messages {
  flex: 1;
  overflow-y: auto;
  padding: 12px;
}

.ac-welcome {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  flex: 1;
  padding: 24px 16px;
  gap: 6px;
}

.ac-welcome-logo {
  font-size: 1.35rem;
  font-weight: 600;
  color: var(--color-text-primary);
  letter-spacing: 0.02em;
}

.ac-welcome-sub {
  font-size: 0.74rem;
  color: var(--color-text-tertiary);
  letter-spacing: 0.06em;
}

.ac-welcome-hints {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 3px;
  margin-top: 18px;
  width: 100%;
  max-width: 620px;
}

.ac-welcome-hints-title {
  font-size: 0.72rem;
  color: var(--color-text-secondary);
  font-weight: 500;
  margin-bottom: 10px;
  letter-spacing: 0.04em;
}

.ac-suggest-grid {
  display: grid;
  grid-template-columns: repeat(2, 1fr);
  gap: 10px;
  width: 100%;
}

.ac-suggest-card {
  display: flex;
  flex-direction: column;
  align-items: flex-start;
  gap: 6px;
  padding: 12px 14px;
  background: var(--color-bg-primary);
  border: 1px solid var(--color-border);
  border-radius: var(--radius-sm, 8px);
  cursor: pointer;
  text-align: left;
  transition: border-color 0.18s ease, box-shadow 0.18s ease, transform 0.18s ease;
}

.ac-suggest-card:hover {
  border-color: var(--color-accent-ring, rgba(64, 158, 255, 0.4));
  box-shadow: 0 2px 10px rgba(0, 0, 0, 0.05);
  transform: translateY(-1px);
}

.ac-suggest-card:active {
  transform: translateY(0);
}

.ac-suggest-tag {
  font-size: 0.62rem;
  font-weight: 600;
  color: var(--color-accent, #409eff);
  background: var(--color-accent-light, rgba(64, 158, 255, 0.08));
  padding: 2px 8px;
  border-radius: var(--radius-full, 999px);
  letter-spacing: 0.04em;
}

.ac-suggest-text {
  font-size: 0.78rem;
  line-height: 1.5;
  color: var(--color-text-primary);
}

@media (max-width: 640px) {
  .ac-suggest-grid {
    grid-template-columns: 1fr;
  }
}

.ac-welcome-divider {
  width: 56%;
  border: none;
  border-top: 1px dashed var(--color-border);
  margin: 12px auto 2px;
}

/* Messages */
.ac-message {
  margin-bottom: 14px;
}

.ac-message.user {
  text-align: right;
}

.ac-message.user .ac-bubble {
  background: var(--color-accent);
  color: #fff;
  display: inline-block;
  max-width: 85%;
  text-align: left;
  border-radius: var(--radius-lg);
  border-bottom-right-radius: 4px;
  box-shadow: var(--shadow-sm);
}

.ac-message.assistant .ac-bubble {
  background: var(--color-bg-primary);
  color: var(--color-text-primary);
  display: inline-block;
  max-width: 92%;
  border: 1px solid var(--color-border);
  border-radius: var(--radius-lg);
  border-bottom-left-radius: 4px;
  box-shadow: var(--shadow-sm);
}

.ac-bubble {
  padding: 12px 16px;
  font-size: 0.85rem;
  line-height: 1.75;
  word-break: break-word;
  letter-spacing: 0.015em;
  text-align: justify;
}

.ac-bubble.streaming {
  border: 1px solid var(--color-border);
}

.cursor-blink {
  animation: blink 0.7s step-end infinite;
  color: var(--color-accent);
}

@keyframes blink {
  50% { opacity: 0; }
}

/* --- v17: 决策轮思考折叠区（极简：浅灰底 + 浅蓝呼吸点） --- */
.ac-thinking {
  margin-bottom: 8px;
  border: 1px solid var(--color-border);
  border-radius: 8px;
  background: #fafafa;
  font-size: 12px;
  overflow: hidden;
}

.ac-thinking summary {
  display: flex;
  align-items: center;
  gap: 6px;
  padding: 6px 10px;
  cursor: pointer;
  color: #8a8a8a;
  user-select: none;
  list-style: none;
}

.ac-thinking summary::-webkit-details-marker {
  display: none;
}

.ac-thinking-dot {
  flex: none;
  width: 6px;
  height: 6px;
  border-radius: 50%;
  background: #c2c2c2;
}

.ac-thinking-dot.active {
  background: var(--color-accent);
  animation: ac-think-breathe 1.6s ease-in-out infinite;
}

@keyframes ac-think-breathe {
  0%, 100% { opacity: 0.35; transform: scale(0.85); }
  50% { opacity: 1; transform: scale(1); }
}

.ac-thinking-content {
  max-height: 180px;
  overflow-y: auto;
  padding: 8px 10px 10px;
  border-top: 1px solid rgba(0, 0, 0, 0.05);
  color: #8a8a8a;
  line-height: 1.6;
  white-space: pre-wrap;
  word-break: break-word;
}

/* 历史消息中的思考折叠区：默认收起，展开时限高滚动 */
.ac-thinking-history {
  margin-bottom: 6px;
}
.ac-thinking-history .ac-thinking-content {
  max-height: 260px;
}

/* --- Markdown 内容排版（v-html 内容需 :deep 穿透） --- */
.ac-bubble :deep(p) { margin: 0 0 0.75em; }
.ac-bubble :deep(p:last-child) { margin-bottom: 0; }

.ac-bubble :deep(h1),
.ac-bubble :deep(h2),
.ac-bubble :deep(h3),
.ac-bubble :deep(h4) {
  margin: 1.15em 0 0.5em;
  font-weight: 600;
  color: var(--color-text-primary);
  line-height: 1.45;
  letter-spacing: 0.02em;
}
.ac-bubble :deep(h1:first-child),
.ac-bubble :deep(h2:first-child),
.ac-bubble :deep(h3:first-child),
.ac-bubble :deep(h4:first-child) { margin-top: 0; }
.ac-bubble :deep(h1) { font-size: 1.05rem; }
.ac-bubble :deep(h2) { font-size: 0.97rem; }
.ac-bubble :deep(h3) { font-size: 0.9rem; }
.ac-bubble :deep(h4) { font-size: 0.85rem; }
.ac-bubble :deep(h2),
.ac-bubble :deep(h3) {
  padding-bottom: 5px;
  border-bottom: 1px solid var(--color-border-light);
}

.ac-bubble :deep(ul),
.ac-bubble :deep(ol) { margin: 0.45em 0 0.75em; padding-left: 1.45em; }
.ac-bubble :deep(li) { margin: 4px 0; line-height: 1.7; }
.ac-bubble :deep(li::marker) { color: var(--color-text-placeholder); }
.ac-bubble :deep(li > ul),
.ac-bubble :deep(li > ol) { margin: 3px 0 3px; }

.ac-bubble :deep(blockquote) {
  margin: 0.6em 0;
  padding: 7px 13px;
  border-left: 2px solid var(--color-accent);
  background: var(--color-bg-secondary);
  border-radius: 0 var(--radius-sm) var(--radius-sm) 0;
  color: var(--color-text-secondary);
  font-size: 0.97em;
}
.ac-bubble :deep(blockquote p) { margin: 0; }

.ac-bubble :deep(code) {
  font-family: var(--font-mono);
  font-size: 0.82em;
  background: var(--color-bg-tertiary);
  border: 1px solid var(--color-border-light);
  border-radius: 4px;
  padding: 1px 5px;
}
.ac-bubble :deep(pre) {
  margin: 0.6em 0;
  padding: 11px 13px;
  background: var(--color-bg-secondary);
  border: 1px solid var(--color-border);
  border-radius: var(--radius);
  overflow-x: auto;
  line-height: 1.6;
}
.ac-bubble :deep(pre code) {
  background: none;
  border: none;
  padding: 0;
  font-size: 0.78rem;
  line-height: 1.65;
}

.ac-bubble :deep(table) {
  border-collapse: collapse;
  margin: 0.6em 0;
  width: 100%;
  font-size: 0.8rem;
}
.ac-bubble :deep(th),
.ac-bubble :deep(td) {
  border: 1px solid var(--color-border);
  padding: 6px 10px;
  text-align: left;
  line-height: 1.55;
}
.ac-bubble :deep(th) {
  background: var(--color-bg-secondary);
  font-weight: 600;
  color: var(--color-text-primary);
}
.ac-bubble :deep(tbody tr:nth-child(even)) {
  background: rgba(0, 0, 0, 0.015);
}

.ac-bubble :deep(hr) {
  border: none;
  border-top: 1px solid var(--color-border);
  margin: 0.8em 0;
}

.ac-bubble :deep(a) {
  color: var(--color-accent);
  text-decoration: none;
}
.ac-bubble :deep(a:hover) { text-decoration: underline; }

.ac-bubble :deep(strong) {
  font-weight: 600;
  color: var(--color-text-primary);
}
.ac-bubble :deep(img) { max-width: 100%; border-radius: var(--radius-sm); }

.ac-sources {
  margin-top: 6px;
  display: flex;
  flex-wrap: wrap;
  gap: 6px;
}

.ac-source-link {
  display: inline-flex;
  align-items: center;
  gap: 4px;
  font-size: 0.68rem;
  color: var(--color-accent);
  background: var(--color-accent-light);
  border-radius: var(--radius-full);
  padding: 2px 10px;
  cursor: pointer;
  transition: background-color var(--dur-fast) var(--ease), color var(--dur-fast) var(--ease);
}

.ac-source-link:hover {
  color: var(--color-accent-hover);
  background: #DDEBF7;
}

.ac-source-link.clickable {
  text-decoration: none;
}
.ac-source-link.clickable:hover {
  text-decoration: underline;
}

/* 研究文献来源：灰色弱高亮、不可点击、只作证据标签 */
.ac-source-tag {
  display: inline-flex;
  align-items: center;
  gap: 4px;
  font-size: 0.66rem;
  color: var(--color-text-secondary);
  background: var(--color-bg-tertiary);
  border: 1px solid var(--color-border);
  border-radius: var(--radius-sm);
  padding: 2px 8px;
  cursor: default;
  user-select: none;
  line-height: 1.4;
}

.ac-system-msg {
  text-align: center;
  font-size: 0.7rem;
  color: var(--color-text-tertiary);
  padding: 6px;
}

/* v19: 完整长报告下载卡片 */
.ac-report-card {
  margin-top: 10px;
  padding: 14px 16px;
  border: 1px solid var(--color-border);
  border-left: 3px solid var(--color-accent, #409eff);
  border-radius: var(--radius-sm, 8px);
  background: var(--color-bg-primary);
  box-shadow: 0 1px 6px rgba(0, 0, 0, 0.04);
  display: flex;
  flex-direction: column;
  gap: 8px;
  max-width: 520px;
}

.ac-report-head {
  display: flex;
  align-items: center;
  gap: 8px;
}

.ac-report-icon {
  font-size: 0.6rem;
  font-weight: 700;
  letter-spacing: 0.04em;
  color: var(--color-accent, #409eff);
  background: var(--color-accent-light, rgba(64, 158, 255, 0.08));
  border-radius: var(--radius-sm, 6px);
  padding: 2px 6px;
  flex-shrink: 0;
}

.ac-report-title {
  font-size: 0.82rem;
  font-weight: 600;
  color: var(--color-text-primary);
  line-height: 1.4;
}

.ac-report-meta {
  font-size: 0.68rem;
  color: var(--color-text-tertiary);
  letter-spacing: 0.02em;
}

.ac-report-sections {
  display: flex;
  flex-wrap: wrap;
  gap: 5px;
}

.ac-report-section {
  font-size: 0.64rem;
  color: var(--color-text-secondary);
  background: var(--color-bg-tertiary);
  border: 1px solid var(--color-border);
  border-radius: var(--radius-full, 999px);
  padding: 2px 9px;
}

.ac-report-download {
  align-self: flex-start;
  margin-top: 2px;
  font-size: 0.72rem;
  font-weight: 500;
  color: var(--color-accent, #409eff);
  background: transparent;
  border: 1px solid var(--color-accent-ring, rgba(64, 158, 255, 0.4));
  border-radius: var(--radius-sm, 8px);
  padding: 5px 14px;
  cursor: pointer;
  transition: background 0.18s ease, color 0.18s ease;
}

.ac-report-download:hover {
  background: var(--color-accent, #409eff);
  color: #fff;
}

/* v19.2: 预览 + 下载双按钮布局 */
.ac-report-buttons {
  display: flex;
  gap: 8px;
  align-items: center;
}

.ac-report-download.secondary {
  color: var(--color-text-secondary);
  border-color: var(--color-border);
}

.ac-report-download.secondary:hover {
  background: var(--color-bg-hover, #f5f5f5);
  color: var(--color-text-primary);
}

.ac-error {
  padding: 10px 12px;
  margin: 8px 0;
  border: 1px solid rgba(255, 77, 79, 0.35);
  border-radius: var(--radius);
  background: var(--color-error-light);
  color: var(--color-error);
  font-size: 0.78rem;
}

.ac-reconnecting {
  padding: 6px 12px;
  margin: 8px 0;
  text-align: center;
  font-size: 0.72rem;
  color: var(--color-warning);
  animation: pulse 1.2s var(--ease) infinite;
}

/* Input area */
.ac-input-area {
  border-top: 1px solid var(--color-border);
  padding: 8px 10px;
}

.ac-input-row {
  display: flex;
  gap: 6px;
  align-items: flex-end;
}

.ac-textarea {
  flex: 1;
  resize: none;
  padding: 6px 10px;
  background: var(--color-bg-primary);
  border: 1px solid var(--color-border);
  border-radius: var(--radius);
  color: var(--color-text-primary);
  font-family: var(--font-sans);
  font-size: 0.8rem;
  line-height: 1.4;
  outline: none;
  max-height: 80px;
  transition: border-color var(--dur-fast) var(--ease), box-shadow var(--dur-fast) var(--ease);
}

.ac-textarea:focus {
  border-color: var(--color-accent);
  box-shadow: var(--shadow-focus);
}

.ac-textarea:disabled {
  opacity: 0.5;
}

.ac-send-btn {
  padding: 6px 14px;
  background: var(--color-accent);
  color: #fff;
  border: none;
  border-radius: var(--radius);
  cursor: pointer;
  font-size: 0.78rem;
  flex-shrink: 0;
  transition: background-color var(--dur-fast) var(--ease), opacity var(--dur-fast) var(--ease);
}

.ac-send-btn:hover:not(:disabled) {
  background: var(--color-accent-hover);
}

.ac-send-btn:disabled {
  opacity: 0.4;
  cursor: not-allowed;
}

.ac-stop-btn {
  padding: 6px 14px;
  background: var(--color-error);
  color: #fff;
  border: none;
  border-radius: var(--radius);
  cursor: pointer;
  font-size: 0.78rem;
  flex-shrink: 0;
  transition: background-color var(--dur-fast) var(--ease);
}

.ac-stop-btn:hover {
  background: #E04345;
}

/* Skill selector 已移除（v17：功能未启用，按钮冗余） */

/* === B-140: 演化卡片 === */
.ac-evolution-card {
  background: var(--color-bg-primary);
  border: 1px solid var(--color-border);
  border-left: 3px solid var(--color-accent);
  border-radius: var(--radius);
  padding: 10px 14px;
  cursor: pointer;
  transition: border-color var(--dur-fast) var(--ease), background-color var(--dur-fast) var(--ease), box-shadow var(--dur-fast) var(--ease);
}

.ac-evolution-card:hover {
  border-color: var(--color-accent);
  background: var(--color-bg-secondary);
  box-shadow: var(--shadow-sm);
}

.ac-evolution-card-header {
  display: flex;
  align-items: center;
  gap: 6px;
  margin-bottom: 6px;
}

.ac-evolution-card-icon {
  font-size: 1.1rem;
}

.ac-evolution-card-title {
  font-size: 0.82rem;
  font-weight: 600;
  color: var(--color-text-primary);
  flex: 1;
}

.ac-evolution-card-expand {
  font-size: 0.65rem;
  color: var(--color-accent);
  white-space: nowrap;
}

.ac-evolution-card-summary {
  font-size: 0.72rem;
  color: var(--color-text-secondary);
  line-height: 1.5;
  overflow: hidden;
  display: -webkit-box;
  -webkit-line-clamp: 3;
  -webkit-box-orient: vertical;
}
</style>
