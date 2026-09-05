<script setup lang="ts">
/**
 * v19: 报告文件库 — 每个用户专属的报告存储区
 * - 左侧：报告列表（服务端 agent/reports/{user_id}/ 目录）
 * - 右侧：平台内 Markdown 预览（不再只能下载）
 * - 支持 下载 .md / 删除
 * 从 AgentChat 报告卡片点"预览"进入时，tab data 携带 reportId 自动打开该报告
 */
import { ref, computed, onMounted, onBeforeUnmount, watch } from 'vue'
import { useWorkspaceStore } from '@/stores/workspace'
import { apiFetch } from '@/services/api'
import { marked } from 'marked'

marked.setOptions({ gfm: true, breaks: true })

const props = defineProps<{ tabId?: string }>()

const workspaceStore = useWorkspaceStore()

interface ReportItem {
  reportId: string
  title: string
  topic: string
  sectionCount: number
  createdAt: string
  finalized: boolean
}

const reports = ref<ReportItem[]>([])
const loading = ref(false)
const loadError = ref('')
const previewing = ref(false)
const previewLoading = ref(false)
const previewTitle = ref('')
const previewMeta = ref('')
const previewContent = ref('')

const tabData = computed(() => {
  const tab = workspaceStore.tabs.find(t => t.id === props.tabId)
  return (tab?.data || {}) as Record<string, unknown>
})

const autoReportId = computed(() => tabData.value.reportId as string | undefined)

// ---------- v19.2: 目录 / 预览可拖动分栏 ----------
const containerEl = ref<HTMLElement | null>(null)
const listWidth = ref(320) // 目录宽度 px，可拖动调节
const dragging = ref(false)

function onDividerDown(e: MouseEvent) {
  e.preventDefault()
  dragging.value = true
  document.addEventListener('mousemove', onDividerMove)
  document.addEventListener('mouseup', onDividerUp)
}

function onDividerMove(e: MouseEvent) {
  if (!dragging.value || !containerEl.value) return
  const rect = containerEl.value.getBoundingClientRect()
  const min = 200
  const max = Math.max(min + 80, rect.width * 0.6)
  listWidth.value = Math.min(Math.max(e.clientX - rect.left, min), max)
}

function onDividerUp() {
  dragging.value = false
  document.removeEventListener('mousemove', onDividerMove)
  document.removeEventListener('mouseup', onDividerUp)
}

onBeforeUnmount(() => {
  document.removeEventListener('mousemove', onDividerMove)
  document.removeEventListener('mouseup', onDividerUp)
})

async function loadList() {
  loading.value = true
  loadError.value = ''
  try {
    const resp = await apiFetch('/api/agent/reports')
    if (!resp.ok) {
      throw new Error(`HTTP ${resp.status}`)
    }
    const data = await resp.json() as { reports?: ReportItem[] }
    reports.value = data.reports || []
  } catch (e) {
    loadError.value = e instanceof Error ? e.message : '加载失败'
  } finally {
    loading.value = false
  }
}

async function openReport(reportId: string) {
  previewLoading.value = true
  previewing.value = true
  previewContent.value = ''
  previewTitle.value = '加载中...'
  previewMeta.value = ''
  try {
    const resp = await apiFetch(`/api/agent/reports/${encodeURIComponent(reportId)}`)
    if (!resp.ok) {
      const payload = await resp.json().catch(() => ({}))
      throw new Error((payload as { detail?: string }).detail || `HTTP ${resp.status}`)
    }
    const data = await resp.json() as {
      title?: string; createdAt?: string; sectionCount?: number; content?: string; totalChars?: number
    }
    previewTitle.value = data.title || reportId
    previewMeta.value = [
      data.createdAt ? `生成于 ${data.createdAt}` : '',
      data.sectionCount ? `${data.sectionCount} 个小节` : '',
      data.totalChars ? `约 ${Math.round(data.totalChars / 1000)} 千字` : '',
    ].filter(Boolean).join(' · ')
    previewContent.value = data.content || ''
  } catch (e) {
    previewTitle.value = '预览失败'
    previewContent.value = `无法读取报告：${e instanceof Error ? e.message : String(e)}`
    previewMeta.value = ''
  } finally {
    previewLoading.value = false
  }
}

// 与 AgentChat 的下载逻辑一致
function download(report: { title: string; content?: string; reportId: string }) {
  const doDownload = (content: string) => {
    const safeTitle = (report.title || '研究报告').replace(/[\\/:*?"<>|]/g, '_').slice(0, 60)
    const blob = new Blob([content], { type: 'text/markdown;charset=utf-8' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = `${safeTitle}.md`
    document.body.appendChild(a)
    a.click()
    document.body.removeChild(a)
    URL.revokeObjectURL(url)
  }
  if (previewing.value && report.content && report.content === previewContent.value) {
    doDownload(report.content)
    return
  }
  // 未预览的报告先拉全文再下载
  apiFetch(`/api/agent/reports/${encodeURIComponent(report.reportId)}`)
    .then(r => (r.ok ? r.json() : Promise.reject(new Error(`HTTP ${r.status}`))))
    .then((data: { content?: string }) => doDownload(data.content || ''))
    .catch(err => alert(`下载失败：${err instanceof Error ? err.message : err}`))
}

async function removeReport(item: ReportItem) {
  if (!confirm(`确定删除报告《${item.title}》吗？此操作不可恢复。`)) return
  try {
    const resp = await apiFetch(`/api/agent/reports/${encodeURIComponent(item.reportId)}`, { method: 'DELETE' })
    if (!resp.ok) throw new Error(`HTTP ${resp.status}`)
    // 若正在预览被删报告则关闭预览
    if (previewing.value && previewTitle.value === item.title) {
      previewing.value = false
      previewContent.value = ''
    }
    await loadList()
  } catch (e) {
    alert(`删除失败：${e instanceof Error ? e.message : e}`)
  }
}

// 全角标点归一化（与 AgentChat.renderSimpleMarkdown 相同策略）
function renderMarkdown(text: string): string {
  if (!text) return ''
  try {
    let normalized = text
      .replace(/\uFF0A/g, '*')
      .replace(/\uFF3F/g, '_')
      .replace(/\uFF40/g, '`')
      .replace(/\uFF5E/g, '~')
    normalized = normalized.replace(/\*\*([^*\n]{1,120}?)\*\*/g, '<strong>$1</strong>')
    return marked.parse(normalized) as string
  } catch {
    return text.replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/\n/g, '<br>')
  }
}

onMounted(async () => {
  await loadList()
  // 从报告卡片"预览"进入 → 自动打开该报告
  if (autoReportId.value) {
    openReport(autoReportId.value)
  }
})

// v19.2: tab 复用场景 —— 文件库 tab 已存在时再次点卡片"预览"，
// 组件不会重新挂载（onMounted 不触发），必须 watch reportId 变化打开新报告
watch(autoReportId, (rid) => {
  if (rid) {
    openReport(rid)
  }
})
</script>

<template>
  <div ref="containerEl" class="report-library" :class="{ dragging }" :style="{ '--rl-list-w': listWidth + 'px' }">
    <!-- 左侧：报告列表（宽度可拖动调节） -->
    <div class="rl-list" :style="{ width: listWidth + 'px' }">
      <div class="rl-list-head">
        <span class="rl-list-title">我的报告文件库</span>
        <button class="rl-refresh-btn" :disabled="loading" @click="loadList">
          {{ loading ? '加载中...' : '刷新' }}
        </button>
      </div>

      <div v-if="loadError" class="rl-error">{{ loadError }}</div>

      <div v-if="!loading && reports.length === 0 && !loadError" class="rl-empty">
        <div class="rl-empty-icon">MD</div>
        <p>暂无研究报告</p>
        <p class="rl-empty-hint">在 AI 助手中提出"完整整理/系统梳理"类问题，<br>生成的完整报告会自动保存到这里</p>
      </div>

      <div
        v-for="item in reports"
        :key="item.reportId"
        class="rl-item"
        :class="{ active: previewing && previewTitle === item.title }"
        @click="openReport(item.reportId)"
      >
        <div class="rl-item-head">
          <span class="rl-item-icon">MD</span>
          <span class="rl-item-title">{{ item.title }}</span>
        </div>
        <div class="rl-item-meta">
          {{ item.createdAt }} · {{ item.sectionCount }} 节
          <span v-if="!item.finalized" class="rl-item-badge">未封卷</span>
        </div>
        <div class="rl-item-actions">
          <button class="rl-btn" @click.stop="openReport(item.reportId)">预览</button>
          <button class="rl-btn" @click.stop="download(item)">下载</button>
          <button class="rl-btn danger" @click.stop="removeReport(item)">删除</button>
        </div>
      </div>
    </div>

    <!-- 中间：可拖动分隔条（细线，hover 高亮） -->
    <div
      class="rl-divider"
      title="拖动调整目录宽度"
      @mousedown="onDividerDown"
    />

    <!-- 右侧：预览区 -->
    <div class="rl-preview">
      <template v-if="previewing">
        <div class="rl-preview-head">
          <div class="rl-preview-info">
            <span class="rl-preview-title">{{ previewTitle }}</span>
            <span class="rl-preview-meta">{{ previewMeta }}</span>
          </div>
          <div class="rl-preview-actions">
            <button
              class="rl-btn"
              :disabled="previewLoading || !previewContent"
              @click="download({ title: previewTitle, content: previewContent, reportId: '' })"
            >下载 .md</button>
            <button class="rl-btn" @click="previewing = false">关闭</button>
          </div>
        </div>
        <div v-if="previewLoading" class="rl-preview-loading">正在加载报告全文...</div>
        <div v-else class="rl-preview-body" v-html="renderMarkdown(previewContent)" />
      </template>
      <div v-else class="rl-preview-placeholder">
        <p>点击左侧报告进行平台内预览</p>
      </div>
    </div>
  </div>
</template>

<style scoped>
.report-library {
  display: flex;
  height: 100%;
  overflow: hidden;
  background: var(--color-bg-primary, #fff);
}
.report-library.dragging {
  cursor: col-resize;
  user-select: none;
}
.report-library.dragging .rl-preview {
  pointer-events: none;
}

/* 左侧列表 */
.rl-list {
  width: 320px;
  flex-shrink: 0;
  overflow-y: auto;
  display: flex;
  flex-direction: column;
}

/* v19.2: 目录区细滚动条 */
.rl-list::-webkit-scrollbar {
  width: 4px;
}
.rl-list::-webkit-scrollbar-track {
  background: transparent;
}
.rl-list::-webkit-scrollbar-thumb {
  background: var(--color-border, #dfe3e8);
  border-radius: 2px;
}
.rl-list::-webkit-scrollbar-thumb:hover {
  background: var(--color-text-tertiary, #c3c9d1);
}
.rl-list {
  scrollbar-width: thin;
  scrollbar-color: var(--color-border, #dfe3e8) transparent;
}

/* v19.2: 可拖动分隔条 —— 视觉上是 1px 细线，热区 9px */
.rl-divider {
  position: relative;
  width: 1px;
  flex-shrink: 0;
  background: var(--color-border, #e5e7eb);
  cursor: col-resize;
  z-index: 2;
  transition: background-color var(--dur-fast, 120ms) var(--ease, cubic-bezier(.22,.61,.36,1));
}
.rl-divider::after {
  content: '';
  position: absolute;
  top: 0;
  left: -4px;
  right: -4px;
  height: 100%;
  z-index: 1;
}
.rl-divider:hover,
.report-library.dragging .rl-divider {
  background: var(--color-accent, #409eff);
}

.rl-list-head {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 12px 14px;
  border-bottom: 1px solid var(--color-border, #e5e7eb);
  position: sticky;
  top: 0;
  background: var(--color-bg-primary, #fff);
  z-index: 1;
}

.rl-list-title {
  font-size: 0.82rem;
  font-weight: 600;
  color: var(--color-text-primary, #1f2937);
}

.rl-refresh-btn {
  font-size: 0.68rem;
  color: var(--color-text-secondary, #6b7280);
  background: var(--color-bg-tertiary, #f5f5f5);
  border: 1px solid var(--color-border, #e5e7eb);
  border-radius: 6px;
  padding: 3px 10px;
  cursor: pointer;
  transition: border-color 0.15s ease, color 0.15s ease;
}
.rl-refresh-btn:hover:not(:disabled) {
  border-color: var(--color-accent, #409eff);
  color: var(--color-accent, #409eff);
}
.rl-refresh-btn:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}

.rl-error {
  margin: 10px 14px;
  padding: 8px 10px;
  font-size: 0.72rem;
  color: #c0392b;
  background: #fdf0ee;
  border: 1px solid #f2cfc9;
  border-radius: 6px;
}

.rl-empty {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 4px;
  padding: 48px 16px;
  text-align: center;
  color: var(--color-text-tertiary, #9ca3af);
  font-size: 0.75rem;
}
.rl-empty-icon {
  font-size: 0.6rem;
  font-weight: 700;
  letter-spacing: 0.04em;
  color: var(--color-accent, #409eff);
  background: var(--color-accent-light, rgba(64, 158, 255, 0.08));
  border-radius: 6px;
  padding: 4px 8px;
  margin-bottom: 4px;
}
.rl-empty-hint {
  font-size: 0.66rem;
  line-height: 1.7;
}

.rl-item {
  padding: 12px 14px;
  border-bottom: 1px solid var(--color-border-light, #f0f0f0);
  cursor: pointer;
  transition: background-color 0.15s ease;
}
.rl-item:hover {
  background: var(--color-bg-hover, #f7f9fb);
}
.rl-item.active {
  background: var(--color-accent-light, rgba(64, 158, 255, 0.06));
  border-left: 2px solid var(--color-accent, #409eff);
  padding-left: 12px;
}

.rl-item-head {
  display: flex;
  align-items: flex-start;
  gap: 7px;
}

.rl-item-icon {
  font-size: 0.56rem;
  font-weight: 700;
  color: var(--color-accent, #409eff);
  background: var(--color-accent-light, rgba(64, 158, 255, 0.08));
  border-radius: 4px;
  padding: 2px 5px;
  flex-shrink: 0;
  margin-top: 1px;
}

.rl-item-title {
  font-size: 0.76rem;
  font-weight: 500;
  color: var(--color-text-primary, #1f2937);
  line-height: 1.45;
  overflow: hidden;
  display: -webkit-box;
  -webkit-line-clamp: 2;
  -webkit-box-orient: vertical;
}

.rl-item-meta {
  margin-top: 4px;
  font-size: 0.64rem;
  color: var(--color-text-tertiary, #9ca3af);
}

.rl-item-badge {
  color: #c47f17;
  background: #fbf3e4;
  border-radius: 4px;
  padding: 1px 5px;
  margin-left: 4px;
}

.rl-item-actions {
  margin-top: 7px;
  display: flex;
  gap: 6px;
}

.rl-btn {
  font-size: 0.66rem;
  color: var(--color-text-secondary, #4b5563);
  background: var(--color-bg-tertiary, #f7f7f7);
  border: 1px solid var(--color-border, #e5e7eb);
  border-radius: 6px;
  padding: 3px 10px;
  cursor: pointer;
  transition: border-color 0.15s ease, color 0.15s ease, background 0.15s ease;
}
.rl-btn:hover:not(:disabled) {
  border-color: var(--color-accent, #409eff);
  color: var(--color-accent, #409eff);
}
.rl-btn:disabled {
  opacity: 0.4;
  cursor: not-allowed;
}
.rl-btn.danger:hover:not(:disabled) {
  border-color: #d9534f;
  color: #d9534f;
}

/* 右侧预览 */
.rl-preview {
  flex: 1;
  min-width: 0;
  display: flex;
  flex-direction: column;
  overflow: hidden;
}

.rl-preview-head {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
  padding: 12px 18px;
  border-bottom: 1px solid var(--color-border, #e5e7eb);
  flex-shrink: 0;
}

.rl-preview-info {
  min-width: 0;
  display: flex;
  flex-direction: column;
  gap: 2px;
}

.rl-preview-title {
  font-size: 0.88rem;
  font-weight: 600;
  color: var(--color-text-primary, #1f2937);
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.rl-preview-meta {
  font-size: 0.66rem;
  color: var(--color-text-tertiary, #9ca3af);
}

.rl-preview-actions {
  display: flex;
  gap: 6px;
  flex-shrink: 0;
}

.rl-preview-loading {
  padding: 40px;
  text-align: center;
  font-size: 0.75rem;
  color: var(--color-text-tertiary, #9ca3af);
}

.rl-preview-body {
  flex: 1;
  overflow-y: auto;
  padding: 20px 26px 48px;
}

.rl-preview-placeholder {
  flex: 1;
  display: flex;
  align-items: center;
  justify-content: center;
  color: var(--color-text-tertiary, #9ca3af);
  font-size: 0.76rem;
}

/* 预览正文 Markdown 排版（v-html 需 :deep 穿透） */
.rl-preview-body :deep(h1),
.rl-preview-body :deep(h2),
.rl-preview-body :deep(h3),
.rl-preview-body :deep(h4) {
  margin: 1.4em 0 0.6em;
  font-weight: 600;
  color: var(--color-text-primary, #1f2937);
  line-height: 1.5;
}
.rl-preview-body :deep(h1:first-child),
.rl-preview-body :deep(h2:first-child),
.rl-preview-body :deep(h3:first-child) { margin-top: 0; }
.rl-preview-body :deep(h1) { font-size: 1.25rem; padding-bottom: 8px; border-bottom: 1px solid var(--color-border, #e5e7eb); }
.rl-preview-body :deep(h2) { font-size: 1.05rem; padding-bottom: 6px; border-bottom: 1px solid var(--color-border-light, #f0f0f0); }
.rl-preview-body :deep(h3) { font-size: 0.94rem; }

.rl-preview-body :deep(p) { margin: 0 0 0.9em; font-size: 0.84rem; line-height: 1.9; color: var(--color-text-primary, #374151); text-align: justify; }

.rl-preview-body :deep(ul),
.rl-preview-body :deep(ol) { margin: 0.5em 0 1em; padding-left: 1.6em; }
.rl-preview-body :deep(li) { margin: 5px 0; line-height: 1.8; font-size: 0.84rem; }

.rl-preview-body :deep(blockquote) {
  margin: 0.8em 0;
  padding: 8px 14px;
  border-left: 2px solid var(--color-accent, #409eff);
  background: var(--color-bg-secondary, #fafbfc);
  color: var(--color-text-secondary, #6b7280);
  font-size: 0.82rem;
}
.rl-preview-body :deep(blockquote p) { margin: 0; }

.rl-preview-body :deep(strong) { font-weight: 600; color: var(--color-text-primary, #1f2937); }

.rl-preview-body :deep(code) {
  font-family: var(--font-mono, monospace);
  font-size: 0.8em;
  background: var(--color-bg-tertiary, #f5f5f5);
  border: 1px solid var(--color-border-light, #f0f0f0);
  border-radius: 4px;
  padding: 1px 5px;
}

.rl-preview-body :deep(table) {
  border-collapse: collapse;
  margin: 0.8em 0;
  width: 100%;
  font-size: 0.78rem;
}
.rl-preview-body :deep(th),
.rl-preview-body :deep(td) {
  border: 1px solid var(--color-border, #e5e7eb);
  padding: 7px 11px;
  text-align: left;
  line-height: 1.6;
}
.rl-preview-body :deep(th) {
  background: var(--color-bg-secondary, #fafbfc);
  font-weight: 600;
}

.rl-preview-body :deep(hr) {
  border: none;
  border-top: 1px solid var(--color-border, #e5e7eb);
  margin: 1.2em 0;
}

@media (max-width: 640px) {
  /* 小屏禁用拖动布局，固定比例 */
  .rl-list {
    width: 44% !important;
  }
  .rl-divider {
    display: none;
  }
  .rl-preview-body {
    padding: 14px 14px 40px;
  }
}
</style>
