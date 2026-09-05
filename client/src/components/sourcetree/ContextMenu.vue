<script setup lang="ts">
import { ref, watch, onUnmounted } from 'vue'

const props = defineProps<{
  visible: boolean
  x: number
  y: number
  targetType: 'chapter' | 'paragraph' | null
  chapters?: { number: number; dynasty: string; title: string }[]
  currentChapterNumber?: number
}>()

const emit = defineEmits<{
  action: [action: string, payload?: Record<string, unknown>]
  close: []
}>()

const showDynastySubmenu = ref(false)

function onKeydown(e: KeyboardEvent) {
  if (e.key === 'Escape') {
    emit('close')
  }
}

watch(() => props.visible, (val) => {
  if (val) {
    document.addEventListener('keydown', onKeydown)
  } else {
    document.removeEventListener('keydown', onKeydown)
  }
})

onUnmounted(() => {
  document.removeEventListener('keydown', onKeydown)
})

function handleAction(action: string, payload?: Record<string, unknown>) {
  emit('action', action, payload)
}

function handleDynastyCompare(targetChapterNumber: number) {
  emit('action', 'dynasty_compare', { targetChapterNumber })
}
</script>

<template>
  <Teleport to="body">
    <div
      v-if="visible"
      class="context-menu-backdrop"
      @click="emit('close')"
    />
    <div
      v-if="visible"
      class="context-menu"
      :style="{ left: x + 'px', top: y + 'px' }"
    >
      <template v-if="targetType === 'chapter'">
        <button class="cm-item" @click="handleAction('send_chapter')">
          发送章节到 Agent
        </button>
        <button class="cm-item" @click="handleAction('open_new_tab')">
          在新标签页打开
        </button>
        <button class="cm-item" @click="handleAction('split_compare')">
          分屏对比
        </button>

        <!-- A12: Dynasty compare submenu -->
        <div class="cm-item cm-submenu-trigger" @mouseenter="(showDynastySubmenu = true)" @mouseleave="showDynastySubmenu = false">
          跨朝代对比 &#9656;
          <div v-if="showDynastySubmenu" class="cm-submenu">
            <div
              v-for="ch in (chapters || []).filter(c => c.number !== currentChapterNumber)"
              :key="ch.number"
              class="cm-item"
              @click="handleDynastyCompare(ch.number)"
            >{{ ch.dynasty }} - {{ ch.title }}</div>
            <div v-if="(chapters || []).filter(c => c.number !== currentChapterNumber).length === 0" class="cm-item cm-item-disabled">
              无其他朝代可对比
            </div>
          </div>
        </div>
      </template>
      <template v-else-if="targetType === 'paragraph'">
        <button class="cm-item" @click="handleAction('send_paragraph')">
          发送段落到 Agent
        </button>
        <button class="cm-item" @click="handleAction('self_annotate')">
          标注此段落
        </button>
        <button class="cm-item" @click="handleAction('copy_paragraph')">
          复制段落文本
        </button>
        <button class="cm-item" @click="handleAction('add_to_compare')">
          加入对比
        </button>
      </template>
    </div>
  </Teleport>
</template>

<style scoped>
.context-menu-backdrop {
  position: fixed;
  inset: 0;
  z-index: 8000;
}

.context-menu {
  position: fixed;
  z-index: 8001;
  background: var(--color-bg-primary);
  border: 1px solid var(--color-border);
  border-radius: var(--radius);
  box-shadow: var(--shadow-md);
  min-width: 160px;
  padding: 4px 0;
}

.cm-item {
  display: block;
  width: 100%;
  text-align: left;
  padding: 7px 14px;
  background: none;
  border: none;
  color: var(--color-text-primary);
  font-family: var(--font-sans);
  font-size: 0.78rem;
  cursor: pointer;
  transition: background-color var(--dur-fast) var(--ease), color var(--dur-fast) var(--ease);
}

.cm-item:hover {
  background: var(--color-accent-light);
  color: var(--color-accent);
}

.cm-item-disabled {
  color: var(--color-text-tertiary);
  cursor: default;
}

.cm-item-disabled:hover {
  background: transparent;
  color: var(--color-text-tertiary);
}

.cm-submenu-trigger {
  position: relative;
}

.cm-submenu {
  position: absolute;
  left: 100%;
  top: -4px;
  background: var(--color-bg-primary);
  border: 1px solid var(--color-border);
  border-radius: var(--radius);
  box-shadow: var(--shadow-md);
  min-width: 180px;
  padding: 4px 0;
  max-height: 240px;
  overflow-y: auto;
}
</style>
