<script setup lang="ts">
import AgentChat from '@/components/agent/AgentChat.vue'
import UserInfo from '@/components/layout/UserInfo.vue'
import { useWorkspaceStore } from '@/stores/workspace'

const workspaceStore = useWorkspaceStore()

// v19.2: 报告文件库入口 —— 位于用户信息按钮旁
function openReportLibrary() {
  workspaceStore.openTab('report-library', '报告文件库', {})
}
</script>

<template>
  <div class="right-column">
    <div class="rc-user-bar">
      <button
        class="rc-library-btn"
        title="报告文件库：查看 / 预览 / 下载已生成的完整报告"
        @click="openReportLibrary"
      >
        <svg class="rc-library-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round">
          <path d="M3 7a2 2 0 0 1 2-2h4l2 2h8a2 2 0 0 1 2 2v8a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z" />
          <path d="M3 11h18" />
        </svg>
        <span class="rc-library-text">文件库</span>
      </button>
      <UserInfo />
    </div>
    <AgentChat />
  </div>
</template>

<style scoped>
.right-column {
  display: flex;
  flex-direction: column;
  height: 100%;
  overflow: hidden;
  background: var(--color-bg-secondary, #F7F8FA);
}

.rc-user-bar {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 8px;
  padding: 8px 12px;
  border-bottom: 1px solid var(--color-border, #E5E7EB);
  flex-shrink: 0;
  position: relative;
  background: var(--color-bg-secondary, #F7F8FA);
}

/* ---------- 文件库按钮（与用户信息按钮同风格） ---------- */
.rc-library-btn {
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 6px;
  height: 32px;
  padding: 0 10px;
  border-radius: var(--radius, 8px);
  border: 1px solid var(--color-border, #E5E7EB);
  background: var(--color-bg-primary, #FFFFFF);
  cursor: pointer;
  white-space: nowrap;
  transition:
    border-color var(--dur-fast, 120ms) var(--ease, cubic-bezier(.22,.61,.36,1)),
    background var(--dur-fast, 120ms) var(--ease, cubic-bezier(.22,.61,.36,1)),
    box-shadow var(--dur-fast, 120ms) var(--ease, cubic-bezier(.22,.61,.36,1)),
    color var(--dur-fast, 120ms) var(--ease, cubic-bezier(.22,.61,.36,1));
}
.rc-library-btn:hover {
  border-color: var(--color-accent, #5B9BD5);
  background: var(--color-bg-secondary, #F7F8FA);
  color: var(--color-accent, #5B9BD5);
  box-shadow: var(--shadow-sm, 0 1px 2px rgba(0, 0, 0, 0.04));
}
.rc-library-btn:active {
  transform: translateY(0.5px);
}
.rc-library-icon {
  width: 14px;
  height: 14px;
  flex-shrink: 0;
  color: var(--color-text-tertiary, #6B7280);
  transition: color var(--dur-fast, 120ms) var(--ease, cubic-bezier(.22,.61,.36,1));
}
.rc-library-btn:hover .rc-library-icon {
  color: var(--color-accent, #5B9BD5);
}
.rc-library-text {
  font-size: 0.78rem;
  font-weight: 500;
  color: var(--color-text-primary, #1A1A1A);
  transition: color var(--dur-fast, 120ms) var(--ease, cubic-bezier(.22,.61,.36,1));
}
.rc-library-btn:hover .rc-library-text {
  color: var(--color-accent, #5B9BD5);
}
</style>
