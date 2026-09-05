<script setup lang="ts">
import { ref, computed } from 'vue'
import { useAuthStore } from '@/stores/auth'
import { useRouter } from 'vue-router'

const authStore = useAuthStore()
const router = useRouter()

const expanded = ref(false)
const stats = ref<{ conversationCount: number; annotationCount: number; lastActive: string | null } | null>(null)
const statsLoading = ref(false)
const logoutConfirming = ref(false)

const displayName = computed(() => {
  if (!authStore.user) return ''
  return authStore.user.displayName || authStore.user.username
})

const avatarLetter = computed(() => {
  const name = displayName.value
  return name ? name.charAt(0).toUpperCase() : '?'
})

const isAuth = computed(() => authStore.isAuthenticated)

function toggleExpanded() {
  expanded.value = !expanded.value
  if (expanded.value) {
    loadStats()
  }
}

function closeExpanded() {
  expanded.value = false
}

async function loadStats() {
  statsLoading.value = true
  try {
    const s = await authStore.fetchStats()
    stats.value = s
  } catch {
    stats.value = null
  } finally {
    statsLoading.value = false
  }
}

function formatLastActive(raw: string | null): string {
  if (!raw) return ''
  try {
    const d = new Date(raw + (raw.includes('T') ? '' : 'Z').replace(' ', 'T'))
    if (isNaN(d.getTime())) return raw
    const now = Date.now()
    const diff = now - d.getTime()
    if (diff < 60000) return ''
    if (diff < 3600000) return `${Math.floor(diff / 60000)} 分钟前`
    if (diff < 86400000) return `${Math.floor(diff / 3600000)} 小时前`
    return `${Math.floor(diff / 86400000)} 天前`
  } catch {
    return raw
  }
}

async function handleLogout() {
  if (!logoutConfirming.value) {
    logoutConfirming.value = true
    return
  }
  await authStore.logout()
  logoutConfirming.value = false
  expanded.value = false
  router.push('/login')
}

function cancelLogout() {
  logoutConfirming.value = false
}
</script>

<template>
  <div v-if="isAuth" class="user-info" :class="{ expanded }">
    <!-- User info button -->
    <button class="ui-avatar-btn" :title="displayName" @click="toggleExpanded">
      <span class="ui-avatar-letter">{{ avatarLetter }}</span>
      <span class="ui-avatar-name">{{ displayName }}</span>
      <span class="ui-avatar-arrow" :class="{ rotated: expanded }">&#9660;</span>
    </button>

    <!-- Expanded panel -->
    <Transition name="ui-slide-up">
      <div v-if="expanded" class="ui-panel">
        <div class="ui-panel-header">
          <div class="ui-panel-avatar">{{ avatarLetter }}</div>
          <div class="ui-panel-names">
            <div class="ui-panel-displayname">{{ displayName }}</div>
            <div class="ui-panel-username">@{{ authStore.user?.username }}</div>
          </div>
          <button class="ui-panel-close" @click="closeExpanded">&times;</button>
        </div>

        <!-- Stats -->
        <div class="ui-panel-stats">
          <div class="ui-stat-item">
            <span class="ui-stat-value">{{ statsLoading ? '...' : (stats?.conversationCount ?? '-') }}</span>
            <span class="ui-stat-label">对话</span>
          </div>
          <div class="ui-stat-item">
            <span class="ui-stat-value">{{ statsLoading ? '...' : (stats?.annotationCount ?? '-') }}</span>
            <span class="ui-stat-label">标注</span>
          </div>
          <div class="ui-stat-item" v-if="stats?.lastActive">
            <span class="ui-stat-value">{{ formatLastActive(stats.lastActive) }}</span>
            <span class="ui-stat-label">最近活跃</span>
          </div>
        </div>

        <!-- Logout -->
        <div class="ui-panel-actions">
          <button
            v-if="!logoutConfirming"
            class="ui-logout-btn"
            @click="handleLogout"
          >退出登录</button>
          <div v-else class="ui-logout-confirm">
            <span class="ui-confirm-text">确定退出?</span>
            <button class="ui-confirm-yes" @click="handleLogout">退出</button>
            <button class="ui-confirm-no" @click="cancelLogout">取消</button>
          </div>
        </div>
      </div>
    </Transition>
  </div>
</template>

<style scoped>
.user-info {
  position: relative;
}

/* ---------- User info button ---------- */
.ui-avatar-btn {
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
  transition:
    border-color var(--dur-fast, 120ms) var(--ease, cubic-bezier(.22,.61,.36,1)),
    background var(--dur-fast, 120ms) var(--ease, cubic-bezier(.22,.61,.36,1)),
    box-shadow var(--dur-fast, 120ms) var(--ease, cubic-bezier(.22,.61,.36,1));
  white-space: nowrap;
}
.ui-avatar-btn:hover {
  border-color: var(--color-accent, #5B9BD5);
  background: var(--color-bg-secondary, #F7F8FA);
  box-shadow: var(--shadow-sm, 0 1px 2px rgba(0, 0, 0, 0.04));
}
.ui-avatar-letter {
  display: flex;
  align-items: center;
  justify-content: center;
  width: 20px;
  height: 20px;
  border-radius: 50%;
  background: var(--color-accent, #5B9BD5);
  color: #FFFFFF;
  font-size: 0.65rem;
  font-weight: 600;
  flex-shrink: 0;
}
.ui-avatar-name {
  font-size: 0.78rem;
  font-weight: 500;
  color: var(--color-text-primary, #1A1A1A);
  max-width: 100px;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}
.ui-avatar-arrow {
  font-size: 0.5rem;
  color: var(--color-text-tertiary, #6B7280);
  transition: transform var(--dur, 220ms) var(--ease, cubic-bezier(.22,.61,.36,1));
  flex-shrink: 0;
}
.ui-avatar-arrow.rotated {
  transform: rotate(180deg);
}

/* ---------- Expanded panel ---------- */
.ui-panel {
  position: absolute;
  top: 100%;
  left: 0;
  margin-top: 6px;
  width: 232px;
  background: var(--color-bg-primary, #FFFFFF);
  border: 1px solid var(--color-border, #E5E7EB);
  border-radius: var(--radius-lg, 10px);
  box-shadow: var(--shadow-lg, 0 12px 28px rgba(0, 0, 0, 0.08));
  z-index: 50;
  overflow: hidden;
}
.ui-panel-header {
  display: flex;
  align-items: center;
  gap: 10px;
  padding: 14px;
  border-bottom: 1px solid var(--color-border-light, #F0F2F5);
  background: var(--color-bg-secondary, #F7F8FA);
}
.ui-panel-avatar {
  width: 36px;
  height: 36px;
  border-radius: 50%;
  background: var(--color-accent, #5B9BD5);
  color: #FFFFFF;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 0.9rem;
  font-weight: 600;
  flex-shrink: 0;
}
.ui-panel-names {
  flex: 1;
  min-width: 0;
}
.ui-panel-displayname {
  font-size: 0.85rem;
  font-weight: 600;
  color: var(--color-text-primary, #1A1A1A);
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}
.ui-panel-username {
  font-size: 0.7rem;
  color: var(--color-text-tertiary, #6B7280);
}
.ui-panel-close {
  background: none;
  border: none;
  font-size: 1.2rem;
  color: var(--color-text-tertiary, #6B7280);
  cursor: pointer;
  padding: 0 2px;
  line-height: 1;
  transition: color var(--dur-fast, 120ms) var(--ease, cubic-bezier(.22,.61,.36,1));
}
.ui-panel-close:hover {
  color: var(--color-text-primary, #1A1A1A);
}

/* ---------- Stats ---------- */
.ui-panel-stats {
  display: flex;
  gap: 0;
  padding: 12px 14px;
  border-bottom: 1px solid var(--color-border-light, #F0F2F5);
}
.ui-stat-item {
  flex: 1;
  text-align: center;
  display: flex;
  flex-direction: column;
  gap: 2px;
  position: relative;
}
.ui-stat-item:not(:last-child)::after {
  content: '';
  position: absolute;
  right: 0;
  top: 20%;
  height: 60%;
  width: 1px;
  background: var(--color-border-light, #F0F2F5);
}
.ui-stat-value {
  font-size: 0.95rem;
  font-weight: 600;
  color: var(--color-text-primary, #1A1A1A);
  font-variant-numeric: tabular-nums;
}
.ui-stat-label {
  font-size: 0.62rem;
  color: var(--color-text-tertiary, #6B7280);
  letter-spacing: 0.04em;
}

/* ---------- Actions ---------- */
.ui-panel-actions {
  padding: 10px 14px;
}
.ui-logout-btn {
  width: 100%;
  padding: 7px 0;
  font-size: 0.76rem;
  font-weight: 500;
  color: var(--color-text-secondary, #4B5563);
  background: var(--color-bg-tertiary, #F0F2F5);
  border: 1px solid var(--color-border, #E5E7EB);
  border-radius: var(--radius, 8px);
  cursor: pointer;
  transition:
    color var(--dur-fast, 120ms) var(--ease, cubic-bezier(.22,.61,.36,1)),
    border-color var(--dur-fast, 120ms) var(--ease, cubic-bezier(.22,.61,.36,1)),
    background var(--dur-fast, 120ms) var(--ease, cubic-bezier(.22,.61,.36,1));
}
.ui-logout-btn:hover {
  color: var(--color-error, #C89292);
  border-color: var(--color-error, #C89292);
  background: var(--color-error-light, #FBF0F0);
}

.ui-logout-confirm {
  display: flex;
  align-items: center;
  gap: 6px;
  flex-wrap: wrap;
}
.ui-confirm-text {
  font-size: 0.72rem;
  color: var(--color-text-secondary, #4B5563);
  width: 100%;
  text-align: center;
  margin-bottom: 2px;
}
.ui-confirm-yes {
  flex: 1;
  padding: 5px 0;
  font-size: 0.7rem;
  border: 1px solid var(--color-text-secondary, #4B5563);
  background: var(--color-text-secondary, #4B5563);
  color: #FFFFFF;
  border-radius: var(--radius-sm, 4px);
  cursor: pointer;
  transition:
    background var(--dur-fast, 120ms) var(--ease, cubic-bezier(.22,.61,.36,1)),
    border-color var(--dur-fast, 120ms) var(--ease, cubic-bezier(.22,.61,.36,1));
}
.ui-confirm-yes:hover {
  background: var(--color-text-primary, #1A1A1A);
  border-color: var(--color-text-primary, #1A1A1A);
}
.ui-confirm-no {
  flex: 1;
  padding: 5px 0;
  font-size: 0.7rem;
  border: 1px solid var(--color-border, #E5E7EB);
  background: var(--color-bg-tertiary, #F0F2F5);
  color: var(--color-text-secondary, #4B5563);
  border-radius: var(--radius-sm, 4px);
  cursor: pointer;
  transition:
    color var(--dur-fast, 120ms) var(--ease, cubic-bezier(.22,.61,.36,1)),
    border-color var(--dur-fast, 120ms) var(--ease, cubic-bezier(.22,.61,.36,1)),
    background var(--dur-fast, 120ms) var(--ease, cubic-bezier(.22,.61,.36,1));
}
.ui-confirm-no:hover {
  color: var(--color-accent, #5B9BD5);
  border-color: var(--color-accent, #5B9BD5);
  background: var(--color-bg-primary, #FFFFFF);
}

/* ---------- Transition ---------- */
.ui-slide-up-enter-active,
.ui-slide-up-leave-active {
  transition: opacity var(--dur, 220ms) var(--ease, cubic-bezier(.22,.61,.36,1)),
              transform var(--dur, 220ms) var(--ease, cubic-bezier(.22,.61,.36,1));
}
.ui-slide-up-enter-from,
.ui-slide-up-leave-to {
  opacity: 0;
  transform: translateY(-4px);
}
</style>
