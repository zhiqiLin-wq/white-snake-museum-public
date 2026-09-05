import { defineStore } from 'pinia'
import { ref, computed } from 'vue'

interface User {
  id: string
  username: string
  displayName: string
  createdAt: string
}

export const useAuthStore = defineStore('auth', () => {
  const user = ref<User | null>(null)
  const token = ref<string | null>(localStorage.getItem('auth_token'))
  const isLoading = ref(false)
  const error = ref<string | null>(null)

  const isAuthenticated = computed(() => !!token.value && !!user.value)

  async function login(username: string, password: string): Promise<boolean> {
    isLoading.value = true
    error.value = null
    try {
      const resp = await fetch('/api/auth/login', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ username, password }),
      })
      const data = await resp.json()
      if (!resp.ok) {
        error.value = data.detail || ''
        return false
      }
      token.value = data.token
      user.value = data.user
      localStorage.setItem('auth_token', data.token)
      return true
    } catch {
      error.value = ''
      return false
    } finally {
      isLoading.value = false
    }
  }

  async function register(username: string, password: string, displayName?: string): Promise<boolean> {
    isLoading.value = true
    error.value = null
    try {
      const body: Record<string, string> = { username, password }
      if (displayName) body.display_name = displayName
      const resp = await fetch('/api/auth/register', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
      })
      const data = await resp.json()
      if (!resp.ok) {
        if (resp.status === 409) {
          error.value = data.detail || '该用户名已被注册，请直接登录'
        } else {
          error.value = data.detail || '注册失败，请稍后重试'
        }
        return false
      }
      token.value = data.token
      user.value = data.user
      localStorage.setItem('auth_token', data.token)
      return true
    } catch {
      error.value = ''
      return false
    } finally {
      isLoading.value = false
    }
  }

  async function logout() {
    const currentToken = token.value
    token.value = null
    user.value = null
    localStorage.removeItem('auth_token')
    if (currentToken) {
      try {
        await fetch('/api/auth/logout', {
          method: 'POST',
          headers: { 'Authorization': `Bearer ${currentToken}` },
        })
      } catch {
        // fire-and-forget
      }
    }
  }

  async function fetchMe(): Promise<boolean> {
    if (!token.value) return false
    try {
      const resp = await fetch('/api/auth/me', {
        headers: { 'Authorization': `Bearer ${token.value}` },
      })
      if (!resp.ok) {
        token.value = null
        user.value = null
        localStorage.removeItem('auth_token')
        return false
      }
      user.value = await resp.json()
      return true
    } catch {
      return false
    }
  }

  async function fetchStats(): Promise<{ conversationCount: number; annotationCount: number; lastActive: string | null }> {
    if (!token.value) throw new Error('Not authenticated')
    const resp = await fetch('/api/auth/stats', {
      headers: { 'Authorization': `Bearer ${token.value}` },
    })
    if (!resp.ok) {
      throw new Error(`Failed to load stats: HTTP ${resp.status}`)
    }
    return await resp.json()
  }

  return {
    user,
    token,
    isLoading,
    error,
    isAuthenticated,
    login,
    register,
    logout,
    fetchMe,
    fetchStats,
  }
})
