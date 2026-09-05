import { useAuthStore } from '@/stores/auth'

export async function apiFetch(url: string, options: RequestInit = {}): Promise<Response> {
  const authStore = useAuthStore()
  const headers = new Headers(options.headers)

  if (authStore.token) {
    headers.set('Authorization', `Bearer ${authStore.token}`)
  }

  if (options.body && typeof options.body === 'string') {
    if (!headers.has('Content-Type')) {
      headers.set('Content-Type', 'application/json')
    }
  }

  const resp = await fetch(url, { ...options, headers })

  if (resp.status === 401) {
    await authStore.logout()
    window.location.href = '/login'
    throw new Error('Session expired')
  }

  return resp
}
