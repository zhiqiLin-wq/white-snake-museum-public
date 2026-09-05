import { ref } from 'vue'
import { SSEEventBus } from '@/services/sseEventBus'

export function useSSEConnection() {
  const isConnected = ref(false)
  const isReconnecting = ref(false)
  const lastEventTime = ref(0)

  const eventBus = SSEEventBus.getInstance()

  // Listen for connection state changes
  eventBus.on('*', () => {
    lastEventTime.value = Date.now()
  })

  async function connect(url: string, body: Record<string, unknown>): Promise<void> {
    try {
      isConnected.value = true
      isReconnecting.value = false
      await eventBus.startConnection(url, body)
    } catch {
      isConnected.value = false
    }
  }

  function disconnect() {
    eventBus.abort()
    isConnected.value = false
    isReconnecting.value = false
  }

  return {
    isConnected,
    isReconnecting,
    lastEventTime,
    connect,
    disconnect,
  }
}
