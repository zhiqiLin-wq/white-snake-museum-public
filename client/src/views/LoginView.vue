<script setup lang="ts">
import { ref } from 'vue'
import { useRouter, useRoute } from 'vue-router'
import { useAuthStore } from '@/stores/auth'

const router = useRouter()
const route = useRoute()
const authStore = useAuthStore()

const username = ref('')
const password = ref('')

async function handleLogin() {
  if (!username.value || !password.value) return
  const ok = await authStore.login(username.value, password.value)
  if (ok) {
    const redirect = (route.query.redirect as string) || '/'
    router.push(redirect)
  }
}
</script>

<template>
  <div class="auth-page">
    <div class="auth-deco" aria-hidden="true">
      <div class="deco-circle deco-1"></div>
      <div class="deco-circle deco-2"></div>
      <div class="deco-circle deco-3"></div>
    </div>

    <Transition name="pop" appear>
      <div class="auth-card">
        <div class="brand">
          <div class="brand-mark"></div>
          <div class="brand-text">
            <div class="brand-title">白蛇传</div>
            <div class="brand-sub">文脉全息博物馆</div>
          </div>
        </div>

        <h2 class="auth-heading">欢迎回来</h2>
        <p class="auth-hint">登录以继续你的研究之旅</p>

        <form class="auth-form" @submit.prevent="handleLogin">
          <label class="field">
            <span class="field-label">用户名</span>
            <input
              v-model="username"
              type="text"
              class="input"
              placeholder="请输入用户名"
              autocomplete="username"
              @keyup.enter="handleLogin"
            />
          </label>

          <label class="field">
            <span class="field-label">密码</span>
            <input
              v-model="password"
              type="password"
              class="input"
              placeholder="请输入密码"
              autocomplete="current-password"
              @keyup.enter="handleLogin"
            />
          </label>

          <button
            class="btn btn-primary btn-lg auth-submit"
            :disabled="authStore.isLoading"
            type="submit"
          >
            <span v-if="!authStore.isLoading">登 录</span>
            <span v-else class="loading-dots">登录中<span>.</span><span>.</span><span>.</span></span>
          </button>
        </form>

        <Transition name="fade" mode="out-in">
          <div v-if="authStore.error" key="err" class="alert alert-error">
            {{ authStore.error }}
          </div>
        </Transition>

        <div class="auth-foot">
          <span>还没有账号？</span>
          <router-link to="/register" class="t-link">注册新账号</router-link>
        </div>
      </div>
    </Transition>
  </div>
</template>

<style scoped>
.auth-page {
  position: relative;
  display: flex;
  align-items: center;
  justify-content: center;
  height: 100vh;
  background: var(--color-bg-primary);
  overflow: hidden;
}

/* 背景装饰：极淡的模糊圆，只做呼吸感，不抢 */
.auth-deco { position: absolute; inset: 0; pointer-events: none; }
.deco-circle {
  position: absolute;
  border-radius: 50%;
  filter: blur(80px);
  opacity: 0.5;
}
.deco-1 { width: 420px; height: 420px; background: #EEF4FB; left: -80px; top: -60px; }
.deco-2 { width: 360px; height: 360px; background: #F3F5F7; right: -60px; bottom: -40px; }
.deco-3 { width: 220px; height: 220px; background: #F0F6FA; right: 35%; top: 18%; opacity: 0.7; }

.auth-card {
  position: relative;
  width: 400px;
  padding: 36px 40px 32px;
  background: var(--color-bg-primary);
  border: 1px solid var(--color-border);
  border-radius: 16px;
  box-shadow: var(--shadow-lg);
}

.brand {
  display: flex;
  align-items: center;
  gap: 12px;
  margin-bottom: 28px;
}
.brand-mark {
  width: 36px;
  height: 36px;
  border-radius: 10px;
  background: linear-gradient(135deg, var(--color-accent) 0%, #8E9FD0 100%);
  position: relative;
  box-shadow: 0 4px 10px rgba(91,155,213,.22);
}
.brand-mark::after {
  content: '';
  position: absolute;
  inset: 9px;
  border-radius: 50%;
  background: rgba(255,255,255,.85);
}
.brand-text .brand-title {
  font-size: 1.05rem;
  font-weight: 600;
  color: var(--color-text-primary);
  letter-spacing: 0.02em;
}
.brand-text .brand-sub {
  font-size: 0.75rem;
  color: var(--color-text-tertiary);
  margin-top: 2px;
}

.auth-heading {
  font-size: 1.4rem;
  font-weight: 600;
  color: var(--color-text-primary);
  letter-spacing: -0.02em;
}
.auth-hint {
  margin-top: 4px;
  margin-bottom: 24px;
  font-size: 0.82rem;
  color: var(--color-text-tertiary);
}

.auth-form { display: flex; flex-direction: column; gap: 16px; }
.field { display: flex; flex-direction: column; gap: 6px; }
.field-label {
  font-size: 0.78rem;
  font-weight: 500;
  color: var(--color-text-secondary);
}

.auth-submit {
  width: 100%;
  margin-top: 8px;
}
.loading-dots span {
  display: inline-block;
  animation: dot-bounce 1.2s infinite ease-in-out both;
}
.loading-dots span:nth-child(2) { animation-delay: -0.16s; }
.loading-dots span:nth-child(3) { animation-delay: -0.32s; }
@keyframes dot-bounce {
  0%, 80%, 100% { opacity: 0.2; transform: translateY(0); }
  40% { opacity: 1; transform: translateY(-2px); }
}

.alert {
  margin-top: 14px;
  padding: 10px 14px;
  border-radius: var(--radius);
  font-size: 0.8rem;
  text-align: center;
}
.alert-error {
  background: var(--color-error-light);
  color: var(--color-error);
  border: 1px solid rgba(255,77,79,.15);
}

.auth-foot {
  margin-top: 22px;
  padding-top: 18px;
  border-top: 1px solid var(--color-border-light);
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 6px;
  font-size: 0.82rem;
  color: var(--color-text-tertiary);
}
</style>
