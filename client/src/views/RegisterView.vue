<script setup lang="ts">
import { ref } from 'vue'
import { useRouter } from 'vue-router'
import { useAuthStore } from '@/stores/auth'

const router = useRouter()
const authStore = useAuthStore()

const username = ref('')
const displayName = ref('')
const password = ref('')
const confirmPassword = ref('')
const localError = ref('')

async function handleRegister() {
  localError.value = ''
  if (!username.value || !password.value) {
    localError.value = '请填写用户名和密码'
    return
  }
  if (username.value.length < 3 || username.value.length > 20) {
    localError.value = '用户名为3-20个字符'
    return
  }
  if (password.value.length < 8) {
    localError.value = '密码至少8个字符'
    return
  }
  if (password.value !== confirmPassword.value) {
    localError.value = '两次密码输入不一致'
    return
  }
  const ok = await authStore.register(username.value, password.value, displayName.value || undefined)
  if (ok) {
    router.push('/')
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

        <h2 class="auth-heading">创建账号</h2>
        <p class="auth-hint">加入研究社区，开启你的文脉探索</p>

        <form class="auth-form" @submit.prevent="handleRegister">
          <label class="field">
            <span class="field-label">用户名</span>
            <input
              v-model="username"
              type="text"
              class="input"
              placeholder="3-20位字母、数字或下划线"
              autocomplete="username"
            />
          </label>

          <label class="field">
            <span class="field-label">显示名称 <em>（可选）</em></span>
            <input
              v-model="displayName"
              type="text"
              class="input"
              placeholder="如何称呼你"
            />
          </label>

          <label class="field">
            <span class="field-label">密码</span>
            <input
              v-model="password"
              type="password"
              class="input"
              placeholder="至少8个字符"
              autocomplete="new-password"
            />
          </label>

          <label class="field">
            <span class="field-label">确认密码</span>
            <input
              v-model="confirmPassword"
              type="password"
              class="input"
              placeholder="再次输入密码"
              autocomplete="new-password"
            />
          </label>

          <button
            class="btn btn-primary btn-lg auth-submit"
            :disabled="authStore.isLoading"
            type="submit"
          >
            <span v-if="!authStore.isLoading">注 册</span>
            <span v-else class="loading-dots">注册中<span>.</span><span>.</span><span>.</span></span>
          </button>
        </form>

        <Transition name="fade" mode="out-in">
          <div v-if="localError || authStore.error" key="err" class="alert alert-error">
            {{ localError || authStore.error }}
          </div>
        </Transition>

        <div class="auth-foot">
          <span>已有账号？</span>
          <router-link to="/login" class="t-link">去登录</router-link>
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
  width: 420px;
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

.auth-form { display: flex; flex-direction: column; gap: 14px; }
.field { display: flex; flex-direction: column; gap: 6px; }
.field-label {
  font-size: 0.78rem;
  font-weight: 500;
  color: var(--color-text-secondary);
}
.field-label em {
  font-style: normal;
  color: var(--color-text-placeholder);
  font-weight: 400;
  margin-left: 4px;
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
