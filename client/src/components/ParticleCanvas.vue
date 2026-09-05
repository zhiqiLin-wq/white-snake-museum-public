<script setup lang="ts">
import { ref, onMounted, onUnmounted } from 'vue'

const canvas = ref<HTMLCanvasElement | null>(null)
interface Particle { x: number; y: number; radius: number; speedX: number; speedY: number; color: string }
let particles: Particle[] = []
let ctx: CanvasRenderingContext2D | null = null
let animId = 0

//  ——  + 
const PALETTE = [
  'hsla(120, 8%, 65%, 0.18)',   // 
  'hsla(160, 6%, 62%, 0.15)',   // 
  'hsla(40, 25%, 70%, 0.14)',   // 
  'hsla(12, 30%, 62%, 0.12)',   // 
  'hsla(90, 5%, 68%, 0.14)',    // 
]

function resize() {
  if (!canvas.value) return
  canvas.value.width = window.innerWidth
  canvas.value.height = window.innerHeight
}

function initParticles() {
  if (!canvas.value) return
  particles = Array.from({ length: 80 }, () => ({
    x: Math.random() * canvas.value!.width,
    y: Math.random() * canvas.value!.height,
    radius: Math.random() * 4 + 1.5,
    speedX: (Math.random() - 0.5) * 0.15,
    speedY: (Math.random() - 0.5) * 0.15,
    color: PALETTE[Math.floor(Math.random() * PALETTE.length)],
  }))
}

function draw() {
  if (!ctx || !canvas.value) return
  const w = canvas.value.width
  const h = canvas.value.height
  ctx.clearRect(0, 0, w, h)
  particles.forEach(p => {
    ctx!.beginPath()
    ctx!.arc(p.x, p.y, p.radius, 0, Math.PI * 2)
    ctx!.fillStyle = p.color
    ctx!.fill()
    p.x += p.speedX
    p.y += p.speedY
    if (p.x < 0) p.x = w; if (p.x > w) p.x = 0
    if (p.y < 0) p.y = h; if (p.y > h) p.y = 0
  })
  animId = requestAnimationFrame(draw)
}

function onResize() { resize(); initParticles() }

onMounted(() => {
  if (canvas.value) {
    ctx = canvas.value.getContext('2d')
    resize(); initParticles(); draw()
  }
  window.addEventListener('resize', onResize)
})

onUnmounted(() => {
  cancelAnimationFrame(animId)
  window.removeEventListener('resize', onResize)
})
</script>

<template>
  <canvas ref="canvas" id="particle-canvas"></canvas>
</template>

<style scoped>
#particle-canvas {
  position: fixed;
  top: 0; left: 0;
  width: 100%; height: 100%;
  pointer-events: none;
  z-index: 2;
  opacity: 0.7;
}
</style>
