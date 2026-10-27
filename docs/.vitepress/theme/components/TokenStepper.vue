<script setup lang="ts">
import { computed, ref } from 'vue'

// 纯教学模拟：例子和概率是手工构造的，并不是模型运行结果。
const prompt = '推理引擎负责'
const steps = [
  { token: '加载', candidates: [['加载', 0.57], ['预测', 0.28], ['保存', 0.15]] },
  { token: '模型', candidates: [['模型', 0.64], ['数据', 0.22], ['参数', 0.14]] },
  { token: '。', candidates: [['。', 0.74], ['并', 0.18], ['！', 0.08]] }
]
const position = ref(0)
const output = computed(() => prompt + steps.slice(0, position.value).map(s => s.token).join(''))
const current = computed(() => steps[Math.min(position.value, steps.length - 1)])
const done = computed(() => position.value >= steps.length)
function next() { if (!done.value) position.value += 1 }
function reset() { position.value = 0 }
</script>

<template>
  <section class="token-demo" aria-label="逐 Token 生成教学模拟">
    <div class="topline"><span class="eyebrow">TOKEN GENERATION</span><span class="tag">教学模拟 · 非真实模型预测</span></div>
    <h3>一次只选择下一个 Token</h3>
    <p class="note">每按一次“生成下一个”，就把演示 Token 接到已有序列后面。候选概率是人为设置的示例数据。</p>
    <div class="result" aria-live="polite">{{ output }}<span class="cursor" v-if="!done">▍</span></div>
    <div class="candidate" v-if="!done">
      <div class="candidate-label">第 {{ position + 1 }} 步 · 示例候选分布</div>
      <div class="bar" v-for="candidate in current.candidates" :key="candidate[0]">
        <span>{{ candidate[0] }}</span>
        <div class="track"><div class="fill" :style="{width: (candidate[1] * 100) + '%'}"></div></div>
        <span class="pct">{{ (candidate[1] * 100).toFixed(0) }}%</span>
      </div>
    </div>
    <div v-else class="complete">已完成三步演示。真实语言模型会继续预测，直到满足停止条件。</div>
    <div class="actions"><button type="button" :disabled="done" @click="next">生成下一个</button><button type="button" class="secondary" @click="reset">重新开始</button></div>
  </section>
</template>

<style scoped>
.token-demo { margin: 24px 0; padding: 24px; border: 1px solid var(--vp-c-divider); border-radius: 18px; background: var(--vp-c-bg-soft); box-shadow: 0 7px 23px rgba(20,50,70,.05); }
.topline { display:flex; justify-content:space-between; align-items:center; gap:10px; flex-wrap:wrap; }
.eyebrow { font-size: 11px; font-weight: 800; letter-spacing: .14em; color: var(--vp-c-brand-1); }
.tag { border:1px solid var(--vp-c-divider); padding:4px 9px; border-radius:999px; color:var(--vp-c-text-2); font-size:12px; }
h3 { margin: 14px 0 4px!important; font-size:21px!important; }
.note { color:var(--vp-c-text-2); font-size:14px; line-height:1.6; margin: 0 0 18px; }
.result { font-size:22px; line-height:1.6; font-weight:650; letter-spacing:.02em; padding:19px; min-height:82px; border:1px solid var(--vp-c-divider); border-radius:12px; background:var(--vp-c-bg); word-break: break-all; }
.cursor { color:var(--vp-c-brand-1); }
.candidate { margin-top:20px; }
.candidate-label {font-weight:650; font-size:13px; margin-bottom:10px; }
.bar {display:grid; grid-template-columns:52px minmax(0,1fr) 42px; gap:10px; align-items:center; margin:10px 0; font-size:13px; }
.track { height:10px; background:var(--vp-c-bg-alt); border-radius:20px; overflow:hidden; }
.fill { height:100%; border-radius:20px; background:linear-gradient(90deg,#3698a2,#6e91d7); }
.pct { text-align:right; color:var(--vp-c-text-2); font-variant-numeric:tabular-nums; }
.complete {margin-top:18px; color:var(--vp-c-text-2); font-size:14px; }
.actions { display:flex; gap:10px; margin-top:18px; flex-wrap:wrap; }
button {cursor:pointer; background:var(--vp-c-brand-1); color:var(--vp-c-white); border:0; padding:10px 16px; border-radius:10px; font-weight:650; }
button.secondary {background:transparent; color:var(--vp-c-text-1); border:1px solid var(--vp-c-divider); }
button:disabled {opacity:.4; cursor:not-allowed; }
@media(max-width:480px){.token-demo{padding:16px}.result{font-size:18px}.bar{grid-template-columns:42px minmax(0,1fr) 35px;gap:7px}}
</style>
