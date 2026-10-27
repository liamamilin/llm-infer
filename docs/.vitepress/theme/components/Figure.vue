<script setup lang="ts">
import { computed } from 'vue'
const props = defineProps<{ name: string; caption?: string }>()
// 图由 scripts/make_figures.py 生成；内联渲染，文字颜色才会跟随亮/暗主题
const files = import.meta.glob('../figures/*.svg', { query: '?raw', import: 'default', eager: true }) as Record<string, string>
const svg = computed(() => files[`../figures/${props.name}.svg`] ?? '')
</script>

<template>
  <figure class="fig">
    <div class="fig-body" v-html="svg" />
    <figcaption v-if="caption">{{ caption }}</figcaption>
  </figure>
</template>

<style scoped>
.fig { margin: 20px 0; }
.fig-body { overflow-x: auto; }
.fig-body :deep(svg) { width: 100%; min-width: 520px; height: auto; display: block; }
figcaption { margin-top: 6px; font-size: 13px; color: var(--vp-c-text-2); text-align: center; }
</style>
