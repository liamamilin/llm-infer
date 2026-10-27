import { defineConfig } from 'vitepress'

export default defineConfig({
  lang: 'zh-CN',
  title: 'Inference 工程实践',
  description: '一条小模型主线，从模型文件到可用服务的可运行教材',
  base: process.env.VITEPRESS_BASE || '/',
  lastUpdated: true,
  cleanUrls: true,
  themeConfig: {
    siteTitle: 'Inference 工程实践',
    nav: [
      { text: '开始学习', link: '/guide/start' },
      { text: '课程目录', link: '/guide/roadmap' },
      { text: '可视化', link: '/guide/visuals' },
      { text: '术语表', link: '/guide/glossary' }
    ],
    sidebar: {
      '/': [
        { text: '阅读指南', items: [
          { text: '从这里开始', link: '/guide/start' },
          { text: '全书路线', link: '/guide/roadmap' },
          { text: '推理可视化', link: '/guide/visuals' },
          { text: '术语表', link: '/guide/glossary' }
        ] },
        { text: '实践章节', items: [
          { text: '00 · 环境与首次推理', link: '/chapters/ch00' },
          { text: '01 · 模型资产', link: '/chapters/ch01' },
          { text: '02 · 从 Token 到输出', link: '/chapters/ch02' },
          { text: '03 · 生成控制', link: '/chapters/ch03' },
          { text: '04 · HTTP 与流式', link: '/chapters/ch04' },
          { text: '05 · 基准测试', link: '/chapters/ch05' },
          { text: '06 · 内存与 KV Cache', link: '/chapters/ch06' },
          { text: '07 · 量化（GPU）', link: '/chapters/ch07' },
          { text: '08 · Prefix Caching（GPU）', link: '/chapters/ch08' },
          { text: '09 · Batching 与并发（GPU）', link: '/chapters/ch09' },
          { text: '10 · 引擎选型', link: '/chapters/ch10' }
        ] }
      ]
    },
    search: { provider: 'local' },
    socialLinks: [],
    outline: { label: '本页内容', level: [2, 3] },
    docFooter: { prev: '上一节', next: '下一节' },
    lastUpdated: { text: '最后更新' },
    footer: { message: '小模型 · 真执行 · 可复现', copyright: 'Inference Practice Book · 实验状态以实际证据为准' }
  }
})
