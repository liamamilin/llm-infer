import DefaultTheme from 'vitepress/theme'
import type { Theme } from 'vitepress'
import TokenStepper from './components/TokenStepper.vue'
import './custom.css'

export default {
  extends: DefaultTheme,
  enhanceApp({ app }) {
    app.component('TokenStepper', TokenStepper)
  }
} satisfies Theme
