# GitHub Pages 发布路径

## 路径

1. 本地：`npm install`（初次生成并提交 `package-lock.json`）→ `npm run docs:build` → `npm run docs:dev` 预览。
2. 获取用户同意公开仓库以及可能暴露的教材内容、日志与图片素材；检查 `.gitignore` 和许可证。
3. 建立 GitHub 仓库并推送 `main`；在 **Settings → Pages → Build and deployment → Source** 设置 **GitHub Actions**。
4. `.github/workflows/pages.yml` 在推送 main 时安装锁定依赖、构建 `docs/.vitepress/dist` 并部署 Pages。
5. 验证 `https://<owner>.github.io/<repo>/`（项目站点）或 `<owner>.github.io`（用户站点）；实际访问并检查 CSS/静态资源/路由/图片正常。
6. 发布失败先查 Actions 日志、`VITEPRESS_BASE`、包锁与构建命令；不要重复无依据修改。

## 注意事项

- GitHub Pages 只承载静态书籍。实验要在读者本机或其自有/授权的 GPU 环境运行。
- 发布前必须有 `package-lock.json`；CI 使用 `npm ci`。由于当前打包环境无法访问 npm registry，锁文件必须在 Codex 的联网环境生成。
- 工作流自动按仓库名称设置项目站点的 base；自定义域名需要另调站点 base。
- `pages.yml` 默认只在主分支发布，PR 仅验证构建，发布活动需 GitHub Pages 权限配置。
- 不允许在仓库、网页源码或 Pages 公开任何 API 密钥。

官方参考：https://vitepress.dev/guide/deploy
