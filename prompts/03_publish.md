# 发布指令（获得用户授权后使用）

用户已同意公开发布本项目时执行：

1. 检查仓库内容、许可证、模型使用条款、资产授权；确保无模型权重、秘钥、私人日志、用户名与本地路径泄露。
2. `python -m unittest discover -s tests -v`、`npm ci`、`npm run docs:build` 真实通过。
3. 经用户确认目标 GitHub 账号、仓库名称、公开/私有设置后，创建或关联仓库，推送 `main`；配置 Settings → Pages → GitHub Actions。
4. 检查 workflow 成功发布并在浏览器实际访问线上地址，确认首页、章节、图像、资源路径和搜索基本可用。
5. 返回实际 URL、commit、Action 结果、未覆盖项；没有权限就明确停在待授权状态。

**不能**仅因仓库里有 Pages workflow 就声称网站已经在线发布。
