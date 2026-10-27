# 实验与证据约定

## 最小路径

- 统一小模型候选（待实机核验）：`HuggingFaceTB/SmolLM2-135M-Instruct` 或类似 0.1B—1B 的公开自回归模型。模型许可、revision、使用要求由 Codex 实时核实。
- 先探测环境，再选模型与引擎；macOS Apple Silicon/CPU 可选 Transformers/llama.cpp；Linux NVIDIA 有条件再引入 vLLM/SGLang。
- 主线实验逐章存放在 `experiments/chXX/`，避免为每章反复下载大权重；缓存位置可配置，任何下载需先说明大小与费用。

## 结果记录

路径建议：`evidence/runs/YYYYMMDD-chXX-<runid>.json`（原始日志可附不含秘密的 txt）。使用以下状态：

- `NOT_RUN` 未运行；`PASS` 真实执行且断言通过；`FAIL` 执行失败；`SKIP` 可选任务且条件不满足；`BLOCKED` 外部条件阻断。
- 最少字段：chapter_id, experiment_id, status, timestamp, platform, model_id, model_revision, runtime, runtime_version, exact_command, assertions, artifact_paths, git_sha, notes。
- 必选实验只有 PASS 才算完成；可选硬件实验允许 SKIP，但必须可见说明。
- 性能报告必须有测试条件、请求/输出长度、并发、预热次数、测量次数、单位、结果分布与失败情况。

## 不允许

- 仅凭 API 请求成功就冒充 GPU 本地模型实测。
- 用 GitHub Action 上的无 GPU runner 伪造 CUDA 性能结果。
- 未完成模型 revision 锁定却声称“100%任何设备可复现”。
- 在教材里硬编码私人绝对路径、密码/密钥、未授权下载的文件。

## 简化质量闭环

开发时用单条命令跑章节实验 + 最小断言；网页构建成功，再人工目测重点页面。只保留能发现真实错误的检查，不另建复杂审批系统。
