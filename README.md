# 大模型 Inference 工程实践 · 交互教材工程

这是提供给 Codex 的 **v2 启动仓库**，不是已完成或已实测的教材。与原来纯 Markdown 文档包相比，这一版增加了在线网站脚手架、交互演示、可复用技能、实战与证据规范、自动部署工作流。

**第一件事：** 打开 [`START_HERE.md`](START_HERE.md)，把 [`prompts/01_kickoff.md`](prompts/01_kickoff.md) 交给 Codex。

## 目标

一条小模型贯穿的实践主线，最终做到模型资产识别、运行、服务、测量、优化、部署，并在 GitHub Pages 上提供面向读者的课程（静态网页，不托管模型推理本身）。

## 网站脚手架

Node.js 22+。当前 ZIP **不含 `package-lock.json`**，由有网络的 Codex 首轮执行并提交锁文件：

```bash
npm install
npm run docs:dev
npm run docs:build
```

之后 CI 使用 `npm ci`。当前制包环境无法联网获取 npm 包，所以这些命令**未在制包环境通过实测**；不能把脚手架当成已发布网站。

实验环境（Python 3.10+，Mac/Linux/CPU 均可）：

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python -m experiments.ch00.smoke
.venv/bin/python -m experiments.ch01.inspect_assets
.venv/bin/python -m experiments.ch02.trace          # 另有 ch03.sampling / ch04.verify / ch05.bench / ch06.kv_cache
```

第 07—09 章需要 NVIDIA GPU（依赖见 `requirements-gpu.txt`，用 `scripts/gpu_sync.sh` 同步到 GPU 机器）：

```bash
python -m experiments.ch07.quant_hf && python -m experiments.ch07.vllm_fp8
python -m experiments.ch08.prefix_cache && python -m experiments.ch09.concurrency
python -m experiments.ch10.decide   # 只读 evidence/public/，不需要 GPU
```

第 11、12 章：

```bash
.venv/bin/python -m experiments.ch11.verify       # 本机；GPU 上加 MODEL_SIZE=large DTYPE=bfloat16
.venv/bin/python -m experiments.ch12.report       # 重新生成评估报告
.venv/bin/python -m experiments.ch12.clean_room   # 克隆已提交的 HEAD，新建 venv，从零复现（需联网，约十几分钟）
```

Python 标准库预检：

```bash
python3 scripts/doctor.py
python3 -m unittest discover -s tests -v   # 有 .venv 时会同时跑实验端到端测试
```

## 重要约定

- 网站正文在 `docs/`；教材状态以实测证据为准。
- 实验在 `experiments/`；先从 `ch00` 开始，其他章节待 Codex 实装。
- 原版文档位于 `reference/v1/` 供参考，新版 `AGENTS.md` 和 `plan/` 优先。
- 公开发布及可能产生费用的行为需用户授权。
