---
title: '评估报告'
description: '由 evidence/public 自动生成，不含手写数字'
---

# 评估报告

> 本页由 `python -m experiments.ch12.report` 从 `evidence/public/*.json` 生成。页面里的每个数字都能在对应证据文件里找到。

## 1. 实验清单

| 章 | 实验 | 状态 | 设备 | 模型 | 断言 | 证据 |
|---|---|---|---|---|---|---|
| ch00 | smoke | PASS | mps | SmolLM2-135M-Instruct | 2/2 | [ch00-smoke.json](https://github.com/liamamilin/llm-infer/blob/main/evidence/public/ch00-smoke.json) |
| ch01 | assets | PASS | none (no model loaded) | SmolLM2-135M-Instruct | 6/6 | [ch01-assets.json](https://github.com/liamamilin/llm-infer/blob/main/evidence/public/ch01-assets.json) |
| ch02 | trace | PASS | mps | SmolLM2-135M-Instruct | 4/4 | [ch02-trace.json](https://github.com/liamamilin/llm-infer/blob/main/evidence/public/ch02-trace.json) |
| ch03 | sampling | PASS | mps | SmolLM2-135M-Instruct | 6/6 | [ch03-sampling.json](https://github.com/liamamilin/llm-infer/blob/main/evidence/public/ch03-sampling.json) |
| ch04 | http_stream | PASS | mps | SmolLM2-135M-Instruct | 5/5 | [ch04-http_stream.json](https://github.com/liamamilin/llm-infer/blob/main/evidence/public/ch04-http_stream.json) |
| ch05 | bench | PASS | mps | SmolLM2-135M-Instruct | 4/4 | [ch05-bench.json](https://github.com/liamamilin/llm-infer/blob/main/evidence/public/ch05-bench.json) |
| ch06 | kv_cache | PASS | mps | SmolLM2-135M-Instruct | 6/6 | [ch06-kv_cache.json](https://github.com/liamamilin/llm-infer/blob/main/evidence/public/ch06-kv_cache.json) |
| ch07 | quant_hf | PASS | cuda | SmolLM2-1.7B-Instruct | 4/4 | [ch07-quant_hf.json](https://github.com/liamamilin/llm-infer/blob/main/evidence/public/ch07-quant_hf.json) |
| ch07 | vllm_fp8 | PASS | cuda | SmolLM2-1.7B-Instruct | 3/3 | [ch07-vllm_fp8.json](https://github.com/liamamilin/llm-infer/blob/main/evidence/public/ch07-vllm_fp8.json) |
| ch08 | prefix_cache | PASS | cuda | SmolLM2-1.7B-Instruct | 4/4 | [ch08-prefix_cache.json](https://github.com/liamamilin/llm-infer/blob/main/evidence/public/ch08-prefix_cache.json) |
| ch09 | concurrency | PASS | cuda | SmolLM2-1.7B-Instruct | 5/5 | [ch09-concurrency.json](https://github.com/liamamilin/llm-infer/blob/main/evidence/public/ch09-concurrency.json) |
| ch10 | decide | PASS | none (reads evidence) | SmolLM2-1.7B-Instruct | 6/6 | [ch10-decide.json](https://github.com/liamamilin/llm-infer/blob/main/evidence/public/ch10-decide.json) |
| ch11 | guardrails | PASS | mps | SmolLM2-135M-Instruct | 25/25 | [ch11-guardrails.json](https://github.com/liamamilin/llm-infer/blob/main/evidence/public/ch11-guardrails.json) |
| ch11 | guardrails_large | PASS | cuda | SmolLM2-1.7B-Instruct | 25/25 | [ch11-guardrails_large.json](https://github.com/liamamilin/llm-infer/blob/main/evidence/public/ch11-guardrails_large.json) |

## 2. 关键数字

### 2.1 单请求延迟（Apple M4 Pro / MPS，SmolLM2-135M，fp32，ch05）

| 提示词 token | TTFT p50 | TPOT p50 |
|---|---|---|
| 32 | 14.9 ms | 10.03 ms |
| 128 | 21.3 ms | 9.92 ms |
| 512 | 52.5 ms | 10.67 ms |

### 2.2 量化（RTX 4080 SUPER，SmolLM2-1.7B，ch07）

| 精度 | 权重内存 | tok/s（p50） | 困惑度 |
|---|---|---|---|
| bf16 | 3423 MB | 58.4 | 9.541 |
| int8 | 1812 MB | 12.9 | 9.698 |
| nf4 | 1007 MB | 42.1 | 10.569 |

### 2.3 引擎对比（同一张卡、同一个模型，ch09/ch10）

| 指标 | Transformers 服务 | vLLM |
|---|---|---|
| 启动到就绪（s） | 7.4 | 34.5 |
| 启动后显存（MB） | 4122 | 16296 |
| 并发 1 吞吐（tok/s） | 56.9 | 169.1 |
| 并发 16 吞吐（tok/s） | 42.0 | 2198.9 |
| 并发 16 首 token p50（s） | 44.789 | 0.073 |

前缀缓存加速：2.53×（ch08）。

### 2.4 服务护栏（ch11）

| 项目 | 135M / MPS | 1.7B / RTX 4080 SUPER |
|---|---|---|
| 24 并发突发：429 / 503 / 200 个数 | 20/1/3 | 20/2/2 |
| 护栏下最长响应（s） | 1.02 | 1.47 |
| 无护栏（ch04 服务）最长响应（s） | 8.07 | 18.29 |
| 排队超时返回 503 的耗时（s） | 1.01 | 1.01 |
| 客户端断开后名额回收（s） | 0.06 | 0.04 |
| kill -9 后恢复就绪（s） | 6.4 | 8.9 |

## 3. 局限（均已在对应章节写明）

- 只测了一个模型家族（SmolLM2）和一张 GPU（RTX 4080 SUPER 32GB）；其他硬件上的数字需要自己重测。
- fp8 只测了吞吐与显存，**没有**测质量，不对其质量下结论。
- 第 07—09 章的 vLLM 证据文件只记录了 torch 版本，没有记录 vllm 版本；版本由 `requirements-gpu.txt` 锁定（0.32.0）。
- SGLang、llama.cpp、TensorRT-LLM 未测试（NOT_RUN）。
- 第 09 章 vLLM 并发 128 档首次出现过 4 个失败请求，三次重测为 0，原因未查明。
- 第 11 章的护栏只覆盖单机单进程；多副本、限流网关、鉴权不在本书验证范围内。
