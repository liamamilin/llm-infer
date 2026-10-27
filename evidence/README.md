# 验证记录

`runs/` 是实验脚本每次运行的原始输出，默认被 git 忽略（可能含设备信息）。
确认无隐私信息后，把代表性的结果复制到 `public/`，章节正文只引用 `public/` 里的文件。

| 实验 | 状态 | 证据 |
|---|---|---|
| ch00 smoke | PASS（Apple M4 Pro / MPS） | [`public/ch00-smoke.json`](public/ch00-smoke.json) |
| ch01 assets | PASS | [`public/ch01-assets.json`](public/ch01-assets.json) |
| ch02 trace | PASS | [`public/ch02-trace.json`](public/ch02-trace.json) |
| ch03 sampling | PASS | [`public/ch03-sampling.json`](public/ch03-sampling.json) |
| ch04 http_stream | PASS | [`public/ch04-http_stream.json`](public/ch04-http_stream.json) |
| ch05 bench | PASS | [`public/ch05-bench.json`](public/ch05-bench.json) |
| ch06 kv_cache | PASS | [`public/ch06-kv_cache.json`](public/ch06-kv_cache.json) |
| ch07 quant_hf | PASS（RTX 4080 SUPER） | [`public/ch07-quant_hf.json`](public/ch07-quant_hf.json) |
| ch07 vllm_fp8 | PASS（RTX 4080 SUPER） | [`public/ch07-vllm_fp8.json`](public/ch07-vllm_fp8.json) |
| ch08 prefix_cache | PASS（RTX 4080 SUPER） | [`public/ch08-prefix_cache.json`](public/ch08-prefix_cache.json) |
| ch09 concurrency | PASS（RTX 4080 SUPER） | [`public/ch09-concurrency.json`](public/ch09-concurrency.json) |
| ch10 decide | PASS（汇总上述证据） | [`public/ch10-decide.json`](public/ch10-decide.json) |
| ch11 guardrails | PASS（Apple M4 Pro / MPS） | [`public/ch11-guardrails.json`](public/ch11-guardrails.json) |
| ch11 guardrails_large | PASS（RTX 4080 SUPER） | [`public/ch11-guardrails_large.json`](public/ch11-guardrails_large.json) |
| ch12 report | PASS（汇总全部证据） | [`public/ch12-report.json`](public/ch12-report.json) |
| ch12 clean_room | PASS（洁净室复现） | [`public/ch12-clean_room.json`](public/ch12-clean_room.json) |

SGLang 等未测试的引擎不在任何结论里。
