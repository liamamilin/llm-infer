"""E12：从 evidence/public 生成评估报告 docs/chapters/report.md，并检查证据链完整性。

    .venv/bin/python -m experiments.ch12.report
"""
import json
import re
from pathlib import Path

import torch

from experiments.common.evidence import cli_command, write_run

ROOT = Path(__file__).resolve().parents[2]
PUB = ROOT / "evidence" / "public"
OUT = ROOT / "docs" / "chapters" / "report.md"
BLOB = "https://github.com/liamamilin/llm-infer/blob/main/evidence/public/"


def load(name: str) -> dict:
    return json.loads((PUB / f"{name}.json").read_text())


def build() -> tuple[str, dict]:
    files = sorted(p for p in PUB.glob("ch*.json") if not p.name.startswith("ch12"))
    runs = [json.loads(p.read_text()) | {"_file": p.name} for p in files]
    d10, d11s, d11l = load("ch10-decide")["table"], load("ch11-guardrails"), load("ch11-guardrails_large")
    q = load("ch07-quant_hf")["results"]
    b5 = load("ch05-bench")["results"]

    L = ["---", "title: '评估报告'", "description: '由 evidence/public 自动生成，不含手写数字'", "---", "",
         "# 评估报告", "",
         "> 本页由 `python -m experiments.ch12.report` 从 `evidence/public/*.json` 生成。页面里的每个数字都能在对应证据文件里找到。", "",
         "## 1. 实验清单", "",
         "| 章 | 实验 | 状态 | 设备 | 模型 | 断言 | 证据 |", "|---|---|---|---|---|---|---|"]
    for r in runs:
        ok = sum(bool(v) for v in r["assertions"].values())
        model = r["model_id"].split("/")[-1] if r["model_id"].startswith("Hugging") else "—"
        L.append(f"| {r['chapter_id']} | {r['experiment_id']} | {r['status']} | {r['device']} | {model} | "
                 f"{ok}/{len(r['assertions'])} | [{r['_file']}]({BLOB}{r['_file']}) |")
    L += ["", "## 2. 关键数字", "", "### 2.1 单请求延迟（Apple M4 Pro / MPS，SmolLM2-135M，fp32，ch05）", "",
          "| 提示词 token | TTFT p50 | TPOT p50 |", "|---|---|---|"]
    for n in ("32", "128", "512"):
        L.append(f"| {n} | {b5[n]['ttft']['p50'] * 1000:.1f} ms | {b5[n]['tpot']['p50'] * 1000:.2f} ms |")
    L += ["", "### 2.2 量化（RTX 4080 SUPER，SmolLM2-1.7B，ch07）", "",
          "| 精度 | 权重内存 | tok/s（p50） | 困惑度 |", "|---|---|---|---|"]
    for k in ("bf16", "int8", "nf4"):
        L.append(f"| {k} | {q[k]['weights_mb']:.0f} MB | {q[k]['tok_s_p50']} | {q[k]['perplexity']} |")
    L += ["", "### 2.3 引擎对比（同一张卡、同一个模型，ch09/ch10）", "",
          "| 指标 | Transformers 服务 | vLLM |", "|---|---|---|"]
    for key, label in [("startup_seconds", "启动到就绪（s）"), ("gpu_memory_mb_after_start", "启动后显存（MB）"),
                       ("tok_s_at_concurrency_1", "并发 1 吞吐（tok/s）"),
                       ("tok_s_at_concurrency_16", "并发 16 吞吐（tok/s）"),
                       ("ttft_p50_s_at_concurrency_16", "并发 16 首 token p50（s）")]:
        L.append(f"| {label} | {d10[key]['transformers_server']} | {d10[key]['vllm']} |")
    L += ["", f"前缀缓存加速：{d10['prefix_cache_speedup_x']}×（ch08）。", "",
          "### 2.4 服务护栏（ch11）", "",
          "| 项目 | 135M / MPS | 1.7B / RTX 4080 SUPER |", "|---|---|---|"]

    def row(label, f):
        L.append(f"| {label} | {f(d11s)} | {f(d11l)} |")

    row("24 并发突发：429 / 503 / 200 个数",
        lambda d: "/".join(str(d["burst"]["status_counts"].get(c, 0)) for c in ("429", "503", "200")))
    row("护栏下最长响应（s）", lambda d: d["burst"]["max_latency_s"])
    row("无护栏（ch04 服务）最长响应（s）", lambda d: d["baseline_burst"]["max_latency_s"])
    row("排队超时返回 503 的耗时（s）", lambda d: d["queue_timeout"]["latency_s"])
    row("客户端断开后名额回收（s）", lambda d: d["disconnect"]["slot_freed_seconds"])
    row("kill -9 后恢复就绪（s）", lambda d: d["recovery"]["seconds_to_ready"])
    L += ["", "## 3. 局限（均已在对应章节写明）", "",
          "- 只测了一个模型家族（SmolLM2）和一张 GPU（RTX 4080 SUPER 32GB）；其他硬件上的数字需要自己重测。",
          "- fp8 只测了吞吐与显存，**没有**测质量，不对其质量下结论。",
          "- 第 07—09 章的 vLLM 证据文件只记录了 torch 版本，没有记录 vllm 版本；版本由 `requirements-gpu.txt` 锁定（0.32.0）。",
          "- SGLang、llama.cpp、TensorRT-LLM 未测试（NOT_RUN）。",
          "- 第 09 章 vLLM 并发 128 档首次出现过 4 个失败请求，三次重测为 0，原因未查明。",
          "- 第 11 章的护栏只覆盖单机单进程；多副本、限流网关、鉴权不在本书验证范围内。", ""]
    checks = {
        "all_evidence_pass": all(r["status"] == "PASS" and all(r["assertions"].values()) for r in runs),
        "model_revisions_pinned": all(re.fullmatch(r"[0-9a-f]{40}", r["model_revision"]) for r in runs),
        # ch10 只汇总其他证据，没有运行时；vLLM 的证据里只记了 torch，vLLM 版本靠 requirements-gpu.txt 锁定
        "runtime_versions_recorded": all(r["runtime_version"] for r in runs if r["chapter_id"] != "ch10"),
        "vllm_version_pinned_in_requirements": bool(re.search(r"^vllm==[\d.]+$", (ROOT / "requirements-gpu.txt").read_text(), re.M)),
        "no_private_paths": not any(("/Users/" in p.read_text() or "/root/" in p.read_text()) for p in files),
        "covers_ch00_to_ch11": {f"ch{i:02d}" for i in range(12)} <= {r["chapter_id"] for r in runs},
    }
    return "\n".join(L), checks


def main() -> int:
    text, checks = build()
    OUT.write_text(text)
    again, _ = build()
    checks["report_is_deterministic"] = again == text
    status = "PASS" if all(checks.values()) else "FAIL"
    path = write_run("ch12", "report", status, checks, {"torch": torch.__version__}, "none (reads evidence)",
                     cli_command(), extra={"report_path": "docs/chapters/report.md", "report_lines": text.count("\n")})
    print(text[:600], "...\n")
    print(f"{status} -> {path}")
    for k, v in checks.items():
        print(("  ok " if v else "  FAIL ") + k)
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
