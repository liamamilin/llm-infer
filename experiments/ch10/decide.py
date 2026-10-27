"""E10：把第 07—09 章的实测证据汇总成引擎选型表（不需要 GPU，只读 evidence/public/）。

运行（在仓库根目录）：
    python -m experiments.ch10.decide
"""
import json

from experiments.common.config import MODEL_LARGE_ID, MODEL_LARGE_REVISION, ROOT
from experiments.common.evidence import cli_command, write_run

PUB = ROOT / "evidence" / "public"


def load(name: str) -> dict:
    return json.loads((PUB / f"{name}.json").read_text())


def row(rows, c):
    return next(r for r in rows if r["concurrency"] == c)


def main() -> int:
    conc, q, f8, pc = (load(n) for n in ("ch09-concurrency", "ch07-quant_hf", "ch07-vllm_fp8", "ch08-prefix_cache"))
    hf, vl = conc["results"]["hf"], conc["results"]["vllm"]
    table = {
        "startup_seconds": {"transformers_server": hf["startup_seconds"], "vllm": vl["startup_seconds"]},
        "gpu_memory_mb_after_start": {"transformers_server": hf["gpu_mb_after_start"], "vllm": vl["gpu_mb_after_start"]},
        "single_request_e2e_p50_s": {"transformers_server": row(hf["rows"], 1)["e2e_p50_s"], "vllm": row(vl["rows"], 1)["e2e_p50_s"]},
        "tok_s_at_concurrency_1": {"transformers_server": row(hf["rows"], 1)["output_tokens_per_s"], "vllm": row(vl["rows"], 1)["output_tokens_per_s"]},
        "tok_s_at_concurrency_16": {"transformers_server": row(hf["rows"], 16)["output_tokens_per_s"], "vllm": row(vl["rows"], 16)["output_tokens_per_s"]},
        "ttft_p50_s_at_concurrency_16": {"transformers_server": row(hf["rows"], 16)["ttft_p50_s"], "vllm": row(vl["rows"], 16)["ttft_p50_s"]},
        "prefix_cache_speedup_x": round(pc["summary"]["off"]["p50_ms"] / pc["summary"]["on_hits"]["p50_ms"], 2),
        "weights_mb": {"bf16": q["results"]["bf16"]["weights_mb"], "nf4": q["results"]["nf4"]["weights_mb"],
                       "vllm_bf16_gib": f8["bf16"]["weights_gib"], "vllm_fp8_gib": f8["fp8"]["weights_gib"]},
    }
    for k, v in table.items():
        print(f"{k:32} {v}")
    assertions = {
        "all_source_evidence_passed": all(d["status"] == "PASS" for d in (conc, q, f8, pc)),
        "sources_ran_on_same_gpu": len({d.get("gpu") for d in (conc, q, f8, pc)}) == 1,
        "transformers_starts_faster": hf["startup_seconds"] < vl["startup_seconds"],
        "vllm_uses_more_gpu_memory_reserving_kv": vl["gpu_mb_after_start"] > 2 * hf["gpu_mb_after_start"],
        "vllm_wins_single_request_too": table["tok_s_at_concurrency_1"]["vllm"] > table["tok_s_at_concurrency_1"]["transformers_server"],
        "vllm_wins_big_at_16": table["tok_s_at_concurrency_16"]["vllm"] > 20 * table["tok_s_at_concurrency_16"]["transformers_server"],
    }
    status = "PASS" if all(assertions.values()) else "FAIL"
    path = write_run("ch10", "decide", status, assertions, {}, "none (reads evidence)", cli_command(),
                     runtime="summary", model_id=MODEL_LARGE_ID, model_revision=MODEL_LARGE_REVISION,
                     extra={"table": table, "sglang": "NOT_RUN: 未安装、未测试，本书不对其给出结论",
                            "gpu": conc.get("gpu")})
    print(f"{status} -> {path}")
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
