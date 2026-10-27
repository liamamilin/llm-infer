"""E08：Prefix Caching——共享长前缀 + 不同短问题，开/关缓存的延迟对比（vLLM）。

需要 NVIDIA GPU。运行：
    python -m experiments.ch08.prefix_cache
"""
import statistics as st

import torch

from experiments.ch07.vllm_fp8 import run_worker
from experiments.common.config import MODEL_LARGE_ID, MODEL_LARGE_REVISION
from experiments.common.evidence import cli_command, write_run


def main() -> int:
    if not torch.cuda.is_available():
        print("SKIP: 需要 CUDA GPU")
        return 0
    off, _ = run_worker({"workload": "prefix", "prefix_caching": False})
    on, _ = run_worker({"workload": "prefix", "prefix_caching": True})
    n = off["prompt_tokens"]
    lo, lon = off["latencies"], on["latencies"]
    summ = lambda v: {"p50_ms": round(st.median(v) * 1000, 1), "max_ms": round(max(v) * 1000, 1)}
    print(f"prompt ≈ {n} tokens, 24 requests sharing a long prefix")
    print("cache OFF all requests:", summ(lo))
    print("cache ON  first request (miss): %.1f ms" % (lon[0] * 1000), "| requests 2-24 (hit):", summ(lon[1:]))
    assertions = {
        "prompt_is_long": n >= 1500,
        "same_outputs_with_and_without_cache": off["first_texts"] == on["first_texts"],
        "first_request_with_cache_is_a_miss": lon[0] > 1.5 * st.median(lon[1:]),
        "cache_hits_faster_than_no_cache": st.median(lon[1:]) < 0.7 * st.median(lo),
    }
    status = "PASS" if all(assertions.values()) else "FAIL"
    path = write_run("ch08", "prefix_cache", status, assertions, {"torch": torch.__version__}, "cuda", cli_command(),
                     runtime="vllm", model_id=MODEL_LARGE_ID, model_revision=MODEL_LARGE_REVISION,
                     extra={"conditions": {"requests": 24, "max_tokens": 1, "sequential": True, "warmup": 3,
                                           "decoding": "greedy", "prompt_tokens": n},
                            "cache_off_ms": [round(x * 1000, 1) for x in lo],
                            "cache_on_ms": [round(x * 1000, 1) for x in lon],
                            "summary": {"off": summ(lo), "on_first": round(lon[0] * 1000, 1), "on_hits": summ(lon[1:])}})
    print(f"{status} -> {path}")
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
