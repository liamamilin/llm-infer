"""E09：并发压测——本书自带的串行服务 vs vLLM（连续批处理），同一模型同一 GPU。

需要 NVIDIA GPU。运行：
    python -m experiments.ch09.concurrency
"""
import torch

from experiments.ch09.loadgen import sweep
from experiments.common.config import MODEL_LARGE_ID, MODEL_LARGE_REVISION
from experiments.common.evidence import cli_command, write_run
from experiments.common.servers import NAME, gpu_used_mb, launch, stop

PORT = 18900


def main() -> int:
    if not torch.cuda.is_available():
        print("SKIP: 需要 CUDA GPU")
        return 0
    results = {}
    for kind, levels in (("hf", (1, 2, 4, 8, 16)), ("vllm", (1, 2, 4, 8, 16, 32, 64, 128))):
        base_mem = gpu_used_mb()
        proc, startup = launch(kind, PORT)
        try:
            print(f"== {kind} ready in {startup:.0f}s, gpu used +{gpu_used_mb() - base_mem} MB")
            rows = sweep(f"http://127.0.0.1:{PORT}", NAME, levels=levels, per_level_factor=2 if kind == "hf" else 4)
            results[kind] = {"startup_seconds": round(startup, 1), "gpu_mb_after_start": gpu_used_mb() - base_mem, "rows": rows}
        finally:
            stop(proc)

    hf, vl = results["hf"]["rows"], results["vllm"]["rows"]
    tput = lambda rows, c: next(r["output_tokens_per_s"] for r in rows if r["concurrency"] == c)
    print(f"throughput @16: hf {tput(hf, 16)} tok/s vs vllm {tput(vl, 16)} tok/s")
    assertions = {
        "no_failures_up_to_64_concurrent": all(r["failures"] == 0 for k in results for r in results[k]["rows"]
                                               if r["concurrency"] <= 64),
        "failure_rate_below_2pct_everywhere": all(r["failures"] / r["requests"] < 0.02
                                                  for k in results for r in results[k]["rows"]),
        "serial_server_does_not_scale": tput(hf, 16) < 1.5 * tput(hf, 1),
        "vllm_scales_with_concurrency": tput(vl, 16) > 4 * tput(vl, 1),
        "vllm_beats_serial_server_at_16": tput(vl, 16) > 3 * tput(hf, 16),
    }
    status = "PASS" if all(assertions.values()) else "FAIL"
    path = write_run("ch09", "concurrency", status, assertions, {"torch": torch.__version__}, "cuda", cli_command(),
                     runtime="transformers(ch04 server) vs vllm", model_id=MODEL_LARGE_ID,
                     model_revision=MODEL_LARGE_REVISION,
                     extra={"conditions": {"dtype": "bfloat16", "output_tokens": 128, "ignore_eos": True,
                                           "decoding": "greedy", "streaming": True, "warmup": "4 requests",
                                           "requests_per_level": "max(8, concurrency x factor); factor hf=2 vllm=4",
                                           "units": "tokens/s aggregate output; seconds"},
                            "results": results})
    print(f"{status} -> {path}")
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
