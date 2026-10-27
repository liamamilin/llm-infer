"""E07b：vLLM 的 bf16 与 fp8 权重量化对比：吞吐、与 bf16 的输出一致度、权重内存。

需要 NVIDIA GPU（Ada/Hopper 及以上支持 fp8）。运行：
    python -m experiments.ch07.vllm_fp8
"""
import json
import re
import subprocess
import sys

import torch

from experiments.common.config import MODEL_LARGE_ID, MODEL_LARGE_REVISION
from experiments.common.evidence import cli_command, write_run


def run_worker(cfg: dict) -> tuple[dict, float | None]:
    p = subprocess.run([sys.executable, "-m", "experiments.common.vllm_worker", json.dumps(cfg)],
                       capture_output=True, text=True)
    log = p.stdout + p.stderr
    line = next((l for l in p.stdout.splitlines() if l.startswith("RESULT_JSON=")), None)
    if line is None:
        print(log[-3000:])
        raise RuntimeError("worker failed")
    m = re.search(r"Model loading took ([\d.]+) ?GiB", log)
    return json.loads(line[len("RESULT_JSON="):]), (float(m.group(1)) if m else None)


def main() -> int:
    if not torch.cuda.is_available():
        print("SKIP: 需要 CUDA GPU")
        return 0
    bf, bf_gib = run_worker({"quantization": None, "workload": "throughput"})
    f8, f8_gib = run_worker({"quantization": "fp8", "workload": "throughput"})
    same_prefix = []
    for a, b in zip(bf["texts"], f8["texts"]):
        n = 0
        for x, y in zip(a, b):
            if x != y:
                break
            n += 1
        same_prefix.append(n / max(1, len(a)))
    agree = sum(same_prefix) / len(same_prefix)
    print(f"bf16: {bf['tok_s']} tok/s, weights {bf_gib} GiB | fp8: {f8['tok_s']} tok/s, weights {f8_gib} GiB | "
          f"identical-prefix fraction vs bf16: {agree:.2f}")
    assertions = {
        "both_generated_exact_token_count": bf["output_tokens"] == f8["output_tokens"] == 64 * 128,
        "fp8_weights_smaller": bf_gib is not None and f8_gib is not None and f8_gib < bf_gib,
        "fp8_outputs_are_text": all(len(t.strip()) > 20 for t in f8["texts"]),
    }
    status = "PASS" if all(assertions.values()) else "FAIL"
    path = write_run("ch07", "vllm_fp8", status, assertions, {"torch": torch.__version__}, "cuda", cli_command(),
                     runtime="vllm", model_id=MODEL_LARGE_ID, model_revision=MODEL_LARGE_REVISION,
                     extra={"conditions": {"prompts": 64, "output_tokens_each": 128, "decoding": "greedy",
                                           "max_model_len": 4096, "batch": "vLLM continuous batching, 64 requests at once"},
                            "bf16": {"tok_s": bf["tok_s"], "seconds": bf["seconds"], "weights_gib": bf_gib},
                            "fp8": {"tok_s": f8["tok_s"], "seconds": f8["seconds"], "weights_gib": f8_gib},
                            "identical_prefix_fraction_vs_bf16": round(agree, 3),
                            "bf16_sample": bf["texts"][0][:200], "fp8_sample": f8["texts"][0][:200]})
    print(f"{status} -> {path}")
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
