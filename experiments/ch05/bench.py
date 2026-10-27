"""E05：公平地测 TTFT / TPOT / tokens/s / 端到端延迟。

方法：固定输入长度与输出长度（忽略 EOS），先预热，再重复测量，报告分布而不是单次。
运行（在仓库根目录）：
    .venv/bin/python -m experiments.ch05.bench
"""
import statistics as st
import time

import torch
import transformers

from experiments.common.engine import Engine
from experiments.common.evidence import cli_command, write_run
from experiments.common.loader import sync

PROMPT_LENS = (32, 128, 512)
OUT_TOKENS = 64
WARMUP = 3
REPEATS = 10
FILLER = "The quick brown fox jumps over the lazy dog. " * 200


def make_ids(eng: Engine, n: int):
    ids = eng.tok(FILLER, return_tensors="pt")["input_ids"][:, :n]
    assert ids.shape[1] == n
    return ids.to(eng.device)


def one_run(eng: Engine, ids) -> dict:
    stamps = []
    t0 = time.perf_counter()
    for _ in eng.stream(ids, max_new_tokens=OUT_TOKENS, ignore_eos=True):
        sync(eng.device)
        stamps.append(time.perf_counter())
    e2e = stamps[-1] - t0
    ttft = stamps[0] - t0
    tpot = (stamps[-1] - stamps[0]) / (len(stamps) - 1)
    return {"tokens": len(stamps), "ttft": ttft, "tpot": tpot, "e2e": e2e, "tok_s_decode": 1 / tpot}


def summarize(vals):
    s = sorted(vals)
    return {"mean": round(st.mean(s), 5), "std": round(st.pstdev(s), 5), "p50": round(s[len(s) // 2], 5),
            "p95": round(s[min(len(s) - 1, int(len(s) * 0.95))], 5), "min": round(s[0], 5), "max": round(s[-1], 5)}


def main() -> int:
    eng = Engine()
    results, cold = {}, {}
    ok_len, ok_identity = True, True
    for n in PROMPT_LENS:
        ids = make_ids(eng, n)
        if not cold:  # 第一次调用：未预热的“冷”结果，单独记录
            cold = one_run(eng, ids)
        for _ in range(WARMUP):
            one_run(eng, ids)
        runs = [one_run(eng, ids) for _ in range(REPEATS)]
        ok_len &= all(r["tokens"] == OUT_TOKENS for r in runs)
        ok_identity &= all(abs(r["e2e"] - (r["ttft"] + (r["tokens"] - 1) * r["tpot"])) < 1e-6 for r in runs)
        results[str(n)] = {k: summarize([r[k] for r in runs]) for k in ("ttft", "tpot", "e2e", "tok_s_decode")}
        r = results[str(n)]
        print(f"prompt={n:>3} out={OUT_TOKENS}  TTFT p50={r['ttft']['p50']*1000:6.1f}ms  "
              f"TPOT p50={r['tpot']['p50']*1000:5.1f}ms  decode={r['tok_s_decode']['p50']:6.1f} tok/s  "
              f"e2e p50={r['e2e']['p50']:.3f}s p95={r['e2e']['p95']:.3f}s")
    print(f"cold first run (no warmup, prompt=32): TTFT={cold['ttft']*1000:.1f}ms e2e={cold['e2e']:.3f}s")

    assertions = {
        "every_run_produced_exact_token_count": bool(ok_len),
        "e2e_equals_ttft_plus_decode_time": bool(ok_identity),
        "ttft_grows_with_prompt_length": results["512"]["ttft"]["p50"] > results["32"]["ttft"]["p50"],
        "measured_repeats": REPEATS >= 10,
    }
    status = "PASS" if all(assertions.values()) else "FAIL"
    path = write_run("ch05", "bench", status, assertions,
                     {"torch": torch.__version__, "transformers": transformers.__version__},
                     eng.device, cli_command(),
                     extra={"conditions": {"dtype": "float32", "batch": 1, "concurrency": 1,
                                           "prompt_tokens": list(PROMPT_LENS), "output_tokens": OUT_TOKENS,
                                           "warmup": WARMUP, "repeats": REPEATS, "decoding": "greedy",
                                           "units": "seconds; tok_s_decode in tokens/second"},
                            "results": results, "cold_first_run": cold})
    print(f"{status} -> {path}")
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
