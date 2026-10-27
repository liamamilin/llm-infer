"""E06：KV Cache 的大小公式、实测显存增长，以及有无缓存的解码耗时对照。

运行（在仓库根目录）：
    .venv/bin/python -m experiments.ch06.kv_cache
"""
import gc
import time

import torch
import transformers

from experiments.common.evidence import cli_command, write_run
from experiments.common.loader import load, sync

LENGTHS = (128, 512, 2048)
FILLER = "The quick brown fox jumps over the lazy dog. " * 400
NEW = 48


def mem(device: str) -> int:
    if device == "mps":
        return torch.mps.current_allocated_memory()
    if device == "cuda":
        return torch.cuda.memory_allocated()
    return 0


def cache_bytes(cache) -> int:
    return sum(t.numel() * t.element_size()
               for layer in cache.layers for t in (layer.keys, layer.values))


def main() -> int:
    gc.collect()
    base_mem = mem("mps") if torch.backends.mps.is_available() else 0
    tok, model, device = load()
    cfg = model.config
    head_dim = cfg.hidden_size // cfg.num_attention_heads
    bytes_el = 4  # float32
    per_token = 2 * cfg.num_hidden_layers * cfg.num_key_value_heads * head_dim * bytes_el
    weights_mem = mem(device) - base_mem
    param_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    print(f"per-token KV = 2 x {cfg.num_hidden_layers} layers x {cfg.num_key_value_heads} kv_heads x {head_dim} x {bytes_el}B = {per_token} B")
    print(f"weights: {param_bytes/1e6:.1f} MB (device allocated after load: {weights_mem/1e6:.1f} MB)")

    ids_all = tok(FILLER, return_tensors="pt")["input_ids"].to(device)
    rows = []
    with torch.no_grad():
        for n in LENGTHS:
            gc.collect()
            before = mem(device)
            out = model(input_ids=ids_all[:, :n], use_cache=True)
            sync(device)
            actual = cache_bytes(out.past_key_values)
            theory = per_token * n
            rows.append({"seq_len": n, "theory_bytes": theory, "actual_cache_bytes": actual,
                         "device_alloc_growth_bytes": mem(device) - before,
                         "full_logits_bytes": n * cfg.vocab_size * bytes_el})
            print(f"seq={n:>4}  theory={theory/1e6:7.2f} MB  actual={actual/1e6:7.2f} MB  "
                  f"device growth={(mem(device)-before)/1e6:7.2f} MB")
            del out
        mha_per_token = 2 * cfg.num_hidden_layers * cfg.num_attention_heads * head_dim * bytes_el

        # 有无 KV Cache 的解码对照（贪心，输出必须一致）
        prompt = ids_all[:, :512]
        def timed(use_cache: bool):
            t = time.perf_counter()
            o = model.generate(input_ids=prompt, max_new_tokens=NEW, min_new_tokens=NEW, do_sample=False,
                               use_cache=use_cache, pad_token_id=tok.eos_token_id)
            sync(device)
            return time.perf_counter() - t, o[0, 512:].tolist()
        timed(True)  # warmup
        t_on, out_on = timed(True)
        t_off, out_off = timed(False)
    print(f"generate {NEW} tokens after 512-token prompt: cache on {t_on:.2f}s, cache off {t_off:.2f}s "
          f"({t_off/t_on:.1f}x slower); outputs equal: {out_on == out_off}")

    assertions = {
        "kv_size_formula_matches_actual_tensors": all(r["theory_bytes"] == r["actual_cache_bytes"] for r in rows),
        "kv_size_linear_in_sequence_length": rows[2]["actual_cache_bytes"] == 4 * rows[1]["actual_cache_bytes"]
                                             and rows[1]["actual_cache_bytes"] == 4 * rows[0]["actual_cache_bytes"],
        "prefill_growth_is_cache_plus_full_logits": all(
            abs(r["device_alloc_growth_bytes"] - r["actual_cache_bytes"] - r["full_logits_bytes"])
            < 0.05 * r["device_alloc_growth_bytes"] for r in rows) if device != "cpu" else True,
        "gqa_cache_is_smaller_than_full_mha": mha_per_token == 3 * per_token,
        "cache_does_not_change_greedy_output": out_on == out_off,
        "cache_speeds_up_decode": t_off > t_on,
    }
    status = "PASS" if all(assertions.values()) else "FAIL"
    path = write_run("ch06", "kv_cache", status, assertions,
                     {"torch": torch.__version__, "transformers": transformers.__version__},
                     device, cli_command(),
                     extra={"dtype": "float32", "per_token_kv_bytes": per_token, "weights_bytes": param_bytes,
                            "rows": rows, "mha_per_token_bytes": mha_per_token,
                            "cache_on_seconds": round(t_on, 3), "cache_off_seconds": round(t_off, 3),
                            "new_tokens": NEW, "prompt_tokens": 512})
    print(f"{status} -> {path}")
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
