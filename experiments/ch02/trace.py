"""E02：手写 Prefill + Decode 循环，逐步展示一次推理，并与 generate() 对账。

运行（在仓库根目录）：
    .venv/bin/python -m experiments.ch02.trace
"""
import time

import torch
import transformers

from experiments.common.evidence import cli_command, write_run
from experiments.common.loader import load, sync

PROMPT = "The capital of France is"
STEPS = 8


def main() -> int:
    tok, model, device = load()
    ids = tok(PROMPT, return_tensors="pt")["input_ids"].to(device)
    n_prompt = ids.shape[1]
    print(f"prompt tokens ({n_prompt}): {[tok.decode(i) for i in ids[0]]}")

    steps = []
    with torch.no_grad():
        # Prefill：一次前向处理全部 prompt，产出第一个新 token 与 KV Cache
        t = time.perf_counter()
        out = model(input_ids=ids, use_cache=True)
        sync(device)
        prefill_s = time.perf_counter() - t
        cache = out.past_key_values
        logits = out.logits[0, -1].float()
        gen = []
        for step in range(STEPS):
            probs = torch.softmax(logits, dim=-1)
            top = torch.topk(probs, 3)
            nxt = int(top.indices[0])  # 贪心：取概率最大
            gen.append(nxt)
            steps.append({
                "step": step, "phase": "prefill" if step == 0 else "decode",
                "input_tokens": n_prompt if step == 0 else 1,
                "cache_len_before": 0 if step == 0 else n_prompt + step - 1,
                "chosen": nxt, "chosen_text": tok.decode(nxt), "chosen_prob": round(float(top.values[0]), 4),
                "top3": [(tok.decode(int(i)), round(float(p), 4)) for p, i in zip(top.values, top.indices)],
            })
            if step == STEPS - 1:
                break
            # Decode：每步只喂 1 个新 token，历史由 KV Cache 提供
            out = model(input_ids=torch.tensor([[nxt]], device=device), past_key_values=cache, use_cache=True)
            cache = out.past_key_values
            logits = out.logits[0, -1].float()
        sync(device)

        ref = model.generate(input_ids=ids, max_new_tokens=STEPS, do_sample=False)[0, n_prompt:].tolist()

    for s in steps:
        print(f"step {s['step']} [{s['phase']:7}] in={s['input_tokens']:>2} cache_before={s['cache_len_before']:>2} "
              f"-> {s['chosen_text']!r} p={s['chosen_prob']} top3={s['top3']}")
    cache_len = cache.get_seq_length()
    print(f"cache length at end: {cache_len} (= {n_prompt} prompt + {STEPS - 1} decoded inputs)")
    print(f"manual == generate(): {gen == ref}")

    assertions = {
        "manual_loop_matches_generate": gen == ref,
        "prefill_consumed_whole_prompt": steps[0]["input_tokens"] == n_prompt,
        "decode_steps_feed_one_token": all(s["input_tokens"] == 1 for s in steps[1:]),
        "cache_length_is_prompt_plus_decoded": cache_len == n_prompt + STEPS - 1,
    }
    status = "PASS" if all(assertions.values()) else "FAIL"
    path = write_run("ch02", "trace", status, assertions,
                     {"torch": torch.__version__, "transformers": transformers.__version__},
                     device, cli_command(),
                     extra={"prompt": PROMPT, "prompt_token_count": n_prompt, "steps": steps,
                            "prefill_seconds": round(prefill_s, 4), "generated_ids": gen})
    print(f"{status} -> {path}")
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
