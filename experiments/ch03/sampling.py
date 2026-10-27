"""E03：生成参数对比——temperature / top_p / max_new_tokens / stop。

运行（在仓库根目录）：
    .venv/bin/python -m experiments.ch03.sampling
"""
import torch
import transformers

from experiments.common.evidence import cli_command, write_run
from experiments.common.loader import load

PROMPT = "Once upon a time, in a small village,"
NEW = 24


def main() -> int:
    tok, model, device = load()
    ids = tok(PROMPT, return_tensors="pt")["input_ids"].to(device)
    n = ids.shape[1]

    def gen(**kw):
        with torch.no_grad():
            out = model.generate(input_ids=ids, max_new_tokens=kw.pop("max_new_tokens", NEW),
                                 pad_token_id=tok.eos_token_id, tokenizer=tok, **kw)
        return tok.decode(out[0, n:], skip_special_tokens=True), out.shape[1] - n

    # 1) 一个固定的“下一个 token 分布”，看 temperature / top_p 如何改变它
    with torch.no_grad():
        logits = model(input_ids=ids).logits[0, -1].float()
    dist = []
    for temp in (0.2, 0.7, 1.0, 1.5):
        p = torch.softmax(logits / temp, dim=-1)
        sp, _ = torch.sort(p, descending=True)
        cum = torch.cumsum(sp, dim=0)
        dist.append({
            "temperature": temp,
            "top1_prob": round(float(sp[0]), 4),
            "entropy_nats": round(float(-(p * torch.log(p.clamp_min(1e-12))).sum()), 3),
            "tokens_to_cover_90pct": int((cum < 0.9).sum()) + 1,
        })
    for d in dist:
        print(d)

    # 2) 真实生成
    runs = {}
    runs["greedy"] = gen(do_sample=False)[0]
    runs["greedy_again"] = gen(do_sample=False)[0]
    for temp in (0.3, 0.8, 1.5):
        torch.manual_seed(7)
        runs[f"temp{temp}_seed7"] = gen(do_sample=True, temperature=temp, top_k=0)[0]
    torch.manual_seed(7)
    runs["temp0.8_seed7_again"] = gen(do_sample=True, temperature=0.8, top_k=0)[0]
    torch.manual_seed(8)
    runs["temp0.8_seed8"] = gen(do_sample=True, temperature=0.8, top_k=0)[0]
    torch.manual_seed(7)
    runs["temp0.8_top_p0.5_seed7"] = gen(do_sample=True, temperature=0.8, top_p=0.5, top_k=0)[0]
    short_text, short_n = gen(do_sample=False, max_new_tokens=5)
    stop_text, _ = gen(do_sample=False, stop_strings=["."])
    for k, v in runs.items():
        print(f"{k:24} {v!r}")
    print(f"max_new_tokens=5 -> {short_n} tokens: {short_text!r}")
    print(f"stop='.'         -> {stop_text!r}")

    assertions = {
        "greedy_is_deterministic": runs["greedy"] == runs["greedy_again"],
        "same_seed_same_output": runs["temp0.8_seed7"] == runs["temp0.8_seed7_again"],
        "different_seed_differs": runs["temp0.8_seed7"] != runs["temp0.8_seed8"],
        "higher_temperature_flattens_distribution": dist[0]["top1_prob"] > dist[-1]["top1_prob"]
                                                    and dist[0]["entropy_nats"] < dist[-1]["entropy_nats"],
        "max_new_tokens_respected": short_n == 5,
        "stop_string_ends_generation": stop_text.rstrip().endswith("."),
    }
    status = "PASS" if all(assertions.values()) else "FAIL"
    path = write_run("ch03", "sampling", status, assertions,
                     {"torch": torch.__version__, "transformers": transformers.__version__},
                     device, cli_command(),
                     extra={"prompt": PROMPT, "distribution": dist, "generations": runs,
                            "max_new_tokens_5": short_text, "stop_dot": stop_text})
    print(f"{status} -> {path}")
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
