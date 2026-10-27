"""E07a：同一个 1.7B 模型的 bf16 / int8 / nf4（bitsandbytes）对比：内存、解码速度、困惑度。

需要 NVIDIA GPU。运行（在仓库根目录）：
    HF_ENDPOINT=https://hf-mirror.com python -m experiments.ch07.quant_hf    # 国内网络用镜像
"""
import gc
import time

import pandas as pd
import torch
import transformers
from huggingface_hub import hf_hub_download
from transformers import BitsAndBytesConfig

from experiments.common.config import MODEL_LARGE_ID, MODEL_LARGE_REVISION
from experiments.common.evidence import cli_command, write_run
from experiments.common.loader import load

PPL_TOKENS = 1024 * 24
NEW = 128
PROMPT = "Explain how a transformer language model generates text, step by step."


def wikitext_ids(tok) -> torch.Tensor:
    path = hf_hub_download("Salesforce/wikitext", "wikitext-2-raw-v1/test-00000-of-00001.parquet",
                           repo_type="dataset")
    text = "\n\n".join(pd.read_parquet(path)["text"])
    return tok(text, return_tensors="pt")["input_ids"][:, :PPL_TOKENS]


@torch.no_grad()
def perplexity(model, ids, window=1024) -> float:
    nll, n = 0.0, 0
    for i in range(0, ids.shape[1] - window + 1, window):
        x = ids[:, i:i + window].to(model.device)
        loss = model(input_ids=x, labels=x).loss
        nll += float(loss) * (window - 1)
        n += window - 1
    return float(torch.exp(torch.tensor(nll / n)))


@torch.no_grad()
def decode_speed(model, tok) -> dict:
    x = tok(PROMPT, return_tensors="pt")["input_ids"].to(model.device)
    kw = dict(max_new_tokens=NEW, min_new_tokens=NEW, do_sample=False, pad_token_id=tok.eos_token_id)
    for _ in range(2):
        model.generate(input_ids=x, **kw)
    torch.cuda.reset_peak_memory_stats()
    times = []
    for _ in range(5):
        torch.cuda.synchronize()
        t = time.perf_counter()
        out = model.generate(input_ids=x, **kw)
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t)
    times.sort()
    return {"tok_s_p50": round(NEW / times[len(times) // 2], 1), "seconds_p50": round(times[len(times) // 2], 3),
            "peak_alloc_mb": round(torch.cuda.max_memory_allocated() / 1e6, 1),
            "sample": tok.decode(out[0, x.shape[1]:], skip_special_tokens=True)[:120]}


VARIANTS = {
    "bf16": {},
    "int8": {"quantization_config": BitsAndBytesConfig(load_in_8bit=True)},
    "nf4": {"quantization_config": BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)},
}


def main() -> int:
    if not torch.cuda.is_available():
        print("SKIP: 需要 CUDA GPU")
        return 0
    results = {}
    for name, kw in VARIANTS.items():
        gc.collect(); torch.cuda.empty_cache()
        t = time.perf_counter()
        tok, model, _ = load(dtype=torch.bfloat16, device="cuda", size="large", **kw)
        if name != "bf16":
            model.to  # 量化模型加载时已放到 GPU
        load_s = time.perf_counter() - t
        ids = wikitext_ids(tok)
        r = {"weights_mb": round(model.get_memory_footprint() / 1e6, 1), "load_seconds": round(load_s, 1),
             "perplexity": round(perplexity(model, ids), 3), **decode_speed(model, tok)}
        results[name] = r
        print(name, r)
        del model; gc.collect(); torch.cuda.empty_cache()

    base = results["bf16"]
    assertions = {
        "weights_shrink_bf16_gt_int8_gt_nf4": base["weights_mb"] > results["int8"]["weights_mb"] > results["nf4"]["weights_mb"],
        "perplexity_finite": all(0 < r["perplexity"] < 1e4 for r in results.values()),
        "quantized_ppl_within_25pct_of_bf16": all(r["perplexity"] < base["perplexity"] * 1.25 for r in results.values()),
        "all_variants_generate_text": all(len(r["sample"].strip()) > 0 for r in results.values()),
    }
    status = "PASS" if all(assertions.values()) else "FAIL"
    path = write_run("ch07", "quant_hf", status, assertions,
                     {"torch": torch.__version__, "transformers": transformers.__version__},
                     "cuda", cli_command(), runtime="transformers+bitsandbytes",
                     model_id=MODEL_LARGE_ID, model_revision=MODEL_LARGE_REVISION,
                     extra={"conditions": {"batch": 1, "new_tokens": NEW, "warmup": 2, "repeats": 5,
                                           "ppl": f"wikitext-2 test, first {PPL_TOKENS} tokens, window 1024"},
                            "results": results})
    print(f"{status} -> {path}")
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
