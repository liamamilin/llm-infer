"""E00：加载固定 revision 的小模型，生成若干新 token，并写入证据。

运行（在仓库根目录）：
    .venv/bin/python -m experiments.ch00.smoke
"""
import time

import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

from experiments.common.config import MODEL_DIR, pick_device
from experiments.common.evidence import cli_command, write_run
from experiments.common.fetch_model import ensure_model

PROMPT = "The capital of France is"
NEW_TOKENS = 8


def main() -> int:
    ensure_model()
    device = pick_device()
    tok = AutoTokenizer.from_pretrained(MODEL_DIR)
    t0 = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(MODEL_DIR, dtype=torch.float32).to(device).eval()
    load_s = time.perf_counter() - t0

    inputs = tok(PROMPT, return_tensors="pt").to(device)
    n_in = inputs["input_ids"].shape[1]
    t1 = time.perf_counter()
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=NEW_TOKENS, do_sample=False)
    if device == "mps":
        torch.mps.synchronize()
    gen_s = time.perf_counter() - t1

    new_ids = out[0, n_in:]
    text = tok.decode(new_ids, skip_special_tokens=True)
    print(f"device={device} load={load_s:.2f}s generate={gen_s:.2f}s")
    print(f"prompt={PROMPT!r}\nnew_ids={new_ids.tolist()}\ncontinuation={text!r}")

    assertions = {
        "generated_at_least_1_token": len(new_ids) >= 1,
        "decoded_non_empty": len(text.strip()) > 0,
    }
    status = "PASS" if all(assertions.values()) else "FAIL"
    path = write_run(
        "ch00", "smoke", status, assertions,
        {"torch": torch.__version__, "transformers": transformers.__version__},
        device, cli_command(),
        extra={"prompt": PROMPT, "new_token_ids": new_ids.tolist(), "continuation": text,
               "load_seconds": round(load_s, 3), "generate_seconds": round(gen_s, 3)},
    )
    print(f"{status} -> {path}")
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
