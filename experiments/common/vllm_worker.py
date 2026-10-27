"""在子进程里跑一次 vLLM 离线推理并输出 JSON（一个进程只用一种配置，避免显存/状态串扰）。

用法：python -m experiments.common.vllm_worker '<json 配置>'
配置字段：quantization, prefix_caching, workload("throughput"|"prefix")
"""
import json
import os
import sys
import time

# FlashInfer 采样内核需要现场用 nvcc 编译；本机 CUDA 工具链与 torch 不匹配，改用内置采样器。
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")

from vllm import LLM, SamplingParams

from experiments.common.config import MODEL_LARGE_DIR
from experiments.common.fetch_model import ensure_model

SYSTEM = ("You are a careful assistant for a company help desk. Follow the policy below. " +
          "Policy item: always answer concisely and politely; cite the policy number. " * 120)


def main(cfg: dict) -> dict:
    ensure_model("large")
    llm = LLM(model=str(MODEL_LARGE_DIR), dtype="bfloat16", max_model_len=4096,
              gpu_memory_utilization=0.5, quantization=cfg.get("quantization"),
              enable_prefix_caching=cfg.get("prefix_caching", False))
    tok = llm.get_tokenizer()
    out = {"cfg": cfg}
    if cfg["workload"] == "throughput":
        prompts = [f"Write a short paragraph about topic number {i}: the sea, trains, bread, or rain." for i in range(64)]
        sp = SamplingParams(temperature=0, max_tokens=128, ignore_eos=True)
        llm.generate(prompts[:8], sp)  # warmup
        t = time.perf_counter()
        res = llm.generate(prompts, sp)
        dt = time.perf_counter() - t
        out.update(seconds=round(dt, 3), output_tokens=sum(len(r.outputs[0].token_ids) for r in res),
                   tok_s=round(sum(len(r.outputs[0].token_ids) for r in res) / dt, 1),
                   texts=[r.outputs[0].text for r in res[:16]])
    else:  # prefix：共享长前缀 + 不同的短问题，每次只生成 1 个 token，近似测 TTFT
        sp = SamplingParams(temperature=0, max_tokens=1)
        questions = [f"Question {i}: what is the refund window for item {i}?" for i in range(24)]
        def run(q, system=SYSTEM):
            prompt = tok.apply_chat_template([{"role": "system", "content": system}, {"role": "user", "content": q}],
                                             tokenize=False, add_generation_prompt=True)
            t = time.perf_counter()
            r = llm.generate([prompt], sp)
            return time.perf_counter() - t, len(r[0].prompt_token_ids), r[0].outputs[0].text
        for _ in range(3):  # 预热内核：用另一段等长的系统提示，不会让正式前缀提前进入缓存
            run("warmup", system=SYSTEM.replace("careful", "unrelated").replace("policy", "rules"))
        rows = [run(q) for q in questions]
        out.update(prompt_tokens=rows[0][1], latencies=[round(r[0], 5) for r in rows], first_texts=[r[2] for r in rows])
    return out


if __name__ == "__main__":
    print("RESULT_JSON=" + json.dumps(main(json.loads(sys.argv[1])), ensure_ascii=False))
