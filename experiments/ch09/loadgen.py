"""E09：并发压测客户端——对任意 OpenAI 兼容的 /v1/chat/completions 端点施压（流式，测 TTFT 与吞吐）。"""
import asyncio
import json
import statistics as st
import time

import httpx

PROMPT = "Write a short paragraph about the sea, trains, bread, or rain. Topic seed: {i}."


async def one(client, base, model, i, out_tokens):
    body = {"model": model, "messages": [{"role": "user", "content": PROMPT.format(i=i)}],
            "max_tokens": out_tokens, "ignore_eos": True, "temperature": 0, "stream": True}
    t0 = time.perf_counter()
    first, n = None, 0
    try:
        async with client.stream("POST", f"{base}/v1/chat/completions", json=body) as r:
            if r.status_code != 200:
                return {"ok": False, "error": f"HTTP {r.status_code}"}
            async for line in r.aiter_lines():
                if not line.startswith("data: ") or line.endswith("[DONE]"):
                    continue
                chunk = json.loads(line[6:])
                if chunk["choices"] and chunk["choices"][0]["delta"].get("content"):
                    n += 1
                    first = first or time.perf_counter()
    except Exception as e:
        return {"ok": False, "error": type(e).__name__}
    end = time.perf_counter()
    return {"ok": n > 0, "ttft": (first or end) - t0, "e2e": end - t0, "tokens": n}


async def run_level(base, model, concurrency, requests, out_tokens=128):
    sem = asyncio.Semaphore(concurrency)
    async with httpx.AsyncClient(timeout=600, limits=httpx.Limits(max_connections=concurrency + 4)) as client:
        async def guarded(i):
            async with sem:
                return await one(client, base, model, i, out_tokens)
        t = time.perf_counter()
        res = await asyncio.gather(*[guarded(i) for i in range(requests)])
        wall = time.perf_counter() - t
    ok = [r for r in res if r["ok"]]
    errors = {}
    for r in res:
        if not r["ok"]:
            errors[r.get("error", "empty response")] = errors.get(r.get("error", "empty response"), 0) + 1
    pct = lambda v, q: sorted(v)[min(len(v) - 1, int(len(v) * q))]
    return {"concurrency": concurrency, "requests": requests, "failures": len(res) - len(ok), "errors": errors,
            "output_tokens_per_s": round(sum(r["tokens"] for r in ok) / wall, 1),
            "ttft_p50_s": round(st.median(r["ttft"] for r in ok), 3), "ttft_p95_s": round(pct([r["ttft"] for r in ok], 0.95), 3),
            "e2e_p50_s": round(st.median(r["e2e"] for r in ok), 3), "e2e_p95_s": round(pct([r["e2e"] for r in ok], 0.95), 3),
            "wall_s": round(wall, 2)}


def sweep(base, model, levels=(1, 2, 4, 8, 16, 32, 64), per_level_factor=3, out_tokens=128):
    asyncio.run(run_level(base, model, 2, 4, out_tokens))  # 预热
    rows = []
    for c in levels:
        row = asyncio.run(run_level(base, model, c, max(8, c * per_level_factor), out_tokens))
        print(row)
        rows.append(row)
    return rows
