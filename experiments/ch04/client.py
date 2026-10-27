"""E04：客户端——普通请求与流式请求，并测量首个 chunk 的到达时间。"""
import json
import time

import httpx


def chat(base: str, messages, stream: bool = False, **params):
    body = {"messages": messages, "stream": stream, **params}
    if not stream:
        return httpx.post(f"{base}/v1/chat/completions", json=body, timeout=120).json()
    pieces, arrivals = [], []
    t0 = time.perf_counter()
    with httpx.stream("POST", f"{base}/v1/chat/completions", json=body, timeout=120) as r:
        for line in r.iter_lines():
            if not line.startswith("data: "):
                continue
            data = line[6:]
            if data == "[DONE]":
                break
            pieces.append(json.loads(data)["choices"][0]["delta"]["content"])
            arrivals.append(time.perf_counter() - t0)
    return "".join(pieces), arrivals


if __name__ == "__main__":
    msgs = [{"role": "user", "content": "Say hello in one short sentence."}]
    text, arrivals = chat("http://127.0.0.1:8000", msgs, stream=True, max_tokens=32)
    print(text)
    print(f"first chunk after {arrivals[0]:.3f}s, last after {arrivals[-1]:.3f}s ({len(arrivals)} chunks)")
