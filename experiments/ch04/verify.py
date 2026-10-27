"""E04：端到端验证——启动服务子进程，检查 /health、普通/流式一致、错误请求被拒绝、流式确实逐步到达。

运行（在仓库根目录）：
    .venv/bin/python -m experiments.ch04.verify
"""
import os
import socket
import subprocess
import sys
import time

import httpx
import torch
import transformers

from experiments.common.evidence import cli_command, write_run
from experiments.ch04.client import chat



def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


PORT = free_port()
BASE = f"http://127.0.0.1:{PORT}"
MSGS = [{"role": "user", "content": "Write a short story about a cat."}]


def main() -> int:
    env = {**os.environ, "PORT": str(PORT)}
    proc = subprocess.Popen([sys.executable, "-m", "experiments.ch04.server"], env=env,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    try:
        t0 = time.time()
        while True:
            try:
                h = httpx.get(f"{BASE}/health", timeout=2).json()
                if h["status"] == "ok":
                    break
            except Exception:
                pass
            if proc.poll() is not None or time.time() - t0 > 120:
                print(proc.stdout.read())
                raise RuntimeError("server failed to start")
            time.sleep(0.5)
        startup_s = time.time() - t0
        print(f"health: {h} (ready after {startup_s:.1f}s)")

        full = chat(BASE, MSGS, max_tokens=40)
        text_full = full["choices"][0]["message"]["content"]
        streamed, arrivals = chat(BASE, MSGS, stream=True, max_tokens=40)
        bad = httpx.post(f"{BASE}/v1/chat/completions", json={"messages": MSGS, "max_tokens": 0}, timeout=10)
        print(f"non-stream: {text_full!r} usage={full['usage']}")
        print(f"stream    : {streamed!r}  chunks={len(arrivals)} first={arrivals[0]:.3f}s last={arrivals[-1]:.3f}s")
        print(f"bad request status: {bad.status_code}")

        assertions = {
            "health_ok": h["status"] == "ok",
            "stream_equals_non_stream": streamed == text_full,
            "usage_counts_completion_tokens": full["usage"]["completion_tokens"] == len(arrivals),
            "stream_arrives_incrementally": len(arrivals) >= 20 and arrivals[0] < arrivals[-1] * 0.5,
            "invalid_request_rejected_422": bad.status_code == 422,
        }
        status = "PASS" if all(assertions.values()) else "FAIL"
        path = write_run("ch04", "http_stream", status, assertions,
                         {"torch": torch.__version__, "transformers": transformers.__version__},
                         h["device"], cli_command(),
                         extra={"startup_seconds": round(startup_s, 1), "chunks": len(arrivals),
                                "first_chunk_seconds": round(arrivals[0], 4),
                                "last_chunk_seconds": round(arrivals[-1], 4),
                                "non_stream_text": text_full, "usage": full["usage"]})
        print(f"{status} -> {path}")
        return 0 if status == "PASS" else 1
    finally:
        proc.terminate()
        proc.wait(timeout=10)


if __name__ == "__main__":
    raise SystemExit(main())
