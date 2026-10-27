"""启动/停止被测服务进程（第 09、10 章共用）。"""
import os
import signal
import subprocess
import sys
import time

import httpx
import torch

from .config import MODEL_LARGE_DIR
from .fetch_model import ensure_model

NAME = "smollm2"


def launch(kind: str, port: int):
    """kind: "hf"（本书第 04 章的服务）或 "vllm"。返回 (Popen, 启动耗时秒数)。"""
    ensure_model("large")
    env = {**os.environ, "PORT": str(port), "MODEL_SIZE": "large", "DTYPE": "bfloat16",
           "VLLM_USE_FLASHINFER_SAMPLER": os.environ.get("VLLM_USE_FLASHINFER_SAMPLER", "0")}
    if kind == "hf":
        cmd = [sys.executable, "-m", "experiments.ch04.server"]
    elif kind == "vllm":
        cmd = [sys.executable, "-m", "vllm.entrypoints.openai.api_server", "--model", str(MODEL_LARGE_DIR),
               "--served-model-name", NAME, "--port", str(port), "--dtype", "bfloat16",
               "--max-model-len", "4096", "--gpu-memory-utilization", "0.5"]
    else:
        raise ValueError(kind)
    log = open(f"/tmp/server_{kind}_{port}.log", "w")
    t0 = time.time()
    proc = subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    while True:
        try:
            if httpx.get(f"http://127.0.0.1:{port}/health", timeout=2).status_code == 200:
                return proc, time.time() - t0
        except Exception:
            pass
        if proc.poll() is not None or time.time() - t0 > 600:
            raise RuntimeError(f"{kind} server failed; see /tmp/server_{kind}_{port}.log")
        time.sleep(1)


def stop(proc) -> None:
    os.killpg(proc.pid, signal.SIGTERM)
    try:
        proc.wait(timeout=30)
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid, signal.SIGKILL)
    time.sleep(3)


def gpu_used_mb() -> int:
    out = subprocess.check_output(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"], text=True)
    return int(out.split()[0])
