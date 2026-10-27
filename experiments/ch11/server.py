"""E11：在第 04 章服务之上加“生产护栏”：就绪探针、准入控制、排队/生成超时、结构化日志、指标。

    PORT=8000 MAX_INFLIGHT=4 QUEUE_TIMEOUT=1 MAX_SECONDS=2 .venv/bin/python -m experiments.ch11.server
"""
import json
import os
import threading
import time
import uuid
from contextlib import asynccontextmanager

import torch
import uvicorn
from fastapi import FastAPI, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse, StreamingResponse

from experiments.ch04.server import ChatRequest
from experiments.common.config import MODEL_ID, MODEL_LARGE_ID
from experiments.common.engine import Engine, QueueTimeout

MODEL_SIZE = os.environ.get("MODEL_SIZE", "small")
MODEL_NAME = MODEL_LARGE_ID if MODEL_SIZE == "large" else MODEL_ID
MAX_INFLIGHT = int(os.environ.get("MAX_INFLIGHT", "4"))      # 正在跑 + 正在排队的请求总数上限
QUEUE_TIMEOUT = float(os.environ.get("QUEUE_TIMEOUT", "5"))  # 排队等引擎的最长秒数
MAX_SECONDS = float(os.environ.get("MAX_SECONDS", "30"))     # 单个请求占用引擎的最长秒数

engine: Engine | None = None
ready = False
started = time.time()
state_lock = threading.Lock()
counters = {"requests": 0, "ok": 0, "rejected_429": 0, "queue_timeout_503": 0, "deadline_timeout": 0,
            "not_ready_503": 0, "errors_500": 0, "inflight": 0}


def log(**fields):
    print(json.dumps({"ts": round(time.time(), 3), **fields}, ensure_ascii=False), flush=True)


def bump(key: str, delta: int = 1):
    with state_lock:
        counters[key] += delta


def load_engine():
    global engine, ready
    dtype = getattr(torch, os.environ.get("DTYPE", "float32"))
    engine = Engine(dtype=dtype, size=MODEL_SIZE)
    engine.complete(engine.encode_chat([{"role": "user", "content": "hi"}]), max_new_tokens=4)  # 预热
    ready = True
    log(event="ready", startup_seconds=round(time.time() - started, 1), device=engine.device)


@asynccontextmanager
async def lifespan(_app):
    threading.Thread(target=load_engine, daemon=True).start()  # 先开端口，再后台加载：存活探针立刻可用
    yield


app = FastAPI(title="llm-infer hardened server", lifespan=lifespan)


@app.get("/healthz")
def healthz():
    """存活：进程还在响应。加载模型期间也返回 200。"""
    return {"status": "alive", "pid": os.getpid()}


@app.get("/readyz")
def readyz():
    """就绪：模型已加载并预热，可以接请求。否则 503，负载均衡据此不转发流量。"""
    if not ready:
        return JSONResponse({"status": "loading"}, status_code=503)
    return {"status": "ready", "model": MODEL_NAME, "device": engine.device}


@app.get("/metrics")
def metrics():
    with state_lock:
        return {**counters, "pid": os.getpid(), "uptime_seconds": round(time.time() - started, 1),
                "limits": {"max_inflight": MAX_INFLIGHT, "queue_timeout": QUEUE_TIMEOUT,
                           "max_seconds": MAX_SECONDS}}


def reject(rid: str, status: int, key: str, message: str, t0: float, headers: dict | None = None):
    bump(key)
    log(event="request", request_id=rid, status=status, error=message, total_ms=round((time.time() - t0) * 1000))
    return JSONResponse({"error": {"message": message, "request_id": rid}}, status_code=status,
                        headers={"X-Request-ID": rid, **(headers or {})})


@app.post("/v1/chat/completions")
def chat(req: ChatRequest, request: Request):
    t0 = time.time()
    rid = request.headers.get("x-request-id") or uuid.uuid4().hex[:12]
    bump("requests")
    if not ready:
        return reject(rid, 503, "not_ready_503", "model is still loading", t0, {"Retry-After": "2"})
    with state_lock:
        admitted = counters["inflight"] < MAX_INFLIGHT
        if admitted:
            counters["inflight"] += 1
    if not admitted:
        return reject(rid, 429, "rejected_429", "too many requests in flight", t0, {"Retry-After": "1"})

    released = False

    def release():
        nonlocal released
        if not released:
            released = True
            bump("inflight", -1)

    try:
        ids = engine.encode_chat([m.model_dump() for m in req.messages])
        it = engine.stream(ids, max_new_tokens=req.max_tokens, temperature=req.temperature, top_p=req.top_p,
                           seed=req.seed, ignore_eos=req.ignore_eos, queue_timeout=QUEUE_TIMEOUT,
                           max_seconds=MAX_SECONDS)
        first = next(it)  # 在发出响应头之前拿到引擎，排队超时才能返回真正的 503 状态码
    except QueueTimeout:
        release()
        return reject(rid, 503, "queue_timeout_503", f"waited more than {QUEUE_TIMEOUT}s in queue", t0,
                      {"Retry-After": "2"})
    except Exception as e:  # noqa: BLE001
        release()
        log(event="exception", request_id=rid, type=type(e).__name__, message=str(e))
        bump("errors_500")
        return JSONResponse({"error": {"message": "internal error", "request_id": rid}}, status_code=500,
                            headers={"X-Request-ID": rid})

    created = int(time.time())
    ttft_ms = round((time.time() - t0) * 1000)

    def tokens():
        yield first
        yield from it

    def finish(n: int, reason: str | None):
        release()
        bump("ok")
        if reason == "timeout":
            bump("deadline_timeout")
        log(event="request", request_id=rid, status=200, stream=req.stream, prompt_tokens=ids.shape[1],
            completion_tokens=n, finish_reason=reason, ttft_ms=ttft_ms, total_ms=round((time.time() - t0) * 1000))

    def close_and_finish(n: int, reason: str):
        """关闭引擎生成器（释放引擎锁），成功后才归还名额。生成器正在别的线程里执行时 close 会抛 ValueError，重试。"""
        while True:
            try:
                it.close()
                break
            except ValueError:
                time.sleep(0.01)
        finish(n, reason)

    headers = {"X-Request-ID": rid}
    if not req.stream:
        parts, n, reason = [], 0, "length"
        try:
            for t in tokens():
                parts.append(t.text)
                n += 1
                reason = t.finished or reason
        finally:
            close_and_finish(n, reason)
        return JSONResponse({
            "object": "chat.completion", "created": created, "model": MODEL_NAME,
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "".join(parts)},
                         "finish_reason": reason}],
            "usage": {"prompt_tokens": ids.shape[1], "completion_tokens": n,
                      "total_tokens": ids.shape[1] + n}}, headers=headers)

    gen = tokens()

    async def sse():
        n, reason = 0, "client_disconnected"
        try:
            while not await request.is_disconnected():
                t = await run_in_threadpool(next, gen, None)
                if t is None:
                    break
                n += 1
                chunk = {"object": "chat.completion.chunk", "created": created, "model": MODEL_NAME,
                         "choices": [{"index": 0, "delta": {"content": t.text}, "finish_reason": t.finished}]}
                yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
                if t.finished:
                    reason = t.finished
                    yield "data: [DONE]\n\n"
                    break
        finally:  # 正常结束、客户端断开、被取消，都要归还名额和引擎锁（放到线程里做，避免在被取消的协程里阻塞）
            threading.Thread(target=close_and_finish, args=(n, reason), daemon=True).start()

    return StreamingResponse(sse(), media_type="text/event-stream", headers=headers)


if __name__ == "__main__":
    uvicorn.run(app, host=os.environ.get("HOST", "127.0.0.1"), port=int(os.environ.get("PORT", "8000")),
                log_level="warning", timeout_graceful_shutdown=30)
