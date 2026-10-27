"""E04：最小 HTTP 推理服务，接口形状兼容 OpenAI chat/completions（含 SSE 流式）。

启动（在仓库根目录）：
    .venv/bin/python -m experiments.ch04.server            # 默认 127.0.0.1:8000
"""
import json
import os
import time
from contextlib import asynccontextmanager

import torch
import uvicorn
from fastapi import FastAPI
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from experiments.common.config import MODEL_ID, MODEL_LARGE_ID
from experiments.common.engine import Engine

MODEL_SIZE = os.environ.get("MODEL_SIZE", "small")
MODEL_NAME = MODEL_LARGE_ID if MODEL_SIZE == "large" else MODEL_ID
engine: Engine | None = None


@asynccontextmanager
async def lifespan(_app):
    global engine
    dtype = getattr(torch, os.environ.get("DTYPE", "float32"))
    engine = Engine(dtype=dtype, size=MODEL_SIZE)
    yield


app = FastAPI(title="llm-infer demo server", lifespan=lifespan)


class Message(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    messages: list[Message]
    max_tokens: int = Field(64, ge=1, le=512)
    temperature: float = Field(0.0, ge=0.0, le=2.0)
    top_p: float = Field(1.0, gt=0.0, le=1.0)
    seed: int | None = None
    stream: bool = False
    ignore_eos: bool = False


@app.get("/health")
def health():
    return {"status": "ok" if engine else "loading", "model": MODEL_NAME,
            "device": engine.device if engine else None}


@app.post("/v1/chat/completions")
def chat(req: ChatRequest):
    ids = engine.encode_chat([m.model_dump() for m in req.messages])
    kw = dict(max_new_tokens=req.max_tokens, temperature=req.temperature, top_p=req.top_p, seed=req.seed,
              ignore_eos=req.ignore_eos)
    created = int(time.time())

    if not req.stream:
        text, reason, n = engine.complete(ids, **kw)
        return JSONResponse({
            "object": "chat.completion", "created": created, "model": MODEL_NAME,
            "choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": reason}],
            "usage": {"prompt_tokens": ids.shape[1], "completion_tokens": n,
                      "total_tokens": ids.shape[1] + n},
        })

    def sse():
        for t in engine.stream(ids, **kw):
            chunk = {"object": "chat.completion.chunk", "created": created, "model": MODEL_NAME,
                     "choices": [{"index": 0, "delta": {"content": t.text}, "finish_reason": t.finished}]}
            yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(sse(), media_type="text/event-stream")


if __name__ == "__main__":
    uvicorn.run(app, host=os.environ.get("HOST", "127.0.0.1"), port=int(os.environ.get("PORT", "8000")))
