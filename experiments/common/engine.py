"""最小生成引擎：手写 Prefill + Decode 循环，逐 token 产出。服务与基准共用。"""
import threading
import time
from dataclasses import dataclass
from typing import Iterator

import torch

from .loader import load, sync


@dataclass
class Token:
    id: int
    text: str
    finished: str | None  # None | "stop" | "length"


class Engine:
    def __init__(self, dtype=torch.float32, device: str | None = None):
        self.tok, self.model, self.device = load(dtype=dtype, device=device)
        self.lock = threading.Lock()  # 单模型单设备：同一时刻只跑一个请求（第 09 章再谈并发）

    def encode_chat(self, messages: list[dict]) -> torch.Tensor:
        text = self.tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        return self.tok(text, return_tensors="pt")["input_ids"].to(self.device)

    @torch.no_grad()
    def stream(self, ids: torch.Tensor, max_new_tokens: int = 64, temperature: float = 0.0,
               top_p: float = 1.0, seed: int | None = None,
               ignore_eos: bool = False) -> Iterator[Token]:
        gen = torch.Generator(device="cpu")
        if seed is not None:
            gen.manual_seed(seed)
        eos = self.tok.eos_token_id
        with self.lock:
            out = self.model(input_ids=ids, use_cache=True)  # prefill
            for i in range(max_new_tokens):
                logits = out.logits[0, -1].float()
                nxt = self._pick(logits, temperature, top_p, gen)
                done = "stop" if (nxt == eos and not ignore_eos) else ("length" if i == max_new_tokens - 1 else None)
                yield Token(nxt, "" if (nxt == eos and not ignore_eos) else self.tok.decode(nxt), done)
                if done:
                    return
                out = self.model(input_ids=torch.tensor([[nxt]], device=self.device),
                                 past_key_values=out.past_key_values, use_cache=True)  # decode

    @staticmethod
    def _pick(logits, temperature, top_p, gen) -> int:
        if temperature <= 0:
            return int(torch.argmax(logits))
        probs = torch.softmax(logits / temperature, dim=-1).cpu()
        if top_p < 1.0:
            sp, si = torch.sort(probs, descending=True)
            keep = torch.cumsum(sp, 0) - sp < top_p
            sp = sp * keep
            probs = torch.zeros_like(probs).scatter_(0, si, sp)
        return int(torch.multinomial(probs / probs.sum(), 1, generator=gen))

    def complete(self, ids, **kw) -> tuple[str, str, int]:
        parts, reason, n = [], "length", 0
        for t in self.stream(ids, **kw):
            parts.append(t.text)
            n += 1
            reason = t.finished or reason
        return "".join(parts), reason, n
