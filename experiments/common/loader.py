"""各章实验共用的模型加载。"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .config import MODEL_DIR, MODEL_LARGE_DIR, pick_device
from .fetch_model import ensure_model


def load(dtype=torch.float32, device: str | None = None, size: str = "small", **model_kwargs):
    ensure_model(size)
    device = device or pick_device()
    path = MODEL_LARGE_DIR if size == "large" else MODEL_DIR
    tok = AutoTokenizer.from_pretrained(path)
    model = AutoModelForCausalLM.from_pretrained(path, dtype=dtype, **model_kwargs)
    if "quantization_config" not in model_kwargs:
        model = model.to(device)
    return tok, model.eval(), device


def sync(device: str) -> None:
    if device == "mps":
        torch.mps.synchronize()
    elif device == "cuda":
        torch.cuda.synchronize()
