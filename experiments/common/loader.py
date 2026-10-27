"""各章实验共用的模型加载。"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .config import MODEL_DIR, pick_device
from .fetch_model import ensure_model


def load(dtype=torch.float32, device: str | None = None):
    ensure_model()
    device = device or pick_device()
    tok = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForCausalLM.from_pretrained(MODEL_DIR, dtype=dtype).to(device).eval()
    return tok, model, device


def sync(device: str) -> None:
    if device == "mps":
        torch.mps.synchronize()
    elif device == "cuda":
        torch.cuda.synchronize()
