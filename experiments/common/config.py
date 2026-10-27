"""全书主线模型的固定配置：所有章节实验共用。"""
from pathlib import Path

MODEL_ID = "HuggingFaceTB/SmolLM2-135M-Instruct"
MODEL_REVISION = "12fd25f77366fa6b3b4b768ec3050bf629380bac"
LICENSE = "apache-2.0"
ALLOW_PATTERNS = [
    "config.json", "generation_config.json", "model.safetensors",
    "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
    "vocab.json", "merges.txt",
]

ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = ROOT / "models" / "smollm2-135m-instruct"

# 第 07—10 章在 GPU 上换用同系列的 1.7B 模型：135M 太小，量化/并发的收益看不出来。
MODEL_LARGE_ID = "HuggingFaceTB/SmolLM2-1.7B-Instruct"
MODEL_LARGE_REVISION = "31b70e2e869a7173562077fd711b654946d38674"
MODEL_LARGE_DIR = ROOT / "models" / "smollm2-1.7b-instruct"
RUNS_DIR = ROOT / "evidence" / "runs"


def pick_device() -> str:
    import torch
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"
