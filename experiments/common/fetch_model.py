"""下载固定 revision 的主线模型到 models/（已存在则跳过）。"""
from huggingface_hub import snapshot_download

from .config import ALLOW_PATTERNS, MODEL_DIR, MODEL_ID, MODEL_REVISION


def ensure_model() -> str:
    path = snapshot_download(
        MODEL_ID, revision=MODEL_REVISION,
        allow_patterns=ALLOW_PATTERNS, local_dir=MODEL_DIR,
    )
    return path


if __name__ == "__main__":
    print(ensure_model())
