"""下载固定 revision 的主线模型到 models/（已存在则跳过）。"""
from huggingface_hub import snapshot_download

from .config import (ALLOW_PATTERNS, MODEL_DIR, MODEL_ID, MODEL_LARGE_DIR, MODEL_LARGE_ID,
                     MODEL_LARGE_REVISION, MODEL_REVISION)


def ensure_model(size: str = "small") -> str:
    """size: "small"（135M，主线）或 "large"（1.7B，GPU 章节）。国内网络可设 HF_ENDPOINT 镜像。"""
    if size == "large":
        return snapshot_download(MODEL_LARGE_ID, revision=MODEL_LARGE_REVISION,
                                 allow_patterns=ALLOW_PATTERNS, local_dir=MODEL_LARGE_DIR)
    return snapshot_download(MODEL_ID, revision=MODEL_REVISION,
                             allow_patterns=ALLOW_PATTERNS, local_dir=MODEL_DIR)


if __name__ == "__main__":
    import sys
    print(ensure_model(sys.argv[1] if len(sys.argv) > 1 else "small"))
