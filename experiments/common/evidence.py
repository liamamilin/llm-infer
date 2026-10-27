"""写入机器可读的实验证据 evidence/runs/*.json。"""
import json
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone

from .config import MODEL_ID, MODEL_REVISION, ROOT, RUNS_DIR


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, stderr=subprocess.DEVNULL, text=True
        ).strip()
    except Exception:
        return "unknown"


def write_run(chapter_id: str, experiment_id: str, status: str, assertions: dict,
              runtime_version: dict, device: str, command: str, notes: str = "",
              extra: dict | None = None, runtime: str = "transformers", model_id: str = MODEL_ID,
              model_revision: str = MODEL_REVISION) -> str:
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    now = datetime.now(timezone.utc)
    record = {
        "chapter_id": chapter_id,
        "experiment_id": experiment_id,
        "status": status,
        "timestamp": now.isoformat(),
        "platform": f"{platform.system()} {platform.machine()} py{platform.python_version()}",
        "device": device,
        "model_id": model_id,
        "model_revision": model_revision,
        "runtime": runtime,
        "runtime_version": runtime_version,
        "exact_command": command,
        "assertions": assertions,
        "git_sha": _git_sha(),
        "notes": notes,
    }
    try:
        import torch
        if torch.cuda.is_available():
            record["gpu"] = torch.cuda.get_device_name(0)
    except Exception:
        pass
    if extra:
        record.update(extra)
    path = RUNS_DIR / f"{now:%Y%m%d}-{chapter_id}-{experiment_id}-{int(time.time())}.json"
    path.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n")
    return str(path)


def cli_command() -> str:
    spec = getattr(sys.modules["__main__"], "__spec__", None)
    return f"python -m {spec.name}" if spec else "python <script>"
