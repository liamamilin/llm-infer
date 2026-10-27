#!/usr/bin/env python3
"""Safe standard-library environment preview. No identities, serials or absolute paths."""
import json
import os
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone

def run(cmd):
    if not shutil.which(cmd[0]):
        return None
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=4, check=False)
        return p.stdout.strip()[:200] if p.returncode == 0 else None
    except (OSError, subprocess.TimeoutExpired):
        return None

def detect():
    return {
        'status': 'PREFLIGHT_ONLY_NOT_INFERENCE_TEST',
        'time_utc': datetime.now(timezone.utc).isoformat(),
        'os': platform.system(),
        'os_release': platform.release(),
        'arch': platform.machine(),
        'python': platform.python_version(),
        'logical_cpus': os.cpu_count(),
        'node': run(['node', '--version']),
        'npm': run(['npm', '--version']),
        'git': run(['git', '--version']),
        'nvidia_smi_available': bool(shutil.which('nvidia-smi')),
        'gpu_probe_required': True,
        'model_inference_verified': False,
    }

if __name__ == '__main__':
    print(json.dumps(detect(), indent=2, ensure_ascii=False))
