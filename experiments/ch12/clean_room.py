"""E12：洁净室复现——把已提交的 HEAD 克隆到临时目录，新建虚拟环境，只按 requirements.txt 安装，再跑关键实验。

    .venv/bin/python -m experiments.ch12.clean_room

只验证“已提交”的内容；工作区里没提交的改动不会被带进去（这正是目的）。
"""
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from experiments.common.evidence import cli_command, write_run

ROOT = Path(__file__).resolve().parents[2]
STEPS = [
    ("ch00_smoke", ["-m", "experiments.ch00.smoke"]),
    ("ch04_http_stream", ["-m", "experiments.ch04.verify"]),
    ("ch11_guardrails", ["-m", "experiments.ch11.verify"]),
    ("evidence_tests", ["-m", "unittest", "tests.test_evidence"]),
]


def sh(cmd, cwd, log):
    t0 = time.time()
    with open(log, "w") as f:
        code = subprocess.run(cmd, cwd=cwd, stdout=f, stderr=subprocess.STDOUT, env={**os.environ, "PYTHONUNBUFFERED": "1"}).returncode
    return code, round(time.time() - t0, 1)


def main() -> int:
    tmp = Path(tempfile.mkdtemp(prefix="cleanroom-"))
    repo, venv = tmp / "repo", tmp / "venv"
    dirty = subprocess.run(["git", "status", "--porcelain"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    sha = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    print(f"cloning HEAD {sha[:8]} into {tmp}" + ("  (working tree has uncommitted changes, NOT included)" if dirty else ""))
    steps = {}
    code, secs = sh(["git", "clone", "--quiet", str(ROOT), str(repo)], tmp, tmp / "clone.log")
    steps["clone"] = {"exit": code, "seconds": secs}
    code, secs = sh([sys.executable, "-m", "venv", str(venv)], tmp, tmp / "venv.log")
    py = str(venv / "bin" / "python")
    steps["venv"] = {"exit": code, "seconds": secs}
    code, secs = sh([py, "-m", "pip", "install", "-q", "-r", "requirements.txt"], repo, tmp / "pip.log")
    steps["pip_install"] = {"exit": code, "seconds": secs}
    print(f"pip install: exit {code} in {secs}s")
    for name, args in STEPS:
        if steps["pip_install"]["exit"] != 0:
            break
        code, secs = sh([py, *args], repo, tmp / f"{name}.log")
        tail = (tmp / f"{name}.log").read_text().strip().splitlines()[-1:] or [""]
        steps[name] = {"exit": code, "seconds": secs}
        print(f"{name}: exit {code} in {secs}s  | {tail[0][:100]}")
    freeze = subprocess.run([py, "-m", "pip", "freeze"], capture_output=True, text=True).stdout
    pins = {m.group(1): m.group(2) for m in re.finditer(r"^(torch|transformers|fastapi|uvicorn|httpx)==(\S+)$", freeze, re.M)}
    wanted = {m.group(1): m.group(2) for m in re.finditer(r"^(torch|transformers|fastapi|uvicorn|httpx)==(\S+)$",
                                                             (ROOT / "requirements.txt").read_text(), re.M)}
    runs = sorted((repo / "evidence" / "runs").glob("*.json")) if (repo / "evidence" / "runs").exists() else []
    run_status = {json.loads(p.read_text())["chapter_id"]: json.loads(p.read_text())["status"] for p in runs}
    a = {"working_tree_was_clean": not dirty,
         "installed_only_from_requirements": steps["pip_install"]["exit"] == 0 and pins == wanted,
         **{f"{n}_exit_0": steps.get(n, {}).get("exit") == 0 for n, _ in STEPS},
         "fresh_runs_all_pass": bool(run_status) and set(run_status.values()) == {"PASS"}
         and {"ch00", "ch04", "ch11"} <= set(run_status)}
    status = "PASS" if all(a.values()) else "FAIL"
    path = write_run("ch12", "clean_room", status, a, {"python": sys.version.split()[0], **pins}, "mps",
                     cli_command(), extra={"cloned_commit": sha, "steps": steps, "fresh_run_status": run_status})
    print(f"{status} -> {path}")
    for k, v in a.items():
        print(("  ok " if v else "  FAIL ") + k)
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
