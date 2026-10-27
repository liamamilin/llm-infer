"""E11：最小进程守护——子进程退出就重启（指数退避），收到 SIGTERM 则转发给子进程并退出，不再重启。

    .venv/bin/python -m experiments.ch11.supervisor -- .venv/bin/python -m experiments.ch11.server
"""
import json
import signal
import subprocess
import sys
import time


def main(cmd: list[str]) -> int:
    stopping = False
    child: subprocess.Popen | None = None

    def on_term(_sig, _frame):
        nonlocal stopping
        stopping = True
        if child and child.poll() is None:
            child.send_signal(signal.SIGTERM)

    signal.signal(signal.SIGTERM, on_term)
    signal.signal(signal.SIGINT, on_term)
    backoff = 0.5
    while not stopping:
        t0 = time.time()
        child = subprocess.Popen(cmd)
        print(json.dumps({"event": "child_started", "pid": child.pid}), flush=True)
        code = child.wait()
        print(json.dumps({"event": "child_exited", "pid": child.pid, "code": code}), flush=True)
        if stopping:
            return 0 if code in (0, -signal.SIGTERM) else code  # uvicorn 优雅退出后会重新抛出 SIGTERM
        backoff = 0.5 if time.time() - t0 > 30 else min(backoff * 2, 30)  # 稳定运行过就重置退避
        time.sleep(backoff)
    return 0


if __name__ == "__main__":
    args = sys.argv[1:]
    if args and args[0] == "--":
        args = args[1:]
    raise SystemExit(main(args))
