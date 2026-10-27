"""E11：端到端验证服务护栏——就绪探针、准入控制、排队/生成超时、断开回收、日志、崩溃恢复、优雅关闭。

    .venv/bin/python -m experiments.ch11.verify             # 本机，135M
    MODEL_SIZE=large DTYPE=bfloat16 python -m experiments.ch11.verify   # GPU，1.7B
"""
import json
import os
import signal
import socket
import statistics
import subprocess
import sys
import tempfile
import threading
import time

import httpx
import torch
import transformers

from experiments.common.config import MODEL_LARGE_ID, MODEL_LARGE_REVISION
from experiments.common.evidence import cli_command, write_run

QUEUE_TIMEOUT, MAX_SECONDS, MAX_INFLIGHT = 1.0, 2.0, 4
BURST = 24
MSGS = [{"role": "user", "content": "Write a short story about a cat."}]


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


CLIENT = httpx.Client(limits=httpx.Limits(max_connections=100))  # 共用连接池：httpx.post 每次新建客户端会重建 SSL 上下文，24 线程会把耗时算进被测延迟


def post(base, rid=None, **body):
    """返回 (状态码, 耗时秒, json, headers)。"""
    t0 = time.perf_counter()
    r = CLIENT.post(f"{base}/v1/chat/completions", json={"messages": MSGS, **body}, timeout=120,
                   headers={"X-Request-ID": rid} if rid else {})
    return r.status_code, time.perf_counter() - t0, r.json(), r.headers


def burst(base, n, **body):
    res = [None] * n

    def one(i):
        res[i] = post(base, **body)[:2]

    ts = [threading.Thread(target=one, args=(i,)) for i in range(n)]
    [t.start() for t in ts]
    [t.join() for t in ts]
    return res


def wait_until(fn, timeout, step=0.1):
    t0 = time.time()
    while time.time() - t0 < timeout:
        v = fn()
        if v:
            return v
        time.sleep(step)
    return None


def try_get(url):
    try:
        return httpx.get(url, timeout=2)
    except Exception:
        return None


def start_server(module, port, env_extra, log_path):
    env = {**os.environ, "PORT": str(port), **env_extra}
    return subprocess.Popen([sys.executable, "-m", module], env=env, stdout=open(log_path, "w"),
                            stderr=subprocess.STDOUT, start_new_session=True)


def main() -> int:
    size = os.environ.get("MODEL_SIZE", "small")
    tmp = tempfile.mkdtemp()
    port = free_port()
    base = f"http://127.0.0.1:{port}"
    log_path = f"{tmp}/hardened.log"
    env = {"MODEL_SIZE": size, "MAX_INFLIGHT": str(MAX_INFLIGHT), "QUEUE_TIMEOUT": str(QUEUE_TIMEOUT),
           "MAX_SECONDS": str(MAX_SECONDS)}
    sup = subprocess.Popen([sys.executable, "-m", "experiments.ch11.supervisor", "--", sys.executable, "-m",
                            "experiments.ch11.server"], env={**os.environ, "PORT": str(port), **env},
                           stdout=open(log_path, "w"), stderr=subprocess.STDOUT)
    a, extra = {}, {}
    try:
        # A. 启动：存活先于就绪，加载期间请求被拒绝
        t0 = time.time()
        alive_at = ready_at = None
        saw_loading = loading_chat = None
        while ready_at is None:
            if time.time() - t0 > 600 or sup.poll() is not None:
                raise RuntimeError(open(log_path).read())
            h = try_get(f"{base}/healthz")
            if h is not None and h.status_code == 200 and alive_at is None:
                alive_at = time.time() - t0
            r = try_get(f"{base}/readyz")
            if r is not None and r.status_code == 503:
                saw_loading = True
                if loading_chat is None:
                    loading_chat = post(base, max_tokens=4)[0]
            if r is not None and r.status_code == 200:
                ready_at = time.time() - t0
            time.sleep(0.05)
        print(f"alive after {alive_at:.2f}s, ready after {ready_at:.2f}s, chat while loading -> {loading_chat}")
        a["alive_before_ready"] = alive_at is not None and alive_at < ready_at
        a["readyz_503_while_loading"] = bool(saw_loading)
        a["chat_rejected_503_while_loading"] = loading_chat == 503
        extra.update(alive_seconds=round(alive_at, 2), ready_seconds=round(ready_at, 2))

        # B. 突发：24 个并发请求，准入控制
        burst_t0 = time.time()
        res = burst(base, BURST, max_tokens=32, ignore_eos=True)
        codes = [c for c, _ in res]
        by = {c: sorted(round(t, 2) for k, t in res if k == c) for c in set(codes)}
        print("burst:", {c: (len(v), v[0], v[-1]) for c, v in by.items()})
        bounded_max = max(t for _, t in res)
        a["burst_only_expected_statuses"] = set(codes) <= {200, 429, 503}
        a["burst_some_succeeded"] = codes.count(200) >= 1
        a["burst_some_rejected_429"] = codes.count(429) >= 1
        a["burst_429_fast"] = all(t < 0.5 for c, t in res if c == 429)
        a["burst_latency_bounded"] = bounded_max < QUEUE_TIMEOUT + MAX_SECONDS + 3
        extra["burst"] = {"requests": BURST, "status_counts": {str(c): codes.count(c) for c in sorted(set(codes))},
                          "p50_latency_429_s": statistics.median(by[429]) if 429 in by else None,
                          "max_latency_s": round(bounded_max, 2)}
        time.sleep(0.3)
        rej_logs = [json.loads(x) for x in open(log_path) if x.startswith("{") and '"status": 429' in x]
        # 服务端是什么时候“处理”到第一个 429 的？晚了说明请求在到达处理函数之前就在排队（事件循环/GIL 被占用）
        handler_delay = min(x["ts"] for x in rej_logs) - burst_t0 if rej_logs else None
        print(f"first 429 handled {handler_delay:.2f}s after burst start; handler time {max(x['total_ms'] for x in rej_logs)}ms")
        extra["burst"]["first_429_handled_after_s"] = round(handler_delay, 2)
        extra["burst"]["max_429_handler_ms"] = max(x["total_ms"] for x in rej_logs)
        wait_until(lambda: httpx.get(f"{base}/metrics").json()["inflight"] == 0, 15)

        # C+D. 排队超时 503；超过生成时限的请求被截断（finish_reason=timeout）
        long = {}

        def run_long():
            long["res"] = post(base, max_tokens=512, ignore_eos=True)

        th = threading.Thread(target=run_long)
        th.start()
        time.sleep(0.4)
        code, lat, _, _ = post(base, max_tokens=8)
        th.join()
        lcode, llat, lbody, _ = long["res"]
        print(f"queue timeout: second request -> {code} after {lat:.2f}s; long request -> {lcode} "
              f"finish={lbody['choices'][0]['finish_reason']} tokens={lbody['usage']['completion_tokens']} in {llat:.2f}s")
        a["queue_timeout_returns_503"] = code == 503
        a["queue_timeout_close_to_limit"] = 0.5 <= lat <= QUEUE_TIMEOUT + 1.0
        a["deadline_truncates_long_request"] = (lcode == 200 and lbody["choices"][0]["finish_reason"] == "timeout"
                                                and lbody["usage"]["completion_tokens"] < 512)
        a["deadline_respected"] = llat < MAX_SECONDS + 2.0
        extra["queue_timeout"] = {"status": code, "latency_s": round(lat, 2)}
        extra["deadline"] = {"latency_s": round(llat, 2), "completion_tokens": lbody["usage"]["completion_tokens"],
                             "finish_reason": lbody["choices"][0]["finish_reason"]}
        wait_until(lambda: httpx.get(f"{base}/metrics").json()["inflight"] == 0, 15)

        # E. 客户端中途断开：名额和引擎锁必须被回收
        with httpx.stream("POST", f"{base}/v1/chat/completions", timeout=60,
                          json={"messages": MSGS, "stream": True, "max_tokens": 512, "ignore_eos": True}) as r:
            n = 0
            for line in r.iter_lines():
                if line.startswith("data: "):
                    n += 1
                if n >= 3:
                    break
        t_disc = time.time()
        freed = wait_until(lambda: httpx.get(f"{base}/metrics").json()["inflight"] == 0, 10, 0.05)
        free_s = time.time() - t_disc
        code2, lat2, _, _ = post(base, max_tokens=8)
        print(f"disconnect: inflight back to 0 after {free_s:.2f}s; next request -> {code2} in {lat2:.2f}s")
        a["disconnect_releases_slot"] = bool(freed)
        a["next_request_ok_after_disconnect"] = code2 == 200
        extra["disconnect"] = {"slot_freed_seconds": round(free_s, 2), "next_status": code2,
                               "next_latency_s": round(lat2, 2)}

        # F. 日志与请求 ID
        code3, _, _, hdr = post(base, rid="trace-me-123", max_tokens=6)
        time.sleep(0.3)
        lines = [json.loads(x) for x in open(log_path) if x.startswith("{")]
        reqs = [x for x in lines if x.get("event") == "request"]
        a["request_id_echoed"] = hdr.get("x-request-id") == "trace-me-123"
        a["request_logged_with_id"] = any(x["request_id"] == "trace-me-123" and x["status"] == 200 for x in reqs)
        a["rejections_logged"] = {429, 503} <= {x["status"] for x in reqs}
        a["logs_are_json_lines_with_latency"] = all("total_ms" in x for x in reqs)
        m = httpx.get(f"{base}/metrics").json()
        a["metrics_match_burst"] = m["rejected_429"] >= codes.count(429) and m["queue_timeout_503"] >= 1
        extra["metrics"] = {k: v for k, v in m.items() if k not in ("pid", "uptime_seconds")}

        # H. 崩溃恢复：kill -9 工作进程，守护进程把它拉起来
        pid_before = httpx.get(f"{base}/metrics").json()["pid"]
        os.kill(pid_before, signal.SIGKILL)
        t_kill = time.time()
        rec = wait_until(lambda: (lambda r: r is not None and r.status_code == 200)(try_get(f"{base}/readyz")), 300, 0.1)
        rec_s = time.time() - t_kill
        pid_after = httpx.get(f"{base}/metrics").json()["pid"]
        code4, _, _, _ = post(base, max_tokens=8)
        print(f"crash: pid {pid_before} killed -> pid {pid_after}, ready again after {rec_s:.1f}s, request -> {code4}")
        a["crash_recovered"] = bool(rec) and pid_after != pid_before
        a["request_ok_after_recovery"] = code4 == 200
        extra["recovery"] = {"seconds_to_ready": round(rec_s, 1)}

        # G. 优雅关闭：处理中的流式请求被完整送完，然后进程以 0 退出
        got, done, term_at = [], False, None
        with httpx.stream("POST", f"{base}/v1/chat/completions", timeout=60,
                          json={"messages": MSGS, "stream": True, "max_tokens": 48, "ignore_eos": True}) as r:
            for line in r.iter_lines():
                if not line.startswith("data: "):
                    continue
                if line[6:] == "[DONE]":
                    done = True
                    break
                got.append(line)
                if len(got) == 5 and term_at is None:
                    term_at = time.time()
                    sup.send_signal(signal.SIGTERM)
        exit_code = sup.wait(timeout=60)
        print(f"graceful: SIGTERM mid-stream -> received {len(got)} chunks, done={done}, supervisor exit={exit_code}")
        a["sigterm_stream_completed"] = done and len(got) == 48
        a["supervisor_exits_zero"] = exit_code == 0
        extra["graceful"] = {"chunks_received": len(got), "exit_code": exit_code}

        # 对照：第 04 章服务（无护栏）在同样的突发下
        bport = free_port()
        bproc = start_server("experiments.ch04.server", bport, {"MODEL_SIZE": size}, f"{tmp}/baseline.log")
        try:
            bbase = f"http://127.0.0.1:{bport}"
            if not wait_until(lambda: (lambda r: r is not None and r.status_code == 200)(try_get(f"{bbase}/health")), 300, 0.5):
                raise RuntimeError("baseline server failed")
            bres = burst(bbase, BURST, max_tokens=32, ignore_eos=True)
        finally:
            bproc.terminate()
            bproc.wait(timeout=30)
        bmax = max(t for _, t in bres)
        print(f"baseline (ch04, no guardrails): statuses={sorted(set(c for c, _ in bres))} max latency {bmax:.2f}s "
              f"vs guarded max {bounded_max:.2f}s")
        a["baseline_all_served_but_slower_tail"] = all(c == 200 for c, _ in bres) and bmax > bounded_max
        extra["baseline_burst"] = {"requests": BURST, "all_200": all(c == 200 for c, _ in bres),
                                   "max_latency_s": round(bmax, 2)}
        extra["limits"] = {"max_inflight": MAX_INFLIGHT, "queue_timeout_s": QUEUE_TIMEOUT, "max_seconds": MAX_SECONDS}

        status = "PASS" if all(a.values()) else "FAIL"
        device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
        path = write_run("ch11", "guardrails" if size == "small" else "guardrails_large", status, a,
                         {"torch": torch.__version__, "transformers": transformers.__version__}, device,
                         cli_command(), extra=extra,
                         **({} if size == "small" else {"model_id": MODEL_LARGE_ID, "model_revision": MODEL_LARGE_REVISION}))
        print(f"{status} -> {path}")
        for k, v in a.items():
            if not v:
                print("FAILED:", k)
        return 0 if status == "PASS" else 1
    finally:
        if sup.poll() is None:
            sup.terminate()
            try:
                sup.wait(timeout=30)
            except subprocess.TimeoutExpired:
                sup.kill()
        subprocess.run(["pkill", "-f", "experiments.ch11.server"], check=False)


if __name__ == "__main__":
    raise SystemExit(main())
