"""从 evidence/public 生成数据图，并写出示意图，输出到 docs/.vitepress/theme/figures/。仅用标准库。

    python3 scripts/make_figures.py
数据图的数字全部来自证据文件；示意图（schematic）不含测量数字。
"""
import json
from html import escape
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PUB = ROOT / "evidence" / "public"
OUT = ROOT / "docs" / ".vitepress" / "theme" / "figures"
A, B, MUTED, GRID = "#4f7cff", "#f0a030", "#8a8f98", "#8a8f9833"  # 两种数据色 + 中性色，深浅主题下都可读
FONT = "font-family=\"system-ui,-apple-system,'PingFang SC','Noto Sans SC',sans-serif\""


def ev(name):
    return json.loads((PUB / f"{name}.json").read_text())


def svg(w, h, body, title, desc):
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" role="img" {FONT} font-size="13" fill="currentColor">'
            f'<title>{escape(title)}</title><desc>{escape(desc)}</desc>{body}</svg>\n')


def t(x, y, s, size=13, anchor="start", fill="currentColor", weight="normal"):
    return f'<text x="{x}" y="{y}" font-size="{size}" text-anchor="{anchor}" fill="{fill}" font-weight="{weight}">{escape(str(s))}</text>'


def bars(x0, y0, w, h, labels, series, title, unit, colors=(A, B), fmt="{:g}"):
    """单个柱状面板。series: [(name, [values])]，多系列为分组柱。"""
    vmax = max(max(v) for _, v in series) * 1.18
    n, k = len(labels), len(series)
    slot = w / n
    bw = min(46, slot * 0.7 / k)
    out = [t(x0, y0 - 14, title, 14, weight="600")]
    for g in range(5):
        y = y0 + h - h * g / 4
        out.append(f'<line x1="{x0}" x2="{x0 + w}" y1="{y:.1f}" y2="{y:.1f}" stroke="{GRID}"/>')
    for i, lab in enumerate(labels):
        cx = x0 + slot * (i + 0.5)
        for j, (_, vals) in enumerate(series):
            bh = h * vals[i] / vmax
            bx = cx - bw * k / 2 + bw * j
            out.append(f'<rect x="{bx:.1f}" y="{y0 + h - bh:.1f}" width="{bw - 3:.1f}" height="{bh:.1f}" rx="3" fill="{colors[j]}"/>')
            out.append(t(bx + (bw - 3) / 2, y0 + h - bh - 5, fmt.format(vals[i]), 11, "middle"))
        out.append(t(cx, y0 + h + 18, lab, 12, "middle"))
    out.append(t(x0 + w, y0 - 14, unit, 11, "end", MUTED))
    if k > 1:
        for j, (name, _) in enumerate(series):
            out.append(f'<rect x="{x0 + 120 + j * 120}" y="{y0 - 25}" width="10" height="10" rx="2" fill="{colors[j]}"/>' + t(x0 + 134 + j * 120, y0 - 15, name, 11))
    return "".join(out)


def lines(x0, y0, w, h, xs, series, title, unit, colors=(A, B)):
    """对数纵轴折线图。series: [(name, [值或 None])]，与 xs 对齐；None 表示该点没测。"""
    import math
    vals = [v for _, vs in series for v in vs if v is not None]
    lo, hi = math.floor(math.log10(min(vals))), math.ceil(math.log10(max(vals)))
    px = lambda i: x0 + 20 + (w - 40) * i / (len(xs) - 1)
    py = lambda v: y0 + h - h * (math.log10(v) - lo) / (hi - lo)
    out = [t(x0, y0 - 28, title, 14, weight="600"), t(x0 + w, y0 - 28, unit, 11, "end", MUTED)]
    for d in range(lo, hi + 1):
        y = y0 + h - h * (d - lo) / (hi - lo)
        out.append(f'<line x1="{x0}" x2="{x0 + w}" y1="{y:.1f}" y2="{y:.1f}" stroke="{GRID}"/>')
        out.append(t(x0 - 6, y + 4, f"{10 ** d:g}", 10, "end", MUTED))
    for i, x in enumerate(xs):
        out.append(t(px(i), y0 + h + 18, x, 11, "middle"))
    for j, (name, vs) in enumerate(series):
        pts = [(px(i), py(v)) for i, v in enumerate(vs) if v is not None]
        out.append(f'<polyline points="{" ".join(f"{a:.1f},{b:.1f}" for a, b in pts)}" fill="none" stroke="{colors[j]}" stroke-width="2.5"/>')
        out += [f'<circle cx="{a:.1f}" cy="{b:.1f}" r="3.5" fill="{colors[j]}"/>' for a, b in pts]
        out.append(f'<rect x="{x0 + j * 90}" y="{y0 - 22}" width="10" height="10" rx="2" fill="{colors[j]}"/>' + t(x0 + 14 + j * 90, y0 - 12, name, 11))
    return "".join(out)


def box(x, y, w, h, label, sub="", fill="none", stroke=A):
    s = f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="8" fill="{fill}" stroke="{stroke}" stroke-width="1.6"/>'
    s += t(x + w / 2, y + h / 2 + (-2 if sub else 4), label, 13, "middle", weight="600")
    if sub:
        s += t(x + w / 2, y + h / 2 + 15, sub, 11, "middle", MUTED)
    return s


def arrow(x1, y1, x2, y2, label="", color=MUTED):
    import math
    ang = math.atan2(y2 - y1, x2 - x1)
    hx, hy = x2 - 8 * math.cos(ang - 0.4), y2 - 8 * math.sin(ang - 0.4)
    gx, gy = x2 - 8 * math.cos(ang + 0.4), y2 - 8 * math.sin(ang + 0.4)
    s = f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" stroke-width="1.6"/><polygon points="{x2},{y2} {hx:.1f},{hy:.1f} {gx:.1f},{gy:.1f}" fill="{color}"/>'
    if label:
        s += t((x1 + x2) / 2, (y1 + y2) / 2 - 6, label, 11, "middle", MUTED)
    return s


FIGS = {}


def data_figs():
    r = ev("ch05-bench")["results"]
    ks = ["32", "128", "512"]
    FIGS["ch05-ttft"] = svg(720, 270, bars(60, 50, 600, 170, [f"提示词 {k} token" for k in ks],
        [("TTFT p50", [round(r[k]["ttft"]["p50"] * 1000, 1) for k in ks]), ("TPOT p50", [round(r[k]["tpot"]["p50"] * 1000, 2) for k in ks])],
        "首 token 延迟随提示词变长，每 token 延迟基本不变", "毫秒（ms）"),
        "TTFT 与 TPOT", "三种提示词长度下的 TTFT p50 与 TPOT p50，来自 ch05-bench.json")

    rows = ev("ch06-kv_cache")["rows"]
    FIGS["ch06-kv"] = svg(720, 270, bars(60, 50, 600, 170, [str(x["seq_len"]) for x in rows],
        [("KV Cache", [round(x["actual_cache_bytes"] / 2**20, 1) for x in rows]), ("完整 logits", [round(x["full_logits_bytes"] / 2**20, 1) for x in rows])],
        "KV Cache 随序列长度线性增长；Prefill 的完整 logits 更大", "MiB（横轴：序列长度 token）"),
        "KV Cache 与 logits 大小", "不同序列长度下 KV Cache 与 Prefill 完整 logits 的大小，来自 ch06-kv_cache.json")

    q = ev("ch07-quant_hf")["results"]
    ks = ["bf16", "int8", "nf4"]
    FIGS["ch07-quant"] = svg(720, 250,
        bars(30, 50, 200, 150, ks, [("MB", [round(q[k]["weights_mb"]) for k in ks])], "权重内存", "MB", (A,)) +
        bars(260, 50, 200, 150, ks, [("tok/s", [q[k]["tok_s_p50"] for k in ks])], "生成速度", "tok/s", (B,)) +
        bars(490, 50, 200, 150, ks, [("ppl", [q[k]["perplexity"] for k in ks])], "困惑度（越低越好）", "ppl", (MUTED,)),
        "三种精度的内存、速度、困惑度", "bf16、int8、nf4 三种精度的权重内存、生成速度与困惑度，来自 ch07-quant_hf.json")

    p = ev("ch08-prefix_cache")["summary"]
    FIGS["ch08-prefix"] = svg(720, 250, bars(160, 50, 400, 150, ["关闭缓存", "命中缓存"],
        [("首 token p50", [p["off"]["p50_ms"], p["on_hits"]["p50_ms"]])], "同一段约 1845 token 的前缀，首 token 延迟 p50", "毫秒（ms）", (A,)),
        "前缀缓存对首 token 延迟的影响", "关闭与命中前缀缓存时首 token 延迟 p50，来自 ch08-prefix_cache.json")

    c = ev("ch09-concurrency")["results"]
    xs = [r["concurrency"] for r in c["vllm"]["rows"]]
    FIGS["ch09-concurrency"] = svg(760, 300, _conc(xs, c),
        "吞吐与首 token 延迟随并发的变化", "vLLM 与串行服务的总吞吐、首 token 延迟 p50 随并发的变化，来自 ch09-concurrency.json")

    g, gl = ev("ch11-guardrails"), ev("ch11-guardrails_large")
    FIGS["ch11-burst"] = svg(720, 250, bars(120, 50, 480, 150, ["135M / Mac", "1.7B / RTX 4080 SUPER"],
        [("无护栏", [g["baseline_burst"]["max_latency_s"], gl["baseline_burst"]["max_latency_s"]]),
         ("有护栏", [g["burst"]["max_latency_s"], gl["burst"]["max_latency_s"]])], "24 个并发请求下最慢的一个响应多久", "秒（s）"),
        "护栏对最坏延迟的影响", "有无护栏时 24 并发突发下的最大响应时间，来自 ch11 证据")


def _conc(xs, c):
    v = {r["concurrency"]: r for r in c["vllm"]["rows"]}
    h = {r["concurrency"]: r for r in c["hf"]["rows"]}
    pair = lambda key: [("vLLM", [v[x][key] for x in xs]), ("串行服务", [h[x][key] if x in h else None for x in xs])]
    return (lines(70, 60, 280, 180, xs, pair("output_tokens_per_s"), "总吞吐 vs 并发", "tok/s") +
            lines(430, 60, 280, 180, xs, pair("ttft_p50_s"), "首 token 延迟 p50 vs 并发", "秒") +
            t(390, 288, "横轴：并发数（串行服务只测到 16）；纵轴为对数刻度", 11, "middle", MUTED))


def schematics():
    # ch02：Prefill 与 Decode
    b = box(20, 70, 130, 56, "提示词", "N 个 token") + box(200, 70, 150, 56, "Prefill", "一次并行处理 N 个", fill="#4f7cff22") \
        + box(400, 70, 130, 56, "第 1 个 token", "同时产出 KV Cache") + box(560, 70, 140, 56, "Decode", "每步 1 个 token，循环", fill="#f0a03022", stroke=B)
    b += arrow(150, 98, 198, 98) + arrow(350, 98, 398, 98) + arrow(530, 98, 558, 98)
    b += f'<path d="M630 128 C630 175 630 175 630 175 L630 175 C560 175 560 135 600 130" fill="none" stroke="{B}" stroke-width="1.6" stroke-dasharray="4 3"/>'
    b += t(630, 200, "读取并追加 KV Cache，直到遇到停止条件", 11, "middle", MUTED)
    FIGS["ch02-phases"] = svg(720, 220, b, "Prefill 与 Decode", "提示词先经过 Prefill 并行处理，随后 Decode 每步生成一个 token 并循环")

    # ch04：请求时序
    ys = [70, 110, 150, 190, 230]
    b = box(30, 20, 120, 34, "客户端", stroke=A) + box(300, 20, 120, 34, "FastAPI 服务", stroke=A) + box(570, 20, 120, 34, "Engine", stroke=A)
    for x in (90, 360, 630):
        b += f'<line x1="{x}" y1="54" x2="{x}" y2="262" stroke="{GRID}" stroke-width="2" stroke-dasharray="4 4"/>'
    b += arrow(90, 80, 358, 80, "POST /v1/chat/completions  stream:true") + arrow(360, 115, 628, 115, "加锁 → Prefill")
    b += arrow(628, 150, 362, 150, "token 1") + arrow(358, 150, 92, 150, "data: {...}")
    b += arrow(628, 185, 362, 185, "token 2…N") + arrow(358, 185, 92, 185, "data: {...} ×N")
    b += arrow(358, 235, 92, 235, "data: [DONE]")
    FIGS["ch04-sequence"] = svg(720, 275, b, "流式请求时序", "客户端发起流式请求，服务调用引擎，每产生一个 token 就立刻以 SSE 发回客户端")

    # ch10：选型流程
    b = box(240, 10, 240, 40, "你的瓶颈是什么？") 
    items = [(10, "单用户、吞吐够用", "留在 Transformers"), (190, "并发高 / 首字慢", "考虑 vLLM（ch09）"), (370, "显存放不下", "量化（ch07）"), (550, "大量共享前缀", "前缀缓存（ch08）")]
    b2 = ""
    for x, a, c in items:
        b2 += arrow(360, 50, x + 80, 98) + box(x, 100, 160, 44, a) + arrow(x + 80, 144, x + 80, 178) + box(x, 180, 160, 44, c, fill="#4f7cff22")
    b2 += t(360, 258, "无论选哪个：先照第 09 章在自己的硬件和流量上测一遍", 12, "middle", MUTED)
    FIGS["ch10-decision"] = svg(720, 270, b + b2, "引擎选型流程", "按瓶颈类型选择：单用户留在 Transformers，并发用 vLLM，显存不足用量化，共享前缀用前缀缓存")

    # ch11：服务架构
    b = box(10, 95, 90, 50, "客户端") + box(140, 95, 120, 50, "/readyz", "未就绪 → 503", stroke=B) \
        + box(300, 95, 130, 50, "准入控制", "超限 → 429", stroke=B) + box(470, 95, 120, 50, "排队", "超时 → 503", stroke=B) \
        + box(620, 95, 90, 50, "引擎", "超时 → 截断", fill="#4f7cff22")
    for a, c in ((100, 140), (260, 300), (430, 470), (590, 620)):
        b += arrow(a, 120, c, 120)
    b += box(140, 10, 570, 40, "守护进程 supervisor", "子进程退出就重启；SIGTERM 时先送完在途请求", stroke=MUTED)
    b += box(300, 190, 290, 40, "JSON 日志 + /metrics", "每个请求带 request_id", stroke=MUTED)
    b += arrow(365, 145, 365, 188, color=MUTED)
    FIGS["ch11-arch"] = svg(720, 250, b, "稳定服务架构", "请求依次经过就绪检查、准入控制、排队与引擎，由守护进程托管并输出日志和指标")

    # ch12：交付流
    b = box(10, 20, 150, 50, "代码 + 锁定依赖", "requirements / revision") + box(10, 110, 150, 50, "实验断言", "失败即退出非 0") \
        + box(230, 65, 150, 50, "evidence/public", "原始结果 JSON", fill="#4f7cff22") + box(450, 20, 130, 50, "评估报告", "程序生成") + box(450, 110, 130, 50, "洁净室复现", "新克隆 + 新 venv", stroke=B)
    b += arrow(160, 45, 230, 82) + arrow(160, 135, 230, 100) + arrow(380, 82, 450, 45) + arrow(380, 100, 450, 135)
    b += box(620, 65, 90, 50, "可交付", fill="#f0a03022", stroke=B) + arrow(580, 45, 620, 82) + arrow(580, 135, 620, 100)
    FIGS["ch12-delivery"] = svg(720, 190, b, "交付流程", "代码和断言产生证据，证据生成报告并经洁净室复现，通过后才算可交付")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    data_figs()
    schematics()
    for name, s in FIGS.items():
        (OUT / f"{name}.svg").write_text(s)
    print(f"wrote {len(FIGS)} figures to {OUT}")


if __name__ == "__main__":
    main()
