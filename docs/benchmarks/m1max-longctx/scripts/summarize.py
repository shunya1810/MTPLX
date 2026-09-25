#!/usr/bin/env python3
"""Summarize results/raw-cells.jsonl into results/summary.{json,csv} and charts/*.svg.

Per cell (one fresh server, N identical requests):
  cold  = request 1, full prefill in a fresh process
          TTFT and E2E (client wall clock), prefill tok/s (server
          prompt_tokens / prompt_eval_time_s), peak memory (server
          peak_memory_bytes after request 1)
  decode = median of the server's decode_tok_s over all N requests
           (requests 2..N hit the RAM prefix cache; same KV length, same text)

Charts are static SVGs in a light and a dark variant (README switches them
with <picture>). No third-party dependencies.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CTX_ORDER = ["2k", "4k", "8k", "16k", "32k", "64k", "128k", "256k"]
KV_LABEL = {"off": "fp16 KV", "q8": "q8 KV", "q4": "q4 KV"}
GB = 1e9
PHYSICAL_RAM_GB = 68719476736 / GB
WORKING_SET_GB = 55662788608 / GB  # MTLDevice.recommendedMaxWorkingSetSize on this machine

# 256K was not re-measured on the baseline commit. Earlier-session numbers, for
# reference only (older commit, manual prefill chunk 512; at the default chunk
# of 2048 the baseline ran out of memory ~48 min into the prefill).
REFERENCE_256K_BASELINE = {
    "q8": {"prefill_s": 5304.0, "decode": 4.86, "peak_gb": 52.34},
    "off": {"prefill_s": 5000.0, "decode": 3.54, "peak_gb": 57.34},
}


def load_rows(path: Path) -> list[dict]:
    latest = {}
    for line in path.read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            latest[r["cell"]] = r  # a re-run of a cell replaces the earlier row
    return list(latest.values())


def cell_summary(r: dict) -> dict:
    reqs = r["requests"]
    first = reqs[0] if reqs else {}
    dec = [q["decode_tok_s"] for q in reqs if q.get("decode_tok_s")]
    shas = {q["text_sha256"] for q in reqs}
    acc = sum(q.get("accepted_drafts") or 0 for q in reqs)
    drafted = sum(q.get("drafted_tokens") or 0 for q in reqs)
    pe = first.get("prompt_eval_time_s")
    ttft = first.get("client_ttft_s")
    e2e = first.get("client_wall_s")
    return {
        "cell": r["cell"], "arm": r["arm"], "commit": r["commit"], "context": r["context"], "kv": r["kv"],
        "prompt_tokens": first.get("prompt_tokens"), "completion_tokens": first.get("completion_tokens"),
        "requests": len(reqs),
        "cold_prefill_s": pe,
        "cold_prefill_tok_s": (first["prompt_tokens"] / pe) if pe and first.get("prompt_tokens") else None,
        "cold_ttft_s": ttft,
        "cold_e2e_s": e2e,
        "cold_decode_phase_s": (e2e - ttft) if e2e and ttft else None,
        "decode_tok_s_median": statistics.median(dec) if dec else None,
        "decode_tok_s_min": min(dec) if dec else None,
        "decode_tok_s_max": max(dec) if dec else None,
        "decode_tok_s_all": dec,
        "draft_acceptance": acc / drafted if drafted else None,
        "peak_mlx_gb": (first.get("peak_memory_bytes") or 0) / GB or None,
        "max_wired_gb": (r.get("max_wired_bytes") or 0) / GB or None,
        "max_swap_used_mb": r.get("max_swap_used_mb"),
        "prefill_chunk_tokens": r.get("prefill_chunk_tokens"),
        "compiled_verify_fallbacks": sum(q.get("compiled_verify_fallbacks") or 0 for q in reqs),
        "finish_reason": first.get("finish_reason"),
        "output_sha256": sorted(shas)[0] if len(shas) == 1 else None,
        "outputs_identical_within_cell": len(shas) == 1,
        "error": r.get("error") or next((q["client_error"] for q in reqs if q.get("client_error")), None),
    }


def find(cells, ctx, kv, arm):
    return next((c for c in cells if c["context"] == ctx and c["kv"] == kv and c["arm"] == arm), None)


def build(rows: list[dict]) -> dict:
    cells = sorted((cell_summary(r) for r in rows),
                   key=lambda c: (CTX_ORDER.index(c["context"]), c["kv"], c["arm"]))
    pairs = []
    ratio = lambda x, y: (x / y) if x and y else None
    for ctx in CTX_ORDER:
        for kv in ("off", "q8"):
            b, o = find(cells, ctx, kv, "baseline"), find(cells, ctx, kv, "optimized")
            if not (b and o):
                continue
            pairs.append({
                "context": ctx, "kv": kv, "prompt_tokens": o["prompt_tokens"],
                "completion_tokens": o["completion_tokens"],
                **{f"{side}_{k}": c[k] for side, c in (("baseline", b), ("optimized", o))
                   for k in ("cold_prefill_tok_s", "cold_ttft_s", "decode_tok_s_median", "cold_e2e_s", "peak_mlx_gb")},
                "decode_speedup": ratio(o["decode_tok_s_median"], b["decode_tok_s_median"]),
                "prefill_speedup": ratio(o["cold_prefill_tok_s"], b["cold_prefill_tok_s"]),
                "ttft_ratio": ratio(o["cold_ttft_s"], b["cold_ttft_s"]),
                "e2e_ratio": ratio(o["cold_e2e_s"], b["cold_e2e_s"]),
                "peak_delta_gb": (o["peak_mlx_gb"] - b["peak_mlx_gb"]) if o["peak_mlx_gb"] and b["peak_mlx_gb"] else None,
                "outputs_identical_across_arms": bool(b["output_sha256"]) and b["output_sha256"] == o["output_sha256"],
            })
    return {"cells": cells, "pairs": pairs, "reference_256k_baseline": REFERENCE_256K_BASELINE}


def fmt(v, nd=1, suffix=""):
    return "—" if v is None else f"{v:,.{nd}f}{suffix}"


def fmt_s(v):
    if v is None:
        return "—"
    return f"{v:,.1f} s" if v < 100 else (f"{v:,.0f} s" if v < 600 else f"{v / 60:,.1f} min")


def pct(new, old, lower_is_better=False):
    if not new or not old:
        return ""
    d = (new / old - 1) * 100
    return f" ({d:+.0f}%)"


HEADERS = {
    "en": ("| context | prompt tokens | prefill tok/s | TTFT (cold) | decode tok/s | E2E (cold) | peak memory | same output |",
           "q4 KV (optimized only, 256K)"),
    "ja": ("| context | プロンプト token 数 | prefill tok/s | TTFT（コールド） | decode tok/s | E2E（コールド） | ピークメモリ | 出力一致 |",
           "q4 KV（optimized のみ、256K）"),
}


def tables_md(s: dict, lang: str = "en") -> str:
    head, q4_title = HEADERS[lang]
    out = []
    for kv in ("off", "q8"):
        out += [f"### {KV_LABEL[kv]}", "",
                head, "|---|---:|---:|---:|---:|---:|---:|:-:|"]
        for ctx in CTX_ORDER:
            b, o = find(s["cells"], ctx, kv, "baseline"), find(s["cells"], ctx, kv, "optimized")
            if not o:
                continue
            if b:
                same = "✅" if b["output_sha256"] and b["output_sha256"] == o["output_sha256"] else "❌"
                out.append(
                    f"| {ctx.upper()} | {o['prompt_tokens']:,} "
                    f"| {fmt(b['cold_prefill_tok_s'])} → **{fmt(o['cold_prefill_tok_s'])}**{pct(o['cold_prefill_tok_s'], b['cold_prefill_tok_s'])} "
                    f"| {fmt_s(b['cold_ttft_s'])} → **{fmt_s(o['cold_ttft_s'])}**{pct(o['cold_ttft_s'], b['cold_ttft_s'])} "
                    f"| {fmt(b['decode_tok_s_median'], 2)} → **{fmt(o['decode_tok_s_median'], 2)}**{pct(o['decode_tok_s_median'], b['decode_tok_s_median'])} "
                    f"| {fmt_s(b['cold_e2e_s'])} → **{fmt_s(o['cold_e2e_s'])}**{pct(o['cold_e2e_s'], b['cold_e2e_s'])} "
                    f"| {fmt(b['peak_mlx_gb'])} → **{fmt(o['peak_mlx_gb'])} GB** "
                    f"| {same} |")
            else:
                ref = REFERENCE_256K_BASELINE.get(kv) if ctx == "256k" else None
                r = (lambda v: f"<sub>ref.</sub> {v} → ") if ref else (lambda v: "")
                out.append(
                    f"| {ctx.upper()} | {o['prompt_tokens']:,} "
                    f"| {r(fmt(o['prompt_tokens'] / ref['prefill_s']) if ref else '')}**{fmt(o['cold_prefill_tok_s'])}** "
                    f"| {r(fmt_s(ref['prefill_s']) if ref else '')}**{fmt_s(o['cold_ttft_s'])}** "
                    f"| {r(fmt(ref['decode'], 2) if ref else '')}**{fmt(o['decode_tok_s_median'], 2)}** "
                    f"| **{fmt_s(o['cold_e2e_s'])}** "
                    f"| {r(fmt(ref['peak_gb']) if ref else '')}**{fmt(o['peak_mlx_gb'])} GB** | — |")
        out.append("")
    q4 = find(s["cells"], "256k", "q4", "optimized")
    if q4:
        out += [f"### {q4_title}", "",
                head.rsplit(" |", 2)[0] + " |",
                "|---|---:|---:|---:|---:|---:|---:|",
                f"| 256K | {q4['prompt_tokens']:,} | {fmt(q4['cold_prefill_tok_s'])} | {fmt_s(q4['cold_ttft_s'])} "
                f"| {fmt(q4['decode_tok_s_median'], 2)} | {fmt_s(q4['cold_e2e_s'])} | {fmt(q4['peak_mlx_gb'])} GB |", ""]
    return "\n".join(out)


def write_csv(cells: list[dict], path: Path) -> None:
    keys = [k for k in cells[0] if k != "decode_tok_s_all"] if cells else []
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        for c in cells:
            w.writerow({k: (round(v, 4) if isinstance(v, float) else v) for k, v in c.items() if k in keys})


# ---------------------------------------------------------------- SVG charts
THEMES = {
    "light": {"bg": "#fcfcfb", "t1": "#0b0b0b", "t2": "#52514e", "grid": "#e4e3de", "axis": "#8a8983",
              "opt": "#2a78d6", "base": "#eb6834", "ref": "#8a8983"},
    "dark": {"bg": "#1a1a19", "t1": "#ffffff", "t2": "#c3c2b7", "grid": "#34332f", "axis": "#6d6c66",
             "opt": "#3987e5", "base": "#d95926", "ref": "#8a8983"},
}
FONT = "-apple-system,BlinkMacSystemFont,Segoe UI,Helvetica,Arial,sans-serif"


def esc(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def nice_step(span: float, target: int = 6) -> float:
    raw = span / target
    mag = 10 ** math.floor(math.log10(raw))
    return next(m * mag for m in (1, 2, 2.5, 5, 10) if m * mag >= raw)


def svg_open(W, H, th, title, subtitle):
    return [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" '
            f'role="img" aria-label="{esc(title)}" font-family="{FONT}">',
            f'<rect width="{W}" height="{H}" rx="8" fill="{th["bg"]}"/>',
            f'<text x="20" y="30" fill="{th["t1"]}" font-size="16" font-weight="600">{esc(title)}</text>',
            f'<text x="20" y="50" fill="{th["t2"]}" font-size="12">{esc(subtitle)}</text>']


def legend(o, th, items, x, y, width):
    """items: (label, color, dash, marker) — marker 'dot' | 'hollow' | 'diamond' | 'swatch' | 'swatch-light'.
    Wraps to a new row past `width`; returns the y of the last row."""
    x0 = x
    for label, color, dash, marker in items:
        w = 36 + 6.4 * len(label)
        if x + w > width and x > x0:
            x, y = x0, y + 20
        if marker.startswith("swatch"):
            op = ' fill-opacity="0.4"' if marker == "swatch-light" else ""
            o.append(f'<rect x="{x + 3}" y="{y - 10}" width="16" height="11" rx="2" fill="{color}"{op}/>')
        else:
            if marker == "dot":
                o.append(f'<line x1="{x}" x2="{x + 22}" y1="{y - 4}" y2="{y - 4}" stroke="{color}" stroke-width="2" '
                         f'stroke-dasharray="{dash or "none"}"/>')
            o.append(mark(x + 11, y - 4, color, th, marker))
        o.append(f'<text x="{x + 28}" y="{y}" fill="{th["t2"]}" font-size="12">{esc(label)}</text>')
        x += w
    return y


def mark(cx, cy, color, th, kind="dot"):
    if kind == "hollow":
        return f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="4.5" fill="{th["bg"]}" stroke="{color}" stroke-width="2"/>'
    if kind == "diamond":
        return (f'<path d="M{cx:.1f} {cy - 6:.1f} L{cx + 6:.1f} {cy:.1f} L{cx:.1f} {cy + 6:.1f} L{cx - 6:.1f} {cy:.1f}Z" '
                f'fill="{color}" stroke="{th["bg"]}" stroke-width="2"/>')
    return f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="4.5" fill="{color}" stroke="{th["bg"]}" stroke-width="2"/>'


def line_chart(th, title, subtitle, ylabel, xcats, xtok, series, *, logy=False, yfmt=None, vfmt=None,
               hlines=(), extra_points=(), ymin=None, ymax=None, legend_items=None):
    """series: dict(name, color, dash, y[list aligned with xcats], label_end)."""
    W, H, L, R, B = 900, 470, 70, 230, 52
    o = svg_open(W, H, th, title, subtitle)
    T = legend(o, th, legend_items or [(s["name"], s["color"], s.get("dash"), "dot") for s in series], 20, 78, W - 20) + 26
    pw, ph = W - L - R, H - T - B
    vals = [v for s in series for v in s["y"] if v] + [p[2] for p in extra_points] + [h[0] for h in hlines]
    yfmt = yfmt or (lambda v: f"{v:g}")
    vfmt = vfmt or yfmt
    if logy:
        lo = 10 ** math.floor(math.log10(min(vals)))
        hi = 10 ** math.ceil(math.log10(max(vals)))
        ticks, v = [], lo
        while v <= hi * 1.001:
            ticks.append(v)
            v *= 10
        fy = lambda v: T + ph * (1 - (math.log10(v) - math.log10(lo)) / (math.log10(hi) - math.log10(lo)))
    else:
        lo = 0 if ymin is None else ymin
        top = ymax or max(vals) * 1.1
        step = nice_step(top - lo)
        hi = lo + step * math.ceil((top - lo) / step)
        ticks = [lo + i * step for i in range(int(round((hi - lo) / step)) + 1)]
        fy = lambda v: T + ph * (1 - (v - lo) / (hi - lo))
    lx = [math.log2(t) for t in xtok]
    fx = lambda i: L + pw * (lx[i] - lx[0]) / (lx[-1] - lx[0])
    for t in ticks:
        o.append(f'<line x1="{L}" x2="{L + pw}" y1="{fy(t):.1f}" y2="{fy(t):.1f}" stroke="{th["grid"]}"/>'
                 f'<text x="{L - 8}" y="{fy(t) + 4:.1f}" fill="{th["t2"]}" font-size="11" text-anchor="end">{yfmt(t)}</text>')
    o.append(f'<line x1="{L}" x2="{L + pw}" y1="{T + ph}" y2="{T + ph}" stroke="{th["axis"]}"/>')
    for i, c in enumerate(xcats):
        o.append(f'<text x="{fx(i):.1f}" y="{T + ph + 18}" fill="{th["t2"]}" font-size="11" text-anchor="middle">{c}</text>')
    o.append(f'<text x="{L + pw / 2}" y="{H - 10}" fill="{th["t2"]}" font-size="11" text-anchor="middle">'
             f'prompt length (tokens, log scale)</text>')
    o.append(f'<text transform="translate(18 {T + ph / 2}) rotate(-90)" fill="{th["t2"]}" font-size="11" '
             f'text-anchor="middle">{esc(ylabel)}</text>')
    for yv, label in hlines:
        o.append(f'<line x1="{L}" x2="{L + pw}" y1="{fy(yv):.1f}" y2="{fy(yv):.1f}" stroke="{th["ref"]}" '
                 f'stroke-width="1.5" stroke-dasharray="2 4"/>'
                 f'<text x="{L + 6}" y="{fy(yv) - 5:.1f}" fill="{th["t2"]}" font-size="11">{esc(label)}</text>')
    ends = []
    for s in series:
        pts = [(fx(i), fy(v), v) for i, v in enumerate(s["y"]) if v]
        if not pts:
            continue
        dash = f' stroke-dasharray="{s["dash"]}"' if s.get("dash") else ""
        o.append(f'<polyline fill="none" stroke="{s["color"]}" stroke-width="2"{dash} stroke-linejoin="round" '
                 f'points="{" ".join(f"{a:.1f},{b:.1f}" for a, b, _ in pts)}"/>')
        o.extend(mark(a, b, s["color"], th) for a, b, _ in pts)
        if s.get("label_end", True):
            ends.append([pts[-1][1], pts[-1][0], f'{s["name"]} {vfmt(pts[-1][2])}'])
    for i_ctx, color, v, kind, label in extra_points:
        o.append(mark(fx(i_ctx), fy(v), color, th, kind))
        if label:
            ends.append([fy(v), fx(i_ctx), label])
    ends.sort(key=lambda e: e[0])
    for i in range(1, len(ends)):
        ends[i][0] = max(ends[i][0], ends[i - 1][0] + 15)
    for ly, x0, text in ends:
        o.append(f'<text x="{x0 + 10:.1f}" y="{ly + 4:.1f}" fill="{th["t1"]}" font-size="11" font-weight="600">{esc(text)}</text>')
    o.append("</svg>")
    return "\n".join(o)


def e2e_chart(th, s, kv):
    """Horizontal stacked bars: cold prefill (TTFT) + decode phase, baseline vs optimized per context."""
    rows = []
    for ctx in CTX_ORDER:
        for arm in ("baseline", "optimized"):
            c = find(s["cells"], ctx, kv, arm)
            if c and c["cold_e2e_s"]:
                rows.append((ctx, arm, c["cold_ttft_s"], c["cold_decode_phase_s"], c["cold_e2e_s"], c["completion_tokens"]))
            elif ctx == "256k" and arm == "baseline":
                rows.append((ctx, arm, None, None, None, None))
    if not rows:
        return None
    W, L, R, T, rowh, gap = 820, 150, 150, 92, 16, 10
    H = T + len(rows) * (rowh + 4) + (len(rows) // 2) * gap + 50
    pw = W - L - R
    top = max(r[4] for r in rows if r[4]) / 60
    step = nice_step(top, 6) * 60
    hi = step * math.ceil(top * 60 / step)
    fx = lambda v: L + pw * v / hi
    o = svg_open(W, H, th, f"End-to-end latency, cold request ({KV_LABEL[kv]})",
                 "Fresh server · prefill (= time to first token), then generation (≤ 256 tokens) · lower is better")
    legend(o, th, [("baseline: prefill", th["base"], None, "swatch"), ("generation", th["base"], None, "swatch-light"),
                   ("optimized: prefill", th["opt"], None, "swatch"), ("generation", th["opt"], None, "swatch-light")],
           20, 78, W - 20)
    ybot = H - 50
    t = 0.0
    while t <= hi + 1e-9:
        o.append(f'<line x1="{fx(t):.1f}" x2="{fx(t):.1f}" y1="{T - 6}" y2="{ybot}" stroke="{th["grid"]}"/>'
                 f'<text x="{fx(t):.1f}" y="{ybot + 16}" fill="{th["t2"]}" font-size="11" text-anchor="middle">{round(t / 60, 2):g}</text>')
        t += step
    o.append(f'<text x="{L + pw / 2}" y="{H - 12}" fill="{th["t2"]}" font-size="11" text-anchor="middle">minutes</text>')
    y = T
    prev_ctx = None
    for ctx, arm, pre, dec, tot, ntok in rows:
        if prev_ctx and ctx != prev_ctx:
            y += gap
        color = th["opt"] if arm == "optimized" else th["base"]
        label = f"{ctx.upper()} {arm}" if arm == "baseline" or ctx != prev_ctx else f"{arm}"
        o.append(f'<text x="{L - 8}" y="{y + rowh - 4}" fill="{th["t2"]}" font-size="11" text-anchor="end">'
                 f'{esc((ctx.upper() + " ") if arm == "baseline" else "")}{arm}</text>')
        if tot is None:
            ref = REFERENCE_256K_BASELINE.get(kv)
            note = "not re-measured (default chunk: OOM during prefill)"
            if ref:
                note += f"; earlier session w/ chunk 512: {ref['prefill_s'] / 60:.0f} min prefill"
            o.append(f'<text x="{L + 4}" y="{y + rowh - 4}" fill="{th["t2"]}" font-size="11" font-style="italic">{esc(note)}</text>')
        else:
            w1 = max(fx(pre) - L, 1)
            w2 = max(fx(tot) - fx(pre) - 2, 1)
            o.append(f'<rect x="{L}" y="{y}" width="{w1:.1f}" height="{rowh}" rx="2" fill="{color}"/>')
            o.append(f'<rect x="{L + w1 + 2:.1f}" y="{y}" width="{w2:.1f}" height="{rowh}" rx="2" fill="{color}" fill-opacity="0.4"/>')
            txt = f"{tot:,.0f} s" if tot < 600 else f"{tot / 60:,.1f} min"
            o.append(f'<text x="{fx(tot) + 6:.1f}" y="{y + rowh - 4}" fill="{th["t1"]}" font-size="11" font-weight="600">{txt}</text>')
        y += rowh + 4
        prev_ctx = ctx
    o.append("</svg>")
    return "\n".join(o)


def charts(s: dict, out_dir: Path) -> list[str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    ctxs = [c for c in CTX_ORDER if any(x["context"] == c for x in s["cells"])]
    if len(ctxs) < 2:
        return []
    xtok = []
    for ctx in ctxs:
        toks = [c["prompt_tokens"] for c in s["cells"] if c["context"] == ctx and c["prompt_tokens"]]
        xtok.append(statistics.median(toks))
    xl = [c.upper() for c in ctxs]
    i256 = ctxs.index("256k") if "256k" in ctxs else None
    written = []
    for mode, th in THEMES.items():
        def get(arm, kv, key):
            return [(find(s["cells"], ctx, kv, arm) or {}).get(key) for ctx in ctxs]

        def series(key, fmt):
            return [
                {"name": "optimized fp16", "color": th["opt"], "y": get("optimized", "off", key)},
                {"name": "optimized q8", "color": th["opt"], "dash": "6 4", "y": get("optimized", "q8", key)},
                {"name": "baseline fp16", "color": th["base"], "y": get("baseline", "off", key)},
                {"name": "baseline q8", "color": th["base"], "dash": "6 4", "y": get("baseline", "q8", key)},
            ]

        legend_items = [("optimized fp16 KV", th["opt"], None, "dot"), ("optimized q8 KV", th["opt"], "6 4", "dot"),
                        ("baseline fp16 KV", th["base"], None, "dot"), ("baseline q8 KV", th["base"], "6 4", "dot")]
        extra_leg = []
        def extras(key, ref_key):
            pts = []
            if i256 is None:
                return pts
            q4 = find(s["cells"], "256k", "q4", "optimized")
            if q4 and q4.get(key):
                pts.append((i256, th["opt"], q4[key], "diamond", None))
            if ref_key:
                for kv in ("q8", "off"):
                    ref = REFERENCE_256K_BASELINE[kv][ref_key]
                    pts.append((i256, th["base"], ref, "hollow", None))
            return pts

        leg_q4 = [("optimized q4 KV (256K)", th["opt"], None, "diamond")]
        leg_ref = [("baseline, earlier session (ref.)", th["base"], None, "hollow")]
        charts_spec = {
            "decode-vs-context": dict(
                title="Decode throughput vs prompt length",
                subtitle="Qwen3.8-27B FP16, MTP depth 3, M1 Max 64 GB · median of 3 requests (256K: 1) · higher is better",
                ylabel="decode tok/s", series=series("decode_tok_s_median", None), yfmt=lambda v: f"{v:g}",
                vfmt=lambda v: f"{v:.1f}",
                extra_points=extras("decode_tok_s_median", "decode"), legend_items=legend_items + leg_q4 + leg_ref),
            "ttft-vs-context": dict(
                title="Time to first token (cold prefill) vs prompt length",
                subtitle="Fresh server, whole prompt prefilled · log–log · lower is better",
                ylabel="seconds (log scale)", series=series("cold_ttft_s", None), logy=True,
                yfmt=lambda v: f"{v:,.0f}" if v >= 1 else f"{v:g}",
                vfmt=lambda v: f"{v:,.0f} s" if v < 600 else f"{v / 60:,.0f} min",
                extra_points=extras("cold_ttft_s", "prefill_s"), legend_items=legend_items + leg_q4 + leg_ref),
            "prefill-vs-context": dict(
                title="Prefill throughput vs prompt length",
                subtitle="prompt tokens / server prompt-eval time, cold request · higher is better",
                ylabel="prefill tok/s", series=series("cold_prefill_tok_s", None), yfmt=lambda v: f"{v:,.0f}",
                vfmt=lambda v: f"{v:,.0f}",
                extra_points=extras("cold_prefill_tok_s", None), legend_items=legend_items + leg_q4),
            "peak-memory-vs-context": dict(
                title="Peak memory vs prompt length",
                subtitle="MLX peak allocation reported by the server after the cold request · lower is better",
                ylabel="GB", series=series("peak_mlx_gb", None), yfmt=lambda v: f"{v:g}", vfmt=lambda v: f"{v:.1f} GB", ymin=0,
                ymax=PHYSICAL_RAM_GB * 1.02,
                hlines=[(PHYSICAL_RAM_GB, f"physical RAM {PHYSICAL_RAM_GB:.1f} GB"),
                        (WORKING_SET_GB, f"GPU recommended working set {WORKING_SET_GB:.1f} GB")],
                extra_points=extras("peak_mlx_gb", "peak_gb"), legend_items=legend_items + leg_q4 + leg_ref),
        }
        # the 256K reference prefill is in seconds; convert to tok/s-free keys
        for name, spec in charts_spec.items():
            svg = line_chart(th, spec.pop("title"), spec.pop("subtitle"), spec.pop("ylabel"), xl, xtok,
                             spec.pop("series"), **spec)
            p = out_dir / f"{name}-{mode}.svg"
            p.write_text(svg)
            written.append(p.name)
        for kv in ("off", "q8"):
            svg = e2e_chart(th, s, kv)
            if svg:
                p = out_dir / f"e2e-{kv}-{mode}.svg"
                p.write_text(svg)
                written.append(p.name)
    return written


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--raw", default=str(ROOT / "results" / "raw-cells.jsonl"))
    p.add_argument("--out-dir", default=str(ROOT / "results"))
    p.add_argument("--charts-dir", default=str(ROOT / "charts"))
    a = p.parse_args()
    s = build(load_rows(Path(a.raw)))
    out = Path(a.out_dir)
    (out / "summary.json").write_text(json.dumps(s, indent=1) + "\n")
    write_csv(s["cells"], out / "summary.csv")
    (out / "tables.md").write_text(tables_md(s))
    (out / "tables.ja.md").write_text(tables_md(s, "ja"))
    for name in charts(s, Path(a.charts_dir)):
        print("wrote", name)
    for c in s["cells"]:
        print(f'{c["cell"]:24} prompt={c["prompt_tokens"]} ttft={c["cold_ttft_s"]} decode={c["decode_tok_s_median"]} '
              f'peak={c["peak_mlx_gb"]} sha={(c["output_sha256"] or "DIFF")[:10]} err={c["error"]}')


if __name__ == "__main__":
    main()
