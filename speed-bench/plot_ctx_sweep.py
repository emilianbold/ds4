#!/usr/bin/env python3
"""Plot a session_concurrency_bench sweep: aggregate decode t/s and session
memory by context size, one line per concurrency level.

    python3 speed-bench/plot_ctx_sweep.py speed-bench/m5_max_ctx_sweep \
        --title "M5 Max, Qwen3.8 Flash Next"

reads <prefix>_plain.csv, <prefix>_spec.csv (batched MTP, dashed) and, when
present, <prefix>_upstream_plain.csv / <prefix>_upstream_spec.csv (a single
reference stream, orange), and writes <prefix>_throughput.svg and
<prefix>_memory.svg.  Cells whose status is not "ok" (no memory fit) leave
a gap.  Same look as plot_speed.py.
"""

import argparse
import csv
import html
import math
from pathlib import Path

TEXT_COLOR = "#1f2933"
MUTED_COLOR = "#64748b"
GRID_COLOR = "#e2e8f0"
AXIS_COLOR = "#334155"
RAMP = {1: "#93c5fd", 2: "#60a5fa", 4: "#2563eb", 8: "#1d4ed8", 16: "#1e3a8a"}
UPSTREAM_COLOR = "#ea580c"

W, H = 960, 540
LEFT, RIGHT, TOP, BOTTOM = 82, 96, 100, 72


def read(path):
    """{concurrency: {ctx: row}} of the ok cells."""
    out = {}
    if not path.exists():
        return out
    with path.open("r", encoding="utf-8-sig", newline="") as fp:
        for row in csv.DictReader(fp):
            if row["status"] != "ok":
                continue
            out.setdefault(int(row["concurrency"]), {})[int(row["ctx"])] = row
    return out


def nice_ceil(value):
    if value <= 0:
        return 1.0
    magnitude = 10 ** math.floor(math.log10(value))
    for step in (1, 2, 2.5, 3, 4, 5, 6, 8, 10):
        if value / magnitude <= step:
            return step * magnitude
    return 10 * magnitude


def nice_step(span, target):
    raw = span / target
    magnitude = 10 ** math.floor(math.log10(raw))
    for step in (1, 2, 2.5, 5, 10):
        if raw / magnitude <= step:
            return step * magnitude
    return 10 * magnitude


def fmt(value):
    return f"{value / 1000:g}k" if abs(value) >= 1000 else f"{value:g}"


def fmt_ctx(ctx):
    return f"{ctx // 1024}k"


def svg(series, ctxs, title, y_label, unit, out):
    """series: list of (label, color, dashed, {ctx: value})."""
    y_max = nice_ceil(max(v for _, _, _, pts in series for v in pts.values()) * 1.08)
    step = nice_step(y_max, 6)
    lo, hi = math.log2(min(ctxs)), math.log2(max(ctxs))
    plot_w, plot_h = W - LEFT - RIGHT, H - TOP - BOTTOM

    def x_of(ctx):
        return LEFT + (math.log2(ctx) - lo) / (hi - lo) * plot_w

    def y_of(v):
        return TOP + plot_h - v / y_max * plot_h

    parts = [f'<?xml version="1.0" encoding="UTF-8"?>',
             f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">',
             "<style>text { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; }"
             f" .title {{ font-size: 22px; font-weight: 700; fill: {TEXT_COLOR}; }}"
             f" .axis-label {{ font-size: 14px; font-weight: 600; fill: {AXIS_COLOR}; }}"
             f" .tick {{ font-size: 12px; fill: {MUTED_COLOR}; }}"
             f" .legend {{ font-size: 12px; font-weight: 600; fill: {TEXT_COLOR}; }}</style>",
             f'<rect width="{W}" height="{H}" fill="#ffffff"/>',
             f'<text class="title" x="{W / 2:.1f}" y="34" text-anchor="middle">{html.escape(title)}</text>']
    v = 0.0
    while v <= y_max + 1e-9:
        y = y_of(v)
        parts.append(f'<line x1="{LEFT}" y1="{y:.2f}" x2="{W - RIGHT}" y2="{y:.2f}" stroke="{GRID_COLOR}" stroke-width="1"/>')
        parts.append(f'<text class="tick" x="{LEFT - 10}" y="{y + 4:.2f}" text-anchor="end">{fmt(v)}</text>')
        v += step
    for ctx in ctxs:
        x = x_of(ctx)
        parts.append(f'<line x1="{x:.2f}" y1="{TOP}" x2="{x:.2f}" y2="{TOP + plot_h}" stroke="{GRID_COLOR}" stroke-width="1"/>')
        parts.append(f'<text class="tick" x="{x:.2f}" y="{TOP + plot_h + 18}" text-anchor="middle">{fmt_ctx(ctx)}</text>')
    parts.append(f'<line x1="{LEFT}" y1="{TOP + plot_h}" x2="{W - RIGHT}" y2="{TOP + plot_h}" stroke="{AXIS_COLOR}" stroke-width="1.2"/>')
    parts.append(f'<line x1="{LEFT}" y1="{TOP}" x2="{LEFT}" y2="{TOP + plot_h}" stroke="{AXIS_COLOR}" stroke-width="1.2"/>')
    parts.append(f'<text class="axis-label" x="{LEFT + plot_w / 2:.1f}" y="{H - 30}" text-anchor="middle">context size (tokens, log scale)</text>')
    parts.append(f'<text class="axis-label" x="20" y="{TOP + plot_h / 2:.1f}" text-anchor="middle" transform="rotate(-90 20 {TOP + plot_h / 2:.1f})">{html.escape(y_label)}</text>')
    for label, color, dashed, pts in series:
        dash = ' stroke-dasharray="7 4"' if dashed else ""
        d = " ".join(f"{'M' if i == 0 else 'L'}{x_of(ctx):.2f},{y_of(pts[ctx]):.2f}" for i, ctx in enumerate(sorted(pts)))
        parts.append(f'<path d="{d}" fill="none" stroke="{color}" stroke-width="2.2"{dash} stroke-linejoin="round"/>')
        for ctx in sorted(pts):
            parts.append(f'<circle cx="{x_of(ctx):.2f}" cy="{y_of(pts[ctx]):.2f}" r="3.2" fill="{color}"/>')
        if not dashed:
            last = max(pts)
            parts.append(f'<text class="tick" x="{x_of(last) + 6:.2f}" y="{y_of(pts[last]) + 4:.2f}" fill="{color}">{fmt(round(pts[last], 1))}{unit}</text>')
    # legend: rows between the title and the plot
    x, y = LEFT + 6, 62
    for label, color, dashed, _ in series:
        dash = ' stroke-dasharray="7 4"' if dashed else ""
        parts.append(f'<line x1="{x}" y1="{y - 4}" x2="{x + 26}" y2="{y - 4}" stroke="{color}" stroke-width="2.5"{dash}/>')
        parts.append(f'<text class="legend" x="{x + 32}" y="{y}">{html.escape(label)}</text>')
        x += 32 + 7 * len(label) + 22
        if x > W - RIGHT - 150:
            x, y = LEFT + 6, y + 17
    parts.append("</svg>")
    out.write_text("\n".join(parts) + "\n", encoding="utf-8")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("prefix", help="CSV prefix, e.g. speed-bench/m5_max_ctx_sweep")
    ap.add_argument("--title", default="")
    args = ap.parse_args()
    prefix = Path(args.prefix)
    plain = read(Path(f"{prefix}_plain.csv"))
    spec = read(Path(f"{prefix}_spec.csv"))
    up_plain = read(Path(f"{prefix}_upstream_plain.csv"))
    up_spec = read(Path(f"{prefix}_upstream_spec.csv"))
    ctxs = sorted({ctx for d in (plain, spec, up_plain, up_spec) for c in d for ctx in d[c]})
    concs = sorted({c for d in (plain, spec) for c in d})

    def col(d, key):
        return {ctx: float(r[key]) for ctx, r in d.items()}

    tps = [(f"C={c} plain", RAMP.get(c, TEXT_COLOR), False, col(plain[c], "decode_agg_tps")) for c in concs if c in plain]
    tps += [(f"C={c} MTP", RAMP.get(c, TEXT_COLOR), True, col(spec[c], "decode_agg_tps")) for c in concs if c in spec]
    if 1 in up_plain:
        tps.append(("upstream C=1 plain", UPSTREAM_COLOR, False, col(up_plain[1], "decode_agg_tps")))
    if 1 in up_spec:
        tps.append(("upstream C=1 MTP", UPSTREAM_COLOR, True, col(up_spec[1], "decode_agg_tps")))
    svg(tps, ctxs, f"{args.title}: decode t/s by context".strip(": "),
        "aggregate decode t/s (all streams)", "", Path(f"{prefix}_throughput.svg"))
    mem = [(f"C={c} plain", RAMP.get(c, TEXT_COLOR), False, col(plain[c], "predicted_gib")) for c in concs if c in plain]
    mem += [(f"C={c} MTP", RAMP.get(c, TEXT_COLOR), True, col(spec[c], "predicted_gib")) for c in concs if c in spec]
    svg(mem, ctxs, f"{args.title}: session memory by context".strip(": "),
        "session state, GiB (caches + shared arena; model excluded)", " GiB", Path(f"{prefix}_memory.svg"))


if __name__ == "__main__":
    main()
