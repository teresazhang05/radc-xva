#!/usr/bin/env python3
"""Generate required figures from metrics_merged.csv with correct semantics.

Outputs in <run_dir>/figures/:
- fig1_bytes_vs_rmax.pdf
- fig2_latency_quantiles.pdf
- fig3_xva_error_cdf.pdf
- fig4_fallback_vs_shock.pdf

Corrections vs prior version:
- Latency quantiles computed from epoch-wise max across ranks (exclude warmup and shadow).
- fig4 plots fallback rate vs shock magnitude (shock_sigma if present; else delta_m_sigma; else falls back to rho_hat_sketch).
"""

import argparse
import csv
import math
import os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def read_rows(path: str):
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def to_float(v, default=float("nan")) -> float:
    try:
        if v in ("", "NaN", "nan", None):
            return default
        return float(v)
    except Exception:
        return default


def to_int(v, default=0) -> int:
    try:
        if v in ("", "NaN", "nan", None):
            return default
        return int(float(v))
    except Exception:
        return default


def epoch_max_latency(rows, mode_pred, warmup_epochs: int):
    by_epoch = {}
    for r in rows:
        if not mode_pred(r):
            continue
        ep = to_int(r.get("epoch"), 0)
        if ep < warmup_epochs:
            continue
        if r.get("is_shadow_epoch") == "1":
            continue
        t = to_float(r.get("t_epoch_total_ms"))
        if not math.isfinite(t):
            continue
        prev = by_epoch.get(ep)
        if prev is None or t > prev:
            by_epoch[ep] = t
    return list(by_epoch.values())


def q_nearest_rank(vals, q: float):
    xs = [v for v in vals if math.isfinite(v)]
    if not xs:
        return float("nan")
    xs.sort()
    n = len(xs)
    idx = int(math.ceil(q * n)) - 1
    idx = max(0, min(idx, n - 1))
    return xs[idx]


def fig1(rows, out_path):
    # Bytes vs rank used (fallback to l_used) and compressed-attempt payload bytes.
    xs = []
    ys = []
    for r in rows:
        if not (r.get("mode") or "").startswith("compressed_"):
            continue
        x = to_int(r.get("r_used"), -1)
        if x < 0:
            x = to_int(r.get("l_used"), to_int(r.get("l"), 0))
        y = to_float(r.get("bytes_comp_attempt_payload"))
        if not math.isfinite(y):
            y = to_float(r.get("bytes_comp_payload"))
        xs.append(x)
        ys.append(y)
    if not xs:
        xs = [0]
        ys = [0.0]
    plt.figure(figsize=(6, 4))
    plt.scatter(xs, ys, s=10, alpha=0.5)
    plt.xlabel("r_used (fallback: l_used)")
    plt.ylabel("compressed-attempt payload bytes")
    plt.title("Bytes vs adaptive rank")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def fig2(rows, out_path, warmup_epochs: int):
    exact_vals = epoch_max_latency(rows, lambda r: r.get("mode") == "exact", warmup_epochs)
    meth_vals = epoch_max_latency(rows, lambda r: (r.get("mode") or "").startswith("compressed_"), warmup_epochs)
    if not exact_vals:
        exact_vals = [0.0]
    if not meth_vals:
        meth_vals = [0.0]

    q_exact = [q_nearest_rank(exact_vals, q) for q in (0.50, 0.95, 0.99)]
    q_meth = [q_nearest_rank(meth_vals, q) for q in (0.50, 0.95, 0.99)]

    labels = ["p50", "p95", "p99"]
    x = [0, 1, 2]
    w = 0.35
    plt.figure(figsize=(6, 4))
    plt.bar([i - w / 2 for i in x], q_exact, width=w, label="exact")
    plt.bar([i + w / 2 for i in x], q_meth, width=w, label="method")
    plt.xticks(x, labels)
    plt.ylabel("epoch latency (ms)")
    plt.title("Epoch latency quantiles (rank-max)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def fig3(rows, out_path, warmup_epochs: int):
    errs = []
    for r in rows:
        if not (r.get("mode") or "").startswith("compressed_"):
            continue
        ep = to_int(r.get("epoch"), 0)
        if ep < warmup_epochs:
            continue
        if r.get("is_shadow_epoch") != "1":
            continue
        if r.get("mode") != "compressed_accept":
            continue
        e = to_float(r.get("xva_err_bps"))
        if math.isfinite(e):
            errs.append(abs(e))
    errs.sort()
    if not errs:
        errs = [0.0]
    y = [(i + 1) / len(errs) for i in range(len(errs))]

    plt.figure(figsize=(6, 4))
    plt.plot(errs, y)
    plt.xlabel("abs(xva_err_bps) on shadow accepted epochs")
    plt.ylabel("CDF")
    plt.title("XVA error CDF (accepted, shadow epochs)")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def fig4(rows, out_path, warmup_epochs: int):
    # Prefer shock_sigma; else delta_m_sigma; else rho_hat_sketch.
    key = None
    for cand in ("shock_sigma", "delta_m_sigma"):
        if any((r.get(cand) not in (None, "", "NaN", "nan")) for r in rows):
            key = cand
            break
    if key is None:
        key = "rho_hat_sketch"
        title = "Fallback vs rho_hat_sketch (NO shock_sigma in logs)"
    else:
        title = f"Fallback rate vs {key}"

    bins = defaultdict(list)
    for r in rows:
        if not (r.get("mode") or "").startswith("compressed_"):
            continue
        ep = to_int(r.get("epoch"), 0)
        if ep < warmup_epochs:
            continue
        x = to_float(r.get(key))
        if not math.isfinite(x):
            continue
        fb = 1.0 if r.get("mode") == "compressed_fallback" else 0.0
        bins[x].append(fb)

    xs = sorted(bins.keys())
    if xs:
        ys = [100.0 * (sum(bins[x]) / len(bins[x])) for x in xs]
    else:
        xs = [0.0]
        ys = [0.0]

    plt.figure(figsize=(6, 4))
    plt.plot(xs, ys, marker="o")
    plt.xlabel(key)
    plt.ylabel("fallback rate (%)")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--warmup_epochs", type=int, default=0)
    args = ap.parse_args()

    rows = read_rows(os.path.join(args.run_dir, "metrics_merged.csv"))
    fig_dir = os.path.join(args.run_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    fig1(rows, os.path.join(fig_dir, "fig1_bytes_vs_rmax.pdf"))
    fig2(rows, os.path.join(fig_dir, "fig2_latency_quantiles.pdf"), args.warmup_epochs)
    fig3(rows, os.path.join(fig_dir, "fig3_xva_error_cdf.pdf"), args.warmup_epochs)
    fig4(rows, os.path.join(fig_dir, "fig4_fallback_vs_shock.pdf"), args.warmup_epochs)

    print(f"Figures written to {fig_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
