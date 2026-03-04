#!/usr/bin/env python3
"""Generate paper tables from metrics_merged.csv with correct definitions.

Outputs under <run_dir>/tables/:
- table1_main.csv
- table2_lower_bound.csv

Definitions:
- p50/p95/p99 exact and method are computed from epoch-wise max across ranks,
  excluding warmup epochs and shadow epochs.
- p99_speedup_exact_over_method = p99_exact_ms / p99_method_ms
- p99_speedup_exact_over_accept_only = p99_exact_ms / p99_method_accept_only_ms
- bytes_effective_payload = bytes_total_payload (counts comp attempt + fallback exact bytes)
- bytes_reduction_pct_effective = 100 * (1 - median(bytes_effective_payload)/median(bytes_exact_payload))
- false_accept_count_shadow = count(shadow epochs with mode=='compressed_accept' and
  xva_err_bps > xva_epsilon_bps), excluding warmup epochs.
"""

import argparse
import csv
import math
import os
from typing import Dict, List, Tuple
from collections import defaultdict


def read_rows(path: str) -> List[Dict[str, str]]:
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


def median(vals: List[float]) -> float:
    xs = [v for v in vals if math.isfinite(v)]
    if not xs:
        return float("nan")
    xs.sort()
    n = len(xs)
    if n % 2 == 1:
        return xs[n // 2]
    return 0.5 * (xs[n // 2 - 1] + xs[n // 2])


def q_nearest_rank(vals: List[float], q: float) -> float:
    xs = [v for v in vals if math.isfinite(v)]
    if not xs:
        return float("nan")
    xs.sort()
    n = len(xs)
    idx = int(math.ceil(q * n)) - 1
    idx = max(0, min(idx, n - 1))
    return xs[idx]


def epoch_max_latency(rows: List[Dict[str, str]], mode_pred, warmup_epochs: int) -> List[float]:
    by_epoch: Dict[int, float] = {}
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


def infer_bytes_per_num(rows: List[Dict[str, str]]) -> int:
    # Prefer factor_dtype column if present; else assume float32 factors.
    for r in rows:
        fd = (r.get("factor_dtype") or "").lower()
        if fd in ("float32", "f32"):
            return 4
        if fd in ("float64", "f64"):
            return 8
    return 4


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--workload", required=True, choices=["w1", "w2_normal", "w2_shock", "w3"])
    ap.add_argument("--warmup_epochs", type=int, default=0)
    args = ap.parse_args()

    run_dir = args.run_dir
    rows = read_rows(os.path.join(run_dir, "metrics_merged.csv"))
    if not rows:
        raise SystemExit(f"ERROR: no rows found in {run_dir}/metrics_merged.csv")

    warmup = args.warmup_epochs
    # If warmup_epochs exists in the file, prefer it (must be consistent).
    warmup_file = to_int(rows[0].get("warmup_epochs"), warmup)
    warmup = warmup_file if warmup_file >= 0 else warmup

    # Identify metadata
    first = rows[0]
    P = to_int(first.get("world_size"), 1)
    N = to_int(first.get("N"), 0)
    S = to_int(first.get("S"), 0)

    # Latency: exact vs method (mixed and accept-only)
    exact_vals = epoch_max_latency(rows, lambda r: r.get("mode") == "exact", warmup)
    meth_vals = epoch_max_latency(rows, lambda r: (r.get("mode") or "").startswith("compressed_"), warmup)
    meth_accept_vals = epoch_max_latency(rows, lambda r: r.get("mode") == "compressed_accept", warmup)

    p50_exact = q_nearest_rank(exact_vals, 0.50)
    p95_exact = q_nearest_rank(exact_vals, 0.95)
    p99_exact = q_nearest_rank(exact_vals, 0.99)

    p50_method = q_nearest_rank(meth_vals, 0.50)
    p95_method = q_nearest_rank(meth_vals, 0.95)
    p99_method = q_nearest_rank(meth_vals, 0.99)

    p50_method_accept = q_nearest_rank(meth_accept_vals, 0.50)
    p95_method_accept = q_nearest_rank(meth_accept_vals, 0.95)
    p99_method_accept = q_nearest_rank(meth_accept_vals, 0.99)

    p99_speedup = (p99_exact / p99_method) if (math.isfinite(p99_exact) and math.isfinite(p99_method) and p99_method > 0) else float("nan")
    p99_speedup_accept = (
        p99_exact / p99_method_accept
        if (math.isfinite(p99_exact) and math.isfinite(p99_method_accept) and p99_method_accept > 0)
        else float("nan")
    )

    # Bytes reduction (effective, includes fallback epochs and attempted compressed payload on fallback)
    bytes_exact = [to_float(r.get("bytes_exact_payload")) for r in rows if r.get("mode") == "exact"]
    bytes_exact_med = median(bytes_exact)
    # If there is no exact mode in this run directory, still allow bytes_exact_payload from any row.
    if not math.isfinite(bytes_exact_med):
        bytes_exact_med = median([to_float(r.get("bytes_exact_payload")) for r in rows])

    method_rows = [r for r in rows if (r.get("mode") or "").startswith("compressed_")]
    bytes_effective = []
    for r in method_rows:
        used = to_float(r.get("bytes_total_payload"))
        if not math.isfinite(used):
            # Backward compatibility with legacy schema.
            be = to_float(r.get("bytes_exact_payload"))
            bc = to_float(r.get("bytes_comp_payload"))
            used = be if r.get("fallback_triggered") == "1" else bc
        if math.isfinite(used):
            bytes_effective.append(used)
    bytes_effective_med = median(bytes_effective)
    bytes_reduction = float("nan")
    if math.isfinite(bytes_exact_med) and bytes_exact_med > 0 and math.isfinite(bytes_effective_med):
        bytes_reduction = 100.0 * (1.0 - bytes_effective_med / bytes_exact_med)

    # Fallback rate (compressed rows only)
    fb = sum(1 for r in method_rows if r.get("mode") == "compressed_fallback")
    denom = len(method_rows)
    fb_rate = (100.0 * fb / denom) if denom > 0 else float("nan")

    # Accuracy on shadow epochs
    eps_bps = to_float(first.get("xva_epsilon_bps"), 0.01)
    shadow_accept_errs = []
    false_accept = 0
    for r in method_rows:
        ep = to_int(r.get("epoch"), 0)
        if ep < warmup:
            continue
        if r.get("is_shadow_epoch") != "1":
            continue
        if r.get("mode") != "compressed_accept":
            continue
        e = to_float(r.get("xva_err_bps"))
        if math.isfinite(e):
            shadow_accept_errs.append(abs(e))
            if e > eps_bps:
                false_accept += 1
    max_shadow_err = max(shadow_accept_errs) if shadow_accept_errs else float("nan")

    # Write tables
    table_dir = os.path.join(run_dir, "tables")
    os.makedirs(table_dir, exist_ok=True)

    with open(os.path.join(table_dir, "table1_main.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "workload", "P", "N", "S", "warmup_epochs",
            "bytes_reduction_pct_effective",
            "fallback_rate_pct_compressed_only",
            "p50_exact_ms", "p95_exact_ms",
            "p50_method_ms", "p95_method_ms",
            "p50_method_accept_only_ms", "p95_method_accept_only_ms",
            "p99_exact_ms", "p99_method_ms",
            "p99_method_accept_only_ms",
            "p99_speedup_exact_over_method",
            "p99_speedup_exact_over_accept_only",
            "max_shadow_xva_err_bps",
            "false_accept_count_shadow",
        ])
        w.writerow([
            args.workload, P, N, S, warmup,
            f"{bytes_reduction:.6f}" if math.isfinite(bytes_reduction) else "NaN",
            f"{fb_rate:.6f}" if math.isfinite(fb_rate) else "NaN",
            f"{p50_exact:.6f}" if math.isfinite(p50_exact) else "NaN",
            f"{p95_exact:.6f}" if math.isfinite(p95_exact) else "NaN",
            f"{p50_method:.6f}" if math.isfinite(p50_method) else "NaN",
            f"{p95_method:.6f}" if math.isfinite(p95_method) else "NaN",
            f"{p50_method_accept:.6f}" if math.isfinite(p50_method_accept) else "NaN",
            f"{p95_method_accept:.6f}" if math.isfinite(p95_method_accept) else "NaN",
            f"{p99_exact:.6f}" if math.isfinite(p99_exact) else "NaN",
            f"{p99_method:.6f}" if math.isfinite(p99_method) else "NaN",
            f"{p99_method_accept:.6f}" if math.isfinite(p99_method_accept) else "NaN",
            f"{p99_speedup:.6f}" if math.isfinite(p99_speedup) else "NaN",
            f"{p99_speedup_accept:.6f}" if math.isfinite(p99_speedup_accept) else "NaN",
            f"{max_shadow_err:.6f}" if math.isfinite(max_shadow_err) else "NaN",
            str(false_accept),
        ])

    # Table2: lower bound ratio
    bytes_per_num = infer_bytes_per_num(rows)
    accept_rows = [r for r in method_rows if r.get("mode") == "compressed_accept"]
    r_med = median([to_float(r.get("r_used")) for r in accept_rows])
    comp_med = median([to_float(r.get("bytes_comp_attempt_payload")) for r in accept_rows])
    if not math.isfinite(comp_med):
        comp_med = median([to_float(r.get("bytes_comp_payload")) for r in accept_rows])

    lb = float("nan")
    ratio = float("nan")
    if r_med > 0 and N > 0 and S > 0 and math.isfinite(comp_med):
        lb = (N + S) * r_med * bytes_per_num
        ratio = (comp_med / lb) if lb > 0 else float("nan")

    with open(os.path.join(table_dir, "table2_lower_bound.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["N", "S", "bytes_per_num", "r_used_median", "bytes_comp_attempt_payload_median", "lower_bound", "ratio"])
        w.writerow([
            N, S, bytes_per_num,
            f"{r_med:.6f}" if math.isfinite(r_med) else "NaN",
            f"{comp_med:.6f}" if math.isfinite(comp_med) else "NaN",
            f"{lb:.6f}" if math.isfinite(lb) else "NaN",
            f"{ratio:.6f}" if math.isfinite(ratio) else "NaN",
        ])

    print(f"Tables written to {table_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
