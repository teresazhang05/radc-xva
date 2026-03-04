#!/usr/bin/env python3
"""Validate a run directory for paper/artifact readiness.

Exit codes:
- 0 pass
- 2 fail

Checks:
- required columns exist
- non-warmup, non-shadow t_epoch_total_ms is finite and > 0
- p99 exact and method are finite
- false_accept_count_shadow == 0
"""

import argparse
import csv
import math
import os
from typing import Dict, List


REQUIRED_COLS = [
    "rank", "epoch", "mode", "t_epoch_total_ms",
    "bytes_exact_payload", "bytes_total_payload",
    "fallback_triggered", "is_shadow_epoch",
]


def read_rows(path: str) -> List[Dict[str, str]]:
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


def q_nearest_rank(vals: List[float], q: float) -> float:
    xs = [v for v in vals if math.isfinite(v)]
    if not xs:
        return float("nan")
    xs.sort()
    n = len(xs)
    idx = int(math.ceil(q * n)) - 1
    idx = max(0, min(idx, n - 1))
    return xs[idx]


def epoch_max(rows: List[Dict[str, str]], mode_pred, warmup_epochs: int) -> List[float]:
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


def median(vals: List[float]) -> float:
    xs = [v for v in vals if math.isfinite(v)]
    if not xs:
        return float("nan")
    xs.sort()
    n = len(xs)
    if n % 2 == 1:
        return xs[n // 2]
    return 0.5 * (xs[n // 2 - 1] + xs[n // 2])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--warmup_epochs", type=int, default=0)
    ap.add_argument("--xva_eps_bps", type=float, default=0.01)
    ap.add_argument("--skip_false_accept_gate", action="store_true")
    ap.add_argument("--require_epsilon_bps", type=float, default=float("nan"))
    ap.add_argument("--max_p99_over_exact_ratio", type=float, default=float("nan"))
    ap.add_argument("--require_fallback_pct_min", type=float, default=float("nan"))
    ap.add_argument("--require_fallback_pct_max", type=float, default=float("nan"))
    ap.add_argument("--require_bytes_reduction_pct_min", type=float, default=float("nan"))
    ap.add_argument("--require_p99_speedup_accept_min", type=float, default=float("nan"))
    args = ap.parse_args()

    path = os.path.join(args.run_dir, "metrics_merged.csv")
    if not os.path.exists(path):
        print(f"FAIL: missing {path}")
        return 2

    rows = read_rows(path)
    if not rows:
        print("FAIL: metrics_merged.csv has no rows")
        return 2

    # required columns
    cols = set(rows[0].keys())
    missing = [c for c in REQUIRED_COLS if c not in cols]
    if missing:
        print(f"FAIL: missing required columns: {missing}")
        return 2

    warmup = args.warmup_epochs
    warmup_file = to_int(rows[0].get("warmup_epochs"), warmup)
    warmup = warmup_file if warmup_file >= 0 else warmup

    # timing nonzero check
    bad = 0
    for r in rows:
        ep = to_int(r.get("epoch"), 0)
        if ep < warmup:
            continue
        if r.get("is_shadow_epoch") == "1":
            continue
        t = to_float(r.get("t_epoch_total_ms"))
        if not (math.isfinite(t) and t > 0):
            bad += 1
    if bad > 0:
        print(f"FAIL: {bad} rows have non-finite or non-positive t_epoch_total_ms (post-warmup, non-shadow)")
        return 2

    # Critical non-NaN checks for payload fields.
    bad_payload = 0
    for r in rows:
        ep = to_int(r.get("epoch"), 0)
        if ep < warmup:
            continue
        be = to_float(r.get("bytes_exact_payload"))
        if not (math.isfinite(be) and be > 0):
            bad_payload += 1
            continue
        if (r.get("mode") or "").startswith("compressed_"):
            bt = to_float(r.get("bytes_total_payload"))
            if not (math.isfinite(bt) and bt > 0):
                bad_payload += 1
    if bad_payload > 0:
        print(f"FAIL: {bad_payload} rows have invalid bytes payload fields")
        return 2

    # p99 finite check
    exact = epoch_max(rows, lambda r: r.get("mode") == "exact", warmup)
    meth = epoch_max(rows, lambda r: (r.get("mode") or "").startswith("compressed_"), warmup)
    meth_accept = epoch_max(rows, lambda r: r.get("mode") == "compressed_accept", warmup)
    p99_exact = q_nearest_rank(exact, 0.99)
    p99_meth = q_nearest_rank(meth, 0.99)
    p99_meth_accept = q_nearest_rank(meth_accept, 0.99)
    if not (math.isfinite(p99_exact) and math.isfinite(p99_meth) and p99_meth > 0):
        print(f"FAIL: invalid p99s exact={p99_exact} method={p99_meth}")
        return 2

    # Optional epsilon gate for compressed rows.
    if math.isfinite(args.require_epsilon_bps):
        eps_rows = []
        for r in rows:
            if not (r.get("mode") or "").startswith("compressed_"):
                continue
            eps = to_float(r.get("xva_epsilon_bps"))
            if math.isfinite(eps):
                eps_rows.append(eps)
        if not eps_rows:
            print("FAIL: no compressed epsilon values available for epsilon gate")
            return 2
        for eps in eps_rows:
            if abs(eps - args.require_epsilon_bps) > 1e-12:
                print(f"FAIL: found xva_epsilon_bps={eps} (expected {args.require_epsilon_bps})")
                return 2

    # false accept check (shadow accepted epochs only)
    if not args.skip_false_accept_gate:
        false_accept = 0
        shadow_accept_errs: List[float] = []
        for r in rows:
            ep = to_int(r.get("epoch"), 0)
            if ep < warmup:
                continue
            if r.get("mode") != "compressed_accept":
                continue
            if r.get("is_shadow_epoch") != "1":
                continue
            e = to_float(r.get("xva_err_bps"))
            if math.isfinite(e):
                shadow_accept_errs.append(abs(e))
                if e > args.xva_eps_bps:
                    false_accept += 1
            else:
                print("FAIL: non-finite xva_err_bps for accepted shadow epoch")
                return 2
        if false_accept > 0:
            print(f"FAIL: false_accept_count_shadow={false_accept}")
            return 2
        if shadow_accept_errs:
            max_shadow_err = max(shadow_accept_errs)
            if max_shadow_err > args.xva_eps_bps:
                print(f"FAIL: max_accepted_shadow_error_bps={max_shadow_err} > epsilon={args.xva_eps_bps}")
                return 2

    # Optional fallback and speed ratio gates.
    method_rows = [r for r in rows if (r.get("mode") or "").startswith("compressed_")]
    fb_rows = [r for r in method_rows if r.get("mode") == "compressed_fallback"]
    fb_pct = (100.0 * len(fb_rows) / len(method_rows)) if method_rows else float("nan")

    if math.isfinite(args.require_fallback_pct_min):
        if not (math.isfinite(fb_pct) and fb_pct >= args.require_fallback_pct_min):
            print(f"FAIL: fallback_pct={fb_pct} < min={args.require_fallback_pct_min}")
            return 2
    if math.isfinite(args.require_fallback_pct_max):
        if not (math.isfinite(fb_pct) and fb_pct <= args.require_fallback_pct_max):
            print(f"FAIL: fallback_pct={fb_pct} > max={args.require_fallback_pct_max}")
            return 2

    if math.isfinite(args.max_p99_over_exact_ratio):
        ratio = p99_meth / p99_exact if (math.isfinite(p99_meth) and math.isfinite(p99_exact) and p99_exact > 0) else float("nan")
        if not (math.isfinite(ratio) and ratio <= args.max_p99_over_exact_ratio):
            print(f"FAIL: p99_method/p99_exact={ratio} > limit={args.max_p99_over_exact_ratio}")
            return 2

    if math.isfinite(args.require_p99_speedup_accept_min):
        speedup_accept = (
            p99_exact / p99_meth_accept
            if (math.isfinite(p99_exact) and math.isfinite(p99_meth_accept) and p99_meth_accept > 0)
            else float("nan")
        )
        if not (math.isfinite(speedup_accept) and speedup_accept >= args.require_p99_speedup_accept_min):
            print(
                f"FAIL: p99_speedup_accept_only={speedup_accept} "
                f"< min={args.require_p99_speedup_accept_min}"
            )
            return 2

    # Optional bytes reduction gate (effective, median based).
    if math.isfinite(args.require_bytes_reduction_pct_min):
        exact_bytes = median([to_float(r.get("bytes_exact_payload")) for r in rows if r.get("mode") == "exact"])
        if not math.isfinite(exact_bytes):
            exact_bytes = median([to_float(r.get("bytes_exact_payload")) for r in rows])
        effective = []
        for r in method_rows:
            v = to_float(r.get("bytes_total_payload"))
            if not math.isfinite(v):
                # Backward compatibility with legacy schema.
                be = to_float(r.get("bytes_exact_payload"))
                bc = to_float(r.get("bytes_comp_payload"))
                v = be if r.get("fallback_triggered") == "1" else bc
            if math.isfinite(v):
                effective.append(v)
        eff_med = median(effective)
        red = 100.0 * (1.0 - eff_med / exact_bytes) if (math.isfinite(eff_med) and math.isfinite(exact_bytes) and exact_bytes > 0) else float("nan")
        if not (math.isfinite(red) and red >= args.require_bytes_reduction_pct_min):
            print(f"FAIL: bytes_reduction_pct_effective={red} < min={args.require_bytes_reduction_pct_min}")
            return 2

    print("PASS")
    speedup = p99_exact / p99_meth
    speedup_accept = p99_exact / p99_meth_accept if (math.isfinite(p99_meth_accept) and p99_meth_accept > 0) else float("nan")
    print(
        f"p99_exact_ms={p99_exact:.6f} p99_method_ms={p99_meth:.6f} "
        f"p99_method_accept_only_ms={p99_meth_accept:.6f} "
        f"speedup={speedup:.3f}x speedup_accept_only={speedup_accept:.3f}x "
        f"fallback_pct={fb_pct:.3f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
