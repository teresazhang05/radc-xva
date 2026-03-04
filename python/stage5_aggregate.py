#!/usr/bin/env python3
"""Aggregate Stage5 ablation runs.

This script expects each ablation run directory to exist at:
  results/<run_id>_combined/metrics_merged.csv

It computes p99 from epoch-wise max across ranks and fails if p99 is NaN.
"""

import argparse
import csv
import math
import os
from dataclasses import dataclass
from typing import Dict, List
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


@dataclass(frozen=True)
class AblationRun:
    run_id: str
    regime: str
    family: str
    warmup_epochs: int


RUNS: List[AblationRun] = [
    AblationRun("stage5_w2_normal_baseline", "normal", "baseline", 10),
    AblationRun("stage5_w2_normal_energy990", "normal", "energy", 10),
    AblationRun("stage5_w2_normal_energy999", "normal", "energy", 10),
    AblationRun("stage5_w2_normal_rmax8", "normal", "rmax", 10),
    AblationRun("stage5_w2_normal_rmax14", "normal", "rmax", 10),
    AblationRun("stage5_w2_normal_sketch32", "normal", "sketch", 10),
    AblationRun("stage5_w2_normal_sketch64", "normal", "sketch", 10),
    AblationRun("stage5_w2_shock_baseline", "shock", "shock", 10),
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


def summarize(repo_root: str, run: AblationRun) -> Dict[str, object]:
    run_dir = os.path.join(repo_root, "results", f"{run.run_id}_combined")
    path = os.path.join(run_dir, "metrics_merged.csv")
    rows_all = read_rows(path)
    rows = rows_all  # use all ranks; epoch_max handles aggregation

    exact = epoch_max(rows, lambda r: r.get("mode") == "exact", run.warmup_epochs)
    meth = epoch_max(rows, lambda r: (r.get("mode") or "").startswith("compressed_"), run.warmup_epochs)
    p99_exact = q_nearest_rank(exact, 0.99)
    p99_meth = q_nearest_rank(meth, 0.99)
    if not (math.isfinite(p99_exact) and math.isfinite(p99_meth) and p99_meth > 0):
        raise SystemExit(2)

    speedup = p99_exact / p99_meth

    method_rows = [r for r in rows_all if (r.get("mode") or "").startswith("compressed_")]
    fb = sum(1 for r in method_rows if r.get("mode") == "compressed_fallback")
    fb_rate = 100.0 * fb / len(method_rows) if method_rows else float("nan")

    # bytes reduction (effective)
    be = [to_float(r.get("bytes_exact_payload")) for r in rows_all]
    be = [x for x in be if math.isfinite(x)]
    be_med = sorted(be)[len(be)//2] if be else float("nan")
    eff = []
    for r in method_rows:
        used = to_float(r.get("bytes_total_payload"))
        if not math.isfinite(used):
            used = to_float(r.get("bytes_exact_payload")) if r.get("fallback_triggered") == "1" else to_float(r.get("bytes_comp_payload"))
        if math.isfinite(used):
            eff.append(used)
    eff.sort()
    eff_med = eff[len(eff)//2] if eff else float("nan")
    bytes_red = 100.0 * (1.0 - eff_med / be_med) if (math.isfinite(be_med) and be_med > 0 and math.isfinite(eff_med)) else float("nan")

    # shadow error
    shadow_err = []
    for r in method_rows:
        if r.get("is_shadow_epoch") != "1":
            continue
        if r.get("mode") != "compressed_accept":
            continue
        e = to_float(r.get("xva_err_bps"))
        if math.isfinite(e):
            shadow_err.append(abs(e))
    max_shadow = max(shadow_err) if shadow_err else float("nan")

    return {
        "run_id": run.run_id,
        "regime": run.regime,
        "family": run.family,
        "p99_exact_ms": p99_exact,
        "p99_method_ms": p99_meth,
        "p99_speedup_exact_over_method": speedup,
        "bytes_reduction_pct_effective": bytes_red,
        "fallback_rate_pct_compressed_only": fb_rate,
        "max_shadow_xva_err_bps": max_shadow,
    }


def write_csv(path: str, rows: List[Dict[str, object]]):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    keys = sorted({k for r in rows for k in r.keys()})
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def fig_speedup_vs_error(rows: List[Dict[str, object]], out_path: str):
    normals = [r for r in rows if r["regime"] == "normal"]
    xs = [max(float(r["max_shadow_xva_err_bps"]), 1e-12) for r in normals]
    ys = [float(r["p99_speedup_exact_over_method"]) for r in normals]
    labels = [str(r["run_id"]) for r in normals]
    plt.figure(figsize=(7.5, 4.5))
    plt.scatter(xs, ys, s=45)
    for x, y, label in zip(xs, ys, labels):
        plt.annotate(label, (x, y), fontsize=8, xytext=(4, 3), textcoords="offset points")
    plt.xscale("log")
    plt.xlabel("max shadow xva_err_bps (log)")
    plt.ylabel("p99 speedup (exact/method)")
    plt.title("Stage5: speedup vs error (normal)")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo_root", default=os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    ap.add_argument("--out_dir", default="results/stage5_final")
    args = ap.parse_args()

    repo_root = os.path.abspath(args.repo_root)
    out_dir = os.path.join(repo_root, args.out_dir) if not os.path.isabs(args.out_dir) else args.out_dir
    os.makedirs(out_dir, exist_ok=True)

    summaries = [summarize(repo_root, r) for r in RUNS]
    write_csv(os.path.join(out_dir, "ablation_summary.csv"), summaries)

    fig_dir = os.path.join(out_dir, "paper_figures")
    os.makedirs(fig_dir, exist_ok=True)
    fig_speedup_vs_error(summaries, os.path.join(fig_dir, "fig_stage5_speedup_vs_error.pdf"))

    print(f"Wrote {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
