#!/usr/bin/env python3
import argparse
import csv
import math
import os
from dataclasses import dataclass
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


@dataclass(frozen=True)
class RunSpec:
  run_id: str
  scale: str
  regime: str
  baseline: str


RUNS: List[RunSpec] = [
    RunSpec("toy_w2_radc", "toy", "normal", "safety_fallback"),
    RunSpec("medium_w2_radc", "medium", "normal", "safety_fallback"),
    RunSpec("medium_w2_nosafety", "medium", "normal", "no_safety"),
    RunSpec("medium_w2_fixedrank", "medium", "normal", "fixed_rank"),
    RunSpec("network_w2_radc", "network", "normal", "safety_fallback"),
    RunSpec("network_w2_real_radc", "network_real", "normal", "safety_fallback"),
    RunSpec("network_w2_nosafety", "network", "normal", "no_safety"),
    RunSpec("network_w2_fixedrank", "network", "normal", "fixed_rank"),
    RunSpec("network_w2_shock_sigma1_radc", "network", "shock", "safety_fallback"),
    RunSpec("network_w2_shock_sigma2_radc", "network", "shock", "safety_fallback"),
    RunSpec("network_w2_shock_sigma4_radc", "network", "shock", "safety_fallback"),
    RunSpec("network_w2_shock_sigma6_radc", "network", "shock", "safety_fallback"),
    RunSpec("network_w2_shock_sigma8_radc", "network", "shock", "safety_fallback"),
    RunSpec("network_w2_shock_sigma10_radc", "network", "shock", "safety_fallback"),
]


def to_float(v: str) -> float:
  try:
    if v in ("", "NaN", "nan", None):
      return float("nan")
    return float(v)
  except Exception:
    return float("nan")


def read_first_row(path: str) -> Dict[str, str]:
  if not os.path.exists(path):
    return {}
  with open(path, "r", encoding="utf-8") as f:
    rows = list(csv.DictReader(f))
  if not rows:
    return {}
  return rows[0]


def write_csv(path: str, rows: List[Dict[str, object]], fieldnames: List[str]) -> None:
  os.makedirs(os.path.dirname(path), exist_ok=True)
  with open(path, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=fieldnames)
    w.writeheader()
    for row in rows:
      w.writerow(row)


def parse_sigma(run_id: str) -> float:
  for tok in run_id.split("_"):
    if tok.startswith("sigma"):
      return to_float(tok.replace("sigma", ""))
  return float("nan")


def make_storyboard(rows: List[Dict[str, object]], out_path: str) -> None:
  normals = [r for r in rows if r["regime"] == "normal" and r["scale"] in ("medium", "network")]
  labels = [f'{r["scale"]}:{r["baseline"]}' for r in normals]
  bytes_vals = [to_float(str(r["bytes_reduction_pct_effective"])) for r in normals]
  speed_vals = [to_float(str(r["p99_speedup_exact_over_method"])) for r in normals]
  speed_accept_vals = [to_float(str(r["p99_speedup_exact_over_accept_only"])) for r in normals]
  fb_vals = [to_float(str(r["fallback_rate_pct_compressed_only"])) for r in normals]
  err_vals = [to_float(str(r["max_shadow_xva_err_bps"])) for r in normals]

  plt.figure(figsize=(11, 8))
  ax1 = plt.subplot(2, 2, 1)
  ax1.bar(range(len(labels)), bytes_vals)
  ax1.set_xticks(range(len(labels)))
  ax1.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
  ax1.set_ylabel("bytes reduction (%)")
  ax1.set_title("Communication Reduction")

  ax2 = plt.subplot(2, 2, 2)
  ax2.bar(range(len(labels)), speed_vals, label="mixed")
  if any(math.isfinite(v) for v in speed_accept_vals):
    ax2.plot(range(len(labels)), speed_accept_vals, marker="o", color="black", label="accept-only")
  ax2.set_xticks(range(len(labels)))
  ax2.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
  ax2.set_ylabel("p99 speedup (exact/method)")
  ax2.set_title("Tail-Latency Speedup")
  ax2.legend(fontsize=7)

  ax3 = plt.subplot(2, 2, 3)
  ax3.bar(range(len(labels)), fb_vals)
  ax3.set_xticks(range(len(labels)))
  ax3.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
  ax3.set_ylabel("fallback rate (%)")
  ax3.set_title("Fallback Rate (compressed rows)")

  ax4 = plt.subplot(2, 2, 4)
  pts = [(x, y, lab) for x, y, lab in zip(speed_vals, err_vals, labels) if math.isfinite(x) and math.isfinite(y) and y > 0]
  if pts:
    ax4.scatter([p[0] for p in pts], [p[1] for p in pts])
    for x, y, lab in pts:
      ax4.annotate(lab, (x, y), fontsize=7, xytext=(4, 3), textcoords="offset points")
    ax4.set_yscale("log")
  ax4.set_xlabel("p99 speedup (exact/method)")
  ax4.set_ylabel("max shadow error (bps, log)")
  ax4.set_title("Speedup vs Safety Error")

  plt.tight_layout()
  plt.savefig(out_path)
  plt.close()


def make_shock_plot(rows: List[Dict[str, object]], out_path: str) -> None:
  shocks = [r for r in rows if r["regime"] == "shock"]
  pts = []
  for r in shocks:
    sigma = parse_sigma(str(r["run_id"]))
    fb = to_float(str(r["fallback_rate_pct_compressed_only"]))
    p99_mixed = to_float(str(r["p99_method_ms"]))
    p99_accept = to_float(str(r["p99_method_accept_only_ms"]))
    if math.isfinite(sigma):
      pts.append((sigma, fb, p99_mixed, p99_accept))
  pts.sort(key=lambda x: x[0])
  xs = [p[0] for p in pts]
  ys_fb = [p[1] for p in pts]
  ys_mixed = [p[2] for p in pts]
  ys_accept = [p[3] for p in pts]
  plt.figure(figsize=(10, 4.5))
  ax1 = plt.subplot(1, 3, 1)
  ax1.plot(xs, ys_fb, marker="o")
  ax1.set_xlabel("shock_sigma")
  ax1.set_ylabel("fallback rate (%)")
  ax1.set_title("Fallback vs sigma")

  ax2 = plt.subplot(1, 3, 2)
  ax2.plot(xs, ys_mixed, marker="o")
  ax2.set_xlabel("shock_sigma")
  ax2.set_ylabel("p99 method ms (mixed)")
  ax2.set_title("Mixed p99 vs sigma")

  ax3 = plt.subplot(1, 3, 3)
  ax3.plot(xs, ys_accept, marker="o")
  ax3.set_xlabel("shock_sigma")
  ax3.set_ylabel("p99 method ms (accept-only)")
  ax3.set_title("Accepted-only p99 vs sigma")

  plt.tight_layout()
  plt.savefig(out_path)
  plt.close()


def main() -> int:
  ap = argparse.ArgumentParser()
  ap.add_argument("--repo_root", default=os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
  ap.add_argument("--out_dir", default="results/top_tier_final")
  args = ap.parse_args()

  repo_root = os.path.abspath(args.repo_root)
  out_dir = os.path.join(repo_root, args.out_dir) if not os.path.isabs(args.out_dir) else args.out_dir
  fig_dir = os.path.join(out_dir, "figures")
  os.makedirs(fig_dir, exist_ok=True)

  rows: List[Dict[str, object]] = []
  for spec in RUNS:
    run_dir = os.path.join(repo_root, "results", f"{spec.run_id}_combined")
    t1 = read_first_row(os.path.join(run_dir, "tables", "table1_main.csv"))
    t2 = read_first_row(os.path.join(run_dir, "tables", "table2_lower_bound.csv"))
    if not t1:
      continue
    rows.append(
        {
            "run_id": spec.run_id,
            "scale": spec.scale,
            "regime": spec.regime,
            "baseline": spec.baseline,
            "bytes_reduction_pct_effective": t1.get("bytes_reduction_pct_effective", "NaN"),
            "fallback_rate_pct_compressed_only": t1.get("fallback_rate_pct_compressed_only", "NaN"),
            "p99_exact_ms": t1.get("p99_exact_ms", "NaN"),
            "p99_method_ms": t1.get("p99_method_ms", "NaN"),
            "p99_method_accept_only_ms": t1.get("p99_method_accept_only_ms", "NaN"),
            "p99_speedup_exact_over_method": t1.get("p99_speedup_exact_over_method", "NaN"),
            "p99_speedup_exact_over_accept_only": t1.get("p99_speedup_exact_over_accept_only", "NaN"),
            "max_shadow_xva_err_bps": t1.get("max_shadow_xva_err_bps", "NaN"),
            "false_accept_count_shadow": t1.get("false_accept_count_shadow", ""),
            "near_opt_ratio": t2.get("ratio", "NaN") if t2 else "NaN",
        }
    )

  write_csv(
      os.path.join(out_dir, "combined_table_across_runs.csv"),
      rows,
      [
          "run_id",
          "scale",
          "regime",
          "baseline",
          "bytes_reduction_pct_effective",
          "fallback_rate_pct_compressed_only",
          "p99_exact_ms",
          "p99_method_ms",
          "p99_method_accept_only_ms",
          "p99_speedup_exact_over_method",
          "p99_speedup_exact_over_accept_only",
          "max_shadow_xva_err_bps",
          "false_accept_count_shadow",
          "near_opt_ratio",
      ],
  )

  make_storyboard(rows, os.path.join(fig_dir, "fig_main_storyboard.pdf"))
  make_shock_plot(rows, os.path.join(fig_dir, "fig_shock_sweep_fallback_vs_sigma.pdf"))
  print(f"Wrote top-tier summary to {out_dir}")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
