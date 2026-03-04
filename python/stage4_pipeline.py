#!/usr/bin/env python3
"""Build a Stage4 summary and artifact bundle.

This script expects each canonical run directory to contain:
- metrics_merged.csv
- tables/table1_main.csv (new schema from this packet)
- tables/table2_lower_bound.csv
- figures/fig1_bytes_vs_rmax.pdf
- figures/fig2_latency_quantiles.pdf
- figures/fig3_xva_error_cdf.pdf
- figures/fig4_fallback_vs_shock.pdf
"""

import argparse
import csv
import hashlib
import os
import shutil
import zipfile
from dataclasses import dataclass
from typing import List, Dict


@dataclass(frozen=True)
class RunSpec:
    name: str
    run_dir: str
    workload: str
    warmup_epochs: int


CANONICAL = [
    RunSpec("medium_w2_exact_f32", "results/medium_w2_exact_f32_combined", "w2_normal", 30),
    RunSpec("medium_w2_radc", "results/medium_w2_radc_combined", "w2_normal", 30),
    RunSpec("network_w2_exact_f32", "results/network_w2_exact_f32_combined", "w2_normal", 30),
    RunSpec("network_w2_radc", "results/network_w2_radc_combined", "w2_normal", 30),
]


REQUIRED = [
    "metrics_merged.csv",
    "tables/table1_main.csv",
    "tables/table2_lower_bound.csv",
    "figures/fig1_bytes_vs_rmax.pdf",
    "figures/fig2_latency_quantiles.pdf",
    "figures/fig3_xva_error_cdf.pdf",
    "figures/fig4_fallback_vs_shock.pdf",
]


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_table1(path: str) -> Dict[str, str]:
    with open(path, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return {}
    return rows[0]


def ensure_exists(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(path)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo_root", default=os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    ap.add_argument("--out_dir", default="results/stage4_final")
    ap.add_argument("--bundle_dir", default="results/artifact_bundle_latest")
    ap.add_argument("--zip_out", default="results/artifact_bundle_latest.zip")
    args = ap.parse_args()

    repo = os.path.abspath(args.repo_root)
    out_dir = os.path.join(repo, args.out_dir) if not os.path.isabs(args.out_dir) else args.out_dir
    bundle_dir = os.path.join(repo, args.bundle_dir) if not os.path.isabs(args.bundle_dir) else args.bundle_dir
    zip_out = os.path.join(repo, args.zip_out) if not os.path.isabs(args.zip_out) else args.zip_out

    os.makedirs(out_dir, exist_ok=True)

    # Summary
    summary = []
    for r in CANONICAL:
        run_abs = os.path.join(repo, r.run_dir)
        for rel in REQUIRED:
            ensure_exists(os.path.join(run_abs, rel))
        t1 = read_table1(os.path.join(run_abs, "tables", "table1_main.csv"))
        row = {"name": r.name, "run_dir": r.run_dir}
        row.update(t1)
        summary.append(row)

    sum_path = os.path.join(out_dir, "stage4_summary.csv")
    with open(sum_path, "w", newline="", encoding="utf-8") as f:
        keys = sorted({k for row in summary for k in row.keys()})
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(summary)

    # Bundle
    if os.path.exists(bundle_dir):
        shutil.rmtree(bundle_dir)
    os.makedirs(bundle_dir, exist_ok=True)
    for r in CANONICAL:
        src = os.path.join(repo, r.run_dir)
        dst = os.path.join(bundle_dir, r.name)
        shutil.copytree(src, dst)

    # Zip
    os.makedirs(os.path.dirname(zip_out), exist_ok=True)
    if os.path.exists(zip_out):
        os.remove(zip_out)
    with zipfile.ZipFile(zip_out, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for root, _, files in os.walk(bundle_dir):
            for fn in files:
                p = os.path.join(root, fn)
                arc = os.path.relpath(p, os.path.dirname(bundle_dir))
                z.write(p, arc)

    # Hash
    with open(os.path.join(out_dir, "artifact_zip_sha256.txt"), "w", encoding="utf-8") as f:
        f.write(sha256_file(zip_out) + "\n")

    print(f"Wrote {sum_path}")
    print(f"Wrote {zip_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
