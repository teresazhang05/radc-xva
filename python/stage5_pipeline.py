#!/usr/bin/env python3
import argparse
import csv
import os
import shutil
import zipfile
from typing import Dict, List

from stage4_pipeline import (
    CANONICAL_RUNS,
    REQUIRED_RELATIVE_FILES,
    copy_file,
    ensure_exists,
    sha256_file,
    write_csv,
)


STAGE4_REQUIRED = [
    "summary_stage_metrics.csv",
    "paper_tables/table_stage2_vs_stage3_delta.csv",
    "paper_tables/table_stage3_final_story.csv",
    "paper_figures/fig_stage2_vs_stage3_bytes_reduction.pdf",
    "paper_figures/fig_stage2_vs_stage3_p99_speedup.pdf",
    "paper_figures/fig_stage2_vs_stage3_fallback_rate.pdf",
    "paper_figures/fig_stage2_vs_stage3_max_shadow_err_bps.pdf",
]

STAGE5_REQUIRED = [
    "ablation_summary.csv",
    "gates_stage5.csv",
    "stage5_report.md",
    "paper_tables/table_stage5_ablation_summary.csv",
    "paper_figures/fig_stage5_speedup_vs_error.pdf",
    "paper_figures/fig_stage5_bytes_vs_fallback.pdf",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repo_root",
        default=os.path.abspath(os.path.join(os.path.dirname(__file__), "..")),
        help="Repository root for radc-xva",
    )
    parser.add_argument(
        "--stage4_out",
        default="results/stage4_final",
        help="Stage 4 output directory (relative to repo_root unless absolute)",
    )
    parser.add_argument(
        "--stage5_out",
        default="results/stage5_final",
        help="Stage 5 output directory (relative to repo_root unless absolute)",
    )
    parser.add_argument(
        "--bundle_dir",
        default="results/artifact_bundle_latest",
        help="Artifact bundle directory (relative to repo_root unless absolute)",
    )
    parser.add_argument(
        "--zip_out",
        default="results/artifact_bundle_latest.zip",
        help="Zip output path (relative to repo_root unless absolute)",
    )
    return parser.parse_args()


def to_abs(repo_root: str, path: str) -> str:
    if os.path.isabs(path):
        return path
    return os.path.join(repo_root, path)


def copy_tree(src_root: str, dst_root: str) -> None:
    for root, _, files in os.walk(src_root):
        for fn in files:
            src = os.path.join(root, fn)
            rel = os.path.relpath(src, src_root)
            copy_file(src, os.path.join(dst_root, rel))


def validate_output_dir(base_dir: str, required_files: List[str]) -> None:
    ensure_exists(base_dir)
    for rel in required_files:
        ensure_exists(os.path.join(base_dir, rel))


def write_gate_audit_table(stage5_out: str) -> None:
    gates_path = os.path.join(stage5_out, "gates_stage5.csv")
    with open(gates_path, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    pass_count = sum(1 for r in rows if str(r.get("passed", "0")) == "1")
    total = len(rows)

    out_rows: List[Dict[str, object]] = []
    for r in rows:
        out_rows.append(
            {
                "run_id": r.get("run_id", ""),
                "regime": r.get("regime", ""),
                "passed": int(str(r.get("passed", "0")) == "1"),
                "detail": r.get("detail", ""),
            }
        )

    out_rows.append(
        {
            "run_id": "TOTAL",
            "regime": "all",
            "passed": f"{pass_count}/{total}",
            "detail": "Stage 5 gate pass count",
        }
    )

    out_dir = os.path.join(stage5_out, "paper_tables")
    os.makedirs(out_dir, exist_ok=True)
    write_csv(
        os.path.join(out_dir, "table_stage5_gate_audit.csv"),
        out_rows,
        ["run_id", "regime", "passed", "detail"],
    )


def make_bundle(
    repo_root: str,
    stage4_out: str,
    stage5_out: str,
    bundle_dir: str,
    zip_out: str,
) -> None:
    if os.path.exists(bundle_dir):
        shutil.rmtree(bundle_dir)
    os.makedirs(bundle_dir, exist_ok=True)

    canonical_root = os.path.join(bundle_dir, "canonical")
    for spec in CANONICAL_RUNS:
        src_base = os.path.join(repo_root, spec.run_dir)
        dst_base = os.path.join(canonical_root, spec.stage, spec.workload)
        for rel in REQUIRED_RELATIVE_FILES:
            copy_file(os.path.join(src_base, rel), os.path.join(dst_base, rel))
        if spec.stage == "stage0":
            for rel in [
                "metrics_rank0.csv",
                "metrics_rank1.csv",
                "events_rank0.jsonl",
                "events_rank1.jsonl",
            ]:
                src = os.path.join(src_base, rel)
                if os.path.exists(src):
                    copy_file(src, os.path.join(dst_base, rel))

    copy_tree(stage4_out, os.path.join(canonical_root, "stage4"))
    copy_tree(stage5_out, os.path.join(canonical_root, "stage5"))

    for rel in [
        "README.md",
        "artifact/AE_INSTRUCTIONS.md",
        "artifact/EXPECTED_OUTPUTS.md",
        "scripts/run_stage0123_full_eval.sh",
        "scripts/run_stage01234_full_eval.sh",
        "scripts/run_stage5_eval.sh",
        "scripts/run_stage5_pipeline.sh",
        "scripts/run_stage012345_full_eval.sh",
    ]:
        src = os.path.join(repo_root, rel)
        if os.path.exists(src):
            copy_file(src, os.path.join(bundle_dir, "metadata", rel))

    manifest_rows: List[Dict[str, object]] = []
    for root, _, files in os.walk(bundle_dir):
        for fn in files:
            p = os.path.join(root, fn)
            rel = os.path.relpath(p, bundle_dir)
            manifest_rows.append(
                {
                    "path": rel,
                    "size_bytes": os.path.getsize(p),
                    "sha256": sha256_file(p),
                }
            )
    manifest_rows.sort(key=lambda r: str(r["path"]))
    write_csv(
        os.path.join(bundle_dir, "MANIFEST.csv"),
        manifest_rows,
        ["path", "size_bytes", "sha256"],
    )

    with open(os.path.join(bundle_dir, "REPRODUCE.txt"), "w", encoding="utf-8") as f:
        f.write("Run full pipeline:\n")
        f.write("bash scripts/run_stage012345_full_eval.sh 2\n")

    os.makedirs(os.path.dirname(zip_out), exist_ok=True)
    if os.path.exists(zip_out):
        os.remove(zip_out)
    with zipfile.ZipFile(zip_out, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for root, _, files in os.walk(bundle_dir):
            for fn in files:
                p = os.path.join(root, fn)
                rel = os.path.relpath(p, os.path.dirname(bundle_dir))
                zf.write(p, rel)


def main() -> int:
    args = parse_args()
    repo_root = os.path.abspath(args.repo_root)
    stage4_out = to_abs(repo_root, args.stage4_out)
    stage5_out = to_abs(repo_root, args.stage5_out)
    bundle_dir = to_abs(repo_root, args.bundle_dir)
    zip_out = to_abs(repo_root, args.zip_out)

    validate_output_dir(stage4_out, STAGE4_REQUIRED)
    validate_output_dir(stage5_out, STAGE5_REQUIRED)
    write_gate_audit_table(stage5_out)
    make_bundle(repo_root, stage4_out, stage5_out, bundle_dir, zip_out)

    print(f"Stage 5 bundle dir: {bundle_dir}")
    print(f"Stage 5 bundle zip: {zip_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
