# Artifact / AE Instructions

This project is implemented in C++17 with MPI (PMPI interception) and a Python analysis pipeline.

## Required environment
- Linux x86_64 or macOS
- OpenMPI or MPICH
- CMake >= 3.16
- GCC >= 10 or Clang >= 12
- Python 3.10+

## Determinism
- Randomness is seeded from `run.seed`, `epoch`, and `rank`.
- Logged outputs are sufficient to regenerate all figures/tables.

## Full reproduction command
```bash
bash scripts/run_stage012345_full_eval.sh 2
```

## Stage outputs
- Stage 0: `results/w1_synth_small`
- Stage 2: `results/stage2_*_combined`
- Stage 3: `results/stage3_*_combined`
- Stage 4 summary/paper outputs: `results/stage4_final`
- Stage 5 ablation/stress outputs: `results/stage5_final`

## Final artifact bundle outputs
- `results/artifact_bundle_latest/`
- `results/artifact_bundle_latest.zip`

The bundle includes:
- Canonical metrics/figures/tables for Stage 0, Stage 2, and Stage 3 runs.
- Stage 4 paper-level summary tables/figures.
- Stage 5 ablation summaries, gate audits, and paper-level Stage 5 figures/tables.
- `MANIFEST.csv` with file size and sha256 for integrity checks.

## Notes
- Power/energy counters are best-effort. If unavailable, metrics log `NaN`.
