#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
NP="${1:-2}"

# Stage 0-4 full build/test/evaluation run.
"$ROOT_DIR/scripts/run_stage01234_full_eval.sh" "$NP"

# Stage 5 ablation + stress suite.
"$ROOT_DIR/scripts/run_stage5_eval.sh" "$NP"

# Stage 5 final artifact package (includes Stage 4 and Stage 5 outputs).
"$ROOT_DIR/scripts/run_stage5_pipeline.sh"

echo "Stage 0-5 full evaluation suite complete."
