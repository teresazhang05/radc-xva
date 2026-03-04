#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
NP="${1:-2}"

"$ROOT_DIR/scripts/build.sh"

# Stage 1 gate: full unit/integration tests.
ctest --test-dir "$ROOT_DIR/build" --output-on-failure

# Stage 0 final run/eval (wrapper path + artifact generation).
"$ROOT_DIR/scripts/run_all.sh" "$ROOT_DIR/configs/w1_synth_small.yaml" "$NP"

# Stage 2 final run/eval (non-intercept compressed protocol in benchmarks).
"$ROOT_DIR/scripts/run_stage2_eval.sh" "$NP"

# Stage 3 final run/eval (compression path moved into MPI interceptor).
"$ROOT_DIR/scripts/run_stage3_eval.sh" "$NP"

echo "Stage 0-3 full evaluation suite complete."
