#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
NP="${1:-2}"

# Stage 0-3 full build/test/evaluation run.
"$ROOT_DIR/scripts/run_stage0123_full_eval.sh" "$NP"

# Stage 4 paper pipeline + artifact packaging.
"$ROOT_DIR/scripts/run_stage4_pipeline.sh"

echo "Stage 0-4 full evaluation suite complete."
