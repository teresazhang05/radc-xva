#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"

if [[ -n "${PYTHON_BIN:-}" ]]; then
  PY_BIN="$PYTHON_BIN"
else
  PY_BIN="$(command -v python3)"
fi

"$PY_BIN" "$ROOT_DIR/python/validate_configs.py" \
  --configs_dir "$ROOT_DIR/configs" \
  --fail_on_deprecated_config 1

# One-command reproducibility entrypoint for the full evaluation suite.
bash "$ROOT_DIR/scripts/run_top_tier_eval.sh"
