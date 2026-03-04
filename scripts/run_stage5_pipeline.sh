#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"

if [[ -n "${PYTHON_BIN:-}" ]]; then
  PY_BIN="$PYTHON_BIN"
elif [[ -x /opt/homebrew/Caskroom/miniforge/base/bin/python3 ]]; then
  PY_BIN="/opt/homebrew/Caskroom/miniforge/base/bin/python3"
else
  PY_BIN="$(command -v python3)"
fi

unset PYTHONHOME
unset PYTHONPATH

export MPLCONFIGDIR="${MPLCONFIGDIR:-$ROOT_DIR/.mplcache}"
mkdir -p "$MPLCONFIGDIR"

"$PY_BIN" "$ROOT_DIR/python/stage5_pipeline.py" \
  --repo_root "$ROOT_DIR" \
  --stage4_out "results/stage4_final" \
  --stage5_out "results/stage5_final" \
  --bundle_dir "results/artifact_bundle_latest" \
  --zip_out "results/artifact_bundle_latest.zip"

echo "Stage 5 pipeline complete."
