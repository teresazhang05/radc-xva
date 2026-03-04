#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 ]]; then
  echo "Usage: $0 <run_dir> <workload> <warmup_epochs> [--validate]" >&2
  exit 2
fi

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
RUN_DIR="$1"
WORKLOAD="$2"
WARMUP="$3"
DO_VALIDATE="${4:-}"

if [[ -n "${PYTHON_BIN:-}" ]]; then
  PY_BIN="$PYTHON_BIN"
elif [[ -x /opt/homebrew/Caskroom/miniforge/base/bin/python3 ]]; then
  PY_BIN="/opt/homebrew/Caskroom/miniforge/base/bin/python3"
else
  PY_BIN="$(command -v python3)"
fi

unset PYTHONHOME
unset PYTHONPATH
if [[ -z "${MPLCONFIGDIR:-}" ]]; then
  export MPLCONFIGDIR="$ROOT_DIR/.mplcache"
fi
mkdir -p "$MPLCONFIGDIR"

"$PY_BIN" "$ROOT_DIR/python/merge_logs.py" \
  --input_dir "$ROOT_DIR/$RUN_DIR" \
  --output "$ROOT_DIR/$RUN_DIR/metrics_merged.csv"

"$PY_BIN" "$ROOT_DIR/python/make_tables.py" \
  --run_dir "$ROOT_DIR/$RUN_DIR" \
  --workload "$WORKLOAD" \
  --warmup_epochs "$WARMUP"

"$PY_BIN" "$ROOT_DIR/python/make_figures.py" \
  --run_dir "$ROOT_DIR/$RUN_DIR" \
  --warmup_epochs "$WARMUP"

if [[ "$DO_VALIDATE" == "--validate" ]]; then
  "$PY_BIN" "$ROOT_DIR/python/validate_run.py" \
    --run_dir "$ROOT_DIR/$RUN_DIR" \
    --warmup_epochs "$WARMUP"
fi

echo "Postprocess complete for $RUN_DIR"
