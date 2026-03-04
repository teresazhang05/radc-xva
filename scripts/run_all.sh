#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"

# New default: top-tier toy+medium+network evaluation suite.
if [[ $# -eq 0 ]]; then
  bash "$ROOT_DIR/scripts/run_all_experiments.sh"
  exit 0
fi

CONFIG="${1:-$ROOT_DIR/configs/w1_synth_small.yaml}"
NP="${2:-2}"

if [[ -n "${PYTHON_BIN:-}" ]]; then
  PY_BIN="$PYTHON_BIN"
elif [[ -x /opt/homebrew/Caskroom/miniforge/base/bin/python3 ]]; then
  PY_BIN="/opt/homebrew/Caskroom/miniforge/base/bin/python3"
else
  PY_BIN="$(command -v python3)"
fi

"$ROOT_DIR/scripts/run_local.sh" "$CONFIG" "$NP"

unset PYTHONHOME
unset PYTHONPATH

RUN_DIR=$(RADC_CONFIG="$CONFIG" "$PY_BIN" - <<'PY'
import os
import sys
cfg = os.environ.get("RADC_CONFIG", "")
out = "results/w1_synth_small"
if cfg and os.path.exists(cfg):
    with open(cfg, "r", encoding="utf-8") as f:
        for line in f:
            t = line.strip()
            if t.startswith("output_dir:"):
                out = t.split(":",1)[1].strip().strip('"').strip("'")
                break
print(out)
PY
)

"$PY_BIN" "$ROOT_DIR/python/merge_logs.py" --input_dir "$ROOT_DIR/$RUN_DIR" --output "$ROOT_DIR/$RUN_DIR/metrics_merged.csv"
"$PY_BIN" "$ROOT_DIR/python/make_figures.py" --run_dir "$ROOT_DIR/$RUN_DIR"
"$PY_BIN" "$ROOT_DIR/python/make_tables.py" --run_dir "$ROOT_DIR/$RUN_DIR"

echo "All outputs generated under $ROOT_DIR/$RUN_DIR"
