#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
BUILD_DIR="$ROOT_DIR/build"
NP="${1:-2}"

if [[ -n "${PYTHON_BIN:-}" ]]; then
  PY_BIN="$PYTHON_BIN"
elif [[ -x /opt/homebrew/Caskroom/miniforge/base/bin/python3 ]]; then
  PY_BIN="/opt/homebrew/Caskroom/miniforge/base/bin/python3"
else
  PY_BIN="$(command -v python3)"
fi

unset PYTHONHOME
unset PYTHONPATH

"$ROOT_DIR/scripts/build.sh"

LIBRADC="$BUILD_DIR/libradc.so"
if [[ ! -f "$LIBRADC" && -f "$BUILD_DIR/libradc.dylib" ]]; then
  LIBRADC="$BUILD_DIR/libradc.dylib"
fi
if [[ ! -f "$LIBRADC" ]]; then
  echo "Could not find preloaded library in $BUILD_DIR" >&2
  exit 1
fi

run_with_wrapper() {
  local cfg="$1"
  local force_exact="$2"
  local suffix="$3"
  local bench_bin="$4"

  if [[ "$(uname -s)" == "Darwin" ]]; then
    mpirun -np "$NP" env \
      DYLD_FORCE_FLAT_NAMESPACE=1 \
      DYLD_INSERT_LIBRARIES="$LIBRADC" \
      RADC_CONFIG="$cfg" \
      RADC_FORCE_EXACT="$force_exact" \
      RADC_RUN_SUFFIX="$suffix" \
      "$BUILD_DIR/$bench_bin" --config "$cfg"
  else
    mpirun -np "$NP" env \
      LD_PRELOAD="$LIBRADC" \
      RADC_CONFIG="$cfg" \
      RADC_FORCE_EXACT="$force_exact" \
      RADC_RUN_SUFFIX="$suffix" \
      "$BUILD_DIR/$bench_bin" --config "$cfg"
  fi
}

run_one() {
  local cfg="$1"
  local bench_bin="$2"

  local run_id output_dir
  run_id="$($PY_BIN - <<PY
import re
p = r"$cfg"
run_id = ""
output_dir = ""
in_run = False
for line in open(p, 'r', encoding='utf-8'):
    t = line.rstrip('\n')
    if t.strip() == 'run:':
        in_run = True
        continue
    if in_run and t and not t.startswith('  '):
        in_run = False
    if in_run:
        s = t.strip()
        if s.startswith('run_id:'):
            run_id = s.split(':',1)[1].strip().strip('"').strip("'")
        if s.startswith('output_dir:'):
            output_dir = s.split(':',1)[1].strip().strip('"').strip("'")
print(run_id)
print(output_dir)
PY
)"

  local base_run_id base_out
  base_run_id="$(echo "$run_id" | sed -n '1p')"
  base_out="$(echo "$run_id" | sed -n '2p')"

  if [[ -z "$base_run_id" || -z "$base_out" ]]; then
    echo "Failed to parse run_id/output_dir from $cfg" >&2
    exit 1
  fi

  local exact_suffix="exact"
  local comp_suffix="compressed"

  echo "[stage3] Running exact baseline via interceptor: $base_run_id"
  run_with_wrapper "$cfg" "1" "$exact_suffix" "$bench_bin"

  echo "[stage3] Running compressed protocol via interceptor: $base_run_id"
  run_with_wrapper "$cfg" "0" "$comp_suffix" "$bench_bin"

  local exact_dir="$ROOT_DIR/${base_out}_${exact_suffix}"
  local comp_dir="$ROOT_DIR/${base_out}_${comp_suffix}"
  local combined_dir="$ROOT_DIR/results/${base_run_id}_combined"

  "$PY_BIN" "$ROOT_DIR/python/merge_logs.py" --input_dir "$exact_dir" --output "$exact_dir/metrics_merged.csv"
  "$PY_BIN" "$ROOT_DIR/python/merge_logs.py" --input_dir "$comp_dir" --output "$comp_dir/metrics_merged.csv"

  "$PY_BIN" "$ROOT_DIR/python/combine_metrics.py" \
    --exact "$exact_dir/metrics_merged.csv" \
    --compressed "$comp_dir/metrics_merged.csv" \
    --output "$combined_dir/metrics_merged.csv"

  "$PY_BIN" "$ROOT_DIR/python/make_figures.py" --run_dir "$combined_dir"
  "$PY_BIN" "$ROOT_DIR/python/make_tables.py" --run_dir "$combined_dir"

  echo "[stage3] Completed workload: $base_run_id"
}

run_one "$ROOT_DIR/configs/stage3_w1_eval.yaml" "bench_stage3_intercept_driver"
run_one "$ROOT_DIR/configs/stage3_w2_normal_eval.yaml" "bench_stage3_intercept_driver"
run_one "$ROOT_DIR/configs/stage3_w2_shock_eval.yaml" "bench_stage3_intercept_driver"
run_one "$ROOT_DIR/configs/stage3_w3_eval.yaml" "bench_stage3_intercept_driver"

echo "Stage 3 evaluation suite complete."
