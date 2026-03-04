#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
BUILD_DIR="$ROOT_DIR/build"

NP_TOY="${NP_TOY:-2}"
NP_MEDIUM="${NP_MEDIUM:-8}"
NP_NETWORK="${NP_NETWORK:-16}"
REAL_HOSTFILE="${REAL_HOSTFILE:-}"
REAL_NP="${REAL_NP:-16}"
ALLOW_EMULATION_ONLY="${ALLOW_EMULATION_ONLY:-1}"

if [[ -n "${PYTHON_BIN:-}" ]]; then
  PY_BIN="$PYTHON_BIN"
elif [[ -x /opt/homebrew/Caskroom/miniforge/base/bin/python3 ]]; then
  PY_BIN="/opt/homebrew/Caskroom/miniforge/base/bin/python3"
else
  PY_BIN="$(command -v python3)"
fi

unset PYTHONHOME
unset PYTHONPATH

export RADC_FAIL_ON_DEPRECATED_CONFIG=1

"$PY_BIN" "$ROOT_DIR/python/validate_configs.py" \
  --configs_dir "$ROOT_DIR/configs" \
  --fail_on_deprecated_config 1

"$ROOT_DIR/scripts/build.sh"

LIBRADC="$BUILD_DIR/libradc.so"
if [[ ! -f "$LIBRADC" && -f "$BUILD_DIR/libradc.dylib" ]]; then
  LIBRADC="$BUILD_DIR/libradc.dylib"
fi
if [[ ! -f "$LIBRADC" ]]; then
  echo "Could not find preloaded library in $BUILD_DIR" >&2
  exit 1
fi

parse_cfg() {
  local cfg="$1"
  "$PY_BIN" - <<PY
p = r"$cfg"
run_id = ""
out = ""
warmup = "0"
for line in open(p, "r", encoding="utf-8"):
    t = line.strip()
    if t.startswith("run_id:"):
        run_id = t.split(":",1)[1].strip().strip('"').strip("'")
    elif t.startswith("output_dir:"):
        out = t.split(":",1)[1].strip().strip('"').strip("'")
    elif t.startswith("warmup_epochs:"):
        warmup = t.split(":",1)[1].strip().strip('"').strip("'")
print(run_id)
print(out)
print(warmup)
PY
}

workload_from_run_id() {
  local run_id="$1"
  if [[ "$run_id" == *shock_sigma* || "$run_id" == *"_shock_"* ]]; then
    echo "w2_shock"
  else
    echo "w2_normal"
  fi
}

run_mpi_intercept() {
  local cfg="$1"
  local np="$2"
  local capture_system="${3:-0}"
  local hostfile="${4:-}"

  local parsed run_id out_dir warmup workload
  parsed="$(parse_cfg "$cfg")"
  run_id="$(echo "$parsed" | sed -n '1p')"
  out_dir="$(echo "$parsed" | sed -n '2p')"
  warmup="$(echo "$parsed" | sed -n '3p')"
  workload="$(workload_from_run_id "$run_id")"

  if [[ "${FORCE_RERUN:-0}" != "1" && -f "$ROOT_DIR/$out_dir/metrics_merged.csv" ]]; then
    echo "[top-tier] skip existing run_id=$run_id (set FORCE_RERUN=1 to rerun)"
    return 0
  fi

  echo "[top-tier] run_id=$run_id np=$np cfg=$cfg hostfile=${hostfile:-<none>}"
  local cmd=("$BUILD_DIR/bench_stage3_intercept_driver" "--config" "$cfg")
  if [[ "$(uname -s)" == "Darwin" ]]; then
    cmd=(env DYLD_FORCE_FLAT_NAMESPACE=1 DYLD_INSERT_LIBRARIES="$LIBRADC" "${cmd[@]}")
  else
    cmd=(env LD_PRELOAD="$LIBRADC" "${cmd[@]}")
  fi

  local mpicmd=(mpirun --oversubscribe -np "$np")
  if [[ -n "$hostfile" ]]; then
    mpicmd+=(--hostfile "$hostfile")
  fi

  mkdir -p "$ROOT_DIR/$out_dir/system"
  if [[ "$capture_system" == "1" ]]; then
    local perf_file="$ROOT_DIR/$out_dir/system/perf_stat.txt"
    if command -v perf >/dev/null 2>&1; then
      "$ROOT_DIR/scripts/capture_net_bytes.sh" "$ROOT_DIR/$out_dir/system/net_bytes.csv" -- \
        perf stat -e cycles,instructions,cache-misses,LLC-load-misses -o "$perf_file" \
        "${mpicmd[@]}" "${cmd[@]}"
    else
      echo "perf not available on this host" > "$perf_file"
      "$ROOT_DIR/scripts/capture_net_bytes.sh" "$ROOT_DIR/$out_dir/system/net_bytes.csv" -- \
        "${mpicmd[@]}" "${cmd[@]}"
    fi
  else
    "${mpicmd[@]}" "${cmd[@]}"
  fi

  cp -f "$cfg" "$ROOT_DIR/$out_dir/run_config.yaml"
  "$PY_BIN" - <<PY
import json, os, platform, subprocess
out = r"$ROOT_DIR/$out_dir/build_info.json"
info = {
  "platform": platform.platform(),
  "python": platform.python_version(),
}
for k, c in [("cmake_version", ["cmake", "--version"]), ("mpirun_version", ["mpirun", "--version"]), ("cxx", ["c++", "--version"])]:
  try:
    info[k] = subprocess.check_output(c, text=True, stderr=subprocess.STDOUT).splitlines()[0]
  except Exception as e:
    info[k] = f"unavailable: {e}"
with open(out, "w", encoding="utf-8") as f:
  json.dump(info, f, indent=2)
PY

  "$ROOT_DIR/scripts/postprocess.sh" "$out_dir" "$workload" "$warmup"
}

combine_pair() {
  local exact_cfg="$1"
  local method_cfg="$2"
  local exact_parsed method_parsed exact_run method_run exact_out method_out method_warmup workload
  exact_parsed="$(parse_cfg "$exact_cfg")"
  method_parsed="$(parse_cfg "$method_cfg")"
  exact_run="$(echo "$exact_parsed" | sed -n '1p')"
  exact_out="$(echo "$exact_parsed" | sed -n '2p')"
  method_run="$(echo "$method_parsed" | sed -n '1p')"
  method_out="$(echo "$method_parsed" | sed -n '2p')"
  method_warmup="$(echo "$method_parsed" | sed -n '3p')"
  workload="$(workload_from_run_id "$method_run")"

  local combined_dir="$ROOT_DIR/results/${method_run}_combined"
  mkdir -p "$combined_dir"

  "$PY_BIN" "$ROOT_DIR/python/combine_metrics.py" \
    --exact "$ROOT_DIR/$exact_out/metrics_merged.csv" \
    --compressed "$ROOT_DIR/$method_out/metrics_merged.csv" \
    --output "$combined_dir/metrics_merged.csv"

  for ef in "$ROOT_DIR/$exact_out"/metrics_rank*.csv; do
    local bn rf rank_idx mf of
    bn="$(basename "$ef")"
    rf="${bn#metrics_rank}"
    rank_idx="${rf%.csv}"
    mf="$ROOT_DIR/$method_out/metrics_rank$rf"
    of="$combined_dir/$bn"
    if [[ ! -f "$mf" ]]; then
      continue
    fi
    head -n 1 "$ef" > "$of"
    tail -n +2 "$ef" >> "$of"
    tail -n +2 "$mf" >> "$of"

    if [[ -f "$ROOT_DIR/$exact_out/events_rank${rank_idx}.jsonl" || -f "$ROOT_DIR/$method_out/events_rank${rank_idx}.jsonl" ]]; then
      {
        [[ -f "$ROOT_DIR/$exact_out/events_rank${rank_idx}.jsonl" ]] && cat "$ROOT_DIR/$exact_out/events_rank${rank_idx}.jsonl"
        [[ -f "$ROOT_DIR/$method_out/events_rank${rank_idx}.jsonl" ]] && cat "$ROOT_DIR/$method_out/events_rank${rank_idx}.jsonl"
      } > "$combined_dir/events_rank${rank_idx}.jsonl"
    fi
  done

  mkdir -p "$combined_dir/system"
  if [[ -f "$ROOT_DIR/$method_out/system/net_bytes.csv" ]]; then
    cp -f "$ROOT_DIR/$method_out/system/net_bytes.csv" "$combined_dir/system/net_bytes.csv"
  fi
  if [[ -f "$ROOT_DIR/$method_out/system/perf_stat.txt" ]]; then
    cp -f "$ROOT_DIR/$method_out/system/perf_stat.txt" "$combined_dir/system/perf_stat.txt"
  fi
  cp -f "$exact_cfg" "$combined_dir/run_config_exact.yaml"
  cp -f "$method_cfg" "$combined_dir/run_config_method.yaml"
  if [[ -f "$ROOT_DIR/$method_out/build_info.json" ]]; then
    cp -f "$ROOT_DIR/$method_out/build_info.json" "$combined_dir/build_info.json"
  fi

  "$PY_BIN" "$ROOT_DIR/python/make_tables.py" \
    --run_dir "$combined_dir" \
    --workload "$workload" \
    --warmup_epochs "$method_warmup"
  "$PY_BIN" "$ROOT_DIR/python/make_figures.py" \
    --run_dir "$combined_dir" \
    --warmup_epochs "$method_warmup"
}

REPORT="$ROOT_DIR/results/top_tier_final/VALIDATION_REPORT.md"
mkdir -p "$(dirname "$REPORT")"
cat > "$REPORT" <<'EOF'
# VALIDATION_REPORT

## Evaluation Setup
- toy sanity: P=2
- medium: P=8
- network-bound: P=16, N=8192, S=8192, float32 (payload/rank/epoch = 256MB)
- shock sweep: sigma in {1,2,4,6,8,10}
- metric basis: per-epoch rank-max latency, excluding warmup and shadow epochs

EOF

append_validation() {
  local run_id="$1"
  local warmup="$2"
  local skip_false_accept="${3:-0}"
  shift 3
  local run_dir="$ROOT_DIR/results/${run_id}_combined"
  local out
  local -a vcmd=("$PY_BIN" "$ROOT_DIR/python/validate_run.py" --run_dir "$run_dir" --warmup_epochs "$warmup")
  if [[ "$skip_false_accept" == "1" ]]; then
    vcmd+=(--skip_false_accept_gate)
  fi
  if [[ $# -gt 0 ]]; then
    vcmd+=("$@")
  fi
  set +e
  out="$("${vcmd[@]}" 2>&1)"
  local rc=$?
  set -e
  {
    echo "## $run_id"
    echo ""
    echo "- validation_exit_code: $rc"
    echo '```'
    echo "$out"
    echo '```'
    if [[ -f "$run_dir/tables/table1_main.csv" ]]; then
      echo "- table1_main:"
      echo '```'
      cat "$run_dir/tables/table1_main.csv"
      echo '```'
    fi
    echo ""
  } >> "$REPORT"
  if [[ $rc -ne 0 ]]; then
    echo "Validation failed for $run_id" >&2
    return 1
  fi
}

TOY_EXACT="$ROOT_DIR/configs/toy_w2_exact_f32.yaml"
MED_EXACT="$ROOT_DIR/configs/medium_w2_exact_f32.yaml"
NET_EXACT="$ROOT_DIR/configs/network_w2_exact_f32.yaml"

EXACT_CFGS=(
  "$TOY_EXACT"
  "$MED_EXACT"
  "$NET_EXACT"
  "$ROOT_DIR/configs/network_w2_shock_sigma1_exact_f32.yaml"
  "$ROOT_DIR/configs/network_w2_shock_sigma2_exact_f32.yaml"
  "$ROOT_DIR/configs/network_w2_shock_sigma4_exact_f32.yaml"
  "$ROOT_DIR/configs/network_w2_shock_sigma6_exact_f32.yaml"
  "$ROOT_DIR/configs/network_w2_shock_sigma8_exact_f32.yaml"
  "$ROOT_DIR/configs/network_w2_shock_sigma10_exact_f32.yaml"
)

METHOD_CFGS=(
  "$ROOT_DIR/configs/toy_w2_radc.yaml"
  "$ROOT_DIR/configs/toy_w2_nosafety.yaml"
  "$ROOT_DIR/configs/toy_w2_fixedrank.yaml"
  "$ROOT_DIR/configs/medium_w2_radc.yaml"
  "$ROOT_DIR/configs/medium_w2_nosafety.yaml"
  "$ROOT_DIR/configs/medium_w2_fixedrank.yaml"
  "$ROOT_DIR/configs/network_w2_radc.yaml"
  "$ROOT_DIR/configs/network_w2_nosafety.yaml"
  "$ROOT_DIR/configs/network_w2_fixedrank.yaml"
  "$ROOT_DIR/configs/network_w2_shock_sigma1_radc.yaml"
  "$ROOT_DIR/configs/network_w2_shock_sigma2_radc.yaml"
  "$ROOT_DIR/configs/network_w2_shock_sigma4_radc.yaml"
  "$ROOT_DIR/configs/network_w2_shock_sigma6_radc.yaml"
  "$ROOT_DIR/configs/network_w2_shock_sigma8_radc.yaml"
  "$ROOT_DIR/configs/network_w2_shock_sigma10_radc.yaml"
)

for cfg in "${EXACT_CFGS[@]}"; do
  run_id="$(parse_cfg "$cfg" | sed -n '1p')"
  np="$NP_NETWORK"
  if [[ "$run_id" == toy_* ]]; then
    np="$NP_TOY"
  elif [[ "$run_id" == medium_* ]]; then
    np="$NP_MEDIUM"
  fi
  run_mpi_intercept "$cfg" "$np" 0
done

for cfg in "${METHOD_CFGS[@]}"; do
  parsed="$(parse_cfg "$cfg")"
  run_id="$(echo "$parsed" | sed -n '1p')"
  warmup="$(echo "$parsed" | sed -n '3p')"

  np="$NP_NETWORK"
  capture=0
  exact="$NET_EXACT"
  if [[ "$run_id" == toy_* ]]; then
    np="$NP_TOY"
    exact="$TOY_EXACT"
  elif [[ "$run_id" == medium_* ]]; then
    np="$NP_MEDIUM"
    exact="$MED_EXACT"
  elif [[ "$run_id" =~ ^network_w2_shock_sigma([0-9]+)_radc$ ]]; then
    sigma="${BASH_REMATCH[1]}"
    exact="$ROOT_DIR/configs/network_w2_shock_sigma${sigma}_exact_f32.yaml"
  else
    if [[ "$run_id" == "network_w2_radc" ]]; then
      capture=1
    fi
  fi

  run_mpi_intercept "$cfg" "$np" "$capture"
  combine_pair "$exact" "$cfg"

  skip_false_accept=0
  declare -a extra_gates=()
  if [[ "$run_id" == *_nosafety ]]; then
    skip_false_accept=1
  else
    extra_gates+=(--require_epsilon_bps 0.01)
  fi

  if [[ "$run_id" == "network_w2_radc" ]]; then
    extra_gates+=(--require_bytes_reduction_pct_min 90.0 --require_fallback_pct_max 5.0)
  fi
  if [[ "$run_id" == "network_w2_shock_sigma1_radc" ]]; then
    extra_gates+=(--require_fallback_pct_max 1.0)
  fi
  if [[ "$run_id" == "network_w2_shock_sigma6_radc" || "$run_id" == "network_w2_shock_sigma8_radc" || "$run_id" == "network_w2_shock_sigma10_radc" ]]; then
    extra_gates+=(--require_fallback_pct_min 95.0)
  fi
  if [[ "$run_id" == "network_w2_shock_sigma10_radc" ]]; then
    extra_gates+=(--max_p99_over_exact_ratio 1.05 --require_fallback_pct_min 99.0)
  fi

  if [[ ${#extra_gates[@]} -gt 0 ]]; then
    append_validation "$run_id" "$warmup" "$skip_false_accept" "${extra_gates[@]}"
  else
    append_validation "$run_id" "$warmup" "$skip_false_accept"
  fi
done

REAL_RAN=0
if [[ -n "$REAL_HOSTFILE" && -f "$REAL_HOSTFILE" ]]; then
  REAL_RAN=1
  REAL_EXACT_CFG="$ROOT_DIR/configs/network_w2_real_exact_f32.yaml"
  REAL_RADC_CFG="$ROOT_DIR/configs/network_w2_real_radc.yaml"
  "$PY_BIN" - <<PY
import re
from pathlib import Path
root = Path(r"$ROOT_DIR")
exact_src = (root / "configs/network_w2_exact_f32.yaml").read_text()
radc_src = (root / "configs/network_w2_radc.yaml").read_text()

def patch(txt, run_id):
    txt = re.sub(r'run_id:\\s*\"[^\"]+\"', f'run_id: \"{run_id}\"', txt)
    txt = re.sub(r'output_dir:\\s*\"[^\"]+\"', f'output_dir: \"results/{run_id}\"', txt)
    txt = re.sub(r'net_emulation_bandwidth_gbps:\\s*[-+0-9.eE]+', 'net_emulation_bandwidth_gbps: 0.0', txt)
    txt = re.sub(r'net_emulation_base_latency_ms:\\s*[-+0-9.eE]+', 'net_emulation_base_latency_ms: 0.0', txt)
    return txt

(root / "configs/network_w2_real_exact_f32.yaml").write_text(patch(exact_src, "network_w2_real_exact_f32"))
(root / "configs/network_w2_real_radc.yaml").write_text(patch(radc_src, "network_w2_real_radc"))
PY
  run_mpi_intercept "$REAL_EXACT_CFG" "$REAL_NP" 1 "$REAL_HOSTFILE"
  run_mpi_intercept "$REAL_RADC_CFG" "$REAL_NP" 1 "$REAL_HOSTFILE"
  combine_pair "$REAL_EXACT_CFG" "$REAL_RADC_CFG"
  append_validation "network_w2_real_radc" "10" "0" \
    --require_epsilon_bps 0.01 \
    --require_bytes_reduction_pct_min 90.0
else
  {
    echo "## real_multi_node_gate"
    echo ""
    if [[ "$ALLOW_EMULATION_ONLY" == "1" ]]; then
      echo "- status: SKIPPED (no REAL_HOSTFILE provided; emulation-only evidence in this environment)"
    else
      echo "- status: FAIL (REAL_HOSTFILE required for real multi-node gate)"
    fi
    echo ""
  } >> "$REPORT"
  if [[ "$ALLOW_EMULATION_ONLY" != "1" ]]; then
    echo "REAL_HOSTFILE not provided and ALLOW_EMULATION_ONLY!=1" >&2
    exit 1
  fi
fi

"$PY_BIN" "$ROOT_DIR/python/top_tier_summary.py" \
  --repo_root "$ROOT_DIR" \
  --out_dir "results/top_tier_final"

echo "Top-tier evaluation complete. See:"
echo "  $ROOT_DIR/results/top_tier_final"
echo "  $REPORT"
