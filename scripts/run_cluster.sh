#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
BUILD_DIR="$ROOT_DIR/build"
CONFIG="${1:-$ROOT_DIR/configs/w1_synth_cluster.yaml}"
NP="${2:-8}"

export RADC_CONFIG="$CONFIG"

"$ROOT_DIR/scripts/build.sh"

LIBRADC="$BUILD_DIR/libradc.so"
if [[ ! -f "$LIBRADC" && -f "$BUILD_DIR/libradc.dylib" ]]; then
  LIBRADC="$BUILD_DIR/libradc.dylib"
fi

if [[ ! -f "$LIBRADC" ]]; then
  echo "Could not find preloaded library in $BUILD_DIR" >&2
  exit 1
fi

if [[ "$(uname -s)" == "Darwin" ]]; then
  mpirun -np "$NP" env DYLD_FORCE_FLAT_NAMESPACE=1 DYLD_INSERT_LIBRARIES="$LIBRADC" \
    "$BUILD_DIR/bench_xva_delta_gamma" --config "$CONFIG"
else
  mpirun -np "$NP" env LD_PRELOAD="$LIBRADC" "$BUILD_DIR/bench_xva_delta_gamma" --config "$CONFIG"
fi

echo "Cluster-style run complete for config: $CONFIG"
