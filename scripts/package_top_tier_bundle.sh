#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
BUNDLE_DIR="$ROOT_DIR/results/top_tier_bundle_latest"
ZIP_OUT="$ROOT_DIR/results/top_tier_bundle_latest.zip"

rm -rf "$BUNDLE_DIR"
mkdir -p "$BUNDLE_DIR"

copy_path() {
  local src="$1"
  local dst="$2"
  if [[ -e "$src" ]]; then
    mkdir -p "$(dirname "$dst")"
    cp -R "$src" "$dst"
  fi
}

copy_path "$ROOT_DIR/CMakeLists.txt" "$BUNDLE_DIR/CMakeLists.txt"
copy_path "$ROOT_DIR/README.md" "$BUNDLE_DIR/README.md"
copy_path "$ROOT_DIR/LICENSE" "$BUNDLE_DIR/LICENSE"
copy_path "$ROOT_DIR/radc" "$BUNDLE_DIR/radc"
copy_path "$ROOT_DIR/bench" "$BUNDLE_DIR/bench"
copy_path "$ROOT_DIR/tests" "$BUNDLE_DIR/tests"
copy_path "$ROOT_DIR/scripts" "$BUNDLE_DIR/scripts"
copy_path "$ROOT_DIR/configs" "$BUNDLE_DIR/configs"
copy_path "$ROOT_DIR/python" "$BUNDLE_DIR/python"
copy_path "$ROOT_DIR/build" "$BUNDLE_DIR/build"
copy_path "$ROOT_DIR/artifact" "$BUNDLE_DIR/artifact"

mkdir -p "$BUNDLE_DIR/results"
for d in "$ROOT_DIR"/results/toy_w2_* "$ROOT_DIR"/results/medium_w2_* "$ROOT_DIR"/results/network_w2_*; do
  if [[ -d "$d" ]]; then
    cp -R "$d" "$BUNDLE_DIR/results/"
  fi
done
copy_path "$ROOT_DIR/results/top_tier_final" "$BUNDLE_DIR/results/top_tier_final"

cat > "$BUNDLE_DIR/REPRODUCE_ONE_COMMAND.sh" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
bash "$ROOT_DIR/scripts/run_top_tier_eval.sh"
EOF
chmod +x "$BUNDLE_DIR/REPRODUCE_ONE_COMMAND.sh"

if [[ -f "$ROOT_DIR/results/top_tier_final/VALIDATION_REPORT.md" ]]; then
  cp -f "$ROOT_DIR/results/top_tier_final/VALIDATION_REPORT.md" \
    "$BUNDLE_DIR/VALIDATION_REPORT.md"
fi

python3 - "$BUNDLE_DIR" <<'PY'
import csv
import hashlib
import os
import sys

bundle = os.path.abspath(sys.argv[1])
rows = []
for root, _, files in os.walk(bundle):
    for fn in files:
        p = os.path.join(root, fn)
        rel = os.path.relpath(p, bundle)
        h = hashlib.sha256()
        with open(p, "rb") as f:
            while True:
                b = f.read(1024 * 1024)
                if not b:
                    break
                h.update(b)
        rows.append({"path": rel, "size_bytes": os.path.getsize(p), "sha256": h.hexdigest()})

rows.sort(key=lambda r: r["path"])
with open(os.path.join(bundle, "MANIFEST.csv"), "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=["path", "size_bytes", "sha256"])
    w.writeheader()
    w.writerows(rows)
PY

rm -f "$ZIP_OUT"
(cd "$ROOT_DIR/results" && zip -qry "$(basename "$ZIP_OUT")" "$(basename "$BUNDLE_DIR")")
shasum -a 256 "$ZIP_OUT" > "$ZIP_OUT.sha256"

echo "Bundle dir: $BUNDLE_DIR"
echo "Bundle zip: $ZIP_OUT"
