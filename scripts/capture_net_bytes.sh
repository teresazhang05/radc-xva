#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 <out_csv> -- <command...>" >&2
  exit 2
fi

OUT_CSV="$1"
shift
if [[ "${1:-}" != "--" ]]; then
  echo "Usage: $0 <out_csv> -- <command...>" >&2
  exit 2
fi
shift

if [[ $# -lt 1 ]]; then
  echo "No command provided" >&2
  exit 2
fi

read_net_totals() {
  python3 - <<'PY'
import os
import re
import subprocess
import sys

def from_proc():
    p = "/proc/net/dev"
    if not os.path.exists(p):
        return None
    rx = 0
    tx = 0
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            if ":" not in line:
                continue
            iface, rest = line.split(":", 1)
            iface = iface.strip()
            if iface == "lo":
                continue
            vals = rest.split()
            if len(vals) < 16:
                continue
            rx += int(vals[0])
            tx += int(vals[8])
    return rx, tx

def from_netstat_macos():
    try:
        out = subprocess.check_output(["netstat", "-ibn"], text=True, stderr=subprocess.DEVNULL)
    except Exception:
        return None
    lines = out.splitlines()
    if not lines:
        return None
    header = re.split(r"\s+", lines[0].strip())
    try:
        i_name = header.index("Name")
        i_ibytes = header.index("Ibytes")
        i_obytes = header.index("Obytes")
    except ValueError:
        return None
    rx = 0
    tx = 0
    for line in lines[1:]:
        cols = re.split(r"\s+", line.strip())
        if len(cols) <= max(i_name, i_ibytes, i_obytes):
            continue
        iface = cols[i_name]
        if iface.startswith("lo"):
            continue
        try:
            rx += int(cols[i_ibytes])
            tx += int(cols[i_obytes])
        except Exception:
            continue
    return rx, tx

vals = from_proc()
if vals is None:
    vals = from_netstat_macos()
if vals is None:
    print("NaN,NaN")
else:
    print(f"{vals[0]},{vals[1]}")
PY
}

mkdir -p "$(dirname "$OUT_CSV")"

BEFORE="$(read_net_totals)"
RX_BEFORE="${BEFORE%%,*}"
TX_BEFORE="${BEFORE##*,}"

set +e
"$@"
RC=$?
set -e

AFTER="$(read_net_totals)"
RX_AFTER="${AFTER%%,*}"
TX_AFTER="${AFTER##*,}"

python3 - "$OUT_CSV" "$RX_BEFORE" "$TX_BEFORE" "$RX_AFTER" "$TX_AFTER" "$RC" "$*" <<'PY'
import csv
import math
import os
import sys

out_csv, rx_b, tx_b, rx_a, tx_a, rc, cmd = sys.argv[1:]

def to_num(v):
    try:
        return float(v)
    except Exception:
        return float("nan")

rb = to_num(rx_b)
tb = to_num(tx_b)
ra = to_num(rx_a)
ta = to_num(tx_a)

rx_delta = (ra - rb) if math.isfinite(ra) and math.isfinite(rb) else float("nan")
tx_delta = (ta - tb) if math.isfinite(ta) and math.isfinite(tb) else float("nan")

with open(out_csv, "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["rx_before_bytes", "tx_before_bytes", "rx_after_bytes", "tx_after_bytes", "rx_delta_bytes", "tx_delta_bytes", "exit_code", "command"])
    w.writerow([rx_b, tx_b, rx_a, tx_a, f"{rx_delta:.0f}" if math.isfinite(rx_delta) else "NaN",
                f"{tx_delta:.0f}" if math.isfinite(tx_delta) else "NaN", rc, cmd])
PY

exit "$RC"
