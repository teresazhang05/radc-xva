#!/usr/bin/env python3
import argparse
import csv
import glob
import os
import sys


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    pattern = os.path.join(args.input_dir, "metrics_rank*.csv")
    files = sorted(glob.glob(pattern))
    if not files:
        print(f"No files found for pattern: {pattern}", file=sys.stderr)
        return 1

    rows = []
    header = None
    for path in files:
        with open(path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames is None:
                continue
            if header is None:
                header = reader.fieldnames
            elif header != reader.fieldnames:
                raise RuntimeError(f"Header mismatch in {path}")
            rows.extend(reader)

    if header is None:
        print("No rows available to merge", file=sys.stderr)
        return 1

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Merged {len(files)} files -> {args.output} ({len(rows)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
