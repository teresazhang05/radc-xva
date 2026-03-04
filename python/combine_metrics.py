#!/usr/bin/env python3
import argparse
import csv
import os


def read_rows(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exact", required=True)
    parser.add_argument("--compressed", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    rows_exact = read_rows(args.exact)
    rows_comp = read_rows(args.compressed)

    header = []
    if rows_exact:
        header = list(rows_exact[0].keys())
    elif rows_comp:
        header = list(rows_comp[0].keys())
    else:
        raise RuntimeError("No rows in either exact or compressed metrics files")

    rows = rows_exact + rows_comp

    def key(r):
        return (int(r.get("epoch", "0")), int(r.get("rank", "0")), r.get("mode", ""))

    rows.sort(key=key)

    out_dir = os.path.dirname(args.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    with open(args.output, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Combined rows: {len(rows)} -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
