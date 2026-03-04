#!/usr/bin/env python3
"""Validate RADC YAML configs against the current schema.

Uses the same flattened key subset approach as the C++ config loader, so it
has no external Python dependencies.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path


DEPRECATED_KEYS: dict[str, str] = {
    "sketch.k_row": "sketch.kG",
    "sketch.k_col": "sketch.kS",
    "sketch.hash_seed_row": "sketch.hash_seed_g0",
    "sketch.sign_seed_row": "sketch.sign_seed_g0",
    "sketch.hash_seed_col": "sketch.hash_seed_s0",
    "sketch.sign_seed_col": "sketch.sign_seed_s0",
    "sketch.jl_epsilon": "safety.jl_epsilon",
    "sketch.jl_delta": "safety.jl_delta",
    "compression.allow_double_downcast": "compression.double_mode",
}

REQUIRED_KEYS: tuple[str, ...] = (
    "run.run_id",
    "run.output_dir",
    "run.epochs",
    "run.warmup_epochs",
    "buffer.N",
    "buffer.S",
    "buffer.layout",
    "buffer.dtype_in",
    "radc.enabled",
    "sketch.kind",
    "sketch.kG",
    "sketch.kS",
    "sketch.num_sketches",
    "sketch.hash_seed_g0",
    "sketch.sign_seed_g0",
    "sketch.hash_seed_s0",
    "sketch.sign_seed_s0",
    "sketch.hash_seed_g1",
    "sketch.sign_seed_g1",
    "sketch.hash_seed_s1",
    "sketch.sign_seed_s1",
    "safety.xva_epsilon_bps",
    "safety.accept_margin",
    "safety.jl_epsilon",
    "safety.jl_delta",
    "compression.double_mode",
    "logging.shadow_exact_every",
)

VALID_DOUBLE_MODE = {"native64", "downcast32", "passthrough"}
VALID_DTYPE_IN = {"float32", "float64"}


@dataclass
class StackItem:
    indent: int
    path: str


def unquote(v: str) -> str:
    s = v.strip()
    if len(s) >= 2 and ((s[0] == '"' and s[-1] == '"') or (s[0] == "'" and s[-1] == "'")):
        return s[1:-1]
    return s


def parse_yaml_subset(path: Path) -> dict[str, str]:
    kv: dict[str, str] = {}
    stack: list[StackItem] = []

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.split("#", 1)[0].rstrip()
        if not line.strip():
            continue

        indent = len(line) - len(line.lstrip(" "))
        trimmed = line.strip()
        if ":" not in trimmed:
            continue

        key, value = trimmed.split(":", 1)
        key = key.strip()
        value = value.strip()

        while stack and indent <= stack[-1].indent:
            stack.pop()

        full_key = f"{stack[-1].path}.{key}" if stack else key

        if value == "":
            stack.append(StackItem(indent=indent, path=full_key))
            continue

        kv[full_key] = unquote(value)

    return kv


def validate_one(path: Path, fail_on_deprecated: bool) -> tuple[list[str], list[str]]:
    errors: list[str] = []
    warnings: list[str] = []

    kv = parse_yaml_subset(path)

    for old, new in DEPRECATED_KEYS.items():
        if old in kv:
            msg = f"DEPRECATED_KEY_USED key={old} replacement={new}"
            warnings.append(msg)
            if fail_on_deprecated:
                errors.append(msg)

    for req in REQUIRED_KEYS:
        if req not in kv:
            errors.append(f"Missing required key: {req}")

    dtype_in = kv.get("buffer.dtype_in")
    if dtype_in is not None and dtype_in not in VALID_DTYPE_IN:
        errors.append(f"buffer.dtype_in must be one of {sorted(VALID_DTYPE_IN)}, got {dtype_in!r}")

    mode = kv.get("compression.double_mode")
    if mode is not None and mode not in VALID_DOUBLE_MODE:
        errors.append(f"compression.double_mode must be one of {sorted(VALID_DOUBLE_MODE)}, got {mode!r}")

    num_sketches = kv.get("sketch.num_sketches")
    if num_sketches is not None:
        try:
            num = int(num_sketches)
        except Exception:
            errors.append(f"sketch.num_sketches must be an integer, got {num_sketches!r}")
        else:
            if num < 1 or num > 2:
                errors.append("sketch.num_sketches must be in [1,2]")

    for seed_key in (
        "sketch.hash_seed_g0",
        "sketch.sign_seed_g0",
        "sketch.hash_seed_s0",
        "sketch.sign_seed_s0",
        "sketch.hash_seed_g1",
        "sketch.sign_seed_g1",
        "sketch.hash_seed_s1",
        "sketch.sign_seed_s1",
    ):
        if seed_key in kv:
            try:
                int(kv[seed_key])
            except Exception:
                errors.append(f"{seed_key} must be an integer, got {kv[seed_key]!r}")

    for numeric_key in (
        "safety.xva_epsilon_bps",
        "safety.accept_margin",
        "safety.jl_epsilon",
        "safety.jl_delta",
        "buffer.N",
        "buffer.S",
    ):
        if numeric_key in kv:
            try:
                float(kv[numeric_key])
            except Exception:
                errors.append(f"{numeric_key} must be numeric, got {kv[numeric_key]!r}")

    return errors, warnings


def discover_configs(configs_dir: Path) -> list[Path]:
    files = sorted(configs_dir.glob("*.yaml")) + sorted(configs_dir.glob("*.yml"))
    return sorted(set(files))


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate config schema and deprecation usage")
    parser.add_argument("--configs_dir", type=Path, default=Path("configs"), help="Directory containing YAML configs")
    parser.add_argument(
        "--fail_on_deprecated_config",
        type=int,
        default=0,
        help="If 1, deprecated keys are treated as errors",
    )
    parser.add_argument("--quiet", action="store_true", help="Only print summary and failures")
    args = parser.parse_args()

    cfg_files = discover_configs(args.configs_dir)
    if not cfg_files:
        print(f"No config files found under {args.configs_dir}")
        return 1

    fail_on_deprecated = bool(args.fail_on_deprecated_config)
    total_errors = 0
    total_warnings = 0

    for cfg_path in cfg_files:
        errors, warnings = validate_one(cfg_path, fail_on_deprecated)
        total_errors += len(errors)
        total_warnings += len(warnings)

        if errors or (warnings and not args.quiet):
            print(f"[{cfg_path}]")
            for w in warnings:
                print(f"  WARN: {w}")
            for e in errors:
                print(f"  ERROR: {e}")

    status = "PASS" if total_errors == 0 else "FAIL"
    print(
        f"CONFIG_VALIDATION_{status}: files={len(cfg_files)} errors={total_errors} warnings={total_warnings} "
        f"fail_on_deprecated={int(fail_on_deprecated)}"
    )
    return 0 if total_errors == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
