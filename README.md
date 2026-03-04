# radc-xva

Rank-Adaptive Differential Communication (RADC) for distributed XVA.

## Status
- Stage 0 complete: wrapper scaffold, logging, pass-through correctness.
- Stage 1 complete: RandSVD/CountSketch/safety kernels + tests.
- Stage 2 complete: non-intercept compressed protocol path in benchmark binaries.
- Stage 3 complete: compression path moved into the `MPI_Allreduce` interceptor for strict v1 eligible calls.
- Stage 4 complete: paper-level summary tables/figures + artifact packaging.
- Stage 5 complete: ablation/stress suite + gate audit + final Stage 0-5 artifact package.

## Strict v1 interceptor scope
- Intercept target: `MPI_Allreduce`
- Eligible compressed path: `MPI_SUM`, contiguous `N*S` row-major buffers, datatype in `{MPI_FLOAT, MPI_DOUBLE}`, and config-compatible dtype.
- `MPI_IN_PLACE` remains pass-through.
- All ineligible calls are exact pass-through to `PMPI_Allreduce`.
- `compression.double_mode` controls `MPI_DOUBLE` behavior:
  - `native64` (default): full native float64 compressed kernels + float64 sketch cert
  - `downcast32`: explicit ablation path (`double -> float32 kernels -> cast back`)
  - `passthrough`: exact `PMPI_Allreduce` with original datatype
- Certification path: exposure-space dual CountSketch (`num_sketches=2`) with one-sided acceptance threshold
  `accept_margin*(1-jl_epsilon)*rho_max`.
- Shadow exact (`logging.shadow_exact_every`) is logging/validation-only and is never used for accept/reject.

## Build
```bash
bash scripts/build.sh
```

## Run benchmark (Stage 2 non-intercept protocol path)
```bash
mpirun -np 2 ./build/bench_xva_delta_gamma --config configs/w1_synth_small.yaml
```

## Run benchmark with interceptor
```bash
export RADC_CONFIG=configs/w1_synth_small.yaml
# Linux
mpirun -np 2 env LD_PRELOAD=./build/libradc.so \
  ./build/bench_xva_delta_gamma --config configs/w1_synth_small.yaml

# macOS
mpirun -np 2 env DYLD_FORCE_FLAT_NAMESPACE=1 DYLD_INSERT_LIBRARIES=./build/libradc.dylib \
  ./build/bench_xva_delta_gamma --config configs/w1_synth_small.yaml
```

## Run tests
```bash
ctest --test-dir build --output-on-failure
```

## Run full evaluations
```bash
python3 python/validate_configs.py --configs_dir configs --fail_on_deprecated_config 1

# Stage 0-2
bash scripts/run_stage012_full_eval.sh 2

# Stage 0-3 (includes Stage 3 interceptor experiments)
bash scripts/run_stage0123_full_eval.sh 2

# Stage 0-4 (includes paper summaries + artifact bundle packaging)
bash scripts/run_stage01234_full_eval.sh 2

# Stage 0-5 (includes Stage 5 ablation/stress + final package refresh)
bash scripts/run_stage012345_full_eval.sh 2
```

## Notes
- Paths are relative to repo root; no absolute path assumptions.
- Results default under `results/<run_id>/` and can be overridden via config.
