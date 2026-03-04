#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include <mpi.h>

#include "radc/compressed_protocol.h"
#include "radc/config.h"
#include "radc/logging.h"
#include "radc/matrix_view.h"

namespace {

struct Cli {
  std::string config;
  bool force_exact = false;
  std::string run_suffix;
};

uint64_t splitmix64(uint64_t x) {
  uint64_t z = x + 0x9E3779B97F4A7C15ULL;
  z = (z ^ (z >> 30U)) * 0xBF58476D1CE4E5B9ULL;
  z = (z ^ (z >> 27U)) * 0x94D049BB133111EBULL;
  return z ^ (z >> 31U);
}

Cli parse_cli(int argc, char** argv) {
  Cli cli{};
  for (int i = 1; i < argc; ++i) {
    if (std::strcmp(argv[i], "--config") == 0 && i + 1 < argc) {
      cli.config = argv[++i];
    } else if (std::strcmp(argv[i], "--force_exact") == 0) {
      cli.force_exact = true;
    } else if (std::strcmp(argv[i], "--run_suffix") == 0 && i + 1 < argc) {
      cli.run_suffix = argv[++i];
    }
  }
  return cli;
}

void apply_cli_overrides(const Cli& cli, radc::Config& cfg) {
  if (cli.force_exact) {
    cfg.radc.enabled = false;
  }
  if (!cli.run_suffix.empty()) {
    cfg.run.run_id += "_" + cli.run_suffix;
    cfg.run.output_dir += "_" + cli.run_suffix;
  }
}

void fill_pca_like(std::vector<float>& A, int64_t N, int64_t S, const radc::Config& cfg,
                   int64_t epoch, int rank) {
  const int r = std::max(1, std::min<int>(cfg.workload.pca_like.true_rank,
                                          static_cast<int>(std::min<int64_t>(N, S))));
  const double noise = cfg.workload.pca_like.noise_scale;

  const uint64_t seed_epoch = splitmix64(cfg.run.seed ^ static_cast<uint64_t>(epoch));
  const uint64_t seed_rank = splitmix64(seed_epoch ^ static_cast<uint64_t>(rank));
  std::mt19937_64 rng(seed_rank);
  std::normal_distribution<double> nd(0.0, 1.0);

  std::vector<double> F(static_cast<size_t>(N) * static_cast<size_t>(r), 0.0);
  std::vector<double> G(static_cast<size_t>(S) * static_cast<size_t>(r), 0.0);

  for (double& x : F) {
    x = nd(rng);
  }
  for (double& x : G) {
    x = nd(rng);
  }

  std::vector<double> sigma(static_cast<size_t>(r), 0.0);
  for (int k = 0; k < r; ++k) {
    sigma[static_cast<size_t>(k)] = 1.0 / static_cast<double>(k + 1);
  }

  for (int64_t n = 0; n < N; ++n) {
    float* row = A.data() + static_cast<size_t>(n) * static_cast<size_t>(S);
    for (int64_t s = 0; s < S; ++s) {
      double acc = 0.0;
      for (int k = 0; k < r; ++k) {
        acc += sigma[static_cast<size_t>(k)] *
               F[static_cast<size_t>(n) * static_cast<size_t>(r) + static_cast<size_t>(k)] *
               G[static_cast<size_t>(s) * static_cast<size_t>(r) + static_cast<size_t>(k)];
      }
      row[s] = static_cast<float>(acc + noise * nd(rng));
    }
  }
}

radc::MetricsRow make_row(const radc::Config& cfg, int rank, int world, int64_t epoch,
                          const radc::ProtocolMetrics& pm, double epoch_ms) {
  radc::MetricsRow row{};
  row.run_id = cfg.run.run_id;
  row.rank = rank;
  row.world_size = world;
  row.epoch = epoch;
  row.mode = pm.mode;
  row.N = cfg.buffer.N;
  row.S = cfg.buffer.S;
  row.dtype_in = cfg.buffer.dtype_in;
  row.dtype_internal = pm.dtype_internal;
  row.layout = cfg.buffer.layout;
  row.factor_dtype = cfg.radc.factor_dtype;
  row.l = pm.l;
  row.l_used = pm.l_used;
  row.r_used = pm.r_used;
  row.r_max = cfg.radc.r_max;
  row.oversample_p = cfg.radc.oversample_p;
  row.power_iters = cfg.radc.power_iters;
  row.energy_at_r = pm.energy_at_r;
  row.frob_norm_exact = pm.frob_norm_exact;
  row.rho_hat_sketch_0 = pm.rho_hat_sketch_0;
  row.rho_hat_sketch_1 = pm.rho_hat_sketch_1;
  row.rho_hat_sketch_max = pm.rho_hat_sketch_max;
  row.rho_hat_sketch = pm.rho_hat_sketch;
  row.rho_max = pm.rho_max;
  row.eps_dollars = pm.eps_dollars;
  row.accept_threshold = pm.accept_threshold;
  row.kG = pm.kG;
  row.kS = pm.kS;
  row.num_sketches = pm.num_sketches;
  row.accepted = pm.accepted;
  row.fallback_triggered = pm.fallback_triggered;
  row.bytes_exact_payload = pm.bytes_exact_payload;
  row.bytes_comp_attempt_payload = pm.bytes_comp_attempt_payload;
  row.bytes_exact_fallback_payload = pm.bytes_exact_fallback_payload;
  row.bytes_total_payload = pm.bytes_total_payload;
  row.bytes_comp_payload = pm.bytes_comp_payload;
  row.bytes_effective_payload = pm.bytes_total_payload;
  row.compression_ratio_payload = pm.compression_ratio_payload;
  row.t_compute_Y_ms = pm.t_compute_Y_ms;
  row.t_allreduce_Y_ms = pm.t_allreduce_Y_ms;
  row.t_qr_ms = pm.t_qr_ms;
  row.t_compute_B_ms = pm.t_compute_B_ms;
  row.t_allreduce_B_ms = pm.t_allreduce_B_ms;
  row.t_small_svd_ms = pm.t_small_svd_ms;
  row.t_reconstruct_ms = pm.t_reconstruct_ms;
  row.t_sketch_local_ms = pm.t_sketch_local_ms;
  row.t_allreduce_sketch_ms = pm.t_allreduce_sketch_ms;
  row.t_verify_ms = pm.t_verify_ms;
  row.t_exact_fallback_ms = pm.t_exact_fallback_ms;
  row.t_epoch_total_ms = epoch_ms;
  row.xva_true = pm.xva_true;
  row.xva_approx = pm.xva_approx;
  row.xva_err_abs = pm.xva_err_abs;
  row.xva_err_bps = pm.xva_err_bps;
  row.L_xva = pm.L_xva;
  row.total_notional = pm.total_notional;
  row.xva_epsilon_bps = pm.xva_epsilon_bps;
  row.accept_margin = pm.accept_margin;
  row.perf_cycles = std::numeric_limits<double>::quiet_NaN();
  row.perf_instructions = std::numeric_limits<double>::quiet_NaN();
  row.perf_cache_misses = std::numeric_limits<double>::quiet_NaN();
  row.perf_llc_load_misses = std::numeric_limits<double>::quiet_NaN();
  row.energy_pkg_joules = std::numeric_limits<double>::quiet_NaN();
  row.warmup_epochs = cfg.run.warmup_epochs;
  row.shock_sigma = cfg.workload.delta_gamma.shock_sigma;
  row.is_shadow_epoch = pm.is_shadow_epoch;
  return row;
}

int run(const radc::Config& cfg) {
  int rank = 0;
  int world = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &world);

  radc::PerRankLogger logger(cfg, rank);
  radc::ProtocolRuntime rt{};
  radc::init_protocol_runtime(cfg, cfg.buffer.N, cfg.buffer.S, rt);

  std::vector<float> local(static_cast<size_t>(cfg.buffer.N) * static_cast<size_t>(cfg.buffer.S), 0.0f);
  std::vector<float> out;

  using SetEpochFn = void (*)(int64_t);
  SetEpochFn set_epoch = reinterpret_cast<SetEpochFn>(dlsym(RTLD_DEFAULT, "radc_set_epoch"));

  for (int64_t epoch = 0; epoch < cfg.run.epochs; ++epoch) {
    if (cfg.mpi.barrier_before_epochs) {
      MPI_Barrier(MPI_COMM_WORLD);
    }

    if (set_epoch != nullptr) {
      set_epoch(epoch);
    }

    fill_pca_like(local, cfg.buffer.N, cfg.buffer.S, cfg, epoch, rank);

    const radc::MatrixView<const float> A_view{local.data(), cfg.buffer.N, cfg.buffer.S, cfg.buffer.S};
    radc::ProtocolMetrics pm{};

    const double t0 = MPI_Wtime();
    radc::run_protocol_epoch(MPI_COMM_WORLD, cfg, epoch, A_view, rt, out, pm, cfg.risk.notional_total);
    const double t1 = MPI_Wtime();
    const double epoch_ms = 1000.0 * (t1 - t0);

    if (cfg.mpi.barrier_after_epochs) {
      MPI_Barrier(MPI_COMM_WORLD);
    }

    logger.log_metric(make_row(cfg, rank, world, epoch, pm, epoch_ms));
  }

  if (rank == 0) {
    double checksum = 0.0;
    const size_t upto = std::min<size_t>(out.size(), 64);
    for (size_t i = 0; i < upto; ++i) {
      checksum += out[i];
    }
    std::cout << "bench_pca_like checksum=" << checksum << std::endl;
  }

  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);

  const Cli cli = parse_cli(argc, argv);
  const std::string cfg_path = radc::resolve_config_path(cli.config);
  setenv("RADC_CONFIG", cfg_path.c_str(), 1);

  radc::Config cfg{};
  try {
    cfg = radc::load_config_from_env_or_default(cli.config);
    apply_cli_overrides(cli, cfg);
  } catch (const std::exception& e) {
    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (rank == 0) {
      std::cerr << "Failed to load config: " << e.what() << std::endl;
    }
    MPI_Finalize();
    return 1;
  }

  int rc = 0;
  try {
    rc = run(cfg);
  } catch (const std::exception& e) {
    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (rank == 0) {
      std::cerr << "Run failed: " << e.what() << std::endl;
    }
    rc = 2;
  }

  MPI_Finalize();
  return rc;
}
