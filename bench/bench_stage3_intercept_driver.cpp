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

#include "radc/config.h"

namespace {

struct Cli {
  std::string config;
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
    }
  }
  return cli;
}

void fill_synth_lowrank(std::vector<double>& A, int64_t N, int64_t S, const radc::Config& cfg,
                        int64_t epoch, int rank) {
  const int r = std::max(1, std::min<int>(cfg.workload.synth_lowrank.true_rank,
                                          static_cast<int>(std::min<int64_t>(N, S))));
  const double noise = cfg.workload.synth_lowrank.noise_scale;
  const bool heavy_tail = cfg.workload.synth_lowrank.heavy_tail;

  std::vector<double> U(static_cast<size_t>(N) * static_cast<size_t>(r), 0.0);
  std::vector<double> V(static_cast<size_t>(S) * static_cast<size_t>(r), 0.0);

  const uint64_t seed_epoch = splitmix64(cfg.run.seed ^ static_cast<uint64_t>(epoch));
  const uint64_t seed_rank = splitmix64(seed_epoch ^ static_cast<uint64_t>(rank));
  std::mt19937_64 rng(seed_rank);
  std::normal_distribution<double> nd(0.0, 1.0);

  for (double& x : U) {
    x = nd(rng);
  }
  for (double& x : V) {
    x = nd(rng);
  }

  for (int64_t n = 0; n < N; ++n) {
    double* row = A.data() + static_cast<size_t>(n) * static_cast<size_t>(S);
    for (int64_t s = 0; s < S; ++s) {
      double acc = 0.0;
      for (int k = 0; k < r; ++k) {
        acc += U[static_cast<size_t>(n) * static_cast<size_t>(r) + static_cast<size_t>(k)] *
               V[static_cast<size_t>(s) * static_cast<size_t>(r) + static_cast<size_t>(k)];
      }

      double eps = nd(rng);
      if (heavy_tail) {
        const double denom = std::max(0.25, std::abs(nd(rng)));
        eps = std::max(-10.0, std::min(10.0, eps / denom));
      }
      row[s] = acc + noise * eps;
    }
  }
}

void fill_delta_gamma(std::vector<double>& A, int64_t N, int64_t S, const radc::Config& cfg,
                      int64_t epoch, int rank, double* local_notional_out) {
  const int d = std::max(1, cfg.workload.delta_gamma.d_factors);
  const auto& dg = cfg.workload.delta_gamma;

  std::vector<double> notionals(static_cast<size_t>(N), 0.0);
  std::vector<double> delta(static_cast<size_t>(N) * static_cast<size_t>(d), 0.0);
  std::vector<double> gamma(static_cast<size_t>(N) * static_cast<size_t>(d), 0.0);
  std::vector<double> a_coef(static_cast<size_t>(N) * static_cast<size_t>(d), 0.0);
  std::vector<double> b_coef(static_cast<size_t>(N), 0.0);
  std::vector<double> x(static_cast<size_t>(S) * static_cast<size_t>(d), 0.0);
  std::vector<double> delta_m(static_cast<size_t>(d), 0.0);

  const uint64_t seed_epoch = splitmix64(cfg.run.seed ^ static_cast<uint64_t>(epoch));
  const uint64_t seed_rank = splitmix64(seed_epoch ^ static_cast<uint64_t>(rank));
  const uint64_t seed_scenarios =
      cfg.workload.common_scenarios_across_ranks ? seed_epoch : splitmix64(seed_epoch ^ static_cast<uint64_t>(rank));
  const uint64_t seed_delta_m =
      cfg.workload.common_delta_m_across_ranks
          ? splitmix64(seed_epoch ^ 0xD15EA5EULL)
          : splitmix64(seed_epoch ^ static_cast<uint64_t>(rank) ^ 0xD15EA5EULL);
  std::mt19937_64 rng(seed_rank);
  std::mt19937_64 rng_scenarios(seed_scenarios);
  std::mt19937_64 rng_delta(seed_delta_m);

  std::uniform_real_distribution<double> notional_dist(dg.notional_min, dg.notional_max);
  std::normal_distribution<double> nd(0.0, 1.0);

  for (int i = 0; i < d; ++i) {
    delta_m[static_cast<size_t>(i)] = dg.scenario_sigma * dg.shock_sigma * nd(rng_delta);
  }

  for (int64_t n = 0; n < N; ++n) {
    notionals[static_cast<size_t>(n)] = notional_dist(rng);
    b_coef[static_cast<size_t>(n)] = 0.1 * nd(rng);
    for (int i = 0; i < d; ++i) {
      delta[static_cast<size_t>(n) * static_cast<size_t>(d) + static_cast<size_t>(i)] =
          dg.delta_scale * nd(rng);
      gamma[static_cast<size_t>(n) * static_cast<size_t>(d) + static_cast<size_t>(i)] =
          dg.gamma_scale * nd(rng);
      a_coef[static_cast<size_t>(n) * static_cast<size_t>(d) + static_cast<size_t>(i)] = nd(rng);
    }
  }

  if (local_notional_out != nullptr) {
    double sum_notional = 0.0;
    for (double x : notionals) {
      sum_notional += x;
    }
    *local_notional_out = sum_notional;
  }

  for (int64_t s = 0; s < S; ++s) {
    for (int i = 0; i < d; ++i) {
      x[static_cast<size_t>(s) * static_cast<size_t>(d) + static_cast<size_t>(i)] =
          dg.scenario_sigma * dg.shock_sigma * nd(rng_scenarios);
    }
  }

  const double lambda = dg.nonlinear_lambda * (dg.shock_sigma / 1.0);

  for (int64_t n = 0; n < N; ++n) {
    const double notional = notionals[static_cast<size_t>(n)];
    const double* drow = delta.data() + static_cast<size_t>(n) * static_cast<size_t>(d);
    const double* grow = gamma.data() + static_cast<size_t>(n) * static_cast<size_t>(d);
    const double* arow = a_coef.data() + static_cast<size_t>(n) * static_cast<size_t>(d);

    double linear = 0.0;
    double quad = 0.0;
    std::vector<double> gdm(static_cast<size_t>(d), 0.0);
    for (int i = 0; i < d; ++i) {
      linear += drow[i] * delta_m[static_cast<size_t>(i)];
      gdm[static_cast<size_t>(i)] = grow[i] * delta_m[static_cast<size_t>(i)];
      quad += 0.5 * delta_m[static_cast<size_t>(i)] * delta_m[static_cast<size_t>(i)] * grow[i];
    }

    double* out_row = A.data() + static_cast<size_t>(n) * static_cast<size_t>(S);
    for (int64_t s = 0; s < S; ++s) {
      const double* xs = x.data() + static_cast<size_t>(s) * static_cast<size_t>(d);
      double cross = 0.0;
      double axpb = b_coef[static_cast<size_t>(n)];
      for (int i = 0; i < d; ++i) {
        cross += gdm[static_cast<size_t>(i)] * xs[i];
        axpb += arow[i] * xs[i];
      }
      const double phi = std::max(0.0, axpb);
      out_row[s] = notional * (linear + cross + quad + lambda * phi * phi);
    }
  }
}

void fill_pca_like(std::vector<double>& A, int64_t N, int64_t S, const radc::Config& cfg, int64_t epoch,
                   int rank) {
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
    double* row = A.data() + static_cast<size_t>(n) * static_cast<size_t>(S);
    for (int64_t s = 0; s < S; ++s) {
      double acc = 0.0;
      for (int k = 0; k < r; ++k) {
        acc += sigma[static_cast<size_t>(k)] *
               F[static_cast<size_t>(n) * static_cast<size_t>(r) + static_cast<size_t>(k)] *
               G[static_cast<size_t>(s) * static_cast<size_t>(r) + static_cast<size_t>(k)];
      }
      row[s] = acc + noise * nd(rng);
    }
  }
}

double fill_workload(std::vector<double>& A, const radc::Config& cfg, int64_t epoch, int rank) {
  if (cfg.workload.kind == "synth_lowrank") {
    fill_synth_lowrank(A, cfg.buffer.N, cfg.buffer.S, cfg, epoch, rank);
    return std::numeric_limits<double>::quiet_NaN();
  }
  if (cfg.workload.kind == "delta_gamma") {
    double local_notional = 0.0;
    fill_delta_gamma(A, cfg.buffer.N, cfg.buffer.S, cfg, epoch, rank, &local_notional);
    return local_notional;
  }
  if (cfg.workload.kind == "pca_like") {
    fill_pca_like(A, cfg.buffer.N, cfg.buffer.S, cfg, epoch, rank);
    return std::numeric_limits<double>::quiet_NaN();
  }
  throw std::runtime_error("Unknown workload.kind: " + cfg.workload.kind);
}

bool is_float32(const radc::Config& cfg) { return cfg.buffer.dtype_in == "float32"; }

int run(const radc::Config& cfg) {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  const int64_t count64 = cfg.buffer.N * cfg.buffer.S;
  if (count64 <= 0 || count64 > static_cast<int64_t>(std::numeric_limits<int>::max())) {
    throw std::runtime_error("Invalid N*S count for MPI_Allreduce driver");
  }
  const int count = static_cast<int>(count64);

  std::vector<double> local_d(static_cast<size_t>(count), 0.0);
  std::vector<double> out_d(static_cast<size_t>(count), 0.0);
  std::vector<float> local_f(static_cast<size_t>(count), 0.0f);
  std::vector<float> out_f(static_cast<size_t>(count), 0.0f);

  using SetEpochFn = void (*)(int64_t);
  SetEpochFn set_epoch = reinterpret_cast<SetEpochFn>(dlsym(RTLD_DEFAULT, "radc_set_epoch"));
  using SetNotionalFn = void (*)(double);
  SetNotionalFn set_notional =
      reinterpret_cast<SetNotionalFn>(dlsym(RTLD_DEFAULT, "radc_set_total_notional"));

  for (int64_t epoch = 0; epoch < cfg.run.epochs; ++epoch) {
    if (cfg.mpi.barrier_before_epochs) {
      MPI_Barrier(MPI_COMM_WORLD);
    }

    if (set_epoch != nullptr) {
      set_epoch(epoch);
    }

    const double local_notional = fill_workload(local_d, cfg, epoch, rank);
    double total_notional = local_notional;
    if (std::isfinite(local_notional) && local_notional > 0.0) {
      if (MPI_Allreduce(MPI_IN_PLACE, &total_notional, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD) != MPI_SUCCESS) {
        throw std::runtime_error("MPI_Allreduce failed for total_notional");
      }
    } else {
      total_notional = std::numeric_limits<double>::quiet_NaN();
    }
    if (set_notional != nullptr) {
      set_notional(total_notional);
    }

    if (is_float32(cfg)) {
      for (int i = 0; i < count; ++i) {
        local_f[static_cast<size_t>(i)] = static_cast<float>(local_d[static_cast<size_t>(i)]);
      }
      if (MPI_Allreduce(local_f.data(), out_f.data(), count, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD) != MPI_SUCCESS) {
        throw std::runtime_error("MPI_Allreduce failed (float32)");
      }
    } else {
      if (MPI_Allreduce(local_d.data(), out_d.data(), count, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD) != MPI_SUCCESS) {
        throw std::runtime_error("MPI_Allreduce failed (float64)");
      }
    }

    if (cfg.mpi.barrier_after_epochs) {
      MPI_Barrier(MPI_COMM_WORLD);
    }
  }

  if (rank == 0) {
    double checksum = 0.0;
    const size_t upto = std::min<size_t>(static_cast<size_t>(count), 64);
    if (is_float32(cfg)) {
      for (size_t i = 0; i < upto; ++i) {
        checksum += static_cast<double>(out_f[i]);
      }
    } else {
      for (size_t i = 0; i < upto; ++i) {
        checksum += out_d[i];
      }
    }
    std::cout << "bench_stage3_intercept_driver checksum=" << checksum << std::endl;
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
