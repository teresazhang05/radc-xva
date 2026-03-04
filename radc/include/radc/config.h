#pragma once

#include <cstdint>
#include <string>

namespace radc {

struct RunSection {
  std::string run_id;
  uint64_t seed;
  std::string output_dir;
  int epochs;
  int warmup_epochs;
  bool overwrite;
};

struct BufferSection {
  int64_t N;
  int64_t S;
  std::string layout;
  std::string dtype_in;
};

struct MpiSection {
  std::string comm;
  bool barrier_before_epochs;
  bool barrier_after_epochs;
};

struct RadcSection {
  bool enabled;
  std::string intercept;
  std::string only_if_op;
  int r_min;
  int r_max;
  int oversample_p;
  int power_iters;
  std::string omega_dist;
  uint64_t omega_seed;
  double energy_capture;
  std::string factor_dtype;
};

struct SketchSection {
  std::string kind;
  int kG;
  int kS;
  int num_sketches;
  uint64_t hash_seed_g0;
  uint64_t sign_seed_g0;
  uint64_t hash_seed_s0;
  uint64_t sign_seed_s0;
  uint64_t hash_seed_g1;
  uint64_t sign_seed_g1;
  uint64_t hash_seed_s1;
  uint64_t sign_seed_s1;
};

struct RiskSection {
  std::string kind;
  int netting_sets_G;
  uint64_t netting_seed;
  std::string collateral_kind;
  double collateral_threshold;
  double a_scalar;
  std::string scenario_weights;
  double notional_total;
};

struct SafetySection {
  double xva_epsilon_bps;
  double accept_margin;
  double jl_epsilon;
  double jl_delta;
  bool always_accept;
  bool verify_enabled;
};

struct CompressionSection {
  std::string double_mode;  // native64 | downcast32 | passthrough
};

struct DetectSection {
  bool enabled;
  double delta_m_sigma_threshold;
  bool force_fallback_if_r_hits_rmax;
};

struct FallbackSection {
  bool enabled;
  std::string mode;
};

struct ProfilingSection {
  bool timing_breakdown;
  bool perf_stat_enable;
  bool rapl_enable;
  double net_emulation_bandwidth_gbps;
  double net_emulation_base_latency_ms;
};

struct LoggingSection {
  std::string per_rank_csv;
  std::string per_rank_jsonl;
  int flush_every_epochs;
  int shadow_exact_every;
};

struct WorkloadSection {
  bool common_scenarios_across_ranks;
  bool common_delta_m_across_ranks;

  struct SynthLowrank {
    int true_rank;
    double noise_scale;
    bool heavy_tail;
  };

  struct DeltaGamma {
    int d_factors;
    double notional_min;
    double notional_max;
    double delta_scale;
    double gamma_scale;
    double scenario_sigma;
    double shock_sigma;
    double nonlinear_lambda;
  };

  struct PcaLike {
    int true_rank;
    double noise_scale;
  };

  std::string kind;
  SynthLowrank synth_lowrank;
  DeltaGamma delta_gamma;
  PcaLike pca_like;
};

struct Config {
  std::string source_path;
  RunSection run;
  MpiSection mpi;
  BufferSection buffer;
  RadcSection radc;
  SketchSection sketch;
  RiskSection risk;
  SafetySection safety;
  CompressionSection compression;
  DetectSection detect;
  FallbackSection fallback;
  ProfilingSection profiling;
  LoggingSection logging;
  WorkloadSection workload;
};

Config default_config();

std::string resolve_config_path(const std::string& cli_config_path = "");

Config load_config_from_path(const std::string& path);

Config load_config_from_env_or_default(const std::string& cli_config_path = "");

}  // namespace radc
