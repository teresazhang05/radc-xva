#pragma once

#include <cstdint>
#include <fstream>
#include <mutex>
#include <string>

#include "radc/config.h"

namespace radc {

struct MetricsRow {
  std::string run_id;
  int rank;
  int world_size;
  int64_t epoch;
  std::string mode;
  int64_t N;
  int64_t S;
  std::string dtype_in;
  std::string dtype_internal;
  int downcast_used;
  std::string layout;
  std::string factor_dtype;
  int l;
  int l_used;
  int r_used;
  int r_max;
  int oversample_p;
  int power_iters;
  double energy_at_r;
  double frob_norm_exact;
  double rho_hat_sketch_0;
  double rho_hat_sketch_1;
  double rho_hat_sketch_max;
  double rho_hat_sketch;
  double rho_max;
  double eps_dollars;
  double accept_threshold;
  int kG;
  int kS;
  int num_sketches;
  int accepted;
  int fallback_triggered;
  int64_t bytes_exact_payload;
  int64_t bytes_comp_attempt_payload;
  int64_t bytes_exact_fallback_payload;
  int64_t bytes_total_payload;
  int64_t bytes_comp_payload;
  int64_t bytes_effective_payload;
  double compression_ratio_payload;
  double t_compute_Y_ms;
  double t_allreduce_Y_ms;
  double t_qr_ms;
  double t_compute_B_ms;
  double t_allreduce_B_ms;
  double t_small_svd_ms;
  double t_reconstruct_ms;
  double t_sketch_local_ms;
  double t_allreduce_sketch_ms;
  double t_verify_ms;
  double t_exact_fallback_ms;
  double t_epoch_total_ms;
  double xva_true;
  double xva_approx;
  double xva_err_abs;
  double xva_err_bps;
  double L_xva;
  double total_notional;
  double xva_epsilon_bps;
  double accept_margin;
  double perf_cycles;
  double perf_instructions;
  double perf_cache_misses;
  double perf_llc_load_misses;
  double energy_pkg_joules;
  int warmup_epochs;
  double shock_sigma;
  int is_shadow_epoch;
};

class PerRankLogger {
 public:
  PerRankLogger(const Config& cfg, int rank);
  ~PerRankLogger();

  PerRankLogger(const PerRankLogger&) = delete;
  PerRankLogger& operator=(const PerRankLogger&) = delete;

  void log_metric(const MetricsRow& row);
  void log_event(int64_t epoch, const std::string& event, const std::string& detail);

 private:
  void write_header_if_needed();
  static std::string replace_rank_token(const std::string& pattern, int rank);
  static std::string csv_escape(const std::string& v);
  static std::string json_escape(const std::string& v);

  Config cfg_;
  int rank_;
  std::ofstream metrics_;
  std::ofstream events_;
  std::mutex mu_;
  int writes_since_flush_;
};

}  // namespace radc
