#include "radc/logging.h"

#include <cmath>
#include <filesystem>
#include <iomanip>
#include <sstream>
#include <stdexcept>

#include "radc/timer.h"

namespace radc {

namespace {

std::string fmt_double(double v) {
  if (std::isnan(v)) {
    return "NaN";
  }
  if (!std::isfinite(v)) {
    return (v > 0.0) ? "Inf" : "-Inf";
  }
  std::ostringstream os;
  os << std::setprecision(17) << v;
  return os.str();
}

}  // namespace

PerRankLogger::PerRankLogger(const Config& cfg, int rank)
    : cfg_(cfg), rank_(rank), writes_since_flush_(0) {
  const std::filesystem::path out_dir(cfg_.run.output_dir);
  std::filesystem::create_directories(out_dir);

  const std::filesystem::path metrics_path = out_dir / replace_rank_token(cfg_.logging.per_rank_csv, rank_);
  const std::filesystem::path events_path = out_dir / replace_rank_token(cfg_.logging.per_rank_jsonl, rank_);

  const bool metrics_exists = std::filesystem::exists(metrics_path);
  const bool events_exists = std::filesystem::exists(events_path);
  const bool metrics_nonempty = metrics_exists && std::filesystem::file_size(metrics_path) > 0;
  const bool events_nonempty = events_exists && std::filesystem::file_size(events_path) > 0;

  const auto mode = cfg_.run.overwrite ? std::ios::trunc : std::ios::app;
  metrics_.open(metrics_path, std::ios::out | mode);
  events_.open(events_path, std::ios::out | mode);

  if (!metrics_.is_open() || !events_.is_open()) {
    throw std::runtime_error("Failed to open metrics/events log files for rank " + std::to_string(rank_));
  }

  if (cfg_.run.overwrite || !metrics_nonempty) {
    write_header_if_needed();
  }
  if (!cfg_.run.overwrite && events_nonempty) {
    events_ << '\n';
  }
}

PerRankLogger::~PerRankLogger() {
  std::lock_guard<std::mutex> lock(mu_);
  if (metrics_.is_open()) {
    metrics_.flush();
    metrics_.close();
  }
  if (events_.is_open()) {
    events_.flush();
    events_.close();
  }
}

void PerRankLogger::write_header_if_needed() {
  metrics_ << "run_id"
           << ",rank"
           << ",world_size"
           << ",epoch"
           << ",mode"
           << ",N"
           << ",S"
           << ",dtype_in"
           << ",dtype_internal"
           << ",downcast_used"
           << ",layout"
           << ",factor_dtype"
           << ",l"
           << ",l_used"
           << ",r_used"
           << ",r_max"
           << ",oversample_p"
           << ",power_iters"
           << ",energy_at_r"
           << ",frob_norm_exact"
           << ",rho_hat_sketch_0"
           << ",rho_hat_sketch_1"
           << ",rho_hat_sketch_max"
           << ",rho_hat_sketch"
           << ",rho_max"
           << ",eps_dollars"
           << ",accept_threshold"
           << ",kG"
           << ",kS"
           << ",num_sketches"
           << ",accepted"
           << ",fallback_triggered"
           << ",bytes_exact_payload"
           << ",bytes_comp_attempt_payload"
           << ",bytes_exact_fallback_payload"
           << ",bytes_total_payload"
           << ",bytes_comp_payload"
           << ",bytes_effective_payload"
           << ",compression_ratio_payload"
           << ",t_compute_Y_ms"
           << ",t_allreduce_Y_ms"
           << ",t_qr_ms"
           << ",t_compute_B_ms"
           << ",t_allreduce_B_ms"
           << ",t_small_svd_ms"
           << ",t_reconstruct_ms"
           << ",t_sketch_local_ms"
           << ",t_allreduce_sketch_ms"
           << ",t_verify_ms"
           << ",t_exact_fallback_ms"
           << ",t_epoch_total_ms"
           << ",xva_true"
           << ",xva_approx"
           << ",xva_err_abs"
           << ",xva_err_bps"
           << ",L_xva"
           << ",total_notional"
           << ",xva_epsilon_bps"
           << ",accept_margin"
           << ",perf_cycles"
           << ",perf_instructions"
           << ",perf_cache_misses"
           << ",perf_llc_load_misses"
           << ",energy_pkg_joules"
           << ",warmup_epochs"
           << ",shock_sigma"
           << ",is_shadow_epoch"
           << "\n";
}

void PerRankLogger::log_metric(const MetricsRow& row) {
  std::lock_guard<std::mutex> lock(mu_);
  metrics_ << csv_escape(row.run_id) << ',' << row.rank << ',' << row.world_size << ',' << row.epoch
           << ',' << row.mode << ',' << row.N << ',' << row.S << ',' << row.dtype_in << ','
           << row.dtype_internal << ',' << row.downcast_used << ',' << row.layout << ',' << row.factor_dtype << ',' << row.l
           << ',' << row.l_used << ',' << row.r_used << ',' << row.r_max << ',' << row.oversample_p << ','
           << row.power_iters << ',' << fmt_double(row.energy_at_r) << ','
           << fmt_double(row.frob_norm_exact) << ',' << fmt_double(row.rho_hat_sketch_0) << ','
           << fmt_double(row.rho_hat_sketch_1) << ',' << fmt_double(row.rho_hat_sketch_max) << ','
           << fmt_double(row.rho_hat_sketch) << ',' << fmt_double(row.rho_max) << ','
           << fmt_double(row.eps_dollars) << ',' << fmt_double(row.accept_threshold) << ','
           << row.kG << ',' << row.kS << ',' << row.num_sketches << ','
           << row.accepted << ',' << row.fallback_triggered << ','
           << row.bytes_exact_payload << ',' << row.bytes_comp_attempt_payload << ','
           << row.bytes_exact_fallback_payload << ',' << row.bytes_total_payload << ','
           << row.bytes_comp_payload << ','
           << row.bytes_effective_payload << ','
           << fmt_double(row.compression_ratio_payload) << ',' << fmt_double(row.t_compute_Y_ms) << ','
           << fmt_double(row.t_allreduce_Y_ms) << ',' << fmt_double(row.t_qr_ms) << ','
           << fmt_double(row.t_compute_B_ms) << ',' << fmt_double(row.t_allreduce_B_ms) << ','
           << fmt_double(row.t_small_svd_ms) << ',' << fmt_double(row.t_reconstruct_ms) << ','
           << fmt_double(row.t_sketch_local_ms) << ',' << fmt_double(row.t_allreduce_sketch_ms) << ','
           << fmt_double(row.t_verify_ms) << ',' << fmt_double(row.t_exact_fallback_ms) << ','
           << fmt_double(row.t_epoch_total_ms) << ',' << fmt_double(row.xva_true) << ','
           << fmt_double(row.xva_approx) << ',' << fmt_double(row.xva_err_abs) << ','
           << fmt_double(row.xva_err_bps) << ',' << fmt_double(row.L_xva) << ','
           << fmt_double(row.total_notional) << ',' << fmt_double(row.xva_epsilon_bps) << ','
           << fmt_double(row.accept_margin) << ','
           << fmt_double(row.perf_cycles) << ',' << fmt_double(row.perf_instructions) << ','
           << fmt_double(row.perf_cache_misses) << ',' << fmt_double(row.perf_llc_load_misses) << ','
           << fmt_double(row.energy_pkg_joules) << ',' << row.warmup_epochs << ','
           << fmt_double(row.shock_sigma) << ',' << row.is_shadow_epoch << '\n';

  ++writes_since_flush_;
  if (cfg_.logging.flush_every_epochs <= 1 || writes_since_flush_ >= cfg_.logging.flush_every_epochs) {
    metrics_.flush();
    writes_since_flush_ = 0;
  }
}

void PerRankLogger::log_event(int64_t epoch, const std::string& event, const std::string& detail) {
  std::lock_guard<std::mutex> lock(mu_);
  events_ << "{\"ts_unix_ns\":" << now_unix_ns() << ",\"rank\":" << rank_ << ",\"epoch\":" << epoch
          << ",\"event\":\"" << json_escape(event) << "\",\"detail\":\""
          << json_escape(detail) << "\"}" << '\n';

  if (cfg_.logging.flush_every_epochs <= 1) {
    events_.flush();
  }
}

std::string PerRankLogger::replace_rank_token(const std::string& pattern, int rank) {
  std::string out = pattern;
  const std::string token = "{rank}";
  const size_t pos = out.find(token);
  if (pos != std::string::npos) {
    out.replace(pos, token.size(), std::to_string(rank));
  }
  return out;
}

std::string PerRankLogger::csv_escape(const std::string& v) {
  if (v.find(',') == std::string::npos && v.find('"') == std::string::npos &&
      v.find('\n') == std::string::npos) {
    return v;
  }
  std::string out = "\"";
  for (char c : v) {
    if (c == '"') {
      out += '"';
    }
    out += c;
  }
  out += '"';
  return out;
}

std::string PerRankLogger::json_escape(const std::string& v) {
  std::string out;
  out.reserve(v.size());
  for (char c : v) {
    switch (c) {
      case '\\':
        out += "\\\\";
        break;
      case '"':
        out += "\\\"";
        break;
      case '\n':
        out += "\\n";
        break;
      case '\r':
        out += "\\r";
        break;
      case '\t':
        out += "\\t";
        break;
      default:
        out += c;
        break;
    }
  }
  return out;
}

}  // namespace radc
