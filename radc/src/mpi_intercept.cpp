#include "radc/mpi_intercept.h"

#include <algorithm>
#include <atomic>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <limits>
#include <memory>
#include <mutex>
#include <string>
#include <stdexcept>
#include <thread>
#include <vector>

#include <mpi.h>

#include "radc/compressed_protocol.h"
#include "radc/config.h"
#include "radc/logging.h"
#include "radc/matrix_view.h"
#include "radc/risk_xva.h"
#include "radc/timer.h"

namespace radc {
namespace {

std::atomic<int64_t> g_current_epoch{-1};
std::atomic<double> g_current_notional_total{std::numeric_limits<double>::quiet_NaN()};

struct InterceptContext {
  Config cfg;
  int rank = 0;
  int world_size = 1;
  std::unique_ptr<PerRankLogger> logger;
  std::atomic<int64_t> call_counter{0};
  ProtocolRuntime protocol_runtime{};
  bool have_protocol_runtime = false;
  RiskState risk_state{};
  bool have_risk = false;
};

bool debug_enabled() {
  const char* v = std::getenv("RADC_DEBUG_WRAPPER");
  return v != nullptr && std::string(v) == "1";
}

bool env_true(const char* name) {
  const char* raw = std::getenv(name);
  if (raw == nullptr) {
    return false;
  }
  std::string v(raw);
  for (char& c : v) {
    c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
  }
  return v == "1" || v == "true" || v == "yes" || v == "on";
}

void apply_env_overrides(Config& cfg) {
  if (env_true("RADC_FORCE_EXACT")) {
    cfg.radc.enabled = false;
  }

  const char* suffix = std::getenv("RADC_RUN_SUFFIX");
  if (suffix != nullptr && std::string(suffix).size() > 0) {
    cfg.run.run_id += std::string("_") + suffix;
    cfg.run.output_dir += std::string("_") + suffix;
  }
}

InterceptContext& context_for_comm(MPI_Comm comm) {
  static std::once_flag once;
  static std::unique_ptr<InterceptContext> ctx;

  std::call_once(once, [&]() {
    auto built = std::make_unique<InterceptContext>();

    int rank = 0;
    int world = 1;
    PMPI_Comm_rank(comm, &rank);
    PMPI_Comm_size(comm, &world);

    built->rank = rank;
    built->world_size = world;

    try {
      built->cfg = load_config_from_env_or_default("");
      apply_env_overrides(built->cfg);
    } catch (...) {
      built->cfg = default_config();
      built->cfg.source_path = "<default_after_load_failure>";
      built->cfg.run.output_dir = std::string("results/") + built->cfg.run.run_id;
      built->cfg.run.overwrite = true;
    }

    try {
      built->logger = std::make_unique<PerRankLogger>(built->cfg, rank);
      built->logger->log_event(-1, "interceptor_initialized",
                               std::string("source_config=") + built->cfg.source_path);
    } catch (const std::exception& e) {
      if (debug_enabled()) {
        std::fprintf(stderr, "[radc] logger init failed on rank %d: %s\n", rank, e.what());
      }
      built->logger.reset();
    } catch (...) {
      if (debug_enabled()) {
        std::fprintf(stderr, "[radc] logger init failed on rank %d: unknown error\n", rank);
      }
      built->logger.reset();
    }

    try {
      RiskConfig rcfg{};
      rcfg.G = built->cfg.risk.netting_sets_G;
      rcfg.netting_seed = built->cfg.risk.netting_seed;
      rcfg.collateral_threshold = static_cast<float>(built->cfg.risk.collateral_threshold);
      rcfg.a_scalar = static_cast<float>(built->cfg.risk.a_scalar);
      rcfg.scenario_weights_uniform = (built->cfg.risk.scenario_weights == "uniform");
      rcfg.notional_total = built->cfg.risk.notional_total;
      built->risk_state =
          build_risk_state(static_cast<int>(built->cfg.buffer.N), static_cast<int>(built->cfg.buffer.S), rcfg);
      built->have_risk = true;
    } catch (...) {
      built->have_risk = false;
    }

    ctx = std::move(built);
  });

  return *ctx;
}

int datatype_size_bytes(MPI_Datatype dtype) {
  if (dtype == MPI_FLOAT) {
    return 4;
  }
  if (dtype == MPI_DOUBLE) {
    return 8;
  }
  return 0;
}

double safe_nan() { return std::numeric_limits<double>::quiet_NaN(); }

double net_emulation_delay_ms(int64_t bytes_effective, const Config& cfg) {
  if (bytes_effective <= 0) {
    return 0.0;
  }
  const double bw_gbps = cfg.profiling.net_emulation_bandwidth_gbps;
  if (!std::isfinite(bw_gbps) || bw_gbps <= 0.0) {
    return 0.0;
  }
  const double base_ms = (std::isfinite(cfg.profiling.net_emulation_base_latency_ms) &&
                          cfg.profiling.net_emulation_base_latency_ms > 0.0)
                             ? cfg.profiling.net_emulation_base_latency_ms
                             : 0.0;
  const double bits = static_cast<double>(bytes_effective) * 8.0;
  const double tx_ms = 1000.0 * bits / (bw_gbps * 1.0e9);
  return base_ms + tx_ms;
}

void maybe_emulate_network(int64_t bytes_effective, const Config& cfg) {
  const double delay_ms = net_emulation_delay_ms(bytes_effective, cfg);
  if (!std::isfinite(delay_ms) || delay_ms <= 0.0) {
    return;
  }
  const auto us = static_cast<int64_t>(std::llround(delay_ms * 1000.0));
  if (us > 0) {
    std::this_thread::sleep_for(std::chrono::microseconds(us));
  }
}

int64_t resolve_epoch(InterceptContext& ctx) {
  const int64_t e = g_current_epoch.load(std::memory_order_relaxed);
  if (e >= 0) {
    return e;
  }
  return ctx.call_counter.fetch_add(1, std::memory_order_relaxed);
}

std::string dtype_name(MPI_Datatype dtype) {
  if (dtype == MPI_FLOAT) {
    return "float32";
  }
  if (dtype == MPI_DOUBLE) {
    return "float64";
  }
  return "unknown";
}

bool dtype_matches_config(const Config& cfg, MPI_Datatype dtype) {
  if (cfg.buffer.dtype_in == "float32") {
    return dtype == MPI_FLOAT;
  }
  if (cfg.buffer.dtype_in == "float64") {
    return dtype == MPI_DOUBLE;
  }
  return false;
}

int64_t expected_matrix_count(const Config& cfg) {
  if (cfg.buffer.N <= 0 || cfg.buffer.S <= 0) {
    return 0;
  }
  if (cfg.buffer.N > (std::numeric_limits<int64_t>::max() / cfg.buffer.S)) {
    return 0;
  }
  return cfg.buffer.N * cfg.buffer.S;
}

bool eligible_for_protocol(const InterceptContext& ctx, const void* sendbuf, void* recvbuf, int count,
                           MPI_Datatype datatype, MPI_Op op) {
  if (ctx.cfg.radc.intercept != "MPI_Allreduce") {
    return false;
  }
  if (ctx.cfg.radc.only_if_op == "MPI_SUM" && op != MPI_SUM) {
    return false;
  }
  if (sendbuf == MPI_IN_PLACE) {
    return false;
  }
  if (sendbuf == nullptr || recvbuf == nullptr) {
    return false;
  }
  if (count <= 0) {
    return false;
  }
  if (ctx.cfg.buffer.layout != "row_major") {
    return false;
  }
  if (datatype != MPI_FLOAT && datatype != MPI_DOUBLE) {
    return false;
  }
  if (!dtype_matches_config(ctx.cfg, datatype)) {
    return false;
  }

  const int64_t expected = expected_matrix_count(ctx.cfg);
  if (expected <= 0) {
    return false;
  }
  return static_cast<int64_t>(count) == expected;
}

void ensure_protocol_runtime(InterceptContext& ctx) {
  if (ctx.have_protocol_runtime) {
    return;
  }
  init_protocol_runtime(ctx.cfg, ctx.cfg.buffer.N, ctx.cfg.buffer.S, ctx.protocol_runtime);
  ctx.have_protocol_runtime = true;
}

void copy_sendbuf_to_f32(const void* sendbuf, int count, MPI_Datatype datatype, std::vector<float>& out) {
  out.resize(static_cast<size_t>(count));
  if (datatype == MPI_DOUBLE) {
    const double* src = static_cast<const double*>(sendbuf);
    for (int i = 0; i < count; ++i) {
      out[static_cast<size_t>(i)] = static_cast<float>(src[i]);
    }
    return;
  }
  if (datatype == MPI_FLOAT) {
    const float* src = static_cast<const float*>(sendbuf);
    std::copy(src, src + count, out.begin());
    return;
  }
  throw std::runtime_error("copy_sendbuf_to_f32: unsupported datatype");
}

void copy_f32_to_recvbuf(const std::vector<float>& in, int count, MPI_Datatype datatype, void* recvbuf) {
  if (static_cast<int>(in.size()) != count) {
    throw std::runtime_error("copy_f32_to_recvbuf: size mismatch");
  }
  if (datatype == MPI_DOUBLE) {
    double* dst = static_cast<double*>(recvbuf);
    for (int i = 0; i < count; ++i) {
      dst[i] = static_cast<double>(in[static_cast<size_t>(i)]);
    }
    return;
  }
  if (datatype == MPI_FLOAT) {
    float* dst = static_cast<float*>(recvbuf);
    std::copy(in.begin(), in.end(), dst);
    return;
  }
  throw std::runtime_error("copy_f32_to_recvbuf: unsupported datatype");
}

void copy_sendbuf_to_f64(const void* sendbuf, int count, MPI_Datatype datatype, std::vector<double>& out) {
  out.resize(static_cast<size_t>(count));
  if (datatype == MPI_DOUBLE) {
    const double* src = static_cast<const double*>(sendbuf);
    std::copy(src, src + count, out.begin());
    return;
  }
  if (datatype == MPI_FLOAT) {
    const float* src = static_cast<const float*>(sendbuf);
    for (int i = 0; i < count; ++i) {
      out[static_cast<size_t>(i)] = static_cast<double>(src[i]);
    }
    return;
  }
  throw std::runtime_error("copy_sendbuf_to_f64: unsupported datatype");
}

void copy_f64_to_recvbuf(const std::vector<double>& in, int count, MPI_Datatype datatype, void* recvbuf) {
  if (static_cast<int>(in.size()) != count) {
    throw std::runtime_error("copy_f64_to_recvbuf: size mismatch");
  }
  if (datatype == MPI_DOUBLE) {
    double* dst = static_cast<double*>(recvbuf);
    std::copy(in.begin(), in.end(), dst);
    return;
  }
  if (datatype == MPI_FLOAT) {
    float* dst = static_cast<float*>(recvbuf);
    for (int i = 0; i < count; ++i) {
      dst[i] = static_cast<float>(in[static_cast<size_t>(i)]);
    }
    return;
  }
  throw std::runtime_error("copy_f64_to_recvbuf: unsupported datatype");
}

MetricsRow make_exact_row(const InterceptContext& ctx, int64_t epoch, MPI_Datatype datatype, int count,
                          bool is_shadow_epoch, double t_epoch_total_ms,
                          double total_notional_override) {
  MetricsRow row{};
  row.run_id = ctx.cfg.run.run_id;
  row.rank = ctx.rank;
  row.world_size = ctx.world_size;
  row.epoch = epoch;
  row.mode = "exact";
  row.N = ctx.cfg.buffer.N;
  row.S = ctx.cfg.buffer.S;
  row.dtype_in = dtype_name(datatype);
  row.dtype_internal = row.dtype_in;
  row.downcast_used = 0;
  row.layout = ctx.cfg.buffer.layout;
  row.factor_dtype = ctx.cfg.radc.factor_dtype;
  row.l = std::max(1, ctx.cfg.radc.r_max + ctx.cfg.radc.oversample_p);
  row.l_used = row.l;
  row.r_used = 0;
  row.r_max = ctx.cfg.radc.r_max;
  row.oversample_p = ctx.cfg.radc.oversample_p;
  row.power_iters = ctx.cfg.radc.power_iters;
  row.energy_at_r = safe_nan();
  row.frob_norm_exact = safe_nan();
  row.rho_hat_sketch_0 = safe_nan();
  row.rho_hat_sketch_1 = safe_nan();
  row.rho_hat_sketch_max = safe_nan();
  row.rho_hat_sketch = safe_nan();
  const bool use_override_notional =
      std::isfinite(total_notional_override) && total_notional_override > 0.0;
  double effective_notional = use_override_notional ? total_notional_override : ctx.cfg.risk.notional_total;
  if (!(std::isfinite(effective_notional) && effective_notional > 0.0)) {
    effective_notional = 1.0;
  }
  row.rho_max = safe_nan();
  row.eps_dollars = safe_nan();
  row.accept_threshold = safe_nan();
  row.kG = ctx.cfg.sketch.kG;
  row.kS = ctx.cfg.sketch.kS;
  row.num_sketches = std::max(1, std::min(2, ctx.cfg.sketch.num_sketches));
  if (ctx.have_risk) {
    std::vector<double> a_g(static_cast<size_t>(ctx.risk_state.G), 1.0);
    std::vector<double> w_s(static_cast<size_t>(ctx.risk_state.S), 0.0);
    std::vector<double> c_g(static_cast<size_t>(ctx.risk_state.G), 0.0);
    for (int g = 0; g < ctx.risk_state.G; ++g) {
      a_g[static_cast<size_t>(g)] = static_cast<double>(ctx.risk_state.a_g[static_cast<size_t>(g)]);
      c_g[static_cast<size_t>(g)] = static_cast<double>(ctx.risk_state.H_g[static_cast<size_t>(g)]);
    }
    for (int s = 0; s < ctx.risk_state.S; ++s) {
      w_s[static_cast<size_t>(s)] = static_cast<double>(ctx.risk_state.w_s[static_cast<size_t>(s)]);
    }
    const SafetyState st = compute_safety_state(
        ctx.risk_state.G, ctx.risk_state.S, a_g, w_s, c_g, effective_notional,
        SafetyParams{ctx.cfg.safety.xva_epsilon_bps, ctx.cfg.safety.accept_margin, ctx.cfg.safety.jl_epsilon,
                     ctx.cfg.safety.jl_delta});
    row.rho_max = st.rho_max;
    row.eps_dollars = st.eps_dollars;
    row.accept_threshold = st.accept_threshold;
    row.L_xva = st.L;
  } else {
    row.L_xva = safe_nan();
  }
  row.accepted = 0;
  row.fallback_triggered = 0;

  const int dtype_bytes = datatype_size_bytes(datatype);
  row.bytes_exact_payload =
      (dtype_bytes > 0 && count > 0) ? static_cast<int64_t>(count) * static_cast<int64_t>(dtype_bytes) : 0;

  const bool run_verify = ctx.cfg.safety.verify_enabled && !ctx.cfg.safety.always_accept;
  row.bytes_comp_attempt_payload =
      (ctx.cfg.buffer.N * static_cast<int64_t>(row.l) + static_cast<int64_t>(row.l) * ctx.cfg.buffer.S) *
          static_cast<int64_t>(dtype_bytes > 0 ? dtype_bytes : sizeof(float)) +
      (run_verify ? static_cast<int64_t>(row.num_sketches) * static_cast<int64_t>(row.kG) *
                        static_cast<int64_t>(row.kS) *
                        static_cast<int64_t>(dtype_bytes > 0 ? dtype_bytes : sizeof(float))
                  : 0);
  row.bytes_exact_fallback_payload = 0;
  row.bytes_total_payload = row.bytes_exact_payload;
  row.bytes_comp_payload = row.bytes_comp_attempt_payload;
  row.bytes_effective_payload = row.bytes_total_payload;
  row.compression_ratio_payload =
      (row.bytes_total_payload > 0) ? static_cast<double>(row.bytes_exact_payload) / row.bytes_total_payload
                                    : safe_nan();

  row.t_compute_Y_ms = safe_nan();
  row.t_allreduce_Y_ms = safe_nan();
  row.t_qr_ms = safe_nan();
  row.t_compute_B_ms = safe_nan();
  row.t_allreduce_B_ms = safe_nan();
  row.t_small_svd_ms = safe_nan();
  row.t_reconstruct_ms = safe_nan();
  row.t_sketch_local_ms = safe_nan();
  row.t_allreduce_sketch_ms = safe_nan();
  row.t_verify_ms = safe_nan();
  row.t_exact_fallback_ms = safe_nan();
  row.t_epoch_total_ms = t_epoch_total_ms;
  row.xva_true = safe_nan();
  row.xva_approx = safe_nan();
  row.xva_err_abs = safe_nan();
  row.xva_err_bps = safe_nan();
  row.total_notional = effective_notional;
  row.xva_epsilon_bps = ctx.cfg.safety.xva_epsilon_bps;
  row.accept_margin = ctx.cfg.safety.accept_margin;
  row.perf_cycles = safe_nan();
  row.perf_instructions = safe_nan();
  row.perf_cache_misses = safe_nan();
  row.perf_llc_load_misses = safe_nan();
  row.energy_pkg_joules = safe_nan();
  row.warmup_epochs = ctx.cfg.run.warmup_epochs;
  row.shock_sigma = ctx.cfg.workload.delta_gamma.shock_sigma;
  row.is_shadow_epoch = is_shadow_epoch ? 1 : 0;
  return row;
}

MetricsRow make_protocol_row(const InterceptContext& ctx, int64_t epoch, MPI_Datatype datatype,
                             const ProtocolMetrics& pm, double t_epoch_total_ms) {
  MetricsRow row{};
  row.run_id = ctx.cfg.run.run_id;
  row.rank = ctx.rank;
  row.world_size = ctx.world_size;
  row.epoch = epoch;
  row.mode = pm.mode;
  row.N = ctx.cfg.buffer.N;
  row.S = ctx.cfg.buffer.S;
  row.dtype_in = dtype_name(datatype);
  row.dtype_internal = pm.dtype_internal;
  row.downcast_used = pm.downcast_used;
  row.layout = ctx.cfg.buffer.layout;
  row.factor_dtype = pm.factor_dtype;
  row.l = pm.l;
  row.l_used = pm.l_used;
  row.r_used = pm.r_used;
  row.r_max = ctx.cfg.radc.r_max;
  row.oversample_p = ctx.cfg.radc.oversample_p;
  row.power_iters = ctx.cfg.radc.power_iters;
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
  row.t_epoch_total_ms = t_epoch_total_ms;
  row.xva_true = pm.xva_true;
  row.xva_approx = pm.xva_approx;
  row.xva_err_abs = pm.xva_err_abs;
  row.xva_err_bps = pm.xva_err_bps;
  row.L_xva = pm.L_xva;
  row.total_notional = pm.total_notional;
  row.xva_epsilon_bps = pm.xva_epsilon_bps;
  row.accept_margin = pm.accept_margin;
  row.perf_cycles = safe_nan();
  row.perf_instructions = safe_nan();
  row.perf_cache_misses = safe_nan();
  row.perf_llc_load_misses = safe_nan();
  row.energy_pkg_joules = safe_nan();
  row.warmup_epochs = ctx.cfg.run.warmup_epochs;
  row.shock_sigma = ctx.cfg.workload.delta_gamma.shock_sigma;
  row.is_shadow_epoch = pm.is_shadow_epoch;
  return row;
}

}  // namespace

void set_current_epoch(int64_t epoch) { g_current_epoch.store(epoch, std::memory_order_relaxed); }

int64_t current_epoch() { return g_current_epoch.load(std::memory_order_relaxed); }

void set_current_notional_total(double total_notional) {
  g_current_notional_total.store(total_notional, std::memory_order_relaxed);
}

double current_notional_total() {
  return g_current_notional_total.load(std::memory_order_relaxed);
}

}  // namespace radc

extern "C" void radc_set_epoch(int64_t epoch) { radc::set_current_epoch(epoch); }
extern "C" void radc_set_total_notional(double total_notional) {
  radc::set_current_notional_total(total_notional);
}

namespace {

int mpi_allreduce_impl(const void* sendbuf, void* recvbuf, int count, MPI_Datatype datatype, MPI_Op op,
                       MPI_Comm comm) {
  static std::atomic<int> debug_once{0};
  if (radc::debug_enabled() && debug_once.fetch_add(1, std::memory_order_relaxed) == 0) {
    std::fprintf(stderr, "[radc] MPI_Allreduce wrapper active\n");
  }

  int initialized = 0;
  PMPI_Initialized(&initialized);
  if (!initialized) {
    return PMPI_Allreduce(sendbuf, recvbuf, count, datatype, op, comm);
  }

  int finalized = 0;
  PMPI_Finalized(&finalized);
  if (finalized) {
    return PMPI_Allreduce(sendbuf, recvbuf, count, datatype, op, comm);
  }

  auto& ctx = radc::context_for_comm(comm);
  const int64_t epoch = radc::resolve_epoch(ctx);
  const bool is_shadow_epoch =
      (ctx.cfg.logging.shadow_exact_every > 0 && epoch >= 0 &&
       (epoch % ctx.cfg.logging.shadow_exact_every == 0));
  const bool eligible = radc::eligible_for_protocol(ctx, sendbuf, recvbuf, count, datatype, op);
  if (!eligible) {
    // Strict v1 rule: only eligible calls are logged as epochs.
    return PMPI_Allreduce(sendbuf, recvbuf, count, datatype, op, comm);
  }

  if (!ctx.cfg.radc.enabled) {
    const double notional_total = radc::current_notional_total();
    const double t0 = MPI_Wtime();
    const int rc = PMPI_Allreduce(sendbuf, recvbuf, count, datatype, op, comm);
    if (rc == MPI_SUCCESS) {
      const int dtype_bytes = radc::datatype_size_bytes(datatype);
      const int64_t bytes_exact =
          (dtype_bytes > 0 && count > 0)
              ? static_cast<int64_t>(count) * static_cast<int64_t>(dtype_bytes)
              : 0;
      radc::maybe_emulate_network(bytes_exact, ctx.cfg);
    }
    const double t1 = MPI_Wtime();
    const double t_epoch_total_ms = 1000.0 * (t1 - t0);

    try {
      if (ctx.logger) {
        ctx.logger->log_metric(
            radc::make_exact_row(ctx, epoch, datatype, count, is_shadow_epoch, t_epoch_total_ms,
                                 notional_total));
      }
    } catch (...) {
      // Must never fail collectives due to logging/metadata errors.
    }
    return rc;
  }

  radc::ProtocolMetrics protocol_metrics{};
  bool used_protocol_path = false;
  int rc = MPI_SUCCESS;
  const double t0 = MPI_Wtime();

  try {
    radc::ensure_protocol_runtime(ctx);
    const double notional_total = radc::current_notional_total();
    if (datatype == MPI_DOUBLE && ctx.cfg.compression.double_mode == "passthrough") {
      rc = PMPI_Allreduce(sendbuf, recvbuf, count, datatype, op, comm);
      if (rc == MPI_SUCCESS) {
        const int dtype_bytes = radc::datatype_size_bytes(datatype);
        const int64_t bytes_exact =
            (dtype_bytes > 0 && count > 0)
                ? static_cast<int64_t>(count) * static_cast<int64_t>(dtype_bytes)
                : 0;
        radc::maybe_emulate_network(bytes_exact, ctx.cfg);
      }
      used_protocol_path = false;
    } else if (datatype == MPI_DOUBLE && ctx.cfg.compression.double_mode == "native64") {
      std::vector<double> local_matrix_f64;
      std::vector<double> out_matrix_f64;
      radc::copy_sendbuf_to_f64(sendbuf, count, datatype, local_matrix_f64);
      const radc::MatrixView<const double> local_view{
          local_matrix_f64.data(), ctx.cfg.buffer.N, ctx.cfg.buffer.S, ctx.cfg.buffer.S};
      radc::run_protocol_epoch(comm, ctx.cfg, epoch, local_view, ctx.protocol_runtime, out_matrix_f64,
                               protocol_metrics, notional_total);
      protocol_metrics.downcast_used = 0;
      radc::copy_f64_to_recvbuf(out_matrix_f64, count, datatype, recvbuf);
      rc = MPI_SUCCESS;
      used_protocol_path = true;
    } else {
      std::vector<float> local_matrix_f32;
      std::vector<float> out_matrix_f32;
      radc::copy_sendbuf_to_f32(sendbuf, count, datatype, local_matrix_f32);
      const radc::MatrixView<const float> local_view{
          local_matrix_f32.data(), ctx.cfg.buffer.N, ctx.cfg.buffer.S, ctx.cfg.buffer.S};
      radc::run_protocol_epoch(comm, ctx.cfg, epoch, local_view, ctx.protocol_runtime, out_matrix_f32,
                               protocol_metrics, notional_total);
      if (datatype == MPI_DOUBLE) {
        protocol_metrics.downcast_used = 1;
      }
      radc::copy_f32_to_recvbuf(out_matrix_f32, count, datatype, recvbuf);
      rc = MPI_SUCCESS;
      used_protocol_path = true;
    }
  } catch (const std::exception& e) {
    if (ctx.logger) {
      ctx.logger->log_event(epoch, "wrapper_error_fallback", e.what());
    }
    rc = PMPI_Allreduce(sendbuf, recvbuf, count, datatype, op, comm);
  } catch (...) {
    if (ctx.logger) {
      ctx.logger->log_event(epoch, "wrapper_error_fallback", "unknown exception");
    }
    rc = PMPI_Allreduce(sendbuf, recvbuf, count, datatype, op, comm);
  }

  const double t1 = MPI_Wtime();
  const double t_epoch_total_ms = 1000.0 * (t1 - t0);

  try {
    if (ctx.logger) {
      if (used_protocol_path) {
        ctx.logger->log_metric(
            radc::make_protocol_row(ctx, epoch, datatype, protocol_metrics, t_epoch_total_ms));
      } else {
        const double notional_total = radc::current_notional_total();
        ctx.logger->log_metric(
            radc::make_exact_row(ctx, epoch, datatype, count, is_shadow_epoch, t_epoch_total_ms,
                                 notional_total));
      }
    }
  } catch (...) {
    // Must never fail collectives due to logging/metadata errors.
  }

  return rc;
}

}  // namespace

#ifdef __APPLE__
extern "C" int radc_interpose_MPI_Allreduce(const void* sendbuf, void* recvbuf, int count,
                                             MPI_Datatype datatype, MPI_Op op, MPI_Comm comm) {
  return mpi_allreduce_impl(sendbuf, recvbuf, count, datatype, op, comm);
}

#define RADC_INTERPOSE(replacement, replacee)                                                       \
  __attribute__((used)) static struct {                                                             \
    const void* replacement;                                                                        \
    const void* replacee;                                                                           \
  } _radc_interpose_##replacee __attribute__((section("__DATA,__interpose"))) = {                  \
      (const void*)(unsigned long)&replacement, (const void*)(unsigned long)&replacee};            \

RADC_INTERPOSE(radc_interpose_MPI_Allreduce, MPI_Allreduce)
#else
extern "C" int MPI_Allreduce(const void* sendbuf, void* recvbuf, int count, MPI_Datatype datatype, MPI_Op op,
                              MPI_Comm comm) {
  return mpi_allreduce_impl(sendbuf, recvbuf, count, datatype, op, comm);
}
#endif
