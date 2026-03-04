#include "radc/compressed_protocol.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <limits>
#include <map>
#include <stdexcept>
#include <string>
#include <thread>
#include <type_traits>
#include <vector>

#include <mpi.h>

#if defined(RADC_HAVE_CBLAS)
#if __has_include(<cblas.h>)
#include <cblas.h>
#elif __has_include(<Accelerate/Accelerate.h>)
#include <Accelerate/Accelerate.h>
#else
#error "RADC_HAVE_CBLAS is set but no CBLAS header is available"
#endif
#endif

#include "radc/rand_range_finder.h"
#include "radc/small_svd.h"

namespace radc {

namespace {

constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();

uint64_t splitmix64(uint64_t x) {
  uint64_t z = x + 0x9E3779B97F4A7C15ULL;
  z = (z ^ (z >> 30U)) * 0xBF58476D1CE4E5B9ULL;
  z = (z ^ (z >> 27U)) * 0x94D049BB133111EBULL;
  return z ^ (z >> 31U);
}

double elapsed_ms(double t0_s, double t1_s) { return 1000.0 * (t1_s - t0_s); }

void gemm_nn(int M, int N, int K, const float* A, const float* B, float* C) {
#if defined(RADC_HAVE_CBLAS)
  cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, K, 1.0f, A, K, B, N, 0.0f, C, N);
#else
  for (int i = 0; i < M; ++i) {
    for (int j = 0; j < N; ++j) {
      double acc = 0.0;
      for (int k = 0; k < K; ++k) {
        acc += static_cast<double>(A[static_cast<size_t>(i) * static_cast<size_t>(K) + static_cast<size_t>(k)]) *
               static_cast<double>(B[static_cast<size_t>(k) * static_cast<size_t>(N) + static_cast<size_t>(j)]);
      }
      C[static_cast<size_t>(i) * static_cast<size_t>(N) + static_cast<size_t>(j)] = static_cast<float>(acc);
    }
  }
#endif
}

void gemm_nn(int M, int N, int K, const double* A, const double* B, double* C) {
#if defined(RADC_HAVE_CBLAS)
  cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, K, 1.0, A, K, B, N, 0.0, C, N);
#else
  for (int i = 0; i < M; ++i) {
    for (int j = 0; j < N; ++j) {
      double acc = 0.0;
      for (int k = 0; k < K; ++k) {
        acc += A[static_cast<size_t>(i) * static_cast<size_t>(K) + static_cast<size_t>(k)] *
               B[static_cast<size_t>(k) * static_cast<size_t>(N) + static_cast<size_t>(j)];
      }
      C[static_cast<size_t>(i) * static_cast<size_t>(N) + static_cast<size_t>(j)] = acc;
    }
  }
#endif
}

void gemm_nt(int M, int N, int K, const float* A, const float* B, float* C) {
#if defined(RADC_HAVE_CBLAS)
  cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, M, N, K, 1.0f, A, K, B, K, 0.0f, C, N);
#else
  for (int i = 0; i < M; ++i) {
    for (int j = 0; j < N; ++j) {
      double acc = 0.0;
      for (int k = 0; k < K; ++k) {
        acc += static_cast<double>(A[static_cast<size_t>(i) * static_cast<size_t>(K) + static_cast<size_t>(k)]) *
               static_cast<double>(B[static_cast<size_t>(j) * static_cast<size_t>(K) + static_cast<size_t>(k)]);
      }
      C[static_cast<size_t>(i) * static_cast<size_t>(N) + static_cast<size_t>(j)] = static_cast<float>(acc);
    }
  }
#endif
}

void gemm_nt(int M, int N, int K, const double* A, const double* B, double* C) {
#if defined(RADC_HAVE_CBLAS)
  cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, M, N, K, 1.0, A, K, B, K, 0.0, C, N);
#else
  for (int i = 0; i < M; ++i) {
    for (int j = 0; j < N; ++j) {
      double acc = 0.0;
      for (int k = 0; k < K; ++k) {
        acc += A[static_cast<size_t>(i) * static_cast<size_t>(K) + static_cast<size_t>(k)] *
               B[static_cast<size_t>(j) * static_cast<size_t>(K) + static_cast<size_t>(k)];
      }
      C[static_cast<size_t>(i) * static_cast<size_t>(N) + static_cast<size_t>(j)] = acc;
    }
  }
#endif
}

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

int dtype_size_bytes(const std::string& dtype_in) {
  if (dtype_in == "float32") {
    return 4;
  }
  if (dtype_in == "float64") {
    return 8;
  }
  return 4;
}

template <typename T>
MPI_Datatype mpi_dtype();

template <>
MPI_Datatype mpi_dtype<float>() {
  return MPI_FLOAT;
}

template <>
MPI_Datatype mpi_dtype<double>() {
  return MPI_DOUBLE;
}

template <typename T>
void allreduce_sum_inplace_vec(std::vector<T>& buf, MPI_Comm comm) {
  if (buf.empty()) {
    return;
  }
  if (buf.size() > static_cast<size_t>(std::numeric_limits<int>::max())) {
    throw std::runtime_error("allreduce buffer too large for int count");
  }
  const int rc = PMPI_Allreduce(MPI_IN_PLACE, buf.data(), static_cast<int>(buf.size()), mpi_dtype<T>(),
                                MPI_SUM, comm);
  if (rc != MPI_SUCCESS) {
    throw std::runtime_error("PMPI_Allreduce failed");
  }
}

template <typename T>
double global_frob_norm(const MatrixView<const T>& A_local, MPI_Comm comm) {
  double local_sum = 0.0;
  for (int64_t n = 0; n < A_local.rows; ++n) {
    const T* row = A_local.data + n * A_local.stride;
    for (int64_t s = 0; s < A_local.cols; ++s) {
      const double v = static_cast<double>(row[s]);
      local_sum += v * v;
    }
  }

  double global_sum = 0.0;
  if (PMPI_Allreduce(&local_sum, &global_sum, 1, MPI_DOUBLE, MPI_SUM, comm) != MPI_SUCCESS) {
    throw std::runtime_error("PMPI_Allreduce for frob norm failed");
  }
  return std::sqrt(global_sum);
}

template <typename T>
void exact_reduce(const MatrixView<const T>& A_local, MPI_Comm comm,
                  std::vector<T>& exact_global) {
  exact_global.resize(static_cast<size_t>(A_local.rows) * static_cast<size_t>(A_local.cols));
  for (int64_t n = 0; n < A_local.rows; ++n) {
    const T* row = A_local.data + n * A_local.stride;
    std::copy(row, row + A_local.cols,
              exact_global.data() + static_cast<size_t>(n) * static_cast<size_t>(A_local.cols));
  }
  allreduce_sum_inplace_vec(exact_global, comm);
}

template <typename T, typename QB, typename SVD>
void reconstruct_truncated(const QB& qb, const SVD& svd, int r,
                           std::vector<T>& U_out, std::vector<T>& V_out,
                           std::vector<T>& A_hat) {
  const int64_t N = qb.N;
  const int64_t S = qb.S;
  const int l = qb.l;
  if (r <= 0 || r > l) {
    throw std::invalid_argument("reconstruct_truncated: invalid rank");
  }

  std::vector<T> U_r(static_cast<size_t>(l) * static_cast<size_t>(r), static_cast<T>(0));
  for (int t = 0; t < l; ++t) {
    for (int j = 0; j < r; ++j) {
      U_r[static_cast<size_t>(t) * static_cast<size_t>(r) + static_cast<size_t>(j)] =
          static_cast<T>(svd.U_l[static_cast<size_t>(t) * static_cast<size_t>(l) + static_cast<size_t>(j)]);
    }
  }

  U_out.assign(static_cast<size_t>(N) * static_cast<size_t>(r), static_cast<T>(0));
  gemm_nn(static_cast<int>(N), r, l, qb.Q.data(), U_r.data(), U_out.data());
  for (int64_t n = 0; n < N; ++n) {
    T* row = U_out.data() + static_cast<size_t>(n) * static_cast<size_t>(r);
    for (int j = 0; j < r; ++j) {
      row[j] *= static_cast<T>(svd.s[static_cast<size_t>(j)]);
    }
  }

  V_out.assign(static_cast<size_t>(S) * static_cast<size_t>(r), static_cast<T>(0));
  for (int64_t s = 0; s < S; ++s) {
    for (int j = 0; j < r; ++j) {
      V_out[static_cast<size_t>(s) * static_cast<size_t>(r) + static_cast<size_t>(j)] =
          static_cast<T>(svd.Vt[static_cast<size_t>(j) * static_cast<size_t>(S) + static_cast<size_t>(s)]);
    }
  }

  A_hat.assign(static_cast<size_t>(N) * static_cast<size_t>(S), static_cast<T>(0));
  gemm_nt(static_cast<int>(N), static_cast<int>(S), r, U_out.data(), V_out.data(), A_hat.data());
}

template <typename T>
double xva_cva_like_any(const MatrixView<const T>& C, const RiskState& st);

template <>
double xva_cva_like_any<float>(const MatrixView<const float>& C, const RiskState& st) {
  return xva_cva_like_f32(C, st);
}

template <>
double xva_cva_like_any<double>(const MatrixView<const double>& C, const RiskState& st) {
  return xva_cva_like(C, st);
}

template <typename T>
struct CertOps;

template <>
struct CertOps<float> {
  using Vec = std::vector<float>;
  static void sketch_local(const float* A_local, int N, int S, const int* net_id, int G,
                           const SketchParams& kp, int sid, Vec& out) {
    sketch_exposures_from_A_local_f32(A_local, N, S, net_id, G, kp, sid, out);
  }
  static void sketch_factors(const float* U, const float* V, int N, int S, int r, const int* net_id,
                             int G, const SketchParams& kp, int sid, Vec& out) {
    sketch_exposures_from_factors_f32(U, V, N, S, r, net_id, G, kp, sid, out);
  }
  static double diff(const Vec& A, const Vec& B) { return frob_norm_diff(A, B); }
};

template <>
struct CertOps<double> {
  using Vec = std::vector<double>;
  static void sketch_local(const double* A_local, int N, int S, const int* net_id, int G,
                           const SketchParams& kp, int sid, Vec& out) {
    sketch_exposures_from_A_local_f64(A_local, N, S, net_id, G, kp, sid, out);
  }
  static void sketch_factors(const double* U, const double* V, int N, int S, int r,
                             const int* net_id, int G, const SketchParams& kp, int sid,
                             Vec& out) {
    sketch_exposures_from_factors_f64(U, V, N, S, r, net_id, G, kp, sid, out);
  }
  static double diff(const Vec& A, const Vec& B) { return frob_norm_diff(A, B); }
};

template <typename T>
void run_protocol_epoch_impl(MPI_Comm comm, const Config& cfg, int64_t epoch,
                             const MatrixView<const T>& A_local,
                             ProtocolRuntime& rt, std::vector<T>& out_matrix,
                             ProtocolMetrics& m, double total_notional_override,
                             bool downcast_used) {
  m = ProtocolMetrics{};
  m.mode = "exact";
  m.dtype_in = cfg.buffer.dtype_in;
  m.dtype_internal = std::is_same<T, float>::value ? "float32" : "float64";
  m.factor_dtype = m.dtype_internal;
  m.downcast_used = downcast_used ? 1 : 0;
  m.kG = rt.sketch.kG;
  m.kS = rt.sketch.kS;
  m.num_sketches = rt.sketch.num_sketches;
  m.xva_epsilon_bps = rt.safety_cfg.xva_epsilon_bps;
  m.accept_margin = rt.safety_cfg.accept_margin;

  const bool use_override_notional =
      std::isfinite(total_notional_override) && total_notional_override > 0.0;
  double effective_notional = use_override_notional ? total_notional_override : cfg.risk.notional_total;
  if (!(std::isfinite(effective_notional) && effective_notional > 0.0)) {
    effective_notional = 1.0;
  }
  m.total_notional = effective_notional;

  std::vector<double> a_g(static_cast<size_t>(rt.risk_state.G), 1.0);
  std::vector<double> w_s(static_cast<size_t>(rt.risk_state.S), 0.0);
  std::vector<double> c_g(static_cast<size_t>(rt.risk_state.G), 0.0);
  for (int g = 0; g < rt.risk_state.G; ++g) {
    a_g[static_cast<size_t>(g)] = static_cast<double>(rt.risk_state.a_g[static_cast<size_t>(g)]);
    c_g[static_cast<size_t>(g)] = static_cast<double>(rt.risk_state.H_g[static_cast<size_t>(g)]);
  }
  for (int s = 0; s < rt.risk_state.S; ++s) {
    w_s[static_cast<size_t>(s)] = static_cast<double>(rt.risk_state.w_s[static_cast<size_t>(s)]);
  }
  const SafetyState epoch_safety =
      compute_safety_state(rt.risk_state.G, rt.risk_state.S, a_g, w_s, c_g, effective_notional,
                           rt.safety_cfg);
  m.L_xva = epoch_safety.L;
  m.eps_dollars = epoch_safety.eps_dollars;
  m.rho_max = epoch_safety.rho_max;
  m.accept_threshold = epoch_safety.accept_threshold;

  const int64_t N = A_local.rows;
  const int64_t S = A_local.cols;
  const int64_t count = N * S;

  if (count <= 0) {
    throw std::runtime_error("run_protocol_epoch: empty matrix");
  }

  m.is_shadow_epoch =
      (cfg.logging.shadow_exact_every > 0 && (epoch % cfg.logging.shadow_exact_every == 0)) ? 1 : 0;

  const int dtype_bytes = dtype_size_bytes(cfg.buffer.dtype_in);
  m.bytes_exact_payload = count * static_cast<int64_t>(dtype_bytes);
  m.bytes_comp_attempt_payload = 0;
  m.bytes_exact_fallback_payload = 0;
  m.bytes_total_payload = m.bytes_exact_payload;
  m.bytes_comp_payload = 0;
  m.compression_ratio_payload = 1.0;

  m.frob_norm_exact = global_frob_norm(A_local, comm);

  if (!cfg.radc.enabled) {
    exact_reduce(A_local, comm, out_matrix);
    maybe_emulate_network(m.bytes_exact_payload, cfg);
    if (m.is_shadow_epoch == 1) {
      MatrixView<const T> exact_view{out_matrix.data(), N, S, S};
      m.xva_true = xva_cva_like_any<T>(exact_view, rt.risk_state);
      m.xva_approx = m.xva_true;
      m.xva_err_abs = 0.0;
      m.xva_err_bps = 0.0;
    }
    m.l = std::max(1, rt.l_next);
    m.l_used = m.l;
    m.r_used = 0;
    return;
  }

  const bool shock_detected = cfg.detect.enabled &&
                              cfg.workload.kind == "delta_gamma" &&
                              std::isfinite(cfg.workload.delta_gamma.shock_sigma) &&
                              std::isfinite(cfg.detect.delta_m_sigma_threshold) &&
                              (cfg.workload.delta_gamma.shock_sigma >= cfg.detect.delta_m_sigma_threshold);
  if (shock_detected) {
    const double t_fb0 = MPI_Wtime();
    exact_reduce(A_local, comm, out_matrix);
    const double t_fb1 = MPI_Wtime();
    m.t_exact_fallback_ms = elapsed_ms(t_fb0, t_fb1);
    m.mode = "compressed_fallback";
    m.accepted = 0;
    m.fallback_triggered = 1;
    m.l = rt.l_next;
    m.l_used = 0;
    m.r_used = 0;
    m.energy_at_r = kNaN;
    m.rho_hat_sketch_0 = kNaN;
    m.rho_hat_sketch_1 = kNaN;
    m.rho_hat_sketch_max = kNaN;
    m.rho_hat_sketch = kNaN;
    m.bytes_comp_attempt_payload = 0;
    m.bytes_exact_fallback_payload = m.bytes_exact_payload;
    m.bytes_total_payload = m.bytes_exact_payload;
    m.bytes_comp_payload = m.bytes_comp_attempt_payload;
    m.compression_ratio_payload =
        (m.bytes_total_payload > 0)
            ? static_cast<double>(m.bytes_exact_payload) / static_cast<double>(m.bytes_total_payload)
            : kNaN;
    maybe_emulate_network(m.bytes_total_payload, cfg);
    if (m.is_shadow_epoch == 1) {
      MatrixView<const T> exact_view{out_matrix.data(), N, S, S};
      m.xva_true = xva_cva_like_any<T>(exact_view, rt.risk_state);
      m.xva_approx = m.xva_true;
      m.xva_err_abs = 0.0;
      m.xva_err_bps = 0.0;
    }
    return;
  }

  const bool run_verify = cfg.safety.verify_enabled && !cfg.safety.always_accept;
  const uint64_t omega_seed_epoch = splitmix64(cfg.run.seed ^ static_cast<uint64_t>(epoch));

  m.t_compute_Y_ms = 0.0;
  m.t_allreduce_Y_ms = 0.0;
  m.t_qr_ms = 0.0;
  m.t_compute_B_ms = 0.0;
  m.t_allreduce_B_ms = 0.0;

  int l_request = std::max(rt.l_min, std::min(rt.l_next, rt.l_max));
  int r_used = 1;
  double energy_at_r = 0.0;
  int64_t qb_payload_bytes = 0;

  std::vector<T> U_factor;
  std::vector<T> V_factor;
  std::vector<T> A_hat;

  if constexpr (std::is_same<T, float>::value) {
    QBResult qb{};
    SmallSVDResult svd{};
    while (true) {
      std::map<std::string, double> qbt;
      qb = distributed_qb_allreduce(comm, A_local, l_request, cfg.radc.power_iters, cfg.radc.omega_dist,
                                    omega_seed_epoch, qbt);
      m.t_compute_Y_ms += qbt["t_compute_Y_ms"];
      m.t_allreduce_Y_ms += qbt["t_allreduce_Y_ms"];
      m.t_qr_ms += qbt["t_qr_ms"];
      m.t_compute_B_ms += qbt["t_compute_B_ms"];
      m.t_allreduce_B_ms += qbt["t_allreduce_B_ms"];
      qb_payload_bytes += (N * static_cast<int64_t>(qb.l) + static_cast<int64_t>(qb.l) * S) *
                          static_cast<int64_t>(sizeof(T));

      const double t_svd0 = MPI_Wtime();
      svd = svd_B_and_energy(qb.l, qb.S, qb.B);
      const double t_svd1 = MPI_Wtime();
      if (!std::isfinite(m.t_small_svd_ms)) {
        m.t_small_svd_ms = 0.0;
      }
      m.t_small_svd_ms += elapsed_ms(t_svd0, t_svd1);

      const int r_cap = std::min(cfg.radc.r_max, qb.l);
      float eatr = 0.0f;
      r_used = select_rank_by_energy(svd.energy_prefix, cfg.radc.r_min, r_cap,
                                     static_cast<float>(cfg.radc.energy_capture), &eatr);
      energy_at_r = static_cast<double>(eatr);

      const bool energy_met =
          (energy_at_r + 1e-6) >= static_cast<double>(cfg.radc.energy_capture);
      const bool can_grow = (l_request < rt.l_max);
      const bool rank_cap_hit = (r_used >= r_cap);
      if (!(can_grow && rank_cap_hit && !energy_met)) {
        m.l = qb.l;
        m.l_used = qb.l;
        break;
      }
      const int grow = std::max(1, cfg.radc.oversample_p);
      const int next_l = std::min(rt.l_max, std::max(l_request + grow, r_used + cfg.radc.oversample_p));
      if (next_l <= l_request) {
        m.l = qb.l;
        m.l_used = qb.l;
        break;
      }
      l_request = next_l;
    }

    m.r_used = r_used;
    m.energy_at_r = energy_at_r;
    rt.l_next = std::max(rt.l_min, std::min(rt.l_max, r_used + cfg.radc.oversample_p));

    const double t0 = MPI_Wtime();
    reconstruct_truncated<T, QBResult, SmallSVDResult>(qb, svd, r_used, U_factor, V_factor, A_hat);
    const double t1 = MPI_Wtime();
    m.t_reconstruct_ms = elapsed_ms(t0, t1);
  } else {
    QBResultF64 qb{};
    SmallSVDResultF64 svd{};
    while (true) {
      std::map<std::string, double> qbt;
      qb = distributed_qb_allreduce_f64(comm, A_local, l_request, cfg.radc.power_iters,
                                        cfg.radc.omega_dist, omega_seed_epoch, qbt);
      m.t_compute_Y_ms += qbt["t_compute_Y_ms"];
      m.t_allreduce_Y_ms += qbt["t_allreduce_Y_ms"];
      m.t_qr_ms += qbt["t_qr_ms"];
      m.t_compute_B_ms += qbt["t_compute_B_ms"];
      m.t_allreduce_B_ms += qbt["t_allreduce_B_ms"];
      qb_payload_bytes += (N * static_cast<int64_t>(qb.l) + static_cast<int64_t>(qb.l) * S) *
                          static_cast<int64_t>(sizeof(T));

      const double t_svd0 = MPI_Wtime();
      svd = svd_B_and_energy_f64(qb.l, qb.S, qb.B);
      const double t_svd1 = MPI_Wtime();
      if (!std::isfinite(m.t_small_svd_ms)) {
        m.t_small_svd_ms = 0.0;
      }
      m.t_small_svd_ms += elapsed_ms(t_svd0, t_svd1);

      const int r_cap = std::min(cfg.radc.r_max, qb.l);
      double eatr = 0.0;
      r_used = select_rank_by_energy(svd.energy_prefix, cfg.radc.r_min, r_cap,
                                     cfg.radc.energy_capture, &eatr);
      energy_at_r = eatr;

      const bool energy_met =
          (energy_at_r + 1e-9) >= static_cast<double>(cfg.radc.energy_capture);
      const bool can_grow = (l_request < rt.l_max);
      const bool rank_cap_hit = (r_used >= r_cap);
      if (!(can_grow && rank_cap_hit && !energy_met)) {
        m.l = qb.l;
        m.l_used = qb.l;
        break;
      }
      const int grow = std::max(1, cfg.radc.oversample_p);
      const int next_l = std::min(rt.l_max, std::max(l_request + grow, r_used + cfg.radc.oversample_p));
      if (next_l <= l_request) {
        m.l = qb.l;
        m.l_used = qb.l;
        break;
      }
      l_request = next_l;
    }

    m.r_used = r_used;
    m.energy_at_r = energy_at_r;
    rt.l_next = std::max(rt.l_min, std::min(rt.l_max, r_used + cfg.radc.oversample_p));

    const double t0 = MPI_Wtime();
    reconstruct_truncated<T, QBResultF64, SmallSVDResultF64>(qb, svd, r_used, U_factor, V_factor,
                                                             A_hat);
    const double t1 = MPI_Wtime();
    m.t_reconstruct_ms = elapsed_ms(t0, t1);
  }

  const int64_t sketch_payload_bytes =
      run_verify
          ? static_cast<int64_t>(rt.sketch.num_sketches) * static_cast<int64_t>(rt.sketch.kG) *
                static_cast<int64_t>(rt.sketch.kS) * static_cast<int64_t>(sizeof(T))
          : 0;
  m.bytes_comp_attempt_payload = qb_payload_bytes + sketch_payload_bytes;

  bool accepted_local = true;
  if (run_verify) {
    const double t_verify0 = MPI_Wtime();
    m.t_sketch_local_ms = 0.0;
    m.t_allreduce_sketch_ms = 0.0;
    m.rho_hat_sketch_0 = kNaN;
    m.rho_hat_sketch_1 = kNaN;

    double rho_hat_max = -std::numeric_limits<double>::infinity();
    for (int sid = 0; sid < rt.sketch.num_sketches; ++sid) {
      typename CertOps<T>::Vec B_sum;
      {
        const double t0 = MPI_Wtime();
        CertOps<T>::sketch_local(A_local.data, static_cast<int>(N), static_cast<int>(S),
                                 rt.risk_state.netting_of_trade.data(), rt.risk_state.G, rt.sketch,
                                 sid, B_sum);
        const double t1 = MPI_Wtime();
        m.t_sketch_local_ms += elapsed_ms(t0, t1);
      }
      {
        const double t0 = MPI_Wtime();
        allreduce_sum_inplace_vec(B_sum, comm);
        const double t1 = MPI_Wtime();
        m.t_allreduce_sketch_ms += elapsed_ms(t0, t1);
      }
      typename CertOps<T>::Vec B_hat;
      {
        const double t0 = MPI_Wtime();
        CertOps<T>::sketch_factors(U_factor.data(), V_factor.data(), static_cast<int>(N),
                                   static_cast<int>(S), r_used,
                                   rt.risk_state.netting_of_trade.data(), rt.risk_state.G,
                                   rt.sketch, sid, B_hat);
        const double t1 = MPI_Wtime();
        m.t_sketch_local_ms += elapsed_ms(t0, t1);
      }

      const double rho = CertOps<T>::diff(B_sum, B_hat);
      if (sid == 0) {
        m.rho_hat_sketch_0 = rho;
      } else if (sid == 1) {
        m.rho_hat_sketch_1 = rho;
      }
      rho_hat_max = std::max(rho_hat_max, rho);
    }

    m.rho_hat_sketch_max = rho_hat_max;
    m.rho_hat_sketch = rho_hat_max;
    const double t_verify1 = MPI_Wtime();
    m.t_verify_ms = elapsed_ms(t_verify0, t_verify1);
    accepted_local =
        std::isfinite(m.rho_hat_sketch_max) && (m.rho_hat_sketch_max <= epoch_safety.accept_threshold);
  } else {
    m.rho_hat_sketch_0 = kNaN;
    m.rho_hat_sketch_1 = kNaN;
    m.rho_hat_sketch_max = kNaN;
    m.rho_hat_sketch = kNaN;
  }

  bool need_fallback_local = !accepted_local;
  if (cfg.detect.force_fallback_if_r_hits_rmax && r_used >= cfg.radc.r_max) {
    need_fallback_local = true;
  }

  int need_fallback_i = need_fallback_local ? 1 : 0;
  if (PMPI_Allreduce(MPI_IN_PLACE, &need_fallback_i, 1, MPI_INT, MPI_MAX, comm) != MPI_SUCCESS) {
    throw std::runtime_error("PMPI_Allreduce failed for fallback OR");
  }
  const bool need_fallback_global = (need_fallback_i != 0);

  std::vector<T> exact_global;
  if (need_fallback_global) {
    const double t0 = MPI_Wtime();
    exact_reduce(A_local, comm, exact_global);
    const double t1 = MPI_Wtime();
    m.t_exact_fallback_ms = elapsed_ms(t0, t1);
    out_matrix = exact_global;
    m.mode = "compressed_fallback";
    m.accepted = 0;
    m.fallback_triggered = 1;
    m.bytes_exact_fallback_payload = m.bytes_exact_payload;
    m.bytes_total_payload = m.bytes_comp_attempt_payload + m.bytes_exact_fallback_payload;
  } else {
    out_matrix = A_hat;
    m.mode = "compressed_accept";
    m.accepted = 1;
    m.fallback_triggered = 0;
    m.bytes_exact_fallback_payload = 0;
    m.bytes_total_payload = m.bytes_comp_attempt_payload;
  }

  m.bytes_comp_payload = m.bytes_comp_attempt_payload;
  m.compression_ratio_payload =
      (m.bytes_total_payload > 0)
          ? static_cast<double>(m.bytes_exact_payload) / static_cast<double>(m.bytes_total_payload)
          : kNaN;

  maybe_emulate_network(m.bytes_total_payload, cfg);

  if (m.is_shadow_epoch == 1) {
    if (exact_global.empty()) {
      exact_reduce(A_local, comm, exact_global);
    }

    MatrixView<const T> exact_view{exact_global.data(), N, S, S};
    MatrixView<const T> approx_view{A_hat.data(), N, S, S};
    m.xva_true = xva_cva_like_any<T>(exact_view, rt.risk_state);
    m.xva_approx = xva_cva_like_any<T>(approx_view, rt.risk_state);
    m.xva_err_abs = std::abs(m.xva_true - m.xva_approx);
    m.xva_err_bps = xva_error_bps(m.xva_true, m.xva_approx, effective_notional);
  }
}

}  // namespace

void init_protocol_runtime(const Config& cfg, int64_t N, int64_t S, ProtocolRuntime& rt) {
  if (N <= 0 || S <= 0) {
    throw std::invalid_argument("init_protocol_runtime: invalid N/S");
  }

  RiskConfig rcfg{};
  rcfg.G = cfg.risk.netting_sets_G;
  rcfg.netting_seed = cfg.risk.netting_seed;
  rcfg.collateral_threshold = static_cast<float>(cfg.risk.collateral_threshold);
  rcfg.a_scalar = static_cast<float>(cfg.risk.a_scalar);
  rcfg.scenario_weights_uniform = (cfg.risk.scenario_weights == "uniform");
  rcfg.notional_total = cfg.risk.notional_total;

  rt.risk_state = build_risk_state(static_cast<int>(N), static_cast<int>(S), rcfg);

  rt.sketch.kG = cfg.sketch.kG;
  rt.sketch.kS = cfg.sketch.kS;
  rt.sketch.num_sketches = std::max(1, std::min(2, cfg.sketch.num_sketches));
  rt.sketch.hash_seed_g[0] = cfg.sketch.hash_seed_g0;
  rt.sketch.sign_seed_g[0] = cfg.sketch.sign_seed_g0;
  rt.sketch.hash_seed_s[0] = cfg.sketch.hash_seed_s0;
  rt.sketch.sign_seed_s[0] = cfg.sketch.sign_seed_s0;
  rt.sketch.hash_seed_g[1] = cfg.sketch.hash_seed_g1;
  rt.sketch.sign_seed_g[1] = cfg.sketch.sign_seed_g1;
  rt.sketch.hash_seed_s[1] = cfg.sketch.hash_seed_s1;
  rt.sketch.sign_seed_s[1] = cfg.sketch.sign_seed_s1;

  rt.safety_cfg.xva_epsilon_bps = cfg.safety.xva_epsilon_bps;
  rt.safety_cfg.accept_margin = cfg.safety.accept_margin;
  rt.safety_cfg.jl_epsilon = cfg.safety.jl_epsilon;
  rt.safety_cfg.jl_delta = cfg.safety.jl_delta;

  std::vector<double> a_g(static_cast<size_t>(rt.risk_state.G), 1.0);
  std::vector<double> w_s(static_cast<size_t>(rt.risk_state.S), 0.0);
  std::vector<double> c_g(static_cast<size_t>(rt.risk_state.G), 0.0);
  for (int g = 0; g < rt.risk_state.G; ++g) {
    a_g[static_cast<size_t>(g)] = static_cast<double>(rt.risk_state.a_g[static_cast<size_t>(g)]);
    c_g[static_cast<size_t>(g)] = static_cast<double>(rt.risk_state.H_g[static_cast<size_t>(g)]);
  }
  for (int s = 0; s < rt.risk_state.S; ++s) {
    w_s[static_cast<size_t>(s)] = static_cast<double>(rt.risk_state.w_s[static_cast<size_t>(s)]);
  }
  const double notional = (cfg.risk.notional_total > 0.0) ? cfg.risk.notional_total : 1.0;
  rt.safety_state = compute_safety_state(rt.risk_state.G, rt.risk_state.S, a_g, w_s, c_g, notional,
                                         rt.safety_cfg);

  const int64_t l_upper_i64 =
      std::min<int64_t>(static_cast<int64_t>(cfg.radc.r_max + cfg.radc.oversample_p),
                        std::min<int64_t>(N, S));
  const int l_max_cfg = std::max(1, static_cast<int>(l_upper_i64));
  const int l_min_cfg =
      std::max(1, std::min(l_max_cfg, cfg.radc.r_min + cfg.radc.oversample_p));
  rt.l_min = l_min_cfg;
  rt.l_max = l_max_cfg;
  rt.l_next = l_min_cfg;

  rt.N = N;
  rt.S = S;
  rt.initialized = true;
}

void run_protocol_epoch(MPI_Comm comm, const Config& cfg, int64_t epoch,
                        const MatrixView<const float>& A_local,
                        ProtocolRuntime& rt,
                        std::vector<float>& out_matrix,
                        ProtocolMetrics& m,
                        double total_notional_override) {
  run_protocol_epoch_impl<float>(comm, cfg, epoch, A_local, rt, out_matrix, m,
                                 total_notional_override, false);
}

void run_protocol_epoch(MPI_Comm comm, const Config& cfg, int64_t epoch,
                        const MatrixView<const double>& A_local,
                        ProtocolRuntime& rt,
                        std::vector<double>& out_matrix,
                        ProtocolMetrics& m,
                        double total_notional_override) {
  run_protocol_epoch_impl<double>(comm, cfg, epoch, A_local, rt, out_matrix, m,
                                  total_notional_override, false);
}

}  // namespace radc
