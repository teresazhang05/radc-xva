#pragma once

#include <cstdint>
#include <limits>
#include <map>
#include <string>
#include <vector>

#include <mpi.h>

#include "radc/config.h"
#include "radc/exposure_cert.h"
#include "radc/matrix_view.h"
#include "radc/risk_xva.h"

namespace radc {

struct ProtocolRuntime {
  bool initialized = false;
  int64_t N = 0;
  int64_t S = 0;
  int l_min = 1;
  int l_max = 1;
  int l_next = 1;
  RiskState risk_state{};
  SketchParams sketch{};
  SafetyParams safety_cfg{};
  SafetyState safety_state{};
};

struct ProtocolMetrics {
  std::string mode = "exact";
  int l = 0;       // legacy projection dim field
  int l_used = 0;  // explicit projection dimension used in current epoch
  int r_used = 0;
  double energy_at_r = 0.0;
  double frob_norm_exact = 0.0;
  double rho_hat_sketch_0 = std::numeric_limits<double>::quiet_NaN();
  double rho_hat_sketch_1 = std::numeric_limits<double>::quiet_NaN();
  double rho_hat_sketch_max = std::numeric_limits<double>::quiet_NaN();
  double rho_hat_sketch = std::numeric_limits<double>::quiet_NaN();  // alias for compatibility
  double rho_max = 0.0;
  double eps_dollars = std::numeric_limits<double>::quiet_NaN();
  double accept_threshold = std::numeric_limits<double>::quiet_NaN();
  int kG = 0;
  int kS = 0;
  int num_sketches = 0;
  int accepted = 0;
  int fallback_triggered = 0;
  int64_t bytes_exact_payload = 0;
  int64_t bytes_comp_payload = 0;  // alias for compatibility
  int64_t bytes_comp_attempt_payload = 0;
  int64_t bytes_exact_fallback_payload = 0;
  int64_t bytes_total_payload = 0;
  double compression_ratio_payload = std::numeric_limits<double>::quiet_NaN();
  double t_compute_Y_ms = std::numeric_limits<double>::quiet_NaN();
  double t_allreduce_Y_ms = std::numeric_limits<double>::quiet_NaN();
  double t_qr_ms = std::numeric_limits<double>::quiet_NaN();
  double t_compute_B_ms = std::numeric_limits<double>::quiet_NaN();
  double t_allreduce_B_ms = std::numeric_limits<double>::quiet_NaN();
  double t_small_svd_ms = std::numeric_limits<double>::quiet_NaN();
  double t_reconstruct_ms = std::numeric_limits<double>::quiet_NaN();
  double t_sketch_local_ms = std::numeric_limits<double>::quiet_NaN();
  double t_allreduce_sketch_ms = std::numeric_limits<double>::quiet_NaN();
  double t_verify_ms = std::numeric_limits<double>::quiet_NaN();
  double t_exact_fallback_ms = std::numeric_limits<double>::quiet_NaN();
  double xva_true = std::numeric_limits<double>::quiet_NaN();
  double xva_approx = std::numeric_limits<double>::quiet_NaN();
  double xva_err_abs = std::numeric_limits<double>::quiet_NaN();
  double xva_err_bps = std::numeric_limits<double>::quiet_NaN();
  double L_xva = std::numeric_limits<double>::quiet_NaN();
  double total_notional = std::numeric_limits<double>::quiet_NaN();
  double xva_epsilon_bps = std::numeric_limits<double>::quiet_NaN();
  double accept_margin = std::numeric_limits<double>::quiet_NaN();
  std::string dtype_in = "float32";
  std::string dtype_internal = "float32";
  std::string factor_dtype = "float32";
  int downcast_used = 0;
  int is_shadow_epoch = 0;
};

void init_protocol_runtime(const Config& cfg, int64_t N, int64_t S, ProtocolRuntime& rt);

void run_protocol_epoch(MPI_Comm comm, const Config& cfg, int64_t epoch,
                        const MatrixView<const float>& A_local,
                        ProtocolRuntime& rt,
                        std::vector<float>& out_matrix,
                        ProtocolMetrics& m,
                        double total_notional_override = std::numeric_limits<double>::quiet_NaN());

void run_protocol_epoch(MPI_Comm comm, const Config& cfg, int64_t epoch,
                        const MatrixView<const double>& A_local,
                        ProtocolRuntime& rt,
                        std::vector<double>& out_matrix,
                        ProtocolMetrics& m,
                        double total_notional_override = std::numeric_limits<double>::quiet_NaN());

}  // namespace radc
