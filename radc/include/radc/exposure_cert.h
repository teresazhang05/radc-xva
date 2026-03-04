#pragma once

#include <cstdint>
#include <vector>

namespace radc {

// Netting IDs: net_id[n] in [0, G-1]
struct SketchParams {
  int kG = 128;
  int kS = 128;
  int num_sketches = 2;
  uint64_t hash_seed_g[2]{777u, 1777u};
  uint64_t sign_seed_g[2]{123u, 1123u};
  uint64_t hash_seed_s[2]{999u, 1999u};
  uint64_t sign_seed_s[2]{456u, 1456u};
};

struct SafetyParams {
  double xva_epsilon_bps = 0.01;
  double accept_margin = 0.90;
  double jl_epsilon = 0.10;
  double jl_delta = 1e-12;
};

struct SafetyState {
  double notional_total = 0.0;
  double L = 0.0;
  double eps_dollars = 0.0;
  double rho_max = 0.0;
  double accept_threshold = 0.0;
};

SafetyState compute_safety_state(int G, int S, const std::vector<double>& a_g,
                                 const std::vector<double>& w_s, const std::vector<double>& c_g,
                                 double notional_total, const SafetyParams& sp);

// Compute B_i = S_G * (P*A_local) * S_S^T without forming E explicitly.
// A_local is N×S row-major.
void sketch_exposures_from_A_local_f32(const float* A_local, int N, int S, const int* net_id, int G,
                                       const SketchParams& kp, int sketch_id,
                                       std::vector<float>& B_out);

void sketch_exposures_from_A_local_f64(const double* A_local, int N, int S, const int* net_id, int G,
                                       const SketchParams& kp, int sketch_id,
                                       std::vector<double>& B_out);

// Compute B_hat = S_G*(P*U)* (S_S*V)^T.
// U is N×r row-major, V is S×r row-major.
// net_id maps trades->netting sets; we must compute U_net = P*U (G×r).
void sketch_exposures_from_factors_f32(const float* U, const float* V, int N, int S, int r,
                                       const int* net_id, int G, const SketchParams& kp,
                                       int sketch_id, std::vector<float>& Bhat_out);

void sketch_exposures_from_factors_f64(const double* U, const double* V, int N, int S, int r,
                                       const int* net_id, int G, const SketchParams& kp,
                                       int sketch_id, std::vector<double>& Bhat_out);

double frob_norm_diff(const std::vector<float>& A, const std::vector<float>& B);
double frob_norm_diff(const std::vector<double>& A, const std::vector<double>& B);

}  // namespace radc
