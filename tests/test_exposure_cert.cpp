#include <cmath>
#include <iostream>
#include <limits>
#include <vector>

#include "radc/exposure_cert.h"

int main() {
  // Safety-state sanity check.
  {
    const int G = 2;
    const int S = 4;
    std::vector<double> a_g{2.0, 1.0};
    std::vector<double> w_s{0.25, 0.25, 0.25, 0.25};
    std::vector<double> c_g{0.0, 0.0};
    radc::SafetyParams sp{};
    sp.xva_epsilon_bps = 0.01;
    sp.accept_margin = 0.90;
    sp.jl_epsilon = 0.10;
    sp.jl_delta = 1e-12;
    const double notional = 1.0e9;

    const radc::SafetyState st =
        radc::compute_safety_state(G, S, a_g, w_s, c_g, notional, sp);
    const double L_expected = std::sqrt((2.0 * 2.0 + 1.0 * 1.0) * (4.0 * 0.25 * 0.25));
    const double eps_expected = (sp.xva_epsilon_bps / 1.0e4) * notional;
    const double rho_expected = eps_expected / L_expected;
    const double thr_expected = sp.accept_margin * (1.0 - sp.jl_epsilon) * rho_expected;

    if (std::abs(st.L - L_expected) > 1e-12 || std::abs(st.eps_dollars - eps_expected) > 1e-6 ||
        std::abs(st.rho_max - rho_expected) > 1e-6 ||
        std::abs(st.accept_threshold - thr_expected) > 1e-6) {
      std::cerr << "SafetyState mismatch" << std::endl;
      return 1;
    }
  }

  // Sketch consistency: if A = U V^T exactly, sketches must match.
  {
    const int N = 3;
    const int S = 4;
    const int G = 2;
    const int r = 2;
    const int net_id[N] = {0, 1, 0};

    const std::vector<float> U = {
        1.0f, 0.5f,   // n=0
        -2.0f, 1.0f,  // n=1
        0.3f, -0.2f   // n=2
    };
    const std::vector<float> V = {
        0.4f, -1.0f,  // s=0
        1.2f, 0.3f,   // s=1
        -0.7f, 0.8f,  // s=2
        2.0f, -0.1f   // s=3
    };

    std::vector<float> A(static_cast<size_t>(N) * static_cast<size_t>(S), 0.0f);
    for (int n = 0; n < N; ++n) {
      for (int s = 0; s < S; ++s) {
        float acc = 0.0f;
        for (int j = 0; j < r; ++j) {
          acc += U[static_cast<size_t>(n) * static_cast<size_t>(r) + static_cast<size_t>(j)] *
                 V[static_cast<size_t>(s) * static_cast<size_t>(r) + static_cast<size_t>(j)];
        }
        A[static_cast<size_t>(n) * static_cast<size_t>(S) + static_cast<size_t>(s)] = acc;
      }
    }

    radc::SketchParams kp{};
    kp.kG = 16;
    kp.kS = 16;
    kp.num_sketches = 2;
    kp.hash_seed_g[0] = 777;
    kp.sign_seed_g[0] = 123;
    kp.hash_seed_s[0] = 999;
    kp.sign_seed_s[0] = 456;
    kp.hash_seed_g[1] = 1777;
    kp.sign_seed_g[1] = 1123;
    kp.hash_seed_s[1] = 1999;
    kp.sign_seed_s[1] = 1456;

    for (int sid = 0; sid < 2; ++sid) {
      std::vector<float> B;
      std::vector<float> Bhat;
      radc::sketch_exposures_from_A_local_f32(A.data(), N, S, net_id, G, kp, sid, B);
      radc::sketch_exposures_from_factors_f32(U.data(), V.data(), N, S, r, net_id, G, kp, sid, Bhat);
      const double err = radc::frob_norm_diff(B, Bhat);
      if (!(std::isfinite(err) && err < 1e-5)) {
        std::cerr << "Sketch mismatch sid=" << sid << " err=" << err << std::endl;
        return 1;
      }
    }
  }

  return 0;
}
