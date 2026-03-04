#include <cmath>
#include <iostream>
#include <vector>

#include "radc/countsketch2d.h"

int main() {
  const int64_t N = 3;
  const int64_t S = 4;
  std::vector<double> A = {
      1.0, 2.0, 3.0, 4.0,
      5.0, 6.0, 7.0, 8.0,
      9.0, 10.0, 11.0, 12.0,
  };

  radc::MatrixView<const double> view{A.data(), N, S, S};
  radc::CountSketch2D cs{};
  cs.k_row = 5;
  cs.k_col = 6;
  cs.hash_seed_row = 11;
  cs.hash_seed_col = 13;
  cs.sign_seed_row = 17;
  cs.sign_seed_col = 19;

  std::vector<float> K;
  cs.sketch_matrix_f64_to_f32(view, K);

  const int l = static_cast<int>(N);
  std::vector<float> Q(static_cast<size_t>(N) * static_cast<size_t>(l), 0.0f);
  for (int i = 0; i < l; ++i) {
    Q[static_cast<size_t>(i) * static_cast<size_t>(l) + static_cast<size_t>(i)] = 1.0f;
  }

  std::vector<float> B(static_cast<size_t>(l) * static_cast<size_t>(S), 0.0f);
  for (int64_t i = 0; i < N; ++i) {
    for (int64_t j = 0; j < S; ++j) {
      B[static_cast<size_t>(i) * static_cast<size_t>(S) + static_cast<size_t>(j)] =
          static_cast<float>(A[static_cast<size_t>(i) * static_cast<size_t>(S) + static_cast<size_t>(j)]);
    }
  }

  std::vector<float> Khat;
  cs.sketch_qb_f32(N, S, l, Q, B, Khat);

  const double diff = cs.frob_norm_diff(K, Khat);
  if (diff > 1e-4) {
    std::cerr << "CountSketch QB mismatch: diff=" << diff << std::endl;
    return 1;
  }

  std::vector<double> Bmat = {
      -0.3, 1.2, 0.7, -0.4,
      0.5, -1.1, 0.0, 0.9,
      2.0, -0.8, 0.2, 1.5,
  };
  std::vector<double> C(A.size(), 0.0);
  for (size_t i = 0; i < A.size(); ++i) {
    C[i] = A[i] + Bmat[i];
  }

  radc::MatrixView<const double> b_view{Bmat.data(), N, S, S};
  radc::MatrixView<const double> c_view{C.data(), N, S, S};
  std::vector<float> KB;
  std::vector<float> KC;
  cs.sketch_matrix_f64_to_f32(b_view, KB);
  cs.sketch_matrix_f64_to_f32(c_view, KC);

  if (K.size() != KB.size() || K.size() != KC.size()) {
    std::cerr << "Sketch output shape mismatch" << std::endl;
    return 1;
  }

  double lin_err = 0.0;
  for (size_t i = 0; i < K.size(); ++i) {
    const double lhs = static_cast<double>(KC[i]);
    const double rhs = static_cast<double>(K[i]) + static_cast<double>(KB[i]);
    const double d = lhs - rhs;
    lin_err += d * d;
  }
  lin_err = std::sqrt(lin_err);
  if (lin_err > 1e-5) {
    std::cerr << "CountSketch linearity check failed: err=" << lin_err << std::endl;
    return 1;
  }

  // Netting sketch path should match explicit sketch of E = M*A.
  const int G = 2;
  std::vector<int> netting = {0, 0, 1};
  std::vector<double> E(static_cast<size_t>(G) * static_cast<size_t>(S), 0.0);
  for (int64_t n = 0; n < N; ++n) {
    const int g = netting[static_cast<size_t>(n)];
    for (int64_t s = 0; s < S; ++s) {
      E[static_cast<size_t>(g) * static_cast<size_t>(S) + static_cast<size_t>(s)] +=
          A[static_cast<size_t>(n) * static_cast<size_t>(S) + static_cast<size_t>(s)];
    }
  }
  radc::MatrixView<const double> e_view{E.data(), G, S, S};
  std::vector<float> K_net;
  std::vector<float> K_explicit;
  cs.sketch_matrix_f64_to_f32_netting(view, netting, G, K_net);
  cs.sketch_matrix_f64_to_f32(e_view, K_explicit);
  if (K_net.size() != K_explicit.size()) {
    std::cerr << "Netting sketch size mismatch" << std::endl;
    return 1;
  }
  double net_err = 0.0;
  for (size_t i = 0; i < K_net.size(); ++i) {
    const double d = static_cast<double>(K_net[i]) - static_cast<double>(K_explicit[i]);
    net_err += d * d;
  }
  net_err = std::sqrt(net_err);
  if (net_err > 1e-5) {
    std::cerr << "Netting sketch mismatch: err=" << net_err << std::endl;
    return 1;
  }

  return 0;
}
