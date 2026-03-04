#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <random>
#include <vector>

#include "radc/small_svd.h"

namespace {

double frob_norm(const std::vector<float>& A) {
  double sum = 0.0;
  for (float v : A) {
    const double x = static_cast<double>(v);
    sum += x * x;
  }
  return std::sqrt(sum);
}

}  // namespace

int main() {
  {
    const int l = 6;
    const int64_t S = 11;
    const int r_true = 3;

    std::vector<float> U(static_cast<size_t>(l) * static_cast<size_t>(r_true), 0.0f);
    std::vector<float> V(static_cast<size_t>(S) * static_cast<size_t>(r_true), 0.0f);

    std::mt19937 rng(1234);
    std::normal_distribution<float> nd(0.0f, 1.0f);

    for (float& v : U) {
      v = nd(rng);
    }
    for (float& v : V) {
      v = nd(rng);
    }

    std::vector<float> B(static_cast<size_t>(l) * static_cast<size_t>(S), 0.0f);
    for (int i = 0; i < l; ++i) {
      for (int64_t j = 0; j < S; ++j) {
        double acc = 0.0;
        for (int k = 0; k < r_true; ++k) {
          acc += static_cast<double>(U[static_cast<size_t>(i) * static_cast<size_t>(r_true) + static_cast<size_t>(k)]) *
                 static_cast<double>(V[static_cast<size_t>(j) * static_cast<size_t>(r_true) + static_cast<size_t>(k)]);
        }
        B[static_cast<size_t>(i) * static_cast<size_t>(S) + static_cast<size_t>(j)] = static_cast<float>(acc);
      }
    }

    const radc::SmallSVDResult svd = radc::svd_B_and_energy(l, S, B);

    if (svd.s.size() != static_cast<size_t>(l) || svd.U_l.size() != static_cast<size_t>(l) * static_cast<size_t>(l) ||
        svd.Vt.size() != static_cast<size_t>(l) * static_cast<size_t>(S) ||
        svd.energy_prefix.size() != static_cast<size_t>(l)) {
      std::cerr << "SVD output shape mismatch" << std::endl;
      return 1;
    }

    for (int i = 1; i < l; ++i) {
      if (svd.s[static_cast<size_t>(i)] > svd.s[static_cast<size_t>(i - 1)] + 1e-4f) {
        std::cerr << "Singular values are not sorted descending" << std::endl;
        return 1;
      }
      if (svd.energy_prefix[static_cast<size_t>(i)] + 1e-6f < svd.energy_prefix[static_cast<size_t>(i - 1)]) {
        std::cerr << "Energy prefix is not monotone" << std::endl;
        return 1;
      }
    }

    if (svd.energy_prefix.back() < 0.99f || svd.energy_prefix.back() > 1.01f) {
      std::cerr << "Energy prefix final value not near 1" << std::endl;
      return 1;
    }

    std::vector<float> Bhat(static_cast<size_t>(l) * static_cast<size_t>(S), 0.0f);
    for (int row = 0; row < l; ++row) {
      for (int64_t col = 0; col < S; ++col) {
        double acc = 0.0;
        for (int k = 0; k < l; ++k) {
          const double u = static_cast<double>(svd.U_l[static_cast<size_t>(row) * static_cast<size_t>(l) + static_cast<size_t>(k)]);
          const double s = static_cast<double>(svd.s[static_cast<size_t>(k)]);
          const double v = static_cast<double>(svd.Vt[static_cast<size_t>(k) * static_cast<size_t>(S) + static_cast<size_t>(col)]);
          acc += u * s * v;
        }
        Bhat[static_cast<size_t>(row) * static_cast<size_t>(S) + static_cast<size_t>(col)] = static_cast<float>(acc);
      }
    }

    std::vector<float> D(B.size(), 0.0f);
    for (size_t i = 0; i < B.size(); ++i) {
      D[i] = B[i] - Bhat[i];
    }
    const double rel_err = frob_norm(D) / std::max(1e-12, frob_norm(B));
    if (rel_err > 1e-2) {
      std::cerr << "SVD reconstruction error too high: " << rel_err << std::endl;
      return 1;
    }

    float energy_at_r = 0.0f;
    const int r = radc::select_rank_by_energy(svd.energy_prefix, 1, l, 0.90f, &energy_at_r);
    if (r < 1 || r > l || energy_at_r < 0.90f - 1e-4f) {
      std::cerr << "select_rank_by_energy failed" << std::endl;
      return 1;
    }
  }

  {
    const int l = 4;
    const int64_t S = 4;
    std::vector<float> B(static_cast<size_t>(l) * static_cast<size_t>(S), 0.0f);
    B[0 * S + 0] = 4.0f;
    B[1 * S + 1] = 3.0f;
    B[2 * S + 2] = 2.0f;
    B[3 * S + 3] = 1.0f;

    const radc::SmallSVDResult svd = radc::svd_B_and_energy(l, S, B);
    float energy_at_r = 0.0f;
    const int r = radc::select_rank_by_energy(svd.energy_prefix, 1, l, 0.83f, &energy_at_r);
    if (r != 2) {
      std::cerr << "Expected rank 2 for diagonal test, got " << r << std::endl;
      return 1;
    }
    if (energy_at_r < 0.83f - 1e-4f) {
      std::cerr << "Energy at selected rank is too low" << std::endl;
      return 1;
    }
  }

  return 0;
}
