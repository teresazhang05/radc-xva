#include "radc/small_svd.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <utility>
#include <vector>

namespace radc {

namespace {

void jacobi_eigendecomp_symmetric(std::vector<double>& A, int n, std::vector<double>& eigvecs,
                                  std::vector<double>& eigvals) {
  eigvecs.assign(static_cast<size_t>(n) * static_cast<size_t>(n), 0.0);
  for (int i = 0; i < n; ++i) {
    eigvecs[static_cast<size_t>(i) * static_cast<size_t>(n) + static_cast<size_t>(i)] = 1.0;
  }

  constexpr int kMaxSweeps = 64;
  const double tol = 1e-12;

  for (int sweep = 0; sweep < kMaxSweeps; ++sweep) {
    double max_offdiag = 0.0;
    int p = 0;
    int q = 1;

    for (int i = 0; i < n; ++i) {
      for (int j = i + 1; j < n; ++j) {
        const double v = std::abs(A[static_cast<size_t>(i) * static_cast<size_t>(n) +
                                    static_cast<size_t>(j)]);
        if (v > max_offdiag) {
          max_offdiag = v;
          p = i;
          q = j;
        }
      }
    }

    if (max_offdiag <= tol) {
      break;
    }

    const double app = A[static_cast<size_t>(p) * static_cast<size_t>(n) + static_cast<size_t>(p)];
    const double aqq = A[static_cast<size_t>(q) * static_cast<size_t>(n) + static_cast<size_t>(q)];
    const double apq = A[static_cast<size_t>(p) * static_cast<size_t>(n) + static_cast<size_t>(q)];

    const double phi = 0.5 * std::atan2(2.0 * apq, aqq - app);
    const double c = std::cos(phi);
    const double s = std::sin(phi);

    for (int k = 0; k < n; ++k) {
      const double aik = A[static_cast<size_t>(p) * static_cast<size_t>(n) + static_cast<size_t>(k)];
      const double aqk = A[static_cast<size_t>(q) * static_cast<size_t>(n) + static_cast<size_t>(k)];
      A[static_cast<size_t>(p) * static_cast<size_t>(n) + static_cast<size_t>(k)] = c * aik - s * aqk;
      A[static_cast<size_t>(q) * static_cast<size_t>(n) + static_cast<size_t>(k)] = s * aik + c * aqk;
    }

    for (int k = 0; k < n; ++k) {
      const double akp = A[static_cast<size_t>(k) * static_cast<size_t>(n) + static_cast<size_t>(p)];
      const double akq = A[static_cast<size_t>(k) * static_cast<size_t>(n) + static_cast<size_t>(q)];
      A[static_cast<size_t>(k) * static_cast<size_t>(n) + static_cast<size_t>(p)] = c * akp - s * akq;
      A[static_cast<size_t>(k) * static_cast<size_t>(n) + static_cast<size_t>(q)] = s * akp + c * akq;
    }

    A[static_cast<size_t>(p) * static_cast<size_t>(n) + static_cast<size_t>(q)] = 0.0;
    A[static_cast<size_t>(q) * static_cast<size_t>(n) + static_cast<size_t>(p)] = 0.0;

    for (int k = 0; k < n; ++k) {
      const double vkp = eigvecs[static_cast<size_t>(k) * static_cast<size_t>(n) + static_cast<size_t>(p)];
      const double vkq = eigvecs[static_cast<size_t>(k) * static_cast<size_t>(n) + static_cast<size_t>(q)];
      eigvecs[static_cast<size_t>(k) * static_cast<size_t>(n) + static_cast<size_t>(p)] = c * vkp - s * vkq;
      eigvecs[static_cast<size_t>(k) * static_cast<size_t>(n) + static_cast<size_t>(q)] = s * vkp + c * vkq;
    }
  }

  eigvals.assign(static_cast<size_t>(n), 0.0);
  for (int i = 0; i < n; ++i) {
    eigvals[static_cast<size_t>(i)] = A[static_cast<size_t>(i) * static_cast<size_t>(n) + static_cast<size_t>(i)];
  }
}

template <typename T>
struct SmallSVDResultT {
  int l;
  int64_t S;
  std::vector<T> U_l;
  std::vector<T> s;
  std::vector<T> Vt;
  std::vector<T> energy_prefix;
};

template <typename T>
SmallSVDResultT<T> svd_B_and_energy_t(int l, int64_t S, const std::vector<T>& B_rowmajor) {
  if (l <= 0 || S <= 0) {
    throw std::invalid_argument("svd_B_and_energy: invalid dimensions");
  }
  if (B_rowmajor.size() != static_cast<size_t>(l) * static_cast<size_t>(S)) {
    throw std::invalid_argument("svd_B_and_energy: invalid B size");
  }

  std::vector<double> gram(static_cast<size_t>(l) * static_cast<size_t>(l), 0.0);
  for (int i = 0; i < l; ++i) {
    for (int j = i; j < l; ++j) {
      double acc = 0.0;
      const T* bi = B_rowmajor.data() + static_cast<size_t>(i) * static_cast<size_t>(S);
      const T* bj = B_rowmajor.data() + static_cast<size_t>(j) * static_cast<size_t>(S);
      for (int64_t s = 0; s < S; ++s) {
        acc += static_cast<double>(bi[s]) * static_cast<double>(bj[s]);
      }
      gram[static_cast<size_t>(i) * static_cast<size_t>(l) + static_cast<size_t>(j)] = acc;
      gram[static_cast<size_t>(j) * static_cast<size_t>(l) + static_cast<size_t>(i)] = acc;
    }
  }

  std::vector<double> eigvecs;
  std::vector<double> eigvals;
  jacobi_eigendecomp_symmetric(gram, l, eigvecs, eigvals);

  std::vector<int> order(static_cast<size_t>(l));
  std::iota(order.begin(), order.end(), 0);
  std::sort(order.begin(), order.end(), [&](int a, int b) {
    return eigvals[static_cast<size_t>(a)] > eigvals[static_cast<size_t>(b)];
  });

  SmallSVDResultT<T> out{};
  out.l = l;
  out.S = S;
  out.U_l.assign(static_cast<size_t>(l) * static_cast<size_t>(l), static_cast<T>(0));
  out.s.assign(static_cast<size_t>(l), static_cast<T>(0));
  out.Vt.assign(static_cast<size_t>(l) * static_cast<size_t>(S), static_cast<T>(0));
  out.energy_prefix.assign(static_cast<size_t>(l), static_cast<T>(0));

  for (int col = 0; col < l; ++col) {
    const int src = order[static_cast<size_t>(col)];
    const double eval = std::max(0.0, eigvals[static_cast<size_t>(src)]);
    out.s[static_cast<size_t>(col)] = static_cast<T>(std::sqrt(eval));
    for (int row = 0; row < l; ++row) {
      out.U_l[static_cast<size_t>(row) * static_cast<size_t>(l) + static_cast<size_t>(col)] =
          static_cast<T>(eigvecs[static_cast<size_t>(row) * static_cast<size_t>(l) +
                                 static_cast<size_t>(src)]);
    }
  }

  constexpr double kSigmaEps = 1.0e-12;
  for (int i = 0; i < l; ++i) {
    const double sigma = static_cast<double>(out.s[static_cast<size_t>(i)]);
    if (sigma <= kSigmaEps) {
      continue;
    }

    for (int64_t s = 0; s < S; ++s) {
      double acc = 0.0;
      for (int k = 0; k < l; ++k) {
        const double u = static_cast<double>(out.U_l[static_cast<size_t>(k) * static_cast<size_t>(l) +
                                                  static_cast<size_t>(i)]);
        const double b = static_cast<double>(
            B_rowmajor[static_cast<size_t>(k) * static_cast<size_t>(S) + static_cast<size_t>(s)]);
        acc += u * b;
      }
      out.Vt[static_cast<size_t>(i) * static_cast<size_t>(S) + static_cast<size_t>(s)] =
          static_cast<T>(acc / sigma);
    }
  }

  double total_energy = 0.0;
  for (T sigma : out.s) {
    const double v = static_cast<double>(sigma);
    total_energy += v * v;
  }

  if (total_energy > 0.0) {
    double running = 0.0;
    for (int i = 0; i < l; ++i) {
      const double v = static_cast<double>(out.s[static_cast<size_t>(i)]);
      running += v * v;
      out.energy_prefix[static_cast<size_t>(i)] = static_cast<T>(running / total_energy);
    }
  }

  return out;
}

template <typename T>
int select_rank_by_energy_t(const std::vector<T>& energy_prefix, int r_min, int r_max,
                            double energy_capture, double* out_energy_at_r) {
  if (energy_prefix.empty()) {
    if (out_energy_at_r != nullptr) {
      *out_energy_at_r = 0.0;
    }
    return std::max(1, r_min);
  }

  const int max_allowed = std::min<int>(std::max(1, r_max), static_cast<int>(energy_prefix.size()));
  const int min_allowed = std::max(1, std::min(r_min, max_allowed));
  const double capture = std::max(0.0, std::min(1.0, energy_capture));

  int chosen = max_allowed;
  for (int r = min_allowed; r <= max_allowed; ++r) {
    if (static_cast<double>(energy_prefix[static_cast<size_t>(r - 1)]) >= capture) {
      chosen = r;
      break;
    }
  }

  if (out_energy_at_r != nullptr) {
    *out_energy_at_r = static_cast<double>(energy_prefix[static_cast<size_t>(chosen - 1)]);
  }
  return chosen;
}

}  // namespace

SmallSVDResult svd_B_and_energy(int l, int64_t S, const std::vector<float>& B_rowmajor) {
  const SmallSVDResultT<float> v = svd_B_and_energy_t<float>(l, S, B_rowmajor);
  return SmallSVDResult{v.l, v.S, v.U_l, v.s, v.Vt, v.energy_prefix};
}

SmallSVDResultF64 svd_B_and_energy_f64(int l, int64_t S, const std::vector<double>& B_rowmajor) {
  const SmallSVDResultT<double> v = svd_B_and_energy_t<double>(l, S, B_rowmajor);
  return SmallSVDResultF64{v.l, v.S, v.U_l, v.s, v.Vt, v.energy_prefix};
}

int select_rank_by_energy(const std::vector<float>& energy_prefix, int r_min, int r_max,
                          float energy_capture, float* out_energy_at_r) {
  double tmp = 0.0;
  const int r = select_rank_by_energy_t<float>(energy_prefix, r_min, r_max,
                                               static_cast<double>(energy_capture), &tmp);
  if (out_energy_at_r != nullptr) {
    *out_energy_at_r = static_cast<float>(tmp);
  }
  return r;
}

int select_rank_by_energy(const std::vector<double>& energy_prefix, int r_min, int r_max,
                          double energy_capture, double* out_energy_at_r) {
  return select_rank_by_energy_t<double>(energy_prefix, r_min, r_max, energy_capture,
                                         out_energy_at_r);
}

}  // namespace radc
