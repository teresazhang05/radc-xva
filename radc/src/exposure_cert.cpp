#include "radc/exposure_cert.h"

#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

namespace radc {

namespace {

uint64_t splitmix64(uint64_t x) {
  uint64_t z = x + 0x9E3779B97F4A7C15ULL;
  z = (z ^ (z >> 30U)) * 0xBF58476D1CE4E5B9ULL;
  z = (z ^ (z >> 27U)) * 0x94D049BB133111EBULL;
  return z ^ (z >> 31U);
}

int hash_bucket(uint64_t seed, int idx, int buckets) {
  const uint64_t h = splitmix64(seed ^ static_cast<uint64_t>(idx));
  return static_cast<int>(h % static_cast<uint64_t>(buckets));
}

float hash_sign(uint64_t seed, int idx) {
  const uint64_t h = splitmix64(seed ^ static_cast<uint64_t>(idx));
  return (h & 1ULL) ? 1.0f : -1.0f;
}

template <typename T>
void sketch_exposures_from_A_local_t(const T* A_local, int N, int S, const int* net_id, int G,
                                     const SketchParams& kp, int sketch_id,
                                     std::vector<T>& B_out) {
  if (A_local == nullptr || net_id == nullptr) {
    throw std::invalid_argument("sketch_exposures_from_A_local: null pointers");
  }
  if (N <= 0 || S <= 0 || G <= 0 || kp.kG <= 0 || kp.kS <= 0) {
    throw std::invalid_argument("sketch_exposures_from_A_local: invalid dimensions");
  }
  if (sketch_id < 0 || sketch_id >= kp.num_sketches || sketch_id >= 2) {
    throw std::invalid_argument("sketch_exposures_from_A_local: invalid sketch_id");
  }

  std::vector<int> row_bucket(static_cast<size_t>(G), 0);
  std::vector<float> row_sign(static_cast<size_t>(G), 1.0f);
  for (int g = 0; g < G; ++g) {
    row_bucket[static_cast<size_t>(g)] = hash_bucket(kp.hash_seed_g[sketch_id], g, kp.kG);
    row_sign[static_cast<size_t>(g)] = hash_sign(kp.sign_seed_g[sketch_id], g);
  }

  std::vector<int> col_bucket(static_cast<size_t>(S), 0);
  std::vector<float> col_sign(static_cast<size_t>(S), 1.0f);
  for (int s = 0; s < S; ++s) {
    col_bucket[static_cast<size_t>(s)] = hash_bucket(kp.hash_seed_s[sketch_id], s, kp.kS);
    col_sign[static_cast<size_t>(s)] = hash_sign(kp.sign_seed_s[sketch_id], s);
  }

  B_out.assign(static_cast<size_t>(kp.kG) * static_cast<size_t>(kp.kS), static_cast<T>(0));
  for (int n = 0; n < N; ++n) {
    const int g = net_id[n];
    if (g < 0 || g >= G) {
      throw std::invalid_argument("sketch_exposures_from_A_local: net_id out of range");
    }
    const int rg = row_bucket[static_cast<size_t>(g)];
    const T sg = static_cast<T>(row_sign[static_cast<size_t>(g)]);
    const T* a_row = A_local + static_cast<size_t>(n) * static_cast<size_t>(S);
    for (int s = 0; s < S; ++s) {
      const int cs = col_bucket[static_cast<size_t>(s)];
      const T ss = static_cast<T>(col_sign[static_cast<size_t>(s)]);
      B_out[static_cast<size_t>(rg) * static_cast<size_t>(kp.kS) + static_cast<size_t>(cs)] +=
          sg * ss * a_row[s];
    }
  }
}

template <typename T>
void sketch_exposures_from_factors_t(const T* U, const T* V, int N, int S, int r,
                                     const int* net_id, int G, const SketchParams& kp,
                                     int sketch_id, std::vector<T>& Bhat_out) {
  if (U == nullptr || V == nullptr || net_id == nullptr) {
    throw std::invalid_argument("sketch_exposures_from_factors: null pointers");
  }
  if (N <= 0 || S <= 0 || r <= 0 || G <= 0 || kp.kG <= 0 || kp.kS <= 0) {
    throw std::invalid_argument("sketch_exposures_from_factors: invalid dimensions");
  }
  if (sketch_id < 0 || sketch_id >= kp.num_sketches || sketch_id >= 2) {
    throw std::invalid_argument("sketch_exposures_from_factors: invalid sketch_id");
  }

  std::vector<int> row_bucket(static_cast<size_t>(G), 0);
  std::vector<float> row_sign(static_cast<size_t>(G), 1.0f);
  for (int g = 0; g < G; ++g) {
    row_bucket[static_cast<size_t>(g)] = hash_bucket(kp.hash_seed_g[sketch_id], g, kp.kG);
    row_sign[static_cast<size_t>(g)] = hash_sign(kp.sign_seed_g[sketch_id], g);
  }

  std::vector<int> col_bucket(static_cast<size_t>(S), 0);
  std::vector<float> col_sign(static_cast<size_t>(S), 1.0f);
  for (int s = 0; s < S; ++s) {
    col_bucket[static_cast<size_t>(s)] = hash_bucket(kp.hash_seed_s[sketch_id], s, kp.kS);
    col_sign[static_cast<size_t>(s)] = hash_sign(kp.sign_seed_s[sketch_id], s);
  }

  std::vector<T> U_net(static_cast<size_t>(G) * static_cast<size_t>(r), static_cast<T>(0));
  for (int n = 0; n < N; ++n) {
    const int g = net_id[n];
    if (g < 0 || g >= G) {
      throw std::invalid_argument("sketch_exposures_from_factors: net_id out of range");
    }
    T* dst = U_net.data() + static_cast<size_t>(g) * static_cast<size_t>(r);
    const T* src = U + static_cast<size_t>(n) * static_cast<size_t>(r);
    for (int j = 0; j < r; ++j) {
      dst[j] += src[j];
    }
  }

  std::vector<T> SU(static_cast<size_t>(kp.kG) * static_cast<size_t>(r), static_cast<T>(0));
  for (int g = 0; g < G; ++g) {
    const int rg = row_bucket[static_cast<size_t>(g)];
    const T sg = static_cast<T>(row_sign[static_cast<size_t>(g)]);
    const T* src = U_net.data() + static_cast<size_t>(g) * static_cast<size_t>(r);
    T* dst = SU.data() + static_cast<size_t>(rg) * static_cast<size_t>(r);
    for (int j = 0; j < r; ++j) {
      dst[j] += sg * src[j];
    }
  }

  std::vector<T> SV(static_cast<size_t>(kp.kS) * static_cast<size_t>(r), static_cast<T>(0));
  for (int s = 0; s < S; ++s) {
    const int cs = col_bucket[static_cast<size_t>(s)];
    const T ss = static_cast<T>(col_sign[static_cast<size_t>(s)]);
    const T* src = V + static_cast<size_t>(s) * static_cast<size_t>(r);
    T* dst = SV.data() + static_cast<size_t>(cs) * static_cast<size_t>(r);
    for (int j = 0; j < r; ++j) {
      dst[j] += ss * src[j];
    }
  }

  Bhat_out.assign(static_cast<size_t>(kp.kG) * static_cast<size_t>(kp.kS), static_cast<T>(0));
  for (int kg = 0; kg < kp.kG; ++kg) {
    const T* su = SU.data() + static_cast<size_t>(kg) * static_cast<size_t>(r);
    for (int ks = 0; ks < kp.kS; ++ks) {
      const T* sv = SV.data() + static_cast<size_t>(ks) * static_cast<size_t>(r);
      double acc = 0.0;
      for (int j = 0; j < r; ++j) {
        acc += static_cast<double>(su[j]) * static_cast<double>(sv[j]);
      }
      Bhat_out[static_cast<size_t>(kg) * static_cast<size_t>(kp.kS) + static_cast<size_t>(ks)] =
          static_cast<T>(acc);
    }
  }
}

template <typename T>
double frob_norm_diff_t(const std::vector<T>& A, const std::vector<T>& B) {
  if (A.size() != B.size()) {
    throw std::invalid_argument("frob_norm_diff: size mismatch");
  }
  double sum = 0.0;
  for (size_t i = 0; i < A.size(); ++i) {
    const double d = static_cast<double>(A[i]) - static_cast<double>(B[i]);
    sum += d * d;
  }
  return std::sqrt(sum);
}

}  // namespace

SafetyState compute_safety_state(int G, int S, const std::vector<double>& a_g,
                                 const std::vector<double>& w_s, const std::vector<double>& c_g,
                                 double notional_total, const SafetyParams& sp) {
  if (G <= 0 || S <= 0) {
    throw std::invalid_argument("compute_safety_state: G and S must be positive");
  }
  if (a_g.size() != static_cast<size_t>(G) || w_s.size() != static_cast<size_t>(S) ||
      c_g.size() != static_cast<size_t>(G)) {
    throw std::invalid_argument("compute_safety_state: vector sizes do not match dimensions");
  }
  if (!(notional_total > 0.0) || !std::isfinite(notional_total)) {
    throw std::invalid_argument("compute_safety_state: notional_total must be finite and > 0");
  }

  double sum_a2 = 0.0;
  for (double v : a_g) {
    sum_a2 += v * v;
  }
  double sum_w2 = 0.0;
  for (double v : w_s) {
    sum_w2 += v * v;
  }

  SafetyState st{};
  st.notional_total = notional_total;
  st.L = std::sqrt(sum_a2 * sum_w2);
  st.eps_dollars = (sp.xva_epsilon_bps / 1.0e4) * notional_total;
  if (st.L <= 0.0 || !std::isfinite(st.L)) {
    st.rho_max = std::numeric_limits<double>::infinity();
  } else {
    st.rho_max = st.eps_dollars / st.L;
  }
  const double margin = std::min(1.0, std::max(0.0, sp.accept_margin));
  const double jl = std::min(0.999999, std::max(0.0, sp.jl_epsilon));
  st.accept_threshold = margin * (1.0 - jl) * st.rho_max;
  return st;
}

void sketch_exposures_from_A_local_f32(const float* A_local, int N, int S, const int* net_id, int G,
                                       const SketchParams& kp, int sketch_id,
                                       std::vector<float>& B_out) {
  sketch_exposures_from_A_local_t<float>(A_local, N, S, net_id, G, kp, sketch_id, B_out);
}

void sketch_exposures_from_A_local_f64(const double* A_local, int N, int S, const int* net_id, int G,
                                       const SketchParams& kp, int sketch_id,
                                       std::vector<double>& B_out) {
  sketch_exposures_from_A_local_t<double>(A_local, N, S, net_id, G, kp, sketch_id, B_out);
}

void sketch_exposures_from_factors_f32(const float* U, const float* V, int N, int S, int r,
                                       const int* net_id, int G, const SketchParams& kp,
                                       int sketch_id, std::vector<float>& Bhat_out) {
  sketch_exposures_from_factors_t<float>(U, V, N, S, r, net_id, G, kp, sketch_id, Bhat_out);
}

void sketch_exposures_from_factors_f64(const double* U, const double* V, int N, int S, int r,
                                       const int* net_id, int G, const SketchParams& kp,
                                       int sketch_id, std::vector<double>& Bhat_out) {
  sketch_exposures_from_factors_t<double>(U, V, N, S, r, net_id, G, kp, sketch_id, Bhat_out);
}

double frob_norm_diff(const std::vector<float>& A, const std::vector<float>& B) {
  return frob_norm_diff_t<float>(A, B);
}

double frob_norm_diff(const std::vector<double>& A, const std::vector<double>& B) {
  return frob_norm_diff_t<double>(A, B);
}

}  // namespace radc
