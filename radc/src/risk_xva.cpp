#include "radc/risk_xva.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <random>
#include <stdexcept>

namespace radc {

namespace {

double compute_a_norm(const RiskState& st) {
  double sum_a2 = 0.0;
  for (float a : st.a_g) {
    sum_a2 += static_cast<double>(a) * static_cast<double>(a);
  }

  double sum_w2 = 0.0;
  for (float w : st.w_s) {
    sum_w2 += static_cast<double>(w) * static_cast<double>(w);
  }

  return std::sqrt(sum_a2) * std::sqrt(sum_w2);
}

template <typename T>
double xva_cva_like_typed(const MatrixView<const T>& C, const RiskState& st) {
  if (C.rows != st.N || C.cols != st.S || C.stride < C.cols) {
    throw std::invalid_argument("xva_cva_like: incompatible matrix dimensions");
  }

  std::vector<double> e(static_cast<size_t>(st.G) * static_cast<size_t>(st.S), 0.0);

  for (int n = 0; n < st.N; ++n) {
    const int g = st.netting_of_trade[static_cast<size_t>(n)];
    const T* row = C.data + static_cast<int64_t>(n) * C.stride;
    for (int s = 0; s < st.S; ++s) {
      e[static_cast<size_t>(g) * static_cast<size_t>(st.S) + static_cast<size_t>(s)] +=
          static_cast<double>(row[s]);
    }
  }

  double total = 0.0;
  for (int g = 0; g < st.G; ++g) {
    const double ag = static_cast<double>(st.a_g[static_cast<size_t>(g)]);
    const double H = static_cast<double>(st.H_g[static_cast<size_t>(g)]);
    for (int s = 0; s < st.S; ++s) {
      const double w = static_cast<double>(st.w_s[static_cast<size_t>(s)]);
      const double ex = e[static_cast<size_t>(g) * static_cast<size_t>(st.S) + static_cast<size_t>(s)];
      const double u = std::max(0.0, ex - H);
      total += ag * w * u;
    }
  }
  return total;
}

}  // namespace

RiskState build_risk_state(int N, int S, const RiskConfig& cfg) {
  if (N <= 0 || S <= 0 || cfg.G <= 0) {
    throw std::invalid_argument("build_risk_state: N,S,G must be positive");
  }

  RiskState st{};
  st.N = N;
  st.S = S;
  st.G = cfg.G;
  st.netting_of_trade.resize(static_cast<size_t>(N));
  st.a_g.assign(static_cast<size_t>(cfg.G), cfg.a_scalar);
  st.H_g.assign(static_cast<size_t>(cfg.G), cfg.collateral_threshold);
  st.w_s.assign(static_cast<size_t>(S), 1.0f / static_cast<float>(S));

  std::mt19937_64 rng(cfg.netting_seed);
  std::uniform_int_distribution<int> dist(0, cfg.G - 1);
  for (int n = 0; n < N; ++n) {
    st.netting_of_trade[static_cast<size_t>(n)] = dist(rng);
  }

  st.L_exposure = compute_a_norm(st);
  st.L_xva = std::sqrt(static_cast<double>(N)) * st.L_exposure;
  return st;
}

double xva_cva_like(const MatrixView<const double>& C, const RiskState& st) {
  return xva_cva_like_typed(C, st);
}

double xva_cva_like_f32(const MatrixView<const float>& C, const RiskState& st) {
  return xva_cva_like_typed(C, st);
}

double lipschitz_L_exposure(const RiskState& st) { return st.L_exposure; }

double lipschitz_L_xva(const RiskState& st) { return st.L_xva; }

double xva_error_bps(double xva_true, double xva_approx, double notional_total) {
  if (notional_total <= 0.0) {
    return std::numeric_limits<double>::quiet_NaN();
  }
  return std::abs(xva_true - xva_approx) * 1.0e4 / notional_total;
}

}  // namespace radc
