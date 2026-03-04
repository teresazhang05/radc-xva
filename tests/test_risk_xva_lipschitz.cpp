#include <cmath>
#include <iostream>
#include <random>
#include <vector>

#include "radc/risk_xva.h"

int main() {
  const int N = 32;
  const int S = 64;

  radc::RiskConfig cfg{};
  cfg.G = 8;
  cfg.netting_seed = 202;
  cfg.collateral_threshold = 0.0f;
  cfg.a_scalar = 1.0f;
  cfg.scenario_weights_uniform = true;
  cfg.notional_total = 1.0e9;

  const radc::RiskState st = radc::build_risk_state(N, S, cfg);

  std::vector<double> C(static_cast<size_t>(N) * static_cast<size_t>(S), 0.0);
  std::vector<double> E(static_cast<size_t>(N) * static_cast<size_t>(S), 0.0);
  std::vector<double> Chat(static_cast<size_t>(N) * static_cast<size_t>(S), 0.0);

  std::mt19937_64 rng(12345);
  std::normal_distribution<double> nd(0.0, 1.0);

  for (size_t i = 0; i < C.size(); ++i) {
    C[i] = nd(rng);
    E[i] = 0.01 * nd(rng);
    Chat[i] = C[i] + E[i];
  }

  const radc::MatrixView<const double> C_view{C.data(), N, S, S};
  const radc::MatrixView<const double> Chat_view{Chat.data(), N, S, S};

  const double xva_c = radc::xva_cva_like(C_view, st);
  const double xva_chat = radc::xva_cva_like(Chat_view, st);
  const double diff = std::abs(xva_c - xva_chat);

  double e_frob = 0.0;
  for (double v : E) {
    e_frob += v * v;
  }
  e_frob = std::sqrt(e_frob);

  const double rhs = radc::lipschitz_L_xva(st) * e_frob;
  if (diff > rhs + 1e-9) {
    std::cerr << "Lipschitz bound failed: diff=" << diff << " rhs=" << rhs << std::endl;
    return 1;
  }

  return 0;
}
