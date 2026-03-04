#include <cmath>
#include <iostream>

#include "radc/safety_gate.h"

int main() {
  radc::SafetyConfig cfg{};
  cfg.xva_epsilon_bps = 0.01;
  cfg.accept_margin = 0.90;
  cfg.jl_epsilon = 0.10;

  const double L_xva = 5.0;
  const double notional = 1.0e9;

  const radc::SafetyState st = radc::make_safety_state(L_xva, notional, cfg);

  const double eps_dollars_expected = (cfg.xva_epsilon_bps / 1.0e4) * notional;
  const double rho_max_expected = eps_dollars_expected / L_xva;

  if (std::abs(st.eps_dollars - eps_dollars_expected) > 1e-9) {
    std::cerr << "eps_dollars mismatch" << std::endl;
    return 1;
  }
  if (std::abs(st.rho_max - rho_max_expected) > 1e-9) {
    std::cerr << "rho_max mismatch" << std::endl;
    return 1;
  }

  const double threshold = cfg.accept_margin * (1.0 - cfg.jl_epsilon) * st.rho_max;
  if (!radc::accept_one_sided(threshold, st, cfg)) {
    std::cerr << "threshold should accept" << std::endl;
    return 1;
  }
  if (radc::accept_one_sided(threshold * 1.000001, st, cfg)) {
    std::cerr << "value above threshold should reject" << std::endl;
    return 1;
  }

  return 0;
}
