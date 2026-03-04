#include "radc/safety_gate.h"

#include <algorithm>
#include <cmath>
#include <limits>

namespace radc {

SafetyState make_safety_state(double L_xva, double notional_total, const SafetyConfig& cfg) {
  SafetyState st{};
  st.eps_dollars = (cfg.xva_epsilon_bps / 1.0e4) * notional_total;
  if (L_xva <= 0.0 || !std::isfinite(L_xva)) {
    st.rho_max = std::numeric_limits<double>::infinity();
  } else {
    st.rho_max = st.eps_dollars / L_xva;
  }
  return st;
}

bool accept_one_sided(double rho_hat, const SafetyState& st, const SafetyConfig& cfg) {
  const double margin = std::min(1.0, std::max(0.0, cfg.accept_margin));
  const double jl = std::min(0.999999, std::max(0.0, cfg.jl_epsilon));
  const double threshold = margin * (1.0 - jl) * st.rho_max;
  return std::isfinite(rho_hat) && rho_hat <= threshold;
}

}  // namespace radc
