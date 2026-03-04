#pragma once

namespace radc {

struct SafetyConfig {
  double xva_epsilon_bps;
  double accept_margin;
  double jl_epsilon;
};

struct SafetyState {
  double rho_max;
  double eps_dollars;
};

SafetyState make_safety_state(double L_xva, double notional_total, const SafetyConfig& cfg);

bool accept_one_sided(double rho_hat, const SafetyState& st, const SafetyConfig& cfg);

}  // namespace radc
