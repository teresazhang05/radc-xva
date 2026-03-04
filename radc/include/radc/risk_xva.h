#pragma once

#include <cstdint>
#include <vector>

#include "radc/matrix_view.h"

namespace radc {

struct RiskConfig {
  int G;
  uint64_t netting_seed;
  float collateral_threshold;
  float a_scalar;
  bool scenario_weights_uniform;
  double notional_total;
};

struct RiskState {
  int N;
  int S;
  int G;
  std::vector<int> netting_of_trade;
  std::vector<float> a_g;
  std::vector<float> w_s;
  std::vector<float> H_g;
  double L_exposure;
  double L_xva;
};

RiskState build_risk_state(int N, int S, const RiskConfig& cfg);

double xva_cva_like(const MatrixView<const double>& C, const RiskState& st);
double xva_cva_like_f32(const MatrixView<const float>& C, const RiskState& st);

double lipschitz_L_exposure(const RiskState& st);

double lipschitz_L_xva(const RiskState& st);

double xva_error_bps(double xva_true, double xva_approx, double notional_total);

}  // namespace radc
