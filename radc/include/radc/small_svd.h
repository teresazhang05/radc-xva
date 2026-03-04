#pragma once

#include <cstdint>
#include <vector>

namespace radc {

struct SmallSVDResult {
  int l;
  int64_t S;
  std::vector<float> U_l;
  std::vector<float> s;
  std::vector<float> Vt;
  std::vector<float> energy_prefix;
};

struct SmallSVDResultF64 {
  int l;
  int64_t S;
  std::vector<double> U_l;
  std::vector<double> s;
  std::vector<double> Vt;
  std::vector<double> energy_prefix;
};

SmallSVDResult svd_B_and_energy(int l, int64_t S, const std::vector<float>& B_rowmajor);
SmallSVDResultF64 svd_B_and_energy_f64(int l, int64_t S, const std::vector<double>& B_rowmajor);

int select_rank_by_energy(const std::vector<float>& energy_prefix, int r_min, int r_max,
                          float energy_capture, float* out_energy_at_r);
int select_rank_by_energy(const std::vector<double>& energy_prefix, int r_min, int r_max,
                          double energy_capture, double* out_energy_at_r);

}  // namespace radc
