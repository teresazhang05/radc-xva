#pragma once

#include <cstdint>
#include <vector>

#include "radc/matrix_view.h"

namespace radc {

struct CountSketch2D {
  int k_row;
  int k_col;
  uint64_t hash_seed_row;
  uint64_t hash_seed_col;
  uint64_t sign_seed_row;
  uint64_t sign_seed_col;

  void sketch_matrix_f64_to_f32(const MatrixView<const double>& A, std::vector<float>& K_out) const;

  // Sketches netted exposure E = M*A (GxS) without materializing E.
  void sketch_matrix_f64_to_f32_netting(const MatrixView<const double>& A,
                                        const std::vector<int>& netting_of_trade, int G,
                                        std::vector<float>& K_out) const;

  void sketch_qb_f32(int64_t N, int64_t S, int l, const std::vector<float>& Q,
                     const std::vector<float>& B, std::vector<float>& Khat_out) const;

  double frob_norm_diff(const std::vector<float>& K, const std::vector<float>& Khat) const;
};

}  // namespace radc
