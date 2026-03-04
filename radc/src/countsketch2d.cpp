#include "radc/countsketch2d.h"

#include <cmath>
#include <stdexcept>

namespace radc {

namespace {

uint64_t splitmix64(uint64_t x) {
  uint64_t z = x + 0x9E3779B97F4A7C15ULL;
  z = (z ^ (z >> 30U)) * 0xBF58476D1CE4E5B9ULL;
  z = (z ^ (z >> 27U)) * 0x94D049BB133111EBULL;
  return z ^ (z >> 31U);
}

int hash_bucket(uint64_t seed, int64_t idx, int buckets) {
  const uint64_t h = splitmix64(seed ^ static_cast<uint64_t>(idx));
  return static_cast<int>(h % static_cast<uint64_t>(buckets));
}

float hash_sign(uint64_t seed, int64_t idx) {
  const uint64_t h = splitmix64(seed ^ static_cast<uint64_t>(idx));
  return (h & 1ULL) ? 1.0f : -1.0f;
}

}  // namespace

void CountSketch2D::sketch_matrix_f64_to_f32(const MatrixView<const double>& A,
                                             std::vector<float>& K_out) const {
  if (k_row <= 0 || k_col <= 0) {
    throw std::invalid_argument("CountSketch2D: k_row and k_col must be positive");
  }
  if (A.stride < A.cols) {
    throw std::invalid_argument("CountSketch2D: invalid matrix stride");
  }

  K_out.assign(static_cast<size_t>(k_row) * static_cast<size_t>(k_col), 0.0f);

  for (int64_t n = 0; n < A.rows; ++n) {
    const int r = hash_bucket(hash_seed_row, n, k_row);
    const float sr = hash_sign(sign_seed_row, n);
    const double* row = A.data + n * A.stride;
    for (int64_t s = 0; s < A.cols; ++s) {
      const int c = hash_bucket(hash_seed_col, s, k_col);
      const float sc = hash_sign(sign_seed_col, s);
      const float v = static_cast<float>(row[s]);
      K_out[static_cast<size_t>(r) * static_cast<size_t>(k_col) + static_cast<size_t>(c)] +=
          sr * sc * v;
    }
  }
}

void CountSketch2D::sketch_matrix_f64_to_f32_netting(const MatrixView<const double>& A,
                                                     const std::vector<int>& netting_of_trade, int G,
                                                     std::vector<float>& K_out) const {
  if (k_row <= 0 || k_col <= 0 || G <= 0) {
    throw std::invalid_argument(
        "CountSketch2D::sketch_matrix_f64_to_f32_netting: invalid sketch/netting dims");
  }
  if (A.stride < A.cols) {
    throw std::invalid_argument("CountSketch2D::sketch_matrix_f64_to_f32_netting: invalid matrix stride");
  }
  if (netting_of_trade.size() != static_cast<size_t>(A.rows)) {
    throw std::invalid_argument("CountSketch2D::sketch_matrix_f64_to_f32_netting: netting size mismatch");
  }

  K_out.assign(static_cast<size_t>(k_row) * static_cast<size_t>(k_col), 0.0f);

  for (int64_t n = 0; n < A.rows; ++n) {
    const int g = netting_of_trade[static_cast<size_t>(n)];
    if (g < 0 || g >= G) {
      throw std::invalid_argument("CountSketch2D::sketch_matrix_f64_to_f32_netting: out-of-range netting id");
    }
    const int r = hash_bucket(hash_seed_row, g, k_row);
    const float sr = hash_sign(sign_seed_row, g);
    const double* row = A.data + n * A.stride;
    for (int64_t s = 0; s < A.cols; ++s) {
      const int c = hash_bucket(hash_seed_col, s, k_col);
      const float sc = hash_sign(sign_seed_col, s);
      const float v = static_cast<float>(row[s]);
      K_out[static_cast<size_t>(r) * static_cast<size_t>(k_col) + static_cast<size_t>(c)] +=
          sr * sc * v;
    }
  }
}

void CountSketch2D::sketch_qb_f32(int64_t N, int64_t S, int l, const std::vector<float>& Q,
                                  const std::vector<float>& B, std::vector<float>& Khat_out) const {
  if (k_row <= 0 || k_col <= 0 || l <= 0 || N <= 0 || S <= 0) {
    throw std::invalid_argument("CountSketch2D::sketch_qb_f32: invalid dimensions");
  }
  if (Q.size() != static_cast<size_t>(N) * static_cast<size_t>(l)) {
    throw std::invalid_argument("CountSketch2D::sketch_qb_f32: invalid Q size");
  }
  if (B.size() != static_cast<size_t>(l) * static_cast<size_t>(S)) {
    throw std::invalid_argument("CountSketch2D::sketch_qb_f32: invalid B size");
  }

  std::vector<float> SNQ(static_cast<size_t>(k_row) * static_cast<size_t>(l), 0.0f);
  for (int64_t n = 0; n < N; ++n) {
    const int r = hash_bucket(hash_seed_row, n, k_row);
    const float sr = hash_sign(sign_seed_row, n);
    const float* q_row = Q.data() + static_cast<size_t>(n) * static_cast<size_t>(l);
    for (int t = 0; t < l; ++t) {
      SNQ[static_cast<size_t>(r) * static_cast<size_t>(l) + static_cast<size_t>(t)] +=
          sr * q_row[t];
    }
  }

  std::vector<float> BSS(static_cast<size_t>(l) * static_cast<size_t>(k_col), 0.0f);
  for (int64_t s = 0; s < S; ++s) {
    const int c = hash_bucket(hash_seed_col, s, k_col);
    const float sc = hash_sign(sign_seed_col, s);
    for (int t = 0; t < l; ++t) {
      BSS[static_cast<size_t>(t) * static_cast<size_t>(k_col) + static_cast<size_t>(c)] +=
          B[static_cast<size_t>(t) * static_cast<size_t>(S) + static_cast<size_t>(s)] * sc;
    }
  }

  Khat_out.assign(static_cast<size_t>(k_row) * static_cast<size_t>(k_col), 0.0f);
  for (int r = 0; r < k_row; ++r) {
    for (int c = 0; c < k_col; ++c) {
      float acc = 0.0f;
      for (int t = 0; t < l; ++t) {
        acc += SNQ[static_cast<size_t>(r) * static_cast<size_t>(l) + static_cast<size_t>(t)] *
               BSS[static_cast<size_t>(t) * static_cast<size_t>(k_col) + static_cast<size_t>(c)];
      }
      Khat_out[static_cast<size_t>(r) * static_cast<size_t>(k_col) + static_cast<size_t>(c)] = acc;
    }
  }
}

double CountSketch2D::frob_norm_diff(const std::vector<float>& K,
                                     const std::vector<float>& Khat) const {
  if (K.size() != Khat.size()) {
    throw std::invalid_argument("CountSketch2D::frob_norm_diff: size mismatch");
  }
  double sum = 0.0;
  for (size_t i = 0; i < K.size(); ++i) {
    const double d = static_cast<double>(K[i]) - static_cast<double>(Khat[i]);
    sum += d * d;
  }
  return std::sqrt(sum);
}

}  // namespace radc
