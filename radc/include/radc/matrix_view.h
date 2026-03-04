#pragma once

#include <cstdint>

namespace radc {

template <typename T>
struct MatrixView {
  T* data;
  int64_t rows;
  int64_t cols;
  int64_t stride;
};

}  // namespace radc
