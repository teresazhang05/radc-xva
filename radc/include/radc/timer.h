#pragma once

#include <chrono>

namespace radc {

class ScopedTimer {
 public:
  explicit ScopedTimer(double* out_ms)
      : out_ms_(out_ms), start_(std::chrono::steady_clock::now()) {}

  ~ScopedTimer() {
    if (out_ms_ == nullptr) {
      return;
    }
    const auto end = std::chrono::steady_clock::now();
    const auto us = std::chrono::duration_cast<std::chrono::microseconds>(end - start_);
    *out_ms_ = static_cast<double>(us.count()) / 1000.0;
  }

 private:
  double* out_ms_;
  std::chrono::steady_clock::time_point start_;
};

uint64_t now_unix_ns();

}  // namespace radc
