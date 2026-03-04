#include "radc/timer.h"

#include <chrono>

namespace radc {

uint64_t now_unix_ns() {
  const auto now = std::chrono::system_clock::now();
  const auto ns = std::chrono::duration_cast<std::chrono::nanoseconds>(now.time_since_epoch());
  return static_cast<uint64_t>(ns.count());
}

}  // namespace radc
