#pragma once

#include <cstdint>

namespace radc {

void set_current_epoch(int64_t epoch);
int64_t current_epoch();

}  // namespace radc

extern "C" {
void radc_set_epoch(int64_t epoch);
}
