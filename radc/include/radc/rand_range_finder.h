#pragma once

#include <cstdint>
#include <map>
#include <string>
#include <vector>

#include <mpi.h>

#include "radc/matrix_view.h"

namespace radc {

struct QBResult {
  int64_t N;
  int64_t S;
  int l;
  std::vector<float> Q;
  std::vector<float> B;
};

struct QBResultF64 {
  int64_t N;
  int64_t S;
  int l;
  std::vector<double> Q;
  std::vector<double> B;
};

QBResult distributed_qb_allreduce(MPI_Comm comm, const MatrixView<const float>& A, int l_target,
                                  int power_iters,
                                  const std::string& omega_dist, uint64_t omega_seed,
                                  std::map<std::string, double>& t_ms);

QBResultF64 distributed_qb_allreduce_f64(MPI_Comm comm, const MatrixView<const double>& A, int l_target,
                                         int power_iters, const std::string& omega_dist,
                                         uint64_t omega_seed, std::map<std::string, double>& t_ms);

}  // namespace radc
