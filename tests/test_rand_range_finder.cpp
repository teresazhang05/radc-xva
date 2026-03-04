#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <map>
#include <string>
#include <vector>

#include <mpi.h>

#include "radc/matrix_view.h"
#include "radc/rand_range_finder.h"

namespace {

template <typename T>
double frob_norm(const std::vector<T>& A) {
  double sum = 0.0;
  for (T v : A) {
    const double d = static_cast<double>(v);
    sum += d * d;
  }
  return std::sqrt(sum);
}

}  // namespace

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);

  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  const int64_t N = 32;
  const int64_t S = 24;
  const int r_true = 3;

  std::vector<float> A_local(static_cast<size_t>(N) * static_cast<size_t>(S), 0.0f);
  for (int64_t n = 0; n < N; ++n) {
    for (int64_t s = 0; s < S; ++s) {
      double v = 0.0;
      for (int k = 0; k < r_true; ++k) {
        const double u = std::sin((static_cast<double>(n) + 1.0) * (k + 1.0) *
                                  (static_cast<double>(rank) + 1.0) * 0.11);
        const double w = std::cos((static_cast<double>(s) + 1.0) * (k + 1.0) * 0.07);
        v += u * w;
      }
      A_local[static_cast<size_t>(n) * static_cast<size_t>(S) + static_cast<size_t>(s)] =
          static_cast<float>(v);
    }
  }

  radc::MatrixView<const float> A_view{A_local.data(), N, S, S};
  std::map<std::string, double> t_ms;
  const radc::QBResult qb =
      radc::distributed_qb_allreduce(MPI_COMM_WORLD, A_view, 7, 1, "rademacher", 424242ULL, t_ms);

  if (qb.N != N || qb.S != S || qb.l <= 0) {
    if (rank == 0) {
      std::cerr << "QB dimensions are invalid" << std::endl;
    }
    MPI_Finalize();
    return 1;
  }

  std::vector<float> A_global = A_local;
  const int count = static_cast<int>(A_global.size());
  if (PMPI_Allreduce(MPI_IN_PLACE, A_global.data(), count, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD) != MPI_SUCCESS) {
    if (rank == 0) {
      std::cerr << "PMPI_Allreduce for A_global failed" << std::endl;
    }
    MPI_Finalize();
    return 1;
  }

  std::vector<double> A_hat(static_cast<size_t>(N) * static_cast<size_t>(S), 0.0);
  for (int64_t n = 0; n < N; ++n) {
    for (int64_t s = 0; s < S; ++s) {
      double acc = 0.0;
      for (int t = 0; t < qb.l; ++t) {
        const double q = static_cast<double>(qb.Q[static_cast<size_t>(n) * static_cast<size_t>(qb.l) +
                                                static_cast<size_t>(t)]);
        const double b = static_cast<double>(qb.B[static_cast<size_t>(t) * static_cast<size_t>(S) +
                                                static_cast<size_t>(s)]);
        acc += q * b;
      }
      A_hat[static_cast<size_t>(n) * static_cast<size_t>(S) + static_cast<size_t>(s)] = acc;
    }
  }

  std::vector<double> diff(A_hat.size(), 0.0);
  for (size_t i = 0; i < diff.size(); ++i) {
    diff[i] = A_global[i] - A_hat[i];
  }

  const double rel_err = frob_norm(diff) / std::max(1e-12, frob_norm(A_global));
  if (rel_err > 5e-2) {
    if (rank == 0) {
      std::cerr << "QB relative reconstruction error too high: " << rel_err << std::endl;
    }
    MPI_Finalize();
    return 1;
  }

  double max_orth_err = 0.0;
  for (int i = 0; i < qb.l; ++i) {
    for (int j = 0; j < qb.l; ++j) {
      double dot = 0.0;
      for (int64_t n = 0; n < N; ++n) {
        dot += static_cast<double>(qb.Q[static_cast<size_t>(n) * static_cast<size_t>(qb.l) +
                                        static_cast<size_t>(i)]) *
               static_cast<double>(qb.Q[static_cast<size_t>(n) * static_cast<size_t>(qb.l) +
                                        static_cast<size_t>(j)]);
      }
      const double target = (i == j) ? 1.0 : 0.0;
      max_orth_err = std::max(max_orth_err, std::abs(dot - target));
    }
  }

  if (max_orth_err > 5e-2) {
    if (rank == 0) {
      std::cerr << "Q orthonormality check failed: max error=" << max_orth_err << std::endl;
    }
    MPI_Finalize();
    return 1;
  }

  const std::vector<std::string> required_keys = {
      "t_compute_Y_ms", "t_allreduce_Y_ms", "t_qr_ms", "t_compute_B_ms", "t_allreduce_B_ms"};
  for (const auto& k : required_keys) {
    if (t_ms.find(k) == t_ms.end()) {
      if (rank == 0) {
        std::cerr << "Missing timing key: " << k << std::endl;
      }
      MPI_Finalize();
      return 1;
    }
  }

  MPI_Finalize();
  return 0;
}
