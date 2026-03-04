#include "radc/rand_range_finder.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <limits>
#include <map>
#include <random>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

#if defined(RADC_HAVE_CBLAS)
#if __has_include(<cblas.h>)
#include <cblas.h>
#elif __has_include(<Accelerate/Accelerate.h>)
#include <Accelerate/Accelerate.h>
#else
#error "RADC_HAVE_CBLAS is set but no CBLAS header is available"
#endif
#endif

namespace radc {

namespace {

uint64_t splitmix64(uint64_t x) {
  uint64_t z = x + 0x9E3779B97F4A7C15ULL;
  z = (z ^ (z >> 30U)) * 0xBF58476D1CE4E5B9ULL;
  z = (z ^ (z >> 27U)) * 0x94D049BB133111EBULL;
  return z ^ (z >> 31U);
}

using Clock = std::chrono::steady_clock;

void accumulate_ms(std::map<std::string, double>& t_ms, const std::string& key,
                   const Clock::time_point& t0, const Clock::time_point& t1) {
  const auto us = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
  t_ms[key] += static_cast<double>(us) / 1000.0;
}

template <typename T>
MPI_Datatype mpi_dtype();

template <>
MPI_Datatype mpi_dtype<float>() {
  return MPI_FLOAT;
}

template <>
MPI_Datatype mpi_dtype<double>() {
  return MPI_DOUBLE;
}

template <typename T>
void allreduce_sum_inplace(std::vector<T>& buf, MPI_Comm comm) {
  if (buf.empty()) {
    return;
  }
  if (buf.size() > static_cast<size_t>(std::numeric_limits<int>::max())) {
    throw std::runtime_error("allreduce_sum_inplace: buffer too large for MPI count int");
  }

  const int n = static_cast<int>(buf.size());
  const int rc = PMPI_Allreduce(MPI_IN_PLACE, buf.data(), n, mpi_dtype<T>(), MPI_SUM, comm);
  if (rc != MPI_SUCCESS) {
    throw std::runtime_error("PMPI_Allreduce failed in distributed_qb_allreduce");
  }
}

template <typename T>
void generate_omega(int64_t S, int l, const std::string& omega_dist, uint64_t omega_seed,
                    std::vector<T>& omega) {
  omega.assign(static_cast<size_t>(S) * static_cast<size_t>(l), static_cast<T>(0));

  if (omega_dist == "gaussian") {
    std::mt19937_64 rng(omega_seed);
    std::normal_distribution<double> nd(0.0, 1.0);
    for (T& v : omega) {
      v = static_cast<T>(nd(rng));
    }
    return;
  }

  if (omega_dist == "rademacher") {
    for (int64_t s = 0; s < S; ++s) {
      for (int t = 0; t < l; ++t) {
        const uint64_t h = splitmix64(omega_seed ^ (static_cast<uint64_t>(s) * 0x9E3779B97F4A7C15ULL) ^
                                      static_cast<uint64_t>(t));
        omega[static_cast<size_t>(s) * static_cast<size_t>(l) + static_cast<size_t>(t)] =
            (h & 1ULL) ? static_cast<T>(1) : static_cast<T>(-1);
      }
    }
    return;
  }

  throw std::invalid_argument("distributed_qb_allreduce: omega_dist must be 'rademacher' or 'gaussian'");
}

void sgemm_nn(int M, int N, int K, const float* A, const float* B, float* C) {
#if defined(RADC_HAVE_CBLAS)
  cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, K, 1.0f, A, K, B, N, 0.0f, C, N);
#else
  for (int i = 0; i < M; ++i) {
    for (int j = 0; j < N; ++j) {
      double acc = 0.0;
      for (int k = 0; k < K; ++k) {
        acc += static_cast<double>(A[static_cast<size_t>(i) * static_cast<size_t>(K) + static_cast<size_t>(k)]) *
               static_cast<double>(B[static_cast<size_t>(k) * static_cast<size_t>(N) + static_cast<size_t>(j)]);
      }
      C[static_cast<size_t>(i) * static_cast<size_t>(N) + static_cast<size_t>(j)] = static_cast<float>(acc);
    }
  }
#endif
}

void sgemm_nn(int M, int N, int K, const double* A, const double* B, double* C) {
#if defined(RADC_HAVE_CBLAS)
  cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, K, 1.0, A, K, B, N, 0.0, C, N);
#else
  for (int i = 0; i < M; ++i) {
    for (int j = 0; j < N; ++j) {
      double acc = 0.0;
      for (int k = 0; k < K; ++k) {
        acc += A[static_cast<size_t>(i) * static_cast<size_t>(K) + static_cast<size_t>(k)] *
               B[static_cast<size_t>(k) * static_cast<size_t>(N) + static_cast<size_t>(j)];
      }
      C[static_cast<size_t>(i) * static_cast<size_t>(N) + static_cast<size_t>(j)] = acc;
    }
  }
#endif
}

void sgemm_tn(int M, int N, int K, const float* A, const float* B, float* C) {
#if defined(RADC_HAVE_CBLAS)
  cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans, M, N, K, 1.0f, A, M, B, N, 0.0f, C, N);
#else
  for (int i = 0; i < M; ++i) {
    for (int j = 0; j < N; ++j) {
      double acc = 0.0;
      for (int k = 0; k < K; ++k) {
        acc += static_cast<double>(A[static_cast<size_t>(k) * static_cast<size_t>(M) + static_cast<size_t>(i)]) *
               static_cast<double>(B[static_cast<size_t>(k) * static_cast<size_t>(N) + static_cast<size_t>(j)]);
      }
      C[static_cast<size_t>(i) * static_cast<size_t>(N) + static_cast<size_t>(j)] = static_cast<float>(acc);
    }
  }
#endif
}

void sgemm_tn(int M, int N, int K, const double* A, const double* B, double* C) {
#if defined(RADC_HAVE_CBLAS)
  cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans, M, N, K, 1.0, A, M, B, N, 0.0, C, N);
#else
  for (int i = 0; i < M; ++i) {
    for (int j = 0; j < N; ++j) {
      double acc = 0.0;
      for (int k = 0; k < K; ++k) {
        acc += A[static_cast<size_t>(k) * static_cast<size_t>(M) + static_cast<size_t>(i)] *
               B[static_cast<size_t>(k) * static_cast<size_t>(N) + static_cast<size_t>(j)];
      }
      C[static_cast<size_t>(i) * static_cast<size_t>(N) + static_cast<size_t>(j)] = acc;
    }
  }
#endif
}

template <typename T>
void matmul_A_X(const MatrixView<const T>& A, const std::vector<T>& X, int x_cols,
                std::vector<T>& Y) {
  const int64_t N = A.rows;
  const int64_t S = A.cols;
  if (A.stride != S) {
    throw std::invalid_argument("matmul_A_X: row-major contiguous stride required");
  }
  if (X.size() != static_cast<size_t>(S) * static_cast<size_t>(x_cols)) {
    throw std::invalid_argument("matmul_A_X: X has incompatible size");
  }

  Y.assign(static_cast<size_t>(N) * static_cast<size_t>(x_cols), static_cast<T>(0));
  sgemm_nn(static_cast<int>(N), x_cols, static_cast<int>(S), A.data, X.data(), Y.data());
}

template <typename T>
void matmul_At_X(const MatrixView<const T>& A, const std::vector<T>& X, int x_cols,
                 std::vector<T>& Y) {
  const int64_t N = A.rows;
  const int64_t S = A.cols;
  if (A.stride != S) {
    throw std::invalid_argument("matmul_At_X: row-major contiguous stride required");
  }
  if (X.size() != static_cast<size_t>(N) * static_cast<size_t>(x_cols)) {
    throw std::invalid_argument("matmul_At_X: X has incompatible size");
  }

  Y.assign(static_cast<size_t>(S) * static_cast<size_t>(x_cols), static_cast<T>(0));
  sgemm_tn(static_cast<int>(S), x_cols, static_cast<int>(N), A.data, X.data(), Y.data());
}

template <typename T>
void matmul_Qt_A(const std::vector<T>& Q, int64_t N, int l, const MatrixView<const T>& A,
                 std::vector<T>& B) {
  const int64_t S = A.cols;
  if (A.stride != S) {
    throw std::invalid_argument("matmul_Qt_A: row-major contiguous stride required");
  }
  if (Q.size() != static_cast<size_t>(N) * static_cast<size_t>(l)) {
    throw std::invalid_argument("matmul_Qt_A: Q has incompatible size");
  }

  B.assign(static_cast<size_t>(l) * static_cast<size_t>(S), static_cast<T>(0));
  sgemm_tn(l, static_cast<int>(S), static_cast<int>(N), Q.data(), A.data, B.data());
}

template <typename T>
void orthonormalize_columns(int64_t N, int l, std::vector<T>& M) {
  if (M.size() != static_cast<size_t>(N) * static_cast<size_t>(l)) {
    throw std::invalid_argument("orthonormalize_columns: matrix size mismatch");
  }

  constexpr double kEps = 1.0e-12;

  for (int j = 0; j < l; ++j) {
    for (int pass = 0; pass < 2; ++pass) {
      for (int i = 0; i < j; ++i) {
        double dot = 0.0;
        for (int64_t n = 0; n < N; ++n) {
          dot += static_cast<double>(M[static_cast<size_t>(n) * static_cast<size_t>(l) + static_cast<size_t>(i)]) *
                 static_cast<double>(M[static_cast<size_t>(n) * static_cast<size_t>(l) + static_cast<size_t>(j)]);
        }
        for (int64_t n = 0; n < N; ++n) {
          M[static_cast<size_t>(n) * static_cast<size_t>(l) + static_cast<size_t>(j)] -=
              static_cast<T>(dot) * M[static_cast<size_t>(n) * static_cast<size_t>(l) + static_cast<size_t>(i)];
        }
      }
    }

    double norm2 = 0.0;
    for (int64_t n = 0; n < N; ++n) {
      const double v = static_cast<double>(M[static_cast<size_t>(n) * static_cast<size_t>(l) + static_cast<size_t>(j)]);
      norm2 += v * v;
    }

    if (norm2 <= kEps) {
      for (int64_t n = 0; n < N; ++n) {
        M[static_cast<size_t>(n) * static_cast<size_t>(l) + static_cast<size_t>(j)] = static_cast<T>(0);
      }
      if (N > 0) {
        M[static_cast<size_t>(j % static_cast<int>(N)) * static_cast<size_t>(l) +
          static_cast<size_t>(j)] = static_cast<T>(1);
      }

      for (int i = 0; i < j; ++i) {
        double dot = 0.0;
        for (int64_t n = 0; n < N; ++n) {
          dot += static_cast<double>(M[static_cast<size_t>(n) * static_cast<size_t>(l) + static_cast<size_t>(i)]) *
                 static_cast<double>(M[static_cast<size_t>(n) * static_cast<size_t>(l) + static_cast<size_t>(j)]);
        }
        for (int64_t n = 0; n < N; ++n) {
          M[static_cast<size_t>(n) * static_cast<size_t>(l) + static_cast<size_t>(j)] -=
              static_cast<T>(dot) * M[static_cast<size_t>(n) * static_cast<size_t>(l) + static_cast<size_t>(i)];
        }
      }

      norm2 = 0.0;
      for (int64_t n = 0; n < N; ++n) {
        const double v = static_cast<double>(M[static_cast<size_t>(n) * static_cast<size_t>(l) + static_cast<size_t>(j)]);
        norm2 += v * v;
      }
      if (norm2 <= kEps) {
        continue;
      }
    }

    const double inv_norm = 1.0 / std::sqrt(norm2);
    for (int64_t n = 0; n < N; ++n) {
      M[static_cast<size_t>(n) * static_cast<size_t>(l) + static_cast<size_t>(j)] *=
          static_cast<T>(inv_norm);
    }
  }
}

template <typename T>
struct QBResultT {
  int64_t N;
  int64_t S;
  int l;
  std::vector<T> Q;
  std::vector<T> B;
};

template <typename T>
QBResultT<T> distributed_qb_allreduce_t(MPI_Comm comm, const MatrixView<const T>& A, int l_target,
                                        int power_iters, const std::string& omega_dist,
                                        uint64_t omega_seed,
                                        std::map<std::string, double>& t_ms) {
  if (A.rows <= 0 || A.cols <= 0 || A.stride < A.cols) {
    throw std::invalid_argument("distributed_qb_allreduce: invalid input matrix");
  }
  if (l_target <= 0) {
    throw std::invalid_argument("distributed_qb_allreduce: invalid l_target");
  }
  if (power_iters < 0 || power_iters > 2) {
    throw std::invalid_argument("distributed_qb_allreduce: power_iters must be in [0,2]");
  }

  t_ms["t_compute_Y_ms"] = 0.0;
  t_ms["t_allreduce_Y_ms"] = 0.0;
  t_ms["t_qr_ms"] = 0.0;
  t_ms["t_compute_B_ms"] = 0.0;
  t_ms["t_allreduce_B_ms"] = 0.0;

  const int64_t N = A.rows;
  const int64_t S = A.cols;

  const int64_t l_req_i64 = static_cast<int64_t>(l_target);
  const int l = static_cast<int>(std::min<int64_t>({l_req_i64, N, S}));
  if (l <= 0) {
    throw std::invalid_argument("distributed_qb_allreduce: l collapsed to zero");
  }

  std::vector<T> omega;
  generate_omega<T>(S, l, omega_dist, omega_seed, omega);

  std::vector<T> Y;
  {
    const auto t0 = Clock::now();
    matmul_A_X(A, omega, l, Y);
    const auto t1 = Clock::now();
    accumulate_ms(t_ms, "t_compute_Y_ms", t0, t1);
  }
  {
    const auto t0 = Clock::now();
    allreduce_sum_inplace(Y, comm);
    const auto t1 = Clock::now();
    accumulate_ms(t_ms, "t_allreduce_Y_ms", t0, t1);
  }
  {
    const auto t0 = Clock::now();
    orthonormalize_columns(N, l, Y);
    const auto t1 = Clock::now();
    accumulate_ms(t_ms, "t_qr_ms", t0, t1);
  }

  std::vector<T> Q = Y;

  for (int iter = 0; iter < power_iters; ++iter) {
    std::vector<T> Z;
    {
      const auto t0 = Clock::now();
      matmul_At_X(A, Q, l, Z);
      const auto t1 = Clock::now();
      accumulate_ms(t_ms, "t_compute_B_ms", t0, t1);
    }
    {
      const auto t0 = Clock::now();
      allreduce_sum_inplace(Z, comm);
      const auto t1 = Clock::now();
      accumulate_ms(t_ms, "t_allreduce_B_ms", t0, t1);
    }

    {
      const auto t0 = Clock::now();
      matmul_A_X(A, Z, l, Y);
      const auto t1 = Clock::now();
      accumulate_ms(t_ms, "t_compute_Y_ms", t0, t1);
    }
    {
      const auto t0 = Clock::now();
      allreduce_sum_inplace(Y, comm);
      const auto t1 = Clock::now();
      accumulate_ms(t_ms, "t_allreduce_Y_ms", t0, t1);
    }
    {
      const auto t0 = Clock::now();
      orthonormalize_columns(N, l, Y);
      const auto t1 = Clock::now();
      accumulate_ms(t_ms, "t_qr_ms", t0, t1);
    }

    Q.swap(Y);
  }

  std::vector<T> B;
  {
    const auto t0 = Clock::now();
    matmul_Qt_A(Q, N, l, A, B);
    const auto t1 = Clock::now();
    accumulate_ms(t_ms, "t_compute_B_ms", t0, t1);
  }
  {
    const auto t0 = Clock::now();
    allreduce_sum_inplace(B, comm);
    const auto t1 = Clock::now();
    accumulate_ms(t_ms, "t_allreduce_B_ms", t0, t1);
  }

  return QBResultT<T>{N, S, l, std::move(Q), std::move(B)};
}

}  // namespace

QBResult distributed_qb_allreduce(MPI_Comm comm, const MatrixView<const float>& A, int l_target,
                                  int power_iters,
                                  const std::string& omega_dist, uint64_t omega_seed,
                                  std::map<std::string, double>& t_ms) {
  const QBResultT<float> q =
      distributed_qb_allreduce_t<float>(comm, A, l_target, power_iters, omega_dist, omega_seed, t_ms);
  return QBResult{q.N, q.S, q.l, std::move(q.Q), std::move(q.B)};
}

QBResultF64 distributed_qb_allreduce_f64(MPI_Comm comm, const MatrixView<const double>& A,
                                         int l_target, int power_iters,
                                         const std::string& omega_dist, uint64_t omega_seed,
                                         std::map<std::string, double>& t_ms) {
  const QBResultT<double> q =
      distributed_qb_allreduce_t<double>(comm, A, l_target, power_iters, omega_dist, omega_seed, t_ms);
  return QBResultF64{q.N, q.S, q.l, std::move(q.Q), std::move(q.B)};
}

}  // namespace radc
