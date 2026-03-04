#include <cmath>
#include <cstdlib>
#include <iostream>
#include <vector>

#include <mpi.h>

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);

  int rank = 0;
  int world = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &world);

  const int count = 32;
  std::vector<double> send(static_cast<size_t>(count), 0.0);
  std::vector<double> recv(static_cast<size_t>(count), 0.0);

  for (int i = 0; i < count; ++i) {
    send[static_cast<size_t>(i)] = static_cast<double>(rank + 1) + 0.01 * static_cast<double>(i + 1);
  }

  int rc = MPI_Allreduce(send.data(), recv.data(), count, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  if (rc != MPI_SUCCESS) {
    if (rank == 0) {
      std::cerr << "MPI_Allreduce failed" << std::endl;
    }
    MPI_Finalize();
    return 2;
  }

  const double expected_base = static_cast<double>(world * (world + 1)) / 2.0;
  for (int i = 0; i < count; ++i) {
    const double expected = expected_base + static_cast<double>(world) * 0.01 * static_cast<double>(i + 1);
    if (std::abs(recv[static_cast<size_t>(i)] - expected) > 1e-12) {
      std::cerr << "Mismatch at i=" << i << " got=" << recv[static_cast<size_t>(i)]
                << " expected=" << expected << std::endl;
      MPI_Finalize();
      return 3;
    }
  }

  for (int i = 0; i < count; ++i) {
    recv[static_cast<size_t>(i)] = static_cast<double>(rank + 1) + 0.02 * static_cast<double>(i + 1);
  }

  rc = MPI_Allreduce(MPI_IN_PLACE, recv.data(), count, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  if (rc != MPI_SUCCESS) {
    if (rank == 0) {
      std::cerr << "MPI_Allreduce MPI_IN_PLACE failed" << std::endl;
    }
    MPI_Finalize();
    return 4;
  }

  for (int i = 0; i < count; ++i) {
    const double expected = expected_base + static_cast<double>(world) * 0.02 * static_cast<double>(i + 1);
    if (std::abs(recv[static_cast<size_t>(i)] - expected) > 1e-12) {
      std::cerr << "MPI_IN_PLACE mismatch at i=" << i << " got=" << recv[static_cast<size_t>(i)]
                << " expected=" << expected << std::endl;
      MPI_Finalize();
      return 5;
    }
  }

  MPI_Finalize();
  return 0;
}
