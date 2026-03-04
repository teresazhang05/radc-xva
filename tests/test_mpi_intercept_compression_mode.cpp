#include <cmath>
#include <fstream>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

#include <mpi.h>

#include "radc/config.h"

namespace {

std::vector<std::string> split_csv_row(const std::string& line) {
  std::vector<std::string> out;
  std::string cur;
  for (char c : line) {
    if (c == ',') {
      out.push_back(cur);
      cur.clear();
    } else {
      cur.push_back(c);
    }
  }
  out.push_back(cur);
  return out;
}

bool starts_with(const std::string& s, const std::string& pfx) {
  return s.size() >= pfx.size() && s.compare(0, pfx.size(), pfx) == 0;
}

int find_col(const std::vector<std::string>& cols, const std::string& name) {
  for (size_t i = 0; i < cols.size(); ++i) {
    if (cols[i] == name) {
      return static_cast<int>(i);
    }
  }
  return -1;
}

}  // namespace

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);

  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  radc::Config cfg{};
  try {
    cfg = radc::load_config_from_env_or_default("");
  } catch (const std::exception& e) {
    if (rank == 0) {
      std::cerr << "Failed to load config: " << e.what() << std::endl;
    }
    MPI_Finalize();
    return 2;
  }

  const int64_t count64 = cfg.buffer.N * cfg.buffer.S;
  if (count64 <= 0 || count64 > static_cast<int64_t>(std::numeric_limits<int>::max())) {
    if (rank == 0) {
      std::cerr << "Invalid test N*S count" << std::endl;
    }
    MPI_Finalize();
    return 3;
  }
  const int count = static_cast<int>(count64);

  std::vector<double> send(static_cast<size_t>(count), 0.0);
  std::vector<double> recv(static_cast<size_t>(count), 0.0);
  std::vector<double> exact(static_cast<size_t>(count), 0.0);

  for (int i = 0; i < count; ++i) {
    send[static_cast<size_t>(i)] =
        static_cast<double>(rank + 1) + 0.001 * static_cast<double>(i + 1);
  }

  if (MPI_Allreduce(send.data(), recv.data(), count, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD) != MPI_SUCCESS) {
    if (rank == 0) {
      std::cerr << "MPI_Allreduce failed through wrapper" << std::endl;
    }
    MPI_Finalize();
    return 4;
  }

  if (PMPI_Allreduce(send.data(), exact.data(), count, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD) != MPI_SUCCESS) {
    if (rank == 0) {
      std::cerr << "PMPI_Allreduce exact reference failed" << std::endl;
    }
    MPI_Finalize();
    return 5;
  }

  for (int i = 0; i < count; ++i) {
    if (!std::isfinite(recv[static_cast<size_t>(i)])) {
      if (rank == 0) {
        std::cerr << "Received non-finite value at i=" << i << std::endl;
      }
      MPI_Finalize();
      return 6;
    }
  }

  MPI_Barrier(MPI_COMM_WORLD);

  if (rank == 0) {
    const std::string metrics_path = cfg.run.output_dir + "/metrics_rank0.csv";
    std::ifstream in(metrics_path);
    if (!in.is_open()) {
      std::cerr << "Could not open metrics file: " << metrics_path << std::endl;
      MPI_Finalize();
      return 7;
    }

    std::string header;
    if (!std::getline(in, header)) {
      std::cerr << "Metrics file missing header: " << metrics_path << std::endl;
      MPI_Finalize();
      return 8;
    }

    const auto cols = split_csv_row(header);
    const int mode_idx = find_col(cols, "mode");
    const int dtype_internal_idx = find_col(cols, "dtype_internal");
    const int downcast_used_idx = find_col(cols, "downcast_used");

    if (mode_idx < 0 || dtype_internal_idx < 0 || downcast_used_idx < 0) {
      std::cerr << "Metrics header missing one of required columns: mode,dtype_internal,downcast_used"
                << std::endl;
      MPI_Finalize();
      return 9;
    }

    std::string line;
    std::string last_line;
    while (std::getline(in, line)) {
      if (!line.empty()) {
        last_line = line;
      }
    }

    if (last_line.empty()) {
      std::cerr << "Metrics file has no data rows" << std::endl;
      MPI_Finalize();
      return 10;
    }

    const auto row = split_csv_row(last_line);
    if (mode_idx >= static_cast<int>(row.size()) ||
        dtype_internal_idx >= static_cast<int>(row.size()) ||
        downcast_used_idx >= static_cast<int>(row.size())) {
      std::cerr << "Malformed data row in metrics file" << std::endl;
      MPI_Finalize();
      return 11;
    }

    const std::string mode = row[static_cast<size_t>(mode_idx)];
    const std::string dtype_internal = row[static_cast<size_t>(dtype_internal_idx)];
    const std::string downcast_used = row[static_cast<size_t>(downcast_used_idx)];

    std::string double_mode = cfg.compression.double_mode;
    if (double_mode.empty()) {
      double_mode = "native64";
    }

    if (double_mode == "passthrough") {
      if (mode != "exact") {
        std::cerr << "Expected exact mode for passthrough, got: " << mode << std::endl;
        MPI_Finalize();
        return 12;
      }
      if (dtype_internal != "float64" || downcast_used != "0") {
        std::cerr << "Passthrough expected dtype_internal=float64 and downcast_used=0, got "
                  << dtype_internal << " / " << downcast_used << std::endl;
        MPI_Finalize();
        return 13;
      }
    } else {
      if (!starts_with(mode, "compressed_")) {
        std::cerr << "Expected compressed_* mode, got: " << mode << std::endl;
        MPI_Finalize();
        return 14;
      }
      if (double_mode == "downcast32") {
        if (dtype_internal != "float32" || downcast_used != "1") {
          std::cerr << "Downcast expected dtype_internal=float32 and downcast_used=1, got "
                    << dtype_internal << " / " << downcast_used << std::endl;
          MPI_Finalize();
          return 15;
        }
      } else if (double_mode == "native64") {
        if (dtype_internal != "float64" || downcast_used != "0") {
          std::cerr << "Native64 expected dtype_internal=float64 and downcast_used=0, got "
                    << dtype_internal << " / " << downcast_used << std::endl;
          MPI_Finalize();
          return 16;
        }
      } else {
        std::cerr << "Unexpected compression.double_mode in test: " << double_mode << std::endl;
        MPI_Finalize();
        return 17;
      }
    }
  }

  MPI_Finalize();
  return 0;
}
