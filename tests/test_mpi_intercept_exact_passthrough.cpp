#include <cmath>
#include <fstream>
#include <iostream>
#include <limits>
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

int find_col(const std::vector<std::string>& cols, const std::string& name) {
  for (size_t i = 0; i < cols.size(); ++i) {
    if (cols[i] == name) {
      return static_cast<int>(i);
    }
  }
  return -1;
}

double parse_double_or_nan(const std::string& s) {
  if (s.empty() || s == "NaN" || s == "nan") {
    return std::numeric_limits<double>::quiet_NaN();
  }
  try {
    return std::stod(s);
  } catch (...) {
    return std::numeric_limits<double>::quiet_NaN();
  }
}

}  // namespace

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);

  int rank = 0;
  int world = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &world);

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

  std::vector<float> send(static_cast<size_t>(count), 0.0f);
  std::vector<float> recv(static_cast<size_t>(count), 0.0f);
  std::vector<float> exact(static_cast<size_t>(count), 0.0f);

  for (int i = 0; i < count; ++i) {
    send[static_cast<size_t>(i)] =
        static_cast<float>(0.01 * static_cast<double>(i + 1) + static_cast<double>(rank + 1));
  }

  if (MPI_Allreduce(send.data(), recv.data(), count, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD) != MPI_SUCCESS) {
    if (rank == 0) {
      std::cerr << "MPI_Allreduce failed through wrapper" << std::endl;
    }
    MPI_Finalize();
    return 4;
  }

  if (PMPI_Allreduce(send.data(), exact.data(), count, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD) != MPI_SUCCESS) {
    if (rank == 0) {
      std::cerr << "PMPI_Allreduce exact reference failed" << std::endl;
    }
    MPI_Finalize();
    return 5;
  }

  for (int i = 0; i < count; ++i) {
    const float got = recv[static_cast<size_t>(i)];
    const float ref = exact[static_cast<size_t>(i)];
    if (!std::isfinite(got) || std::fabs(static_cast<double>(got - ref)) > 1e-6) {
      if (rank == 0) {
        std::cerr << "Mismatch at i=" << i << " got=" << got << " ref=" << ref << std::endl;
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
      std::cerr << "Metrics file missing header" << std::endl;
      MPI_Finalize();
      return 8;
    }
    const auto cols = split_csv_row(header);
    const int mode_idx = find_col(cols, "mode");
    const int bytes_exact_idx = find_col(cols, "bytes_exact_payload");
    const int t_epoch_idx = find_col(cols, "t_epoch_total_ms");
    if (mode_idx < 0 || bytes_exact_idx < 0 || t_epoch_idx < 0) {
      std::cerr << "Metrics header missing required columns" << std::endl;
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
    if (mode_idx >= static_cast<int>(row.size()) || bytes_exact_idx >= static_cast<int>(row.size()) ||
        t_epoch_idx >= static_cast<int>(row.size())) {
      std::cerr << "Malformed metrics row" << std::endl;
      MPI_Finalize();
      return 11;
    }

    if (row[static_cast<size_t>(mode_idx)] != "exact") {
      std::cerr << "Expected exact mode for pass-through baseline, got: "
                << row[static_cast<size_t>(mode_idx)] << std::endl;
      MPI_Finalize();
      return 12;
    }

    const double bytes_exact = parse_double_or_nan(row[static_cast<size_t>(bytes_exact_idx)]);
    const double expected_bytes = static_cast<double>(count) * 4.0;
    if (!std::isfinite(bytes_exact) || std::fabs(bytes_exact - expected_bytes) > 0.5) {
      std::cerr << "Unexpected bytes_exact_payload: got=" << bytes_exact
                << " expected=" << expected_bytes << std::endl;
      MPI_Finalize();
      return 13;
    }

    const double t_epoch_ms = parse_double_or_nan(row[static_cast<size_t>(t_epoch_idx)]);
    if (!std::isfinite(t_epoch_ms) || t_epoch_ms <= 0.0) {
      std::cerr << "Non-positive t_epoch_total_ms: " << t_epoch_ms << std::endl;
      MPI_Finalize();
      return 14;
    }
  }

  MPI_Finalize();
  return 0;
}
