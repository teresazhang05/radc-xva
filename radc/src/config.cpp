#include "radc/config.h"

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace radc {

namespace {

struct StackItem {
  int indent;
  std::string path;
};

std::string ltrim(const std::string& s) {
  size_t i = 0;
  while (i < s.size() && (s[i] == ' ' || s[i] == '\t' || s[i] == '\r' || s[i] == '\n')) {
    ++i;
  }
  return s.substr(i);
}

std::string rtrim(const std::string& s) {
  size_t end = s.size();
  while (end > 0) {
    const char c = s[end - 1];
    if (c == ' ' || c == '\t' || c == '\r' || c == '\n') {
      --end;
      continue;
    }
    break;
  }
  return s.substr(0, end);
}

std::string trim(const std::string& s) { return rtrim(ltrim(s)); }

std::string strip_comments(const std::string& line) {
  bool in_single = false;
  bool in_double = false;
  for (size_t i = 0; i < line.size(); ++i) {
    const char c = line[i];
    if (c == '\'' && !in_double) {
      in_single = !in_single;
      continue;
    }
    if (c == '"' && !in_single) {
      in_double = !in_double;
      continue;
    }
    if (c == '#' && !in_single && !in_double) {
      return line.substr(0, i);
    }
  }
  return line;
}

std::string unquote(const std::string& s) {
  if (s.size() >= 2) {
    if ((s.front() == '"' && s.back() == '"') || (s.front() == '\'' && s.back() == '\'')) {
      return s.substr(1, s.size() - 2);
    }
  }
  return s;
}

bool parse_bool(const std::string& raw, bool default_value) {
  const std::string v = trim(raw);
  if (v == "true" || v == "True" || v == "1") {
    return true;
  }
  if (v == "false" || v == "False" || v == "0") {
    return false;
  }
  return default_value;
}

int parse_int(const std::string& raw, int default_value) {
  try {
    return std::stoi(trim(raw));
  } catch (...) {
    return default_value;
  }
}

int64_t parse_i64(const std::string& raw, int64_t default_value) {
  try {
    return std::stoll(trim(raw));
  } catch (...) {
    return default_value;
  }
}

uint64_t parse_u64(const std::string& raw, uint64_t default_value) {
  try {
    return static_cast<uint64_t>(std::stoull(trim(raw)));
  } catch (...) {
    return default_value;
  }
}

double parse_double(const std::string& raw, double default_value) {
  try {
    return std::stod(trim(raw));
  } catch (...) {
    return default_value;
  }
}

std::unordered_map<std::string, std::string> parse_yaml_subset(const std::string& path) {
  std::ifstream in(path);
  if (!in.is_open()) {
    throw std::runtime_error("Could not open config file: " + path);
  }

  std::unordered_map<std::string, std::string> out;
  std::vector<StackItem> stack;

  std::string line;
  while (std::getline(in, line)) {
    line = strip_comments(line);
    line = rtrim(line);
    if (trim(line).empty()) {
      continue;
    }

    int indent = 0;
    while (indent < static_cast<int>(line.size()) && line[static_cast<size_t>(indent)] == ' ') {
      ++indent;
    }

    const std::string content = trim(line.substr(static_cast<size_t>(indent)));
    if (content.empty()) {
      continue;
    }

    const size_t colon = content.find(':');
    if (colon == std::string::npos) {
      continue;
    }

    std::string key = trim(content.substr(0, colon));
    std::string value = trim(content.substr(colon + 1));

    while (!stack.empty() && indent <= stack.back().indent) {
      stack.pop_back();
    }

    std::string full_key;
    if (stack.empty()) {
      full_key = key;
    } else {
      full_key = stack.back().path + "." + key;
    }

    if (value.empty()) {
      stack.push_back(StackItem{indent, full_key});
      continue;
    }

    out[full_key] = unquote(value);
  }

  return out;
}

std::string get_str(const std::unordered_map<std::string, std::string>& kv, const std::string& key,
                    const std::string& fallback) {
  const auto it = kv.find(key);
  return it == kv.end() ? fallback : it->second;
}

bool env_true(const char* name) {
  const char* v = std::getenv(name);
  if (v == nullptr) {
    return false;
  }
  const std::string s = trim(v);
  return s == "1" || s == "true" || s == "True" || s == "yes" || s == "on";
}

void deprecated_key_used(const std::string& path, const std::string& key,
                         const std::string& replacement) {
  std::cerr << "DEPRECATED_KEY_USED file=" << path << " key=" << key
            << " replacement=" << replacement << "\n";
}

}  // namespace

Config default_config() {
  Config cfg{};

  cfg.run.run_id = "w1_synth_small";
  cfg.run.seed = 12345;
  cfg.run.output_dir = "results/w1_synth_small";
  cfg.run.epochs = 20;
  cfg.run.warmup_epochs = 2;
  cfg.run.overwrite = true;

  cfg.mpi.comm = "WORLD";
  cfg.mpi.barrier_before_epochs = true;
  cfg.mpi.barrier_after_epochs = true;

  cfg.buffer.N = 64;
  cfg.buffer.S = 128;
  cfg.buffer.layout = "row_major";
  cfg.buffer.dtype_in = "float64";

  cfg.radc.enabled = true;
  cfg.radc.intercept = "MPI_Allreduce";
  cfg.radc.only_if_op = "MPI_SUM";
  cfg.radc.r_min = 4;
  cfg.radc.r_max = 16;
  cfg.radc.oversample_p = 8;
  cfg.radc.power_iters = 1;
  cfg.radc.omega_dist = "rademacher";
  cfg.radc.omega_seed = 424242;
  cfg.radc.energy_capture = 0.999;
  cfg.radc.factor_dtype = "float32";

  cfg.sketch.kind = "countsketch2d";
  cfg.sketch.kG = 128;
  cfg.sketch.kS = 128;
  cfg.sketch.num_sketches = 2;
  cfg.sketch.hash_seed_g0 = 777;
  cfg.sketch.sign_seed_g0 = 123;
  cfg.sketch.hash_seed_s0 = 999;
  cfg.sketch.sign_seed_s0 = 456;
  cfg.sketch.hash_seed_g1 = 1777;
  cfg.sketch.sign_seed_g1 = 1123;
  cfg.sketch.hash_seed_s1 = 1999;
  cfg.sketch.sign_seed_s1 = 1456;

  cfg.risk.kind = "cva_like_scoped";
  cfg.risk.netting_sets_G = 8;
  cfg.risk.netting_seed = 202;
  cfg.risk.collateral_kind = "threshold";
  cfg.risk.collateral_threshold = 0.0;
  cfg.risk.a_scalar = 1.0;
  cfg.risk.scenario_weights = "uniform";
  cfg.risk.notional_total = 1.0e9;

  cfg.safety.xva_epsilon_bps = 0.01;
  cfg.safety.accept_margin = 0.90;
  cfg.safety.jl_epsilon = 0.10;
  cfg.safety.jl_delta = 1.0e-12;
  cfg.safety.always_accept = false;
  cfg.safety.verify_enabled = true;
  cfg.compression.double_mode = "native64";

  cfg.detect.enabled = false;
  cfg.detect.delta_m_sigma_threshold = 6.0;
  cfg.detect.force_fallback_if_r_hits_rmax = true;

  cfg.fallback.enabled = true;
  cfg.fallback.mode = "collective_flag";

  cfg.profiling.timing_breakdown = true;
  cfg.profiling.perf_stat_enable = false;
  cfg.profiling.rapl_enable = false;
  cfg.profiling.net_emulation_bandwidth_gbps = 0.0;
  cfg.profiling.net_emulation_base_latency_ms = 0.0;

  cfg.logging.per_rank_csv = "metrics_rank{rank}.csv";
  cfg.logging.per_rank_jsonl = "events_rank{rank}.jsonl";
  cfg.logging.flush_every_epochs = 1;
  cfg.logging.shadow_exact_every = 20;

  cfg.workload.kind = "synth_lowrank";
  cfg.workload.common_scenarios_across_ranks = true;
  cfg.workload.common_delta_m_across_ranks = true;
  cfg.workload.synth_lowrank.true_rank = 8;
  cfg.workload.synth_lowrank.noise_scale = 0.01;
  cfg.workload.synth_lowrank.heavy_tail = false;

  cfg.workload.delta_gamma.d_factors = 16;
  cfg.workload.delta_gamma.notional_min = 1.0e6;
  cfg.workload.delta_gamma.notional_max = 5.0e6;
  cfg.workload.delta_gamma.delta_scale = 1.0;
  cfg.workload.delta_gamma.gamma_scale = 0.2;
  cfg.workload.delta_gamma.scenario_sigma = 1.0;
  cfg.workload.delta_gamma.shock_sigma = 1.0;
  cfg.workload.delta_gamma.nonlinear_lambda = 0.05;

  cfg.workload.pca_like.true_rank = 8;
  cfg.workload.pca_like.noise_scale = 0.05;
  return cfg;
}

std::string resolve_config_path(const std::string& cli_config_path) {
  if (!cli_config_path.empty()) {
    return cli_config_path;
  }

  const char* env = std::getenv("RADC_CONFIG");
  if (env != nullptr && std::string(env).size() > 0) {
    return std::string(env);
  }

  return "configs/w1_synth_small.yaml";
}

Config load_config_from_path(const std::string& path) {
  Config cfg = default_config();
  cfg.source_path = path;

  const auto kv = parse_yaml_subset(path);
  const bool fail_on_deprecated = env_true("RADC_FAIL_ON_DEPRECATED_CONFIG");

  const std::vector<std::pair<std::string, std::string>> deprecated_keys = {
      {"sketch.k_row", "sketch.kG"},
      {"sketch.k_col", "sketch.kS"},
      {"sketch.hash_seed_row", "sketch.hash_seed_g0"},
      {"sketch.sign_seed_row", "sketch.sign_seed_g0"},
      {"sketch.hash_seed_col", "sketch.hash_seed_s0"},
      {"sketch.sign_seed_col", "sketch.sign_seed_s0"},
      {"sketch.jl_epsilon", "safety.jl_epsilon"},
      {"sketch.jl_delta", "safety.jl_delta"},
  };
  for (const auto& kvp : deprecated_keys) {
    if (kv.find(kvp.first) != kv.end()) {
      deprecated_key_used(path, kvp.first, kvp.second);
      if (fail_on_deprecated) {
        throw std::runtime_error("Deprecated config key used with strict mode: " + kvp.first);
      }
    }
  }

  cfg.run.run_id = get_str(kv, "run.run_id", cfg.run.run_id);
  cfg.run.seed = parse_u64(get_str(kv, "run.seed", ""), cfg.run.seed);
  cfg.run.output_dir = get_str(kv, "run.output_dir", cfg.run.output_dir);
  cfg.run.epochs = parse_int(get_str(kv, "run.epochs", ""), cfg.run.epochs);
  cfg.run.warmup_epochs = parse_int(get_str(kv, "run.warmup_epochs", ""), cfg.run.warmup_epochs);
  cfg.run.overwrite = parse_bool(get_str(kv, "run.overwrite", ""), cfg.run.overwrite);

  cfg.mpi.comm = get_str(kv, "mpi.comm", cfg.mpi.comm);
  cfg.mpi.barrier_before_epochs =
      parse_bool(get_str(kv, "mpi.barrier_before_epochs", ""), cfg.mpi.barrier_before_epochs);
  cfg.mpi.barrier_after_epochs =
      parse_bool(get_str(kv, "mpi.barrier_after_epochs", ""), cfg.mpi.barrier_after_epochs);

  cfg.buffer.N = parse_i64(get_str(kv, "buffer.N", ""), cfg.buffer.N);
  cfg.buffer.S = parse_i64(get_str(kv, "buffer.S", ""), cfg.buffer.S);
  cfg.buffer.layout = get_str(kv, "buffer.layout", cfg.buffer.layout);
  cfg.buffer.dtype_in = get_str(kv, "buffer.dtype_in", cfg.buffer.dtype_in);

  cfg.radc.enabled = parse_bool(get_str(kv, "radc.enabled", ""), cfg.radc.enabled);
  cfg.radc.intercept = get_str(kv, "radc.intercept", cfg.radc.intercept);
  cfg.radc.only_if_op = get_str(kv, "radc.only_if_op", cfg.radc.only_if_op);
  cfg.radc.r_min = parse_int(get_str(kv, "radc.r_min", ""), cfg.radc.r_min);
  cfg.radc.r_max = parse_int(get_str(kv, "radc.r_max", ""), cfg.radc.r_max);
  cfg.radc.oversample_p = parse_int(get_str(kv, "radc.oversample_p", ""), cfg.radc.oversample_p);
  cfg.radc.power_iters = parse_int(get_str(kv, "radc.power_iters", ""), cfg.radc.power_iters);
  cfg.radc.omega_dist = get_str(kv, "radc.omega_dist", cfg.radc.omega_dist);
  cfg.radc.omega_seed = parse_u64(get_str(kv, "radc.omega_seed", ""), cfg.radc.omega_seed);
  cfg.radc.energy_capture = parse_double(get_str(kv, "radc.energy_capture", ""), cfg.radc.energy_capture);
  cfg.radc.factor_dtype = get_str(kv, "radc.factor_dtype", cfg.radc.factor_dtype);

  cfg.sketch.kind = get_str(kv, "sketch.kind", cfg.sketch.kind);
  cfg.sketch.kG = parse_int(get_str(kv, "sketch.kG", ""), cfg.sketch.kG);
  cfg.sketch.kS = parse_int(get_str(kv, "sketch.kS", ""), cfg.sketch.kS);
  cfg.sketch.num_sketches =
      parse_int(get_str(kv, "sketch.num_sketches", ""), cfg.sketch.num_sketches);
  cfg.sketch.hash_seed_g0 =
      parse_u64(get_str(kv, "sketch.hash_seed_g0", ""), cfg.sketch.hash_seed_g0);
  cfg.sketch.sign_seed_g0 =
      parse_u64(get_str(kv, "sketch.sign_seed_g0", ""), cfg.sketch.sign_seed_g0);
  cfg.sketch.hash_seed_s0 =
      parse_u64(get_str(kv, "sketch.hash_seed_s0", ""), cfg.sketch.hash_seed_s0);
  cfg.sketch.sign_seed_s0 =
      parse_u64(get_str(kv, "sketch.sign_seed_s0", ""), cfg.sketch.sign_seed_s0);
  cfg.sketch.hash_seed_g1 =
      parse_u64(get_str(kv, "sketch.hash_seed_g1", ""), cfg.sketch.hash_seed_g1);
  cfg.sketch.sign_seed_g1 =
      parse_u64(get_str(kv, "sketch.sign_seed_g1", ""), cfg.sketch.sign_seed_g1);
  cfg.sketch.hash_seed_s1 =
      parse_u64(get_str(kv, "sketch.hash_seed_s1", ""), cfg.sketch.hash_seed_s1);
  cfg.sketch.sign_seed_s1 =
      parse_u64(get_str(kv, "sketch.sign_seed_s1", ""), cfg.sketch.sign_seed_s1);

  // Backward compatibility with legacy keys.
  cfg.sketch.kG = parse_int(get_str(kv, "sketch.k_row", ""), cfg.sketch.kG);
  cfg.sketch.kS = parse_int(get_str(kv, "sketch.k_col", ""), cfg.sketch.kS);
  cfg.sketch.hash_seed_g0 =
      parse_u64(get_str(kv, "sketch.hash_seed_row", ""), cfg.sketch.hash_seed_g0);
  cfg.sketch.sign_seed_g0 =
      parse_u64(get_str(kv, "sketch.sign_seed_row", ""), cfg.sketch.sign_seed_g0);
  cfg.sketch.hash_seed_s0 =
      parse_u64(get_str(kv, "sketch.hash_seed_col", ""), cfg.sketch.hash_seed_s0);
  cfg.sketch.sign_seed_s0 =
      parse_u64(get_str(kv, "sketch.sign_seed_col", ""), cfg.sketch.sign_seed_s0);

  cfg.risk.kind = get_str(kv, "risk.kind", cfg.risk.kind);
  cfg.risk.netting_sets_G =
      parse_int(get_str(kv, "risk.netting_sets_G", ""), cfg.risk.netting_sets_G);
  cfg.risk.netting_seed = parse_u64(get_str(kv, "risk.netting_seed", ""), cfg.risk.netting_seed);
  cfg.risk.collateral_kind = get_str(kv, "risk.collateral_kind", cfg.risk.collateral_kind);
  cfg.risk.collateral_threshold =
      parse_double(get_str(kv, "risk.collateral_threshold", ""), cfg.risk.collateral_threshold);
  cfg.risk.a_scalar = parse_double(get_str(kv, "risk.a_scalar", ""), cfg.risk.a_scalar);
  cfg.risk.scenario_weights = get_str(kv, "risk.scenario_weights", cfg.risk.scenario_weights);
  cfg.risk.notional_total =
      parse_double(get_str(kv, "risk.notional_total", ""), cfg.risk.notional_total);

  cfg.safety.xva_epsilon_bps =
      parse_double(get_str(kv, "safety.xva_epsilon_bps", ""), cfg.safety.xva_epsilon_bps);
  cfg.safety.accept_margin =
      parse_double(get_str(kv, "safety.accept_margin", ""), cfg.safety.accept_margin);
  cfg.safety.jl_epsilon =
      parse_double(get_str(kv, "safety.jl_epsilon", ""), cfg.safety.jl_epsilon);
  cfg.safety.jl_delta = parse_double(get_str(kv, "safety.jl_delta", ""), cfg.safety.jl_delta);
  // Backward compatibility with legacy placement under sketch.*
  cfg.safety.jl_epsilon =
      parse_double(get_str(kv, "sketch.jl_epsilon", ""), cfg.safety.jl_epsilon);
  cfg.safety.jl_delta = parse_double(get_str(kv, "sketch.jl_delta", ""), cfg.safety.jl_delta);
  cfg.safety.always_accept =
      parse_bool(get_str(kv, "safety.always_accept", ""), cfg.safety.always_accept);
  cfg.safety.verify_enabled =
      parse_bool(get_str(kv, "safety.verify_enabled", ""), cfg.safety.verify_enabled);
  cfg.compression.double_mode =
      get_str(kv, "compression.double_mode", cfg.compression.double_mode);
  if (cfg.compression.double_mode == "native64" || cfg.compression.double_mode == "downcast32" ||
      cfg.compression.double_mode == "passthrough") {
    // ok
  } else {
    throw std::runtime_error("Invalid compression.double_mode: " + cfg.compression.double_mode);
  }
  const std::string legacy_allow_double_downcast =
      get_str(kv, "compression.allow_double_downcast", "");
  if (!legacy_allow_double_downcast.empty()) {
    deprecated_key_used(path, "compression.allow_double_downcast", "compression.double_mode");
    if (fail_on_deprecated) {
      throw std::runtime_error("Deprecated config key used with strict mode: compression.allow_double_downcast");
    }
    if (parse_bool(legacy_allow_double_downcast, false)) {
      cfg.compression.double_mode = "downcast32";
    }
  }

  cfg.detect.enabled = parse_bool(get_str(kv, "detect.enabled", ""), cfg.detect.enabled);
  cfg.detect.delta_m_sigma_threshold = parse_double(
      get_str(kv, "detect.delta_m_sigma_threshold", ""), cfg.detect.delta_m_sigma_threshold);
  cfg.detect.force_fallback_if_r_hits_rmax = parse_bool(
      get_str(kv, "detect.force_fallback_if_r_hits_rmax", ""), cfg.detect.force_fallback_if_r_hits_rmax);

  cfg.fallback.enabled = parse_bool(get_str(kv, "fallback.enabled", ""), cfg.fallback.enabled);
  cfg.fallback.mode = get_str(kv, "fallback.mode", cfg.fallback.mode);

  cfg.profiling.timing_breakdown =
      parse_bool(get_str(kv, "profiling.timing_breakdown", ""), cfg.profiling.timing_breakdown);
  cfg.profiling.perf_stat_enable =
      parse_bool(get_str(kv, "profiling.perf_stat_enable", ""), cfg.profiling.perf_stat_enable);
  cfg.profiling.rapl_enable =
      parse_bool(get_str(kv, "profiling.rapl_enable", ""), cfg.profiling.rapl_enable);
  cfg.profiling.net_emulation_bandwidth_gbps = parse_double(
      get_str(kv, "profiling.net_emulation_bandwidth_gbps", ""),
      cfg.profiling.net_emulation_bandwidth_gbps);
  cfg.profiling.net_emulation_base_latency_ms = parse_double(
      get_str(kv, "profiling.net_emulation_base_latency_ms", ""),
      cfg.profiling.net_emulation_base_latency_ms);

  cfg.logging.per_rank_csv = get_str(kv, "logging.per_rank_csv", cfg.logging.per_rank_csv);
  cfg.logging.per_rank_jsonl =
      get_str(kv, "logging.per_rank_jsonl", cfg.logging.per_rank_jsonl);
  cfg.logging.flush_every_epochs =
      parse_int(get_str(kv, "logging.flush_every_epochs", ""), cfg.logging.flush_every_epochs);
  cfg.logging.shadow_exact_every =
      parse_int(get_str(kv, "logging.shadow_exact_every", ""), cfg.logging.shadow_exact_every);

  cfg.workload.kind = get_str(kv, "workload.kind", cfg.workload.kind);
  cfg.workload.common_scenarios_across_ranks = parse_bool(
      get_str(kv, "workload.common_scenarios_across_ranks", ""),
      cfg.workload.common_scenarios_across_ranks);
  cfg.workload.common_delta_m_across_ranks = parse_bool(
      get_str(kv, "workload.common_delta_m_across_ranks", ""),
      cfg.workload.common_delta_m_across_ranks);
  cfg.workload.synth_lowrank.true_rank = parse_int(
      get_str(kv, "workload.synth_lowrank.true_rank", ""), cfg.workload.synth_lowrank.true_rank);
  cfg.workload.synth_lowrank.noise_scale = parse_double(
      get_str(kv, "workload.synth_lowrank.noise_scale", ""), cfg.workload.synth_lowrank.noise_scale);
  cfg.workload.synth_lowrank.heavy_tail = parse_bool(
      get_str(kv, "workload.synth_lowrank.heavy_tail", ""), cfg.workload.synth_lowrank.heavy_tail);

  cfg.workload.delta_gamma.d_factors = parse_int(
      get_str(kv, "workload.delta_gamma.d_factors", ""), cfg.workload.delta_gamma.d_factors);
  cfg.workload.delta_gamma.notional_min = parse_double(
      get_str(kv, "workload.delta_gamma.notional_min", ""), cfg.workload.delta_gamma.notional_min);
  cfg.workload.delta_gamma.notional_max = parse_double(
      get_str(kv, "workload.delta_gamma.notional_max", ""), cfg.workload.delta_gamma.notional_max);
  cfg.workload.delta_gamma.delta_scale = parse_double(
      get_str(kv, "workload.delta_gamma.delta_scale", ""), cfg.workload.delta_gamma.delta_scale);
  cfg.workload.delta_gamma.gamma_scale = parse_double(
      get_str(kv, "workload.delta_gamma.gamma_scale", ""), cfg.workload.delta_gamma.gamma_scale);
  cfg.workload.delta_gamma.scenario_sigma = parse_double(
      get_str(kv, "workload.delta_gamma.scenario_sigma", ""), cfg.workload.delta_gamma.scenario_sigma);
  cfg.workload.delta_gamma.shock_sigma = parse_double(
      get_str(kv, "workload.delta_gamma.shock_sigma", ""), cfg.workload.delta_gamma.shock_sigma);
  cfg.workload.delta_gamma.nonlinear_lambda = parse_double(
      get_str(kv, "workload.delta_gamma.nonlinear_lambda", ""), cfg.workload.delta_gamma.nonlinear_lambda);

  cfg.workload.pca_like.true_rank = parse_int(
      get_str(kv, "workload.pca_like.true_rank", ""), cfg.workload.pca_like.true_rank);
  cfg.workload.pca_like.noise_scale = parse_double(
      get_str(kv, "workload.pca_like.noise_scale", ""), cfg.workload.pca_like.noise_scale);

  return cfg;
}

Config load_config_from_env_or_default(const std::string& cli_config_path) {
  const std::string path = resolve_config_path(cli_config_path);

  Config cfg = load_config_from_path(path);

  if (cfg.run.output_dir.empty()) {
    cfg.run.output_dir = std::string("results/") + cfg.run.run_id;
  }

  std::filesystem::path p(cfg.run.output_dir);
  if (!p.is_absolute()) {
    const std::filesystem::path config_path(path);
    const std::filesystem::path config_dir = config_path.parent_path();
    const std::filesystem::path repo_root = config_dir.parent_path();
    cfg.run.output_dir = (repo_root / p).lexically_normal().string();
  } else {
    cfg.run.output_dir = p.lexically_normal().string();
  }

  return cfg;
}

}  // namespace radc
