// Diagnostic (2026-08-24, range-widening plan step 1): before spending a retrain's wall-clock time
// on denser sampling above gamma=3.5, check whether PENN's predictions actually carry useful …
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include <torch/csrc/jit/serialization/pickle.h>
#include <torch/torch.h>

#include "single_vehicle_cbf_rate/penn_gamma_selector.hpp"

using nodelib::PennGammaSelector;

namespace {

struct Obstacle {
  Eigen::Vector3d center;
  double radius_phys;
};

struct Scene {
  Eigen::Vector3d start;
  Eigen::Vector3d goal;
  std::vector<Obstacle> obstacles;
};

// Mirrors body_rate_closed_loop_diag.cpp's loadScenes exactly (same .pt scene format) -- duplicated
// rather than shared to keep each diagnostic tool self-contained, matching this directory's …
std::vector<Scene> loadScenes(const std::string& path) {
  std::ifstream file(path, std::ios::binary);
  if (!file) {
    std::fprintf(stderr, "cannot open %s\n", path.c_str());
    std::exit(1);
  }
  std::vector<char> data((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
  const torch::IValue loaded = torch::jit::pickle_load(data);
  const auto list = loaded.toList();

  std::vector<Scene> scenes;
  for (size_t i = 0; i < list.size(); ++i) {
    const auto dict = list.get(i).toGenericDict();
    const torch::Tensor start_t = dict.at("start").toTensor();
    const torch::Tensor goal_t = dict.at("goal").toTensor();
    const torch::Tensor obs_t = dict.at("obstacles").toTensor();

    Scene scene;
    scene.start = Eigen::Vector3d(start_t[0].item<double>(), start_t[1].item<double>(),
                                   start_t[2].item<double>());
    scene.goal = Eigen::Vector3d(goal_t[0].item<double>(), goal_t[1].item<double>(),
                                  goal_t[2].item<double>());
    const auto acc = obs_t.accessor<double, 2>();
    for (int64_t k = 0; k < obs_t.size(0); ++k) {
      scene.obstacles.push_back({Eigen::Vector3d(acc[k][0], acc[k][1], acc[k][2]), acc[k][3]});
    }
    scenes.push_back(std::move(scene));
  }
  return scenes;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 4) {
    std::fprintf(stderr,
                  "usage: %s <scenes_dir_with_scenes_seed*.pt> <model_path> <out_csv>\n", argv[0]);
    return 1;
  }
  const std::string scenes_dir = argv[1];
  const std::string model_path = argv[2];
  const std::string out_csv = argv[3];

  PennGammaSelector selector(model_path.c_str());

  PennGammaSelector::GammaSelectionParams params;
  params.gamma_min = 1.3;
  params.gamma_max_candidate = 8.0;
  params.gamma_steps = 20;
  params.epistemic_threshold = 0.10687255859375;
  if (const char* env = std::getenv("EPISTEMIC_THRESHOLD")) params.epistemic_threshold = std::atof(env);
  params.cvar_boundary = 0.0023882027249783394;
  if (const char* env = std::getenv("CVAR_BOUNDARY")) params.cvar_boundary = std::atof(env);
  params.lcb_k = 0.0;

  const double obs_safety_margin = 0.5;  // matches deployed obs_safety_margin

  std::ofstream csv(out_csv);
  csv << "seed,scene,gamma,divergence,mean_deadlock,mean_risk,passed_epistemic,passed_gate\n";

  namespace fs = std::filesystem;
  std::vector<fs::path> files;
  for (const auto& entry : fs::directory_iterator(scenes_dir)) {
    if (entry.path().filename().string().rfind("scenes_seed", 0) == 0) {
      files.push_back(entry.path());
    }
  }
  std::sort(files.begin(), files.end());

  int n_ok = 0, n_failed = 0;
  for (const auto& f : files) {
    const std::string stem = f.stem().string();
    const int seed = std::atoi(stem.c_str() + std::string("scenes_seed").size());
    const auto scenes = loadScenes(f.string());

    for (size_t s = 0; s < scenes.size(); ++s) {
      const auto& scene = scenes[s];
      params.goal = scene.goal;
      std::vector<PennGammaSelector::ObstacleInput> obstacles;
      for (const auto& o : scene.obstacles) {
        obstacles.push_back({o.center, o.radius_phys + obs_safety_margin});
      }
      // QUERY_VEL="x,y,z" overrides the at-rest default (2026-08-24): every real episode DOES start
      // at rest, so vel=0 is a legitimate query condition, not an edge case to dismiss -- but …
      Eigen::Vector3d query_vel(0.0, 0.0, 0.0);
      if (const char* env = std::getenv("QUERY_VEL")) {
        double vx = 0, vy = 0, vz = 0;
        std::sscanf(env, "%lf,%lf,%lf", &vx, &vy, &vz);
        query_vel = Eigen::Vector3d(vx, vy, vz);
      }
      const auto table = selector.diagnoseGammaSelection(scene.start, query_vel, obstacles, params);
      if (!table.has_value()) {
        ++n_failed;
        continue;
      }
      ++n_ok;
      for (const auto& row : *table) {
        csv << seed << ',' << s << ',' << row.gamma << ',' << row.divergence << ','
            << row.mean_deadlock << ',' << row.mean_risk << ','
            << (row.passed_epistemic ? 1 : 0) << ',' << (row.passed_deadlock_risk ? 1 : 0) << '\n';
      }
    }
  }
  std::printf("wrote %s: %d scenes ok, %d failed (empty obstacles or inference exception)\n",
              out_csv.c_str(), n_ok, n_failed);
  return 0;
}
