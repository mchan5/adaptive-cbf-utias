// Standalone diagnostic for investigating why PennGammaSelector::selectGamma
// has been falling back to gamma_min on every call in practice (see
// params_single_vehicle_cbf_rate_arc.yaml's penn_enabled comment). Loads the
// real checkpoint and runs the real inference path against a frozen-vehicle
// state captured from an actual CBF-QP deadlock (2026-08-10 randomized
// stress test, trial 4), printing the full per-candidate-gamma table
// (divergence, predicted deadlock/risk, pass/fail against each gate)
// instead of just the final selected gamma.
#include <array>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include <Eigen/Dense>

#include "single_vehicle_cbf_rate/penn_gamma_selector.hpp"

using nodelib::PennGammaSelector;

// Offline dev tool: resolve default asset paths from $ARC_ROOT (exported by
// scripts/setup.sh); fall back to a repo-root-relative path. Always overridable
// via argv.
static std::string arcPath(const char* rel) {
  const char* root = std::getenv("ARC_ROOT");
  return (root && *root ? std::string(root) + "/" : std::string()) + rel;
}

int main(int argc, char** argv) {
  const std::string model_path =
      argc > 1 ? argv[1]
                : arcPath("src/arc/single_vehicle_cbf_rate_arc/"
                          "nn_model/checkpoint/gat_penn_weights.pt");

  PennGammaSelector selector(model_path.c_str());

  // Frozen state from trial 4's deadlock (gate 1 dodged, gate 2 immediately
  // ahead and to the side -- see conversation for the full trace). Swap in
  // an early-flight state (e.g. pos=(0.02,0.3,0.15), vel=(0.1,2.6,0.4)) to
  // reproduce the far-field high-epistemic-divergence comparison case.
  const Eigen::Vector3d pos_enu(0.97, 1.08, -0.03);
  const Eigen::Vector3d vel_enu(0.0, 0.0, 0.0);
  const Eigen::Vector3d goal(0.0, 5.5, 1.5);  // matches waypoint_{x,y,z} in the live yaml

  // trial_04.json obstacles: flat [x,y,z,diameter]*, z-stacked pairs.
  // ObstacleInput.radius mirrors production (obstaclesCallback): diameter/2
  // + obs_safety_margin (0.5), not the CBF's fixed obs_d_min.
  const double margin = 0.5;
  const std::vector<std::array<double, 4>> raw_obstacles = {
      {0.0, 0.9, 0.52, 1.03},    {0.0, 0.9, 1.61, 1.03},   {-1.06, 2.07, 0.49, 0.66},
      {1.06, 2.07, 0.49, 0.66},  {-1.06, 2.07, 1.56, 0.66}, {1.06, 2.07, 1.56, 0.66},
      {0.25, 3.22, 0.35, 0.82},  {0.25, 3.22, 1.44, 0.82},  {-1.22, 4.3, 0.45, 0.96},
      {1.22, 4.3, 0.45, 0.96},   {-1.22, 4.3, 1.37, 0.96},  {1.22, 4.3, 1.37, 0.96},
      {-1.04, 5.3, 0.37, 1.08},  {1.04, 5.3, 0.37, 1.08},   {-1.04, 5.3, 1.32, 1.08},
      {1.04, 5.3, 1.32, 1.08},
  };
  std::vector<PennGammaSelector::ObstacleInput> obstacles;
  for (const auto& o : raw_obstacles) {
    obstacles.push_back({Eigen::Vector3d(o[0], o[1], o[2]), o[3] / 2.0 + margin});
  }

  // Params from params_single_vehicle_cbf_rate_arc.yaml.
  // Updated 2026-08-22 to the validated checkpoint's actual calibration
  // (was the stale/OOD params_single_vehicle_cbf_rate_arc.yaml values:
  // gamma_max_candidate=10.0 against a training range that tops out at
  // 3.5, and epistemic/cvar thresholds calibrated for the pre-min_h-label
  // checkpoint). See eval_scene_distribution.py's DEPLOY_GAMMA_MIN/MAX and
  // nn_model/checkpoint/cccp_threshold_Quadrotor3D_gat.json.
  PennGammaSelector::GammaSelectionParams params;
  params.goal = goal;
  params.gamma_min = 0.5;
  params.gamma_max_candidate = 2.5;
  params.gamma_steps = 20;
  params.epistemic_threshold = 0.174864;
  params.cvar_boundary = 0.0055;
  params.deadlock_threshold = 5.0;  // vestigial as of the 2026-08-22 gate fix -- no longer a hard gate

  const auto selected = selector.selectGamma(pos_enu, vel_enu, obstacles, params);
  std::printf("selectGamma() result: %s\n\n",
              selected.has_value() ? std::to_string(*selected).c_str() : "nullopt (inference failed)");

  const auto table = selector.diagnoseGammaSelection(pos_enu, vel_enu, obstacles, params);
  if (!table.has_value()) {
    std::printf("diagnoseGammaSelection() failed (empty obstacles or inference exception)\n");
    return 1;
  }

  std::printf("%8s %12s %10s %14s %10s %10s %10s\n", "gamma", "divergence", "eps_thr",
              "pass_eps", "deadlock", "risk", "pass_dr");
  for (const auto& row : *table) {
    std::printf("%8.3f %12.6f %10.6f %14s %10.4f %10.4f %10s\n", row.gamma, row.divergence,
                params.epistemic_threshold, row.passed_epistemic ? "yes" : "NO", row.mean_deadlock,
                row.mean_risk, row.passed_deadlock_risk ? "yes" : "no");
  }

  int n_passed_eps = 0;
  for (const auto& row : *table) {
    if (row.passed_epistemic) ++n_passed_eps;
  }
  std::printf("\n%d/%zu candidates passed the epistemic-divergence gate (threshold=%.6f)\n",
              n_passed_eps, table->size(), params.epistemic_threshold);

  return 0;
}
