// Closed-loop trend check: replay the frozen test scenes through the real RateAutopilotCore +
// rigid-body integration (8 scenes x 4 arms: PENN-adaptive plus three fixed gammas), asking …
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>
#include <thread>
#include <vector>

#include <rcl/time.h>
#include <torch/csrc/jit/serialization/pickle.h>
#include <torch/torch.h>

#include "single_vehicle_cbf_rate/rate_autopilot_core.hpp"
#include "deploy_yaml_params.hpp"

using nodelib::RateAutopilotCore;

namespace {

constexpr double kSensorMaxRange = 8.0;
constexpr double kSensorZMin = 0.1;
constexpr double kSensorZMax = 2.5;

struct Obstacle {
  Eigen::Vector3d center;
  double radius_phys;  // physical radius, NO safety margin (matches marker.scale.x/2 contract)
};

struct Scene {
  Eigen::Vector3d start;
  Eigen::Vector3d goal;
  std::vector<Obstacle> obstacles;
};

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
      scene.obstacles.push_back(
          {Eigen::Vector3d(acc[k][0], acc[k][1], acc[k][2]), acc[k][3]});
    }
    scenes.push_back(std::move(scene));
  }
  return scenes;
}

struct SeedScenes {
  int seed;
  std::vector<Scene> scenes;
};

std::vector<SeedScenes> loadSeedSource(const std::string& path) {
  namespace fs = std::filesystem;
  if (!fs::is_directory(path)) {
    return {{-1, loadScenes(path)}};
  }
  std::vector<fs::path> files;
  for (const auto& entry : fs::directory_iterator(path)) {
    if (entry.path().filename().string().rfind("scenes_seed", 0) == 0) {
      files.push_back(entry.path());
    }
  }
  std::sort(files.begin(), files.end());

  std::vector<SeedScenes> out;
  for (const auto& f : files) {
    const std::string stem = f.stem().string();  // "scenes_seed20260906"
    const int seed = std::atoi(stem.c_str() + std::string("scenes_seed").size());
    out.push_back({seed, loadScenes(f.string())});
  }
  return out;
}

// Obstacle model for this run.
const bool g_cylinder_barrier = std::getenv("CYLINDER_BARRIER") != nullptr;

// Distance from `pos` to an obstacle centre, projected to the horizontal plane
// in cylinder mode so the clearance metrics match the active barrier.
double obstacleCenterDist(const Eigen::Vector3d& pos, const Eigen::Vector3d& center) {
  return g_cylinder_barrier ? (pos - center).head<2>().norm() : (pos - center).norm();
}

RateAutopilotCore::Params baseParams() {
  // Loads config/params_single_vehicle_cbf_rate_arc.yaml -- the SAME file the deployed node ships
  // with -- so this harness measures the config that actually flies, not a hand-maintained …
  RateAutopilotCore::Params p;
  diagtools::loadDeployedParams(p);
  p.cbf_cylinder_barrier = g_cylinder_barrier;

  // Env-var A/B overrides, applied ON TOP of the deployed values.
  if (const char* env = std::getenv("NOM_ACCEL_MAX")) p.nom_accel_max = std::atof(env);
  if (const char* env = std::getenv("NOM_POS_KP")) p.nom_pos_kp = std::atof(env);
  if (const char* env = std::getenv("NOM_VEL_KD")) p.nom_vel_kd = std::atof(env);
  if (const char* env = std::getenv("PENN_GAMMA_MIN")) {
    p.penn_gamma_min = std::atof(env);
  }
  if (const char* env = std::getenv("PENN_GAMMA_MAX")) {
    p.penn_gamma_max = std::atof(env);
  }
  if (const char* env = std::getenv("PENN_GAMMA_STEPS")) {
    p.penn_gamma_steps = std::atoi(env);
  }
  if (const char* env = std::getenv("PENN_UPDATE_TICKS")) {
    p.penn_update_ticks = std::atoi(env);
  }
  if (const char* env = std::getenv("EPISTEMIC_THRESHOLD")) {
    p.epistemic_threshold = std::atof(env);
  }
  if (const char* env = std::getenv("CVAR_BOUNDARY")) {
    p.cvar_boundary = std::atof(env);
  }
  if (const char* env = std::getenv("LCB_K")) {
    p.lcb_k = std::atof(env);
  }
  if (const char* env = std::getenv("CONTINUITY_WEIGHT")) {
    p.continuity_weight = std::atof(env);
  }
  if (const char* env = std::getenv("FALLBACK_STEP")) {
    p.fallback_step = std::atof(env);
  }
  // FEASIBILITY_GATE=1 (2026-08-26, PLAN_adaptive_recovery §1.2 A/B check).
  if (std::getenv("FEASIBILITY_GATE") != nullptr) {
    p.penn_feasibility_gate = true;
  }
  p.deadlock_threshold = 1.0e9;  // vestigial, never gates -- see penn_gamma_selector.cpp

  if (const char* env = std::getenv("ESCAPE_SEARCH")) {
    p.escape_search_enabled = (std::atoi(env) != 0);
  }
  return p;
}

visualization_msgs::msg::MarkerArray visibleMarkers(const Eigen::Vector3d& pos,
                                                      const std::vector<Obstacle>& obstacles) {
  visualization_msgs::msg::MarkerArray markers;
  for (size_t i = 0; i < obstacles.size(); ++i) {
    const auto& o = obstacles[i];
    if (o.center.z() < kSensorZMin || o.center.z() > kSensorZMax) continue;
    const double dist_to_surface = obstacleCenterDist(pos, o.center) - o.radius_phys;
    if (dist_to_surface > kSensorMaxRange) continue;

    visualization_msgs::msg::Marker m;
    m.id = static_cast<int>(i);
    m.pose.position.x = o.center.x();
    m.pose.position.y = o.center.y();
    m.pose.position.z = o.center.z();
    m.scale.x = 2.0 * o.radius_phys;
    markers.markers.push_back(m);
  }
  return markers;
}

struct EpisodeResult {
  bool reached_goal{false};
  double t_goal{0.0};
  double min_h_physical{std::numeric_limits<double>::infinity()};  // true clearance to surface
  double min_h_margin{std::numeric_limits<double>::infinity()};    // clearance to safety-margin buffer
  bool infeasible_any{false};
  int n_ticks{0};
  // Ticks where gamma_used != cbf_gamma_obs (the fallback value), i.e. PENN actually selected the
  // gain this tick rather than the tick falling back (beyond penn_proximity_gate_m, no obstacles …
  int n_penn_active_ticks{0};
  // Mean gamma_used across every tick (PENN-selected AND fallback ticks).
  double mean_gamma_used{0.0};
};

EpisodeResult runEpisode(const Scene& scene, RateAutopilotCore::Params params, double dt,
                          double max_time_s, bool debug_ticks = false) {
  auto logger = rclcpp::get_logger("body_rate_closed_loop_diag");
  // Simulated clock, advanced by exactly dt per tick (2026-08-22).
  auto clock = std::make_shared<rclcpp::Clock>(RCL_ROS_TIME);
  rcl_enable_ros_time_override(clock->get_clock_handle());
  int64_t sim_time_ns = 1000000000LL;  // start at t=1s, not 0, so time deltas are well-defined
  rcl_set_ros_time_override(clock->get_clock_handle(), sim_time_ns);
  const int64_t dt_ns = static_cast<int64_t>(dt * 1e9);

  params.waypoint = scene.goal;
  RateAutopilotCore core(params, logger, clock);
  // Hard-fail rather than warn: RateAutopilotCore's own loadPennModel() warns-and-falls-back on a
  // bad model path, which is the right behavior for a flying vehicle but means an arm claiming …
  if (params.penn_enabled && !core.pennModelLoaded()) {
    std::fprintf(stderr,
                  "FATAL: penn_enabled=true but PENN model failed to load from '%s' -- "
                  "this episode would silently run fixed cbf_gamma_obs=%.2f under the "
                  "\"adaptive\" label. Refusing to produce a mislabeled result.\n",
                  params.penn_model_path.c_str(), params.cbf_gamma_obs);
    std::exit(1);
  }

  Eigen::Vector3d pos = scene.start;
  Eigen::Vector3d vel = Eigen::Vector3d::Zero();
  Eigen::Matrix3d R = Eigen::Matrix3d::Identity();
  core.setOdometry(pos, vel, R, clock->now());

  EpisodeResult result;
  const int max_steps = static_cast<int>(max_time_s / dt);
  const double margin = params.obs_safety_margin;
  double gamma_sum = 0.0;

  for (int step = 0; step < max_steps; ++step) {
    if ((scene.goal - pos).norm() < 0.3) {
      result.reached_goal = true;
      result.t_goal = step * dt;
      break;
    }

    core.handleObstacles(visibleMarkers(pos, scene.obstacles));
    const auto tick = core.tick();
    if (!tick.feasible) result.infeasible_any = true;
    ++result.n_ticks;
    if (std::abs(tick.gamma_used - params.cbf_gamma_obs) > 1e-6) {
      ++result.n_penn_active_ticks;
    }
    gamma_sum += tick.gamma_used;

    const Eigen::Vector3d accel =
        (tick.T / params.vehicle_mass) * (R * Eigen::Vector3d::UnitZ()) -
        nodelib::kQuadG * Eigen::Vector3d::UnitZ();
    const double omega_norm = tick.omega.norm();
    if (omega_norm > 1e-9) {
      R = R * Eigen::AngleAxisd(omega_norm * dt, tick.omega / omega_norm).toRotationMatrix();
    }
    vel = vel + accel * dt;
    pos = pos + vel * dt;

    double closest = std::numeric_limits<double>::infinity();
    for (const auto& o : scene.obstacles) {
      const double dist_to_surface = obstacleCenterDist(pos, o.center) - o.radius_phys;
      result.min_h_physical = std::min(result.min_h_physical, dist_to_surface);
      result.min_h_margin = std::min(result.min_h_margin, dist_to_surface - margin);
      closest = std::min(closest, dist_to_surface);
    }
    if (debug_ticks) {
      std::fprintf(stderr, "  step=%4d gamma_used=%.3f feasible=%d closest_surface=%+.3f pos=[%.2f %.2f %.2f]\n",
                   step, tick.gamma_used, tick.feasible ? 1 : 0, closest, pos.x(), pos.y(), pos.z());
    }

    sim_time_ns += dt_ns;
    rcl_set_ros_time_override(clock->get_clock_handle(), sim_time_ns);
    core.setOdometry(pos, vel, R, clock->now());
  }

  if (!result.reached_goal) result.t_goal = max_time_s;
  result.mean_gamma_used = result.n_ticks > 0 ? gamma_sum / result.n_ticks : 0.0;
  return result;
}

// GREEDY_SWITCH mode (2026-08-24): measures the headroom available from letting gamma vary WITHIN
// an episode, not just picking one constant value per scene.
struct WindowResult {
  Eigen::Vector3d pos;
  bool infeasible_any{false};
  double min_h_margin{std::numeric_limits<double>::infinity()};
};

WindowResult simulateWindow(const Scene& scene, RateAutopilotCore::Params params, double dt,
                             const Eigen::Vector3d& pos0, const Eigen::Vector3d& vel0,
                             const Eigen::Matrix3d& R0, int64_t sim_time_ns0, int max_ticks,
                             double margin) {
  auto logger = rclcpp::get_logger("greedy_rollout");
  auto clock = std::make_shared<rclcpp::Clock>(RCL_ROS_TIME);
  rcl_enable_ros_time_override(clock->get_clock_handle());
  int64_t sim_time_ns = sim_time_ns0;
  rcl_set_ros_time_override(clock->get_clock_handle(), sim_time_ns);
  const int64_t dt_ns = static_cast<int64_t>(dt * 1e9);

  params.waypoint = scene.goal;
  RateAutopilotCore core(params, logger, clock);

  Eigen::Vector3d pos = pos0, vel = vel0;
  Eigen::Matrix3d R = R0;
  core.setOdometry(pos, vel, R, clock->now());

  WindowResult wr;
  wr.pos = pos0;

  for (int k = 0; k < max_ticks; ++k) {
    if ((scene.goal - pos).norm() < 0.3) break;
    core.handleObstacles(visibleMarkers(pos, scene.obstacles));
    const auto tick = core.tick();
    if (!tick.feasible) wr.infeasible_any = true;

    const Eigen::Vector3d accel = (tick.T / params.vehicle_mass) * (R * Eigen::Vector3d::UnitZ()) -
                                   nodelib::kQuadG * Eigen::Vector3d::UnitZ();
    const double omega_norm = tick.omega.norm();
    if (omega_norm > 1e-9) {
      R = R * Eigen::AngleAxisd(omega_norm * dt, tick.omega / omega_norm).toRotationMatrix();
    }
    vel = vel + accel * dt;
    pos = pos + vel * dt;

    for (const auto& o : scene.obstacles) {
      const double dist_to_surface = obstacleCenterDist(pos, o.center) - o.radius_phys;
      wr.min_h_margin = std::min(wr.min_h_margin, dist_to_surface - margin);
    }

    sim_time_ns += dt_ns;
    rcl_set_ros_time_override(clock->get_clock_handle(), sim_time_ns);
    core.setOdometry(pos, vel, R, clock->now());

    if ((scene.goal - pos).norm() < 0.3) break;
  }
  wr.pos = pos;
  return wr;
}

EpisodeResult runEpisodeGreedySwitch(const Scene& scene, RateAutopilotCore::Params base_params, double dt,
                                      double max_time_s, const std::vector<double>& candidate_gammas,
                                      int horizon_ticks, int decision_interval_ticks,
                                      double proximity_gate_m, double infeasible_penalty) {
  EpisodeResult result;
  const int max_steps = static_cast<int>(max_time_s / dt);
  const double margin = base_params.obs_safety_margin;

  auto logger = rclcpp::get_logger("body_rate_closed_loop_diag");
  auto clock = std::make_shared<rclcpp::Clock>(RCL_ROS_TIME);
  rcl_enable_ros_time_override(clock->get_clock_handle());
  int64_t sim_time_ns = 1000000000LL;
  rcl_set_ros_time_override(clock->get_clock_handle(), sim_time_ns);
  const int64_t dt_ns = static_cast<int64_t>(dt * 1e9);

  RateAutopilotCore::Params real_params = base_params;
  real_params.penn_enabled = false;
  real_params.waypoint = scene.goal;
  real_params.cbf_gamma_obs = candidate_gammas.front();  // overwritten before first real tick
  RateAutopilotCore real_core(real_params, logger, clock);

  Eigen::Vector3d pos = scene.start;
  Eigen::Vector3d vel = Eigen::Vector3d::Zero();
  Eigen::Matrix3d R = Eigen::Matrix3d::Identity();
  real_core.setOdometry(pos, vel, R, clock->now());

  double gamma_sum = 0.0;
  int ticks_done = 0;
  double current_gamma = candidate_gammas.front();

  while (ticks_done < max_steps) {
    if ((scene.goal - pos).norm() < 0.3) {
      result.reached_goal = true;
      result.t_goal = ticks_done * dt;
      break;
    }

    // Proximity gate (mirrors pennSelectAlpha's real gating logic exactly: CENTER distance, not
    // surface distance, to whichever obstacle is closest) -- added after the first pilot run …
    double min_center_dist = std::numeric_limits<double>::infinity();
    for (const auto& o : scene.obstacles) {
      min_center_dist = std::min(min_center_dist, obstacleCenterDist(pos, o.center));
    }
    const int window = std::min(horizon_ticks, max_steps - ticks_done);
    int best_idx = -1;
    if (min_center_dist > proximity_gate_m) {
      // Score every candidate via a throwaway rollout clone (see comment above).
      double best_score = std::numeric_limits<double>::infinity();
      const double dist_before = (scene.goal - pos).norm();
      for (size_t ci = 0; ci < candidate_gammas.size(); ++ci) {
        RateAutopilotCore::Params p = base_params;
        p.penn_enabled = false;
        p.cbf_gamma_obs = candidate_gammas[ci];
        const WindowResult wr = simulateWindow(scene, p, dt, pos, vel, R, sim_time_ns, window, margin);

        const double progress = dist_before - (scene.goal - wr.pos).norm();
        double score = -progress;
        // The infeasibility penalty is env-tunable (default 1.0 = the original behavior) because at
        // the original fixed 1.0 this mode was degenerate: over a decision window, progress …
        if (wr.infeasible_any) score += infeasible_penalty;
        if (wr.min_h_margin < 0.0) score += 1000.0;  // safety veto

        if (score < best_score) {
          best_score = score;
          best_idx = static_cast<int>(ci);
        }
      }
    }

    if (best_idx >= 0) current_gamma = candidate_gammas[best_idx];
    real_core.setFixedGamma(current_gamma);
    const int commit = std::min(decision_interval_ticks, max_steps - ticks_done);
    for (int k = 0; k < commit; ++k) {
      if ((scene.goal - pos).norm() < 0.3) break;
      real_core.handleObstacles(visibleMarkers(pos, scene.obstacles));
      const auto tick = real_core.tick();
      if (!tick.feasible) result.infeasible_any = true;
      ++result.n_ticks;
      gamma_sum += tick.gamma_used;

      const Eigen::Vector3d accel = (tick.T / real_params.vehicle_mass) * (R * Eigen::Vector3d::UnitZ()) -
                                     nodelib::kQuadG * Eigen::Vector3d::UnitZ();
      const double omega_norm = tick.omega.norm();
      if (omega_norm > 1e-9) {
        R = R * Eigen::AngleAxisd(omega_norm * dt, tick.omega / omega_norm).toRotationMatrix();
      }
      vel = vel + accel * dt;
      pos = pos + vel * dt;

      for (const auto& o : scene.obstacles) {
        const double dist_to_surface = obstacleCenterDist(pos, o.center) - o.radius_phys;
        result.min_h_physical = std::min(result.min_h_physical, dist_to_surface);
        result.min_h_margin = std::min(result.min_h_margin, dist_to_surface - margin);
      }

      sim_time_ns += dt_ns;
      rcl_set_ros_time_override(clock->get_clock_handle(), sim_time_ns);
      real_core.setOdometry(pos, vel, R, clock->now());
      ++ticks_done;

      if ((scene.goal - pos).norm() < 0.3) {
        result.reached_goal = true;
        result.t_goal = ticks_done * dt;
        break;
      }
    }
    if (result.reached_goal) break;
  }

  if (!result.reached_goal) result.t_goal = max_time_s;
  result.mean_gamma_used = result.n_ticks > 0 ? gamma_sum / result.n_ticks : 0.0;
  return result;
}

}  // namespace

namespace {

double episodeCost(const EpisodeResult& r, double max_time_s) {
  // Mirrors eval_scene_distribution.cost(): margin-unsafe episodes get a large fixed penalty (so a
  // fast-but-grazing arm never wins on cost), otherwise cost is just time-to-goal.
  return r.min_h_margin < 0.0 ? 2.0 * max_time_s : r.t_goal;
}

}  // namespace

// Resolve default asset paths from $ARC_ROOT (exported by scripts/setup.sh);
// fall back to a repo-root-relative path. Overridable via argv.
static std::string arcPath(const char* rel) {
  const char* root = std::getenv("ARC_ROOT");
  return (root && *root ? std::string(root) + "/" : std::string()) + rel;
}

int main(int argc, char** argv) {
  const std::string scenes_path =
      argc > 1 ? argv[1]
                : arcPath("research/penn_gat/results/"
                          "frozen_20260822/closed_loop_diag_scenes.pt");
  const std::string penn_model_path =
      argc > 2 ? argv[2]
                : arcPath("src/arc/single_vehicle_cbf_rate_arc/"
                          "nn_model/checkpoint/gat_penn_weights.pt");
  const std::string csv_path = argc > 3 ? argv[3] : "";

  const auto seed_sets = loadSeedSource(scenes_path);
  const double dt = 1.0 / 100.0;
  const double max_time_s = 25.0;  // matches eval_scene_distribution.py's MAX_STEPS*DT (500*0.05)

  // GREEDY_SWITCH=1: measures within-episode gamma-switching headroom (see runEpisodeGreedySwitch's
  // comment).
  if (std::getenv("GREEDY_SWITCH") != nullptr) {
    std::vector<double> gammas = {3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0};
    if (const char* list = std::getenv("GREEDY_GAMMAS")) {
      gammas.clear();
      std::string s(list);
      size_t pos = 0;
      while (pos <= s.size()) {
        const size_t comma = s.find(',', pos);
        const std::string tok = s.substr(pos, comma == std::string::npos ? comma : comma - pos);
        if (!tok.empty()) gammas.push_back(std::atof(tok.c_str()));
        if (comma == std::string::npos) break;
        pos = comma + 1;
      }
    }
    const int horizon = std::getenv("GREEDY_HORIZON") ? std::atoi(std::getenv("GREEDY_HORIZON")) : 20;
    const int interval = std::getenv("GREEDY_INTERVAL") ? std::atoi(std::getenv("GREEDY_INTERVAL")) : 10;
    const double proximity_gate =
        std::getenv("GREEDY_PROXIMITY_GATE") ? std::atof(std::getenv("GREEDY_PROXIMITY_GATE")) : 3.0;
    const double infeasible_penalty = std::getenv("GREEDY_INFEASIBLE_PENALTY")
                                          ? std::atof(std::getenv("GREEDY_INFEASIBLE_PENALTY"))
                                          : 1.0;

    std::ofstream csv;
    if (!csv_path.empty()) {
      csv.open(csv_path);
      csv << "seed,scene,n_obs,t_goal,min_h_physical,min_h_margin,reached_goal,infeasible_any,"
             "n_ticks,mean_gamma_used\n";
    }
    std::printf("GREEDY_SWITCH: %zu candidate gammas, horizon=%d ticks, interval=%d ticks, "
                "proximity_gate=%.2fm, infeasible_penalty=%.3f\n\n",
                gammas.size(), horizon, interval, proximity_gate, infeasible_penalty);

    int hard = 0;
    double sum_t = 0.0, sum_h = 0.0;
    size_t n = 0;
    for (const auto& ss : seed_sets) {
      for (size_t s = 0; s < ss.scenes.size(); ++s) {
        RateAutopilotCore::Params params = baseParams();
        const auto r = runEpisodeGreedySwitch(ss.scenes[s], params, dt, max_time_s, gammas, horizon,
                                               interval, proximity_gate, infeasible_penalty);
        if (r.min_h_physical < 0.0) ++hard;
        sum_t += r.t_goal;
        sum_h += r.min_h_physical;
        ++n;
        if (csv.is_open()) {
          csv << ss.seed << ',' << s << ',' << ss.scenes[s].obstacles.size() << ',' << r.t_goal << ','
              << r.min_h_physical << ',' << r.min_h_margin << ',' << (r.reached_goal ? 1 : 0) << ','
              << (r.infeasible_any ? 1 : 0) << ',' << r.n_ticks << ',' << r.mean_gamma_used << '\n';
        }
        std::printf("seed %d scene %2zu: n_obs=%zu t_goal=%.2fs min_h=%.3f %s%s\n", ss.seed, s,
                    ss.scenes[s].obstacles.size(), r.t_goal, r.min_h_physical,
                    r.min_h_physical < 0.0 ? "HARD" : "ok", r.infeasible_any ? " [infeasible]" : "");
      }
    }
    if (csv.is_open()) csv.close();
    std::printf("\ngreedy_switch: hard=%d/%zu mean_t_goal=%.3fs mean_min_h=%.3f\n", hard, n,
                sum_t / n, sum_h / n);
    return 0;
  }

  struct Arm {
    std::string name;
    bool penn_enabled;
    double fixed_gamma;
  };
  // GAMMA_SWEEP=1 replaces the standard arm set with a fine sweep of fixed gammas -- used to locate
  // the deadlock boundary empirically and test the gamma^3 feasible-set-collapse prediction (see …
  std::vector<Arm> arms;
  if (std::getenv("GAMMA_SWEEP") != nullptr) {
    std::vector<double> gammas;
    if (const char* list = std::getenv("GAMMA_LIST")) {
      std::string s(list);
      size_t pos = 0;
      while (pos <= s.size()) {
        const size_t comma = s.find(',', pos);
        const std::string tok = s.substr(pos, comma == std::string::npos ? comma : comma - pos);
        if (!tok.empty()) gammas.push_back(std::atof(tok.c_str()));
        if (comma == std::string::npos) break;
        pos = comma + 1;
      }
    } else {
      gammas = {0.5, 0.7, 0.9, 1.1, 1.3, 1.5, 2.0, 2.5};
    }
    for (double g : gammas) {
      char buf[32];
      std::snprintf(buf, sizeof(buf), "fixed=%.2f", g);
      arms.push_back({buf, false, g});
    }
  } else if (std::getenv("ADAPTIVE_ONLY") != nullptr) {
    // Parameter sweeps (LCB_K=...) only need the adaptive arm: the fixed-gamma arms don't read
    // lcb_k at all, so re-running them once per swept value is ~75% wasted compute and produces …
    arms = {{"adaptive(PENN)", true, 0.0}};
  } else {
    // FIXED_ARMS="0.5,1.5,2.5,3.0" overrides the fixed-gamma baseline set. Added 2026-08-23 because
    // the hardcoded default below is only a FAIR baseline while penn_gamma_max == 2.5.
    std::vector<double> fixed_gammas;
    if (const char* list = std::getenv("FIXED_ARMS")) {
      std::string s(list);
      size_t pos = 0;
      while (pos <= s.size()) {
        const size_t comma = s.find(',', pos);
        const std::string tok = s.substr(pos, comma == std::string::npos ? comma : comma - pos);
        if (!tok.empty()) fixed_gammas.push_back(std::atof(tok.c_str()));
        if (comma == std::string::npos) break;
        pos = comma + 1;
      }
    } else {
      // Must span at least [penn_gamma_min, penn_gamma_max] (now 2.0..9.5, read from the deployed
      // YAML) or the paired-speed comparison silently favours the adaptive arm -- it can select …
      fixed_gammas = {2.0, 3.5, 5.0, 6.5, 8.0, 9.5};
    }
    arms.push_back({"adaptive(PENN)", true, 0.0});
    for (double g : fixed_gammas) {
      char buf[32];
      std::snprintf(buf, sizeof(buf), "fixed=%.1f", g);
      arms.push_back({buf, false, g});
    }
  }
  // Index of "adaptive(PENN)" in arms, if present -- drives the paired-speed
  // report. -1 if this run doesn't include it (e.g. GAMMA_SWEEP mode).
  int adaptive_idx = -1;
  for (size_t a = 0; a < arms.size(); ++a) {
    if (arms[a].penn_enabled) adaptive_idx = static_cast<int>(a);
  }

  std::ofstream csv;
  if (!csv_path.empty()) {
    csv.open(csv_path);
    csv << "seed,arm,scene,n_obs,t_goal,min_h_physical,min_h_margin,reached_goal,"
           "infeasible_any,n_ticks,n_penn_active_ticks,mean_gamma_used\n";
  }

  std::printf("%zu seed(s), %zu arms, dt=%.3fs, max_time=%.1fs, simulated-clock replay\n\n",
              seed_sets.size(), arms.size(), dt, max_time_s);

  // Per-episode rows are only worth printing at small scale (the original 8-scene sanity gate, or a
  // GAMMA_SWEEP run) -- at campaign scale (1400+ episodes) they're noise; the CSV has full per- …
  size_t total_episodes = 0;
  for (const auto& ss : seed_sets) total_episodes += ss.scenes.size() * arms.size();
  const bool verbose = total_episodes <= 64;
  if (verbose) {
    std::printf("%-16s %6s %8s %8s %10s %10s %8s\n", "arm", "seed", "scene", "n_obs", "t_goal",
                "min_h_phys", "HARD?");
    std::printf("--------------------------------------------------------------------------\n");
  }

  std::vector<std::vector<std::vector<EpisodeResult>>> per_seed_arm(seed_sets.size());

  for (size_t si = 0; si < seed_sets.size(); ++si) {
    const auto& scenes = seed_sets[si].scenes;
    per_seed_arm[si].resize(arms.size());

    for (size_t a = 0; a < arms.size(); ++a) {
      RateAutopilotCore::Params params = baseParams();
      params.penn_enabled = arms[a].penn_enabled;
      if (arms[a].penn_enabled) {
        params.penn_model_path = penn_model_path;
        // cbf_gamma_obs is the FALLBACK gamma used whenever PENN does not run -- beyond
        // penn_proximity_gate_m (3m) from the nearest obstacle, with no obstacles in view, or on …
        if (const char* env = std::getenv("CBF_GAMMA_OBS")) {
          params.cbf_gamma_obs = std::atof(env);
        }
      } else {
        params.cbf_gamma_obs = arms[a].fixed_gamma;
      }

      for (size_t s = 0; s < scenes.size(); ++s) {
        // DEBUG_SCENE=<index>, added 2026-08-23 for the entry-certification investigation: prints
        // gamma_used/feasible/closest-obstacle every tick for exactly one scene index, to see the …
        const char* dbg = std::getenv("DEBUG_SCENE");
        const bool debug_ticks = dbg != nullptr && std::atoi(dbg) == static_cast<int>(s);
        if (debug_ticks) {
          std::fprintf(stderr, "=== BEGIN seed=%d arm=%s scene=%zu ===\n", seed_sets[si].seed,
                       arms[a].name.c_str(), s);
        }
        const auto r = runEpisode(scenes[s], params, dt, max_time_s, debug_ticks);
        per_seed_arm[si][a].push_back(r);
        if (csv.is_open()) {
          csv << seed_sets[si].seed << ',' << arms[a].name << ',' << s << ','
              << scenes[s].obstacles.size() << ',' << r.t_goal << ',' << r.min_h_physical << ','
              << r.min_h_margin << ',' << (r.reached_goal ? 1 : 0) << ','
              << (r.infeasible_any ? 1 : 0) << ',' << r.n_ticks << ',' << r.n_penn_active_ticks
              << ',' << r.mean_gamma_used << '\n';
        }
        if (verbose) {
          std::printf("%-16s %6d %6zu %8zu %9.2fs %10.3f %8s%s\n", arms[a].name.c_str(),
                      seed_sets[si].seed, s, scenes[s].obstacles.size(), r.t_goal,
                      r.min_h_physical, r.min_h_physical < 0.0 ? "YES" : "no",
                      r.infeasible_any ? "  [QP infeasible some tick(s)]" : "");
        }
      }
    }

    // Per-seed summary, mirroring reproduce_headline.py's table: hard collisions, margin-
    // violations, and paired speed vs. whichever fixed arm had the lowest mean cost on THIS seed …
    if (adaptive_idx >= 0 && arms.size() > 1) {
      int best_fixed = -1;
      double best_cost = std::numeric_limits<double>::infinity();
      for (size_t a = 0; a < arms.size(); ++a) {
        if (static_cast<int>(a) == adaptive_idx) continue;
        double sum = 0.0;
        for (const auto& r : per_seed_arm[si][a]) sum += episodeCost(r, max_time_s);
        const double mean_cost = sum / per_seed_arm[si][a].size();
        if (mean_cost < best_cost) {
          best_cost = mean_cost;
          best_fixed = static_cast<int>(a);
        }
      }

      const auto& a_res = per_seed_arm[si][adaptive_idx];
      const auto& c_res = per_seed_arm[si][best_fixed];
      int n_hard = 0, n_viol = 0;
      double a_safe_t_sum = 0.0, c_safe_t_sum = 0.0;
      int both_n = 0;
      double a_both_sum = 0.0, c_both_sum = 0.0;
      int a_safe_n = 0;
      for (size_t s = 0; s < a_res.size(); ++s) {
        const bool a_safe = a_res[s].min_h_margin >= 0.0;
        const bool c_safe = c_res[s].min_h_margin >= 0.0;
        if (a_res[s].min_h_physical < 0.0) ++n_hard;
        if (!a_safe) ++n_viol;
        if (a_safe) { a_safe_t_sum += a_res[s].t_goal; ++a_safe_n; }
        if (a_safe && c_safe) {
          a_both_sum += a_res[s].t_goal;
          c_both_sum += c_res[s].t_goal;
          ++both_n;
        }
        if (c_safe) c_safe_t_sum += c_res[s].t_goal;
      }
      const double paired_pct = both_n > 0
          ? 100.0 * (c_both_sum / both_n - a_both_sum / both_n) / (c_both_sum / both_n)
          : std::numeric_limits<double>::quiet_NaN();
      std::printf("seed %d: best_fixed=%s  adaptive %d/%zu margin-viol  %d/%zu HARD  "
                  "paired=%+.1f%% (n_both=%d)\n",
                  seed_sets[si].seed, arms[best_fixed].name.c_str(), n_viol, a_res.size(), n_hard,
                  a_res.size(), paired_pct, both_n);
    }
  }
  if (csv.is_open()) csv.close();

  std::printf("\n%-16s %10s %14s %14s %14s\n", "arm", "hard/N", "mean t_goal", "mean min_h_phys",
              "PENN duty %");
  std::printf("------------------------------------------------------------------------------------\n");
  for (size_t a = 0; a < arms.size(); ++a) {
    int hard = 0;
    double sum_t = 0.0, sum_h = 0.0;
    long long sum_ticks = 0, sum_penn_ticks = 0;
    size_t n = 0;
    for (size_t si = 0; si < seed_sets.size(); ++si) {
      for (const auto& r : per_seed_arm[si][a]) {
        if (r.min_h_physical < 0.0) ++hard;
        sum_t += r.t_goal;
        sum_h += r.min_h_physical;
        sum_ticks += r.n_ticks;
        sum_penn_ticks += r.n_penn_active_ticks;
        ++n;
      }
    }
    const double duty_pct = sum_ticks > 0 ? 100.0 * sum_penn_ticks / sum_ticks : 0.0;
    std::printf("%-16s %6d/%-3zu %13.2fs %14.3f %13.1f%%\n", arms[a].name.c_str(), hard, n,
                sum_t / n, sum_h / n, duty_pct);
  }

  nodelib::printPennSelectorStats();
  return 0;
}
