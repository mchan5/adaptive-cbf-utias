// Phase 2 of the Pareto-reframe plan (2026-08-25, see the "Pareto Reframe"
// plan artifact from today): episode-level adaptive selection failed in
// five formulations because its headroom hides in rare events invisible at
// t=0 (see frozen_widerange_20260824/MANIFEST.md). Per-state switching does
// not have that problem -- an oracle version of it (this project's own
// runEpisodeGreedySwitch, corrected: see body_rate_closed_loop_diag.cpp's
// GREEDY_INFEASIBLE_PENALTY comment for the scorer bug that made it
// unmeasurable until today) beats fixed gamma=8@a8 by 18.7% time at
// identical margin violations on 350 held-out standard-distribution scenes.
//
// This tool exports PER-DECISION-POINT training labels from that same
// corrected oracle, so PENN+GAT can be trained to approximate it from
// observable state alone (no privileged future-rollout access) instead of
// being trained on the old point-mass, whole-episode-from-rest labels.
//
// Design mirrors runEpisodeGreedySwitch (body_rate_closed_loop_diag.cpp)
// exactly -- same decision loop, same proximity gate, same committed
// trajectory -- so the labels come from the states the winning policy
// actually visits, not an arbitrary distribution. At each decision point
// this ALSO does what runEpisodeGreedySwitch's throwaway scoring clones do,
// but records the outcome for every candidate as a training row instead of
// only using it to pick the winner:
//   - progress_deficit: max(0, ideal_progress - actual_progress) over the
//     decision window, ideal_progress = min(v_max*window_s, dist_before).
//     v_max=3.0, matching drone_data_generation.py's label convention (see
//     body_rate_horizon_label_export.cpp's Stage-1 comment for why this
//     constant must match exactly).
//   - min_h_horizon: min over the window of h = ||p-c||^2 - r_eff^2
//     (SQUARED form, r_eff = physical_radius + obs_safety_margin), at the
//     PRE-MOVE position each tick -- same convention as Stage 1's
//     body_rate_horizon_label_export.cpp, verified against
//     drone_data_generation.py:469-486 by direct read.
//
// Difference from Stage 1: R0 there was identity (no attitude source
// available for arbitrary point-mass-matched query states). Here the
// window rollout starts from the REAL current attitude of the real running
// episode, because this tool generates its own episode rather than
// replaying externally-supplied query states -- exactly the
// "self-consistent body-rate-carrier-sourced R0" the original plan flagged
// as required before any Stage-2 label is trusted.
//
// Obstacle visibility filter matches worker()'s obs_visible gate: rows are
// only emitted at decision points where at least one obstacle is currently
// sensed, because the deployed selector is never queried otherwise
// (obstacle_cbf_node.py short-circuits to the default gamma when
// self._obstacles is empty) -- an empty-obstacle graph sample is a state
// distribution the deployed system would never produce.
//
// Output is a flat CSV, not a pickle -- graph construction (edge_index/
// edge_attr via GATModule3D.create_graph) stays in Python exactly as
// worker() already does it, so the pickle schema train_drone.py reads is
// reproduced with zero changes to that file. See
// penn_gat_training/pareto_label_csv_to_pickle.py for the CSV->pickle step.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

#include <rcl/time.h>
#include <torch/csrc/jit/serialization/pickle.h>
#include <torch/torch.h>

#include "single_vehicle_cbf_rate/rate_autopilot_core.hpp"

using nodelib::RateAutopilotCore;

namespace {

constexpr double kSensorMaxRange = 8.0;
constexpr double kSensorZMin = 0.1;
constexpr double kSensorZMax = 2.5;
constexpr double kVMax = 3.0;  // MUST match drone_data_generation.py's v_max default

struct Obstacle {
  Eigen::Vector3d center;
  double radius_phys;
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
      scene.obstacles.push_back({Eigen::Vector3d(acc[k][0], acc[k][1], acc[k][2]), acc[k][3]});
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
    const std::string stem = f.stem().string();
    const int seed = std::atoi(stem.c_str() + std::string("scenes_seed").size());
    out.push_back({seed, loadScenes(f.string())});
  }
  return out;
}

RateAutopilotCore::Params baseParams() {
  RateAutopilotCore::Params p;
  p.vehicle_mass = 2.064;
  p.obs_d_min = 1.0;
  p.obs_safety_margin = 0.5;
  p.control_rate_hz = 100.0;
  // Corrected 2026-08-26 (PLAN_adaptive_recovery §0.2): this tool's
  // defaults were (13.0, 6.5, 6.5), the exact operating point an earlier
  // SITL investigation rejected -- the entire Pareto campaign
  // (frozen_widerange_20260824, pareto_phase1/2_20260825) was generated at
  // an authority setting the project doesn't ship. Matches
  // body_rate_closed_loop_diag.cpp's validated corrected-authority value.
  p.nom_accel_max = std::getenv("NOM_ACCEL_MAX") ? std::atof(std::getenv("NOM_ACCEL_MAX")) : 12.0;
  p.nom_pos_kp = std::getenv("NOM_POS_KP") ? std::atof(std::getenv("NOM_POS_KP")) : 6.0;
  p.nom_vel_kd = std::getenv("NOM_VEL_KD") ? std::atof(std::getenv("NOM_VEL_KD")) : 6.0;
  p.qp_thrust_min = 4.7;
  p.qp_thrust_floor_gain = 8.0;
  p.qp_tau_min = -50.0;
  p.qp_tau_max = 50.0;
  p.qp_omega_max = 6.0;
  p.qp_rho = 1.0;
  p.qp_slack_weight = 1e4;
  const double vehicle_thrust_scaling = 0.08259793214;
  const double vehicle_idle_thrust = 0.2535571429;
  const int vehicle_num_rotors = 4;
  p.T_max = (1.0 - vehicle_idle_thrust) / vehicle_thrust_scaling * vehicle_num_rotors;
  p.penn_enabled = false;
  return p;
}

visualization_msgs::msg::MarkerArray visibleMarkers(const Eigen::Vector3d& pos,
                                                      const std::vector<Obstacle>& obstacles) {
  visualization_msgs::msg::MarkerArray markers;
  for (size_t i = 0; i < obstacles.size(); ++i) {
    const auto& o = obstacles[i];
    if (o.center.z() < kSensorZMin || o.center.z() > kSensorZMax) continue;
    const double dist_to_surface = (pos - o.center).norm() - o.radius_phys;
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

std::vector<const Obstacle*> visibleObstacles(const Eigen::Vector3d& pos,
                                               const std::vector<Obstacle>& obstacles) {
  std::vector<const Obstacle*> out;
  for (const auto& o : obstacles) {
    if (o.center.z() < kSensorZMin || o.center.z() > kSensorZMax) continue;
    const double dist_to_surface = (pos - o.center).norm() - o.radius_phys;
    if (dist_to_surface > kSensorMaxRange) continue;
    out.push_back(&o);
  }
  return out;
}

struct LabelResult {
  double min_h_horizon{std::numeric_limits<double>::infinity()};  // squared HOCBF form
  double progress_deficit{0.0};
  bool infeasible_any{false};
};

// Mirrors simulateWindow (body_rate_closed_loop_diag.cpp) exactly for the
// trajectory/feasibility side, and rolloutLabel
// (body_rate_horizon_label_export.cpp) exactly for the squared-h/progress-
// deficit label side -- this is genuinely both at once, computed together
// so the two tools' formulas don't have to be kept in sync by hand.
LabelResult scoreCandidate(const Scene& scene, RateAutopilotCore::Params params, double dt,
                            const Eigen::Vector3d& pos0, const Eigen::Vector3d& vel0,
                            const Eigen::Matrix3d& R0, int64_t sim_time_ns0, int max_ticks,
                            double obs_safety_margin) {
  auto logger = rclcpp::get_logger("pareto_label_export_scorer");
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

  LabelResult result;
  const double dist0 = (scene.goal - pos).norm();
  double dist_final = dist0;

  for (int k = 0; k < max_ticks; ++k) {
    if ((scene.goal - pos).norm() < 0.3) break;

    for (const auto& o : scene.obstacles) {
      const double r_eff = o.radius_phys + obs_safety_margin;
      const double h = (pos - o.center).squaredNorm() - r_eff * r_eff;
      result.min_h_horizon = std::min(result.min_h_horizon, h);
    }

    core.handleObstacles(visibleMarkers(pos, scene.obstacles));
    const auto tick = core.tick();
    if (!tick.feasible) result.infeasible_any = true;

    const Eigen::Vector3d accel = (tick.T / params.vehicle_mass) * (R * Eigen::Vector3d::UnitZ()) -
                                   nodelib::kQuadG * Eigen::Vector3d::UnitZ();
    const double omega_norm = tick.omega.norm();
    if (omega_norm > 1e-9) {
      R = R * Eigen::AngleAxisd(omega_norm * dt, tick.omega / omega_norm).toRotationMatrix();
    }
    vel = vel + accel * dt;
    pos = pos + vel * dt;
    dist_final = (scene.goal - pos).norm();

    sim_time_ns += dt_ns;
    rcl_set_ros_time_override(clock->get_clock_handle(), sim_time_ns);
    core.setOdometry(pos, vel, R, clock->now());
  }

  if (!std::isfinite(result.min_h_horizon)) result.min_h_horizon = 0.0;
  const double window_s = static_cast<double>(max_ticks) * dt;
  const double ideal_progress = std::min(kVMax * window_s, dist0);
  const double actual_progress = dist0 - dist_final;
  result.progress_deficit = std::max(0.0, ideal_progress - actual_progress);
  return result;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 3) {
    std::fprintf(stderr, "usage: %s <scenes_dir_or_file> <out.csv>\n", argv[0]);
    return 1;
  }
  const std::string scenes_path = argv[1];
  const std::string csv_path = argv[2];

  const auto seed_sets = loadSeedSource(scenes_path);
  const double dt = 1.0 / 100.0;
  const double max_time_s = 25.0;
  const int max_steps = static_cast<int>(max_time_s / dt);

  // Widened 4 -> 6 candidates (2026-08-26, PLAN_adaptive_recovery §2.1):
  // the Pareto set's 4-candidates/state training data was still 33.8%
  // gamma-invariant on min_h; more candidates per state raises the odds a
  // given decision point actually shows within-state contrast on top of
  // the decision-weight emission gate above. Still overridable via GAMMAS.
  std::vector<double> candidate_gammas = {2.0, 3.5, 5.0, 6.5, 8.0, 9.5};
  if (const char* list = std::getenv("GAMMAS")) {
    candidate_gammas.clear();
    std::string s(list);
    size_t pos = 0;
    while (pos <= s.size()) {
      const size_t comma = s.find(',', pos);
      const std::string tok = s.substr(pos, comma == std::string::npos ? comma : comma - pos);
      if (!tok.empty()) candidate_gammas.push_back(std::atof(tok.c_str()));
      if (comma == std::string::npos) break;
      pos = comma + 1;
    }
  }
  const int horizon_ticks = std::getenv("HORIZON") ? std::atoi(std::getenv("HORIZON")) : 150;
  const int decision_interval_ticks = std::getenv("INTERVAL") ? std::atoi(std::getenv("INTERVAL")) : 25;
  const double proximity_gate_m =
      std::getenv("PROXIMITY_GATE") ? std::atof(std::getenv("PROXIMITY_GATE")) : 3.0;

  const RateAutopilotCore::Params base_params = baseParams();
  const double margin = base_params.obs_safety_margin;

  std::ofstream csv(csv_path);
  csv << "seed,scene,decision_idx,tick,px,py,pz,vx,vy,vz,gx,gy,gz,obstacles,"
         "gamma,progress_deficit,min_h_horizon,infeasible\n";

  std::fprintf(stderr, "%zu seed(s), %zu candidate gammas, horizon=%d ticks, interval=%d ticks, "
                        "authority=(%.1f,%.1f,%.1f)\n",
                seed_sets.size(), candidate_gammas.size(), horizon_ticks, decision_interval_ticks,
                base_params.nom_accel_max, base_params.nom_pos_kp, base_params.nom_vel_kd);

  size_t total_rows = 0;
  for (const auto& ss : seed_sets) {
    for (size_t s = 0; s < ss.scenes.size(); ++s) {
      const auto& scene = ss.scenes[s];

      auto logger = rclcpp::get_logger("pareto_label_export");
      auto clock = std::make_shared<rclcpp::Clock>(RCL_ROS_TIME);
      rcl_enable_ros_time_override(clock->get_clock_handle());
      int64_t sim_time_ns = 1000000000LL;
      rcl_set_ros_time_override(clock->get_clock_handle(), sim_time_ns);
      const int64_t dt_ns = static_cast<int64_t>(dt * 1e9);

      RateAutopilotCore::Params real_params = base_params;
      real_params.waypoint = scene.goal;
      real_params.cbf_gamma_obs = candidate_gammas.front();
      RateAutopilotCore real_core(real_params, logger, clock);

      Eigen::Vector3d pos = scene.start;
      Eigen::Vector3d vel = Eigen::Vector3d::Zero();
      Eigen::Matrix3d R = Eigen::Matrix3d::Identity();
      real_core.setOdometry(pos, vel, R, clock->now());

      double current_gamma = candidate_gammas.front();
      int ticks_done = 0;
      int decision_idx = 0;

      while (ticks_done < max_steps) {
        if ((scene.goal - pos).norm() < 0.3) break;

        double min_center_dist = std::numeric_limits<double>::infinity();
        for (const auto& o : scene.obstacles) {
          min_center_dist = std::min(min_center_dist, (pos - o.center).norm());
        }
        const int window = std::min(horizon_ticks, max_steps - ticks_done);
        int best_idx = -1;

        if (min_center_dist > proximity_gate_m) {
          const auto visible = visibleObstacles(pos, scene.obstacles);
          // Decision-weight emission (2026-08-26, PLAN_adaptive_recovery
          // §2.2): port of drone_data_generation.py's _decision_weight().
          // The old gate emitted a row at EVERY decision point with a
          // visible obstacle inside proximity_gate_m, uniformly -- most of
          // that 3m band is cruise-through space where gamma doesn't change
          // the outcome (measured: 33.8%/50.0% gamma-invariant labels in
          // the data this produced, body_rate_label_test_20260825). This
          // instead accepts a decision point with probability rising
          // exponentially as the nearest visible obstacle's EFFECTIVE
          // boundary (radius+margin, not center) is approached, plus a
          // floor so moderate-clearance states are still sampled sometimes
          // -- same decay/floor constants as the point-mass generator by
          // default, overridable for this exporter specifically.
          double emit_prob = 0.0;
          if (!visible.empty()) {
            double min_gap = std::numeric_limits<double>::infinity();
            for (const auto* o : visible) {
              min_gap = std::min(min_gap, (pos - o->center).norm() - (o->radius_phys + margin));
            }
            static const double kDecay =
                std::getenv("DECISION_WEIGHT_DECAY") ? std::atof(std::getenv("DECISION_WEIGHT_DECAY")) : 1.0;
            static const double kFloor =
                std::getenv("DECISION_WEIGHT_FLOOR") ? std::atof(std::getenv("DECISION_WEIGHT_FLOOR")) : 0.05;
            emit_prob = std::min(1.0, kFloor + std::exp(-std::max(0.0, min_gap) / kDecay));
          }
          const bool emit = !visible.empty() &&
                            (static_cast<double>(std::rand()) / RAND_MAX) < emit_prob;
          if (emit) {
            double best_score = std::numeric_limits<double>::infinity();
            std::vector<LabelResult> labels(candidate_gammas.size());
            for (size_t ci = 0; ci < candidate_gammas.size(); ++ci) {
              RateAutopilotCore::Params p = base_params;
              p.penn_enabled = false;
              p.cbf_gamma_obs = candidate_gammas[ci];
              labels[ci] = scoreCandidate(scene, p, dt, pos, vel, R, sim_time_ns, window, margin);

              // Minimizing progress_deficit is equivalent to maximizing raw
              // progress (same ideal_progress for every candidate at a given
              // decision point, since window length is shared) -- matches
              // runEpisodeGreedySwitch's winning scorer at
              // GREEDY_INFEASIBLE_PENALTY=0 (infeasibility term dropped,
              // safety veto kept). min_h_horizon<0 is the squared-form
              // equivalent of that scorer's min_h_margin<0 veto.
              double score = labels[ci].progress_deficit;
              if (labels[ci].min_h_horizon < 0.0) score += 1000.0;
              if (score < best_score) {
                best_score = score;
                best_idx = static_cast<int>(ci);
              }
            }

            std::ostringstream obs_field;
            for (size_t oi = 0; oi < visible.size(); ++oi) {
              if (oi) obs_field << ';';
              obs_field << visible[oi]->center.x() << ',' << visible[oi]->center.y() << ','
                        << visible[oi]->center.z() << ',' << (visible[oi]->radius_phys + margin);
            }
            for (size_t ci = 0; ci < candidate_gammas.size(); ++ci) {
              csv << (ss.seed < 0 ? 0 : ss.seed) << ',' << s << ',' << decision_idx << ','
                  << ticks_done << ',' << pos.x() << ',' << pos.y() << ',' << pos.z() << ','
                  << vel.x() << ',' << vel.y() << ',' << vel.z() << ',' << scene.goal.x() << ','
                  << scene.goal.y() << ',' << scene.goal.z() << ',' << '"' << obs_field.str()
                  << '"' << ',' << candidate_gammas[ci] << ',' << labels[ci].progress_deficit
                  << ',' << labels[ci].min_h_horizon << ',' << (labels[ci].infeasible_any ? 1 : 0)
                  << '\n';
              ++total_rows;
            }
            ++decision_idx;
          }
        }

        if (best_idx >= 0) current_gamma = candidate_gammas[best_idx];
        real_core.setFixedGamma(current_gamma);
        const int commit = std::min(decision_interval_ticks, max_steps - ticks_done);
        bool reached = false;
        for (int k = 0; k < commit; ++k) {
          if ((scene.goal - pos).norm() < 0.3) {
            reached = true;
            break;
          }
          real_core.handleObstacles(visibleMarkers(pos, scene.obstacles));
          const auto tick = real_core.tick();

          const Eigen::Vector3d accel = (tick.T / real_params.vehicle_mass) *
                                             (R * Eigen::Vector3d::UnitZ()) -
                                         nodelib::kQuadG * Eigen::Vector3d::UnitZ();
          const double omega_norm = tick.omega.norm();
          if (omega_norm > 1e-9) {
            R = R * Eigen::AngleAxisd(omega_norm * dt, tick.omega / omega_norm).toRotationMatrix();
          }
          vel = vel + accel * dt;
          pos = pos + vel * dt;

          sim_time_ns += dt_ns;
          rcl_set_ros_time_override(clock->get_clock_handle(), sim_time_ns);
          real_core.setOdometry(pos, vel, R, clock->now());
          ++ticks_done;

          if ((scene.goal - pos).norm() < 0.3) {
            reached = true;
            break;
          }
        }
        if (reached) break;
      }

      if ((s + 1) % 25 == 0) {
        std::fprintf(stderr, "  seed=%d scene %zu/%zu, %zu rows so far\n", ss.seed, s + 1,
                      ss.scenes.size(), total_rows);
      }
    }
  }
  csv.close();
  std::printf("Wrote %zu label rows to %s\n", total_rows, csv_path.c_str());
  return 0;
}
