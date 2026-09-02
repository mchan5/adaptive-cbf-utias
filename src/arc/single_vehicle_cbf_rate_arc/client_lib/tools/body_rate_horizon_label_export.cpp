// Stage 1 of the body-rate-native training-label hypothesis test (2026-08-25 plan): the PENN risk
// head's predictions turn backwards above ~gamma=4.5 (higher gamma predicted MORE dangerous), and …
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iterator>
#include <limits>
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
constexpr double kVMax = 3.0;       // MUST match drone_data_generation.py's v_max default
constexpr double kHorizonS = 4.0;   // MUST match drone_data_generation.py's HORIZON_S

struct Obstacle {
  Eigen::Vector3d center;
  double radius_phys;  // physical radius, NO safety margin
};

struct QueryRow {
  int row_id{0};
  Eigen::Vector3d pos0, vel0, goal;
  std::vector<Obstacle> obstacles;
  std::vector<double> gammas;
};

std::vector<QueryRow> loadQueryRows(const std::string& path) {
  std::ifstream file(path, std::ios::binary);
  if (!file) {
    std::fprintf(stderr, "cannot open %s\n", path.c_str());
    std::exit(1);
  }
  std::vector<char> data((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
  const torch::IValue loaded = torch::jit::pickle_load(data);
  const auto list = loaded.toList();

  std::vector<QueryRow> rows;
  for (size_t i = 0; i < list.size(); ++i) {
    const auto dict = list.get(i).toGenericDict();
    QueryRow row;
    row.row_id = static_cast<int>(dict.at("row_id").toInt());
    const torch::Tensor pos0_t = dict.at("pos0").toTensor();
    const torch::Tensor vel0_t = dict.at("vel0").toTensor();
    const torch::Tensor goal_t = dict.at("goal").toTensor();
    const torch::Tensor obs_t = dict.at("obstacles").toTensor();
    const torch::Tensor gammas_t = dict.at("gammas").toTensor();

    row.pos0 = Eigen::Vector3d(pos0_t[0].item<double>(), pos0_t[1].item<double>(), pos0_t[2].item<double>());
    row.vel0 = Eigen::Vector3d(vel0_t[0].item<double>(), vel0_t[1].item<double>(), vel0_t[2].item<double>());
    row.goal = Eigen::Vector3d(goal_t[0].item<double>(), goal_t[1].item<double>(), goal_t[2].item<double>());

    const auto obs_acc = obs_t.accessor<double, 2>();
    for (int64_t k = 0; k < obs_t.size(0); ++k) {
      row.obstacles.push_back({Eigen::Vector3d(obs_acc[k][0], obs_acc[k][1], obs_acc[k][2]), obs_acc[k][3]});
    }
    const auto gammas_acc = gammas_t.accessor<double, 1>();
    for (int64_t k = 0; k < gammas_t.size(0); ++k) {
      row.gammas.push_back(gammas_acc[k]);
    }
    rows.push_back(std::move(row));
  }
  return rows;
}

RateAutopilotCore::Params baseParams() {
  RateAutopilotCore::Params p;
  p.vehicle_mass = 2.064;
  p.obs_d_min = 1.0;
  p.obs_safety_margin = 0.5;
  p.control_rate_hz = 100.0;
  p.nom_accel_max = 8.0;
  p.nom_pos_kp = 4.0;
  p.nom_vel_kd = 4.0;
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

struct LabelResult {
  double min_h_horizon{std::numeric_limits<double>::infinity()};
  double progress_deficit{0.0};
  bool infeasible_any{false};
  bool reached_goal{false};
  int n_ticks{0};
};

LabelResult rolloutLabel(const QueryRow& row, double gamma, double dt, int max_ticks,
                          double obs_safety_margin) {
  auto logger = rclcpp::get_logger("body_rate_horizon_label_export");
  auto clock = std::make_shared<rclcpp::Clock>(RCL_ROS_TIME);
  rcl_enable_ros_time_override(clock->get_clock_handle());
  int64_t sim_time_ns = 1000000000LL;
  rcl_set_ros_time_override(clock->get_clock_handle(), sim_time_ns);
  const int64_t dt_ns = static_cast<int64_t>(dt * 1e9);

  RateAutopilotCore::Params p = baseParams();
  p.cbf_gamma_obs = gamma;
  p.waypoint = row.goal;
  RateAutopilotCore core(p, logger, clock);

  Eigen::Vector3d pos = row.pos0;
  Eigen::Vector3d vel = row.vel0;
  Eigen::Matrix3d R = Eigen::Matrix3d::Identity();
  core.setOdometry(pos, vel, R, clock->now());

  LabelResult result;
  const double dist0 = (row.goal - pos).norm();
  double dist_final = dist0;

  for (int k = 0; k < max_ticks; ++k) {
    if ((row.goal - pos).norm() < 0.3) {
      result.reached_goal = true;
      break;
    }

    for (const auto& o : row.obstacles) {
      const double r_eff = o.radius_phys + obs_safety_margin;
      const double h = (pos - o.center).squaredNorm() - r_eff * r_eff;
      result.min_h_horizon = std::min(result.min_h_horizon, h);
    }

    core.handleObstacles(visibleMarkers(pos, row.obstacles));
    const auto tick = core.tick();
    if (!tick.feasible) result.infeasible_any = true;
    ++result.n_ticks;

    const Eigen::Vector3d accel = (tick.T / p.vehicle_mass) * (R * Eigen::Vector3d::UnitZ()) -
                                   nodelib::kQuadG * Eigen::Vector3d::UnitZ();
    const double omega_norm = tick.omega.norm();
    if (omega_norm > 1e-9) {
      R = R * Eigen::AngleAxisd(omega_norm * dt, tick.omega / omega_norm).toRotationMatrix();
    }
    vel = vel + accel * dt;
    pos = pos + vel * dt;
    dist_final = (row.goal - pos).norm();

    sim_time_ns += dt_ns;
    rcl_set_ros_time_override(clock->get_clock_handle(), sim_time_ns);
    core.setOdometry(pos, vel, R, clock->now());
  }

  if (!std::isfinite(result.min_h_horizon)) result.min_h_horizon = 0.0;  // matches np.isfinite fallback
  const double horizon_s = static_cast<double>(max_ticks) * dt;
  const double ideal_progress = std::min(kVMax * horizon_s, dist0);
  const double actual_progress = dist0 - dist_final;
  result.progress_deficit = std::max(0.0, ideal_progress - actual_progress);
  return result;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 3) {
    std::fprintf(stderr, "usage: %s <query_rows.pt> <out.csv>\n", argv[0]);
    return 1;
  }
  const std::string rows_path = argv[1];
  const std::string csv_path = argv[2];

  const auto rows = loadQueryRows(rows_path);
  const double dt = 1.0 / 100.0;
  const int max_ticks = static_cast<int>(std::round(kHorizonS / dt));  // 400 @ 100Hz
  const double obs_safety_margin = baseParams().obs_safety_margin;

  std::ofstream csv(csv_path);
  csv << "row_id,gamma,min_h_horizon,progress_deficit,infeasible_any,reached_goal,n_ticks\n";

  size_t done = 0;
  for (const auto& row : rows) {
    for (const double gamma : row.gammas) {
      const auto r = rolloutLabel(row, gamma, dt, max_ticks, obs_safety_margin);
      csv << row.row_id << ',' << gamma << ',' << r.min_h_horizon << ',' << r.progress_deficit << ','
          << (r.infeasible_any ? 1 : 0) << ',' << (r.reached_goal ? 1 : 0) << ',' << r.n_ticks << '\n';
    }
    ++done;
    if (done % 50 == 0) {
      std::fprintf(stderr, "  %zu/%zu rows\n", done, rows.size());
    }
  }
  csv.close();
  std::printf("Wrote %zu rows x %zu gammas to %s\n", rows.size(),
              rows.empty() ? 0 : rows[0].gammas.size(), csv_path.c_str());
  return 0;
}
