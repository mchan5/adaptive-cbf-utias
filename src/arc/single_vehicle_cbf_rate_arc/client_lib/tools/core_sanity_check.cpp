// Standalone regression check for RateAutopilotCore, run after the
// rate_autopilot_core/single_vehicle_cbf_rate_client split (see client_lib/CMakeLists.txt) to …
#include <cmath>
#include <cstdio>
#include <cstdlib>

#include "single_vehicle_cbf_rate/rate_autopilot_core.hpp"

using nodelib::RateAutopilotCore;

namespace {

int g_failures = 0;

void check(bool cond, const char* what) {
  if (!cond) {
    std::fprintf(stderr, "FAIL: %s\n", what);
    ++g_failures;
  } else {
    std::printf("ok:   %s\n", what);
  }
}

RateAutopilotCore::Params f450SimParams() {
  // Mirrors params_single_vehicle_cbf_rate_arc.yaml's defaults.
  RateAutopilotCore::Params p;
  p.vehicle_mass = 2.064;
  p.obs_d_min = 1.0;
  p.obs_safety_margin = 0.5;
  p.control_rate_hz = 100.0;
  p.cbf_gamma_obs = 4.0;
  p.qp_thrust_min = 4.7;
  p.qp_thrust_floor_gain = 8.0;
  p.qp_tau_min = -50.0;
  p.qp_tau_max = 50.0;
  p.qp_omega_max = 6.0;
  p.qp_rho = 1.0;
  p.qp_slack_weight = 1e4;
  // Same computeThrustMax() formula as the PX4 client, with the SITL x500
  // thrust-curve constants.
  const double vehicle_thrust_scaling = 0.08259793214;
  const double vehicle_idle_thrust = 0.2535571429;
  const int vehicle_num_rotors = 4;
  p.T_max = (1.0 - vehicle_idle_thrust) / vehicle_thrust_scaling * vehicle_num_rotors;
  p.penn_enabled = false;  // no model file available in this standalone check
  p.waypoint = Eigen::Vector3d(0.0, 5.5, 1.5);
  return p;
}

}  // namespace

int main() {
  auto logger = rclcpp::get_logger("core_sanity_check");
  auto clock = std::make_shared<rclcpp::Clock>(RCL_STEADY_TIME);

  // Scenario 1: hovering exactly at the waypoint, at rest
  {
    RateAutopilotCore::Params p = f450SimParams();
    RateAutopilotCore core(p, logger, clock);
    core.setOdometry(p.waypoint, Eigen::Vector3d::Zero(), Eigen::Matrix3d::Identity(), clock->now());

    RateAutopilotCore::TickResult result;
    for (int i = 0; i < 20; ++i) {
      result = core.tick();
    }

    check(result.feasible, "hover: QP feasible");
    const double hover_thrust = p.vehicle_mass * nodelib::kQuadG;
    check(std::abs(result.T - hover_thrust) < 0.05,
          "hover: settles near mg thrust (within 0.05N)");
    check(result.omega.norm() < 0.05, "hover: near-zero body rate at steady state");
    check(result.certified_count == 0 && result.uncertified_count == 0,
          "hover: no obstacles seen -> zero obstacle rows");
    std::printf("  T=%.4fN (mg=%.4fN) omega=[%.4f %.4f %.4f]\n", result.T, hover_thrust,
                result.omega.x(), result.omega.y(), result.omega.z());
  }

  // ---- Scenario 2: displaced from the waypoint -> should command a
  // nonzero pitch/roll rate toward it, not just sit still. ----------------
  {
    RateAutopilotCore::Params p = f450SimParams();
    RateAutopilotCore core(p, logger, clock);
    const Eigen::Vector3d displaced = p.waypoint - Eigen::Vector3d(2.0, 0.0, 0.0);
    core.setOdometry(displaced, Eigen::Vector3d::Zero(), Eigen::Matrix3d::Identity(), clock->now());

    const auto result = core.tick();
    check(result.feasible, "displaced: QP feasible");
    check(result.omega.norm() > 1e-3, "displaced: nonzero body rate commanded toward goal");
    std::printf("  T=%.4fN omega=[%.4f %.4f %.4f]\n", result.T, result.omega.x(), result.omega.y(),
                result.omega.z());
  }

  // ---- Scenario 3: an obstacle directly between the vehicle and goal,
  // close enough that the QP's collision row should be active. ------------
  {
    RateAutopilotCore::Params p = f450SimParams();
    RateAutopilotCore core(p, logger, clock);
    const Eigen::Vector3d start(0.0, 0.0, 1.5);
    core.setOdometry(start, Eigen::Vector3d(0.0, 1.0, 0.0), Eigen::Matrix3d::Identity(), clock->now());

    visualization_msgs::msg::MarkerArray obstacles;
    visualization_msgs::msg::Marker m;
    m.id = 1;
    m.pose.position.x = 0.0;
    m.pose.position.y = 1.0;
    m.pose.position.z = 1.5;
    m.scale.x = 1.0;  // diameter
    obstacles.markers.push_back(m);
    core.handleObstacles(obstacles);

    const auto result = core.tick();
    check(result.feasible, "obstacle: QP still feasible with one nearby obstacle");
    // First sighting -> not yet certified -> should be routed through the
    // uncertified (slack-relaxed) row, not the certified one.
    check(result.uncertified_count == 1 && result.certified_count == 0,
          "obstacle: first-sighting obstacle is uncertified, not certified");
    std::printf("  T=%.4fN omega=[%.4f %.4f %.4f] delta=%.4f\n", result.T, result.omega.x(),
                result.omega.y(), result.omega.z(), result.delta);
  }

  if (g_failures > 0) {
    std::fprintf(stderr, "\n%d check(s) FAILED\n", g_failures);
    return 1;
  }
  std::printf("\nall checks passed\n");
  return 0;
}
