// Tier 0 unit tests -- gtest port of core_sanity_check.cpp's three
// physical-sanity scenarios (previously print-and-eyeball, now real
// assertions). See penn_gat_training/TESTING.md for the full V&V pyramid
// this fits into. No ROS init required (rclcpp::Logger/Clock work
// standalone), no PENN model needed (penn_enabled: false).
#include <gtest/gtest.h>

#include "single_vehicle_cbf_rate/rate_autopilot_core.hpp"

using nodelib::RateAutopilotCore;

namespace {

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
  const double vehicle_thrust_scaling = 0.08259793214;
  const double vehicle_idle_thrust = 0.2535571429;
  const int vehicle_num_rotors = 4;
  p.T_max = (1.0 - vehicle_idle_thrust) / vehicle_thrust_scaling * vehicle_num_rotors;
  p.penn_enabled = false;
  p.waypoint = Eigen::Vector3d(0.0, 5.5, 1.5);
  return p;
}

rclcpp::Clock::SharedPtr steadyClock() {
  return std::make_shared<rclcpp::Clock>(RCL_STEADY_TIME);
}

}  // namespace

TEST(CoreSanity, HoveringAtWaypointSettlesNearHoverThrust) {
  auto logger = rclcpp::get_logger("core_sanity_check_test");
  auto clock = steadyClock();
  RateAutopilotCore::Params p = f450SimParams();
  RateAutopilotCore core(p, logger, clock);
  core.setOdometry(p.waypoint, Eigen::Vector3d::Zero(), Eigen::Matrix3d::Identity(), clock->now());

  RateAutopilotCore::TickResult result;
  for (int i = 0; i < 20; ++i) {
    result = core.tick();
  }

  EXPECT_TRUE(result.feasible);
  const double hover_thrust = p.vehicle_mass * nodelib::kQuadG;
  EXPECT_NEAR(result.T, hover_thrust, 0.05);
  EXPECT_LT(result.omega.norm(), 0.05);
  EXPECT_EQ(result.certified_count, 0u);
  EXPECT_EQ(result.uncertified_count, 0u);
}

TEST(CoreSanity, DisplacedFromWaypointCommandsNonzeroRateTowardIt) {
  auto logger = rclcpp::get_logger("core_sanity_check_test");
  auto clock = steadyClock();
  RateAutopilotCore::Params p = f450SimParams();
  RateAutopilotCore core(p, logger, clock);
  const Eigen::Vector3d displaced = p.waypoint - Eigen::Vector3d(2.0, 0.0, 0.0);
  core.setOdometry(displaced, Eigen::Vector3d::Zero(), Eigen::Matrix3d::Identity(), clock->now());

  const auto result = core.tick();
  EXPECT_TRUE(result.feasible);
  EXPECT_GT(result.omega.norm(), 1e-3);
}

TEST(CoreSanity, NearbyUnseenObstacleRoutesThroughUncertifiedRow) {
  auto logger = rclcpp::get_logger("core_sanity_check_test");
  auto clock = steadyClock();
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
  EXPECT_TRUE(result.feasible);
  // First sighting -> not yet certified -> routed through the uncertified
  // (slack-relaxed) row, not the certified one.
  EXPECT_EQ(result.uncertified_count, 1u);
  EXPECT_EQ(result.certified_count, 0u);
}
