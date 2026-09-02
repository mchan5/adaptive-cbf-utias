// Tier 0 unit test: RateAutopilotCore::pennSelectAlpha's proximity-gate
// logic (penn_proximity_gate_m) -- guards against a silent regression of
// the exact mechanism that made this project's within-episode gamma-
// switching experiment unsafe on 2026-08-24 (see TESTING.md and
// results/ history): switching gamma is only safe far from obstacles, and
// the gate is what enforces that in the deployed selector. This is not a
// hypothetical -- an earlier diagnostic harness that switched gamma
// without respecting this gate produced real hard collisions.
//
// Loads the real deploy checkpoint (same convention as
// penn_gamma_parity_check.cpp / penn_gamma_diagnostic.cpp / body_rate_
// closed_loop_diag.cpp -- this project treats checkpoint-loading tools as
// ordinary diagnostics, not something requiring a mock). Deterministic:
// PENN inference is a pure function of fixed weights + input, no RNG.
#include <cstdlib>
#include <string>

#include <gtest/gtest.h>

#include "single_vehicle_cbf_rate/rate_autopilot_core.hpp"

using nodelib::RateAutopilotCore;

namespace {

// Resolve from $ARC_ROOT (exported by scripts/setup.sh); fall back to a
// repo-root-relative path when run from the workspace root.
const std::string kCheckpointPath = [] {
  const char* root = std::getenv("ARC_ROOT");
  return (root && *root ? std::string(root) + "/" : std::string()) +
         "src/arc/single_vehicle_cbf_rate_arc/nn_model/checkpoint/gat_penn_weights.pt";
}();

RateAutopilotCore::Params basePennParams() {
  RateAutopilotCore::Params p;
  p.vehicle_mass = 2.064;
  p.obs_d_min = 1.0;
  p.obs_safety_margin = 0.5;
  p.control_rate_hz = 100.0;
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

  p.penn_enabled = true;
  p.penn_model_path = kCheckpointPath;
  p.penn_gamma_min = 1.3;
  p.penn_gamma_max = 8.0;
  p.penn_gamma_steps = 20;
  p.penn_update_ticks = 1;  // re-select every tick so the test isn't cadence-flaky
  p.penn_proximity_gate_m = 3.0;
  // Deliberately OUTSIDE [penn_gamma_min, penn_gamma_max] (1.3-8.0) -- makes
  // the two test cases below unambiguous. If the fallback were inside the
  // candidate range (e.g. 4.0, the actual deploy value), a gate bug that
  // silently always falls back could still coincidentally land inside
  // [1.3, 8.0] and pass the "within gate" test for the wrong reason.
  p.cbf_gamma_obs = 0.1;
  p.waypoint = Eigen::Vector3d(0.0, 5.5, 1.5);
  return p;
}

visualization_msgs::msg::MarkerArray oneObstacleAt(double x, double y, double z, double diameter) {
  visualization_msgs::msg::MarkerArray obstacles;
  visualization_msgs::msg::Marker m;
  m.id = 1;
  m.pose.position.x = x;
  m.pose.position.y = y;
  m.pose.position.z = z;
  m.scale.x = diameter;
  obstacles.markers.push_back(m);
  return obstacles;
}

}  // namespace

TEST(PennProximityGate, BeyondGateFallsBackToFixedGamma) {
  auto logger = rclcpp::get_logger("penn_gamma_gate_test");
  auto clock = std::make_shared<rclcpp::Clock>(RCL_STEADY_TIME);
  RateAutopilotCore::Params p = basePennParams();
  RateAutopilotCore core(p, logger, clock);
  ASSERT_TRUE(core.pennModelLoaded()) << "checkpoint failed to load from " << kCheckpointPath;

  const Eigen::Vector3d start(0.0, 0.0, 1.5);
  core.setOdometry(start, Eigen::Vector3d::Zero(), Eigen::Matrix3d::Identity(), clock->now());
  // Obstacle center 5m away -- beyond penn_proximity_gate_m (3.0).
  core.handleObstacles(oneObstacleAt(0.0, 5.0, 1.5, 1.0));

  const auto result = core.tick();
  EXPECT_DOUBLE_EQ(result.gamma_used, p.cbf_gamma_obs)
      << "gate must hold the fixed fallback gamma when the nearest obstacle is beyond "
         "penn_proximity_gate_m -- switching this far out is untested and was never the "
         "regime PENN was calibrated for";
}

TEST(PennProximityGate, WithinGateSelectorRunsAndStaysInCandidateRange) {
  auto logger = rclcpp::get_logger("penn_gamma_gate_test");
  auto clock = std::make_shared<rclcpp::Clock>(RCL_STEADY_TIME);
  RateAutopilotCore::Params p = basePennParams();
  RateAutopilotCore core(p, logger, clock);
  ASSERT_TRUE(core.pennModelLoaded()) << "checkpoint failed to load from " << kCheckpointPath;

  const Eigen::Vector3d start(0.0, 0.0, 1.5);
  core.setOdometry(start, Eigen::Vector3d::Zero(), Eigen::Matrix3d::Identity(), clock->now());
  // Obstacle center 1.5m away -- within penn_proximity_gate_m (3.0).
  core.handleObstacles(oneObstacleAt(0.0, 1.5, 1.5, 1.0));

  const auto result = core.tick();
  // Not asserting an exact learned value (that's a model-output regression
  // test, not a gate-logic test) -- just that the selector actually ran
  // and returned something inside its own declared candidate range, i.e.
  // the gate did NOT short-circuit to the fixed fallback this time.
  EXPECT_GE(result.gamma_used, p.penn_gamma_min);
  EXPECT_LE(result.gamma_used, p.penn_gamma_max);
}
