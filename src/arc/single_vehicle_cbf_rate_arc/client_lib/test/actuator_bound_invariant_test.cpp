// Tier 0 unit test: no matter how aggressively nom_accel_max/nom_pos_kp/ nom_vel_kd are configured,
// the QP's OWN actuator bounds (T in [T_min,T_max], |omega|<=qp_omega_max) must never be violated …
#include <gtest/gtest.h>

#include "single_vehicle_cbf_rate/rate_autopilot_core.hpp"

using nodelib::RateAutopilotCore;

namespace {

RateAutopilotCore::Params extremeAuthorityParams() {
  RateAutopilotCore::Params p;
  p.vehicle_mass = 2.064;
  p.obs_d_min = 1.0;
  p.obs_safety_margin = 0.5;
  p.control_rate_hz = 100.0;
  p.cbf_gamma_obs = 8.0;
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

  // Deliberately far past the vehicle's real physical ceiling (~13.4 m/s^2) -- the point of this
  // test is that the QP's hard bounds must still hold even when the nominal law's own ceiling is …
  p.nom_accel_max = 100.0;
  p.nom_pos_kp = 50.0;
  p.nom_vel_kd = 50.0;
  return p;
}

}  // namespace

TEST(ActuatorBoundInvariant, ExtremeNominalAuthorityStaysWithinQpBounds) {
  auto logger = rclcpp::get_logger("actuator_bound_invariant_test");
  auto clock = std::make_shared<rclcpp::Clock>(RCL_STEADY_TIME);
  RateAutopilotCore::Params p = extremeAuthorityParams();
  RateAutopilotCore core(p, logger, clock);

  // Large displacement -- exactly the condition that produces the biggest nominal-law command, and
  // therefore the case most likely to expose a clamp/QP-formulation regression.
  const Eigen::Vector3d far_away(0.0, -20.0, 1.5);
  core.setOdometry(far_away, Eigen::Vector3d::Zero(), Eigen::Matrix3d::Identity(), clock->now());

  for (int i = 0; i < 5; ++i) {
    const auto result = core.tick();
    ASSERT_TRUE(result.feasible) << "tick " << i;
    EXPECT_GE(result.T, 0.0) << "tick " << i;
    EXPECT_LE(result.T, p.T_max + 1e-6) << "tick " << i << ": commanded thrust exceeded T_max";
    EXPECT_LE(result.omega.norm(), p.qp_omega_max + 1e-6)
        << "tick " << i << ": commanded body rate exceeded qp_omega_max";
  }
}
