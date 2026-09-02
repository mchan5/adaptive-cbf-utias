#pragma once

#include <map>
#include <optional>
#include <set>
#include <string>

#include <Eigen/Dense>
#include <rclcpp/clock.hpp>
#include <rclcpp/logger.hpp>
#include <rclcpp/time.hpp>

#include "nav_msgs/msg/odometry.hpp"
#include "visualization_msgs/msg/marker_array.hpp"

#include "single_vehicle_cbf_rate/body_rate_qp_osqp.hpp"
#include "single_vehicle_cbf_rate/entry_certification_gate.hpp"
#include "single_vehicle_cbf_rate/penn_gamma_selector.hpp"
#include "single_vehicle_cbf_rate/quadrotor_dynamics.hpp"

namespace nodelib {

// Vehicle-agnostic CBF-QP + PENN/GAT orchestration: nominal law, stall-escape, obstacle entry-
// certification bookkeeping, PENN gamma selection, QP solve, and T (thrust) lifecycle integration.
class RateAutopilotCore {
 public:
  struct Params {
    double vehicle_mass{2.064};
    double obs_d_min{1.0};
    double obs_safety_margin{0.5};
    double control_rate_hz{100.0};
    double cbf_gamma_obs{4.0};

    double nom_pos_kp{1.5};
    double nom_vel_kd{2.0};
    double nom_pos_ki{0.3};
    double nom_pos_i_max{2.0};
    double nom_accel_max{3.0};
    double nom_yaw_des{0.0};
    double nom_k_r{5.0};
    double nom_k_t{8.0};
    // Braking gain used ONLY when the QP is infeasible (see tick()'s fallback).
    double nom_infeasible_brake_kv{4.0};
    // Accel ceiling for the braking fallback specifically -- deliberately higher than nom_accel_max
    // (3.0, sized for smooth cruise tracking, only ~22% of the x500's physical ~13.4 m/s^2 per …
    double nom_infeasible_brake_accel_max{8.0};
    // Post-infeasibility recovery cooldown (added 2026-08-24, see TESTING.md / config yaml's
    // nom_accel_max history for the incident: a real SITL hard collision traced to the …
    int infeasible_recovery_ticks{30};

    double stall_speed_threshold{0.08};
    double stall_duration_sec{2.5};
    double escape_duration_sec{2.5};
    double escape_lateral_accel{2.5};
    double escape_goal_pull_scale{0.3};
    double goal_arrival_tolerance{0.3};

    double qp_thrust_min{4.7};
    double qp_thrust_floor_gain{8.0};
    double qp_tau_min{-50.0};
    double qp_tau_max{50.0};
    double qp_omega_max{6.0};
    double qp_rho{1.0};
    double qp_slack_weight{1e4};
    // Multi-direction escape search (see escapeSearchDirections()).
    bool escape_search_enabled{true};
    // Computed by the caller from its own actuator/thrust-curve model --
    // see computeThrustMax()-equivalent logic in each vehicle-specific node.
    double T_max{0.0};

    bool uncertified_v_cap_enabled{false};
    double uncertified_v_cap{1.0};

    // Obstacle geometry model for the collision barrier. false (default) -> sphere: h = |p - c|^2 -
    // d_min^2, full 3-D distance (original certified behavior; a finite obstacle can be overflown).
    bool cbf_cylinder_barrier{false};

    bool penn_enabled{true};
    std::string penn_model_path;  // absolute path; empty/missing -> falls back to cbf_gamma_obs
    double penn_gamma_min{0.5};
    double penn_gamma_max{10.0};
    int penn_gamma_steps{20};
    int penn_update_ticks{20};
    double penn_proximity_gate_m{3.0};
    double epistemic_threshold{0.046695};
    double cvar_boundary{0.1570};
    double deadlock_threshold{5.0};
    double lcb_k{0.0};  // see PennGammaSelector::GammaSelectionParams::lcb_k
    double continuity_weight{0.0};  // see PennGammaSelector::GammaSelectionParams::continuity_weight
    double fallback_step{0.5};
    // see PennGammaSelector::GammaSelectionParams::fallback_step Feasibility-aware gate
    // (2026-08-26, PLAN_adaptive_recovery §1.2): additionally require a candidate gamma to leave …
    bool penn_feasibility_gate{false};

    Eigen::Vector3d waypoint{Eigen::Vector3d::Zero()};
  };

  struct TickResult {
    bool feasible{false};
    double T{0.0};
    Eigen::Vector3d omega{Eigen::Vector3d::Zero()};
    double delta{0.0};
    double gamma_used{0.0};
    bool used_penn{false};
    size_t certified_count{0};
    size_t uncertified_count{0};
    // Raw UDE disturbance estimate this tick (N, world frame).
    Eigen::Vector3d d_hat{Eigen::Vector3d::Zero()};
  };

  RateAutopilotCore(const Params& params, rclcpp::Logger logger, rclcpp::Clock::SharedPtr clock);

  // Pure state updates -- mirrors what odomCallback/obstaclesCallback used
  // to do directly on the old combined class.
  void setOdometry(const Eigen::Vector3d& pos_enu, const Eigen::Vector3d& vel_enu,
                    const Eigen::Matrix3d& R, const rclcpp::Time& stamp);
  void handleOdometry(const nav_msgs::msg::Odometry& msg, const rclcpp::Time& now);
  void handleObstacles(const visualization_msgs::msg::MarkerArray& msg);

  // Call from the vehicle-specific node's own arming-state transition (disarmed->armed) -- resets T
  // to hover thrust and clears the position integrator, same as it always has.
  void resetT(const std::string& reason);

  bool odomValid() const { return odom_valid_; }
  double odomAgeSec(const rclcpp::Time& now) const;

  Eigen::Vector3d position() const { return pos_enu_; }

  // Advance the controller by one control-period tick.
  TickResult tick(std::optional<double> measured_dt_sec = std::nullopt);

  void setWaypoint(const Eigen::Vector3d& wp) { params_.waypoint = wp; }

  // Diagnostic-only: mutates cbf_gamma_obs directly without going through updateTunableParams
  // (which deliberately excludes it -- see that function's comment).
  void setFixedGamma(double gamma) { params_.cbf_gamma_obs = gamma; }
  double T() const { return T_; }
  const Params& params() const { return params_; }

  // Updates exactly the subset of Params that the original (pre-refactor) combined node re-read
  // from ROS on every call (waypoint, nominal-law gains, stall/escape params, PENN selection …
  void updateTunableParams(const Params& p);

  // True iff loadPennModel() actually populated penn_model_ -- distinct from params_.penn_enabled,
  // which only says a model WAS REQUESTED.
  bool pennModelLoaded() const { return penn_model_.has_value(); }

 private:
  void loadPennModel();
  void rebuildQp();
  void maybeCertify(int obstacle_id, const Eigen::Vector3d& center, double d_min);
  double pennSelectAlpha(const Eigen::Vector3d& pos_enu);
  // Probes QP feasibility for a candidate gamma at the current state, using the stateless nominal-
  // tracking command (positionTrackingAccel() + attitudeTrackingCommand()) as the desired command …
  bool probeGammaFeasible(double gamma, const std::vector<Eigen::Vector3d>& certified_positions,
                           const std::vector<Eigen::Vector3d>& uncertified_positions) const;
  Eigen::Vector3d updateStallEscape(const Eigen::Vector3d& e_p);
  std::pair<double, Eigen::Vector3d> nominalCommand(double dt);
  Eigen::Vector3d positionTrackingAccel() const;
  // accel_max_override, added 2026-08-23: defaults to params_.nom_accel_max (every existing call
  // site, unchanged).
  std::pair<double, Eigen::Vector3d> attitudeTrackingCommand(
      const Eigen::Vector3d& a_des, double accel_max_override = -1.0) const;
  std::vector<Eigen::Vector3d> escapeSearchDirections() const;
  void integrateAndClampT(double tau, double dt);

  Params params_;
  rclcpp::Logger logger_;
  rclcpp::Clock::SharedPtr clock_;
  double dt_;

  double T_min_{4.7};
  double T_{0.0};
  Eigen::Vector3d pos_int_{Eigen::Vector3d::Zero()};

  bool stall_clock_running_{false};
  rclcpp::Time stall_clock_start_;
  bool escape_active_{false};
  rclcpp::Time escape_until_;
  int escape_attempt_count_{0};
  // Escape-direction search commitment (2026-08-22): re-running the multi-direction search every
  // tick let the winning direction flip as position/geometry shifted slightly tick to tick, …
  int last_searched_attempt_{0};
  int committed_escape_index_{-1};

  Eigen::Vector3d pos_enu_{Eigen::Vector3d::Zero()};
  Eigen::Vector3d vel_enu_{Eigen::Vector3d::Zero()};
  Eigen::Matrix3d R_{Eigen::Matrix3d::Identity()};
  bool odom_valid_{false};
  rclcpp::Time last_odom_time_;
  uint64_t tick_count_{0};
  // See Params::infeasible_recovery_ticks -- counts down once per nominalCommand() call (exactly
  // one per tick()); >0 means the nominal law's accel_max ceiling is still ramping back up from a …
  int recovery_ticks_remaining_{0};

  struct ObstacleEntry {
    Eigen::Vector3d center;
    double radius;
  };
  std::map<int, ObstacleEntry> obstacles_seen_;
  ObstacleRegistry registry_;
  // uncertified_warned_ids_ is a PERMANENT, insert-only set -- it exists solely to log the "FAILED
  // entry certification" warning once per obstacle id, never to drive QP routing (see …
  std::set<int> uncertified_warned_ids_;
  // currently_certified_ids_ (added 2026-08-24, fixing a real bug: tick() used to route the
  // certified/uncertified QP split off uncertified_warned_ids_ above, which meant an obstacle …
  std::set<int> currently_certified_ids_;
  bool size_mismatch_warned_{false};

  std::optional<PennGammaSelector> penn_model_;
  double gamma_obs_selected_;
  int penn_tick_counter_{0};

  std::optional<BodyRateCBFQP> qp_;
};

}  // namespace nodelib
