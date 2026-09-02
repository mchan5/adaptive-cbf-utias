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

// Vehicle-agnostic CBF-QP + PENN/GAT orchestration: nominal law, stall-
// escape, obstacle entry-certification bookkeeping, PENN gamma selection,
// QP solve, and T (thrust) lifecycle integration.
//
// Extracted out of what used to be a single PX4-only RateAutopilotClient so
// the same control logic can drive a second, differently-actuated vehicle
// (Crazyflie) without duplicating it. This class works entirely in physical
// units (Newtons, rad/s) and has no PX4/px4_msgs dependency and never
// publishes or subscribes to anything itself -- nav_msgs/visualization_msgs
// are pulled in only for the handleOdometry/handleObstacles convenience
// parsers below, since both vehicle backends consume the exact same
// state_estimator/local_position/odom and /arc/obstacles contracts; that
// parsing would otherwise be duplicated verbatim in both vehicle-specific
// nodes; it is not PX4-specific.
//
// What deliberately did NOT move here, and stays in each vehicle-specific
// node instead: actuator/thrust-curve conversion (PX4's normalized-thrust
// curve vs. Crazyflie's raw setpoint units are not remotely the same
// concept), arming-state tracking, and all actual message
// publish/subscribe. T_max (Newtons) is the one actuator-curve-derived
// number this class needs, so it's a constructor parameter the caller
// computes from its own vehicle model, not something computed here.
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
    // Braking gain used ONLY when the QP is infeasible (see tick()'s
    // fallback). Added 2026-08-23: the previous fallback ("hold T, command
    // zero body rate", i.e. coast at current velocity/attitude) let a real
    // hard collision happen -- 9 consecutive infeasible ticks (90ms) at
    // gamma=8.0 during a close approach, with no correction, drove straight
    // into an obstacle before gamma recovered one tick too late
    // (dense_20260921 scene 25, range-widening plan). The QP has no
    // feasible barrier-respecting solution on an infeasible tick, so there
    // is nothing "correct" to fall back to -- killing velocity is strictly
    // safer than preserving it in the failure mode that actually happened,
    // and harmless when not near an obstacle. Applied as a_brake =
    // -nom_infeasible_brake_kv * vel through the SAME attitudeTrackingCommand
    // path used everywhere else, not a new untested code path.
    double nom_infeasible_brake_kv{4.0};
    // Accel ceiling for the braking fallback specifically -- deliberately
    // higher than nom_accel_max (3.0, sized for smooth cruise tracking, only
    // ~22% of the x500's physical ~13.4 m/s^2 per the airframe analysis).
    // 8.0 leaves real margin below the physical limit and below what the
    // qp_omega_max/thrust bounds can realize, while still letting an
    // emergency stop use meaningfully more authority than cruise ever does.
    double nom_infeasible_brake_accel_max{8.0};
    // Post-infeasibility recovery cooldown (added 2026-08-24, see
    // TESTING.md / config yaml's nom_accel_max history for the incident:
    // a real SITL hard collision traced to the infeasible-brake fallback
    // being completely memoryless -- the instant a tick reports feasible
    // again, the nominal law resumed FULL authority with no ramp, letting
    // the vehicle re-attempt an aggressive maneuver back into the same
    // tight geometry that just made the QP infeasible. For
    // infeasible_recovery_ticks after any infeasible tick, nominalCommand()
    // linearly ramps its accel_max ceiling back up from ~50% instead of
    // jumping straight to 100%. 30 ticks = 0.3s at 100Hz is a starting
    // value on the scale of a previously-documented "9 consecutive
    // infeasible ticks (90ms)" incident, not a first-principles constant --
    // validate/tune against real SITL data, don't trust by inspection alone
    // (see this project's repeated lesson on offline-vs-SITL fidelity).
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
    // Ablatable on purpose: a companion "raise gamma while escaping"
    // recovery added at the same time measured as contributing nothing
    // (17.40s vs 17.39s mean time-to-goal over the 8-scene closed-loop
    // diagnostic) and was removed rather than left in as inert
    // safety-critical code. This one earned its place in the same
    // ablation -- 18.80s -> 17.39s, with mean clearance improving
    // 0.738 -> 0.758 -- so it stays, but it stays measurable.
    bool escape_search_enabled{true};
    // Computed by the caller from its own actuator/thrust-curve model --
    // see computeThrustMax()-equivalent logic in each vehicle-specific node.
    double T_max{0.0};

    bool uncertified_v_cap_enabled{false};
    double uncertified_v_cap{1.0};

    // Obstacle geometry model for the collision barrier.
    // false (default) -> sphere: h = |p - c|^2 - d_min^2, full 3-D distance
    //   (original certified behavior; a finite obstacle can be overflown).
    // true -> vertical cylinder: barrier uses the HORIZONTAL distance only, so
    //   the obstacle footprint must be cleared at any altitude. Matches the real
    //   0.75 m floor cylinders. See PLAN_cylinder_barrier_20260831.md.
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
    double fallback_step{0.5};  // see PennGammaSelector::GammaSelectionParams::fallback_step
    // Feasibility-aware gate (2026-08-26, PLAN_adaptive_recovery §1.2):
    // additionally require a candidate gamma to leave the QP feasible
    // (probed with the current state/obstacles and a nominal-tracking
    // command) before it can be selected, instead of trusting the learned
    // min_h head alone. Motivated by gate_calibration_20260824/FINDINGS.md:
    // QP infeasibility enriches 4.3x in collision episodes vs the learned
    // proxy. Candidates are tried in the existing rank order (gated by
    // epistemic+cvar exactly as before); the first that is ALSO
    // QP-feasible wins. false = old behavior (learned gates only).
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
    // Raw UDE disturbance estimate this tick (N, world frame). Zero unless
    // ude_enabled and the estimator is active -- kept here so the
    // CbfDiagnostics side channel has the field wired even before the UDE
    // estimator is ported onto this branch.
    Eigen::Vector3d d_hat{Eigen::Vector3d::Zero()};
  };

  RateAutopilotCore(const Params& params, rclcpp::Logger logger, rclcpp::Clock::SharedPtr clock);

  // Pure state updates -- mirrors what odomCallback/obstaclesCallback used
  // to do directly on the old combined class.
  void setOdometry(const Eigen::Vector3d& pos_enu, const Eigen::Vector3d& vel_enu,
                    const Eigen::Matrix3d& R, const rclcpp::Time& stamp);
  void handleOdometry(const nav_msgs::msg::Odometry& msg, const rclcpp::Time& now);
  void handleObstacles(const visualization_msgs::msg::MarkerArray& msg);

  // Call from the vehicle-specific node's own arming-state transition
  // (disarmed->armed) -- resets T to hover thrust and clears the position
  // integrator, same as it always has.
  void resetT(const std::string& reason);

  bool odomValid() const { return odom_valid_; }
  double odomAgeSec(const rclcpp::Time& now) const;

  // Current ENU position as tracked from odometry. Exposed so a
  // vehicle-specific node can snap its goal waypoint to where the drone is
  // right now (e.g. auto-hover-in-place on arm). Only meaningful once
  // odomValid() is true.
  Eigen::Vector3d position() const { return pos_enu_; }

  // Advance the controller by one control-period tick. Defaults to the fixed
  // dt = 1/control_rate_hz (every offline/deterministic caller -- diag tools,
  // unit tests -- relies on this exact behavior and passes nothing). A live
  // client driven by a wall-clock timer against a sim-time clock that isn't
  // guaranteed to match it (SITL: PX4/Gazebo lockstep runs on sim-time, but
  // nothing downstream is on that clock) should instead pass the real
  // elapsed time so the integrators below don't silently mis-integrate.
  TickResult tick(std::optional<double> measured_dt_sec = std::nullopt);

  void setWaypoint(const Eigen::Vector3d& wp) { params_.waypoint = wp; }

  // Diagnostic-only: mutates cbf_gamma_obs directly without going through
  // updateTunableParams (which deliberately excludes it -- see that
  // function's comment). Exists for offline harnesses (e.g.
  // body_rate_closed_loop_diag's GREEDY_SWITCH mode) that want to change
  // the fixed gamma on a LIVE core between ticks without reconstructing it
  // and losing internal state (T_ hover-thrust reset, stall/escape timers).
  // Not used by the deployed node.
  void setFixedGamma(double gamma) { params_.cbf_gamma_obs = gamma; }
  double T() const { return T_; }
  const Params& params() const { return params_; }

  // Updates exactly the subset of Params that the original (pre-refactor)
  // combined node re-read from ROS on every call (waypoint, nominal-law
  // gains, stall/escape params, PENN selection params, uncertified_v_cap) --
  // i.e. everything that was genuinely live-tunable via `ros2 param set`
  // while running. Deliberately does NOT touch vehicle_mass, QP bounds,
  // T_max, or the PENN model path: those were only ever applied at
  // construction or on a gamma-triggered rebuildQp() in the original too,
  // so treating them as live here would silently diverge from that
  // behavior (Params would say one thing, the already-built QP object
  // another). Call this once per tick from the vehicle-specific node if you
  // want that same live-tuning surface preserved; skip it if you don't.
  void updateTunableParams(const Params& p);

  // True iff loadPennModel() actually populated penn_model_ -- distinct
  // from params_.penn_enabled, which only says a model WAS REQUESTED.
  // loadPennModel() warns-and-continues on a missing/bad model path rather
  // than throwing, which is the right behavior for a flying vehicle (never
  // hard-fail a deployed node over a missing ML model, degrade to fixed
  // gamma instead) -- but it means an "adaptive" arm can silently run pure
  // fixed-gamma with no error at all. Added 2026-08-23 after exactly that
  // happened in body_rate_closed_loop_diag.cpp (a relative weights path
  // didn't resolve; the resulting campaign ran fixed gamma under the
  // "adaptive" label and was only caught because deadlock counts didn't
  // match a frozen reference). Diagnostic/campaign code that claims to
  // measure the adaptive policy should assert this true after construction
  // for any arm with penn_enabled set; the deployed node should keep
  // warning and flying, not assert.
  bool pennModelLoaded() const { return penn_model_.has_value(); }

 private:
  void loadPennModel();
  void rebuildQp();
  void maybeCertify(int obstacle_id, const Eigen::Vector3d& center, double d_min);
  double pennSelectAlpha(const Eigen::Vector3d& pos_enu);
  // Probes QP feasibility for a candidate gamma at the current state, using
  // the stateless nominal-tracking command (positionTrackingAccel() +
  // attitudeTrackingCommand()) as the desired command -- same approximation
  // the escape-direction search already uses for its own candidate probes.
  // Does not mutate qp_/gamma_obs_selected_.
  bool probeGammaFeasible(double gamma, const std::vector<Eigen::Vector3d>& certified_positions,
                           const std::vector<Eigen::Vector3d>& uncertified_positions) const;
  Eigen::Vector3d updateStallEscape(const Eigen::Vector3d& e_p);
  std::pair<double, Eigen::Vector3d> nominalCommand(double dt);
  Eigen::Vector3d positionTrackingAccel() const;
  // accel_max_override, added 2026-08-23: defaults to params_.nom_accel_max
  // (every existing call site, unchanged). The infeasibility braking
  // fallback passes a higher value -- nom_accel_max is sized for smooth
  // cruise tracking (3.0 m/s^2, ~22% of the x500's physical ~13.4 m/s^2
  // capability per the range-widening plan's airframe analysis), which is
  // the wrong ceiling for an emergency stop; braking should be allowed to
  // use much more of the vehicle's real authority than cruise does.
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
  // Escape-direction search commitment (2026-08-22): re-running the
  // multi-direction search every tick let the winning direction flip as
  // position/geometry shifted slightly tick to tick, producing an
  // oscillating limit cycle instead of an escape (observed directly on the
  // closed-loop diagnostic). Search once when a new attempt starts, then
  // hold that choice for the attempt's full duration, same discipline the
  // original single-direction escape already used. -1 = the baseline
  // (single fixed-direction) escape won; >=0 indexes escapeSearchDirections().
  int last_searched_attempt_{0};
  int committed_escape_index_{-1};

  Eigen::Vector3d pos_enu_{Eigen::Vector3d::Zero()};
  Eigen::Vector3d vel_enu_{Eigen::Vector3d::Zero()};
  Eigen::Matrix3d R_{Eigen::Matrix3d::Identity()};
  bool odom_valid_{false};
  rclcpp::Time last_odom_time_;
  uint64_t tick_count_{0};
  // See Params::infeasible_recovery_ticks -- counts down once per
  // nominalCommand() call (exactly one per tick()); >0 means the nominal
  // law's accel_max ceiling is still ramping back up from a recent
  // infeasible tick.
  int recovery_ticks_remaining_{0};

  struct ObstacleEntry {
    Eigen::Vector3d center;
    double radius;
  };
  std::map<int, ObstacleEntry> obstacles_seen_;
  ObstacleRegistry registry_;
  // uncertified_warned_ids_ is a PERMANENT, insert-only set -- it exists
  // solely to log the "FAILED entry certification" warning once per
  // obstacle id, never to drive QP routing (see maybeCertify()'s comment).
  std::set<int> uncertified_warned_ids_;
  // currently_certified_ids_ (added 2026-08-24, fixing a real bug: tick()
  // used to route the certified/uncertified QP split off
  // uncertified_warned_ids_ above, which meant an obstacle that failed
  // certification once stayed on the soft/slack QP row PERMANENTLY for the
  // rest of the episode, even though ObstacleRegistry::registerAndCheck()
  // already correctly re-evaluates a failed obstacle fresh on every call --
  // that live retry result was being computed and silently discarded. This
  // set tracks the LIVE certification state instead, updated every
  // maybeCertify() call; tick() routes off this one.
  std::set<int> currently_certified_ids_;
  bool size_mismatch_warned_{false};

  std::optional<PennGammaSelector> penn_model_;
  double gamma_obs_selected_;
  int penn_tick_counter_{0};

  std::optional<BodyRateCBFQP> qp_;
};

}  // namespace nodelib
