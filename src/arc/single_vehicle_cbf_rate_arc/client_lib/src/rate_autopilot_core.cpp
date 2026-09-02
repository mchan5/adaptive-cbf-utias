#include "single_vehicle_cbf_rate/rate_autopilot_core.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <limits>
#include <string>

#include "rclcpp/logging.hpp"

namespace nodelib {

RateAutopilotCore::RateAutopilotCore(const Params& params, rclcpp::Logger logger,
                                      rclcpp::Clock::SharedPtr clock)
    : params_(params),
      logger_(logger),
      clock_(std::move(clock)),
      dt_(1.0 / params.control_rate_hz),
      T_min_(params.qp_thrust_min),
      stall_clock_start_(clock_->now()),
      escape_until_(clock_->now()),
      last_odom_time_(clock_->now()),
      registry_(params.cbf_gamma_obs, params.cbf_gamma_obs, params.vehicle_mass,
                params.cbf_cylinder_barrier),
      gamma_obs_selected_(params.cbf_gamma_obs) {
  resetT("construction");
  loadPennModel();
  rebuildQp();
}

void RateAutopilotCore::resetT(const std::string& reason) {
  T_ = params_.vehicle_mass * kQuadG;
  pos_int_.setZero();
  RCLCPP_WARN(logger_, "T reset to hover thrust %.4fN (reason: %s)", T_, reason.c_str());
}

void RateAutopilotCore::loadPennModel() {
  if (params_.penn_model_path.empty()) {
    RCLCPP_WARN(logger_, "No penn_model_path configured -- using fixed cbf_gamma_obs");
    return;
  }
  if (!std::filesystem::exists(params_.penn_model_path)) {
    RCLCPP_WARN(logger_, "PENN model not found at %s -- using fixed cbf_gamma_obs",
                params_.penn_model_path.c_str());
    return;
  }
  try {
    penn_model_.emplace(params_.penn_model_path.c_str());
    RCLCPP_INFO(logger_, "PENN+GAT model loaded from %s", params_.penn_model_path.c_str());
  } catch (const std::exception& e) {
    RCLCPP_ERROR(logger_, "PENN model load failed: %s -- using fixed cbf_gamma_obs", e.what());
  }
}

void RateAutopilotCore::rebuildQp() {
  const double gamma = gamma_obs_selected_;
  // GAMMA_RATIO_1/2/3 (diagnostic, 2026-08-23): multipliers applied to the single selected gamma to
  // form the three HOCBF gains separately, i.e. g_i = ratio_i * gamma.
  double r1 = 1.0, r2 = 1.0, r3 = 1.0;
  if (const char* e = std::getenv("GAMMA_RATIO_1")) r1 = std::atof(e);
  if (const char* e = std::getenv("GAMMA_RATIO_2")) r2 = std::atof(e);
  if (const char* e = std::getenv("GAMMA_RATIO_3")) r3 = std::atof(e);
  qp_.emplace(params_.vehicle_mass, r1 * gamma, r2 * gamma, r3 * gamma, T_min_,
              params_.qp_thrust_floor_gain,
              params_.qp_tau_min, params_.qp_tau_max, params_.qp_omega_max, params_.qp_rho,
              params_.qp_slack_weight, params_.cbf_cylinder_barrier);
}

void RateAutopilotCore::setOdometry(const Eigen::Vector3d& pos_enu, const Eigen::Vector3d& vel_enu,
                                     const Eigen::Matrix3d& R, const rclcpp::Time& stamp) {
  last_odom_time_ = stamp;
  pos_enu_ = pos_enu;
  vel_enu_ = vel_enu;
  R_ = R;
  odom_valid_ = true;
}

void RateAutopilotCore::handleOdometry(const nav_msgs::msg::Odometry& msg, const rclcpp::Time& now) {
  const Eigen::Vector3d pos(msg.pose.pose.position.x, msg.pose.pose.position.y,
                             msg.pose.pose.position.z);
  const auto& q = msg.pose.pose.orientation;
  const Eigen::Matrix3d R = Eigen::Quaterniond(q.w, q.x, q.y, q.z).toRotationMatrix();

  const Eigen::Vector3d v_raw(msg.twist.twist.linear.x, msg.twist.twist.linear.y,
                               msg.twist.twist.linear.z);
  Eigen::Vector3d vel;
  if (msg.child_frame_id == "base_link") {
    vel = R * v_raw;
  } else {
    vel = v_raw;
  }
  setOdometry(pos, vel, R, now);
}

void RateAutopilotCore::handleObstacles(const visualization_msgs::msg::MarkerArray& msg) {
  const double margin = params_.obs_safety_margin;
  const double d_min = params_.obs_d_min;

  std::set<int> seen_ids;
  for (const auto& marker : msg.markers) {
    const Eigen::Vector3d center(marker.pose.position.x, marker.pose.position.y,
                                  marker.pose.position.z);
    const double radius = marker.scale.x / 2.0 + margin;
    obstacles_seen_[marker.id] = ObstacleEntry{center, radius};
    seen_ids.insert(marker.id);

    if (radius > d_min && !size_mismatch_warned_) {
      RCLCPP_WARN(logger_,
                  "Obstacle id=%d reports radius+margin=%.2fm, larger than the shared "
                  "obs_d_min=%.2fm used by the QP for every obstacle -- this obstacle may be "
                  "under-cleared.",
                  marker.id, radius, d_min);
      size_mismatch_warned_ = true;
    }

    maybeCertify(marker.id, center, d_min);
  }

  for (auto it = obstacles_seen_.begin(); it != obstacles_seen_.end();) {
    if (seen_ids.count(it->first) == 0) {
      it = obstacles_seen_.erase(it);
    } else {
      ++it;
    }
  }
}

void RateAutopilotCore::maybeCertify(int obstacle_id, const Eigen::Vector3d& center, double d_min) {
  if (!odom_valid_) {
    return;  // can't certify without a state to certify against yet
  }
  const QuadState state{pos_enu_, vel_enu_, R_, T_};
  const double gamma = gamma_obs_selected_;
  // Same GAMMA_RATIO_1/2 multipliers rebuildQp() applies -- the barrier chain's g1/g2 must match
  // the QP's or the certification gate would be checking a different barrier than the one being …
  double r1 = 1.0, r2 = 1.0;
  if (const char* e = std::getenv("GAMMA_RATIO_1")) r1 = std::atof(e);
  if (const char* e = std::getenv("GAMMA_RATIO_2")) r2 = std::atof(e);
  registry_.setGammas(r1 * gamma, r2 * gamma);
  const auto reg_result = registry_.registerAndCheck(obstacle_id, state, center, d_min);
  // Live certification state for tick()'s QP routing -- see this set's declaration comment. Updated
  // every call, independent of the permanent uncertified_warned_ids_ below.
  if (reg_result.certified) {
    currently_certified_ids_.insert(obstacle_id);
  } else {
    currently_certified_ids_.erase(obstacle_id);
  }
  if (!reg_result.certified && uncertified_warned_ids_.count(obstacle_id) == 0) {
    uncertified_warned_ids_.insert(obstacle_id);
    const auto& chain = *reg_result.chain;
    RCLCPP_WARN(logger_,
                "Obstacle id=%d FAILED entry certification at first sighting (psi0=%.3f psi1=%.3f "
                "psi2=%.3f, gamma=%.3f) -- routed through the QP's slack-relaxed row (D14), not "
                "excluded.",
                obstacle_id, chain.psi0, chain.psi1, chain.psi2, gamma);
  }
}

double RateAutopilotCore::odomAgeSec(const rclcpp::Time& now) const {
  return (now - last_odom_time_).seconds();
}

void RateAutopilotCore::updateTunableParams(const Params& p) {
  params_.waypoint = p.waypoint;
  params_.nom_pos_kp = p.nom_pos_kp;
  params_.nom_vel_kd = p.nom_vel_kd;
  params_.nom_pos_ki = p.nom_pos_ki;
  params_.nom_pos_i_max = p.nom_pos_i_max;
  params_.nom_accel_max = p.nom_accel_max;
  params_.nom_yaw_des = p.nom_yaw_des;
  params_.nom_k_r = p.nom_k_r;
  params_.nom_k_t = p.nom_k_t;
  params_.nom_infeasible_brake_kv = p.nom_infeasible_brake_kv;
  params_.nom_infeasible_brake_accel_max = p.nom_infeasible_brake_accel_max;
  params_.stall_speed_threshold = p.stall_speed_threshold;
  params_.stall_duration_sec = p.stall_duration_sec;
  params_.escape_duration_sec = p.escape_duration_sec;
  params_.escape_lateral_accel = p.escape_lateral_accel;
  params_.escape_goal_pull_scale = p.escape_goal_pull_scale;
  params_.goal_arrival_tolerance = p.goal_arrival_tolerance;
  params_.penn_enabled = p.penn_enabled;
  params_.penn_gamma_min = p.penn_gamma_min;
  params_.penn_gamma_max = p.penn_gamma_max;
  params_.penn_gamma_steps = p.penn_gamma_steps;
  params_.penn_proximity_gate_m = p.penn_proximity_gate_m;
  params_.epistemic_threshold = p.epistemic_threshold;
  params_.cvar_boundary = p.cvar_boundary;
  params_.deadlock_threshold = p.deadlock_threshold;
  params_.lcb_k = p.lcb_k;
  params_.continuity_weight = p.continuity_weight;
  params_.uncertified_v_cap_enabled = p.uncertified_v_cap_enabled;
  params_.uncertified_v_cap = p.uncertified_v_cap;
  params_.penn_feasibility_gate = p.penn_feasibility_gate;
}

double RateAutopilotCore::pennSelectAlpha(const Eigen::Vector3d& pos_enu) {
  if (!params_.penn_enabled) {
    return params_.cbf_gamma_obs;
  }
  if (!penn_model_.has_value() || obstacles_seen_.empty()) {
    return params_.cbf_gamma_obs;
  }

  double min_obstacle_dist = std::numeric_limits<double>::infinity();
  for (const auto& [id, entry] : obstacles_seen_) {
    min_obstacle_dist = std::min(min_obstacle_dist, (pos_enu - entry.center).norm());
  }
  if (min_obstacle_dist > params_.penn_proximity_gate_m) {
    return params_.cbf_gamma_obs;
  }

  PennGammaSelector::GammaSelectionParams gp;
  gp.goal = params_.waypoint;
  gp.gamma_min = params_.penn_gamma_min;
  gp.gamma_max_candidate = params_.penn_gamma_max;
  gp.gamma_steps = params_.penn_gamma_steps;
  gp.epistemic_threshold = params_.epistemic_threshold;
  gp.cvar_boundary = params_.cvar_boundary;
  gp.deadlock_threshold = params_.deadlock_threshold;
  gp.lcb_k = params_.lcb_k;
  gp.prev_gamma = gamma_obs_selected_;
  gp.continuity_weight = params_.continuity_weight;
  gp.fallback_step = params_.fallback_step;

  std::vector<PennGammaSelector::ObstacleInput> obstacles;
  obstacles.reserve(obstacles_seen_.size());
  for (const auto& [id, entry] : obstacles_seen_) {
    obstacles.push_back({entry.center, entry.radius});
  }

  if (!params_.penn_feasibility_gate) {
    const auto gamma = penn_model_->selectGamma(pos_enu, vel_enu_, obstacles, gp);
    if (!gamma.has_value()) {
      RCLCPP_WARN_THROTTLE(logger_, *clock_, 5000, "PENN inference failed -- using fixed cbf_gamma_obs");
      return params_.cbf_gamma_obs;
    }
    return *gamma;
  }

  // Feasibility-aware gate (PLAN_adaptive_recovery §1.2): take the full candidate table instead of
  // selectGamma()'s single pick, walk it in the same rank order selectGamma would (learned-gate- …
  const char* rule_env = std::getenv("SELECTION_RULE");
  const bool min_deficit_rule = (rule_env == nullptr || std::string(rule_env) == "min_deficit");
  if (!min_deficit_rule) {
    const auto gamma = penn_model_->selectGamma(pos_enu, vel_enu_, obstacles, gp);
    return gamma.value_or(params_.cbf_gamma_obs);
  }

  const auto table = penn_model_->diagnoseGammaSelection(pos_enu, vel_enu_, obstacles, gp);
  if (!table.has_value() || table->empty()) {
    RCLCPP_WARN_THROTTLE(logger_, *clock_, 5000, "PENN inference failed -- using fixed cbf_gamma_obs");
    return params_.cbf_gamma_obs;
  }

  std::vector<Eigen::Vector3d> certified_positions;
  std::vector<Eigen::Vector3d> uncertified_positions;
  for (const auto& [id, entry] : obstacles_seen_) {
    if (currently_certified_ids_.count(id) != 0) {
      certified_positions.push_back(entry.center);
    } else {
      uncertified_positions.push_back(entry.center);
    }
  }

  std::vector<PennGammaSelector::GammaCandidateDiagnostic> gated;
  for (const auto& row : *table) {
    if (row.passed_epistemic && row.passed_deadlock_risk) gated.push_back(row);
  }
  std::sort(gated.begin(), gated.end(),
            [](const auto& a, const auto& b) { return a.mean_deadlock < b.mean_deadlock; });

  for (const auto& row : gated) {
    if (probeGammaFeasible(row.gamma, certified_positions, uncertified_positions)) {
      return row.gamma;
    }
  }

  // No learned-gate-passed candidate is QP-feasible either: same
  // conservative ratchet §1.1 uses, not the old circular fallback.
  return std::max(params_.penn_gamma_min, gamma_obs_selected_ - params_.fallback_step);
}

Eigen::Vector3d RateAutopilotCore::updateStallEscape(const Eigen::Vector3d& e_p) {
  const rclcpp::Time now = clock_->now();
  const double dist_to_goal = e_p.norm();
  const double arrival_tol = params_.goal_arrival_tolerance;

  if (dist_to_goal < arrival_tol) {
    stall_clock_running_ = false;
    escape_active_ = false;
    return Eigen::Vector3d::Zero();
  }

  if (escape_active_) {
    if (now >= escape_until_) {
      escape_active_ = false;
      stall_clock_running_ = false;
    }
  } else {
    const double speed = vel_enu_.norm();
    if (speed < params_.stall_speed_threshold) {
      if (!stall_clock_running_) {
        stall_clock_start_ = now;
        stall_clock_running_ = true;
      } else if ((now - stall_clock_start_).seconds() >= params_.stall_duration_sec) {
        escape_active_ = true;
        escape_until_ = now + rclcpp::Duration::from_seconds(params_.escape_duration_sec);
        ++escape_attempt_count_;
        stall_clock_running_ = false;
        RCLCPP_WARN(logger_,
                    "Stall detected (speed=%.3fm/s < %.3fm/s for %.1fs, dist_to_goal=%.2fm) -- "
                    "escape attempt #%d",
                    speed, params_.stall_speed_threshold, params_.stall_duration_sec, dist_to_goal,
                    escape_attempt_count_);
      }
    } else {
      stall_clock_running_ = false;
    }
  }

  if (!escape_active_) {
    return Eigen::Vector3d::Zero();
  }

  Eigen::Vector2d goal_dir_xy = -e_p.head<2>();
  if (goal_dir_xy.norm() < 1e-6) {
    goal_dir_xy = Eigen::Vector2d::UnitY();
  }
  goal_dir_xy.normalize();
  const double side = (escape_attempt_count_ % 2 == 1) ? 1.0 : -1.0;
  const Eigen::Vector2d perp = side * Eigen::Vector2d(-goal_dir_xy.y(), goal_dir_xy.x());

  return params_.escape_lateral_accel * Eigen::Vector3d(perp.x(), perp.y(), 0.0);
}

bool RateAutopilotCore::probeGammaFeasible(double gamma,
                                            const std::vector<Eigen::Vector3d>& certified_positions,
                                            const std::vector<Eigen::Vector3d>& uncertified_positions) const {
  double r1 = 1.0, r2 = 1.0, r3 = 1.0;
  if (const char* e = std::getenv("GAMMA_RATIO_1")) r1 = std::atof(e);
  if (const char* e = std::getenv("GAMMA_RATIO_2")) r2 = std::atof(e);
  if (const char* e = std::getenv("GAMMA_RATIO_3")) r3 = std::atof(e);
  const BodyRateCBFQP probe_qp(params_.vehicle_mass, r1 * gamma, r2 * gamma, r3 * gamma, T_min_,
                                params_.qp_thrust_floor_gain, params_.qp_tau_min, params_.qp_tau_max,
                                params_.qp_omega_max, params_.qp_rho, params_.qp_slack_weight,
                                params_.cbf_cylinder_barrier);
  const QuadState state{pos_enu_, vel_enu_, R_, T_};
  const auto [tau_nom, omega_nom] = attitudeTrackingCommand(positionTrackingAccel());
  std::optional<double> v_cap;
  if (params_.uncertified_v_cap_enabled) {
    v_cap = params_.uncertified_v_cap;
  }
  const auto result = probe_qp.solve(state, certified_positions, uncertified_positions,
                                      params_.obs_d_min, tau_nom, omega_nom, v_cap);
  return result.feasible;
}

Eigen::Vector3d RateAutopilotCore::positionTrackingAccel() const {
  const Eigen::Vector3d& wp = params_.waypoint;
  const Eigen::Vector3d e_p = pos_enu_ - wp;
  const Eigen::Vector3d e_v = vel_enu_;  // v_des = 0
  return -params_.nom_pos_kp * e_p - params_.nom_vel_kd * e_v - params_.nom_pos_ki * pos_int_;
}

std::pair<double, Eigen::Vector3d> RateAutopilotCore::attitudeTrackingCommand(
    const Eigen::Vector3d& a_des_in, double accel_max_override) const {
  const double accel_max = accel_max_override > 0.0 ? accel_max_override : params_.nom_accel_max;
  Eigen::Vector3d a_des = a_des_in;
  const double a_norm = a_des.norm();
  if (a_norm > accel_max) {
    a_des *= (accel_max / a_norm);
  }

  const Eigen::Vector3d F_des = params_.vehicle_mass * (a_des + kQuadG * Eigen::Vector3d::UnitZ());
  const double T_des = F_des.norm();
  const Eigen::Vector3d b3_des = T_des > 1e-6 ? Eigen::Vector3d(F_des / T_des) : R_.col(2);

  const double psi_des = params_.nom_yaw_des;
  Eigen::Vector3d c1(std::cos(psi_des), std::sin(psi_des), 0.0);
  Eigen::Vector3d b2_des = b3_des.cross(c1);
  double b2_norm = b2_des.norm();
  if (b2_norm < 1e-6) {
    c1 = Eigen::Vector3d(0.0, 1.0, 0.0);
    b2_des = b3_des.cross(c1);
    b2_norm = b2_des.norm();
  }
  b2_des /= b2_norm;
  const Eigen::Vector3d b1_des = b2_des.cross(b3_des);

  Eigen::Matrix3d R_des;
  R_des.col(0) = b1_des;
  R_des.col(1) = b2_des;
  R_des.col(2) = b3_des;

  const Eigen::Matrix3d e_R_hat = 0.5 * (R_des.transpose() * R_ - R_.transpose() * R_des);
  const Eigen::Vector3d e_R(e_R_hat(2, 1), e_R_hat(0, 2), e_R_hat(1, 0));

  Eigen::Vector3d omega_nom = -params_.nom_k_r * e_R;
  omega_nom.z() = 0.0;

  const double T_des_clamped = std::clamp(T_des, T_min_, params_.T_max);
  const double tau_nom = params_.nom_k_t * (T_des_clamped - T_);

  return {tau_nom, omega_nom};
}

std::pair<double, Eigen::Vector3d> RateAutopilotCore::nominalCommand(double dt) {
  const Eigen::Vector3d& wp = params_.waypoint;
  const Eigen::Vector3d e_p = pos_enu_ - wp;

  const double k_i = params_.nom_pos_ki;
  const double i_bound = params_.nom_pos_i_max / std::max(k_i, 1e-9);
  pos_int_ = (pos_int_ + e_p * dt).cwiseMax(-i_bound).cwiseMin(i_bound);

  Eigen::Vector3d a_des = positionTrackingAccel();

  const Eigen::Vector3d escape_bias = updateStallEscape(e_p);
  if (escape_active_) {
    a_des = params_.escape_goal_pull_scale * a_des + escape_bias;
  }

  // Post-infeasibility recovery ramp (see Params::infeasible_recovery_ticks' comment) -- skipped
  // during an active escape maneuver, which already has its own dedicated, separately-tested …
  double accel_max_override = -1.0;
  if (recovery_ticks_remaining_ > 0 && !escape_active_) {
    const double frac = 1.0 - static_cast<double>(recovery_ticks_remaining_) /
                                   static_cast<double>(params_.infeasible_recovery_ticks);
    accel_max_override = params_.nom_accel_max * (0.5 + 0.5 * frac);
    --recovery_ticks_remaining_;
  }

  return attitudeTrackingCommand(a_des, accel_max_override);
}

std::vector<Eigen::Vector3d> RateAutopilotCore::escapeSearchDirections() const {
  // Multi-direction escape search (2026-08-22): the single fixed perpendicular-to-goal escape bias
  // (updateStallEscape()) can itself be a bad direction in a dense cluster -- confirmed on the …
  std::vector<Eigen::Vector3d> dirs;
  constexpr int kN = 8;
  constexpr double kTwoPi = 6.283185307179586;
  for (int k = 0; k < kN; ++k) {
    const double angle = kTwoPi * static_cast<double>(k) / static_cast<double>(kN);
    dirs.push_back(params_.escape_lateral_accel *
                   Eigen::Vector3d(std::cos(angle), std::sin(angle), 0.0));
  }
  // A lateral cluster may still have vertical room even when every lateral
  // direction is blocked.
  dirs.push_back(params_.escape_lateral_accel * Eigen::Vector3d(0.0, 0.0, 1.0));
  dirs.push_back(params_.escape_lateral_accel * Eigen::Vector3d(0.0, 0.0, -1.0));
  return dirs;
}

void RateAutopilotCore::integrateAndClampT(double tau, double dt) {
  const double T_new = T_ + tau * dt;
  const double T_clamped = std::clamp(T_new, T_min_, params_.T_max);
  if (T_clamped != T_new) {
    RCLCPP_WARN_THROTTLE(logger_, *clock_, 2000,
                         "T integration clamped: unclamped=%.4fN -> %.4fN (bounds=[%.3f, %.3f]) -- "
                         "integration drift or a persistently saturating tau; investigate if this "
                         "recurs.",
                         T_new, T_clamped, T_min_, params_.T_max);
  }
  T_ = T_clamped;
}

RateAutopilotCore::TickResult RateAutopilotCore::tick(std::optional<double> measured_dt_sec) {
  ++tick_count_;
  // Clamp a caller-supplied dt to a band around the nominal control period -- bounds a single
  // missed/late wall-clock tick without letting a real stall (e.g. odom-stale hold, see the …
  const double dt = measured_dt_sec.has_value()
                         ? std::clamp(*measured_dt_sec, 0.5 * dt_, 3.0 * dt_)
                         : dt_;
  TickResult result;

  ++penn_tick_counter_;
  if (penn_tick_counter_ >= params_.penn_update_ticks) {
    penn_tick_counter_ = 0;
    const double new_gamma = pennSelectAlpha(pos_enu_);
    if (new_gamma != gamma_obs_selected_) {
      gamma_obs_selected_ = new_gamma;
      rebuildQp();
    }
  }

  const double d_min = params_.obs_d_min;
  const QuadState state{pos_enu_, vel_enu_, R_, T_};
  const auto [tau_nom, omega_nom] = nominalCommand(dt);

  std::vector<Eigen::Vector3d> certified_positions;
  std::vector<Eigen::Vector3d> uncertified_positions;
  for (const auto& [id, entry] : obstacles_seen_) {
    if (currently_certified_ids_.count(id) != 0) {
      certified_positions.push_back(entry.center);
    } else {
      uncertified_positions.push_back(entry.center);
    }
  }


  std::optional<double> v_cap;
  if (params_.uncertified_v_cap_enabled) {
    v_cap = params_.uncertified_v_cap;
  }

  const auto& active_qp = *qp_;
  auto qp_result =
      active_qp.solve(state, certified_positions, uncertified_positions, d_min, tau_nom, omega_nom, v_cap);

  // Escape-direction search: while an escape is active, try a spread of candidate directions
  // (escapeSearchDirections()) IN ADDITION to the single fixed direction already solved above, …
  if (escape_active_ && params_.escape_search_enabled) {
    const auto search_dirs = escapeSearchDirections();
    const Eigen::Vector3d base_accel = positionTrackingAccel();

    if (escape_attempt_count_ != last_searched_attempt_) {
      last_searched_attempt_ = escape_attempt_count_;
      committed_escape_index_ = -1;  // baseline (already solved above) wins unless beaten below
      double best_score = qp_result.feasible ? qp_result.omega.norm() - 5.0 * qp_result.delta : -1.0;
      for (size_t k = 0; k < search_dirs.size(); ++k) {
        const Eigen::Vector3d a_des_c = params_.escape_goal_pull_scale * base_accel + search_dirs[k];
        const auto [tau_c, omega_c] = attitudeTrackingCommand(a_des_c);
        const auto candidate = active_qp.solve(state, certified_positions, uncertified_positions,
                                                d_min, tau_c, omega_c, v_cap);
        if (!candidate.feasible) continue;
        const double score = candidate.omega.norm() - 5.0 * candidate.delta;
        if (score > best_score) {
          best_score = score;
          qp_result = candidate;
          committed_escape_index_ = static_cast<int>(k);
        }
      }
    } else if (committed_escape_index_ >= 0 &&
              static_cast<size_t>(committed_escape_index_) < search_dirs.size()) {
      // Continuing an already-committed attempt: reuse the same direction (recomputed from current
      // state, same as the baseline always is -- only the CHOICE of direction persists, not a …
      const Eigen::Vector3d a_des_c =
          params_.escape_goal_pull_scale * base_accel + search_dirs[committed_escape_index_];
      const auto [tau_c, omega_c] = attitudeTrackingCommand(a_des_c);
      const auto candidate = active_qp.solve(state, certified_positions, uncertified_positions,
                                             d_min, tau_c, omega_c, v_cap);
      if (candidate.feasible) {
        qp_result = candidate;
      }
    }
    // committed_escape_index_ == -1 on a continuing attempt: the baseline
    // (tau_nom/omega_nom, already solved into qp_result above) stands.
  }

  // Diagnostic dump (Phase 3 follow-up, 2026-08-22): print each certified obstacle's linear barrier
  // row -- c_tau*tau - c_omega1*omega_x + c_omega2*omega_y >= RHS -- alongside the nominal and …
  if (std::getenv("DUMP_QP_ROWS") != nullptr && tick_count_ % 20 == 0) {
    std::fprintf(stderr, "[QPROWS tick=%llu] tau_nom=%.4f omega_nom=[%.4f %.4f %.4f]\n",
                 static_cast<unsigned long long>(tick_count_), tau_nom, omega_nom.x(),
                 omega_nom.y(), omega_nom.z());
    for (size_t i = 0; i < qp_result.certified_barrier_info.size(); ++i) {
      const auto& c = qp_result.certified_barrier_info[i];
      const double rhs = -c.L_ftilde - gamma_obs_selected_ * c.psi2;  // g3_ == gamma_obs_selected_, see rebuildQp()
      std::fprintf(stderr,
                   "  obs[%zu] c_tau=%.4f c_omega1=%.4f c_omega2=%.4f psi2=%.4f rhs=%.4f\n", i,
                   c.c_tau, c.c_omega1, c.c_omega2, c.psi2, rhs);
    }
    std::fprintf(stderr, "  -> solved: feasible=%d tau=%.4f omega=[%.4f %.4f %.4f] delta=%.4f\n",
                 qp_result.feasible, qp_result.tau, qp_result.omega.x(), qp_result.omega.y(),
                 qp_result.omega.z(), qp_result.delta);
  }

  Eigen::Vector3d omega_out = Eigen::Vector3d::Zero();
  if (qp_result.feasible) {
    integrateAndClampT(qp_result.tau, dt);
    omega_out = qp_result.omega;
  } else {
    // Braking fallback (2026-08-23, see nom_infeasible_brake_kv's comment): was "hold T, command
    // zero body rate" (coast at current velocity/attitude), which let a real hard collision …
    RCLCPP_ERROR_THROTTLE(logger_, *clock_, 1000,
                          "QP solve infeasible -- braking this cycle (was: holding T, zero body rate).");
    const Eigen::Vector3d a_brake = -params_.nom_infeasible_brake_kv * vel_enu_;
    // Braking is allowed more authority than cruise (see accel_max_override's comment) -- but this
    // path is open-loop, unlike the QP it's replacing, so nothing else clamps the resulting body …
    const auto [tau_brake, omega_brake] =
        attitudeTrackingCommand(a_brake, params_.nom_infeasible_brake_accel_max);
    integrateAndClampT(tau_brake, dt);
    omega_out = omega_brake.cwiseMax(-params_.qp_omega_max).cwiseMin(params_.qp_omega_max);
    // Arm the recovery ramp (see Params::infeasible_recovery_ticks) -- unconditional overwrite, so
    // a fresh infeasible tick during an already-active cooldown extends it rather than letting it …
    recovery_ticks_remaining_ = params_.infeasible_recovery_ticks;
  }

  if (qp_result.delta > 1e-3) {
    RCLCPP_WARN_THROTTLE(logger_, *clock_, 1000,
                         "slack in use: delta=%.4f across %zu uncertified obstacle(s) -- QP could "
                         "not satisfy the hard part of that row within actuator bounds this cycle.",
                         qp_result.delta, uncertified_positions.size());
  }

  result.feasible = qp_result.feasible;
  result.T = T_;
  result.omega = omega_out;
  result.delta = qp_result.delta;
  result.gamma_used = gamma_obs_selected_;
  result.used_penn = params_.penn_enabled && penn_model_.has_value();
  result.certified_count = certified_positions.size();
  result.uncertified_count = uncertified_positions.size();

  if (tick_count_ % 100 == 0) {
    RCLCPP_INFO(logger_,
                "pos=[%.2f %.2f %.2f]  T=%.3fN  omega=[%.3f %.3f %.3f]  certified=%zu "
                "uncertified=%zu  delta=%.4f  gamma=%.3f%s  feasible=%d",
                pos_enu_.x(), pos_enu_.y(), pos_enu_.z(), T_, omega_out.x(), omega_out.y(),
                omega_out.z(), result.certified_count, result.uncertified_count, qp_result.delta,
                gamma_obs_selected_, result.used_penn ? "(PENN)" : "(fixed)", qp_result.feasible);
  }

  return result;
}

}  // namespace nodelib
