#include "single_vehicle_cbf_rate/single_vehicle_cbf_rate_client.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <cstdlib>

#include "ament_index_cpp/get_package_share_directory.hpp"

namespace nodelib {
namespace {

std::array<float, 3> interconvertFluFrd(float x, float y, float z) {
  return {x, -y, -z};
}

}  // namespace

RateAutopilotClient::RateAutopilotClient(const rclcpp::NodeOptions& options)
    : Node("single_vehicle_cbf_rate_node", options) {
  // bridge / thrust-curve params
  this->declare_parameter<double>("vehicle_thrust_scaling", vehicle_thrust_scaling_);
  this->declare_parameter<double>("vehicle_idle_thrust", vehicle_idle_thrust_);
  this->declare_parameter<int>("vehicle_num_rotors", vehicle_num_rotors_);
  vehicle_thrust_scaling_ = this->get_parameter("vehicle_thrust_scaling").as_double();
  vehicle_idle_thrust_ = this->get_parameter("vehicle_idle_thrust").as_double();
  vehicle_num_rotors_ = this->get_parameter("vehicle_num_rotors").as_int();

  // waypoint / PENN goal
  this->declare_parameter<double>("waypoint_x", 0.0);
  this->declare_parameter<double>("waypoint_y", 0.0);
  this->declare_parameter<double>("waypoint_z", 1.5);

  // nominal position/attitude control gains, feeding the CBF-QP's nominal reference
  this->declare_parameter<double>("nom_pos_kp", 1.5);
  this->declare_parameter<double>("nom_vel_kd", 2.0);
  this->declare_parameter<double>("nom_pos_ki", 0.3);
  this->declare_parameter<double>("nom_pos_i_max", 2.0);
  this->declare_parameter<double>("nom_accel_max", 3.0);
  this->declare_parameter<double>("nom_yaw_des", 0.0);
  this->declare_parameter<double>("nom_k_r", 5.0);
  this->declare_parameter<double>("nom_k_t", 8.0);

  // stall-triggered escape maneuver
  this->declare_parameter<double>("stall_speed_threshold", 0.08);
  this->declare_parameter<double>("stall_duration_sec", 2.5);
  this->declare_parameter<double>("escape_duration_sec", 2.5);
  this->declare_parameter<double>("escape_lateral_accel", 2.5);
  this->declare_parameter<double>("escape_goal_pull_scale", 0.3);
  this->declare_parameter<double>("goal_arrival_tolerance", 0.3);

  // vehicle + QP parameters
  RateAutopilotCore::Params params;
  this->declare_parameter<double>("vehicle_mass", params.vehicle_mass);
  this->declare_parameter<double>("obs_d_min", params.obs_d_min);
  this->declare_parameter<double>("obs_safety_margin", params.obs_safety_margin);

  this->declare_parameter<double>("qp_thrust_min", 4.7);
  this->declare_parameter<double>("qp_thrust_floor_gain", 8.0);
  this->declare_parameter<double>("qp_tau_min", -50.0);
  this->declare_parameter<double>("qp_tau_max", 50.0);
  this->declare_parameter<double>("qp_omega_max", 6.0);
  this->declare_parameter<double>("qp_rho", 1.0);
  this->declare_parameter<double>("control_rate_hz", params.control_rate_hz);

  this->declare_parameter<double>("qp_slack_weight", 1e4);
  this->declare_parameter<bool>("uncertified_v_cap_enabled", false);
  this->declare_parameter<double>("uncertified_v_cap", 1.0);

  this->declare_parameter<double>("cbf_gamma_obs", params.cbf_gamma_obs);
  // Obstacle model: false -> sphere barrier (default, original certified behavior); true ->
  // vertical-cylinder barrier (horizontal distance only).
  this->declare_parameter<bool>("cbf_cylinder_barrier", params.cbf_cylinder_barrier);

  // PENN+GAT adaptive gamma parameters
  this->declare_parameter<bool>("penn_enabled", true);
  this->declare_parameter<std::string>(
      "penn_model_dir", ament_index_cpp::get_package_share_directory("single_vehicle_cbf_rate_arc") + "/nn_model");
  this->declare_parameter<std::string>("penn_model_path", "checkpoint/gat_penn_weights.pt");
  this->declare_parameter<double>("penn_gamma_min", 0.5);
  this->declare_parameter<double>("penn_gamma_max", 10.0);
  this->declare_parameter<int>("penn_gamma_steps", 20);
  this->declare_parameter<int>("penn_update_ticks", 20);
  this->declare_parameter<double>("penn_proximity_gate_m", 3.0);
  this->declare_parameter<double>("epistemic_threshold", 0.046695);
  this->declare_parameter<double>("cvar_boundary", 0.1570);
  this->declare_parameter<double>("deadlock_threshold", 5.0);
  this->declare_parameter<double>("lcb_k", 0.0);
  this->declare_parameter<double>("penn_fallback_step", 0.5);
  this->declare_parameter<bool>("penn_feasibility_gate", false);

  params.vehicle_mass = this->get_parameter("vehicle_mass").as_double();
  params.obs_d_min = this->get_parameter("obs_d_min").as_double();
  params.obs_safety_margin = this->get_parameter("obs_safety_margin").as_double();
  params.control_rate_hz = this->get_parameter("control_rate_hz").as_double();
  params.cbf_gamma_obs = this->get_parameter("cbf_gamma_obs").as_double();
  params.cbf_cylinder_barrier = this->get_parameter("cbf_cylinder_barrier").as_bool();

  params.qp_thrust_min = this->get_parameter("qp_thrust_min").as_double();
  params.qp_thrust_floor_gain = this->get_parameter("qp_thrust_floor_gain").as_double();
  params.qp_tau_min = this->get_parameter("qp_tau_min").as_double();
  params.qp_tau_max = this->get_parameter("qp_tau_max").as_double();
  params.qp_omega_max = this->get_parameter("qp_omega_max").as_double();
  params.qp_rho = this->get_parameter("qp_rho").as_double();
  params.qp_slack_weight = this->get_parameter("qp_slack_weight").as_double();

  params.T_max = computeThrustMax();

  const std::string model_dir = this->get_parameter("penn_model_dir").as_string();
  const std::string model_rel = this->get_parameter("penn_model_path").as_string();
  params.penn_model_path = model_dir + "/" + model_rel;
  params.penn_enabled = this->get_parameter("penn_enabled").as_bool();
  params.penn_gamma_min = this->get_parameter("penn_gamma_min").as_double();
  params.penn_gamma_max = this->get_parameter("penn_gamma_max").as_double();
  params.penn_gamma_steps = static_cast<int>(this->get_parameter("penn_gamma_steps").as_int());
  params.penn_update_ticks = static_cast<int>(this->get_parameter("penn_update_ticks").as_int());
  params.penn_proximity_gate_m = this->get_parameter("penn_proximity_gate_m").as_double();
  params.epistemic_threshold = this->get_parameter("epistemic_threshold").as_double();
  params.cvar_boundary = this->get_parameter("cvar_boundary").as_double();
  params.deadlock_threshold = this->get_parameter("deadlock_threshold").as_double();
  params.lcb_k = this->get_parameter("lcb_k").as_double();
  params.fallback_step = this->get_parameter("penn_fallback_step").as_double();
  params.penn_feasibility_gate = this->get_parameter("penn_feasibility_gate").as_bool();

  params.waypoint = Eigen::Vector3d(this->get_parameter("waypoint_x").as_double(),
                                     this->get_parameter("waypoint_y").as_double(),
                                     this->get_parameter("waypoint_z").as_double());
  params.nom_pos_kp = this->get_parameter("nom_pos_kp").as_double();
  params.nom_vel_kd = this->get_parameter("nom_vel_kd").as_double();
  params.nom_pos_ki = this->get_parameter("nom_pos_ki").as_double();
  params.nom_pos_i_max = this->get_parameter("nom_pos_i_max").as_double();
  params.nom_accel_max = this->get_parameter("nom_accel_max").as_double();
  params.nom_yaw_des = this->get_parameter("nom_yaw_des").as_double();
  params.nom_k_r = this->get_parameter("nom_k_r").as_double();
  params.nom_k_t = this->get_parameter("nom_k_t").as_double();
  params.stall_speed_threshold = this->get_parameter("stall_speed_threshold").as_double();
  params.stall_duration_sec = this->get_parameter("stall_duration_sec").as_double();
  params.escape_duration_sec = this->get_parameter("escape_duration_sec").as_double();
  params.escape_lateral_accel = this->get_parameter("escape_lateral_accel").as_double();
  params.escape_goal_pull_scale = this->get_parameter("escape_goal_pull_scale").as_double();
  params.goal_arrival_tolerance = this->get_parameter("goal_arrival_tolerance").as_double();
  params.uncertified_v_cap_enabled = this->get_parameter("uncertified_v_cap_enabled").as_bool();
  params.uncertified_v_cap = this->get_parameter("uncertified_v_cap").as_double();

  core_ = std::make_unique<RateAutopilotCore>(params, this->get_logger(), this->get_clock());

  const auto qos_px4 = rclcpp::QoS(10)
                           .reliability(RMW_QOS_POLICY_RELIABILITY_BEST_EFFORT)
                           .durability(RMW_QOS_POLICY_DURABILITY_VOLATILE);
  const auto qos_px4_out = rclcpp::QoS(10)
                                .reliability(RMW_QOS_POLICY_RELIABILITY_BEST_EFFORT)
                                .durability(RMW_QOS_POLICY_DURABILITY_TRANSIENT_LOCAL);
  const auto qos_sub = rclcpp::QoS(1)
                           .reliability(RMW_QOS_POLICY_RELIABILITY_BEST_EFFORT)
                           .durability(RMW_QOS_POLICY_DURABILITY_VOLATILE);

  odom_sub_ = this->create_subscription<nav_msgs::msg::Odometry>(
      "state_estimator/local_position/odom", qos_sub,
      [this](const nav_msgs::msg::Odometry::ConstSharedPtr& msg) { odomCallback(msg); });
  obstacles_sub_ = this->create_subscription<visualization_msgs::msg::MarkerArray>(
      "/arc/obstacles", qos_sub,
      [this](const visualization_msgs::msg::MarkerArray::ConstSharedPtr& msg) { obstaclesCallback(msg); });
  // The versioned suffix this checkout's uxrce_dds_client publishes under has drifted before
  // (previously "_v1", not "_v4") and differs between SITL and a real board's firmware build --
  // a parameter, not a hardcoded string, so each launch file can pin the value that's actually
  // live (config/px4/verify_vehicle_status_topic.sh confirms which). Default matches SITL.
  const std::string vehicle_status_topic =
      this->declare_parameter<std::string>("vehicle_status_topic", "fmu/out/vehicle_status");
  vehicle_status_sub_ = this->create_subscription<px4_msgs::msg::VehicleStatus>(
      vehicle_status_topic, qos_px4_out,
      [this](const px4_msgs::msg::VehicleStatus::ConstSharedPtr& msg) { vehicleStatusCallback(msg); });

  // Runtime goal-waypoint command. Relative name -- resolves under this node's namespace, i.e.
  // /<uav_prefix>/goal_waypoint (e.g. /uav_0/goal_waypoint).
  const auto qos_goal = rclcpp::QoS(1).reliability(RMW_QOS_POLICY_RELIABILITY_RELIABLE)
                            .durability(RMW_QOS_POLICY_DURABILITY_VOLATILE);
  goal_sub_ = this->create_subscription<geometry_msgs::msg::PointStamped>(
      "goal_waypoint", qos_goal,
      [this](const geometry_msgs::msg::PointStamped::ConstSharedPtr& msg) { goalCallback(msg); });

  rates_setpoint_pub_ = this->create_publisher<px4_msgs::msg::VehicleRatesSetpoint>(
      "fmu/in/vehicle_rates_setpoint", qos_px4);
  rates_setpoint_debug_pub_ = this->create_publisher<px4_msgs::msg::VehicleRatesSetpoint>(
      "fsc_autopilot_ros2/rate_setpoint_debug", qos_px4);
  vehicle_com_mode_pub_ = this->create_publisher<px4_msgs::msg::OffboardControlMode>(
      "fmu/in/offboard_control_mode", qos_px4);

  // Side-channel diagnostics -- visualization / experiment logging only. BEST_EFFORT + shallow
  // depth so a slow subscriber cannot back-pressure this 100 Hz node.
  const auto qos_diag = rclcpp::QoS(5)
                            .reliability(RMW_QOS_POLICY_RELIABILITY_BEST_EFFORT)
                            .durability(RMW_QOS_POLICY_DURABILITY_VOLATILE);
  cbf_diag_pub_ = this->create_publisher<single_vehicle_cbf_rate_arc::msg::CbfDiagnostics>(
      "cbf/diagnostics", qos_diag);

  const double dt = 1.0 / params.control_rate_hz;
  control_timer_ = this->create_wall_timer(std::chrono::duration<double>(dt), [this] { tick(); });

  RCLCPP_INFO(this->get_logger(),
              "single_vehicle_cbf_rate_node started: control_rate=%.1fHz (dt=%.4fs), "
              "T_min=%.3fN T_max=%.3fN",
              params.control_rate_hz, dt, params.qp_thrust_min, params.T_max);
}

double RateAutopilotClient::computeThrustMax() const {
  return (1.0 - vehicle_idle_thrust_) / vehicle_thrust_scaling_ * vehicle_num_rotors_;
}

void RateAutopilotClient::vehicleStatusCallback(const px4_msgs::msg::VehicleStatus::ConstSharedPtr& msg) {
  const bool now_armed = (msg->arming_state == px4_msgs::msg::VehicleStatus::ARMING_STATE_ARMED);
  if (now_armed != armed_) {
    RCLCPP_WARN(this->get_logger(), "arming_state changed: %s -> %s", armed_ ? "ARMED" : "DISARMED",
                now_armed ? "ARMED" : "DISARMED");
  }
  if (last_arming_state_.has_value() && now_armed && !(*last_arming_state_)) {
    core_->resetT("re-armed");
    // Auto-hover-in-place: snap the goal to wherever the drone is the instant it arms/enters
    // offboard, so it holds position with zero operator action.
    if (core_->odomValid()) {
      waypoint_override_ = core_->position();
      const Eigen::Vector3d& wp = *waypoint_override_;
      RCLCPP_INFO(this->get_logger(),
                  "armed: holding current position [%.3f, %.3f, %.3f] as goal", wp.x(), wp.y(),
                  wp.z());
    } else {
      RCLCPP_WARN(this->get_logger(),
                  "armed before odometry valid -- cannot hold-in-place, falling back to "
                  "waypoint_x/y/z params");
    }
  }
  last_arming_state_ = now_armed;
  armed_ = now_armed;
}

void RateAutopilotClient::odomCallback(const nav_msgs::msg::Odometry::ConstSharedPtr& msg) {
  core_->handleOdometry(*msg, this->get_clock()->now());
}

void RateAutopilotClient::obstaclesCallback(
    const visualization_msgs::msg::MarkerArray::ConstSharedPtr& msg) {
  core_->handleObstacles(*msg);

  last_obstacles_.clear();
  last_obstacles_.reserve(msg->markers.size());
  for (const auto& m : msg->markers) {
    last_obstacles_.emplace_back(
        Eigen::Vector3d(m.pose.position.x, m.pose.position.y, m.pose.position.z),
        0.5 * m.scale.x);
  }
}

void RateAutopilotClient::goalCallback(const geometry_msgs::msg::PointStamped::ConstSharedPtr& msg) {
  waypoint_override_ = Eigen::Vector3d(msg->point.x, msg->point.y, msg->point.z);
  RCLCPP_INFO(this->get_logger(), "goal_waypoint received: [%.3f, %.3f, %.3f] (overrides waypoint_x/y/z params)",
              msg->point.x, msg->point.y, msg->point.z);
}

void RateAutopilotClient::tick() {
  if (!core_->odomValid()) {
    publishRateSetpoint(core_->T(), Eigen::Vector3d::Zero());
    return;
  }

  const rclcpp::Time now = this->get_clock()->now();
  const double age = core_->odomAgeSec(now);
  if (age > 1.0) {
    RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 1000,
                         "Odometry feed stale (%.1fs) -- holding last T, zero rates.", age);
    publishRateSetpoint(core_->T(), Eigen::Vector3d::Zero());
    return;
  }

  // Re-read the params that were live-tunable (via `ros2 param set`) in the
  // pre-refactor combined node -- see RateAutopilotCore::updateTunableParams.
  RateAutopilotCore::Params live = core_->params();
  live.waypoint = waypoint_override_.value_or(
      Eigen::Vector3d(this->get_parameter("waypoint_x").as_double(),
                      this->get_parameter("waypoint_y").as_double(),
                      this->get_parameter("waypoint_z").as_double()));
  live.nom_pos_kp = this->get_parameter("nom_pos_kp").as_double();
  live.nom_vel_kd = this->get_parameter("nom_vel_kd").as_double();
  live.nom_pos_ki = this->get_parameter("nom_pos_ki").as_double();
  live.nom_pos_i_max = this->get_parameter("nom_pos_i_max").as_double();
  live.nom_accel_max = this->get_parameter("nom_accel_max").as_double();
  live.nom_yaw_des = this->get_parameter("nom_yaw_des").as_double();
  live.nom_k_r = this->get_parameter("nom_k_r").as_double();
  live.nom_k_t = this->get_parameter("nom_k_t").as_double();
  live.stall_speed_threshold = this->get_parameter("stall_speed_threshold").as_double();
  live.stall_duration_sec = this->get_parameter("stall_duration_sec").as_double();
  live.escape_duration_sec = this->get_parameter("escape_duration_sec").as_double();
  live.escape_lateral_accel = this->get_parameter("escape_lateral_accel").as_double();
  live.escape_goal_pull_scale = this->get_parameter("escape_goal_pull_scale").as_double();
  live.goal_arrival_tolerance = this->get_parameter("goal_arrival_tolerance").as_double();
  live.penn_enabled = this->get_parameter("penn_enabled").as_bool();
  live.penn_gamma_min = this->get_parameter("penn_gamma_min").as_double();
  live.penn_gamma_max = this->get_parameter("penn_gamma_max").as_double();
  live.penn_gamma_steps = static_cast<int>(this->get_parameter("penn_gamma_steps").as_int());
  live.penn_proximity_gate_m = this->get_parameter("penn_proximity_gate_m").as_double();
  live.epistemic_threshold = this->get_parameter("epistemic_threshold").as_double();
  live.cvar_boundary = this->get_parameter("cvar_boundary").as_double();
  live.deadlock_threshold = this->get_parameter("deadlock_threshold").as_double();
  live.lcb_k = this->get_parameter("lcb_k").as_double();
  live.uncertified_v_cap_enabled = this->get_parameter("uncertified_v_cap_enabled").as_bool();
  live.uncertified_v_cap = this->get_parameter("uncertified_v_cap").as_double();
  core_->updateTunableParams(live);

  // cbf_gamma_obs is deliberately excluded from updateTunableParams (a gamma change needs
  // rebuildQp()); push it in via the dedicated setter so the fixed-gamma arm of the experiment …
  core_->setFixedGamma(this->get_parameter("cbf_gamma_obs").as_double());

  // Measure real elapsed time between ticks rather than trusting the nominal control period: this
  // timer is wall-clock, but under SITL, PX4/Gazebo run lockstepped to each other on sim-time …
  std::optional<double> measured_dt;
  if (last_tick_time_.has_value() && std::getenv("DISABLE_MEASURED_DT") == nullptr) {
    measured_dt = (now - *last_tick_time_).seconds();
  }
  last_tick_time_ = now;

  const auto result = core_->tick(measured_dt);
  publishRateSetpoint(result.T, result.feasible ? result.omega : Eigen::Vector3d::Zero());
  publishDiagnostics(result, live.waypoint, live.goal_arrival_tolerance);
}

void RateAutopilotClient::publishDiagnostics(const RateAutopilotCore::TickResult& result,
                                             const Eigen::Vector3d& waypoint,
                                             double goal_arrival_tolerance) {
  single_vehicle_cbf_rate_arc::msg::CbfDiagnostics d;
  d.header.stamp = this->get_clock()->now();
  d.header.frame_id = "map";

  d.gamma_used = result.gamma_used;
  d.used_penn = result.used_penn;
  d.gamma_fixed_param = this->get_parameter("cbf_gamma_obs").as_double();
  d.penn_gamma_min = this->get_parameter("penn_gamma_min").as_double();
  d.penn_gamma_max = this->get_parameter("penn_gamma_max").as_double();

  d.feasible = result.feasible;
  d.delta = result.delta;
  d.certified_count = static_cast<uint32_t>(result.certified_count);
  d.uncertified_count = static_cast<uint32_t>(result.uncertified_count);
  d.thrust = result.T;

  d.d_hat.x = result.d_hat.x();
  d.d_hat.y = result.d_hat.y();
  d.d_hat.z = result.d_hat.z();

  d.waypoint.x = waypoint.x();
  d.waypoint.y = waypoint.y();
  d.waypoint.z = waypoint.z();

  const Eigen::Vector3d pos = core_->position();
  const double dist_to_goal = (pos - waypoint).norm();
  d.dist_to_goal = dist_to_goal;
  d.reached = dist_to_goal < goal_arrival_tolerance;

  double min_obstacle_dist = 1e9;
  for (const auto& [center, radius] : last_obstacles_) {
    min_obstacle_dist = std::min(min_obstacle_dist, (pos - center).norm() - radius);
  }
  d.min_obstacle_dist = min_obstacle_dist;

  cbf_diag_pub_->publish(d);
}

double RateAutopilotClient::normalizedThrustFromNewtons(double thrust_newtons) const {
  return std::clamp(vehicle_thrust_scaling_ * thrust_newtons / vehicle_num_rotors_ +
                         vehicle_idle_thrust_,
                     0.0, 1.0);
}

void RateAutopilotClient::publishRateSetpoint(double thrust_newtons, const Eigen::Vector3d& omega_flu) {
  const auto omega_frd =
      interconvertFluFrd(static_cast<float>(omega_flu.x()), static_cast<float>(omega_flu.y()),
                         static_cast<float>(omega_flu.z()));
  const double t_normalized = normalizedThrustFromNewtons(thrust_newtons);

  const uint64_t now_us = this->get_clock()->now().nanoseconds() / 1000ULL;

  px4_msgs::msg::VehicleRatesSetpoint rates_setpoint;
  rates_setpoint.timestamp = now_us;
  rates_setpoint.roll = omega_frd[0];
  rates_setpoint.pitch = omega_frd[1];
  rates_setpoint.yaw = omega_frd[2];
  rates_setpoint.thrust_body = {0.0f, 0.0f, -static_cast<float>(t_normalized)};

  if (armed_) {
    rates_setpoint_pub_->publish(rates_setpoint);
  }
  rates_setpoint_debug_pub_->publish(rates_setpoint);

  px4_msgs::msg::OffboardControlMode control_mode;
  control_mode.timestamp = now_us;
  control_mode.position = false;
  control_mode.velocity = false;
  control_mode.acceleration = false;
  control_mode.attitude = false;
  control_mode.body_rate = true;
  control_mode.thrust_and_torque = true;
  vehicle_com_mode_pub_->publish(control_mode);
}

}  // namespace nodelib
