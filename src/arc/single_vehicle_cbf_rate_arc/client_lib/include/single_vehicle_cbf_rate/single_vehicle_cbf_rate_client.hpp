#pragma once

#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include <Eigen/Dense>

#include "rclcpp/rclcpp.hpp"
#include "geometry_msgs/msg/point_stamped.hpp"
#include "nav_msgs/msg/odometry.hpp"
#include "visualization_msgs/msg/marker_array.hpp"
#include "px4_msgs/msg/offboard_control_mode.hpp"
#include "px4_msgs/msg/vehicle_rates_setpoint.hpp"
#include "px4_msgs/msg/vehicle_status.hpp"
#include "single_vehicle_cbf_rate_arc/msg/cbf_diagnostics.hpp"

#include "single_vehicle_cbf_rate/rate_autopilot_core.hpp"

namespace nodelib {

// PX4-specific I/O around RateAutopilotCore: reads ROS params into a
// RateAutopilotCore::Params, owns the PX4 thrust-curve conversion
// (computeThrustMax/normalizedThrustFromNewtons -- specific to PX4's
// normalized-actuator-output convention, not something the vehicle-agnostic
// core knows about), tracks PX4 arming state, and publishes
// VehicleRatesSetpoint/OffboardControlMode. All the actual CBF-QP/PENN
// control logic lives in RateAutopilotCore now; this class does not
// duplicate any of it.
class RateAutopilotClient : public rclcpp::Node {
 public:
  explicit RateAutopilotClient(const rclcpp::NodeOptions& options);

 private:
  double computeThrustMax() const;

  void vehicleStatusCallback(const px4_msgs::msg::VehicleStatus::ConstSharedPtr& msg);
  void odomCallback(const nav_msgs::msg::Odometry::ConstSharedPtr& msg);
  void obstaclesCallback(const visualization_msgs::msg::MarkerArray::ConstSharedPtr& msg);
  void goalCallback(const geometry_msgs::msg::PointStamped::ConstSharedPtr& msg);

  void tick();

  double normalizedThrustFromNewtons(double thrust_newtons) const;
  void publishRateSetpoint(double thrust_newtons, const Eigen::Vector3d& omega_flu);

  // Side-channel diagnostics for visualization / experiment logging. Nothing
  // in the control loop consumes /uav_0/cbf/diagnostics -- see the message
  // definition and EXPERIMENT_GUI_PLAN_20260829.md.
  void publishDiagnostics(const RateAutopilotCore::TickResult& result,
                          const Eigen::Vector3d& waypoint, double goal_arrival_tolerance);

  // ── bridge / thrust-curve params (PX4-specific) ──────────────────────────
  double vehicle_thrust_scaling_{0.08259793214};
  double vehicle_idle_thrust_{0.2535571429};
  int vehicle_num_rotors_{4};
  bool armed_{false};
  std::optional<bool> last_arming_state_;

  std::unique_ptr<RateAutopilotCore> core_;

  // Runtime goal override: set from the "goal_waypoint" topic and, on the
  // disarmed->armed transition, snapped to the current position for
  // hold-in-place. Once set, takes priority over the waypoint_x/y/z params
  // in tick(); before any message arrives it stays empty and the params
  // remain the default.
  std::optional<Eigen::Vector3d> waypoint_override_;

  // Wall-clock time of the previous tick() call, used to measure the real
  // elapsed dt to pass into core_->tick() instead of trusting the nominal
  // control period -- see RateAutopilotCore::tick()'s comment for why.
  std::optional<rclcpp::Time> last_tick_time_;

  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_sub_;
  rclcpp::Subscription<visualization_msgs::msg::MarkerArray>::SharedPtr obstacles_sub_;
  rclcpp::Subscription<px4_msgs::msg::VehicleStatus>::SharedPtr vehicle_status_sub_;
  rclcpp::Subscription<geometry_msgs::msg::PointStamped>::SharedPtr goal_sub_;

  rclcpp::Publisher<px4_msgs::msg::VehicleRatesSetpoint>::SharedPtr rates_setpoint_pub_;
  rclcpp::Publisher<px4_msgs::msg::VehicleRatesSetpoint>::SharedPtr rates_setpoint_debug_pub_;
  rclcpp::Publisher<px4_msgs::msg::OffboardControlMode>::SharedPtr vehicle_com_mode_pub_;
  rclcpp::Publisher<single_vehicle_cbf_rate_arc::msg::CbfDiagnostics>::SharedPtr cbf_diag_pub_;

  // Latest obstacle centres + physical radii (scale.x/2, no safety margin)
  // parsed from /arc/obstacles, kept only to compute min_obstacle_dist for
  // the diagnostics message. The core keeps its own copy for the QP.
  std::vector<std::pair<Eigen::Vector3d, double>> last_obstacles_;

  rclcpp::TimerBase::SharedPtr control_timer_;
};

}  // namespace nodelib
