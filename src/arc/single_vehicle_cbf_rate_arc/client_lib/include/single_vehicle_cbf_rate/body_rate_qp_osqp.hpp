#pragma once

#include <optional>
#include <vector>

#include <Eigen/Dense>

#include "single_vehicle_cbf_rate/body_rate_hocbf.hpp"
#include "single_vehicle_cbf_rate/quadrotor_dynamics.hpp"

namespace nodelib {

// Body-rate CBF-QP, OSQP (v1 C API) backend. Ported from body_rate_qp_osqp.py.
// Decision vector: [tau, omega1, omega2, omega3, delta].
class BodyRateCBFQP {
 public:
  // cylinder_mode: false -> sphere collision barrier (original certified model);
  //                true  -> vertical-cylinder barrier (horizontal distance only).
  BodyRateCBFQP(double m, double g1, double g2, double g3, double T_min, double g_T,
                double tau_min, double tau_max, double omega_max, double rho = 1.0,
                double slack_weight = 1e4, bool cylinder_mode = false);

  struct Result {
    double tau{0.0};
    Eigen::Vector3d omega{Eigen::Vector3d::Zero()};
    double delta{0.0};
    bool feasible{false};
    std::vector<BarrierChain> certified_barrier_info;
    std::vector<BarrierChain> uncertified_barrier_info;
    ThrustFloorBarrier thrust_floor_info;
  };

  Result solve(const QuadState& state, const std::vector<Eigen::Vector3d>& certified_obstacles,
               const std::vector<Eigen::Vector3d>& uncertified_obstacles, double d_min,
               double tau_nom, const Eigen::Vector3d& omega_nom,
               std::optional<double> uncertified_v_cap = std::nullopt) const;

 private:
  double m_;
  double g1_, g2_, g3_;
  double T_min_, g_T_;
  double tau_min_, tau_max_, omega_max_;
  double rho_;
  double slack_weight_;
  bool cylinder_mode_{false};

  static QuadState velocityCappedState(const QuadState& state, const Eigen::Vector3d& p_obs,
                                        double v_cap, bool cylinder_mode = false);
};

} 
