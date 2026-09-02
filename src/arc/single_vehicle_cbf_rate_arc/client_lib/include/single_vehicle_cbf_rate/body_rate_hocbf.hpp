#pragma once

#include <Eigen/Dense>

#include "single_vehicle_cbf_rate/quadrotor_dynamics.hpp"

namespace nodelib {

struct BarrierChain {
  double h{0.0};
  double psi0{0.0};
  double psi1{0.0};
  double psi2{0.0};
  double L_ftilde{0.0};
  double c_tau{0.0};
  double c_omega1{0.0};
  double c_omega2{0.0};
  Eigen::Vector3d r_tilde{Eigen::Vector3d::Zero()};
};

// cylinder_mode == false -> sphere barrier (full 3-D distance, original behavior).
// cylinder_mode == true  -> vertical-cylinder barrier (horizontal distance only).
BarrierChain collisionBarrierChain(const QuadState& state, const Eigen::Vector3d& p_obs,
                                    double d_min, double g1, double g2, double m,
                                    bool cylinder_mode = false);

// Thrust-floor barrier (relative degree 1 in tau)
struct ThrustFloorBarrier {
  double h_T{0.0};
  double rhs{0.0};
};

ThrustFloorBarrier thrustFloorBarrier(const QuadState& state, double T_min, double g_T);

}  
