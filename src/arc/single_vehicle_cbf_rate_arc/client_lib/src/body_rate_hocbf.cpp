#include "single_vehicle_cbf_rate/body_rate_hocbf.hpp"

namespace nodelib {

// `cylinder_mode` switches the obstacle model: false (default) -- sphere: h = |p - c|^2 - d_min^2,
// full 3-D distance.
BarrierChain collisionBarrierChain(const QuadState& state, const Eigen::Vector3d& p_obs,
                                    double d_min, double g1, double g2, double m,
                                    bool cylinder_mode) {
  const Eigen::Vector3d rel_p = state.p - p_obs;
  const Eigen::Vector3d& v_full = state.v;
  const Eigen::Matrix3d& R = state.R;
  const double T = state.T;

  // In cylinder mode the position/velocity used by the barrier are horizontal
  // projections; in sphere mode they are the full 3-D vectors (identity change).
  const Eigen::Vector3d rel =
      cylinder_mode ? Eigen::Vector3d(rel_p.x(), rel_p.y(), 0.0) : rel_p;
  const Eigen::Vector3d v =
      cylinder_mode ? Eigen::Vector3d(v_full.x(), v_full.y(), 0.0) : v_full;

  const double h = rel.dot(rel) - d_min * d_min;
  const double dh = 2.0 * rel.dot(v_full);

  const Eigen::Vector3d a = (T / m) * (R * Eigen::Vector3d::UnitZ()) - kQuadG * Eigen::Vector3d::UnitZ();
  const double ddh = 2.0 * v.dot(v) + 2.0 * rel.dot(a);

  const double psi0 = h;
  const double psi1 = dh + g1 * psi0;
  const double dpsi1 = ddh + g1 * dh;
  const double psi2 = dpsi1 + g2 * psi1;

  const Eigen::Vector3d r_tilde = R.transpose() * rel;

  const double c_tau = (2.0 / m) * r_tilde.z();
  const double c_omega2 = (2.0 * T / m) * r_tilde.x();
  const double c_omega1 = (2.0 * T / m) * r_tilde.y();

  const double L_ftilde = 6.0 * v.dot(a) + 2.0 * g1 * v.dot(v) + 2.0 * g1 * rel.dot(a) + g2 * dpsi1;

  BarrierChain chain;
  chain.h = h;
  chain.psi0 = psi0;
  chain.psi1 = psi1;
  chain.psi2 = psi2;
  chain.L_ftilde = L_ftilde;
  chain.c_tau = c_tau;
  chain.c_omega1 = c_omega1;
  chain.c_omega2 = c_omega2;
  chain.r_tilde = r_tilde;
  return chain;
}

ThrustFloorBarrier thrustFloorBarrier(const QuadState& state, double T_min, double g_T) {
  ThrustFloorBarrier result;
  result.h_T = state.T - T_min;
  result.rhs = -g_T * result.h_T;
  return result;
}

} 