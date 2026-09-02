#include "single_vehicle_cbf_rate/body_rate_qp_osqp.hpp"

#include <cmath>

#include <osqp.h>

namespace nodelib {

BodyRateCBFQP::BodyRateCBFQP(double m, double g1, double g2, double g3, double T_min, double g_T,
                             double tau_min, double tau_max, double omega_max, double rho,
                             double slack_weight, bool cylinder_mode)
    : m_(m),
      g1_(g1),
      g2_(g2),
      g3_(g3),
      T_min_(T_min),
      g_T_(g_T),
      tau_min_(tau_min),
      tau_max_(tau_max),
      omega_max_(omega_max),
      rho_(rho),
      slack_weight_(slack_weight),
      cylinder_mode_(cylinder_mode) {}

QuadState BodyRateCBFQP::velocityCappedState(const QuadState& state, const Eigen::Vector3d& p_obs,
                                              double v_cap, bool cylinder_mode) {
  const Eigen::Vector3d rel_p = state.p - p_obs;
  // In cylinder mode the closing-speed cap acts on the horizontal radial
  // direction only (consistent with the vertical-cylinder collision barrier);
  // in sphere mode it is the full 3-D radial direction (original behavior).
  const Eigen::Vector3d rel =
      cylinder_mode ? Eigen::Vector3d(rel_p.x(), rel_p.y(), 0.0) : rel_p;
  const double dist = rel.norm();
  if (dist < 1e-6) {
    return state;
  }
  const Eigen::Vector3d radial_dir = rel / dist;  // points AWAY from the obstacle
  const double v_radial = -radial_dir.dot(state.v);  // positive == closing
  if (v_radial <= v_cap) {
    return state;
  }
  const double v_reduction = v_radial - v_cap;
  QuadState capped = state;
  capped.v = state.v + v_reduction * radial_dir;
  return capped;
}

namespace {
struct CscBuilder {
  static constexpr int kCols = 5;
  std::vector<std::vector<std::pair<OSQPInt, OSQPFloat>>> col_entries{kCols};
  OSQPInt next_row{0};

  void addEntry(int col, double value) { col_entries[col].push_back({next_row, value}); }
  void endRow() { ++next_row; }

  void toCsc(std::vector<OSQPInt>* p, std::vector<OSQPInt>* i, std::vector<OSQPFloat>* x) const {
    p->assign(kCols + 1, 0);
    i->clear();
    x->clear();
    for (int c = 0; c < kCols; ++c) {
      (*p)[c + 1] = (*p)[c] + static_cast<OSQPInt>(col_entries[c].size());
      for (const auto& [row, val] : col_entries[c]) {
        i->push_back(row);
        x->push_back(val);
      }
    }
  }
};

}

BodyRateCBFQP::Result BodyRateCBFQP::solve(const QuadState& state,
                                            const std::vector<Eigen::Vector3d>& certified_obstacles,
                                            const std::vector<Eigen::Vector3d>& uncertified_obstacles,
                                            double d_min, double tau_nom,
                                            const Eigen::Vector3d& omega_nom,
                                            std::optional<double> uncertified_v_cap) const {
  Result result;

  result.certified_barrier_info.reserve(certified_obstacles.size());
  for (const auto& p_o : certified_obstacles) {
    result.certified_barrier_info.push_back(
        collisionBarrierChain(state, p_o, d_min, g1_, g2_, m_, cylinder_mode_));
  }

  result.uncertified_barrier_info.reserve(uncertified_obstacles.size());
  for (const auto& p_o : uncertified_obstacles) {
    const QuadState eval_state =
        uncertified_v_cap ? velocityCappedState(state, p_o, *uncertified_v_cap, cylinder_mode_)
                          : state;
    result.uncertified_barrier_info.push_back(
        collisionBarrierChain(eval_state, p_o, d_min, g1_, g2_, m_, cylinder_mode_));
  }

  result.thrust_floor_info = thrustFloorBarrier(state, T_min_, g_T_);

  std::vector<OSQPInt> p_p = {0, 1, 2, 3, 4, 5};
  std::vector<OSQPInt> p_i = {0, 1, 2, 3, 4};
  std::vector<OSQPFloat> p_x = {rho_, 1.0, 1.0, 1.0, 2.0 * slack_weight_};
  OSQPCscMatrix* P = OSQPCscMatrix_new(5, 5, 5, p_x.data(), p_i.data(), p_p.data());

  const OSQPFloat q[5] = {
      static_cast<OSQPFloat>(-rho_ * tau_nom),
      static_cast<OSQPFloat>(-omega_nom.x()),
      static_cast<OSQPFloat>(-omega_nom.y()),
      static_cast<OSQPFloat>(-omega_nom.z()),
      0.0,
  };

  // A / l / u
  CscBuilder builder;
  std::vector<OSQPFloat> l_vals;
  std::vector<OSQPFloat> u_vals;

  for (const auto& c : result.certified_barrier_info) {
    builder.addEntry(0, c.c_tau);
    builder.addEntry(1, -c.c_omega1);
    builder.addEntry(2, c.c_omega2);
    l_vals.push_back(static_cast<OSQPFloat>(-c.L_ftilde - g3_ * c.psi2));
    u_vals.push_back(OSQP_INFTY);
    builder.endRow();
  }

  for (const auto& c : result.uncertified_barrier_info) {
    builder.addEntry(0, c.c_tau);
    builder.addEntry(1, -c.c_omega1);
    builder.addEntry(2, c.c_omega2);
    builder.addEntry(4, 1.0);
    l_vals.push_back(static_cast<OSQPFloat>(-c.L_ftilde - g3_ * c.psi2));
    u_vals.push_back(OSQP_INFTY);
    builder.endRow();
  }

  // thrust floor: tau >= tf_rhs (no slack)
  builder.addEntry(0, 1.0);
  l_vals.push_back(static_cast<OSQPFloat>(result.thrust_floor_info.rhs));
  u_vals.push_back(OSQP_INFTY);
  builder.endRow();

  // box bounds, one row per control dimension (no slack)
  const double box_lb[4] = {tau_min_, -omega_max_, -omega_max_, -omega_max_};
  const double box_ub[4] = {tau_max_, omega_max_, omega_max_, omega_max_};
  for (int i = 0; i < 4; ++i) {
    builder.addEntry(i, 1.0);
    l_vals.push_back(static_cast<OSQPFloat>(box_lb[i]));
    u_vals.push_back(static_cast<OSQPFloat>(box_ub[i]));
    builder.endRow();
  }

  // delta >= 0
  builder.addEntry(4, 1.0);
  l_vals.push_back(0.0);
  u_vals.push_back(OSQP_INFTY);
  builder.endRow();

  std::vector<OSQPInt> a_p, a_i;
  std::vector<OSQPFloat> a_x;
  builder.toCsc(&a_p, &a_i, &a_x);
  const OSQPInt m_rows = builder.next_row;
  OSQPCscMatrix* A = OSQPCscMatrix_new(m_rows, 5, static_cast<OSQPInt>(a_x.size()), a_x.data(),
                                       a_i.data(), a_p.data());

  OSQPSettings settings;
  osqp_set_default_settings(&settings);
  settings.verbose = 0;
  settings.eps_abs = 1e-5;
  settings.eps_rel = 1e-5;
  settings.max_iter = 4000;
  settings.polishing = 1;

  OSQPSolver* solver = nullptr;
  const OSQPInt setup_flag =
      osqp_setup(&solver, P, q, A, l_vals.data(), u_vals.data(), m_rows, 5, &settings);

  if (setup_flag == 0) {
    osqp_solve(solver);
    const OSQPInt status = solver->info->status_val;
    result.feasible = (status == OSQP_SOLVED || status == OSQP_SOLVED_INACCURATE);
    if (result.feasible) {
      const OSQPFloat* x = solver->solution->x;
      result.tau = x[0];
      result.omega = Eigen::Vector3d(x[1], x[2], x[3]);
      result.delta = x[4];
    }
    osqp_cleanup(solver);
  } else {
    result.feasible = false;
  }

  OSQPCscMatrix_free(P);
  OSQPCscMatrix_free(A);

  return result;
}

} 
