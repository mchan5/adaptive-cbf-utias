#pragma once

#include <map>
#include <optional>
#include <utility>

#include <Eigen/Dense>

#include "single_vehicle_cbf_rate/body_rate_hocbf.hpp"
#include "single_vehicle_cbf_rate/quadrotor_dynamics.hpp"

namespace nodelib {

// Is this obstacle, at this instant, inside the region where the
// forward-invariance proof applies? Ported from entry_certification_gate.py.
struct CertificationResult {
  bool certified{false};
  BarrierChain chain;
};

CertificationResult isEntryCertified(const QuadState& state, const Eigen::Vector3d& p_obs,
                                      double d_min, double g1, double g2, double m,
                                      bool cylinder_mode = false);

// Tracks which obstacles have been entry-certified, and UNDER WHAT GAMMA.
// Originally certified once at first sighting and trusted forever
// (known_ids_, a plain set) -- fixed 2026-08-23 after a real hard collision
// (dense_20260921 scene 25) traced to exactly this: a certified obstacle
// becomes a HARD, zero-slack QP constraint (body_rate_qp_osqp.cpp splits
// certified vs uncertified_barrier_info this way), so a certification
// computed under the gamma active at first sighting being trusted after
// gamma later changes means the QP can be forced to satisfy a constraint
// under dynamics the forward-invariance proof was never checked against for
// that obstacle. Confirmed mechanistically: no single FIXED gamma (1.3-8.0)
// ever produces a collision in that scene; only the adaptive arm, which is
// the only one that changes gamma mid-approach, does. Tolerable at the old
// [0.5,2.5] range (max absolute swing 2.0); not at [1.3,8.0] (swing 6.7).
// No Python counterpart exists to keep in parity with (entry_certification_
// gate.py is not present in this repo -- point-mass has no analogous
// certified/uncertified split at all), so this is a C++-only fix.
class ObstacleRegistry {
 public:
  ObstacleRegistry(double g1, double g2, double m, bool cylinder_mode = false)
      : g1_(g1), g2_(g2), m_(m), cylinder_mode_(cylinder_mode) {}

  double g1() const { return g1_; }
  double g2() const { return g2_; }
  void setGammas(double g1, double g2) {
    g1_ = g1;
    g2_ = g2;
  }

  // Returns {certified, chain}. chain is empty only in the fast path (known,
  // already certified under the CURRENT g1/g2 -- no re-evaluation needed);
  // populated whenever certification was freshly (re-)computed, whether or
  // not it succeeded, matching the original contract callers rely on to log
  // a failed-certification warning.
  struct RegisterResult {
    bool certified{false};
    std::optional<BarrierChain> chain;
  };
  RegisterResult registerAndCheck(int obstacle_id, const QuadState& state,
                                   const Eigen::Vector3d& p_obs, double d_min);

 private:
  double g1_;
  double g2_;
  double m_;
  bool cylinder_mode_{false};
  // obstacle_id -> (g1,g2) it is currently certified under. Absent means
  // never certified (either never seen, or last attempt failed -- a failed
  // obstacle stays uncertified/soft, the safe default, and is retried the
  // next time gamma changes rather than permanently given up on).
  std::map<int, std::pair<double, double>> certified_gammas_;
};

}
