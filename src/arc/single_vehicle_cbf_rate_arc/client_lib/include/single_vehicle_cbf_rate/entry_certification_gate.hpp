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

  // Returns {certified, chain}.
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
  // obstacle_id -> (g1,g2) it is currently certified under.
  std::map<int, std::pair<double, double>> certified_gammas_;
};

}
