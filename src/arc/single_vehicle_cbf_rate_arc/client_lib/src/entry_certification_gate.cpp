#include "single_vehicle_cbf_rate/entry_certification_gate.hpp"

namespace nodelib {

CertificationResult isEntryCertified(const QuadState& state, const Eigen::Vector3d& p_obs,
                                      double d_min, double g1, double g2, double m,
                                      bool cylinder_mode) {
  CertificationResult result;
  result.chain = collisionBarrierChain(state, p_obs, d_min, g1, g2, m, cylinder_mode);
  result.certified = result.chain.psi0 >= 0 && result.chain.psi1 >= 0 && result.chain.psi2 >= 0;
  return result;
}

ObstacleRegistry::RegisterResult ObstacleRegistry::registerAndCheck(
    int obstacle_id, const QuadState& state, const Eigen::Vector3d& p_obs, double d_min) {
  RegisterResult out;
  const auto it = certified_gammas_.find(obstacle_id);
  if (it != certified_gammas_.end() && it->second.first == g1_ && it->second.second == g2_) {
    out.certified = true;  // already certified under the CURRENTLY active gamma
    return out;
  }
  // Either never certified, or gamma has changed since it last was --
  // re-evaluate under the CURRENT g1_/g2_ before trusting it as a hard,
  // zero-slack QP constraint again.
  const CertificationResult cert =
      isEntryCertified(state, p_obs, d_min, g1_, g2_, m_, cylinder_mode_);
  out.certified = cert.certified;
  out.chain = cert.chain;
  if (cert.certified) {
    certified_gammas_[obstacle_id] = {g1_, g2_};
  } else {
    certified_gammas_.erase(obstacle_id);
  }
  return out;
}

} 
