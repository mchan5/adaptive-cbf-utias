#pragma once

#include <memory>
#include <optional>
#include <vector>

#include <Eigen/Dense>

namespace nodelib {

// Diagnostic-only (2026-08-23): prints the fraction of selectGamma() calls that fell through to the
// safety-only fallback (nothing passed both the epistemic and cvar gates) vs. the speed-aware …
void printPennSelectorStats();

class PennGammaSelector {
 public:

  explicit PennGammaSelector(const char* weights_path);
  ~PennGammaSelector();
  PennGammaSelector(PennGammaSelector&&) noexcept;
  PennGammaSelector& operator=(PennGammaSelector&&) noexcept;
  PennGammaSelector(const PennGammaSelector&) = delete;
  PennGammaSelector& operator=(const PennGammaSelector&) = delete;

  struct ObstacleInput {
    Eigen::Vector3d center;
    double radius;
  };

  struct GammaSelectionParams {
    Eigen::Vector3d goal;
    double gamma_min{0.5};
    double gamma_max_candidate{10.0};
    int gamma_steps{20};
    double epistemic_threshold{0.0};
    double cvar_boundary{0.0};
    double deadlock_threshold{0.0};
    // Lower-confidence-bound weight on the risk head's predicted min_h: mean_risk -= lcb_k *
    // sqrt(var_al + var_ep) before gating (var_al = mean ensemble-member variance, var_ep = …
    double lcb_k{0.0};
    // Continuity penalty added to the ranking score (2026-08-24, range-widening plan): score +=
    // continuity_weight * |gamma_i - prev_gamma|.
    double prev_gamma{0.0};
    double continuity_weight{0.0};
    // Fallback ratchet step (2026-08-26, PLAN_adaptive_recovery §1.1): when no candidate passes the
    // epistemic/cvar gates, the old fallback returned max(all_preds, key=risk) -- the model's own …
    double fallback_step{0.5};
  };

  std::optional<double> selectGamma(const Eigen::Vector3d& pos_enu, const Eigen::Vector3d& vel_enu,
                                     const std::vector<ObstacleInput>& obstacles,
                                     const GammaSelectionParams& params) const;

  // Diagnostic-only: mirrors selectGamma's sweep but returns the full per-candidate table instead
  // of just the first accepted gamma. Does not affect selectGamma's behavior.
  struct GammaCandidateDiagnostic {
    double gamma;
    double divergence;
    double mean_deadlock;
    double mean_risk;
    bool passed_epistemic;
    bool passed_deadlock_risk;
  };
  std::optional<std::vector<GammaCandidateDiagnostic>> diagnoseGammaSelection(
      const Eigen::Vector3d& pos_enu, const Eigen::Vector3d& vel_enu,
      const std::vector<ObstacleInput>& obstacles, const GammaSelectionParams& params) const;

  // Debug-only: dumps the raw (n_obs+1, 21) zij matrix -- the network's actual input before any
  // embedding/attention -- as a flat row-major vector, for diffing against gat_3d.py's …
  std::vector<float> debugZij(const Eigen::Vector3d& pos_enu, const Eigen::Vector3d& vel_enu,
                               const std::vector<ObstacleInput>& obstacles,
                               const Eigen::Vector3d& goal) const;

  // Debug-only: the pooled 16-dim robot embedding (psi1-psi3 output, before the ensemble MLP).
  std::vector<float> debugRobotEmb(const Eigen::Vector3d& pos_enu, const Eigen::Vector3d& vel_enu,
                                    const std::vector<ObstacleInput>& obstacles,
                                    const Eigen::Vector3d& goal) const;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  
