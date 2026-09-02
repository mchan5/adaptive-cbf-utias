#pragma once

#include <memory>
#include <optional>
#include <vector>

#include <Eigen/Dense>

namespace nodelib {

// Diagnostic-only (2026-08-23): prints the fraction of selectGamma() calls
// that fell through to the safety-only fallback (nothing passed both the
// epistemic and cvar gates) vs. the speed-aware gated path, IF
// PENN_SELECTOR_STATS is set in the environment. No-op otherwise, and does
// not affect selectGamma's behavior either way -- see penn_gamma_selector.cpp.
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
    // Lower-confidence-bound weight on the risk head's predicted min_h:
    // mean_risk -= lcb_k * sqrt(var_al + var_ep) before gating (var_al =
    // mean ensemble-member variance, var_ep = variance of ensemble means --
    // law of total variance). 0.0 = off (raw ensemble mean only, the
    // pre-2026-08-22 behavior). Mirrors AdaptivePolicy.select()'s lcb_k in
    // eval_scene_distribution.py; calibrated value lives in
    // diag_lcb_cv.py's output. Requires the aleatoric head to actually
    // carry per-scene signal to do anything useful -- see penn.py's
    // log_std_max clamp history.
    double lcb_k{0.0};
    // Continuity penalty added to the ranking score (2026-08-24, range-
    // widening plan): score += continuity_weight * |gamma_i - prev_gamma|.
    // Diagnosed root cause for why the adaptive policy underperformed a
    // constant fixed gamma at the widened range despite a comparable mean
    // gamma_used -- SELECTION_RULE=max_gamma raised mean gamma_used (4.9 ->
    // 6.4) but made time-to-goal WORSE, ruling out "wrong average setpoint"
    // and pointing at switching/variance cost instead: every
    // penn_update_ticks (0.1s) the selector can jump the full candidate
    // range with no cost for doing so. 0.0 = off (reproduces every prior
    // result bit-for-bit).
    double prev_gamma{0.0};
    double continuity_weight{0.0};
    // Fallback ratchet step (2026-08-26, PLAN_adaptive_recovery §1.1):
    // when no candidate passes the epistemic/cvar gates, the old fallback
    // returned max(all_preds, key=risk) -- the model's own least-distrusted
    // prediction, i.e. exactly the judgment the gate just rejected as
    // unreliable. That fallback fired on ~13% of ticks and was implicated
    // in 60% of collision episodes (gate_calibration_20260824/FINDINGS.md).
    // Replaced with a monotone, model-free ratchet toward safety:
    // max(gamma_min, prev_gamma - fallback_step), matching
    // online_adaptive_cbf.py's select_best_parameters fallback
    // (gamma0 = max(lower_bound, current_gamma0 - step_size)). 0.0 =
    // disabled (falls straight to gamma_min every time gated fails).
    double fallback_step{0.5};
  };

  std::optional<double> selectGamma(const Eigen::Vector3d& pos_enu, const Eigen::Vector3d& vel_enu,
                                     const std::vector<ObstacleInput>& obstacles,
                                     const GammaSelectionParams& params) const;

  // Diagnostic-only: mirrors selectGamma's sweep but returns the full
  // per-candidate table instead of just the first accepted gamma. Does not
  // affect selectGamma's behavior. Added to investigate why the epistemic
  // divergence gate has been rejecting every candidate in practice (see
  // params_single_vehicle_cbf_rate_arc.yaml's penn_enabled comment).
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

  // Debug-only: dumps the raw (n_obs+1, 21) zij matrix -- the network's
  // actual input before any embedding/attention -- as a flat row-major
  // vector, for diffing against gat_3d.py's create_graph output on the
  // same state. Added 2026-08-22 to isolate a residual parity discrepancy
  // (1/300 states in penn_gamma_parity_check) to feature construction vs.
  // the attention/pooling math.
  std::vector<float> debugZij(const Eigen::Vector3d& pos_enu, const Eigen::Vector3d& vel_enu,
                               const std::vector<ObstacleInput>& obstacles,
                               const Eigen::Vector3d& goal) const;

  // Debug-only: the pooled 16-dim robot embedding (psi1-psi3 output, before
  // the ensemble MLP). Isolates whether a parity discrepancy is in the
  // attention/pooling math specifically vs. downstream in the ensemble.
  std::vector<float> debugRobotEmb(const Eigen::Vector3d& pos_enu, const Eigen::Vector3d& vel_enu,
                                    const std::vector<ObstacleInput>& obstacles,
                                    const Eigen::Vector3d& goal) const;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  
