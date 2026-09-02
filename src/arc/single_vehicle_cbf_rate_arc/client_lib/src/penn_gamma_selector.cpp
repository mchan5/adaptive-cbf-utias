#include "single_vehicle_cbf_rate/penn_gamma_selector.hpp"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iterator>
#include <limits>
#include <stdexcept>
#include <string>

#include <torch/csrc/jit/serialization/pickle.h>
#include <torch/torch.h>

namespace nodelib {

// Diagnostic-only selection-path counters, added 2026-08-23 to investigate
// why the adaptive policy underperforms a constant gamma=3.0 despite a real
// oracle showing headroom above it. selectGamma() has two paths: GATED
// (passes epistemic+cvar, ranked by predicted progress_deficit -- i.e.
// speed-aware) and FALLBACK (nothing passed both gates, ranked by predicted
// min_h alone -- pure safety, ignores speed entirely). If FALLBACK fires
// often, that's a direct, mechanistic explanation for a slow-but-safe
// selector. Gated behind PENN_SELECTOR_STATS so it costs nothing when unset.
std::atomic<long> g_gated_count{0};
std::atomic<long> g_fallback_count{0};

void printPennSelectorStats() {
  if (std::getenv("PENN_SELECTOR_STATS") == nullptr) return;
  const long g = g_gated_count.load(), f = g_fallback_count.load();
  const long t = g + f;
  std::fprintf(stderr, "PENN selector path stats: gated=%ld (%.1f%%) fallback=%ld (%.1f%%) total=%ld\n",
               g, t ? 100.0 * g / t : 0.0, f, t ? 100.0 * f / t : 0.0, t);
}

namespace {

// SELECTION_RULE env switch, added 2026-08-23 (range-widening plan, Phase 1).
// The gates (epistemic + cvar) decide which candidates are SAFE; this only
// changes how the safe set is ranked. Diagnosed root cause: the deployed
// min_deficit rule chooses a mean gamma well below the oracle's usual pick
// (the top of the candidate range, on both the narrow and the newly-widened
// range) -- the network was trained on point-mass, where per-scene gamma
// diversity is real and rewards hedging, but body-rate's landscape is much
// flatter, so hedging mostly just costs speed. These alternatives test
// whether a different ranking recovers parity with the best fixed gamma
// without touching the network at all.
//   min_deficit (default): argmin predicted progress_deficit -- unchanged
//     behavior, reproduces every existing result bit-for-bit when unset.
//   max_gamma: ignore the performance prediction entirely, pick the highest
//     gated-safe gamma. This is an upper bound on what ranking alone can
//     achieve -- it can reach parity with the best fixed constant (that IS
//     what "always pick the top of the safe range" means) but not exceed
//     it, since it carries no scene-specific information.
//   blend: argmin (predicted progress_deficit - BLEND_BETA * gamma), a
//     tunable compromise between the two.
enum class SelectionRule { kMinDeficit, kMaxGamma, kBlend };

SelectionRule selectionRuleFromEnv() {
  static const SelectionRule rule = []() {
    const char* env = std::getenv("SELECTION_RULE");
    if (env == nullptr) return SelectionRule::kMinDeficit;
    const std::string s(env);
    if (s == "max_gamma") return SelectionRule::kMaxGamma;
    if (s == "blend") return SelectionRule::kBlend;
    return SelectionRule::kMinDeficit;
  }();
  return rule;
}

double blendBetaFromEnv() {
  static const double beta = []() {
    const char* env = std::getenv("BLEND_BETA");
    return env != nullptr ? std::atof(env) : 0.1;
  }();
  return beta;
}

torch::Tensor ensembleLinearForward(const torch::Tensor& input, const torch::Tensor& weights,
                                     const torch::Tensor& biases) {
  torch::Tensor input3d =
      input.dim() == 2 ? input.unsqueeze(0).repeat({weights.size(0), 1, 1}) : input;
  return torch::baddbmm(biases, input3d, weights);
}


torch::Tensor jensenRenyiDivergence(const torch::Tensor& mu, const torch::Tensor& var) {
  const int64_t n_act = mu.size(0);
  const int64_t es = mu.size(1);
  const int64_t d_s = mu.size(2);

  const torch::Tensor mu_diff = mu.unsqueeze(1) - mu.unsqueeze(2);    // (N,E,E,D)
  const torch::Tensor var_sum = var.unsqueeze(1) + var.unsqueeze(2);  // (N,E,E,D)

  const torch::Tensor err = (mu_diff * mu_diff / var_sum).sum(-1);  // (N,E,E)
  const torch::Tensor det = torch::log(var_sum).sum(-1);            // (N,E,E)

  torch::Tensor log_z = (-0.5 * (err + det)).reshape({n_act, es * es});
  const torch::Tensor mx = std::get<0>(log_z.max(1, /*keepdim=*/true));  // (N,1)
  const torch::Tensor exp_mean = torch::exp(log_z - mx).mean(1, /*keepdim=*/true);  // (N,1)
  const torch::Tensor entropy_mean = (-mx - torch::log(exp_mean)).squeeze(1);       // (N,)

  const torch::Tensor total_entropy = torch::log(var).sum(-1);  // (N,E)
  const torch::Tensor mean_entropy =
      total_entropy.mean(1) / 2.0 + d_s * std::log(2.0) / 2.0;  // (N,)

  return (entropy_mean - mean_entropy).abs();  // (N,)
}

std::vector<char> readFile(const std::string& path) {
  std::ifstream file(path, std::ios::binary);
  if (!file) {
    throw std::runtime_error("PennGammaSelector: cannot open weights file " + path);
  }
  return std::vector<char>((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
}

// Sensing envelope -- mirrors drone_data_generation.py's visible_obstacles()
// / lidar_obstacle_publisher_node.py's max_range_m/z_min_m/z_max_m defaults.
// Added 2026-08-22: AdaptivePolicy.select() (Python) defensively re-filters
// its obstacle input by this same envelope before building a graph, even
// though real deployment already expects /arc/obstacles to be pre-filtered
// upstream by whichever obstacle publisher is running. selectGamma() had no
// equivalent, so it and the Python reference could disagree whenever a
// caller (or a test harness) handed either implementation an obstacle
// outside the sensing envelope -- found via a parity-check mismatch that
// turned out not to be a code bug in either implementation, but this
// missing defensive filter plus a bug in the test harness that saved an
// unfiltered obstacle list. Kept here for defense-in-depth/parity even
// though the real obstacle source should already filter.
constexpr double kSensorMaxRange = 8.0;
constexpr double kSensorZMin = 0.1;
constexpr double kSensorZMax = 2.5;

std::vector<PennGammaSelector::ObstacleInput> filterVisible(
    const Eigen::Vector3d& pos_enu, const std::vector<PennGammaSelector::ObstacleInput>& obstacles) {
  std::vector<PennGammaSelector::ObstacleInput> out;
  out.reserve(obstacles.size());
  for (const auto& o : obstacles) {
    if (o.center.z() < kSensorZMin || o.center.z() > kSensorZMax) continue;
    const double dist_to_surface = (pos_enu - o.center).norm() - o.radius;
    if (dist_to_surface <= kSensorMaxRange) out.push_back(o);
  }
  return out;
}

}
struct PennGammaSelector::Impl {
  double robot_radius{0.25};
  double gamma_max{15.0};  // normalisation constant baked into the checkpoint
  double log_std_min{-20.0};
  double log_std_max{-2.0};
  int64_t ensemble_size{3};
  int64_t n_output{2};

  torch::Tensor psi1_w0, psi1_b0, psi1_w2, psi1_b2;
  torch::Tensor psi2_w0, psi2_b0, psi2_w2, psi2_b2;
  torch::Tensor psi3_w0, psi3_b0, psi3_w2, psi3_b2;
  torch::Tensor lin1_w, lin1_b;
  torch::Tensor lin2_w, lin2_b;
  torch::Tensor lin3_w, lin3_b;
  torch::Tensor lin4_w, lin4_b;
  torch::Tensor lin5_w, lin5_b;

  explicit Impl(const std::string& weights_path) {
    const std::vector<char> data = readFile(weights_path);
    const torch::IValue loaded = torch::jit::pickle_load(data);
    const auto dict = loaded.toGenericDict();

    const auto get = [&dict](const std::string& key) -> torch::Tensor {
      return dict.at(key).toTensor();
    };

    psi1_w0 = get("psi1.0.weight");
    psi1_b0 = get("psi1.0.bias");
    psi1_w2 = get("psi1.2.weight");
    psi1_b2 = get("psi1.2.bias");
    psi2_w0 = get("psi2.0.weight");
    psi2_b0 = get("psi2.0.bias");
    psi2_w2 = get("psi2.2.weight");
    psi2_b2 = get("psi2.2.bias");
    psi3_w0 = get("psi3.0.weight");
    psi3_b0 = get("psi3.0.bias");
    psi3_w2 = get("psi3.2.weight");
    psi3_b2 = get("psi3.2.bias");

    lin1_w = get("lin1.weights");
    lin1_b = get("lin1.biases");
    lin2_w = get("lin2.weights");
    lin2_b = get("lin2.biases");
    lin3_w = get("lin3.weights");
    lin3_b = get("lin3.biases");
    lin4_w = get("lin4.weights");
    lin4_b = get("lin4.biases");
    lin5_w = get("lin5.weights");
    lin5_b = get("lin5.biases");

    gamma_max = get("gamma_max").item<double>();
    log_std_min = get("log_std_min").item<double>();
    log_std_max = get("log_std_max").item<double>();
    ensemble_size = get("ensemble_size").item<int64_t>();
    n_output = get("n_output").item<int64_t>();
    robot_radius = get("robot_radius").item<double>();
  }

  // robot_vel_norm: matches ProbabilisticEnsembleGAT.v_max_norm in
  // nn_gat_iccbf_predict.py -- a hardcoded normalisation constant, not
  // something persisted in the checkpoint, so it's hardcoded here too.
  static constexpr double kRobotVelNorm = 3.0;

  torch::Tensor buildRobotEdgeFeatures(const Eigen::Vector3d& pos_enu, const Eigen::Vector3d& vel_enu_in,
                                        const std::vector<ObstacleInput>& obstacles,
                                        const Eigen::Vector3d& goal,
                                        Eigen::Vector3d* rotated_vel_out = nullptr) const {
    const int n_obs = static_cast<int>(obstacles.size());
    const int num_edges = n_obs + 1;  // each obstacle, plus the goal

    // Frame canonicalization (ported from gat_3d.py's _canonicalize_frame,
    // 2026-08-22): rotate about z so (goal - pos) points along +x, before
    // deriving any feature. Every edge attribute below is a raw [dx,dy,dz]
    // difference plus a raw [vx,vy,vz] velocity, and NEITHER is rotation-
    // invariant on its own even though both are "relative" quantities --
    // only the derived scalar distances (already rotation-invariant) were
    // safe as-is. This C++ port never had the fix that gat_3d.py got: a
    // scene rotated 90 degrees is physically identical but was measured to
    // flip the selected gamma on 59% of scenes. Rotating each edge's
    // difference vector directly is mathematically identical to rotating
    // the absolute positions first and then differencing (rotation is
    // linear: R(a)-R(b) = R(a-b)), so this reproduces gat_3d.py's result
    // without restructuring this function around absolute coordinates.
    const double ryaw = goal.y() - pos_enu.y();
    const double rxaw = goal.x() - pos_enu.x();
    const double yaw = std::atan2(ryaw, rxaw);
    const double cy = std::cos(-yaw);
    const double sy = std::sin(-yaw);
    const auto rot_xy = [cy, sy](double vx, double vy) {
      return std::make_pair(cy * vx - sy * vy, sy * vx + cy * vy);
    };

    Eigen::Vector3d vel_enu = vel_enu_in;
    {
      const auto [rvx, rvy] = rot_xy(vel_enu_in.x(), vel_enu_in.y());
      vel_enu.x() = rvx;
      vel_enu.y() = rvy;
    }
    if (rotated_vel_out != nullptr) {
      *rotated_vel_out = vel_enu;
    }

    std::vector<double> dist_robot_obs(n_obs);
    for (int j = 0; j < n_obs; ++j) {
      const double raw = (pos_enu - obstacles[j].center).norm();
      dist_robot_obs[j] = std::max(0.0, raw - (robot_radius + obstacles[j].radius));
    }
    const double robot_prox =
        n_obs > 0 ? *std::min_element(dist_robot_obs.begin(), dist_robot_obs.end()) : 0.0;

    std::vector<double> dist_goal_obs(n_obs);
    for (int j = 0; j < n_obs; ++j) {
      const double raw = (goal - obstacles[j].center).norm();
      dist_goal_obs[j] = std::max(0.0, raw - obstacles[j].radius);
    }
    const double goal_prox =
        n_obs > 0 ? *std::min_element(dist_goal_obs.begin(), dist_goal_obs.end()) : 0.0;

    const double dist_robot_goal = std::max(0.0, (pos_enu - goal).norm() - robot_radius);

    std::vector<float> flat(static_cast<size_t>(num_edges) * 21, 0.0f);
    auto row = [&](int idx) { return flat.data() + static_cast<size_t>(idx) * 21; };

    for (int j = 0; j < n_obs; ++j) {
      float* r = row(j);
      // x_robot (7): one-hot(3), prox(1), vel(3)
      r[0] = 1.0f;
      r[3] = static_cast<float>(robot_prox);
      r[4] = static_cast<float>(vel_enu.x());
      r[5] = static_cast<float>(vel_enu.y());
      r[6] = static_cast<float>(vel_enu.z());
      // x_dst == obstacle_j (7): one-hot(3), prox(1), vel(3, hardcoded 0)
      r[8] = 1.0f;
      r[10] = static_cast<float>(dist_robot_obs[j]);
      // edge_attr (7): [dx,dy,dz,dist,dvx,dvy,dvz] -- ported from
      // gat_3d.py's create_graph as [pos[dst]-pos[src], dist, vel[dst]-
      // vel[src]] for edge (src=robot, dst=this obstacle), i.e.
      // obstacle_pos - robot_pos and obstacle_vel(=0) - robot_vel. Fixed
      // 2026-08-22: this used to compute pos_enu - obstacles[j].center
      // (src-dst, backwards) and +vel_enu (should be -vel_enu since
      // obstacle_vel is 0) -- both differential terms were sign-flipped
      // relative to the Python reference, silently, since this port was
      // first written (independent of the frame-canonicalization and
      // missing-velocity-feature bugs fixed the same day).
      Eigen::Vector3d d = obstacles[j].center - pos_enu;
      {
        const auto [rdx, rdy] = rot_xy(d.x(), d.y());
        d.x() = rdx;
        d.y() = rdy;
      }
      r[14] = static_cast<float>(d.x());
      r[15] = static_cast<float>(d.y());
      r[16] = static_cast<float>(d.z());
      r[17] = static_cast<float>(dist_robot_obs[j]);
      r[18] = static_cast<float>(-vel_enu.x());
      r[19] = static_cast<float>(-vel_enu.y());
      r[20] = static_cast<float>(-vel_enu.z());
    }

    {
      float* r = row(n_obs);  // goal row
      r[0] = 1.0f;
      r[3] = static_cast<float>(robot_prox);
      r[4] = static_cast<float>(vel_enu.x());
      r[5] = static_cast<float>(vel_enu.y());
      r[6] = static_cast<float>(vel_enu.z());
      r[9] = 1.0f;  // goal one-hot
      r[10] = static_cast<float>(goal_prox);
      // Same dst-src / dst_vel-src_vel convention as the obstacle rows
      // above (goal_vel is always 0 in create_graph, same as obstacle_vel).
      Eigen::Vector3d d = goal - pos_enu;
      {
        const auto [rdx, rdy] = rot_xy(d.x(), d.y());
        d.x() = rdx;
        d.y() = rdy;
      }
      r[14] = static_cast<float>(d.x());
      r[15] = static_cast<float>(d.y());
      r[16] = static_cast<float>(d.z());
      r[17] = static_cast<float>(dist_robot_goal);
      r[18] = static_cast<float>(-vel_enu.x());
      r[19] = static_cast<float>(-vel_enu.y());
      r[20] = static_cast<float>(-vel_enu.z());
    }

    return torch::from_blob(flat.data(), {num_edges, 21}, torch::kFloat32).clone();
  }

  // (M,21) -> (1,16). Ported from GATGraphNetwork3D::extract_robot_embedding.
  torch::Tensor extractRobotEmbedding(const torch::Tensor& zij) const {
    const torch::Tensor h = torch::relu(torch::linear(zij, psi1_w0, psi1_b0));
    const torch::Tensor q_ij = torch::linear(h, psi1_w2, psi1_b2);  // (M,16)

    const torch::Tensor s = torch::relu(torch::linear(q_ij, psi2_w0, psi2_b0));
    const torch::Tensor scores = torch::linear(s, psi2_w2, psi2_b2).squeeze(-1);  // (M,)
    const torch::Tensor attn = torch::softmax(scores, /*dim=*/0);  // single group: all edges share src==0

    const torch::Tensor t = torch::relu(torch::linear(q_ij, psi3_w0, psi3_b0));
    const torch::Tensor msg = torch::linear(t, psi3_w2, psi3_b2);  // (M,16)

    const torch::Tensor node_emb = (attn.unsqueeze(-1) * msg).sum(0);  // (16,)
    return node_emb.unsqueeze(0);                                     // (1,16)
  }

  struct EnsembleOutputs {
    torch::Tensor mus;       // (ensemble_size, N, n_output)
    torch::Tensor log_stds;  // (ensemble_size, N, n_output)
    torch::Tensor div;       // (N,)
  };

  // X_input (N, 16+gamma_dim) -> ensemble mu/log_std/divergence. Ported from
  // EnsembleStochasticLinear::forward + JensenRenyiDivergence.
  EnsembleOutputs ensembleForward(const torch::Tensor& X_input) const {
    torch::Tensor x = torch::relu(ensembleLinearForward(X_input, lin1_w, lin1_b));
    x = torch::relu(ensembleLinearForward(x, lin2_w, lin2_b));
    x = torch::relu(ensembleLinearForward(x, lin3_w, lin3_b));
    x = torch::relu(ensembleLinearForward(x, lin4_w, lin4_b));
    x = ensembleLinearForward(x, lin5_w, lin5_b);  // (E,N,2*n_output)

    const torch::Tensor mu = x.slice(2, 0, n_output);  // (E,N,2)
    const torch::Tensor log_std =
        x.slice(2, n_output, 2 * n_output).clamp(log_std_min, log_std_max);  // (E,N,2)
    const torch::Tensor std_dev = torch::exp(log_std);

    const torch::Tensor mu_perm = mu.permute({1, 0, 2});               // (N,E,2)
    const torch::Tensor var_perm = std_dev.pow(2).permute({1, 0, 2});  // (N,E,2)

    EnsembleOutputs out;
    out.mus = mu;
    out.log_stds = log_std;
    out.div = jensenRenyiDivergence(mu_perm, var_perm);  // (N,)
    return out;
  }

  std::optional<double> selectGamma(const Eigen::Vector3d& pos_enu, const Eigen::Vector3d& vel_enu,
                                     const std::vector<ObstacleInput>& obstacles_raw,
                                     const GammaSelectionParams& params) const {
    // Defensive re-filter to the sensing envelope, matching
    // AdaptivePolicy.select() -- see filterVisible's comment. In real
    // deployment /arc/obstacles should already be filtered upstream, but
    // this keeps the two implementations agreeing even if a caller (or a
    // test) hands in something that isn't.
    const std::vector<ObstacleInput> obstacles = filterVisible(pos_enu, obstacles_raw);
    if (obstacles.empty()) {
      return std::nullopt;
    }

    try {
      torch::NoGradGuard no_grad;

      Eigen::Vector3d rotated_vel;
      const torch::Tensor zij =
          buildRobotEdgeFeatures(pos_enu, vel_enu, obstacles, params.goal, &rotated_vel);
      const torch::Tensor robot_emb = extractRobotEmbedding(zij);  // (1,16)

      // Direct robot-velocity bypass feature (ported 2026-08-22 -- this was
      // missing entirely, which is the actual root cause of PENN inference
      // throwing on every call and silently falling back to a fixed gamma,
      // previously misattributed to the epistemic gate rejecting everything
      // in the 2026-08-10 investigation this diagnostic tool was built for.
      // See nn_gat_iccbf_predict.py's _robot_vel_feature / gat_3d.py's
      // extract_robot_velocity docstring: added 2026-08-19, after this C++
      // port was written, to give the performance head a direct path to
      // velocity that the pooled graph embedding wasn't preserving.
      // Normalised by the SAME hardcoded v_max_norm=3.0 Python uses, and
      // built from the frame-canonicalized velocity so it's consistent
      // with every other feature in this function.
      const torch::Tensor robot_vel = torch::tensor(
          {static_cast<float>(rotated_vel.x() / kRobotVelNorm),
           static_cast<float>(rotated_vel.y() / kRobotVelNorm),
           static_cast<float>(rotated_vel.z() / kRobotVelNorm)},
          torch::kFloat32).unsqueeze(0);  // (1,3)

      const int steps = params.gamma_steps;
      const torch::Tensor candidates =
          torch::linspace(params.gamma_min, params.gamma_max_candidate, steps).unsqueeze(1);  // (steps,1)
      const torch::Tensor robot_emb_rep = robot_emb.expand({steps, robot_emb.size(1)});
      const torch::Tensor robot_vel_rep = robot_vel.expand({steps, robot_vel.size(1)});
      const torch::Tensor gamma_norm = candidates / gamma_max;
      const torch::Tensor X_input =
          torch::cat({robot_emb_rep, robot_vel_rep, gamma_norm}, 1);  // (steps,20)

      const EnsembleOutputs out = ensembleForward(X_input);
      const torch::Tensor mean_over_ensemble = out.mus.mean(0);  // (steps,2): col0=deadlock, col1=risk

      // LCB inputs for the risk channel (index 1): var_al = mean ensemble-
      // member variance (aleatoric), var_ep = variance of ensemble means
      // (epistemic disagreement about the mean) -- law of total variance,
      // mirrors AdaptivePolicy.select()'s var_al/var_ep exactly
      // (eval_scene_distribution.py:271-273).
      const torch::Tensor mu_ch1 = out.mus.select(2, 1);            // (E, steps)
      const torch::Tensor log_std_ch1 = out.log_stds.select(2, 1);  // (E, steps)
      const torch::Tensor var_al_t = log_std_ch1.exp().pow(2).mean(0);        // (steps,)
      const torch::Tensor var_ep_t = mu_ch1.var(0, /*unbiased=*/false);       // (steps,)

      const auto candidates_acc = candidates.accessor<float, 2>();
      const auto div_acc = out.div.accessor<float, 1>();
      const auto mean_acc = mean_over_ensemble.accessor<float, 2>();
      const auto var_al_acc = var_al_t.accessor<float, 1>();
      const auto var_ep_acc = var_ep_t.accessor<float, 1>();

      // Gate every candidate (epistemic, then risk), then take the argmin
      // predicted deadlock/performance among those that pass -- matches
      // AdaptivePolicy.select() in eval_scene_distribution.py, NOT the old
      // "largest gamma down, first feasible" rule this used to implement
      // (2026-08-22 fix: that rule was never what got validated).
      //
      // Sign convention on cvar_boundary ALSO flipped 2026-08-22: mean_risk
      // is the risk head's prediction, and the head's training TARGET
      // changed 2026-08-21 from risk_horizon (an accumulated-violation-time
      // integral, >=0, low=safe -- "mean_risk < cvar_boundary" was the
      // correct accept condition under that label) to min_h_horizon (the
      // worst instantaneous barrier value over the horizon -- unbounded,
      // LOW/NEGATIVE=dangerous). Reusing the old "<" comparison against a
      // min_h-trained head would accept the candidates the model is most
      // confident are UNSAFE. See drone_data_generation.py's worker() and
      // eval_scene_distribution.py's AdaptivePolicy.select() for the
      // matching Python-side change.
      //
      // deadlock_threshold is no longer a separate gate: the validated
      // harness never gates on it, only ranks by it among already-safe
      // candidates, so keeping it as a second hard gate here was an
      // undocumented extra constraint the offline validation never tested.
      const SelectionRule rule = selectionRuleFromEnv();
      const double blend_beta = blendBetaFromEnv();

      int best_gated_idx = -1;
      double best_gated_score = 0.0;

      for (int i = 0; i < steps; ++i) {
        const double div_i = div_acc[i];
        const double mean_deadlock = mean_acc[i][0];
        const double gamma_i = candidates_acc[i][0];
        double mean_risk = mean_acc[i][1];
        if (params.lcb_k > 0.0) {
          mean_risk -= params.lcb_k * std::sqrt(static_cast<double>(var_al_acc[i]) +
                                                 static_cast<double>(var_ep_acc[i]));
        }

        if (params.epistemic_threshold > 0 && div_i > params.epistemic_threshold) {
          continue;
        }
        if (mean_risk < params.cvar_boundary) {
          continue;
        }
        // Ranking score among gated (safe) candidates -- lower wins in all
        // three rules, so max_gamma is expressed as negated gamma.
        double score;
        switch (rule) {
          case SelectionRule::kMaxGamma:
            score = -gamma_i;
            break;
          case SelectionRule::kBlend:
            score = mean_deadlock - blend_beta * gamma_i;
            break;
          case SelectionRule::kMinDeficit:
          default:
            score = mean_deadlock;
            break;
        }
        if (params.continuity_weight > 0.0) {
          score += params.continuity_weight * std::abs(gamma_i - params.prev_gamma);
        }
        if (best_gated_idx < 0 || score < best_gated_score) {
          best_gated_idx = i;
          best_gated_score = score;
        }
      }

      if (best_gated_idx >= 0) {
        g_gated_count.fetch_add(1, std::memory_order_relaxed);
        return candidates_acc[best_gated_idx][0];
      }
      g_fallback_count.fetch_add(1, std::memory_order_relaxed);
      // Conservative ratchet (2026-08-26, PLAN_adaptive_recovery §1.1):
      // no candidate was trustworthy, so degrade monotonically toward
      // gamma_min instead of consulting the model's own (just-rejected)
      // risk ranking. Matches online_adaptive_cbf.py's fallback rule.
      const double ratcheted = params.prev_gamma - params.fallback_step;
      return std::max(params.gamma_min, ratcheted);
    } catch (const std::exception& e) {
      std::fprintf(stderr, "PennGammaSelector::selectGamma exception: %s\n", e.what());
      return std::nullopt;  // inference failure -- caller falls back to a fixed gamma
    }
  }

  std::optional<std::vector<PennGammaSelector::GammaCandidateDiagnostic>> diagnoseGammaSelection(
      const Eigen::Vector3d& pos_enu, const Eigen::Vector3d& vel_enu,
      const std::vector<ObstacleInput>& obstacles_raw, const GammaSelectionParams& params) const {
    const std::vector<ObstacleInput> obstacles = filterVisible(pos_enu, obstacles_raw);
    if (obstacles.empty()) {
      return std::nullopt;
    }

    try {
      torch::NoGradGuard no_grad;

      Eigen::Vector3d rotated_vel;
      const torch::Tensor zij =
          buildRobotEdgeFeatures(pos_enu, vel_enu, obstacles, params.goal, &rotated_vel);
      const torch::Tensor robot_emb = extractRobotEmbedding(zij);
      const torch::Tensor robot_vel = torch::tensor(
          {static_cast<float>(rotated_vel.x() / kRobotVelNorm),
           static_cast<float>(rotated_vel.y() / kRobotVelNorm),
           static_cast<float>(rotated_vel.z() / kRobotVelNorm)},
          torch::kFloat32).unsqueeze(0);  // (1,3) -- see selectGamma's comment

      const int steps = params.gamma_steps;
      const torch::Tensor candidates =
          torch::linspace(params.gamma_min, params.gamma_max_candidate, steps).unsqueeze(1);
      const torch::Tensor robot_emb_rep = robot_emb.expand({steps, robot_emb.size(1)});
      const torch::Tensor robot_vel_rep = robot_vel.expand({steps, robot_vel.size(1)});
      const torch::Tensor gamma_norm = candidates / gamma_max;
      const torch::Tensor X_input = torch::cat({robot_emb_rep, robot_vel_rep, gamma_norm}, 1);

      const EnsembleOutputs out = ensembleForward(X_input);
      const torch::Tensor mean_over_ensemble = out.mus.mean(0);

      const torch::Tensor mu_ch1 = out.mus.select(2, 1);
      const torch::Tensor log_std_ch1 = out.log_stds.select(2, 1);
      const torch::Tensor var_al_t = log_std_ch1.exp().pow(2).mean(0);
      const torch::Tensor var_ep_t = mu_ch1.var(0, /*unbiased=*/false);

      const auto candidates_acc = candidates.accessor<float, 2>();
      const auto div_acc = out.div.accessor<float, 1>();
      const auto mean_acc = mean_over_ensemble.accessor<float, 2>();
      const auto var_al_acc = var_al_t.accessor<float, 1>();
      const auto var_ep_acc = var_ep_t.accessor<float, 1>();

      std::vector<PennGammaSelector::GammaCandidateDiagnostic> table;
      table.reserve(steps);
      for (int i = 0; i < steps; ++i) {
        PennGammaSelector::GammaCandidateDiagnostic row;
        row.gamma = candidates_acc[i][0];
        row.divergence = div_acc[i];
        row.mean_deadlock = mean_acc[i][0];
        row.mean_risk = mean_acc[i][1];
        if (params.lcb_k > 0.0) {
          row.mean_risk -= params.lcb_k * std::sqrt(static_cast<double>(var_al_acc[i]) +
                                                     static_cast<double>(var_ep_acc[i]));
        }
        row.passed_epistemic =
            !(params.epistemic_threshold > 0 && row.divergence > params.epistemic_threshold);
        // Matches selectGamma's corrected gate (2026-08-22): reject BELOW
        // cvar_boundary now (min_h semantics), and deadlock_threshold is no
        // longer a hard gate -- see selectGamma's comment for why. Field
        // name kept as passed_deadlock_risk for minimal API churn; it now
        // reflects the risk gate alone.
        row.passed_deadlock_risk = row.mean_risk >= params.cvar_boundary;
        table.push_back(row);
      }
      return table;
    } catch (const std::exception&) {
      return std::nullopt;
    }
  }
};

PennGammaSelector::PennGammaSelector(const char* weights_path)
    : impl_(std::make_unique<Impl>(std::string(weights_path))) {}

PennGammaSelector::~PennGammaSelector() = default;
PennGammaSelector::PennGammaSelector(PennGammaSelector&&) noexcept = default;
PennGammaSelector& PennGammaSelector::operator=(PennGammaSelector&&) noexcept = default;

std::optional<double> PennGammaSelector::selectGamma(const Eigen::Vector3d& pos_enu,
                                                      const Eigen::Vector3d& vel_enu,
                                                      const std::vector<ObstacleInput>& obstacles,
                                                      const GammaSelectionParams& params) const {
  return impl_->selectGamma(pos_enu, vel_enu, obstacles, params);
}

std::optional<std::vector<PennGammaSelector::GammaCandidateDiagnostic>>
PennGammaSelector::diagnoseGammaSelection(const Eigen::Vector3d& pos_enu, const Eigen::Vector3d& vel_enu,
                                           const std::vector<ObstacleInput>& obstacles,
                                           const GammaSelectionParams& params) const {
  return impl_->diagnoseGammaSelection(pos_enu, vel_enu, obstacles, params);
}

std::vector<float> PennGammaSelector::debugZij(const Eigen::Vector3d& pos_enu,
                                                const Eigen::Vector3d& vel_enu,
                                                const std::vector<ObstacleInput>& obstacles,
                                                const Eigen::Vector3d& goal) const {
  const torch::Tensor zij = impl_->buildRobotEdgeFeatures(pos_enu, vel_enu, obstacles, goal);
  const torch::Tensor flat = zij.contiguous().view(-1);
  std::vector<float> out(flat.numel());
  std::memcpy(out.data(), flat.data_ptr<float>(), flat.numel() * sizeof(float));
  return out;
}

std::vector<float> PennGammaSelector::debugRobotEmb(const Eigen::Vector3d& pos_enu,
                                                      const Eigen::Vector3d& vel_enu,
                                                      const std::vector<ObstacleInput>& obstacles,
                                                      const Eigen::Vector3d& goal) const {
  const torch::Tensor zij = impl_->buildRobotEdgeFeatures(pos_enu, vel_enu, obstacles, goal);
  const torch::Tensor emb = impl_->extractRobotEmbedding(zij).contiguous().view(-1);
  std::vector<float> out(emb.numel());
  std::memcpy(out.data(), emb.data_ptr<float>(), emb.numel() * sizeof(float));
  return out;
}

}
