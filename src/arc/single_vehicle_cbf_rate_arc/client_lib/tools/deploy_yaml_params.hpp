// deploy_yaml_params.hpp
//
// Single source of truth for the offline harnesses: load the controller
// parameters from the SAME file the deployed node ships with
// (config/params_single_vehicle_cbf_rate_arc.yaml) instead of a hand-
// maintained snapshot in baseParams() that silently drifts.
//
// Motivation (PLAN_adaptive_hardening_20260829.md Phase 0.1): on 2026-08-29
// an audit found body_rate_closed_loop_diag.cpp's baseParams() had drifted
// from the deployed YAML on six parameters (penn_gamma_min/max,
// epistemic_threshold, cvar_boundary, lcb_k, nom_accel_max/kp/kd) plus
// cbf_gamma_obs. The default candidate grid [0.5, 2.5] was almost entirely
// outside the deployed checkpoint's training support {2, 3.5, ... 9.5} --
// the exact configuration that manufactured a false "regression" in
// PLAN_adaptive_recovery_20260826.md part 7 before it was caught by hand.
//
// This loader fails LOUDLY (exit 1) if the YAML cannot be read: a missing
// config is a bug in how the tool was invoked, never something to paper
// over with a stale default. Env-var A/B overrides (NOM_ACCEL_MAX,
// PENN_GAMMA_MIN, EPISTEMIC_THRESHOLD, ...) are still honoured -- the caller
// applies them AFTER calling loadDeployedParams().
#pragma once

#include <cstdio>
#include <cstdlib>
#include <exception>
#include <string>
#include <vector>

#include <yaml-cpp/yaml.h>

#include "single_vehicle_cbf_rate/rate_autopilot_core.hpp"

namespace diagtools {

// Overwrites every field of `p` that also appears in the deployed parameter
// file. Fields absent from the YAML keep their RateAutopilotCore::Params
// constructor default. Path resolution: $DEPLOY_YAML if set, else
// `default_rel_path` (relative to the tool's working directory -- the diag
// tools already resolve nn_model/checkpoint/... and results/... the same
// way, i.e. they are run from the package source directory).
inline void loadDeployedParams(nodelib::RateAutopilotCore::Params& p) {
  // Resolution order: $DEPLOY_YAML, then a few paths relative to the
  // working directory the offline tools are actually launched from (the
  // package source dir, and penn_gat_training/ where run_full_validation.sh
  // runs). First one that loads wins.
  std::vector<std::string> candidates;
  if (const char* env = std::getenv("DEPLOY_YAML")) candidates.emplace_back(env);
  candidates.emplace_back("config/params_single_vehicle_cbf_rate_arc.yaml");
  candidates.emplace_back(
      "../single_vehicle_cbf_rate_arc/config/params_single_vehicle_cbf_rate_arc.yaml");
  candidates.emplace_back(
      "single_vehicle_cbf_rate_arc/config/params_single_vehicle_cbf_rate_arc.yaml");

  YAML::Node root;
  std::string path;
  std::string last_err;
  for (const auto& c : candidates) {
    try {
      root = YAML::LoadFile(c);
      path = c;
      break;
    } catch (const std::exception& e) {
      last_err = e.what();
    }
  }
  if (path.empty()) {
    std::fprintf(stderr,
        "FATAL: could not load the deployed config (last error: %s).\n"
        "Tried: $DEPLOY_YAML, config/..., ../single_vehicle_cbf_rate_arc/config/..., "
        "single_vehicle_cbf_rate_arc/config/...\n"
        "This harness no longer carries a hardcoded parameter snapshot -- it reads the\n"
        "same YAML the node ships with, so an offline campaign measures what actually\n"
        "flies. Set DEPLOY_YAML to the absolute path of "
        "params_single_vehicle_cbf_rate_arc.yaml.\n",
        last_err.c_str());
    std::exit(1);
  }
  std::fprintf(stderr, "deploy_yaml_params: loaded %s\n", path.c_str());

  YAML::Node r = root["/**/autopilot_sv_cbf_rate_node"]["ros__parameters"];
  if (!r || !r.IsMap()) {
    std::fprintf(stderr,
        "FATAL: '%s' has no '/**/autopilot_sv_cbf_rate_node: ros__parameters:' map.\n",
        path.c_str());
    std::exit(1);
  }

  auto d = [&](const char* k, double& dst) { if (r[k]) dst = r[k].as<double>(); };
  auto i = [&](const char* k, int& dst) { if (r[k]) dst = r[k].as<int>(); };
  auto b = [&](const char* k, bool& dst) { if (r[k]) dst = r[k].as<bool>(); };

  d("vehicle_mass", p.vehicle_mass);
  d("obs_d_min", p.obs_d_min);
  d("obs_safety_margin", p.obs_safety_margin);
  d("control_rate_hz", p.control_rate_hz);
  d("cbf_gamma_obs", p.cbf_gamma_obs);

  d("nom_pos_kp", p.nom_pos_kp);
  d("nom_vel_kd", p.nom_vel_kd);
  d("nom_pos_ki", p.nom_pos_ki);
  d("nom_pos_i_max", p.nom_pos_i_max);
  d("nom_accel_max", p.nom_accel_max);
  d("nom_yaw_des", p.nom_yaw_des);
  d("nom_k_r", p.nom_k_r);
  d("nom_k_t", p.nom_k_t);
  d("nom_infeasible_brake_kv", p.nom_infeasible_brake_kv);
  d("nom_infeasible_brake_accel_max", p.nom_infeasible_brake_accel_max);
  i("infeasible_recovery_ticks", p.infeasible_recovery_ticks);

  d("stall_speed_threshold", p.stall_speed_threshold);
  d("stall_duration_sec", p.stall_duration_sec);
  d("escape_duration_sec", p.escape_duration_sec);
  d("escape_lateral_accel", p.escape_lateral_accel);
  d("escape_goal_pull_scale", p.escape_goal_pull_scale);
  d("goal_arrival_tolerance", p.goal_arrival_tolerance);

  d("qp_thrust_min", p.qp_thrust_min);
  d("qp_thrust_floor_gain", p.qp_thrust_floor_gain);
  d("qp_tau_min", p.qp_tau_min);
  d("qp_tau_max", p.qp_tau_max);
  d("qp_omega_max", p.qp_omega_max);
  d("qp_rho", p.qp_rho);
  d("qp_slack_weight", p.qp_slack_weight);

  b("uncertified_v_cap_enabled", p.uncertified_v_cap_enabled);
  d("uncertified_v_cap", p.uncertified_v_cap);

  b("penn_enabled", p.penn_enabled);
  d("penn_gamma_min", p.penn_gamma_min);
  d("penn_gamma_max", p.penn_gamma_max);
  i("penn_gamma_steps", p.penn_gamma_steps);
  i("penn_update_ticks", p.penn_update_ticks);
  d("penn_proximity_gate_m", p.penn_proximity_gate_m);
  d("epistemic_threshold", p.epistemic_threshold);
  d("cvar_boundary", p.cvar_boundary);
  // deadlock_threshold is deliberately NOT read here: it is vestigial (the
  // selector no longer gates on it, see penn_gamma_selector.cpp), and the
  // diag harness pins it to a sentinel 1e9 of its own. Reading the YAML's
  // 5.0 would be "more faithful" but changes an inert value for no benefit.
  d("lcb_k", p.lcb_k);
  d("continuity_weight", p.continuity_weight);
  d("penn_fallback_step", p.fallback_step);  // YAML key -> struct field name
  b("penn_feasibility_gate", p.penn_feasibility_gate);

  // T_max: the deployed node computes this from its actuator curve. Mirror
  // that from the same three values the node reads, so the QP thrust bound
  // matches deployment exactly.
  if (r["vehicle_thrust_scaling"] && r["vehicle_idle_thrust"] && r["vehicle_num_rotors"]) {
    const double s = r["vehicle_thrust_scaling"].as<double>();
    const double idle = r["vehicle_idle_thrust"].as<double>();
    const int n = r["vehicle_num_rotors"].as<int>();
    p.T_max = (1.0 - idle) / s * n;
  }
}

}  // namespace diagtools
