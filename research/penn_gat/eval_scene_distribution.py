"""Stage 1 of the barrier-label-collapse plan's scene-level reframe (2026-08-20): every prior
evaluation compared arms on ONE hand-built course, which measures the within-course …"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "nn_model"))

import drone_data_generation as G  # noqa: E402
from gat_3d import GATModule3D  # noqa: E402
from penn.nn_gat_iccbf_predict import ProbabilisticEnsembleGAT  # noqa: E402

EVAL_SEED = 20260821  # disjoint from training's np.random.seed(42) top-level
                       # call and from every worker's np.random.seed() reseed
DT = 0.05
BND = (-1.0, 8.0, -3.0, 3.0, 0.3, 3.0)
A_MAX, K_V, V_MAX, K_P = 5.0, 2.0, 3.0, 0.5
MAX_STEPS = 500
QUERY_TICKS = 10
# penn_update_ticks -- was 20 (1.0s); halved 2026-08-20 after diag_cost_decomposition.py --query-
# ticks 10 showed this alone cuts the margin-violation rate in half for a gate that targets min_h …

DEPLOY_GAMMA_MIN, DEPLOY_GAMMA_MAX, DEPLOY_STEPS = 0.5, 2.5, 20
# DEPLOY_GAMMA_MIN/MAX env overrides, added 2026-08-23 for the range-cap study.
DEPLOY_GAMMA_MIN = float(os.environ.get("DEPLOY_GAMMA_MIN", DEPLOY_GAMMA_MIN))
DEPLOY_GAMMA_MAX = float(os.environ.get("DEPLOY_GAMMA_MAX", DEPLOY_GAMMA_MAX))
# The fixed-gamma reference set is deliberately NOT tied to the deploy range: it is the baseline the
# adaptive policy is compared against, so widening it with the deploy range would silently change …
CONST_CANDIDATES = np.round(np.linspace(
    float(os.environ.get("CONST_GAMMA_MIN", 0.5)),
    float(os.environ.get("CONST_GAMMA_MAX", 2.5)), 21), 3)

# CHECKPOINT_NAME env override added 2026-08-23 for the range-widening retrain, so diag_lcb_cv.py
# and friends can be pointed at a candidate checkpoint (Quadrotor3D_gat_widerange) without …
_CKPT_NAME = os.environ.get("CHECKPOINT_NAME", "Quadrotor3D_gat")
DEFAULT_CKPT = os.path.join(os.path.dirname(__file__),
                            f"nn_model/checkpoint/{_CKPT_NAME}.pth")
DEFAULT_THRESH = os.path.join(os.path.dirname(__file__),
                              f"nn_model/checkpoint/cccp_threshold_{_CKPT_NAME}.json")


# ── Held-out scene set ────────────────────────────────────────────────────

def make_holdout_scenes(n, seed=EVAL_SEED):
    """Deterministic held-out scenes, disjoint from anything training saw (training scenes are
    drawn from unseeded per-worker RNGs, so there's no literal overlap risk -- this seed …"""
    rng_state = np.random.get_state()
    np.random.seed(seed)
    scenes = []
    while len(scenes) < n:
        if np.random.random() < G.GATE_SCENE_PROB:
            start, goal, obstacles = G.random_gate_scene()
        else:
            start, goal, obstacles = G.random_scene(n_obs_range=(1, 9), tight_prob=0.7)
        if obstacles:
            eff = [(c, r + G.SAFETY_MARGIN) for c, r in obstacles]
            scenes.append((start, goal, eff))
    np.random.set_state(rng_state)
    return scenes


# ── Fixed-gamma episode (shared by baseline + oracle) ───────────────────────

def _min_h_physical(pos, obstacles):
    """Ground-truth barrier value against the PHYSICAL obstacle surface, i.e. with SAFETY_MARGIN
    subtracted back out of each effective radius."""
    h = np.inf
    for c, r_eff in obstacles:
        r_phys = r_eff - G.SAFETY_MARGIN
        d2 = float(np.dot(pos - c, pos - c))
        h = min(h, d2 - r_phys ** 2)
    return h


def run_fixed(start, goal, obstacles, gamma):
    """Returns (min_h_margin, min_h_physical, t_goal)."""
    pos, vel = start.copy(), np.zeros(3)
    min_h, min_h_phys, t = np.inf, np.inf, None
    for step in range(MAX_STEPS):
        if float(np.linalg.norm(goal - pos)) < 0.3:
            t = step * DT
            break
        a_safe, h_step, _ = G.step_dynamics(
            pos, vel, goal, obstacles, gamma, BND, 0.5, V_MAX, K_P, K_V, A_MAX, True)
        min_h = min(min_h, h_step)
        if obstacles:
            min_h_phys = min(min_h_phys, _min_h_physical(pos, obstacles))
        vel = vel + a_safe * DT
        pos = pos + vel * DT
    if t is None:
        t = MAX_STEPS * DT
    return ((min_h if np.isfinite(min_h) else 0.0),
            (min_h_phys if np.isfinite(min_h_phys) else 0.0), t)


def cost(min_h, t):
    """Deployed intent: unsafe is disqualifying, otherwise minimize time."""
    return MAX_STEPS * DT * 2.0 if min_h < 0 else t


# ── Real adaptive policy (mirrors cbf_obstacle_node.py:_penn_select_alpha) ──

class AdaptivePolicy:
    def __init__(self, ckpt_path, thresh_path, device="cpu", gate_mode="network",
                 risk_margin=0.0, lcb_k=0.0, fallback_step=0.5,
                 fallback_mode="ratchet", fallback_seed_gamma=None):
        """gate_mode: "network" -- deployed rule: reject on network-predicted mean risk_horizon >
        cvar_boundary (what shipped)."""
        # "fully_oracle" additionally replaces the performance argmin with ground-truth
        # progress_deficit (from the same simulate_horizon call the min_h gate already makes -- …
        assert gate_mode in ("network", "oracle_min_h", "fully_oracle")
        self.gate_mode = gate_mode
        self.gm = GATModule3D(robot_radius=0.25, device=device)
        self.penn = ProbabilisticEnsembleGAT(
            self.gm.gat, n_output=2, n_hidden=40, n_ensemble=3,
            gamma_dim=1, device=device, lr=3e-4)
        self.penn.load_model(ckpt_path)
        self.penn.eval()
        self.device = device
        with open(thresh_path) as f:
            th = json.load(f)
        # cccp_calibrate_drone.py writes "raw_epistemic_threshold", not "epistemic_threshold" --
        # reading the wrong key here silently fell back to the stale default (1.097304, calibrated …
        self.ep_thr = float(th.get("raw_epistemic_threshold",
                                   th.get("epistemic_threshold", 1.097304)))
        # cvar_boundary's sign convention flipped 2026-08-20 along with the risk label swap
        # (risk_horizon -> min_h_horizon, see drone_data_generation.py's worker()): it is now a …
        self.risk_thr = float(th.get("cvar_boundary", -0.05))
        # risk_margin / lcb_k (2026-08-21): diag_minh_head.py found the risk head predicts min_h
        # well in absolute terms (corr +0.991, RMSE 0.173, near-zero bias) but that truly-unsafe …
        self.risk_margin = float(risk_margin)
        self.lcb_k = float(lcb_k)
        # Fallback rule (2026-08-29, PLAN_adaptive_hardening §0.4).
        assert fallback_mode in ("ratchet", "legacy_maxminh")
        self.fallback_mode = fallback_mode
        self.fallback_step = float(fallback_step)
        self.fallback_seed_gamma = (DEPLOY_GAMMA_MAX if fallback_seed_gamma is None
                                    else float(fallback_seed_gamma))
        self._last_gamma = None  # mirrors C++ gamma_obs_selected_; see reset()
        self.n_fallback = 0
        self.n_queries = 0

    def reset(self):
        """Clear the ratchet state -- call between independent episodes, the way the deployed
        node's resetT() re-seeds gamma_obs_selected_ on re-arm."""
        self._last_gamma = None

    def select(self, pos, vel, goal, obstacles_raw, prev_gamma=None):
        """obstacles_raw: list of (center, eff_radius) -- the TRUE/full scene obstacle set.
        Returns gamma."""
        obstacles_visible = G.visible_obstacles(pos, obstacles_raw)
        if not obstacles_visible:
            # cbf_alpha_obs default when nothing is currently sensed.
            self._last_gamma = 1.0
            return 1.0
        robot = [float(pos[0]), float(pos[1]), float(pos[2]),
                float(vel[0]), float(vel[1]), float(vel[2])]
        obs_list = [[float(c[0]), float(c[1]), float(c[2]), float(r), 0.0, 0.0, 0.0]
                   for c, r in obstacles_visible]
        goal_l = [float(goal[0]), float(goal[1]), float(goal[2])]
        graph = self.gm.create_graph(robot=robot, obstacles=obs_list, goal=goal_l)
        cands = torch.linspace(DEPLOY_GAMMA_MIN, DEPLOY_GAMMA_MAX, DEPLOY_STEPS).unsqueeze(1)
        safety_preds, perf_preds, divs = self.penn.predict_single_scene(graph, cands)

        oracle_min_h, oracle_perf = None, None
        if self.gate_mode in ("oracle_min_h", "fully_oracle"):
            oracle_min_h, oracle_perf = {}, {}
            for i in range(DEPLOY_STEPS):
                g = float(cands[i].item())
                _r, p, _fc, mh = G.simulate_horizon(
                    pos, vel, goal, obstacles_raw, g,
                    boundary=BND, horizon_s=G.HORIZON_S, apply_cbf=True)
                oracle_min_h[g] = mh
                oracle_perf[g] = p

        gated, all_preds = [], []
        for i in range(DEPLOY_STEPS):
            g = float(cands[i].item())
            div_i = divs[i]
            mean_risk = float(np.mean([m for m, _ in safety_preds[i]]))
            if self.lcb_k > 0.0:
                # Lower confidence bound on predicted min_h.
                mus = np.array([m for m, _ in safety_preds[i]], dtype=float)
                var_al = float(np.mean([v for _, v in safety_preds[i]]))
                var_ep = float(mus.var())
                mean_risk = mean_risk - self.lcb_k * float(np.sqrt(var_al + var_ep))
            mean_perf = (oracle_perf[g] if self.gate_mode == "fully_oracle"
                        else float(np.mean([m for m, _ in perf_preds[i]])))
            all_preds.append((g, mean_risk, mean_perf))
            if self.ep_thr > 0 and div_i > self.ep_thr:
                continue
            if self.gate_mode in ("oracle_min_h", "fully_oracle"):
                if oracle_min_h[g] < 0:
                    continue
            # mean_risk is now the network's predicted min_h_horizon (see the label-swap comment in
            # drone_data_generation.py's worker()) -- LOW/NEGATIVE is dangerous, so reject below …
            elif mean_risk < self.risk_thr + self.risk_margin:
                continue
            gated.append((g, mean_risk, mean_perf))

        self.n_queries += 1
        if not gated:
            self.n_fallback += 1
            if self.gate_mode in ("oracle_min_h", "fully_oracle"):
                chosen = max(oracle_min_h, key=lambda g: oracle_min_h[g])
            elif self.fallback_mode == "ratchet":
                # Deployed rule: degrade monotonically toward gamma_min
                # instead of consulting the just-rejected model ranking.
                prev = prev_gamma
                if prev is None:
                    prev = self._last_gamma
                if prev is None:
                    prev = self.fallback_seed_gamma
                chosen = max(DEPLOY_GAMMA_MIN, float(prev) - self.fallback_step)
            else:  # "legacy_maxminh" -- pre-2026-08-29 Python behaviour
                chosen = float(max(all_preds, key=lambda f: f[1])[0])
        else:
            chosen = min(gated, key=lambda f: f[2])[0]
        self._last_gamma = float(chosen)
        return float(chosen)


def run_adaptive(policy, start, goal, obstacles):
    """Returns (min_h_margin, min_h_physical, t_goal)."""
    pos, vel = start.copy(), np.zeros(3)
    min_h, min_h_phys, t = np.inf, np.inf, None
    policy.reset()  # ratchet state must not leak between episodes
    gamma = policy.select(pos, vel, goal, obstacles)
    tick = 0
    for step in range(MAX_STEPS):
        if float(np.linalg.norm(goal - pos)) < 0.3:
            t = step * DT
            break
        tick += 1
        if tick >= QUERY_TICKS:
            tick = 0
            gamma = policy.select(pos, vel, goal, obstacles, prev_gamma=gamma)
        a_safe, h_step, _ = G.step_dynamics(
            pos, vel, goal, obstacles, gamma, BND, 0.5, V_MAX, K_P, K_V, A_MAX, True)
        min_h = min(min_h, h_step)
        if obstacles:
            min_h_phys = min(min_h_phys, _min_h_physical(pos, obstacles))
        vel = vel + a_safe * DT
        pos = pos + vel * DT
    if t is None:
        t = MAX_STEPS * DT
    return ((min_h if np.isfinite(min_h) else 0.0),
            (min_h_phys if np.isfinite(min_h_phys) else 0.0), t)


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-scenes", type=int, default=50)
    ap.add_argument("--checkpoint", default=DEFAULT_CKPT)
    ap.add_argument("--thresholds", default=DEFAULT_THRESH)
    ap.add_argument("--out", default=None,
                    help="write a JSON summary here, e.g. results/scene_eval_<tag>.json")
    ap.add_argument("--gate-mode", default="network",
                    choices=("network", "oracle_min_h", "fully_oracle"),
                    help="'oracle_min_h'/'fully_oracle' are diagnostic ablations, not "
                        "deployable policies -- see AdaptivePolicy's docstring.")
    args = ap.parse_args()

    scenes = make_holdout_scenes(args.n_scenes)
    print(f"Held-out scenes: {len(scenes)}  (seed={EVAL_SEED}, disjoint from training)")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Thresholds: {args.thresholds}\n")

    t0 = time.time()

    # fixed-gamma sweep: baseline + oracle from the SAME data
    C = np.zeros((len(scenes), len(CONST_CANDIDATES)))
    for i, (s, g, o) in enumerate(scenes):
        for j, gam in enumerate(CONST_CANDIDATES):
            mh, _mh_phys, t = run_fixed(s, g, o, float(gam))
            C[i, j] = cost(mh, t)

    per_scene_oracle = C.min(axis=1)
    V_oracle = per_scene_oracle.mean()
    mean_per_const = C.mean(axis=0)
    j_best = int(mean_per_const.argmin())
    best_const_gamma = float(CONST_CANDIDATES[j_best])
    V_baseline = mean_per_const[j_best]

    print(f"[fixed-gamma sweep done in {time.time()-t0:.1f}s]")
    print(f"Best-fitted-constant gamma = {best_const_gamma:.2f}  "
          f"(mean cost {V_baseline:.3f}s)")
    print(f"Per-scene oracle mean cost = {V_oracle:.3f}s")
    ceiling = 100 * (V_baseline - V_oracle) / V_baseline if V_baseline > 1e-9 else 0.0
    print(f"Ceiling (oracle gap) = {ceiling:.1f}%\n")

    # real adaptive policy
    t1 = time.time()
    policy = AdaptivePolicy(args.checkpoint, args.thresholds, gate_mode=args.gate_mode)
    adaptive_costs = np.zeros(len(scenes))
    adaptive_min_h = np.zeros(len(scenes))
    adaptive_min_h_phys = np.zeros(len(scenes))
    for i, (s, g, o) in enumerate(scenes):
        mh, mh_phys, t = run_adaptive(policy, s, g, o)
        adaptive_costs[i] = cost(mh, t)
        adaptive_min_h[i] = mh
        adaptive_min_h_phys[i] = mh_phys

    V_adaptive = adaptive_costs.mean()
    n_unsafe_adaptive = int(np.sum(adaptive_min_h < 0))
    n_hard_collision_adaptive = int(np.sum(adaptive_min_h_phys < 0))
    const_runs = [run_fixed(s, g, o, best_const_gamma) for s, g, o in scenes]
    unsafe_at_best_const = int(np.sum([r[0] < 0 for r in const_runs]))
    hard_collision_at_best_const = int(np.sum([r[1] < 0 for r in const_runs]))

    denom = V_baseline - V_oracle
    capture = 100 * (V_baseline - V_adaptive) / denom if denom > 1e-9 else float("nan")
    win_vs_const = 100 * (V_baseline - V_adaptive) / V_baseline if V_baseline > 1e-9 else 0.0

    print(f"[adaptive policy eval done in {time.time()-t1:.1f}s]")
    print(f"PENN fallback fired on {policy.n_fallback}/{policy.n_queries} queries "
          f"({100*policy.n_fallback/max(policy.n_queries,1):.1f}%)\n")

    print("=" * 90)
    print(f"{'arm':<28} {'mean cost':>10} {'margin viol':>12} {'HARD collision':>15} "
          f"{'vs const':>10} {'captures':>10}")
    print("-" * 90)
    print(f"{'best fitted constant':<28} {V_baseline:>9.3f}s {unsafe_at_best_const:>9}/{len(scenes)} "
          f"{hard_collision_at_best_const:>12}/{len(scenes)} {'—':>10} {0.0:>9.1f}%")
    print(f"{'adaptive (current ckpt)':<28} {V_adaptive:>9.3f}s {n_unsafe_adaptive:>9}/{len(scenes)} "
          f"{n_hard_collision_adaptive:>12}/{len(scenes)} {win_vs_const:>9.1f}% {capture:>9.1f}%")
    print(f"{'per-scene oracle':<28} {V_oracle:>9.3f}s {'—':>11} {'—':>15} "
          f"{100*(V_baseline-V_oracle)/V_baseline:>9.1f}% {100.0:>9.1f}%")
    print("=" * 90)
    print("'margin viol' = crossed the SAFETY_MARGIN-inflated boundary (existing metric).")
    print("'HARD collision' = crossed the PHYSICAL obstacle surface -- the metric that")
    print("actually matters; a margin violation with zero hard collisions is a graze,")
    print("not a crash. Report both, always -- see Phase 1 of the redesign plan.")

    if unsafe_at_best_const > 0:
        print(f"\nNOTE: the best-fitted-constant baseline itself has a margin violation on "
              f"{unsafe_at_best_const}/{len(scenes)} held-out scenes -- it was chosen "
              f"purely to minimize mean cost, which can prefer a faster-but-occasionally-"
              f"unsafe gamma if violations are rare in this scene set. If this fires, "
              f"switch cost() to a hard safety gate before comparing capture fractions.")

    result = {
        "n_scenes": len(scenes),
        "eval_seed": EVAL_SEED,
        "checkpoint": args.checkpoint,
        "gate_mode": args.gate_mode,
        "best_const_gamma": best_const_gamma,
        "V_baseline": V_baseline,
        "V_oracle": V_oracle,
        "V_adaptive": V_adaptive,
        "ceiling_pct": ceiling,
        "capture_pct": capture,
        "win_vs_const_pct": win_vs_const,
        "unsafe_best_const": unsafe_at_best_const,
        "unsafe_adaptive": n_unsafe_adaptive,
        "hard_collision_best_const": hard_collision_at_best_const,
        "hard_collision_adaptive": n_hard_collision_adaptive,
        "penn_fallback_rate": policy.n_fallback / max(policy.n_queries, 1),
    }
    if args.out:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(result, f, indent=2)
        print(f"\nWrote {args.out}")

    return result


if __name__ == "__main__":
    main()
