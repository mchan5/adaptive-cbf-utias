"""
diag_horizon_blindspot.py

Step-1 follow-up diagnostic from the scene-level reframe's Stage-3 failure
(2026-08-20): the post-fix adaptive policy is unsafe on 18/50 held-out
scenes despite the risk gate (cvar_boundary) supposedly filtering out any
gamma predicted unsafe. Hypothesis: the gate bounds risk within one 4s
receding-horizon window at query time, but the deployed node re-queries
every ~1s -- a state that looks safe over the next 4s can still be one from
which, one second later (closer to the obstacle, possibly faster), no
gamma keeps the barrier satisfied. This checks that hypothesis directly by
comparing, at the query immediately before each violation:
  (a) the MODEL's predicted risk_horizon at the selected gamma  (calibration)
  (b) GROUND-TRUTH risk_horizon at the same state/gamma, horizon=4s          (should ~match (a) if the model is well fit)
  (c) GROUND-TRUTH risk over a LONGER window (8s, 12s) from the same state/gamma
      (reveals whether danger was already present just past the training horizon)

If (a)~(b) but (c) >> cvar_boundary while (b) <= cvar_boundary: the gate is
well-calibrated for what it measures, and the bug is the horizon length /
query cadence, not the risk head. If (a) << (b): the risk head itself is
mispredicting even within the window it was trained on.
"""
import numpy as np
import torch

import eval_scene_distribution as E
import drone_data_generation as G

CVAR = None  # filled from policy


def predicted_risk(policy, pos, vel, goal, obstacles, gamma):
    robot = [float(pos[0]), float(pos[1]), float(pos[2]),
            float(vel[0]), float(vel[1]), float(vel[2])]
    obs_list = [[float(c[0]), float(c[1]), float(c[2]), float(r), 0.0, 0.0, 0.0]
               for c, r in obstacles]
    graph = policy.gm.create_graph(robot=robot, obstacles=obs_list, goal=list(goal))
    cand = torch.tensor([[gamma]], dtype=torch.float)
    safety_preds, _perf, divs = policy.penn.predict_single_scene(graph, cand)
    mean_risk = float(np.mean([m for m, _ in safety_preds[0]]))
    return mean_risk, float(divs[0])


def trace_scene(policy, start, goal, obstacles):
    """Replays run_adaptive, recording (pos,vel,gamma) at every query and the
    step index of the first violation, if any."""
    pos, vel = start.copy(), np.zeros(3)
    min_h, first_violation_step = np.inf, None
    queries = []  # (step_idx, pos, vel, gamma)
    gamma = policy.select(pos, vel, goal, obstacles)
    queries.append((0, pos.copy(), vel.copy(), gamma))
    tick = 0
    for step in range(E.MAX_STEPS):
        if float(np.linalg.norm(goal - pos)) < 0.3:
            break
        tick += 1
        if tick >= E.QUERY_TICKS:
            tick = 0
            gamma = policy.select(pos, vel, goal, obstacles)
            queries.append((step, pos.copy(), vel.copy(), gamma))
        a_safe, h_step, _ = G.step_dynamics(
            pos, vel, goal, obstacles, gamma, E.BND, 0.5, E.V_MAX, E.K_P, E.K_V, E.A_MAX, True)
        if h_step < 0 and first_violation_step is None:
            first_violation_step = step
        min_h = min(min_h, h_step)
        vel = vel + a_safe * E.DT
        pos = pos + vel * E.DT
    return min_h, first_violation_step, queries


def main():
    scenes = E.make_holdout_scenes(50)
    policy = E.AdaptivePolicy(E.DEFAULT_CKPT, E.DEFAULT_THRESH)
    global CVAR
    CVAR = policy.risk_thr
    print(f"cvar_boundary = {CVAR:.4f}   (risk gate threshold)\n")

    rows = []
    for i, (s, g, o) in enumerate(scenes):
        min_h, viol_step, queries = trace_scene(policy, s, g, o)
        if min_h >= 0:
            continue
        # find the query interval containing (or immediately before) the violation
        q_before = None
        for (qstep, qpos, qvel, qgamma) in queries:
            if viol_step is not None and qstep <= viol_step:
                q_before = (qstep, qpos, qvel, qgamma)
        if q_before is None:
            q_before = queries[-1]
        qstep, qpos, qvel, qgamma = q_before

        pred_risk, jrd = predicted_risk(policy, qpos, qvel, g, o, qgamma)
        gt4, _, _, _ = G.simulate_horizon(qpos, qvel, g, o, qgamma, horizon_s=4.0, apply_cbf=True)
        gt8, _, _, _ = G.simulate_horizon(qpos, qvel, g, o, qgamma, horizon_s=8.0, apply_cbf=True)
        gt12, _, _, _ = G.simulate_horizon(qpos, qvel, g, o, qgamma, horizon_s=12.0, apply_cbf=True)
        seconds_to_violation = (viol_step - qstep) * E.DT if viol_step is not None else float("nan")

        rows.append((i, qgamma, pred_risk, gt4, gt8, gt12, seconds_to_violation, jrd))

    print(f"{'scene':>5} {'gamma':>6} {'model':>8} {'GT@4s':>8} {'GT@8s':>8} {'GT@12s':>8} "
          f"{'t_to_viol':>10} {'JRD':>6}  gate-pass?")
    print("-" * 90)
    for i, gamma, pred, gt4, gt8, gt12, tv, jrd in rows:
        passed = "YES (bug: model)" if pred <= CVAR else "no"
        gt4_flag = "<=CVAR" if gt4 <= CVAR else ">CVAR"
        print(f"{i:>5} {gamma:>6.2f} {pred:>8.4f} {gt4:>8.4f} {gt8:>8.4f} {gt12:>8.4f} "
              f"{tv:>9.2f}s {jrd:>6.3f}  model={passed}  GT@4s={gt4_flag}")

    preds = np.array([r[2] for r in rows])
    gt4s = np.array([r[3] for r in rows])
    gt8s = np.array([r[4] for r in rows])
    gt12s = np.array([r[5] for r in rows])

    print(f"\nn = {len(rows)} unsafe scenes' pre-violation query")
    print(f"mean |model_pred - GT@4s| = {np.mean(np.abs(preds - gt4s)):.4f}  "
          f"(calibration error against the model's OWN training objective)")
    print(f"model passed the gate (pred<=cvar) on {100*np.mean(preds <= CVAR):.0f}% of these")
    print(f"GT@4s was ALSO <=cvar on {100*np.mean(gt4s <= CVAR):.0f}% of these "
          f"(if high: the 4s window genuinely looked safe, ground truth agrees)")
    print(f"GT@8s was <=cvar on {100*np.mean(gt8s <= CVAR):.0f}% of these")
    print(f"GT@12s was <=cvar on {100*np.mean(gt12s <= CVAR):.0f}% of these")
    print(f"\nmean GT risk: @4s={gt4s.mean():.4f}  @8s={gt8s.mean():.4f}  @12s={gt12s.mean():.4f}")


if __name__ == "__main__":
    main()
