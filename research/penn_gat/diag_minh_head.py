"""
diag_minh_head.py

The min_h_horizon label swap (2026-08-20) did not move the shipped
`network`-gate violation rate (36%, unchanged). Two very different causes
produce that same symptom, and they need opposite fixes:

  (a) the risk head predicts min_h badly -> more/better training, different
      architecture, longer schedule
  (b) the risk head predicts min_h fine, but something downstream (the
      epistemic JRD gate, or a miscalibrated cvar_boundary) discards the
      good candidates anyway -> no retrain needed at all, just fix the gate

This is the same technique that cracked the risk-integral bug earlier
(diag_horizon_blindspot.py): instrument the EXACT query states, compare the
network's prediction to ground truth from the real simulator, and let the
comparison say which layer is at fault -- rather than retraining blindly and
guessing from the aggregate number.

Reports:
  1. Prediction quality across ALL queries (corr, RMSE, bias, sign accuracy)
     -- is the head learning min_h at all?
  2. The same restricted to pre-violation query states -- does it fail
     specifically where it matters, even if it's fine on average?
  3. Gate accounting: per query, how many candidates each gate rejected, how
     often a genuinely-safe candidate existed but was rejected anyway, and
     which gate did the rejecting. This is what separates (a) from (b).
"""
import numpy as np
import torch

import eval_scene_distribution as E
import drone_data_generation as G


def analyze(n_scenes=50):
    scenes = E.make_holdout_scenes(n_scenes)
    pol = E.AdaptivePolicy(E.DEFAULT_CKPT, E.DEFAULT_THRESH, gate_mode="network")
    cands = torch.linspace(E.DEPLOY_GAMMA_MIN, E.DEPLOY_GAMMA_MAX,
                           E.DEPLOY_STEPS).unsqueeze(1)

    print(f"epistemic threshold (JRD)  = {pol.ep_thr:.6f}")
    print(f"cvar_boundary (min_h lower bound) = {pol.risk_thr:.6f}\n")

    all_pred, all_true = [], []
    viol_pred, viol_true = [], []
    # gate accounting, per query
    n_q = 0
    n_ep_rej_total = 0
    n_risk_rej_total = 0
    n_fallback = 0
    n_safe_existed_but_all_rejected = 0
    n_safe_existed_but_ep_rejected_them = 0
    n_selected_actually_unsafe = 0
    n_scenes_viol = 0

    for (s, g, o) in scenes:
        pos, vel = s.copy(), np.zeros(3)
        tick = 0
        episode_queries = []   # (step, pos, vel) for every query in this episode
        min_h_run = np.inf
        first_viol_step = None

        # replay the episode exactly as run_adaptive does, recording queries
        gamma = pol.select(pos, vel, g, o)
        episode_queries.append((0, pos.copy(), vel.copy()))
        for step in range(E.MAX_STEPS):
            if float(np.linalg.norm(g - pos)) < 0.3:
                break
            tick += 1
            if tick >= E.QUERY_TICKS:
                tick = 0
                gamma = pol.select(pos, vel, g, o)
                episode_queries.append((step, pos.copy(), vel.copy()))
            a, h, _ = G.step_dynamics(pos, vel, g, o, gamma, E.BND, 0.5,
                                      E.V_MAX, E.K_P, E.K_V, E.A_MAX, True)
            if h < 0 and first_viol_step is None:
                first_viol_step = step
            min_h_run = min(min_h_run, h)
            vel = vel + a * E.DT
            pos = pos + vel * E.DT

        scene_violated = min_h_run < 0
        if scene_violated:
            n_scenes_viol += 1

        # re-evaluate every query state: network prediction vs ground truth
        for (qstep, qp, qv) in episode_queries:
            vis = G.visible_obstacles(qp, o)
            if not vis:
                continue
            robot = [float(qp[0]), float(qp[1]), float(qp[2]),
                     float(qv[0]), float(qv[1]), float(qv[2])]
            obs_list = [[float(c[0]), float(c[1]), float(c[2]), float(r), 0., 0., 0.]
                        for c, r in vis]
            graph = pol.gm.create_graph(robot=robot, obstacles=obs_list,
                                        goal=[float(x) for x in g])
            safety_preds, _perf, divs = pol.penn.predict_single_scene(graph, cands)

            n_q += 1
            n_ep_rej = n_risk_rej = 0
            any_true_safe = False
            any_true_safe_survived = False
            ep_killed_a_safe_one = False

            for i in range(E.DEPLOY_STEPS):
                gam = float(cands[i].item())
                pred = float(np.mean([m for m, _ in safety_preds[i]]))
                _r, _p, _fc, true_mh = G.simulate_horizon(
                    qp, qv, g, o, gam, boundary=E.BND,
                    horizon_s=G.HORIZON_S, apply_cbf=True)

                all_pred.append(pred)
                all_true.append(true_mh)
                is_pre_viol = (scene_violated and first_viol_step is not None
                               and qstep <= first_viol_step)
                if is_pre_viol:
                    viol_pred.append(pred)
                    viol_true.append(true_mh)

                ep_rej = pol.ep_thr > 0 and divs[i] > pol.ep_thr
                risk_rej = pred < pol.risk_thr
                if ep_rej:
                    n_ep_rej += 1
                if risk_rej and not ep_rej:
                    n_risk_rej += 1

                if true_mh >= 0:
                    any_true_safe = True
                    if not ep_rej and not risk_rej:
                        any_true_safe_survived = True
                    if ep_rej:
                        ep_killed_a_safe_one = True

            n_ep_rej_total += n_ep_rej
            n_risk_rej_total += n_risk_rej
            if n_ep_rej + n_risk_rej >= E.DEPLOY_STEPS:
                n_fallback += 1
            if any_true_safe and not any_true_safe_survived:
                n_safe_existed_but_all_rejected += 1
                if ep_killed_a_safe_one:
                    n_safe_existed_but_ep_rejected_them += 1

    ap = np.array(all_pred); at = np.array(all_true)
    print("=" * 74)
    print("1. RISK-HEAD PREDICTION QUALITY  (predicted min_h vs ground truth)")
    print("=" * 74)
    print(f"  samples                {len(ap):,}")
    print(f"  corr(pred, true)       {np.corrcoef(ap, at)[0,1]:+.3f}")
    print(f"  RMSE                   {np.sqrt(((ap-at)**2).mean()):.3f}")
    print(f"  bias (pred - true)     {(ap-at).mean():+.3f}")
    print(f"  true min_h  mean/std   {at.mean():+.3f} / {at.std():.3f}")
    print(f"  pred min_h  mean/std   {ap.mean():+.3f} / {ap.std():.3f}")
    sign_acc = ((ap >= 0) == (at >= 0)).mean()
    print(f"  sign agreement         {100*sign_acc:.1f}%   "
          f"(does it get safe-vs-unsafe right at all?)")
    unsafe = at < 0
    if unsafe.any():
        print(f"  on TRULY UNSAFE cands: pred mean {ap[unsafe].mean():+.3f} "
              f"vs true {at[unsafe].mean():+.3f}   "
              f"-> called SAFE anyway {100*(ap[unsafe] >= 0).mean():.1f}% of the time")

    if viol_pred:
        vp = np.array(viol_pred); vt = np.array(viol_true)
        print("\n" + "=" * 74)
        print("2. SAME, RESTRICTED TO PRE-VIOLATION QUERY STATES")
        print("=" * 74)
        print(f"  samples                {len(vp):,}")
        print(f"  corr(pred, true)       {np.corrcoef(vp, vt)[0,1]:+.3f}")
        print(f"  RMSE                   {np.sqrt(((vp-vt)**2).mean()):.3f}")
        print(f"  bias (pred - true)     {(vp-vt).mean():+.3f}")
        vu = vt < 0
        if vu.any():
            print(f"  on TRULY UNSAFE cands: called SAFE anyway "
                  f"{100*(vp[vu] >= 0).mean():.1f}% of the time")

    print("\n" + "=" * 74)
    print("3. GATE ACCOUNTING  -- which layer is discarding good candidates?")
    print("=" * 74)
    print(f"  scenes with a violation          {n_scenes_viol}/{n_scenes}")
    print(f"  total queries                    {n_q:,}")
    print(f"  candidates rejected by EPISTEMIC {n_ep_rej_total:,} "
          f"({100*n_ep_rej_total/max(n_q*E.DEPLOY_STEPS,1):.1f}% of all candidates)")
    print(f"  candidates rejected by RISK gate {n_risk_rej_total:,} "
          f"({100*n_risk_rej_total/max(n_q*E.DEPLOY_STEPS,1):.1f}% of all candidates)")
    print(f"  queries where EVERYTHING rejected (fallback) {n_fallback:,} "
          f"({100*n_fallback/max(n_q,1):.1f}%)")
    print(f"\n  queries where a genuinely-safe gamma EXISTED but every")
    print(f"  candidate was rejected anyway:   {n_safe_existed_but_all_rejected:,} "
          f"({100*n_safe_existed_but_all_rejected/max(n_q,1):.1f}% of queries)")
    print(f"    ...of those, epistemic gate killed a safe one: "
          f"{n_safe_existed_but_ep_rejected_them:,}")
    print("\n  READING THIS: high epistemic rejection + many 'safe existed but")
    print("  rejected' queries => the gate is the problem, not the head (no")
    print("  retrain needed). Poor corr/sign-agreement in section 1 => the head")
    print("  genuinely isn't learning min_h and the label swap needs more than")
    print("  a threshold fix.")


if __name__ == "__main__":
    analyze()
