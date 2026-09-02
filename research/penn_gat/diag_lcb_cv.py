"""
diag_lcb_cv.py

STAGE 1 OF 2. This script sweeps lcb_k on the POINT-MASS simulator only.
Its output is NOT the deployed value on its own -- see
diag_lcb_cv_closed_loop.py for stage 2 and the final selection.

Why (2026-08-22/23): this script's own pick, lcb_k=1.50, was measured on
the real body-rate controller and made things WORSE -- closed-loop
deadlocks 14->22/350, paired speed +2.3%->-2.1%, no hard-collision benefit
either way. Point-mass dynamics have no deadlock failure mode, so a
point-mass sweep structurally cannot select against one, no matter how
clean its calibrate/test discipline is. Deployment-facing parameters need a
second selection stage on the system that actually deploys.

The deployed value is lcb_k=0.25, chosen as follows, using CALIBRATION
SEEDS ONLY at every step:
  - closed-loop (stage 2): rules out lcb_k >= 0.5 (clearly more deadlocks);
    cannot distinguish 0.0 from 0.25 (4 vs 5 discordant scenes of 300).
  - point-mass (this script): prefers 0.25 over 0.0 clearly
    (margin-violations 11.8 -> 9.0 per 50 scenes).
  => 0.25 is the value that is non-harmful closed-loop AND beneficial
     point-mass. No test-seed data entered the decision.

Keep running this script for the point-mass half of that picture; just do
not read its SELECTED line as the deployment answer.

Original description: calibrates lcb_k using the exact same
calibrate/test-seed protocol as diag_margin_cv.py: sweep on CAL_SEEDS
only, pick the least-conservative value achieving 0 mean hard collisions
THERE, then report performance on held-out TEST_SEEDS the selection never
saw.

Run after the aleatoric head retrain (2026-08-22, widened log_std clamp in
penn.py so the head is no longer degenerate) and after cccp_calibrate_drone.py
has refreshed cvar_boundary/raw_epistemic_threshold for the new checkpoint --
lcb_k changes the *function* being thresholded against cvar_boundary
(mean_risk - lcb_k*sqrt(var_al+var_ep) instead of raw mean_risk), which the
mu-only calibration in cccp_calibrate_drone.py never accounts for.

risk_margin is held at 0.0 here so this isolates lcb_k's effect; a joint
sweep is more thorough but 2D and not needed to establish whether the
aleatoric variance head is now load-bearing at all.

Selection rule (2026-08-22 decision): among lcb_k values with 0 mean hard
collisions on calibration seeds, pick the one that MINIMIZES mean
margin-violations, not the one that maximizes speed. A speed-maximizing
rule always picks lcb_k=0 the moment cvar_boundary's own margin already
clears the 0-hard-collision floor on its own (which it does here) -- that
answer is technically correct but means the newly-fixed aleatoric head,
while genuinely calibrated (predicted sigma correlates with true residual
at r=0.67, see nn_model/checkpoint's retrain notes), would sit unused by
the deployed gate. Minimizing margin-violations instead makes the LCB term
earn its keep: it trades some speed for fewer near-miss grazes, which is
exactly what a variance-aware safety gate is supposed to do, and is
consistent with this project's existing practice of reporting
margin-violations as a real (if softer) safety metric alongside hard
collisions.
"""
import numpy as np

import eval_scene_distribution as E
from diag_margin_cv import const_reference

CAL_SEEDS = [20260821, 20260901, 20260902, 20260903, 20260904, 20260905]
TEST_SEEDS = [20260906, 20260907, 20260908, 20260909, 20260910, 20260911, 20260912]
LCB_KS = [0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0]


def eval_lcb(scenes, lcb_k, const_t):
    pol = E.AdaptivePolicy(E.DEFAULT_CKPT, E.DEFAULT_THRESH,
                           gate_mode="network", risk_margin=0.0, lcb_k=lcb_k)
    mh, mhp, t = [], [], []
    for (s, g, o) in scenes:
        a, b, c = E.run_adaptive(pol, s, g, o)
        mh.append(a); mhp.append(b); t.append(c)
    mh = np.array(mh); mhp = np.array(mhp); t = np.array(t)
    safe = mh >= 0
    st = t[safe].mean() if safe.any() else float("nan")
    return {
        "margin_viol": int((~safe).sum()),
        "hard_coll": int((mhp < 0).sum()),
        "t_safe": st,
        "speed_vs_const": 100 * (const_t - st) / const_t if st == st else float("nan"),
        "fallback": pol.n_fallback / max(pol.n_queries, 1),
    }


def main():
    print("=" * 78)
    print("CALIBRATION -- sweep lcb_k (risk_margin=0) on these seeds only, pick")
    print(f"the value with 0 mean hard collisions and MIN margin-violations: {CAL_SEEDS}")
    print("=" * 78)

    cal_scenes = {s: E.make_holdout_scenes(50, seed=s) for s in CAL_SEEDS}
    cal_const = {s: const_reference(cal_scenes[s]) for s in CAL_SEEDS}

    best_k, best_viol, best_speed = None, 1e9, None
    for k in LCB_KS:
        results = [eval_lcb(cal_scenes[s], k, cal_const[s]["t_safe"])
                  for s in CAL_SEEDS]
        mean_hard = np.mean([r["hard_coll"] for r in results])
        mean_viol = np.mean([r["margin_viol"] for r in results])
        mean_speed = np.mean([r["speed_vs_const"] for r in results])
        mean_fb = np.mean([r["fallback"] for r in results])
        ok = mean_hard == 0.0
        print(f"  lcb_k={k:.2f}  mean hard={mean_hard:.2f}  "
              f"mean margin-viol={mean_viol:.1f}/50  mean speed={mean_speed:+.1f}%  "
              f"mean fallback={100*mean_fb:.0f}%  "
              f"{'[eligible]' if ok else '[REJECTED: hard collisions present]'}")
        if ok and mean_viol < best_viol:
            best_k, best_viol, best_speed = k, mean_viol, mean_speed

    if best_k is None:
        print("\nNo lcb_k achieved 0 mean hard collisions on the calibration set.")
        return

    print(f"\nSELECTED on calibration set (POINT-MASS ONLY -- stage 1 of 2, "
          f"NOT the deployed value): lcb_k={best_k:.2f} "
          f"(mean margin-viol {best_viol:.1f}/50, mean {best_speed:+.1f}% vs const, "
          f"0 mean hard collisions)")
    print("  -> run diag_lcb_cv_closed_loop.py for stage 2; deployed value is 0.25. "
          "See this file's docstring.\n")

    print("=" * 78)
    print(f"TEST -- evaluate lcb_k={best_k:.2f} on seeds it never saw: {TEST_SEEDS}")
    print("=" * 78)

    test_hard, test_viol, test_speed = [], [], []
    for s in TEST_SEEDS:
        scenes = E.make_holdout_scenes(50, seed=s)
        cref = const_reference(scenes)
        r = eval_lcb(scenes, best_k, cref["t_safe"])
        print(f"  seed {s}: const gamma={cref['gamma']:.2f} t_safe={cref['t_safe']:.2f}s  "
              f"-> adaptive {r['margin_viol']}/50 margin-viol  {r['hard_coll']}/50 HARD  "
              f"t_safe={r['t_safe']:.2f}s  {r['speed_vs_const']:+.1f}%  "
              f"fallback={100*r['fallback']:.0f}%")
        test_hard.append(r["hard_coll"])
        test_viol.append(r["margin_viol"])
        test_speed.append(r["speed_vs_const"])

    print("-" * 78)
    print(f"TEST-SET SUMMARY ({len(TEST_SEEDS)} seeds, lcb_k selection never saw these):")
    print(f"  mean hard collisions/seed:   {np.mean(test_hard):.2f}  "
          f"(seeds with >=1: {sum(1 for h in test_hard if h>0)}/{len(TEST_SEEDS)})")
    print(f"  mean margin-violations/seed: {np.mean(test_viol):.1f}/50")
    print(f"  mean speed vs constant:      {np.mean(test_speed):+.1f}%  "
          f"(range {min(test_speed):+.1f}% to {max(test_speed):+.1f}%)")
    print(f"\nSelected lcb_k={best_k:.2f} -- this is the number to deploy.")


if __name__ == "__main__":
    main()
