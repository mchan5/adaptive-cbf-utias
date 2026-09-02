"""
diag_gate_sweep.py

diag_minh_head.py established that the risk head predicts min_h well
(corr +0.991, RMSE 0.173, bias +0.006) but that the decision boundary lives
inside its noise floor: truly-unsafe candidates sit at a mean true min_h of
-0.073, ~2.4x smaller than the head's own error, so 37.8% of them are
predicted safe and pass the gate. The epistemic gate was exonerated in the
same run (5.3% rejection; killed a safe candidate in 2/667 queries).

That diagnosis implies the fix is a THRESHOLD change, not a retrain. Two
candidate knobs, swept here against real episode outcomes:

  risk_margin -- require predicted min_h > cvar_boundary + margin. Pushes
                 the decision point out of the noise floor by brute force.
  lcb_k       -- gate on mu - k*sqrt(var_aleatoric + var_epistemic) instead
                 of the ensemble mean. The deployed rule currently discards
                 sigma entirely, so the PENN's predictive-variance head is
                 trained and then ignored at selection time; this turns it
                 back on. Should dominate a flat margin IF the head's
                 uncertainty is genuinely state-dependent (tight where it's
                 confident, wide near the boundary) rather than
                 homoscedastic -- cccp_calibrate_drone.py's risk-noise print
                 has hinted at near-constant sigma before, so this is a real
                 open question, not a foregone win.

Reports margin-violation rate AND hard-collision rate AND time-when-safe for
each setting, so the safety/speed trade is visible rather than collapsed
into one scalar (Phase 1 discipline -- see the redesign plan).
"""
import numpy as np

import eval_scene_distribution as E


def run_setting(scenes, risk_margin=0.0, lcb_k=0.0):
    pol = E.AdaptivePolicy(E.DEFAULT_CKPT, E.DEFAULT_THRESH,
                           gate_mode="network",
                           risk_margin=risk_margin, lcb_k=lcb_k)
    mh, mhp, t = [], [], []
    for (s, g, o) in scenes:
        a, b, c = E.run_adaptive(pol, s, g, o)
        mh.append(a); mhp.append(b); t.append(c)
    mh = np.array(mh); mhp = np.array(mhp); t = np.array(t)
    safe = mh >= 0
    reached = t < E.MAX_STEPS * E.DT - 1e-9
    safe_t = t[safe & reached]
    return {
        "margin_viol": int((~safe).sum()),
        "hard_coll": int((mhp < 0).sum()),
        "t_safe": float(safe_t.mean()) if len(safe_t) else float("nan"),
        "timeouts": int((~reached).sum()),
        "fallback_rate": pol.n_fallback / max(pol.n_queries, 1),
    }


def main():
    scenes = E.make_holdout_scenes(50)
    n = len(scenes)

    # reference points from the fixed sweep
    C = np.zeros((n, len(E.CONST_CANDIDATES)))
    T = np.zeros((n, len(E.CONST_CANDIDATES)))
    MH = np.zeros((n, len(E.CONST_CANDIDATES)))
    for i, (s, g, o) in enumerate(scenes):
        for j, gam in enumerate(E.CONST_CANDIDATES):
            a, _b, c = E.run_fixed(s, g, o, float(gam))
            MH[i, j], T[i, j] = a, c
            C[i, j] = E.cost(a, c)
    jb = int(C.mean(axis=0).argmin())
    best_const = float(E.CONST_CANDIDATES[jb])
    const_t = T[MH[:, jb] >= 0, jb].mean()
    print(f"reference: best fitted constant gamma={best_const:.2f}  "
          f"0/{n} viol  t_safe={const_t:.2f}s")
    orc_t = np.array([T[i, int(C[i].argmin())] for i in range(n)]).mean()
    print(f"reference: per-scene oracle          0/{n} viol  t_safe={orc_t:.2f}s\n")

    print(f"{'setting':<26} {'margin-viol':>12} {'HARD':>6} {'t_safe':>9} "
          f"{'vs const':>10} {'fallback':>10}")
    print("-" * 78)

    rows = []
    for margin in (0.0, 0.05, 0.1, 0.2, 0.3, 0.5):
        r = run_setting(scenes, risk_margin=margin)
        speed = 100 * (const_t - r["t_safe"]) / const_t if r["t_safe"] == r["t_safe"] else float("nan")
        print(f"{'margin=' + format(margin, '.2f'):<26} "
              f"{r['margin_viol']:>9}/{n} {r['hard_coll']:>5} "
              f"{r['t_safe']:>8.2f}s {speed:>9.1f}% {100*r['fallback_rate']:>9.1f}%")
        rows.append(("margin", margin, r, speed))

    print()
    for k in (0.5, 1.0, 1.5, 2.0, 3.0):
        r = run_setting(scenes, lcb_k=k)
        speed = 100 * (const_t - r["t_safe"]) / const_t if r["t_safe"] == r["t_safe"] else float("nan")
        print(f"{'lcb_k=' + format(k, '.1f'):<26} "
              f"{r['margin_viol']:>9}/{n} {r['hard_coll']:>5} "
              f"{r['t_safe']:>8.2f}s {speed:>9.1f}% {100*r['fallback_rate']:>9.1f}%")
        rows.append(("lcb", k, r, speed))

    print("\nA setting is only interesting if margin-viol drops a lot while")
    print("t_safe stays well under the constant's, i.e. it buys safety without")
    print("spending the entire speed advantage. Watch the fallback rate too: a")
    print("gate that rejects everything collapses to the constant-in-disguise")
    print("failure mode even when its violation number looks great.")


if __name__ == "__main__":
    main()
