"""
diag_margin_cv.py

diag_gate_sweep.py picked risk_margin=0.30 by eyeballing a sweep on ONE
scene-set seed, then diag_validate_margin.py found it didn't generalize
(0.30 gave +14.7% on the tuning seed, averaged ~+2% and produced a hard
collision on 3 fresh seeds). diag_ceiling_across_seeds.py then confirmed
the underlying ceiling (best-constant vs per-scene-oracle gap) IS robust
across 13 seeds (25.6% mean, std 3.8) -- so the target is real, the
single-seed margin selection was the actual mistake.

This does it properly: sweep margin on CALIBRATION seeds only, pick the
value that meets a safety bar with the best mean cost THERE, then report
performance on separate TEST seeds the margin selection never saw. This is
the number that's allowed to be believed.

Safety bar: among calibration seeds, only consider margins with a
mean hard-collision rate of exactly 0 -- hard collisions are the one metric
this project has treated as non-negotiable all session.
"""
import numpy as np

import eval_scene_distribution as E

CAL_SEEDS = [20260821, 20260901, 20260902, 20260903, 20260904, 20260905]
TEST_SEEDS = [20260906, 20260907, 20260908, 20260909, 20260910, 20260911, 20260912]
MARGINS = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5]


def const_reference(scenes):
    n = len(scenes)
    C = np.zeros((n, len(E.CONST_CANDIDATES)))
    T = np.zeros((n, len(E.CONST_CANDIDATES)))
    MH = np.zeros((n, len(E.CONST_CANDIDATES)))
    for i, (s, g, o) in enumerate(scenes):
        for j, gam in enumerate(E.CONST_CANDIDATES):
            a, _b, c = E.run_fixed(s, g, o, float(gam))
            MH[i, j], T[i, j] = a, c
            C[i, j] = E.cost(a, c)
    jb = int(C.mean(axis=0).argmin())
    return {
        "gamma": float(E.CONST_CANDIDATES[jb]),
        "viol": int((MH[:, jb] < 0).sum()),
        "t_safe": float(T[MH[:, jb] >= 0, jb].mean()),
    }


def eval_margin(scenes, margin, const_t):
    pol = E.AdaptivePolicy(E.DEFAULT_CKPT, E.DEFAULT_THRESH,
                           gate_mode="network", risk_margin=margin)
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
    print("CALIBRATION -- sweep margin on these seeds only, pick the value")
    print(f"with 0 mean hard collisions and best speed: {CAL_SEEDS}")
    print("=" * 78)

    cal_scenes = {s: E.make_holdout_scenes(50, seed=s) for s in CAL_SEEDS}
    cal_const = {s: const_reference(cal_scenes[s]) for s in CAL_SEEDS}

    best_margin, best_speed = None, -1e9
    for margin in MARGINS:
        results = [eval_margin(cal_scenes[s], margin, cal_const[s]["t_safe"])
                  for s in CAL_SEEDS]
        mean_hard = np.mean([r["hard_coll"] for r in results])
        mean_viol = np.mean([r["margin_viol"] for r in results])
        mean_speed = np.mean([r["speed_vs_const"] for r in results])
        ok = mean_hard == 0.0
        print(f"  margin={margin:.2f}  mean hard={mean_hard:.2f}  "
              f"mean margin-viol={mean_viol:.1f}/50  mean speed={mean_speed:+.1f}%  "
              f"{'[eligible]' if ok else '[REJECTED: hard collisions present]'}")
        if ok and mean_speed > best_speed:
            best_margin, best_speed = margin, mean_speed

    print(f"\nSELECTED on calibration set: margin={best_margin:.2f} "
          f"(mean {best_speed:+.1f}% vs const, 0 mean hard collisions)\n")

    print("=" * 78)
    print(f"TEST -- evaluate margin={best_margin:.2f} on seeds it never saw: {TEST_SEEDS}")
    print("=" * 78)

    test_hard, test_viol, test_speed = [], [], []
    for s in TEST_SEEDS:
        scenes = E.make_holdout_scenes(50, seed=s)
        cref = const_reference(scenes)
        r = eval_margin(scenes, best_margin, cref["t_safe"])
        print(f"  seed {s}: const gamma={cref['gamma']:.2f} t_safe={cref['t_safe']:.2f}s  "
              f"-> adaptive {r['margin_viol']}/50 margin-viol  {r['hard_coll']}/50 HARD  "
              f"t_safe={r['t_safe']:.2f}s  {r['speed_vs_const']:+.1f}%  "
              f"fallback={100*r['fallback']:.0f}%")
        test_hard.append(r["hard_coll"])
        test_viol.append(r["margin_viol"])
        test_speed.append(r["speed_vs_const"])

    print("-" * 78)
    print(f"TEST-SET SUMMARY ({len(TEST_SEEDS)} seeds, margin selection never saw these):")
    print(f"  mean hard collisions/seed:   {np.mean(test_hard):.2f}  "
          f"(seeds with >=1: {sum(1 for h in test_hard if h>0)}/{len(TEST_SEEDS)})")
    print(f"  mean margin-violations/seed: {np.mean(test_viol):.1f}/50")
    print(f"  mean speed vs constant:      {np.mean(test_speed):+.1f}%  "
          f"(range {min(test_speed):+.1f}% to {max(test_speed):+.1f}%)")
    print("\nThis is the number to actually report -- calibrated on one set of")
    print("seeds, measured on a completely different set that never influenced")
    print("the margin choice.")


if __name__ == "__main__":
    main()
