"""
diag_paired_speed.py

The headline "+29.4% faster than the best fitted constant" (diag_margin_cv.py)
is computed as t_safe: the mean time-to-goal over episodes where THAT ARM did
not margin-violate. The constant violates on ~0-2/50 so it averages over
nearly all scenes; the adaptive policy violates on ~14/50 so it averages over
only ~36. If the scenes adaptive grazes on are the HARD ones -- which is
exactly where grazing happens -- then the two averages are taken over
different-difficulty subsets and the comparison is survivorship-biased in
adaptive's favour.

This measures it three ways per seed:
  unpaired  -- the current method (each arm over its own safe episodes)
  paired    -- both arms over the SAME scenes (those where both are safe)
  hardness  -- the constant's own time on the scenes adaptive grazed vs the
               scenes it didn't. If grazed scenes are slower FOR THE CONSTANT
               TOO, they're objectively harder and the bias is real.
"""
import numpy as np

import eval_scene_distribution as E

TEST_SEEDS = [20260906, 20260907, 20260908, 20260909, 20260910, 20260911, 20260912]


def one_seed(seed):
    scenes = E.make_holdout_scenes(50, seed=seed)
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
    c_mh, c_t = MH[:, jb], T[:, jb]

    pol = E.AdaptivePolicy(E.DEFAULT_CKPT, E.DEFAULT_THRESH,
                           gate_mode="network", risk_margin=0.0)
    a_mh = np.zeros(n); a_t = np.zeros(n)
    for i, (s, g, o) in enumerate(scenes):
        m, _p, t = E.run_adaptive(pol, s, g, o)
        a_mh[i], a_t[i] = m, t

    a_safe = a_mh >= 0
    c_safe = c_mh >= 0
    both = a_safe & c_safe

    unpaired = 100 * (c_t[c_safe].mean() - a_t[a_safe].mean()) / c_t[c_safe].mean()
    paired = 100 * (c_t[both].mean() - a_t[both].mean()) / c_t[both].mean()
    # hardness: constant's OWN time on scenes adaptive grazed vs not
    hard_t = c_t[~a_safe].mean() if (~a_safe).any() else float("nan")
    easy_t = c_t[a_safe].mean() if a_safe.any() else float("nan")

    return {
        "seed": seed, "gamma": float(E.CONST_CANDIDATES[jb]),
        "n_a_viol": int((~a_safe).sum()), "n_both": int(both.sum()),
        "unpaired": unpaired, "paired": paired,
        "const_t_on_grazed": hard_t, "const_t_on_clean": easy_t,
    }


def main():
    print(f"{'seed':>10} {'a-viol':>7} {'n both':>7} {'UNPAIRED':>10} {'PAIRED':>9} "
          f"{'const t: grazed':>16} {'clean':>8}")
    print("-" * 76)
    rows = []
    for s in TEST_SEEDS:
        r = one_seed(s)
        print(f"{r['seed']:>10} {r['n_a_viol']:>5}/50 {r['n_both']:>7} "
              f"{r['unpaired']:>+9.1f}% {r['paired']:>+8.1f}% "
              f"{r['const_t_on_grazed']:>15.2f}s {r['const_t_on_clean']:>7.2f}s")
        rows.append(r)

    up = np.array([r["unpaired"] for r in rows])
    pa = np.array([r["paired"] for r in rows])
    hg = np.array([r["const_t_on_grazed"] for r in rows])
    cl = np.array([r["const_t_on_clean"] for r in rows])
    print("-" * 76)
    print(f"mean UNPAIRED (reported number): {up.mean():+.1f}%")
    print(f"mean PAIRED   (bias-corrected):  {pa.mean():+.1f}%")
    print(f"difference attributable to subset selection: {up.mean()-pa.mean():+.1f} pts")
    print()
    print(f"constant's own mean time on scenes ADAPTIVE GRAZED: {hg.mean():.2f}s")
    print(f"constant's own mean time on scenes adaptive was clean: {cl.mean():.2f}s")
    if hg.mean() > cl.mean():
        print(f"  -> grazed scenes are {100*(hg.mean()-cl.mean())/cl.mean():.0f}% slower for the")
        print("     CONSTANT too, i.e. they are objectively harder scenes. The")
        print("     unpaired number is survivorship-biased; PAIRED is the honest one.")
    else:
        print("  -> grazed scenes are NOT harder for the constant, so the unpaired")
        print("     comparison is not obviously biased by scene difficulty.")


if __name__ == "__main__":
    main()
