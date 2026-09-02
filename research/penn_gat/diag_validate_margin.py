"""Validate margin=0.30 on FRESH scene draws it was not selected on."""
import numpy as np
import eval_scene_distribution as E

for seed in (20260901, 20260902, 20260903):
    scenes = E.make_holdout_scenes(50, seed=seed)
    n = len(scenes)

    # refit the best constant ON THIS SEED (strongest baseline)
    C = np.zeros((n, len(E.CONST_CANDIDATES)))
    T = np.zeros((n, len(E.CONST_CANDIDATES)))
    MH = np.zeros((n, len(E.CONST_CANDIDATES)))
    for i, (s, g, o) in enumerate(scenes):
        for j, gam in enumerate(E.CONST_CANDIDATES):
            a, _b, c = E.run_fixed(s, g, o, float(gam))
            MH[i, j], T[i, j] = a, c
            C[i, j] = E.cost(a, c)
    jb = int(C.mean(axis=0).argmin())
    bc = float(E.CONST_CANDIDATES[jb])
    const_viol = int((MH[:, jb] < 0).sum())
    const_t = T[MH[:, jb] >= 0, jb].mean()

    print("=" * 72)
    print(f"seed {seed}:  best constant gamma={bc:.2f}  "
          f"{const_viol}/{n} viol  t_safe={const_t:.2f}s")

    for margin in (0.30,):
        pol = E.AdaptivePolicy(E.DEFAULT_CKPT, E.DEFAULT_THRESH,
                               gate_mode="network", risk_margin=margin)
        mh, mhp, t = [], [], []
        for (s, g, o) in scenes:
            a, b, c = E.run_adaptive(pol, s, g, o)
            mh.append(a); mhp.append(b); t.append(c)
        mh = np.array(mh); mhp = np.array(mhp); t = np.array(t)
        safe = mh >= 0
        st = t[safe].mean()
        speed = 100 * (const_t - st) / const_t
        depths = mh[~safe]
        # NOTE: converting an h-value to a penetration depth in metres needs the r_eff of the
        # obstacle that produced it.
        pen = []
        print(f"  margin={margin:.2f}: {int((~safe).sum())}/{n} margin-viol  "
              f"{int((mhp < 0).sum())}/{n} HARD  t_safe={st:.2f}s  "
              f"{speed:+.1f}% vs const  fallback={100*pol.n_fallback/max(pol.n_queries,1):.0f}%")
        if len(pen):
            print(f"    worst penetration into 500mm buffer: {max(pen):.1f} mm  "
                  f"(all: {np.round(pen,1).tolist()})")
