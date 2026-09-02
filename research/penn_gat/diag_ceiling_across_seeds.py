"""
diag_ceiling_across_seeds.py

The whole scene-level reframe rests on one number: the gap between the best
FITTED CONSTANT gamma and the PER-SCENE ORACLE, measured at 24-32% and
treated since as the ceiling worth chasing. It was measured on a single
scene-set seed (EVAL_SEED = 20260821).

diag_validate_margin.py (2026-08-21) found that seed is not typical: its
best fitted constant is 0.80 -> 8.53s, while three fresh seeds give
1.00-1.20 -> 7.32-8.14s. A conservative/slow best-constant means a WEAK
baseline, which inflates every "% faster than the constant" number measured
against it -- including, potentially, the ceiling itself. If the ceiling is
seed-specific, the premise of the redesign is too.

This measures ceiling, best-constant gamma, and best-constant time across
many independent seeds, purely from the fixed-gamma sweep (no network, no
checkpoint -- so it's immune to retrain noise and fast).
"""
import numpy as np

import eval_scene_distribution as E


def one_seed(seed, n_scenes=50):
    scenes = E.make_holdout_scenes(n_scenes, seed=seed)
    n = len(scenes)
    C = np.zeros((n, len(E.CONST_CANDIDATES)))
    T = np.zeros((n, len(E.CONST_CANDIDATES)))
    MH = np.zeros((n, len(E.CONST_CANDIDATES)))
    for i, (s, g, o) in enumerate(scenes):
        for j, gam in enumerate(E.CONST_CANDIDATES):
            a, _b, c = E.run_fixed(s, g, o, float(gam))
            MH[i, j], T[i, j] = a, c
            C[i, j] = E.cost(a, c)
    mean_per_const = C.mean(axis=0)
    jb = int(mean_per_const.argmin())
    V_base = mean_per_const[jb]
    V_orc = C.min(axis=1).mean()
    ceiling = 100 * (V_base - V_orc) / V_base if V_base > 1e-9 else 0.0
    return {
        "seed": seed,
        "best_gamma": float(E.CONST_CANDIDATES[jb]),
        "V_base": float(V_base),
        "V_orc": float(V_orc),
        "ceiling": float(ceiling),
        "const_viol": int((MH[:, jb] < 0).sum()),
        "t_safe": float(T[MH[:, jb] >= 0, jb].mean()),
    }


def main():
    seeds = [20260821] + list(range(20260901, 20260913))
    print(f"{'seed':>10} {'best_g':>8} {'const viol':>11} {'t_safe':>9} "
          f"{'V_base':>9} {'V_oracle':>10} {'ceiling':>9}")
    print("-" * 72)
    rows = []
    for s in seeds:
        r = one_seed(s)
        tag = "  <-- EVAL_SEED" if s == 20260821 else ""
        print(f"{r['seed']:>10} {r['best_gamma']:>8.2f} {r['const_viol']:>8}/50 "
              f"{r['t_safe']:>8.2f}s {r['V_base']:>8.2f}s {r['V_orc']:>9.2f}s "
              f"{r['ceiling']:>8.1f}%{tag}")
        rows.append(r)

    cl = np.array([r["ceiling"] for r in rows])
    bg = np.array([r["best_gamma"] for r in rows])
    ts = np.array([r["t_safe"] for r in rows])
    ev = rows[0]
    print("-" * 72)
    print(f"ceiling across {len(rows)} seeds: mean {cl.mean():.1f}%  "
          f"median {np.median(cl):.1f}%  min {cl.min():.1f}%  max {cl.max():.1f}%  "
          f"std {cl.std():.1f}")
    print(f"EVAL_SEED ceiling = {ev['ceiling']:.1f}%  "
          f"-> percentile among these seeds: "
          f"{100*(cl < ev['ceiling']).mean():.0f}%")
    print(f"best-constant gamma: mean {bg.mean():.2f}  "
          f"EVAL_SEED {ev['best_gamma']:.2f}")
    print(f"best-constant t_safe: mean {ts.mean():.2f}s  "
          f"EVAL_SEED {ev['t_safe']:.2f}s  "
          f"-> EVAL_SEED baseline is "
          f"{100*(ev['t_safe']-ts.mean())/ts.mean():+.1f}% slower than typical")
    print("\nIf EVAL_SEED sits far above the median ceiling and its baseline is")
    print("markedly slower than typical, then every '% faster than the best")
    print("constant' figure measured on it -- including the 24-32% ceiling the")
    print("redesign is premised on -- is inflated and needs restating.")


if __name__ == "__main__":
    main()
