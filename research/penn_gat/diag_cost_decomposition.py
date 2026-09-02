"""
diag_cost_decomposition.py

The Stage-3 summary reports a single scalar `mean cost` per arm, where an
unsafe episode is charged a flat MAX_STEPS*DT*2 = 50s penalty. That conflates
two completely different failure modes -- "the policy is slow" and "the
policy is fast but violates on some scenes" -- and the redesign for each is
totally different. This decomposes it:

  - violation RATE, reported at BOTH the margin boundary (existing metric)
    and the physical obstacle surface (2026-08-20, redesign Phase 1 --
    a millimetre graze into the 0.5m SAFETY_MARGIN and an actual collision
    were being scored identically; see eval_scene_distribution.py's
    _min_h_physical())
  - mean time-to-goal among SAFE episodes only (is the speed benefit real?)
  - the same for the best fitted constant and the per-scene oracle

If adaptive's safe-episode time is already at/near oracle while its
violation rate is high, the remaining problem is purely a safety/recursive-
feasibility one, not a performance-prediction one -- a much better-posed
problem than "the architecture is myopic".

Also checks whether violations are preceded by the fallback firing (i.e. the
policy drove itself into a state where NO candidate gamma passes the gate),
which is the signature of a recursive-feasibility failure rather than a
single-step prediction error.

--query-ticks overrides eval_scene_distribution.QUERY_TICKS for this run
only (2026-08-20, redesign Phase 2's cheapest hypothesis: does re-selecting
gamma more often reduce cornering, with no retrain required?).
"""
import argparse

import numpy as np

import eval_scene_distribution as E
import drone_data_generation as G


def run_arm(scenes, mode, policy=None):
    """Returns per-scene (min_h_margin, min_h_phys, t_goal, n_fallback_this_episode)."""
    out = []
    for (s, g, o) in scenes:
        if mode == "adaptive":
            fb0 = policy.n_fallback
            mh, mh_phys, t = E.run_adaptive(policy, s, g, o)
            out.append((mh, mh_phys, t, policy.n_fallback - fb0))
        else:
            mh, mh_phys, t = E.run_fixed(s, g, o, mode)
            out.append((mh, mh_phys, t, 0))
    return out


def summarize(name, rows, n_scenes):
    mh = np.array([r[0] for r in rows])
    mh_phys = np.array([r[1] for r in rows])
    t = np.array([r[2] for r in rows])
    fb = np.array([r[3] for r in rows])
    safe = mh >= 0
    hard_safe = mh_phys >= 0
    reached = t < E.MAX_STEPS * E.DT - 1e-9
    safe_t = t[safe & reached]
    print(f"  {name:<34} margin-viol {int((~safe).sum()):>2}/{n_scenes}  "
          f"({100*(~safe).mean():>4.0f}%)   "
          f"HARD-collision {int((~hard_safe).sum()):>2}/{n_scenes}  "
          f"({100*(~hard_safe).mean():>4.0f}%)   "
          f"safe-episode t_goal {safe_t.mean() if len(safe_t) else float('nan'):>6.2f}s   "
          f"fallbacks/ep {fb.mean():>4.1f}")
    if (~safe).any() and fb.any():
        print(f"  {'':<34} fallbacks on VIOLATING eps {fb[~safe].mean():>4.1f} "
              f"vs safe eps {fb[safe].mean():>4.1f}")
    return safe, hard_safe, t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--query-ticks", type=int, default=None,
                    help="override E.QUERY_TICKS for this run (Phase 2: query-cadence test)")
    ap.add_argument("--n-scenes", type=int, default=50)
    args = ap.parse_args()

    if args.query_ticks is not None:
        print(f"[overriding QUERY_TICKS: {E.QUERY_TICKS} -> {args.query_ticks} "
              f"({args.query_ticks * E.DT:.2f}s cadence)]\n")
        E.QUERY_TICKS = args.query_ticks

    scenes = E.make_holdout_scenes(args.n_scenes)
    n = len(scenes)

    # oracle + best constant from the fixed sweep
    C = np.zeros((n, len(E.CONST_CANDIDATES)))
    MH = np.zeros((n, len(E.CONST_CANDIDATES)))
    MH_PHYS = np.zeros((n, len(E.CONST_CANDIDATES)))
    T = np.zeros((n, len(E.CONST_CANDIDATES)))
    for i, (s, g, o) in enumerate(scenes):
        for j, gam in enumerate(E.CONST_CANDIDATES):
            mh, mh_phys, t = E.run_fixed(s, g, o, float(gam))
            MH[i, j], MH_PHYS[i, j], T[i, j] = mh, mh_phys, t
            C[i, j] = E.cost(mh, t)
    jb = int(C.mean(axis=0).argmin())
    best_const = float(E.CONST_CANDIDATES[jb])

    print(f"held-out scenes: {n}   best fitted constant gamma = {best_const:.2f}\n")
    print("ARM DECOMPOSITION (violation rate vs. speed-when-safe)\n")

    summarize(f"best fitted constant ({best_const:.2f})",
              [(MH[i, jb], MH_PHYS[i, jb], T[i, jb], 0) for i in range(n)], n)

    # per-scene oracle: pick the min-cost gamma per scene
    orc = [(MH[i, int(C[i].argmin())], MH_PHYS[i, int(C[i].argmin())],
            T[i, int(C[i].argmin())], 0) for i in range(n)]
    summarize("per-scene oracle", orc, n)

    for mode in ("network", "oracle_min_h", "fully_oracle"):
        pol = E.AdaptivePolicy(E.DEFAULT_CKPT, E.DEFAULT_THRESH, gate_mode=mode)
        rows = run_arm(scenes, "adaptive", pol)
        summarize(f"adaptive [{mode}]", rows, n)

    # what does the oracle's speed actually look like vs the constant?
    print()
    safe_const = MH[:, jb] >= 0
    orc_t = np.array([r[2] for r in orc])
    orc_mh = np.array([r[0] for r in orc])
    print(f"speed headroom, safe episodes only: constant {T[safe_const, jb].mean():.2f}s "
          f"-> oracle {orc_t[orc_mh >= 0].mean():.2f}s "
          f"({100*(T[safe_const, jb].mean() - orc_t[orc_mh>=0].mean())/T[safe_const, jb].mean():.0f}% faster)")

    # how many scenes have NO safe constant at all?
    any_safe = (MH >= 0).any(axis=1)
    print(f"scenes where at least one constant gamma is safe: {int(any_safe.sum())}/{n}")
    print(f"scenes where the best-fitted constant is safe:    {int(safe_const.sum())}/{n}")


if __name__ == "__main__":
    main()
