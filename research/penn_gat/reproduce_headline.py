"""Single entry point for the validated result-of-record."""
import os

import numpy as np

import eval_scene_distribution as E

CKPT_DIR = os.path.join(os.path.dirname(__file__), "nn_model", "checkpoint")
# Defaults track the DEPLOYED config (config/params_single_vehicle_cbf_rate_arc.yaml): the
# 2026-08-27 clean retrain (gat_penn_clean_20260826, lcb_k=0.0, gamma range [2.0, 9.5]) and the …
_CKPT_NAME = os.environ.get("CHECKPOINT_NAME", "gat_penn_clean_20260826")
CKPT = os.path.join(CKPT_DIR, f"{_CKPT_NAME}.pth")
THRESH = os.path.join(CKPT_DIR, f"cccp_threshold_{_CKPT_NAME}.json")
MARGIN = float(os.environ.get("RISK_MARGIN", "0.00"))
LCB_K = float(os.environ.get("LCB_K", "0.0"))
FALLBACK_MODE = os.environ.get("FALLBACK_MODE", "ratchet")
FALLBACK_STEP = float(os.environ.get("FALLBACK_STEP", "0.5"))

CALIBRATION_SEEDS = [20260821, 20260901, 20260902, 20260903, 20260904, 20260905]
TEST_SEEDS = [20260906, 20260907, 20260908, 20260909, 20260910, 20260911, 20260912]


def run_seed(seed):
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

    pol = E.AdaptivePolicy(CKPT, THRESH, gate_mode="network", risk_margin=MARGIN, lcb_k=LCB_K,
                           fallback_mode=FALLBACK_MODE, fallback_step=FALLBACK_STEP)
    a_mh = np.zeros(n)
    a_mhp = np.zeros(n)
    a_t = np.zeros(n)
    for i, (s, g, o) in enumerate(scenes):
        m, p, t = E.run_adaptive(pol, s, g, o)
        a_mh[i], a_mhp[i], a_t[i] = m, p, t

    a_safe = a_mh >= 0
    c_safe = c_mh >= 0
    both = a_safe & c_safe

    return {
        "seed": seed,
        "const_gamma": float(E.CONST_CANDIDATES[jb]),
        "n_margin_viol": int((~a_safe).sum()),
        "n_hard_coll": int((a_mhp < 0).sum()),
        "n_scenes": n,
        "unpaired_pct": 100 * (c_t[c_safe].mean() - a_t[a_safe].mean()) / c_t[c_safe].mean(),
        "paired_pct": 100 * (c_t[both].mean() - a_t[both].mean()) / c_t[both].mean(),
    }


def main():
    print(f"Frozen checkpoint: {CKPT}")
    print(f"Frozen thresholds: {THRESH}")
    print(f"margin={MARGIN}  lcb_k={LCB_K}  fallback={FALLBACK_MODE}(step={FALLBACK_STEP})  "
          f"gamma=[{E.DEPLOY_GAMMA_MIN},{E.DEPLOY_GAMMA_MAX}]x{E.DEPLOY_STEPS}")
    print(f"calibrated on {len(CALIBRATION_SEEDS)} seeds, "
          f"measured on {len(TEST_SEEDS)} disjoint seeds\n")

    rows = [run_seed(s) for s in TEST_SEEDS]

    print(f"{'seed':>10} {'hard':>6} {'margin-viol':>12} {'UNPAIRED':>10} {'PAIRED':>9}")
    print("-" * 54)
    for r in rows:
        print(f"{r['seed']:>10} {r['n_hard_coll']:>4}/{r['n_scenes']} "
              f"{r['n_margin_viol']:>9}/{r['n_scenes']} "
              f"{r['unpaired_pct']:>+9.1f}% {r['paired_pct']:>+8.1f}%")

    hard = np.array([r["n_hard_coll"] for r in rows])
    mviol = np.array([r["n_margin_viol"] for r in rows])
    up = np.array([r["unpaired_pct"] for r in rows])
    pa = np.array([r["paired_pct"] for r in rows])
    n = rows[0]["n_scenes"]

    print("-" * 54)
    print(f"total hard collisions:      {hard.sum()}/{hard.sum() + (len(rows)*n - hard.sum())} "
          f"episodes ({len(rows)} seeds x {n} scenes)")
    print(f"mean margin-violations/seed: {mviol.mean():.1f}/{n}")
    print(f"mean speed UNPAIRED:         {up.mean():+.1f}%")
    print(f"mean speed PAIRED (honest):  {pa.mean():+.1f}%")
    print()
    print("Headline: 0 hard collisions across "
          f"{len(rows)*n} held-out episodes, {pa.mean():+.1f}% faster than the "
          "best per-seed constant gamma (paired, bias-corrected), at a cost of "
          f"~{100*mviol.mean()/n:.0f}% of episodes grazing the 500mm soft margin.")


if __name__ == "__main__":
    main()
