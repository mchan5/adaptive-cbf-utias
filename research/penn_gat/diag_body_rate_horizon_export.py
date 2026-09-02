"""
diag_body_rate_horizon_export.py

Stage 1 of the body-rate-native training-label hypothesis test (see the
2026-08-25 plan / results/frozen_authority_20260824/MANIFEST.md's sibling
adaptive-selector history). Two prior fixes to the PENN risk head's
backwards-above-gamma~4.5 predictions failed while holding the training
LABEL SOURCE fixed -- a point-mass simulator. The one remaining, never-
attempted hypothesis: the point-mass rollout shows a transient "dangerous"
reading at high gamma that real body-rate dynamics don't share.

This script samples query states EXACTLY the way worker() does (imports
drone_data_generation and calls its functions directly -- does not copy the
sampling logic, so it can't silently drift from what actually trains the
network), computes point-mass min_h_horizon via the existing
simulate_horizon() (free, no new code), and exports the same query states to
a .pt file for the new C++ tool (body_rate_horizon_label_export) to compute
body-rate-native min_h_horizon at the same states/gammas. Then joins the two
back together for a direct comparison.

Usage:
    python diag_body_rate_horizon_export.py           # sample + export query rows
    python diag_body_rate_horizon_export.py --analyze  # after the C++ tool has run, join + report
"""
import argparse
import csv
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
import drone_data_generation as G  # noqa: E402

N_STATES = 500
CANDIDATE_GAMMAS = [1.5, 3.0, 4.5, 6.0, 7.5, 9.0]
SEED = 20260825
BOUNDARY = (-1.0, 8.0, -3.0, 3.0, 0.3, 3.0)

OUT_DIR = os.path.join(os.path.dirname(__file__), "results", "body_rate_label_test_20260825")
QUERY_PATH = os.path.join(OUT_DIR, "query_rows.pt")
BODYRATE_CSV = os.path.join(OUT_DIR, "bodyrate_labels.csv")
REPORT_PATH = os.path.join(OUT_DIR, "stage1_report.csv")


def sample_query_state():
    """Mirrors worker()'s scene/carrier/query-state sampling exactly
    (drone_data_generation.py:698-797), stopping before the label call.
    Returns None for states the deployed node would never query PENN from
    (no obstacle currently sensed), matching worker()'s own skip rule."""
    if np.random.random() < G.GATE_SCENE_PROB:
        start, goal, obstacles = G.random_gate_scene()
    else:
        start, goal, obstacles = G.random_scene()

    eff_obstacles = [(c, r + G.SAFETY_MARGIN) for c, r in obstacles]
    carrier_gamma = np.random.uniform(*G.GAMMA_RANGE)
    pos = start.copy()
    vel = np.zeros(3)
    carrier_states = [(pos.copy(), vel.copy())]
    for _ in range(G.CARRIER_MAX_STEPS):
        if float(np.linalg.norm(goal - pos)) < 0.3:
            break
        a_safe, _h, _c = G.step_dynamics(
            pos, vel, goal, eff_obstacles, carrier_gamma, BOUNDARY,
            alpha_bnd=0.5, v_max=3.0, k_p=0.5, k_v=2.0, a_max=5.0, apply_cbf=True,
        )
        vel = vel + a_safe * 0.05
        pos = pos + vel * 0.05
        carrier_states.append((pos.copy(), vel.copy()))

    weights = np.array([G._decision_weight(p, G.visible_obstacles(p, eff_obstacles))
                        for p, _v in carrier_states])
    weights = weights / weights.sum()
    query_idx = np.random.choice(len(carrier_states), p=weights)
    pos0, vel0 = carrier_states[query_idx]

    if np.random.random() < 0.5:
        err0 = goal - pos0
        dist0 = float(np.linalg.norm(err0))
        goal_dir = err0 / dist0 if dist0 > 1e-6 else np.zeros(3)
        speed = np.random.uniform(0.0, 3.0)
        direction = goal_dir + np.random.normal(0.0, 0.15, size=3)
        dir_norm = float(np.linalg.norm(direction))
        direction = direction / dir_norm if dir_norm > 1e-6 else goal_dir
        vel0 = direction * speed

    if not G.visible_obstacles(pos0, obstacles):
        return None

    return pos0, vel0, goal, obstacles, eff_obstacles


def export_query_rows():
    np.random.seed(SEED)
    os.makedirs(OUT_DIR, exist_ok=True)
    rows = []
    pointmass_min_h = []  # list of lists, [row][gamma_idx]
    while len(rows) < N_STATES:
        r = sample_query_state()
        if r is None:
            continue
        pos0, vel0, goal, obstacles, eff_obstacles = r
        row_id = len(rows)
        pm_row = []
        for g in CANDIDATE_GAMMAS:
            _risk, _pd, _fc, min_h = G.simulate_horizon(
                pos0, vel0, goal, eff_obstacles, g,
                boundary=BOUNDARY, horizon_s=G.HORIZON_S, apply_cbf=True,
            )
            pm_row.append(min_h)
        pointmass_min_h.append(pm_row)
        rows.append({
            "row_id": int(row_id),
            "pos0": torch.tensor(pos0, dtype=torch.float64),
            "vel0": torch.tensor(vel0, dtype=torch.float64),
            "goal": torch.tensor(goal, dtype=torch.float64),
            "obstacles": torch.tensor(
                [[c[0], c[1], c[2], r] for c, r in obstacles], dtype=torch.float64),
            "gammas": torch.tensor(CANDIDATE_GAMMAS, dtype=torch.float64),
        })
        if len(rows) % 100 == 0:
            print(f"  sampled {len(rows)}/{N_STATES}")

    torch.save(rows, QUERY_PATH)
    np.save(os.path.join(OUT_DIR, "pointmass_min_h.npy"), np.array(pointmass_min_h))
    print(f"Wrote {len(rows)} query rows to {QUERY_PATH}")
    print(f"Wrote point-mass labels to {OUT_DIR}/pointmass_min_h.npy")
    print(f"\nNext: run the C++ tool against {QUERY_PATH}, writing to {BODYRATE_CSV}, "
          f"then rerun this script with --analyze")


def analyze():
    pm = np.load(os.path.join(OUT_DIR, "pointmass_min_h.npy"))  # [n_rows, n_gammas]
    n_rows = pm.shape[0]
    bodyrate = {g: [None] * n_rows for g in CANDIDATE_GAMMAS}
    with open(BODYRATE_CSV) as f:
        for row in csv.DictReader(f):
            rid = int(row["row_id"])
            g = float(row["gamma"])
            bodyrate[min(CANDIDATE_GAMMAS, key=lambda x: abs(x - g))][rid] = float(row["min_h_horizon"])

    def backwards_frac(get_val, lo_gamma=4.5, hi_gamma=9.0):
        n, n_backwards = 0, 0
        for i in range(n_rows):
            lo = get_val(i, lo_gamma)
            hi = get_val(i, hi_gamma)
            if lo is None or hi is None:
                continue
            n += 1
            if hi < lo:
                n_backwards += 1
        return n_backwards, n

    gi = {g: idx for idx, g in enumerate(CANDIDATE_GAMMAS)}
    pm_back, pm_n = backwards_frac(lambda i, g: pm[i, gi[g]])
    br_back, br_n = backwards_frac(lambda i, g: bodyrate[g][i])

    with open(REPORT_PATH, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["source", "gamma", "mean_min_h", "median_min_h"])
        for g in CANDIDATE_GAMMAS:
            vals = pm[:, gi[g]]
            w.writerow(["pointmass", g, f"{vals.mean():.4f}", f"{np.median(vals):.4f}"])
        for g in CANDIDATE_GAMMAS:
            vals = np.array([v for v in bodyrate[g] if v is not None])
            w.writerow(["bodyrate", g, f"{vals.mean():.4f}" if len(vals) else "",
                        f"{np.median(vals):.4f}" if len(vals) else ""])

    print(f"\n=== Stage 1 result (n={n_rows} query states) ===")
    print(f"Point-mass:  min_h(9.0) < min_h(4.5) on {pm_back}/{pm_n} states "
          f"({100*pm_back/pm_n:.1f}%) -- backwards rate")
    print(f"Body-rate:   min_h(9.0) < min_h(4.5) on {br_back}/{br_n} states "
          f"({100*br_back/br_n:.1f}%) -- backwards rate")
    print(f"\nPer-gamma means:")
    for g in CANDIDATE_GAMMAS:
        pm_mean = pm[:, gi[g]].mean()
        br_vals = np.array([v for v in bodyrate[g] if v is not None])
        br_mean = br_vals.mean() if len(br_vals) else float("nan")
        print(f"  gamma={g:4.1f}  pointmass_mean_min_h={pm_mean:8.3f}  bodyrate_mean_min_h={br_mean:8.3f}")
    print(f"\nFull report: {REPORT_PATH}")

    if pm_back / max(pm_n, 1) < 0.30:
        print("\nWARNING: point-mass backwards rate on THIS harness/grid is well below the "
              "documented ~51% baseline -- treat this run as an inconclusive sanity-check "
              "failure, not evidence, per the plan's stated gate.")
    elif br_back / max(br_n, 1) < pm_back / max(pm_n, 1) * 0.5:
        print("\nGO: body-rate backwards rate is substantially lower than point-mass on the "
              "same states -- hypothesis supported. Proceed to Stage 2.")
    else:
        print("\nSTOP: body-rate shows a comparable backwards rate to point-mass -- the "
              "effect is likely in the shared HOCBF math, not a point-mass-vs-body-rate "
              "dynamics artifact. Do not proceed to Stage 2 on this basis.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--analyze", action="store_true")
    args = ap.parse_args()
    if args.analyze:
        analyze()
    else:
        export_query_rows()
