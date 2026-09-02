"""
diag_gen_closed_loop_scenes.py

Phase 3 of the report-readiness plan (2026-08-22): the validated headline
result was measured entirely on the point-mass, sequential-projection
training simulator (drone_data_generation.py::step_dynamics). The deployed
controller (body_rate_hocbf.cpp + body_rate_qp_osqp.cpp) is a full 12-state
body-rate model solved as one joint QP -- architecturally different, never
tested. This script exports scenes for a closed-loop C++ harness
(body_rate_closed_loop_diag.cpp) to replay against the real controller, so
that question can get a direct empirical answer instead of staying an
assumption.

Three modes:
  - Default (no args): the original 8-scene single-file diagnostic (seed
    20260906, results/frozen_20260822/closed_loop_diag_scenes.pt) -- kept
    as the cheap sanity gate re-run after every checkpoint change (see
    penn.py's log_std_max comment for the most recent one).
  - CALIBRATION=1: closed-loop scenes from CALIBRATION_SEEDS, for TUNING
    closed-loop-specific parameters (currently lcb_k). Added 2026-08-23
    after the first lcb_k closed-loop sweep was run on TEST_SEEDS scenes by
    mistake -- that made lcb_k in-sample to the very seeds
    reproduce_headline.py and the campaign both measure on, which breaks
    this project's calibrate-on-one-set/measure-on-a-disjoint-one
    discipline. Anything that SELECTS a parameter must use this mode; only
    the final measurement uses CAMPAIGN=1.
  - CAMPAIGN=1: the seeded closed-loop campaign MEASUREMENT. Loops over
    TEST_SEEDS (the same 7 seeds reproduce_headline.py measures on,
    disjoint from CALIBRATION_SEEDS), N_SCENES=50 each (matches the
    point-mass convention), one file per seed under
    results/closed_loop_campaign_<date>/. Never sweep on these.
  - DENSE=1: stratified-dense CALIBRATION scenes, forced to n_obs in [7, 9]
    (no gate scenes, which have a different obstacle-count distribution).
    Added 2026-08-23 after a controller-tuning change (nom_accel_max/kp)
    looked dominant on the standard 6 CALIBRATION_SEEDS (55 dense scenes
    total) but failed on TEST_SEEDS specifically in dense clutter -- 24.5%
    deadlock at 7+ obstacles vs 9.1% calibration had estimated, driven by
    genuine QP infeasibility the 55-scene sample never exercised. Uses
    DENSE_CALIBRATION_SEEDS, disjoint from both CALIBRATION_SEEDS and
    TEST_SEEDS, so nothing already run is contaminated. Use this whenever a
    change plausibly trades against actuator margin (controller gains,
    thrust/omega limits, obs_d_min) -- gamma/lcb_k selection itself doesn't
    need it, since neither interacts with actuator limits the same way.

Saves a torch-picklable list of scene dicts: start, goal, obstacles (each
[cx, cy, cz, radius_physical] -- NO safety margin baked in.

make_holdout_scenes() returns EFFECTIVE radii (physical + SAFETY_MARGIN --
see its "eff = [(c, r + G.SAFETY_MARGIN) for c, r in obstacles]" line), the
convention every eval_scene_distribution.py function expects. But
handleObstacles() on the C++ side reads marker.scale.x as raw physical
diameter and adds obs_safety_margin itself (radius = scale.x/2 + margin),
matching what synthetic_obstacle_publisher_node.py actually publishes.
Feeding it an already-margined radius would double-apply the margin -- so
this script subtracts SAFETY_MARGIN back out before saving, same as
eval_scene_distribution._min_h_physical() does for physical-collision
checks.
"""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(__file__))
import eval_scene_distribution as E  # noqa: E402

CALIBRATION_SEEDS = [20260821, 20260901, 20260902, 20260903, 20260904, 20260905]
TEST_SEEDS = [20260906, 20260907, 20260908, 20260909, 20260910, 20260911, 20260912]
DENSE_CALIBRATION_SEEDS = [20260920, 20260921, 20260922, 20260923, 20260924, 20260925]


def make_dense_holdout_scenes(n, seed):
    """Same effective-radius convention as E.make_holdout_scenes, but forces
    n_obs in [7, 9] and skips gate scenes entirely, so every scene lands in
    the density stratum where the 2026-08-23 controller-tuning failure
    concentrated. Not a replacement for make_holdout_scenes's mixed
    distribution -- deliberately narrower, for stress-testing actuator
    margin specifically."""
    import numpy as np
    rng_state = np.random.get_state()
    np.random.seed(seed)
    scenes = []
    while len(scenes) < n:
        start, goal, obstacles = E.G.random_scene(n_obs_range=(7, 9), tight_prob=0.7)
        if obstacles:
            eff = [(c, r + E.G.SAFETY_MARGIN) for c, r in obstacles]
            scenes.append((start, goal, eff))
    np.random.set_state(rng_state)
    return scenes


def write_scene_file(out_path, seed, n_scenes, dense=False):
    scenes = (make_dense_holdout_scenes(n_scenes, seed=seed) if dense
              else E.make_holdout_scenes(n_scenes, seed=seed))
    out = []
    for start, goal, obstacles in scenes:
        out.append({
            "start": torch.tensor(start, dtype=torch.float64),
            "goal": torch.tensor(goal, dtype=torch.float64),
            "obstacles": torch.tensor(
                [[c[0], c[1], c[2], r - E.G.SAFETY_MARGIN] for c, r in obstacles],
                dtype=torch.float64),
        })
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    torch.save(out, out_path)
    print(f"Wrote {len(out)} scenes (seed={seed}) to {out_path}")
    return out


def main():
    if os.environ.get("DENSE", "0") == "1":
        out_dir = os.path.join(os.path.dirname(__file__), "results",
                               "closed_loop_calibration_dense_20260823")
        for seed in DENSE_CALIBRATION_SEEDS:
            scenes_out = write_scene_file(os.path.join(out_dir, f"scenes_seed{seed}.pt"),
                                          seed, n_scenes=50, dense=True)
            n_obs = [d["obstacles"].shape[0] for d in scenes_out]
            print(f"  seed {seed}: n_obs min={min(n_obs)} max={max(n_obs)} "
                  f"mean={sum(n_obs)/len(n_obs):.1f}")
        return

    if os.environ.get("CALIBRATION", "0") == "1":
        out_dir = os.path.join(os.path.dirname(__file__), "results",
                               "closed_loop_calibration_20260823")
        for seed in CALIBRATION_SEEDS:
            write_scene_file(os.path.join(out_dir, f"scenes_seed{seed}.pt"), seed, n_scenes=50)
        return

    if os.environ.get("CAMPAIGN", "0") == "1":
        out_dir = os.path.join(os.path.dirname(__file__), "results", "closed_loop_campaign_20260822")
        for seed in TEST_SEEDS:
            write_scene_file(os.path.join(out_dir, f"scenes_seed{seed}.pt"), seed, n_scenes=50)
        return

    out_path = os.path.join(os.path.dirname(__file__), "results", "frozen_20260822",
                            "closed_loop_diag_scenes.pt")
    scenes_out = write_scene_file(out_path, seed=20260906, n_scenes=8)
    for i, d in enumerate(scenes_out):
        print(f"  scene {i}: start={d['start'].round(decimals=2)} goal={d['goal'].round(decimals=2)} "
              f"n_obstacles={d['obstacles'].shape[0]}")


if __name__ == "__main__":
    main()
