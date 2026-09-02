"""Phase 3 of the report-readiness plan (2026-08-22): the validated headline result was measured
entirely on the point-mass, sequential-projection training simulator …"""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(__file__))
import eval_scene_distribution as E  # noqa: E402

CALIBRATION_SEEDS = [20260821, 20260901, 20260902, 20260903, 20260904, 20260905]
TEST_SEEDS = [20260906, 20260907, 20260908, 20260909, 20260910, 20260911, 20260912]
DENSE_CALIBRATION_SEEDS = [20260920, 20260921, 20260922, 20260923, 20260924, 20260925]


def make_dense_holdout_scenes(n, seed):
    """Same effective-radius convention as E.make_holdout_scenes, but forces n_obs in [7, 9] and
    skips gate scenes entirely, so every scene lands in the density stratum where the …"""
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
