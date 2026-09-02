"""
gen_hardware_scenes_20260831.py

Generate the 20-scene hardware test set for the fixed-vs-adaptive-gamma
single_vehicle_cbf_rate_arc campaign.

Fixed decisions (2026-08-31, with the project owner):
  * Fixed start / goal for all 20 scenes, chosen to sit clear of the arena
    net and to force a diagonal crossing of the obstacle band.
  * Every obstacle is IDENTICAL, matching config/obstacles.json:
    physical radius 0.06 m (0.12 m diameter), centre height z = OBST_Z.
  * obs_safety_margin = 0.5 m (deployed YAML) => each obstacle carries a
    ~0.56 m keep-out bubble. A physically-narrow gap is therefore a
    reroute / hold test, not a fly-through test.
  * Split: 12 "passable" scenes  (every obstacle pair >= PASS_GAP centre-to
    -centre, i.e. the controller can fly between them) and 8 "reroute"
    scenes (at least one pair in [REROUTE_LO, REROUTE_HI] centre-to-centre
    -- sub-margin, must be routed around; all other pairs still passable).
  * Obstacle-count mix across the 20: 5x2, 6x3, 5x4, 3x5, 1x6 (mean 3.5).
    Reroute scenes are drawn only from n_obs >= 3.

Outputs (under results/hardware_scenes_20260831/):
  * scenes_seed20260831.pt   -- campaign format for body_rate_closed_loop_diag
                                and the SITL replay harness. Obstacle tensor
                                rows are [cx, cy, cz, radius_phys] (NO margin
                                baked in -- matches loadScenes()).
  * hardware/scene_XX.json    -- flat [x, y, z, diameter, ...] per scene, the
                                exact contract config/obstacles.json uses
                                (synthetic_obstacle_publisher reads field 4
                                as DIAMETER and adds obs_safety_margin itself).
  * MANIFEST.csv             -- id, class, n_obs, and the three clearance
                                stats that every scene was checked against.

Deterministic: master seed 20260831, one derived seed per scene.
"""
import csv
import json
import os

import numpy as np
import torch

HERE = os.path.dirname(__file__)
OUT_DIR = os.path.join(HERE, "results", "hardware_scenes_20260831")

# geometry
MASTER_SEED = 20260831
START = np.array([-2.0, 0.5, 1.0])
GOAL = np.array([2.0, 5.5, 1.0])
OBST_Z = 1.0                     # obstacle centre height == flight altitude
RADIUS_PHYS = 0.06              # matches config/obstacles.json (0.12 m diameter)
DIAMETER = 2.0 * RADIUS_PHYS
SAFETY_MARGIN = 0.5             # deployed obs_safety_margin

# obstacle centres live in this box (keeps them off the net and away from
# the takeoff / arrival zones)
OBST_X = (-2.0, 2.0)
OBST_Y = (1.3, 4.7)

PASS_GAP = 1.15                 # centre-to-centre >= this  => flyable corridor
REROUTE_LO, REROUTE_HI = 0.87, 1.00   # a deliberately sub-margin gap (0.75-0.88 m physical)
START_GOAL_CLEAR = 1.25        # every obstacle centre this far from start & goal
SEG_BLOCK = 0.55               # >=1 obstacle within this of the straight path
                               # (scene must actually obstruct the direct line)

N_OBS_MIX = [2] * 5 + [3] * 6 + [4] * 5 + [5] * 3 + [6] * 1
# indices into the (sorted) mix that become reroute scenes: 3x n=3, 3x n=4,
# 1x n=5, 1x n=6  -> 8 total, all n_obs >= 3
REROUTE_SLOTS = {5, 6, 7, 11, 12, 13, 16, 19}


def seg_clearance(p, a, b):
    """Min distance from point p to segment a-b."""
    ab = b - a
    t = np.clip(np.dot(p - a, ab) / np.dot(ab, ab), 0.0, 1.0)
    return np.linalg.norm(p - (a + t * ab))


def try_scene(rng, n_obs, want_reroute):
    """One rejection-sampling attempt. Returns (obstacles Nx3, stats) or None."""
    obs = []
    for _ in range(n_obs):
        for _try in range(200):
            c = np.array([rng.uniform(*OBST_X), rng.uniform(*OBST_Y), OBST_Z])
            if np.linalg.norm(c - START) < START_GOAL_CLEAR:
                continue
            if np.linalg.norm(c - GOAL) < START_GOAL_CLEAR:
                continue
            # never let two obstacles physically overlap or sit absurdly close
            if any(np.linalg.norm(c - o) < REROUTE_LO for o in obs):
                continue
            obs.append(c)
            break
        else:
            return None
    if len(obs) != n_obs:
        return None
    obs = np.array(obs)

    # pairwise centre distances
    pd = [np.linalg.norm(obs[i] - obs[j])
          for i in range(n_obs) for j in range(i + 1, n_obs)]
    min_pair = min(pd) if pd else float("inf")

    # straight-path clearance
    seg_clr = min(seg_clearance(o, START, GOAL) for o in obs)
    if seg_clr > SEG_BLOCK:
        return None  # direct line not obstructed -> trivial scene

    if want_reroute:
        has_gap = any(REROUTE_LO <= d <= REROUTE_HI for d in pd)
        others_ok = all(d >= PASS_GAP for d in pd if not (REROUTE_LO <= d <= REROUTE_HI))
        if not (has_gap and others_ok):
            return None
    else:
        if min_pair < PASS_GAP:
            return None

    stats = dict(min_pair=min_pair, seg_clr=seg_clr,
                 sg_clr=min(min(np.linalg.norm(o - START), np.linalg.norm(o - GOAL))
                            for o in obs))
    return obs, stats


def main():
    os.makedirs(os.path.join(OUT_DIR, "hardware"), exist_ok=True)
    scenes_pt = []
    manifest = []

    for idx, n_obs in enumerate(N_OBS_MIX):
        want_reroute = idx in REROUTE_SLOTS
        rng = np.random.default_rng(MASTER_SEED + 1000 * idx)
        for attempt in range(20000):
            got = try_scene(rng, n_obs, want_reroute)
            if got is not None:
                break
        else:
            raise RuntimeError(f"scene {idx} (n_obs={n_obs}, reroute={want_reroute}) "
                               "did not converge -- loosen constraints")
        obs, stats = got
        cls = "reroute" if want_reroute else "passable"

        scenes_pt.append({
            "start": torch.tensor(START, dtype=torch.float64),
            "goal": torch.tensor(GOAL, dtype=torch.float64),
            "obstacles": torch.tensor(
                [[o[0], o[1], o[2], RADIUS_PHYS] for o in obs], dtype=torch.float64),
        })

        flat = []
        for o in obs:
            flat += [round(float(o[0]), 4), round(float(o[1]), 4),
                     round(float(o[2]), 4), round(DIAMETER, 4)]
        with open(os.path.join(OUT_DIR, "hardware", f"scene_{idx:02d}.json"), "w") as f:
            json.dump(flat, f)

        manifest.append(dict(
            id=f"scene_{idx:02d}", cls=cls, n_obs=n_obs,
            min_pair_ctr_dist=round(stats["min_pair"], 3),
            min_seg_clearance=round(stats["seg_clr"], 3),
            min_start_goal_clearance=round(stats["sg_clr"], 3),
            attempts=attempt + 1))

    torch.save(scenes_pt, os.path.join(OUT_DIR, "scenes_seed20260831.pt"))

    with open(os.path.join(OUT_DIR, "MANIFEST.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(manifest[0].keys()))
        w.writeheader()
        w.writerows(manifest)

    n_re = sum(1 for m in manifest if m["cls"] == "reroute")
    print(f"wrote {len(scenes_pt)} scenes to {OUT_DIR}")
    print(f"  passable={len(manifest) - n_re}  reroute={n_re}")
    print(f"  start={START.tolist()}  goal={GOAL.tolist()}  obst_z={OBST_Z}")
    print(f"  obstacle: r_phys={RADIUS_PHYS} m (d={DIAMETER} m), all identical")
    for m in manifest:
        print(f"  {m['id']}  {m['cls']:8s} n={m['n_obs']}  "
              f"min_pair={m['min_pair_ctr_dist']:.2f}  seg_clr={m['min_seg_clearance']:.2f}  "
              f"sg_clr={m['min_start_goal_clearance']:.2f}")


if __name__ == "__main__":
    main()
