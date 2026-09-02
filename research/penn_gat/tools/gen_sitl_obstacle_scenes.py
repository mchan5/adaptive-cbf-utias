"""
gen_sitl_obstacle_scenes.py

Phase C of the 2026-08-22 aleatoric-head-fix plan: obstacle-only scenes for
the Gazebo/PX4 SITL spot-check, compatible with
`cbf_rate_arc_sitl_test_launch.py`'s `obstacle_file` override (flat
[x,y,z,diameter,...] JSON, see that file's `_launch_obstacle_publisher`).

The SITL launch's spawn (PX4 gz_x500 default, ~origin) and waypoint
(hardcoded (0, 5.5, 1.5) in `ground_station_stub_node`) are NOT wired
through as launch args (see the paper's Future Work) -- scenes here are
therefore obstacle-only and placed along that fixed corridor, not full
{start, goal, obstacles} scenes like the point-mass/closed-loop pipelines
use. Diameter = 2*radius_physical; SAFETY_MARGIN is added by the deployed
QP itself (`obs_safety_margin` in the sim YAML), matching how
`synthetic_obstacle_publisher_node.py` and `handleObstacles()` already
treat obstacle_file input -- do not pre-inflate here.

Extended 2026-08-24 (see penn_gat_training/TESTING.md, Tier 4/4b) beyond the
original hardcoded 8-scene/one-seed spot-check set:

Modes (mirrors diag_gen_closed_loop_scenes.py's env-var convention):
  - Default (no env vars): the original 2026-08-22 8-scene spot-check,
    unchanged (seed 20261005, results/sitl_spotcheck_20260822/) -- kept as
    the cheap sanity/regression re-check, NOT a validation claim (n=8 gives
    ~37.5% upper-bound on true failure rate per the rule of three; see
    TESTING.md).
  - SITL_BATTERY=1: a properly-sized Tier 4 validation battery.
    SITL_BATTERY_SEED (default 20260824, disjoint from 20261005) and
    SITL_BATTERY_N (default 50, the TESTING.md Tier-4 minimum for a
    "validated" claim) control seed/count. Writes to
    results/sitl_battery_<seed>_<date>/.
  - SITL_EMPTY=1: Tier 4b's obstacle-free scenes (arming-transient stress
    test -- see TESTING.md). No obstacle randomization possible with zero
    obstacles, so this just writes SITL_EMPTY_N (default 5) identical empty
    `[]` scene files to results/sitl_empty_20260824/ for
    tools/run_arming_transient_sweep.sh to cycle through.

Usage:
    python gen_sitl_obstacle_scenes.py                  # original 8-scene set
    SITL_BATTERY=1 python gen_sitl_obstacle_scenes.py    # n=50 validation battery
    SITL_EMPTY=1 python gen_sitl_obstacle_scenes.py      # Tier 4b empty scenes
"""
import json
import os

import numpy as np

N_SCENES = 8
SEED = 20261005  # disjoint from every other seed used this session
OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "results", "sitl_spotcheck_20260822")

# Corridor: start ~(0,0,~0.1) (fixed by PX4 gz_x500), goal (0, 5.5, 1.5)
# (fixed in cbf_rate_arc_sitl_test_launch.py). Obstacles placed between
# y=1.0 and y=4.8 (clear of both spawn and the 0.3m goal-arrival tolerance
# the closed-loop harness uses), x in a band the vehicle must actually
# maneuver around, z near flight altitude.
Y_RANGE = (1.0, 4.8)
X_RANGE = (-1.4, 1.4)
Z_RANGE = (1.0, 1.9)
RADIUS_RANGE = (0.4, 0.8)  # matches the training distribution's obstacle scale
N_OBS_RANGE = (1, 4)  # kept small -- SITL has no simulated-clock speedup, and a
                      # sparser scene is a fairer first test of the pipeline
                      # than immediately reproducing the closed-loop campaign's
                      # up-to-8-obstacle density


def gen_scene(rng):
    n_obs = int(rng.integers(N_OBS_RANGE[0], N_OBS_RANGE[1] + 1))
    obstacles = []
    for _ in range(n_obs):
        x = rng.uniform(*X_RANGE)
        y = rng.uniform(*Y_RANGE)
        z = rng.uniform(*Z_RANGE)
        r = rng.uniform(*RADIUS_RANGE)
        obstacles.append((x, y, z, r))
    return obstacles


def write_scene_set(out_dir, seed, n_scenes):
    rng = np.random.default_rng(seed)
    os.makedirs(out_dir, exist_ok=True)
    manifest = []
    for i in range(n_scenes):
        obstacles = gen_scene(rng)
        flat = []
        for x, y, z, r in obstacles:
            flat.extend([float(x), float(y), float(z), float(2 * r)])
        path = os.path.join(out_dir, f"scene_{i:02d}.json")
        with open(path, "w") as f:
            json.dump(flat, f)
        manifest.append({"scene": i, "n_obstacles": len(obstacles), "path": os.path.basename(path)})
        print(f"scene {i:02d}: {len(obstacles)} obstacles -> {path}")

    with open(os.path.join(out_dir, "manifest.json"), "w") as f:
        json.dump({"seed": seed, "y_range": Y_RANGE, "x_range": X_RANGE,
                   "z_range": Z_RANGE, "radius_range": RADIUS_RANGE,
                   "scenes": manifest}, f, indent=2)
    print(f"\nWrote {n_scenes} scenes to {out_dir}")


def write_empty_scene_set(out_dir, n_scenes):
    os.makedirs(out_dir, exist_ok=True)
    for i in range(n_scenes):
        path = os.path.join(out_dir, f"scene_{i:02d}.json")
        with open(path, "w") as f:
            json.dump([], f)
    with open(os.path.join(out_dir, "manifest.json"), "w") as f:
        json.dump({"empty": True, "n_scenes": n_scenes}, f, indent=2)
    print(f"Wrote {n_scenes} empty (obstacle-free) scenes to {out_dir}")


def main():
    if os.environ.get("SITL_EMPTY", "0") == "1":
        n = int(os.environ.get("SITL_EMPTY_N", "5"))
        out_dir = os.path.join(os.path.dirname(__file__), "..", "results", "sitl_empty_20260824")
        write_empty_scene_set(out_dir, n)
        return

    if os.environ.get("SITL_BATTERY", "0") == "1":
        seed = int(os.environ.get("SITL_BATTERY_SEED", "20260824"))
        n = int(os.environ.get("SITL_BATTERY_N", "50"))
        out_dir = os.path.join(os.path.dirname(__file__), "..", "results",
                               f"sitl_battery_{seed}_20260824")
        write_scene_set(out_dir, seed, n)
        return

    write_scene_set(OUT_DIR, SEED, N_SCENES)


if __name__ == "__main__":
    main()
