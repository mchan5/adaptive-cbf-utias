"""Assemble the curated 20-scene deliverable once curate.sh has filled
<final>/scenes/scene_00..19.json."""
import glob
import json
import os
import sys

import torch

sys.path.insert(0, "/home/matt/arc_2026/control-barrier-functions-arc/arc_experiment_gui")
from arc_experiment_gui.campaign import Plan  # noqa: E402

FINAL = sys.argv[1]
SCENES = os.path.join(FINAL, "scenes")
START = [0.0, 0.5, 1.5]
GOAL = [0.0, 5.5, 1.5]

files = sorted(glob.glob(os.path.join(SCENES, "scene_*.json")))
assert len(files) == 20, f"expected 20 scenes, found {len(files)}"

pt = []
for f in files:
    flat = json.load(open(f))
    obs = [flat[i:i + 4] for i in range(0, len(flat), 4)]
    pt.append({
        "start": torch.tensor(START, dtype=torch.float64),
        "goal": torch.tensor(GOAL, dtype=torch.float64),
        "obstacles": torch.tensor(
            [[o[0], o[1], o[2], o[3] / 2.0] for o in obs], dtype=torch.float64),
    })
torch.save(pt, os.path.join(FINAL, "scenes_seed20260831.pt"))

Plan.create(
    root=FINAL,
    scene_ids=list(range(20)),
    arms=["fixed_g5", "adaptive"],
    trials_per_cell=3,
    seed=20260831,
    regime="sitl",
    scene_dir=SCENES,
)

open(os.path.join(FINAL, "README.md"), "w").write(f"""# Curated hardware scene set - 2026-08-31

20 SITL-screened scenes for the fixed-vs-adaptive-gamma
`single_vehicle_cbf_rate_arc` hardware campaign. Every scene was flown in
PX4+Gazebo SITL under BOTH arms and kept only if BOTH were "strong":
reached goal (<=0.6 m), no hard collision (drone centre never inside the
0.06 m obstacle), no veer (max |x| in the corridor <= 2.5 m), no flyaway.

## Geometry (arc_experiment_gui corridor convention)

    start (0.0, 0.5, 1.5)   goal (0.0, 5.5, 1.5)   straight line 5.0 m (+y)

Obstacles: identical, 0.12 m diameter (0.06 m radius, from
config/obstacles.json), centre z = 1.5 m. obs_safety_margin 0.5 m.
12 "passable" (all pairs >= 1.15 m centre-to-centre) + 8 "reroute"
(one 0.87-1.00 m sub-margin gap). See MANIFEST.csv for per-scene stats
and the SITL t_goal / min-clearance of both arms.

## Files

* `scenes/scene_00.json .. scene_19.json` - flat [x,y,z,diameter,...],
  the config/obstacles.json contract. This is the GUI `scene_dir`.
* `plan.json` - arc_experiment_gui Plan (20 scenes x [fixed_g5, adaptive]
  x 3 trials, sitl regime). Load it from the campaign tab.
* `scenes_seed20260831.pt` - torch format for body_rate_closed_loop_diag
  and SITL replay.
* `sitl_runs/` - the raw screening logs (cand_*_fixed.log / _adaptive.log).

## Open in the GUI

    ros2 run arc_experiment_gui dashboard --ros-args -p uav_prefix:=uav_0
    # campaign tab -> Load -> {FINAL}

Config used for screening: deployed params, authority (8,4,4),
goal_arrival_tolerance raised 0.3 -> 0.6 m (kills the at-goal stall-escape
wander). fixed_g5 = penn_enabled:false + cbf_gamma_obs 5.0;
adaptive = penn_enabled:true.
""")
print(f"finalized: {FINAL}")
print(f"  scenes_seed20260831.pt ({len(pt)} scenes)")
print(f"  plan.json  (20 x [fixed_g5, adaptive] x 3, sitl)")
