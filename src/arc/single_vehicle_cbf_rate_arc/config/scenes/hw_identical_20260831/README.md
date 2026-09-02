# Frozen campaign scene set — `hw_identical_20260831`

The 20 obstacle scenes for the adaptive-vs-fixed-gamma campaign
(`EXPERIMENT_GUI_PLAN_20260829.md` §4.5,
`HARDWARE_THESIS_PLAN_20260828.md` §4.3). The **same files** are used for
SITL, synthetic-obstacle hardware, and as the LiDAR ground-truth overlay,
so per-scene sim-to-real gaps are directly computable.

Supersedes `../hw_campaign`, whose obstacles were mixed-radius spheres
(0.4–0.8 m) that no physical obstacle matches. Here every obstacle is
**identical** and sized to the real thing: a 0.12 m diameter floor
cylinder, which the QP's default spherical barrier treats as a 0.12 m
sphere at the flight altitude. Cylinder and sphere therefore agree in the
horizontal plane, so a scene means the same geometry whether it is flown
against `cbf_cylinder_barrier: false` (sphere) or `true` (vertical
cylinder) — the sim-to-real comparison no longer depends on which branch
is active.

**Do not regenerate mid-campaign.** If a regeneration is ever needed,
bump the directory name and record it here.

## Provenance

| | |
|---|---|
| Generator | `research/penn_gat/gen_hardware_scenes_20260831.py` |
| Seed | `20260831` |
| Date frozen | 2026-08-31 |
| Source dir (pre-copy) | `research/penn_gat/results/hardware_scenes_final_20260831/scenes/` |

Every scene was flown in PX4+Gazebo SITL under **both** arms and kept only
if both were "strong": reached goal (≤0.6 m), no hard collision (drone
centre never inside the 0.06 m obstacle), no veer (max |x| in the corridor
≤ 2.5 m), no flyaway. Per-scene screening stats (t_goal, min clearance per
arm) are in the source dir's `MANIFEST.csv`; the raw screening logs are in
its `sitl_runs/`.

## Format

Each `scene_NN.json` is a flat JSON list `[x, y, z, diameter, ...]` in the
map/ENU frame, one group of 4 per obstacle — the `config/obstacles.json`
contract. Consumed directly by both launch files' `obstacle_file` argument
(which feeds `synthetic_obstacle_publisher`'s `obstacles` parameter).
`obs_safety_margin` (0.5 m) is added by the QP itself — obstacle sizes here
are physical, not pre-inflated.

Corridor: start `(0.0, 0.5, 1.5)` → goal `(0.0, 5.5, 1.5)`, a straight 5.0 m
run along +y. Obstacle centres lie in `x ∈ [-1.83, 1.93]`,
`y ∈ [1.30, 4.66]`, all at `z = 1.5` with diameter `0.12`. 2–6 obstacles per
scene: 12 **passable** (all pairs ≥ 1.15 m centre-to-centre) + 8 **reroute**
(one deliberately sub-margin 0.87–1.00 m gap).

`manifest.json` records the seed, the corridor and obstacle geometry, and
each scene's class and obstacle count.
