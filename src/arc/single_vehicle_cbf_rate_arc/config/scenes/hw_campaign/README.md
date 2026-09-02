# Frozen campaign scene set — `hw_campaign`

The 20 obstacle scenes for the adaptive-vs-fixed-gamma campaign
(`EXPERIMENT_GUI_PLAN_20260829.md` §4.5,
`HARDWARE_THESIS_PLAN_20260828.md` §4.3). The **same files** are used for
SITL, synthetic-obstacle hardware, and as the LiDAR ground-truth overlay,
so per-scene sim-to-real gaps are directly computable.

**Do not regenerate mid-campaign.** If a regeneration is ever needed,
bump the directory name and record it here.

## Provenance

| | |
|---|---|
| Generator | `research/penn_gat/tools/gen_sitl_obstacle_scenes.py` |
| Generator commit | `d29ee86` |
| Command | `SITL_BATTERY=1 SITL_BATTERY_SEED=20260829 SITL_BATTERY_N=20 python3 research/penn_gat/tools/gen_sitl_obstacle_scenes.py` |
| Seed | `20260829` |
| Date frozen | 2026-08-29 |
| Source dir (pre-copy) | `research/penn_gat/results/sitl_battery_20260829_20260824/` |

## Format

Each `scene_NN.json` is a flat JSON list `[x, y, z, diameter, ...]` in the
map/ENU frame, one group of 4 per obstacle. Consumed directly by both
launch files' `obstacle_file` argument (which feeds
`synthetic_obstacle_publisher`'s `obstacles` parameter). `obs_safety_margin`
is added by the QP itself — obstacle sizes here are physical, not
pre-inflated.

Corridor (fixed by the SITL spawn at ~origin and the goal at
`(0, 5.5, 1.5)`): obstacles are placed in `x ∈ [-1.4, 1.4]`,
`y ∈ [1.0, 4.8]`, `z ∈ [1.0, 1.9]`, radius `∈ [0.4, 0.8]`, 1–4 per scene.

`manifest.json` records the seed, the sampling ranges, and per-scene
obstacle counts.
