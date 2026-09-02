"""Parses a SITL spot-check's per-episode stdout logs (produced by e.g. run_sitl_spotcheck.sh --
see results/sitl_spotcheck_20260824/MANIFEST.md) into a results.json matching the schema …"""
import json
import math
import os
import re
import sys

GOAL = (0.0, 5.5, 1.5)
GOAL_TOL = 0.5

POS_RE = re.compile(r"pos=\[([-\d.]+) ([-\d.]+) ([-\d.]+)\].*?feasible=(\d)")


def parse_log(path):
    positions = []
    any_infeasible = False
    with open(path) as f:
        for line in f:
            m = POS_RE.search(line)
            if m:
                x, y, z, feas = float(m.group(1)), float(m.group(2)), float(m.group(3)), int(m.group(4))
                positions.append((x, y, z))
                if feas == 0:
                    any_infeasible = True
    return positions, any_infeasible


def main():
    runs_dir = sys.argv[1]
    scenes_dir = sys.argv[2]
    log_prefix = sys.argv[3]

    scene_files = sorted(f for f in os.listdir(scenes_dir) if re.match(r"scene_\d+\.json$", f))
    results = []
    for scene_file in scene_files:
        idx = int(re.match(r"scene_(\d+)\.json$", scene_file).group(1))
        with open(os.path.join(scenes_dir, scene_file)) as f:
            obstacles = json.load(f)
        n_obs = len(obstacles) // 4

        log_path = os.path.join(runs_dir, f"{log_prefix}_scene_{idx:02d}.log")
        if not os.path.exists(log_path):
            print(f"scene {idx:02d}: MISSING LOG {log_path}")
            continue
        positions, any_infeasible = parse_log(log_path)
        if not positions:
            print(f"scene {idx:02d}: NO POSITION TICKS PARSED (armed/spawn failure?)")
            results.append({"scene": scene_file, "n_obstacles": n_obs, "n_log_ticks": 0,
                             "reached_goal": False, "min_clearance": None,
                             "any_infeasible": None, "final_pos": None})
            continue

        final = positions[-1]
        reached = math.dist(final, GOAL) < GOAL_TOL

        min_clearance = None
        if n_obs > 0:
            min_clearance = min(
                math.dist(p, tuple(obstacles[4 * j:4 * j + 3])) - obstacles[4 * j + 3] / 2.0
                for p in positions for j in range(n_obs))

        results.append({
            "scene": scene_file, "n_obstacles": n_obs, "n_log_ticks": len(positions),
            "reached_goal": reached, "min_clearance": min_clearance,
            "any_infeasible": any_infeasible, "final_pos": list(final),
        })
        mc = round(min_clearance, 3) if min_clearance is not None else None
        print(f"scene {idx:02d}: n_obs={n_obs} ticks={len(positions):3d} reached={reached} "
              f"min_clearance={mc} any_infeasible={any_infeasible} "
              f"final={[round(v, 2) for v in final]}")

    n = len(results)
    reached = sum(1 for r in results if r["reached_goal"])
    infeasible = sum(1 for r in results if r["any_infeasible"])
    hard = sum(1 for r in results if r["min_clearance"] is not None and r["min_clearance"] < 0)
    clearances = [r["min_clearance"] for r in results if r["min_clearance"] is not None]
    mc_str = f"{min(clearances):.3f}" if clearances else "n/a"
    print(f"\n{log_prefix}: reached_goal={reached}/{n}  hard_collisions={hard}/{n}  "
          f"infeasible={infeasible}/{n}  min_clearance={mc_str}")

    out_json = os.path.join(os.path.dirname(runs_dir.rstrip("/")), "results.json")
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote {out_json}")


if __name__ == "__main__":
    main()
