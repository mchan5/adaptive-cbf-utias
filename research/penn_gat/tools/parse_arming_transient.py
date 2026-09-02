"""
parse_arming_transient.py

Parses Tier 4b (arming-transient stress test) SITL logs -- see
run_arming_transient_sweep.sh and TESTING.md. No obstacles, no goal, so
"overshoot" is detected directly from odometry rather than
reached_goal/hard_collision: a cycle is flagged OVERSHOOT if position
departs more than OVERSHOOT_RADIUS_M from the spawn point, or altitude (z)
goes outside [Z_MIN, Z_MAX], within the short observation window. Also
surfaces the two auxiliary signals found by hand on 2026-08-24
(integration-clamp warnings, QP-infeasible tick count) for cross-checking.

Usage:
    python parse_arming_transient.py <runs_dir>
"""
import os
import re
import sys
from collections import defaultdict

POS_RE = re.compile(r"pos=\[([-0-9.]+) ([-0-9.]+) ([-0-9.]+)\]")
OVERSHOOT_RADIUS_M = 3.0  # spawn is ~origin; a benign arming hover should never leave this
Z_MIN, Z_MAX = -0.5, 4.0  # goal altitude is 1.5m; this window is generous, not tight


def parse_log(path):
    positions = []
    infeasible_ticks = 0
    integration_clamped = False
    with open(path, errors="replace") as f:
        for line in f:
            m = POS_RE.search(line)
            if m:
                positions.append(tuple(float(g) for g in m.groups()))
            if "QP solve infeasible" in line:
                infeasible_ticks += 1
            if "integration clamped" in line:
                integration_clamped = True

    if not positions:
        return {"n_pos_ticks": 0, "overshoot": None, "max_radius": None, "max_z": None,
                "min_z": None, "infeasible_ticks": infeasible_ticks,
                "integration_clamped": integration_clamped}

    max_radius = max((x ** 2 + y ** 2) ** 0.5 for x, y, _ in positions)
    zs = [z for _, _, z in positions]
    max_z, min_z = max(zs), min(zs)
    overshoot = max_radius > OVERSHOOT_RADIUS_M or max_z > Z_MAX or min_z < Z_MIN
    return {"n_pos_ticks": len(positions), "overshoot": overshoot, "max_radius": max_radius,
            "max_z": max_z, "min_z": min_z, "infeasible_ticks": infeasible_ticks,
            "integration_clamped": integration_clamped}


def main():
    runs_dir = sys.argv[1] if len(sys.argv) > 1 else "."
    by_authority = defaultdict(list)
    for fname in sorted(os.listdir(runs_dir)):
        if not fname.endswith(".log"):
            continue
        m = re.match(r"(a[0-9_]+)_cycle_(\d+)\.log", fname)
        if not m:
            continue
        tag = m.group(1)
        result = parse_log(os.path.join(runs_dir, fname))
        by_authority[tag].append(result)
        flag = "OVERSHOOT" if result["overshoot"] else ("ok" if result["overshoot"] is False else "no-data")
        print(f"{fname}: {flag}  max_radius={result['max_radius']}  "
              f"z=[{result['min_z']},{result['max_z']}]  infeasible_ticks={result['infeasible_ticks']}  "
              f"clamp_warning={result['integration_clamped']}")

    print(f"\n{'authority':<12} {'n':>4} {'overshoot':>10} {'rate':>8}")
    print("-" * 40)
    for tag in sorted(by_authority):
        results = by_authority[tag]
        n = len(results)
        n_overshoot = sum(1 for r in results if r["overshoot"])
        rate = n_overshoot / n if n else float("nan")
        print(f"{tag:<12} {n:>4} {n_overshoot:>10} {rate:>7.1%}")


if __name__ == "__main__":
    main()
