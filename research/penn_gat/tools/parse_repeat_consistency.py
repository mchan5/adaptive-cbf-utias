"""Quantifies run-to-run path spread across N repeats of the SAME SITL scene (produced by
run_sitl_repeat_battery.sh), to measure whether the measured-dt fix …"""
import math
import os
import re
import sys
from itertools import combinations


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


def max_pairwise_spread(points):
    if len(points) < 2:
        return 0.0
    return max(math.dist(a, b) for a, b in combinations(points, 2))


def main():
    runs_dir = sys.argv[1]
    tag = sys.argv[2]

    logs = sorted(
        f for f in os.listdir(runs_dir)
        if re.match(rf"{re.escape(tag)}_rep_\d+\.log$", f))
    if len(logs) < 2:
        print(f"Need >=2 repeat logs for '{tag}' in {runs_dir}, found {len(logs)}")
        sys.exit(1)

    traces = []
    for log in logs:
        positions, any_infeasible = parse_log(os.path.join(runs_dir, log))
        traces.append(positions)
        print(f"{log}: {len(positions)} ticks, any_infeasible={any_infeasible}, "
              f"final={[round(v, 2) for v in positions[-1]] if positions else None}")

    n_ticks = min(len(t) for t in traces)
    if n_ticks == 0:
        print(f"\n{tag}: at least one repeat produced zero position ticks -- can't align, aborting")
        sys.exit(1)

    per_tick_spread = []
    for i in range(n_ticks):
        pts = [t[i] for t in traces]
        per_tick_spread.append(max_pairwise_spread(pts))

    max_spread = max(per_tick_spread)
    max_spread_tick = per_tick_spread.index(max_spread)
    final_positions = [t[-1] for t in traces]
    final_spread = max_pairwise_spread(final_positions)

    print(f"\n{tag}: {len(traces)} repeats, aligned over {n_ticks} common ticks")
    print(f"{tag}: max pairwise spread over all ticks = {max_spread:.3f}m (at tick {max_spread_tick})")
    print(f"{tag}: spread of each repeat's FINAL logged position = {final_spread:.3f}m")
    print(f"{tag}: per-tick spread (every 5th tick): "
          f"{[round(s, 2) for s in per_tick_spread[::5]]}")


if __name__ == "__main__":
    main()
