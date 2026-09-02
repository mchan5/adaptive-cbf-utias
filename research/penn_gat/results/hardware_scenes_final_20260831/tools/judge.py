"""Judge one SITL run log against the "strong result" bar."""
import json
import math
import re
import sys

LINE = re.compile(r"\[(\d+\.\d+)\].*?pos=\[([-\d.]+) ([-\d.]+) ([-\d.]+)\].*?feasible=(\d)")
GOAL = (0.0, 5.5, 1.5)
GOAL_TOL = 0.6

log, scene = sys.argv[1], sys.argv[2]
obs = json.load(open(scene))
nobs = len(obs) // 4

t0 = None
reached = False
t_goal = None
min_clear = math.inf
max_absx_corr = 0.0
max_absx_all = 0.0
min_y = math.inf
max_y = -math.inf
infeas = 0
n = 0
try:
    for ln in open(log, errors="replace"):
        m = LINE.search(ln)
        if not m:
            continue
        n += 1
        t = float(m.group(1))
        x, y, z = (float(m.group(i)) for i in (2, 3, 4))
        feas = int(m.group(5))
        if t0 is None:
            t0 = t
        if feas == 0:
            infeas += 1
        max_absx_all = max(max_absx_all, abs(x))
        min_y = min(min_y, y)
        max_y = max(max_y, y)
        if 1.0 < y < 5.0:
            max_absx_corr = max(max_absx_corr, abs(x))
        for k in range(nobs):
            ox, oy, oz, d = obs[4 * k:4 * k + 4]
            min_clear = min(min_clear, math.dist((x, y, z), (ox, oy, oz)) - d / 2.0)
        if not reached and math.dist((x, y, z), GOAL) < GOAL_TOL:
            reached = True
            t_goal = round(t - t0, 2)
except FileNotFoundError:
    pass

mc = None if min_clear is math.inf else round(min_clear, 3)
res = dict(
    log=log.split("/")[-1], ticks=n, reached=reached, t_goal=t_goal,
    min_clear=mc, max_absx_corridor=round(max_absx_corr, 2),
    max_absx_all=round(max_absx_all, 2),
    y_range=[None if min_y is math.inf else round(min_y, 2),
             None if max_y == -math.inf else round(max_y, 2)],
    infeasible_ticks=infeas,
)
strong = (
    n > 20 and res["reached"] and mc is not None and mc >= 0.0
    and max_absx_corr <= 2.5 and max_absx_all <= 3.5
    and min_y >= -1.0 and max_y <= 7.0 and infeas < 20
)
res["strong"] = bool(strong)
fail = []
if n <= 20:
    fail.append("no_ticks")
if not res["reached"]:
    fail.append("not_reached")
if mc is not None and mc < 0.0:
    fail.append(f"hard_collision({mc})")
if max_absx_corr > 2.5:
    fail.append(f"veer({round(max_absx_corr,2)})")
if max_absx_all > 3.5 or min_y < -1.0 or max_y > 7.0:
    fail.append("flyaway")
if infeas >= 20:
    fail.append(f"infeasible({infeas})")
res["fail"] = fail
print(json.dumps(res))
sys.exit(0 if strong else 1)
