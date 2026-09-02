"""Candidate-scene pool for the curated 20-scene hardware set."""
import csv
import json
import os
import sys

import numpy as np

OUT = sys.argv[1]
N_PASS = int(sys.argv[2])
N_RE = int(sys.argv[3])
SEED_BASE = int(sys.argv[4])

START = np.array([0.0, 0.5, 1.5])
GOAL = np.array([0.0, 5.5, 1.5])
OBST_Z = 1.5
RADIUS_PHYS = 0.06
DIAMETER = 0.12

OBST_X = (-2.0, 2.0)
OBST_Y = (1.3, 4.7)
PASS_GAP = 1.15
REROUTE_LO, REROUTE_HI = 0.87, 1.00
START_GOAL_CLEAR = 1.25
SEG_BLOCK = 0.55

# obstacle-count mix, scaled: base 5x2,6x3,5x4,3x5,1x6 -> weight by class need
COUNT_MIX_PASS = [2, 2, 2, 2, 2, 3, 3, 3, 4, 4, 5, 5]          # 12
COUNT_MIX_RE = [3, 3, 3, 4, 4, 4, 5, 6]                         # 8


def seg_clearance(p, a, b):
    ab = b - a
    t = np.clip(np.dot(p - a, ab) / np.dot(ab, ab), 0.0, 1.0)
    return np.linalg.norm(p - (a + t * ab))


def try_scene(rng, n_obs, want_reroute):
    obs = []
    for _ in range(n_obs):
        for _t in range(300):
            c = np.array([rng.uniform(*OBST_X), rng.uniform(*OBST_Y), OBST_Z])
            if np.linalg.norm(c - START) < START_GOAL_CLEAR:
                continue
            if np.linalg.norm(c - GOAL) < START_GOAL_CLEAR:
                continue
            if any(np.linalg.norm(c - o) < REROUTE_LO for o in obs):
                continue
            obs.append(c)
            break
        else:
            return None
    if len(obs) != n_obs:
        return None
    obs = np.array(obs)
    pd = [np.linalg.norm(obs[i] - obs[j])
          for i in range(n_obs) for j in range(i + 1, n_obs)]
    min_pair = min(pd) if pd else float("inf")
    seg_clr = min(seg_clearance(o, START, GOAL) for o in obs)
    if seg_clr > SEG_BLOCK:
        return None
    if want_reroute:
        has_gap = any(REROUTE_LO <= d <= REROUTE_HI for d in pd)
        others_ok = all(d >= PASS_GAP for d in pd if not (REROUTE_LO <= d <= REROUTE_HI))
        if not (has_gap and others_ok):
            return None
    else:
        if min_pair < PASS_GAP:
            return None
    return obs, dict(min_pair=round(min_pair, 3), seg_clr=round(seg_clr, 3))


def main():
    os.makedirs(OUT, exist_ok=True)
    rows = []
    cid = 0
    for cls, mix, n_want in (("passable", COUNT_MIX_PASS, N_PASS),
                             ("reroute", COUNT_MIX_RE, N_RE)):
        made = 0
        k = 0
        while made < n_want:
            n_obs = mix[k % len(mix)]
            k += 1
            rng = np.random.default_rng(SEED_BASE + 7919 * cid + 17 * k)
            got = None
            for _ in range(40000):
                got = try_scene(rng, n_obs, cls == "reroute")
                if got:
                    break
            if not got:
                continue
            obs, st = got
            flat = []
            for o in obs:
                flat += [round(float(o[0]), 4), round(float(o[1]), 4),
                         round(float(o[2]), 4), DIAMETER]
            with open(os.path.join(OUT, f"cand_{cid:03d}.json"), "w") as f:
                json.dump(flat, f)
            rows.append(dict(cand=f"cand_{cid:03d}", cls=cls, n_obs=n_obs,
                             min_pair_ctr=st["min_pair"], seg_clr=st["seg_clr"]))
            cid += 1
            made += 1
    with open(os.path.join(OUT, "pool_manifest.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"pool: {len(rows)} candidates -> {OUT}")
    for r in rows:
        print(f"  {r['cand']} {r['cls']:8s} n={r['n_obs']} "
              f"min_pair={r['min_pair_ctr']} seg_clr={r['seg_clr']}")


if __name__ == "__main__":
    main()
