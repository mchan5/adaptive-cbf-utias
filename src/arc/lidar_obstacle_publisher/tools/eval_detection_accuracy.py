#!/usr/bin/env python3
"""Detected-vs-surveyed accuracy report for the LiDAR obstacle pipeline.

Replays a recorded bench bag through LidarObstaclePublisherNode (the same path
as test/test_bag_replay.py) and, instead of pass/fail, prints per-obstacle
detection rate, centroid error, radius, and ID stability, plus a spurious-marker
rate. This is the number to put in a report: "mean XY error X cm, Y% detection,
0 ID switches over N frames".

Usage (in a sourced ROS 2 env, on the machine that has the bag):

    python3 tools/eval_detection_accuracy.py [fixture.yaml] [--tol 0.30] [--json out.json]

fixture defaults to test/fixtures/bench_replay.yaml -- same schema (bag_path,
pointcloud_topic, odom_topic, params, obstacles, expect).
"""

import argparse
import json
import math
import os
import sys

import yaml

_HERE = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_FIXTURE = os.path.join(_HERE, '..', 'test', 'fixtures', 'bench_replay.yaml')


def _percentile(sorted_vals, q):
    if not sorted_vals:
        return float('nan')
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    pos = q * (len(sorted_vals) - 1)
    lo = int(math.floor(pos))
    hi = min(lo + 1, len(sorted_vals) - 1)
    return sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * (pos - lo)


def _match_obstacle(frames, centre, tol):
    """Per frame, the nearest marker within tol (XY). Returns lists of
    (frame_idx, marker_id, xy_err, r3d_err, radius)."""
    cx, cy, cz = centre
    hits = []
    for i, frame in enumerate(frames):
        best = None
        for m in frame:
            dx = m.pose.position.x - cx
            dy = m.pose.position.y - cy
            dz = m.pose.position.z - cz
            xy = math.hypot(dx, dy)
            if xy <= tol and (best is None or xy < best[2]):
                best = (i, m.id, xy, math.sqrt(dx * dx + dy * dy + dz * dz), m.scale.x / 2.0)
        if best is not None:
            hits.append(best)
    return hits


def evaluate(cfg, frames, n_clouds, tol):
    n_frames = len(frames)
    obstacles = []
    claimed_per_frame = [set() for _ in range(n_frames)]

    for obs in cfg.get('obstacles', []):
        hits = _match_obstacle(frames, obs['center_xyz'], tol)
        for (fi, mid, *_rest) in hits:
            claimed_per_frame[fi].add(mid)
        xy_errs = sorted(h[2] for h in hits)
        r3d_errs = sorted(h[3] for h in hits)
        radii = [h[4] for h in hits]
        ids_in_order = [h[1] for h in hits]
        id_switches = sum(1 for a, b in zip(ids_in_order, ids_in_order[1:]) if a != b)
        obstacles.append({
            'name': obs['name'],
            'center_xyz': list(obs['center_xyz']),
            'detections': len(hits),
            'detection_rate': (len(hits) / n_frames) if n_frames else 0.0,
            'xy_err_mean_m': (sum(xy_errs) / len(xy_errs)) if xy_errs else float('nan'),
            'xy_err_p95_m': _percentile(xy_errs, 0.95),
            'xy_err_max_m': xy_errs[-1] if xy_errs else float('nan'),
            'err3d_mean_m': (sum(r3d_errs) / len(r3d_errs)) if r3d_errs else float('nan'),
            'radius_mean_m': (sum(radii) / len(radii)) if radii else float('nan'),
            'distinct_ids': sorted(set(ids_in_order)),
            'id_switches': id_switches,
        })

    spurious_counts = [len(frame) - len(claimed_per_frame[i]) for i, frame in enumerate(frames)]
    total_markers = sum(len(f) for f in frames)
    max_radius_seen = max((m.scale.x / 2.0 for f in frames for m in f), default=float('nan'))

    return {
        'bag_path': cfg.get('bag_path'),
        'tol_m': tol,
        'n_clouds': n_clouds,
        'n_published_frames': n_frames,
        'obstacles': obstacles,
        'spurious_per_frame_mean': (sum(spurious_counts) / n_frames) if n_frames else 0.0,
        'spurious_per_frame_max': max(spurious_counts, default=0),
        'markers_per_frame_mean': (total_markers / n_frames) if n_frames else 0.0,
        'max_reported_radius_m': max_radius_seen,
    }


def _fmt(x, nd=3):
    return 'n/a' if x is None or (isinstance(x, float) and math.isnan(x)) else f'{x:.{nd}f}'


def print_report(rep):
    print(f"\nbag: {rep['bag_path']}   clouds: {rep['n_clouds']}   "
          f"published frames: {rep['n_published_frames']}   match tol: {rep['tol_m']} m")
    print(f"markers/frame: mean {_fmt(rep['markers_per_frame_mean'], 2)}   "
          f"spurious/frame: mean {_fmt(rep['spurious_per_frame_mean'], 2)} "
          f"max {rep['spurious_per_frame_max']}   "
          f"max radius seen: {_fmt(rep['max_reported_radius_m'])} m")
    print()
    hdr = f"{'obstacle':<12} {'det%':>6} {'xyErr':>8} {'p95':>7} {'max':>7} {'r_mean':>7} {'ids':>5} {'switch':>7}"
    print(hdr)
    print('-' * len(hdr))
    for o in rep['obstacles']:
        print(f"{o['name']:<12} {100 * o['detection_rate']:>6.1f} "
              f"{_fmt(o['xy_err_mean_m']):>8} {_fmt(o['xy_err_p95_m']):>7} {_fmt(o['xy_err_max_m']):>7} "
              f"{_fmt(o['radius_mean_m']):>7} {len(o['distinct_ids']):>5} {o['id_switches']:>7}")

    if rep['obstacles']:
        det = [o for o in rep['obstacles'] if o['detections']]
        if det:
            mean_xy = sum(o['xy_err_mean_m'] for o in det) / len(det)
            mean_rate = sum(o['detection_rate'] for o in det) / len(det)
            switches = sum(o['id_switches'] for o in det)
            print(f"\nHEADLINE: mean XY error {100 * mean_xy:.1f} cm | "
                  f"detection {100 * mean_rate:.0f}% | {switches} ID switch(es) | "
                  f"{rep['spurious_per_frame_mean']:.2f} spurious markers/frame "
                  f"over {rep['n_published_frames']} frames")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('fixture', nargs='?', default=_DEFAULT_FIXTURE)
    ap.add_argument('--tol', type=float, default=None, help='match tolerance m (default: fixture expect.centroid_tol_m or 0.3)')
    ap.add_argument('--json', dest='json_out', default=None, help='also write the report as JSON here')
    args = ap.parse_args(argv)

    with open(args.fixture) as f:
        cfg = yaml.safe_load(f)

    bag = cfg.get('bag_path')
    if not bag:
        sys.exit(f'{args.fixture}: bag_path is null -- record a bench bag first (see fixtures/README.md)')
    bag_path = os.path.join(os.path.dirname(os.path.abspath(args.fixture)), bag)
    if not os.path.exists(bag_path):
        sys.exit(f'bag not found: {bag_path}')

    tol = args.tol if args.tol is not None else float(cfg.get('expect', {}).get('centroid_tol_m', 0.3))

    try:
        import rclpy
        from lidar_obstacle_publisher.replay import replay
    except ImportError as e:
        sys.exit(f'need a sourced ROS 2 environment: {e}')

    rclpy.init()
    try:
        frames, n_clouds = replay(cfg, bag_path)
    finally:
        rclpy.shutdown()

    rep = evaluate(cfg, frames, n_clouds, tol)
    print_report(rep)
    if args.json_out:
        with open(args.json_out, 'w') as f:
            json.dump(rep, f, indent=2)
        print(f'\nwrote {args.json_out}')


if __name__ == '__main__':
    main()
