#!/usr/bin/env python3
"""Export a SITL trial's trajectory into a per-scene reference the operator
console overlays as a faint "ghost" during the matched hardware run.

Trials are hardware-only now; the SITL twin for each frozen scene is just a
visual/quantitative reference. This tool turns one recorded SITL rosbag into
``scene_NN.ref.json`` sitting next to ``scene_NN.json`` in the campaign scene
dir. The console loads it automatically when that scene is selected
(``_load_reference`` in main.py); if the file is absent the console just shows
"SITL ref: not exported" and carries on.

Output schema (all keys optional except ``scene`` and ``trail``)::

    {
      "scene": 3,
      "source_bag": "/abs/path/to/bag_dir",
      "exported_at": "2026-08-31T02:10:00",
      "trail": [[x, y], ...],            # ENU, downsampled, from odom
      "t_goal": 6.21,                    # s, first goal publish -> first `reached`
      "min_obstacle_dist": 0.214,        # m, min over the diagnostics stream
      "obstacles": [[x, y, z, d], ...]   # last non-empty /arc/obstacles frame
    }

Usage::

    source install/setup.bash
    python3 arc_experiment_gui/tools/export_sitl_reference.py \\
        --bag ~/arc_campaigns/20260831_sitl/scene03_adaptive_trial1 \\
        --scene-dir <ws>/install/single_vehicle_cbf_rate_arc/share/single_vehicle_cbf_rate_arc/config/scenes/hw_campaign \\
        --scene 3

    # or point it at a dir of bags named sceneNN_*: one .ref.json per scene
    python3 .../export_sitl_reference.py --bag-root ~/arc_campaigns/20260831_sitl \\
        --scene-dir .../hw_campaign --arm adaptive

The bag needs, at minimum, the odom topic. ``t_goal`` / ``min_obstacle_dist``
/ ``obstacles`` are filled only if the matching topics are present.
"""

import argparse
import glob
import json
import math
import os
import re
import sys
from datetime import datetime

DEFAULT_ODOM = '/uav_0/state_estimator/local_position/odom'
DEFAULT_DIAG = '/uav_0/cbf/diagnostics'
DEFAULT_GOAL = '/uav_0/goal_waypoint'
DEFAULT_OBS = '/arc/obstacles'
_MIN_STEP_M = 0.05          # trail downsample distance
_GOAL_Y_MIN = 3.0          # a goal publish past this y is "the goal", not the start


def _read_bag(bag_path):
    import rosbag2_py
    from rclpy.serialization import deserialize_message
    from rosidl_runtime_py.utilities import get_message

    reader = rosbag2_py.SequentialReader()
    reader.open(
        rosbag2_py.StorageOptions(uri=bag_path, storage_id=''),
        rosbag2_py.ConverterOptions(input_serialization_format='cdr',
                                    output_serialization_format='cdr'),
    )
    types = {t.name: t.type for t in reader.get_all_topics_and_types()}
    while reader.has_next():
        topic, data, t_ns = reader.read_next()
        if topic in types:
            yield topic, deserialize_message(data, get_message(types[topic])), t_ns


def export_one(bag_path, scene_id, scene_dir, topics, dry_run=False):
    odom_t, diag_t, goal_t, obs_t = topics
    trail = []
    t_goal_start_ns = None
    t_reached_ns = None
    min_obs = math.inf
    last_obs_frame = None
    n_odom = 0

    for topic, msg, t_ns in _read_bag(bag_path):
        if topic == odom_t:
            n_odom += 1
            p = msg.pose.pose.position
            if not trail or math.hypot(p.x - trail[-1][0], p.y - trail[-1][1]) >= _MIN_STEP_M:
                trail.append([round(p.x, 3), round(p.y, 3)])
        elif topic == goal_t and t_goal_start_ns is None:
            if getattr(msg, 'point', None) is not None and msg.point.y >= _GOAL_Y_MIN:
                t_goal_start_ns = t_ns
        elif topic == diag_t:
            d = getattr(msg, 'min_obstacle_dist', math.inf)
            if d is not None and d < 1e8:
                min_obs = min(min_obs, d)
            if t_reached_ns is None and getattr(msg, 'reached', False):
                t_reached_ns = t_ns
        elif topic == obs_t and getattr(msg, 'markers', None):
            last_obs_frame = [[round(m.pose.position.x, 3), round(m.pose.position.y, 3),
                               round(m.pose.position.z, 3), round(m.scale.x, 3)]
                              for m in msg.markers]

    if n_odom == 0:
        print(f'  !! no odom on {odom_t} in {bag_path} -- skipping', file=sys.stderr)
        return None

    out = {
        'scene': scene_id,
        'source_bag': os.path.abspath(bag_path),
        'exported_at': datetime.now().isoformat(timespec='seconds'),
        'trail': trail,
    }
    if t_goal_start_ns is None:            # no goal topic -- fall back to bag start
        t_goal_start_ns = None
    if t_reached_ns is not None and t_goal_start_ns is not None:
        out['t_goal'] = round((t_reached_ns - t_goal_start_ns) / 1e9, 2)
    if min_obs < math.inf:
        out['min_obstacle_dist'] = round(min_obs, 3)
    if last_obs_frame is not None:
        out['obstacles'] = last_obs_frame

    ref_path = os.path.join(scene_dir, f'scene_{scene_id:02d}.ref.json')
    print(f'  scene {scene_id:02d}: {len(trail)} trail pts'
          + (f", t_goal={out['t_goal']}s" if 't_goal' in out else ', t_goal=?')
          + (f", min_dist={out['min_obstacle_dist']}m" if 'min_obstacle_dist' in out else '')
          + f'  -> {ref_path}')
    if not dry_run:
        with open(ref_path, 'w') as f:
            json.dump(out, f, indent=2)
    return ref_path


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--scene-dir', required=True,
                    help='campaign scene dir holding scene_NN.json (ref files land here)')
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument('--bag', help='one rosbag2 dir (needs --scene)')
    g.add_argument('--bag-root', help='dir of bags named sceneNN_* -- exports all')
    ap.add_argument('--scene', type=int, help='scene id for --bag')
    ap.add_argument('--arm', default=None,
                    help='[--bag-root] only use bags whose name contains this arm tag')
    ap.add_argument('--odom-topic', default=DEFAULT_ODOM)
    ap.add_argument('--diag-topic', default=DEFAULT_DIAG)
    ap.add_argument('--goal-topic', default=DEFAULT_GOAL)
    ap.add_argument('--obstacles-topic', default=DEFAULT_OBS)
    ap.add_argument('--dry-run', action='store_true')
    a = ap.parse_args(argv)

    topics = (a.odom_topic, a.diag_topic, a.goal_topic, a.obstacles_topic)
    if not os.path.isdir(a.scene_dir):
        ap.error(f'--scene-dir not a dir: {a.scene_dir}')

    if a.bag:
        if a.scene is None:
            ap.error('--bag requires --scene')
        export_one(a.bag, a.scene, a.scene_dir, topics, a.dry_run)
        return 0

    n = 0
    for bag in sorted(glob.glob(os.path.join(a.bag_root, 'scene*'))):
        if not os.path.isdir(bag):
            continue
        m = re.match(r'scene(\d+)_', os.path.basename(bag))
        if not m:
            continue
        if a.arm and a.arm not in os.path.basename(bag):
            continue
        if export_one(bag, int(m.group(1)), a.scene_dir, topics, a.dry_run):
            n += 1
    print(f'exported {n} reference file(s)')
    return 0


if __name__ == '__main__':
    sys.exit(main())
