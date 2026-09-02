"""Campaign plan + manifest/CSV bookkeeping for the trial runner."""

import csv
import json
import os
import random
import subprocess
from dataclasses import dataclass, field, asdict

# arm name -> {penn_enabled, cbf_gamma_obs} to push before the trial
DEFAULT_ARMS = {
    'adaptive': {'penn_enabled': True},
    'fixed-5.0': {'penn_enabled': False, 'cbf_gamma_obs': 5.0},
    'fixed-8.0': {'penn_enabled': False, 'cbf_gamma_obs': 8.0},
}

MANIFEST_FIELDS = [
    'scene_id', 'arm', 'trial_index', 'timestamp', 'regime', 'git_hash',
    'config_dirty', 'outcome', 't_goal', 'min_obstacle_dist', 'penn_duty_cycle',
    'bag_path', 'notes',
]


def _git(repo, *args):
    try:
        return subprocess.run(['git', '-C', repo, *args], capture_output=True,
                              text=True, timeout=3).stdout.strip()
    except Exception:  # noqa: BLE001
        return ''


def git_state(repo):
    h = _git(repo, 'rev-parse', '--short', 'HEAD')
    dirty = bool(_git(repo, 'status', '--porcelain'))
    return h, dirty


@dataclass
class Plan:
    root: str
    scenes: list                 # [scene_id, ...]
    arms: list                   # [arm_name, ...]
    trials_per_cell: int
    seed: int
    regime: str                  # 'sitl' | 'synthetic' | 'lidar'
    scene_dir: str
    order: dict = field(default_factory=dict)   # scene_id -> [arm order per trial round]

    @staticmethod
    def create(root, scene_ids, arms, trials_per_cell, seed, regime, scene_dir):
        rng = random.Random(seed)
        order = {}
        for sid in scene_ids:
            rounds = []
            for _ in range(trials_per_cell):
                a = list(arms)
                rng.shuffle(a)
                rounds.append(a)
            order[str(sid)] = rounds
        p = Plan(root, list(scene_ids), list(arms), trials_per_cell, seed,
                 regime, scene_dir, order)
        os.makedirs(root, exist_ok=True)
        with open(os.path.join(root, 'plan.json'), 'w') as f:
            json.dump(asdict(p), f, indent=2)
        return p

    @staticmethod
    def load(root):
        with open(os.path.join(root, 'plan.json')) as f:
            d = json.load(f)
        return Plan(**d)

    def scene_path(self, scene_id):
        return os.path.join(self.scene_dir, f'scene_{int(scene_id):02d}.json')

    def sequence(self):
        """Flat ordered list of (scene_id, trial_index, arm) cells."""
        cells = []
        for sid in self.scenes:
            for k, rnd in enumerate(self.order[str(sid)]):
                for arm in rnd:
                    cells.append((sid, k, arm))
        return cells


class Manifest:
    def __init__(self, root):
        self.root = root
        self.path = os.path.join(root, 'manifest.json')
        self.csv_path = os.path.join(root, 'trials.csv')
        self.rows = []
        if os.path.exists(self.path):
            with open(self.path) as f:
                self.rows = json.load(f)

    def done_keys(self):
        return {(str(r['scene_id']), r['trial_index'], r['arm'])
                for r in self.rows if r.get('outcome')}

    def append(self, row):
        full = {k: row.get(k, '') for k in MANIFEST_FIELDS}
        self.rows.append(full)
        with open(self.path, 'w') as f:
            json.dump(self.rows, f, indent=2)
        write_header = not os.path.exists(self.csv_path)
        with open(self.csv_path, 'a', newline='') as f:
            w = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS)
            if write_header:
                w.writeheader()
            w.writerow(full)
        return full
