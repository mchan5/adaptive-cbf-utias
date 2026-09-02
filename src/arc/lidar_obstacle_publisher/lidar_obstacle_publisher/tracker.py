"""Greedy nearest-centroid obstacle ID tracker."""

import numpy as np


class GreedyCentroidTracker:
    def __init__(self):
        self._centroids = {}   # id -> np.ndarray (3,), last frame's matched centroid
        self._next_id = 0

    def update(self, clusters, max_jump_m):
        """clusters: list of (centroid, radius, num_points) for the current frame. Returns {id:
        (centroid, radius)} and advances internal state."""
        prev_ids = list(self._centroids.keys())
        used = set()
        assigned = {}

        for centroid, radius, _n in clusters:
            best_id, best_dist = None, max_jump_m
            for pid in prev_ids:
                if pid in used:
                    continue
                d = float(np.linalg.norm(centroid - self._centroids[pid]))
                if d < best_dist:
                    best_dist = d
                    best_id = pid
            if best_id is None:
                best_id = self._next_id
                self._next_id += 1
            used.add(best_id)
            assigned[best_id] = (centroid, radius)

        self._centroids = {pid: centroid for pid, (centroid, _r) in assigned.items()}
        return assigned
