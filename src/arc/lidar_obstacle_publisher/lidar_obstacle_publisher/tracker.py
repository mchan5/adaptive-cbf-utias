"""Greedy nearest-centroid obstacle ID tracker.

Split out of lidar_obstacle_publisher_node so the node body reads as
filter -> cluster -> track -> publish, and the association heuristic is
testable on its own.

Why stable IDs matter at all: single_vehicle_cbf_rate_arc's
RateAutopilotCore::handleObstacles() keys its per-obstacle entry-certification
state and the QP certified/uncertified routing on marker.id. If IDs churned
frame to frame, one physical obstacle would flip certified<->uncertified every
tick. Greedy nearest-centroid is the weakest tracker that still gives stable
IDs for well-separated, slow-moving obstacles -- which is all this deployment
has (obstacles <= 0.5 m radius, effectively static).

A track not matched in a frame is simply dropped -- no timeout, no DELETE
marker. handleObstacles() already treats "absent from this MarkerArray" as
"gone", so holding stale tracks here would only fight that.
"""

import numpy as np


class GreedyCentroidTracker:
    def __init__(self):
        self._centroids = {}   # id -> np.ndarray (3,), last frame's matched centroid
        self._next_id = 0

    def update(self, clusters, max_jump_m):
        """clusters: list of (centroid, radius, num_points) for the current frame.

        Returns {id: (centroid, radius)} and advances internal state. Each
        cluster is matched to the nearest unused previous-frame track within
        max_jump_m; unmatched clusters get a fresh monotonic id.
        """
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
