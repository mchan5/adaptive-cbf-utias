"""Hysteresis obstacle tracker: greedy 2D association + birth/death debouncing.

Replaces GreedyCentroidTracker. The consumer (single_vehicle_cbf_rate_arc) keys
entry certification and the GAT graph on marker.id, and DROPS an obstacle from
the QP the instant its id is absent from one MarkerArray. Two behaviours matter:

  * birth hysteresis -- a detection must persist ``min_hits`` consecutive frames
    before it is published at all, so clustering flicker never reaches the QP or
    triggers a certification attempt.
  * death hysteresis -- a confirmed track that stops matching keeps being
    published at its last position for ``max_misses`` frames ("coasting")
    before deletion, so a one-frame sensor dropout does not yank an obstacle
    out of the QP.

IDs are monotonic and NEVER reused: the certification registry
(entry_certification_gate.cpp) caches a per-id result and never prunes it, so a
recycled id would silently inherit a dead obstacle's certification.

numpy-only, no rclpy.
"""

import numpy as np


def _ema_nan(prev, new, alpha):
    """EMA that tolerates NaN/inf on either side -- a missing new value keeps
    prev; a first finite value replaces a non-finite prev."""
    new = float(new)
    if not np.isfinite(new):
        return prev
    if not np.isfinite(prev):
        return new
    return (1.0 - alpha) * prev + alpha * new


class _Track:
    __slots__ = ('id', 'center_xy', 'radius', 'z_min', 'z_max', 'confidence',
                 'hits', 'misses', 'confirmed')

    def __init__(self, track_id, det):
        self.id = track_id
        self.center_xy = np.asarray(det['centroid_xy'], dtype=float).copy()
        self.radius = float(det['radius'])
        self.z_min = float(det.get('z_min', np.nan))
        self.z_max = float(det.get('z_max', np.nan))
        self.confidence = float(det.get('confidence', 1.0))
        self.hits = 1
        self.misses = 0
        self.confirmed = False

    def matched(self, det, alpha):
        c = np.asarray(det['centroid_xy'], dtype=float)
        self.center_xy = (1.0 - alpha) * self.center_xy + alpha * c
        self.radius = (1.0 - alpha) * self.radius + alpha * float(det['radius'])
        self.z_min = _ema_nan(self.z_min, det.get('z_min', np.nan), alpha)
        self.z_max = _ema_nan(self.z_max, det.get('z_max', np.nan), alpha)
        self.confidence = float(det.get('confidence', self.confidence))
        self.hits += 1
        self.misses = 0

    def missed(self):
        self.hits = 0        # birth requires CONSECUTIVE hits
        self.misses += 1


class HysteresisTracker:
    def __init__(self):
        self._tracks = []
        self._next_id = 0

    def update(self, detections, max_jump_m=0.5, min_hits=3, max_misses=5, ema_alpha=0.5):
        """Advance one frame. ``detections`` is a list of dicts, each with at
        least ``centroid_xy`` (2,) and ``radius`` (``z_min``/``z_max``/
        ``confidence`` optional). Returns ``{id: {center_xy, radius, z_min,
        z_max, confidence, coasting}}`` for confirmed tracks only, including
        those coasting through a dropout."""
        claimed = set()
        for det in detections:
            c = np.asarray(det['centroid_xy'], dtype=float)
            best, best_d = None, max_jump_m
            for tr in self._tracks:
                if tr.id in claimed:
                    continue
                d = float(np.linalg.norm(c - tr.center_xy))
                if d < best_d:
                    best, best_d = tr, d
            if best is None:
                tr = _Track(self._next_id, det)
                self._next_id += 1
                if tr.hits >= min_hits:
                    tr.confirmed = True
                self._tracks.append(tr)
                claimed.add(tr.id)
            else:
                best.matched(det, ema_alpha)
                if best.hits >= min_hits:
                    best.confirmed = True
                claimed.add(best.id)

        survivors = []
        for tr in self._tracks:
            if tr.id in claimed:
                survivors.append(tr)
                continue
            tr.missed()
            if tr.confirmed and tr.misses <= max_misses:
                survivors.append(tr)     # coasting
            # unconfirmed & unmatched, or confirmed & too stale -> dropped
        self._tracks = survivors

        return {
            tr.id: {
                'center_xy': tr.center_xy.copy(),
                'radius': tr.radius,
                'z_min': tr.z_min,
                'z_max': tr.z_max,
                'confidence': tr.confidence,
                'coasting': tr.misses > 0,
            }
            for tr in self._tracks if tr.confirmed
        }
