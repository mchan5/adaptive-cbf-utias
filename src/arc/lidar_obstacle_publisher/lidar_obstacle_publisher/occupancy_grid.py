"""Temporal 2D occupancy grid + connected-component clustering.

Replaces the per-frame 3D voxel clustering. Real Mid-360 returns are sparse and
patchy at range: one cloud of a 0.75 m pole fragments into several disconnected
3D voxel blobs, each becoming its own short-lived "obstacle", and a single
dropped frame deletes a track. Here every cloud is projected to the world-frame
XY plane -- the arena obstacles are vertical (cylinders, poles, boxes, people;
nothing to fly over or under) -- and folded into a grid that decays between
frames. Sparse hits on the same object accumulate over ~3 frames into one
stable cluster instead of flickering; a removed object fades out over ~1 s
rather than vanishing instantly.

Per-cell vertical span (z_lo/z_hi) is tracked alongside for visualisation and
for the eventual cylinder-height output; the clustering itself is purely 2D.

numpy-only, no rclpy, so it is unit-testable without a ROS environment (the
same split as tracker.py / filters.py).
"""

import numpy as np

_SQRT2 = float(np.sqrt(2.0))
_NB8 = ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1))


def _label_8(mask):
    """8-connected component labels for a 2D boolean array. 0 = background,
    1..K = components. Seeds only from occupied cells, so cost scales with the
    occupied-cell count, not the whole grid."""
    nx, ny = mask.shape
    labels = np.zeros((nx, ny), dtype=np.int32)
    cur = 0
    for seed in np.argwhere(mask):
        sx, sy = int(seed[0]), int(seed[1])
        if labels[sx, sy]:
            continue
        cur += 1
        labels[sx, sy] = cur
        stack = [(sx, sy)]
        while stack:
            x, y = stack.pop()
            for dx, dy in _NB8:
                ax, ay = x + dx, y + dy
                if 0 <= ax < nx and 0 <= ay < ny and mask[ax, ay] and not labels[ax, ay]:
                    labels[ax, ay] = cur
                    stack.append((ax, ay))
    return labels


class OccupancyGridClusterer:
    """A decaying world-frame XY occupancy grid. Feed it one world-frame cloud
    per tick (``decay`` then ``add_cloud``), then read ``clusters``.

    Grid geometry (origin / size / resolution) is fixed at construction -- it
    sizes the backing arrays. The decay/threshold knobs are passed per call so
    they retune live.
    """

    def __init__(self, origin_xy, size_xy, resolution_m):
        self._origin = np.asarray(origin_xy, dtype=float)   # world XY of cell (0, 0)'s min corner
        self._res = float(resolution_m)
        self._nx = max(1, int(np.ceil(float(size_xy[0]) / self._res)))
        self._ny = max(1, int(np.ceil(float(size_xy[1]) / self._res)))
        self._occ = np.zeros((self._nx, self._ny), dtype=float)
        self._z_lo = np.full((self._nx, self._ny), np.inf)
        self._z_hi = np.full((self._nx, self._ny), -np.inf)

    @property
    def shape(self):
        return (self._nx, self._ny)

    def decay(self, dt, half_life_s):
        """Multiplicative time decay. dt is real elapsed seconds since the last
        call; time-based (not per-frame) so behaviour is independent of cloud
        rate. A no-op for non-positive dt / half-life."""
        if dt > 0.0 and half_life_s > 0.0:
            self._occ *= 0.5 ** (dt / half_life_s)

    def add_cloud(self, points_world, hit_increment=0.35, reset_epsilon=0.05):
        """Fold one world-frame cloud (N x 3) in. Each grid cell touched by at
        least one point gains ``hit_increment`` (saturating at 1.0), regardless
        of how many points landed in it -- the evidence unit is "seen this
        frame", so a single dense frame can't saturate a cell on its own. A
        cell that was effectively empty (occupancy < ``reset_epsilon``) has its
        vertical span reset before this cloud's z values fold in."""
        if points_world.shape[0] == 0:
            return
        idx = np.floor((points_world[:, :2] - self._origin) / self._res).astype(np.int64)
        ix, iy = idx[:, 0], idx[:, 1]
        inb = (ix >= 0) & (ix < self._nx) & (iy >= 0) & (iy < self._ny)
        if not inb.any():
            return
        ix, iy, z = ix[inb], iy[inb], points_world[inb, 2]

        flat = ix * self._ny + iy
        uniq = np.unique(flat)
        ux, uy = uniq // self._ny, uniq % self._ny

        empty = self._occ[ux, uy] < reset_epsilon
        self._z_lo[ux[empty], uy[empty]] = np.inf
        self._z_hi[ux[empty], uy[empty]] = -np.inf

        np.minimum.at(self._z_lo, (ix, iy), z)
        np.maximum.at(self._z_hi, (ix, iy), z)
        self._occ[ux, uy] = np.minimum(1.0, self._occ[ux, uy] + hit_increment)

    def occupied_cell_count(self, threshold):
        return int((self._occ > threshold).sum())

    def clusters(self, threshold=0.5, min_cells=4, max_radius=0.5):
        """8-connected components of the cells above ``threshold``. Returns a
        list of dicts sorted by descending confidence:
          centroid_xy (2,), radius, z_min, z_max, confidence, n_cells
        ``radius`` is the max cell-centre-to-centroid distance plus a cell half
        diagonal (so it encloses the cells), clamped to ``max_radius``."""
        mask = self._occ > threshold
        if not mask.any():
            return []
        labels = _label_8(mask)
        half_diag = 0.5 * self._res * _SQRT2
        out = []
        for lab in range(1, int(labels.max()) + 1):
            cells = np.argwhere(labels == lab)
            if cells.shape[0] < min_cells:
                continue
            cx, cy = cells[:, 0], cells[:, 1]
            w = self._occ[cx, cy]
            centres = self._origin + (cells + 0.5) * self._res
            centroid = np.average(centres, axis=0, weights=w)
            reach = float(np.linalg.norm(centres - centroid, axis=1).max()) + half_diag
            out.append({
                'centroid_xy': centroid,
                'radius': min(reach, float(max_radius)),
                'z_min': float(self._z_lo[cx, cy].min()),
                'z_max': float(self._z_hi[cx, cy].max()),
                'confidence': float(w.mean()),
                'n_cells': int(cells.shape[0]),
            })
        out.sort(key=lambda c: c['confidence'], reverse=True)
        return out
