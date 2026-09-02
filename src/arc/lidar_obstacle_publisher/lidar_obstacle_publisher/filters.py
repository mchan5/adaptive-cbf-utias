"""Pure spatial pre-filters for the LiDAR obstacle pipeline.

numpy-only (no rclpy) so they are unit-testable without a ROS environment, the
same split as tracker.py. Each returns a boolean keep-mask (True == keep) the
caller applies to its point array.
"""

import numpy as np


def self_filter_mask(points_body, half_extents_xyz):
    """Keep-mask for points OUTSIDE the vehicle's own footprint.

    ``half_extents_xyz`` is the half-size of an axis-aligned box centred on the
    body origin; any point with |x| < hx AND |y| < hy AND |z| < hz is the drone
    itself (frame, arms, props, legs, payload) seen in the Mid-360's near field,
    and is dropped. An all-zero half-extent disables the filter -- nothing lies
    strictly inside a zero-width box. Meant to run in the BODY frame so it stays
    locked to the airframe regardless of mount offset or attitude; the spherical
    ``min_range_m`` cull is too blunt for an asymmetric airframe.
    """
    hx, hy, hz = half_extents_xyz
    inside = (
        (np.abs(points_body[:, 0]) < hx)
        & (np.abs(points_body[:, 1]) < hy)
        & (np.abs(points_body[:, 2]) < hz)
    )
    return ~inside


def roi_mask(points_world, xy_min, xy_max):
    """Keep-mask for points inside the world-frame arena box.

    Points whose world X or Y falls outside ``[xy_min, xy_max]`` (room walls,
    neighbouring equipment) are dropped before clustering. A wall's bounding
    disc would otherwise exceed ``max_obstacle_radius_m`` and, chained cell by
    cell, inflate the obstacle count the GAT/QP see. Apply only when the arena
    ROI is enabled; keep the flight geofence strictly inside it.
    """
    return (
        (points_world[:, 0] >= xy_min[0])
        & (points_world[:, 0] <= xy_max[0])
        & (points_world[:, 1] >= xy_min[1])
        & (points_world[:, 1] <= xy_max[1])
    )
