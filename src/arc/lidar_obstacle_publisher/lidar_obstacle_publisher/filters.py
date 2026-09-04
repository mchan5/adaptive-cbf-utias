"""Pure spatial pre-filters for the LiDAR obstacle pipeline.

numpy-only (no rclpy) so they are unit-testable without a ROS environment, the
same split as hysteresis_tracker.py. Each takes an (N, 3) array and returns a
boolean keep-mask (True == keep) the caller applies to its point array.
"""


def self_filter_mask(points_body, radius_m, z_min_m, z_max_m):
    """Keep-mask for points OUTSIDE the vehicle's own body.

    Drops points inside a vertical cylinder centred on the body origin: within
    ``radius_m`` in the body XY plane AND between ``z_min_m`` and ``z_max_m`` in
    body z. That is the drone itself -- arms, props, legs, payload -- seen in
    the Mid-360's near field. A quadrotor is roughly radially symmetric about
    the flight-controller origin, so one radius fits it (no three hand-tuned box
    half-extents, no single zero silently disabling everything).

    Runs in the BODY frame (after the mount transform) so it stays locked to the
    airframe regardless of mount offset or attitude. Real obstacles are cleared
    to obs_d_min (~1 m) laterally, so a sub-metre self column never clips one.
    ``radius_m <= 0`` disables it (the caller checks).
    """
    inside = (
        (points_body[:, 0] ** 2 + points_body[:, 1] ** 2 < radius_m * radius_m)
        & (points_body[:, 2] >= z_min_m)
        & (points_body[:, 2] <= z_max_m)
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
