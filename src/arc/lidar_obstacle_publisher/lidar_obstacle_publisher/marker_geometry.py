"""Pure marker-geometry helper: turn a track's vertical span into a centre
height + extent, falling back to the full sensing band when the span is not yet
populated. Kept numpy-free and rclpy-free so the NaN-fallback path is unit
tested without a ROS environment."""

import math


def vertical_extent(z_min, z_max, band_lo, band_hi, floor):
    """Return ``(z_centre, height)`` for a marker.

    When ``z_min``/``z_max`` are finite and ordered, use them. Otherwise the
    track has no observed span yet -> centre on the sensing band and span it,
    but never report a height below ``floor`` (min_obstacle_diameter_m), so a
    degenerate band can't produce a zero-height cylinder.
    """
    if math.isfinite(z_min) and math.isfinite(z_max) and z_max > z_min:
        return 0.5 * (z_min + z_max), z_max - z_min
    return 0.5 * (band_lo + band_hi), max(band_hi - band_lo, floor)
