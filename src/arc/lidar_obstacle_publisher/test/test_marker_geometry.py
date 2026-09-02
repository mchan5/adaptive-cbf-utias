"""Unit tests for marker_geometry.vertical_extent -- centre/height with a
sensing-band fallback when a track has no observed vertical span."""

import math

from lidar_obstacle_publisher.marker_geometry import vertical_extent

BAND_LO, BAND_HI, FLOOR = 0.1, 2.5, 0.2


def test_uses_the_observed_span_when_valid():
    z_c, h = vertical_extent(0.4, 1.6, BAND_LO, BAND_HI, FLOOR)
    assert z_c == 1.0 and abs(h - 1.2) < 1e-9


def test_falls_back_to_the_band_when_span_is_nan():
    z_c, h = vertical_extent(math.nan, math.nan, BAND_LO, BAND_HI, FLOOR)
    assert abs(z_c - 1.3) < 1e-9 and abs(h - 2.4) < 1e-9


def test_falls_back_when_span_is_inverted_or_degenerate():
    for lo, hi in [(1.0, 1.0), (1.5, 0.5), (math.inf, -math.inf)]:
        z_c, h = vertical_extent(lo, hi, BAND_LO, BAND_HI, FLOOR)
        assert abs(z_c - 1.3) < 1e-9 and abs(h - 2.4) < 1e-9


def test_fallback_height_never_below_floor():
    # a degenerate band narrower than the floor
    z_c, h = vertical_extent(math.nan, math.nan, 1.0, 1.05, FLOOR)
    assert h == FLOOR
