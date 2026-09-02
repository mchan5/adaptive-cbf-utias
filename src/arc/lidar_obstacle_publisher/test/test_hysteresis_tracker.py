"""Unit tests for HysteresisTracker -- the birth/death debounce and the
monotonic-id property that single_vehicle_cbf_rate_arc's entry certification
depends on."""

import numpy as np

from lidar_obstacle_publisher.hysteresis_tracker import HysteresisTracker


def _det(x, y, radius=0.3, z_min=0.5, z_max=1.5, confidence=0.9):
    return {
        'centroid_xy': np.array([x, y], dtype=float),
        'radius': radius,
        'z_min': z_min,
        'z_max': z_max,
        'confidence': confidence,
    }


def _run(tracker, frames, **kw):
    kw.setdefault('min_hits', 3)
    kw.setdefault('max_misses', 5)
    kw.setdefault('ema_alpha', 1.0)   # snap, so positions are exact in assertions
    out = None
    for dets in frames:
        out = tracker.update(dets, **kw)
    return out


def test_detection_is_not_published_until_min_hits():
    t = HysteresisTracker()
    assert t.update([_det(0, 0)], min_hits=3, ema_alpha=1.0) == {}
    assert t.update([_det(0, 0)], min_hits=3, ema_alpha=1.0) == {}
    out = t.update([_det(0, 0)], min_hits=3, ema_alpha=1.0)
    assert list(out) == [0]


def test_single_frame_flicker_never_publishes():
    t = HysteresisTracker()
    out = _run(t, [[_det(0, 0)], [], [_det(0, 0)], [], [_det(0, 0)], []], min_hits=3)
    assert out == {}


def test_confirmed_track_coasts_through_a_dropout_then_recovers():
    t = HysteresisTracker()
    _run(t, [[_det(1, 1)]] * 3, min_hits=3)          # confirmed
    out = t.update([], min_hits=3, max_misses=5)     # one missed frame
    assert list(out) == [0] and out[0]['coasting'] is True
    assert np.allclose(out[0]['center_xy'], [1, 1])  # held at last pose
    out = t.update([_det(1.05, 1.0)], min_hits=3, ema_alpha=1.0)
    assert list(out) == [0] and out[0]['coasting'] is False


def test_confirmed_track_dropped_after_max_misses():
    t = HysteresisTracker()
    _run(t, [[_det(0, 0)]] * 3, min_hits=3)
    for _ in range(5):
        out = t.update([], min_hits=3, max_misses=5)
        assert list(out) == [0]                       # still coasting
    assert t.update([], min_hits=3, max_misses=5) == {}   # 6th miss -> gone


def test_ids_are_monotonic_and_never_reused():
    t = HysteresisTracker()
    _run(t, [[_det(0, 0)]] * 3, min_hits=3)           # id 0 confirmed
    for _ in range(7):
        t.update([], min_hits=3, max_misses=5)        # id 0 fully expires
    out = _run(t, [[_det(0, 0)]] * 3, min_hits=3)     # same location, new object
    assert list(out) == [1]                           # NOT 0


def test_two_detections_do_not_collapse_onto_one_track():
    t = HysteresisTracker()
    # first frame seeds one track at origin
    t.update([_det(0.0, 0.0)], min_hits=1, ema_alpha=1.0)
    # next frame: two detections both within max_jump of that track
    out = t.update([_det(0.1, 0.0), _det(0.4, 0.0)],
                   min_hits=1, max_jump_m=0.5, ema_alpha=1.0)
    assert sorted(out) == [0, 1]


def test_birth_gap_restarts_the_count_with_a_fresh_id():
    t = HysteresisTracker()
    # 2 hits then a miss before confirming -> the unconfirmed track is discarded
    _run(t, [[_det(0, 0)], [_det(0, 0)], []], min_hits=3)
    assert t.update([_det(0, 0)], min_hits=3) == {}   # fresh count starts over
    assert t.update([_det(0, 0)], min_hits=3) == {}
    out = t.update([_det(0, 0)], min_hits=3)
    assert list(out) == [1]                           # a new id, not the discarded 0


def test_min_hits_one_confirms_immediately():
    t = HysteresisTracker()
    out = t.update([_det(2, 3)], min_hits=1, ema_alpha=1.0)
    assert list(out) == [0]
    assert np.allclose(out[0]['center_xy'], [2, 3])


def test_ema_smooths_center():
    t = HysteresisTracker()
    _run(t, [[_det(0, 0)]] * 3, min_hits=3, ema_alpha=0.5)
    out = t.update([_det(0.3, 0.0)], min_hits=3, ema_alpha=0.5)   # within max_jump
    assert 0.0 < out[0]['center_xy'][0] < 0.3        # moved partway, not snapped
