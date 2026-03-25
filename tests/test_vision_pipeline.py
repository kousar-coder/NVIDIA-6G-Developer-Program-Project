"""
SenseForge — Vision Pipeline Test Suite
════════════════════════════════════════
35 tests covering degradation modes, IoU, tracker, depth estimation.
"""

import sys
import os
import pytest
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vision.degradation import (
    DegradationMode,
    apply_degradation,
    get_camera_confidence,
)
from vision.detector import Detection, YOLODetector
from vision.tracker import VisionTracker, VisionTrack, _iou
from vision.depth import bbox_to_depth_m


def _make_frame(h=480, w=640, value=128):
    return np.full((h, w, 3), value, dtype=np.uint8)


# ═══════════════════════════════════════════════════════════════════════
# Degradation Tests
# ═══════════════════════════════════════════════════════════════════════

class TestDegradation:

    def test_clear_no_change(self):
        frame = _make_frame()
        result = apply_degradation(frame, DegradationMode.CLEAR)
        assert np.array_equal(result, frame)

    def test_fog_brightens(self):
        frame = _make_frame(value=50)
        result = apply_degradation(frame, DegradationMode.FOG, intensity=0.8)
        assert result.mean() > frame.mean()

    def test_fog_intensity_range(self):
        frame = _make_frame()
        r_low = apply_degradation(frame, DegradationMode.FOG, intensity=0.2)
        r_high = apply_degradation(frame, DegradationMode.FOG, intensity=0.9)
        assert r_high.mean() > r_low.mean()

    def test_night_darkens(self):
        frame = _make_frame(value=180)
        result = apply_degradation(frame, DegradationMode.NIGHT, intensity=0.8)
        assert result.mean() < frame.mean()

    def test_night_intensity_scale(self):
        frame = _make_frame(value=180)
        r_low = apply_degradation(frame, DegradationMode.NIGHT, intensity=0.2)
        r_high = apply_degradation(frame, DegradationMode.NIGHT, intensity=0.9)
        assert r_high.mean() < r_low.mean()

    def test_occlusion_darker(self):
        frame = _make_frame(value=150)
        result = apply_degradation(frame, DegradationMode.OCCLUSION, intensity=0.8)
        assert result.mean() < frame.mean()

    def test_rain_output_shape(self):
        frame = _make_frame()
        result = apply_degradation(frame, DegradationMode.RAIN, intensity=0.5)
        assert result.shape == frame.shape

    def test_all_modes_return_uint8(self):
        frame = _make_frame()
        for mode in DegradationMode:
            r = apply_degradation(frame, mode, intensity=0.5)
            assert r.dtype == np.uint8

    def test_all_modes_preserve_shape(self):
        frame = _make_frame()
        for mode in DegradationMode:
            r = apply_degradation(frame, mode, intensity=0.5)
            assert r.shape == frame.shape

    def test_deterministic(self):
        frame = _make_frame()
        r1 = apply_degradation(frame, DegradationMode.FOG, 0.5, seed=42)
        r2 = apply_degradation(frame, DegradationMode.FOG, 0.5, seed=42)
        assert np.array_equal(r1, r2)

    def test_intensity_clamp(self):
        frame = _make_frame()
        r = apply_degradation(frame, DegradationMode.FOG, intensity=5.0)
        assert r.dtype == np.uint8


class TestCameraConfidence:

    def test_clear_full(self):
        assert get_camera_confidence(DegradationMode.CLEAR) == 1.0

    def test_fog_reduces(self):
        c = get_camera_confidence(DegradationMode.FOG, 0.8)
        assert c < 0.5

    def test_night_reduces(self):
        c = get_camera_confidence(DegradationMode.NIGHT, 0.9)
        assert c < 0.2

    def test_occlusion_reduces(self):
        c = get_camera_confidence(DegradationMode.OCCLUSION, 0.5)
        assert c < 1.0

    def test_rain_reduces(self):
        c = get_camera_confidence(DegradationMode.RAIN, 0.7)
        assert c < 0.8

    def test_minimum_clamp(self):
        c = get_camera_confidence(DegradationMode.NIGHT, 1.0)
        assert c >= 0.02

    def test_all_modes_in_range(self):
        for mode in DegradationMode:
            c = get_camera_confidence(mode, 0.5)
            assert 0.02 <= c <= 1.0


# ═══════════════════════════════════════════════════════════════════════
# IoU Tests
# ═══════════════════════════════════════════════════════════════════════

class TestIoU:

    def test_identical_boxes(self):
        assert abs(_iou((0, 0, 10, 10), (0, 0, 10, 10)) - 1.0) < 1e-6

    def test_no_overlap(self):
        assert _iou((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0

    def test_partial_overlap(self):
        iou = _iou((0, 0, 10, 10), (5, 5, 15, 15))
        assert 0.0 < iou < 1.0

    def test_contained(self):
        iou = _iou((0, 0, 20, 20), (5, 5, 15, 15))
        assert 0.0 < iou < 1.0

    def test_symmetric(self):
        assert abs(_iou((0, 0, 10, 10), (5, 5, 15, 15)) -
                   _iou((5, 5, 15, 15), (0, 0, 10, 10))) < 1e-6


# ═══════════════════════════════════════════════════════════════════════
# Vision Tracker Tests
# ═══════════════════════════════════════════════════════════════════════

class TestVisionTracker:

    def test_birth(self):
        tracker = VisionTracker()
        dets = [Detection(bbox=(10, 10, 50, 100), confidence=0.8)]
        tracker.update(dets)
        assert len(tracker.tracks) >= 1

    def test_confirm(self):
        tracker = VisionTracker(confirm_hits=2)
        det = Detection(bbox=(10, 10, 50, 100), confidence=0.8)
        tracker.update([det])
        confirmed = tracker.update([det])
        assert len(confirmed) >= 1

    def test_death(self):
        tracker = VisionTracker(max_missed=2)
        det = Detection(bbox=(10, 10, 50, 100), confidence=0.8)
        tracker.update([det])
        tracker.update([det])
        tracker.update([])
        tracker.update([])
        tracker.update([])
        assert len(tracker.tracks) == 0

    def test_multiple_tracks(self):
        tracker = VisionTracker()
        dets = [
            Detection(bbox=(10, 10, 50, 100), confidence=0.8),
            Detection(bbox=(200, 10, 250, 100), confidence=0.7),
        ]
        tracker.update(dets)
        assert len(tracker.tracks) == 2

    def test_reset(self):
        tracker = VisionTracker()
        dets = [Detection(bbox=(10, 10, 50, 100), confidence=0.8)]
        tracker.update(dets)
        tracker.reset()
        assert len(tracker.tracks) == 0


# ═══════════════════════════════════════════════════════════════════════
# Depth Estimation Tests
# ═══════════════════════════════════════════════════════════════════════

class TestDepth:

    def test_heuristic_fallback(self):
        d = bbox_to_depth_m((100, 100, 200, 300), None, 480)
        assert 0.5 <= d <= 150

    def test_large_bbox_close(self):
        d = bbox_to_depth_m((0, 0, 640, 480), None, 480)
        assert d < 5  # Very large box → very close

    def test_small_bbox_far(self):
        d = bbox_to_depth_m((300, 200, 320, 230), None, 480)
        assert d > 10  # Small box → far

    def test_depth_monotonicity(self):
        # Larger bbox → closer depth
        d_small = bbox_to_depth_m((300, 200, 320, 230), None, 480)
        d_large = bbox_to_depth_m((200, 100, 440, 400), None, 480)
        assert d_large < d_small

    def test_max_range_clamp(self):
        d = bbox_to_depth_m((300, 200, 305, 202), None, 480, max_range=100)
        assert d <= 100
