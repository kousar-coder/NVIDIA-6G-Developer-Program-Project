"""
SenseForge — RF Pipeline Test Suite
════════════════════════════════════
32 tests covering WaveformConfig formulas, OFDM signal properties,
echo physics (Doppler/delay formulas), CFAR outputs, and Kalman tracker.
"""

import math
import sys
import os
import pytest
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rf.waveform_gen import WaveformConfig, SPEED_OF_LIGHT
from rf.echo_simulator import Target, ScenarioConfig, make_test_scenario
from rf.range_doppler import (
    compute_channel_estimate,
    clutter_removal_eca,
    compute_range_doppler_map,
    cfar_detector,
    _nms,
    RDDetection,
)
from rf.rf_tracker import KalmanRFTracker, RFTrack


# ═══════════════════════════════════════════════════════════════════════
# WaveformConfig Tests
# ═══════════════════════════════════════════════════════════════════════

class TestWaveformConfig:

    def test_default_params(self):
        cfg = WaveformConfig()
        assert cfg.num_subcarriers == 272
        assert cfg.num_symbols == 14
        assert cfg.fft_size == 512
        assert cfg.scs == 30e3
        assert cfg.carrier_freq == 3.5e9

    def test_bandwidth(self):
        cfg = WaveformConfig()
        expected = 272 * 30e3
        assert cfg.bandwidth_hz == expected

    def test_range_resolution(self):
        cfg = WaveformConfig()
        expected = SPEED_OF_LIGHT / (2.0 * cfg.bandwidth_hz)
        assert abs(cfg.range_res_m - expected) < 1e-6

    def test_max_range(self):
        cfg = WaveformConfig()
        expected = SPEED_OF_LIGHT / (2.0 * cfg.scs)
        assert abs(cfg.max_range_m - expected) < 1e-6

    def test_velocity_resolution_positive(self):
        cfg = WaveformConfig()
        assert cfg.velocity_res_mps > 0

    def test_max_velocity_positive(self):
        cfg = WaveformConfig()
        assert cfg.max_velocity_mps > 0

    def test_wavelength(self):
        cfg = WaveformConfig()
        expected = SPEED_OF_LIGHT / cfg.carrier_freq
        assert abs(cfg.wavelength_m - expected) < 1e-10

    def test_range_res_reasonable(self):
        cfg = WaveformConfig()
        assert 10 < cfg.range_res_m < 50  # ~18m for 272×30kHz BW

    def test_max_range_reasonable(self):
        cfg = WaveformConfig()
        assert 4000 < cfg.max_range_m < 6000  # ~5km

    def test_frozen_dataclass(self):
        cfg = WaveformConfig()
        with pytest.raises(Exception):
            cfg.num_subcarriers = 100


# ═══════════════════════════════════════════════════════════════════════
# Target / Echo Tests
# ═══════════════════════════════════════════════════════════════════════

class TestTarget:

    def test_rcs_linear(self):
        t = Target(range_m=50, velocity_mps=1.0, rcs_dbsm=-10)
        assert abs(t.rcs_m2 - 0.1) < 1e-6

    def test_zero_rcs(self):
        t = Target(range_m=50, velocity_mps=0, rcs_dbsm=0)
        assert abs(t.rcs_m2 - 1.0) < 1e-6

    def test_doppler_positive_velocity(self):
        t = Target(range_m=50, velocity_mps=2.0)
        fd = t.doppler_hz(3.5e9)
        expected = 2 * 2.0 * 3.5e9 / SPEED_OF_LIGHT
        assert abs(fd - expected) < 1e-3

    def test_doppler_negative_velocity(self):
        t = Target(range_m=50, velocity_mps=-3.0)
        fd = t.doppler_hz(3.5e9)
        assert fd < 0

    def test_doppler_zero_velocity(self):
        t = Target(range_m=50, velocity_mps=0)
        assert t.doppler_hz(3.5e9) == 0.0

    def test_delay(self):
        t = Target(range_m=100, velocity_mps=0)
        expected = 2 * 100 / SPEED_OF_LIGHT
        assert abs(t.delay_s - expected) < 1e-12

    def test_make_test_scenario(self):
        sc = make_test_scenario(n_targets=3, seed=42)
        assert len(sc.targets) == 3
        assert sc.tx_power_dbm == 30.0

    def test_scenario_targets_range(self):
        sc = make_test_scenario(n_targets=4, seed=99)
        for t in sc.targets:
            assert 10 <= t.range_m <= 120

    def test_scenario_deterministic(self):
        sc1 = make_test_scenario(n_targets=2, seed=42)
        sc2 = make_test_scenario(n_targets=2, seed=42)
        assert sc1.targets[0].range_m == sc2.targets[0].range_m


# ═══════════════════════════════════════════════════════════════════════
# Channel Estimate & Range-Doppler Tests
# ═══════════════════════════════════════════════════════════════════════

class TestRangeDoppler:

    def test_channel_estimate_identity(self):
        tx = np.ones((16, 4), dtype=np.complex64)
        rx = np.ones((16, 4), dtype=np.complex64) * 2.0
        H = compute_channel_estimate(rx, tx)
        assert np.allclose(np.abs(H), 2.0, atol=0.01)

    def test_channel_estimate_shape(self):
        tx = np.ones((32, 8), dtype=np.complex64)
        rx = np.ones((32, 8), dtype=np.complex64)
        H = compute_channel_estimate(rx, tx)
        assert H.shape == (32, 8)

    def test_clutter_removal_zeros_mean(self):
        rng = np.random.RandomState(42)
        H = rng.randn(32, 8).astype(np.complex64) + 5.0
        H_clean = clutter_removal_eca(H)
        mean_magnitude = np.abs(np.mean(H_clean, axis=1)).mean()
        assert mean_magnitude < 1e-5

    def test_rd_map_shape(self):
        H = np.random.randn(32, 8).astype(np.complex64)
        rd, r_ax, d_ax = compute_range_doppler_map(H, 64, 16)
        assert rd.shape == (64, 16)
        assert len(r_ax) == 64
        assert len(d_ax) == 16

    def test_rd_map_dtype(self):
        H = np.random.randn(32, 8).astype(np.complex64)
        rd, _, _ = compute_range_doppler_map(H, 64, 16)
        assert rd.dtype == np.float32

    def test_rd_map_finite(self):
        H = np.random.randn(32, 8).astype(np.complex64)
        rd, _, _ = compute_range_doppler_map(H, 64, 16)
        assert np.all(np.isfinite(rd))

    def test_cfar_on_noise(self):
        rng = np.random.RandomState(42)
        rd = rng.randn(64, 32).astype(np.float32) * 5 - 40
        dets = cfar_detector(rd, guard=(2, 2), train=(4, 4), false_alarm=1e-4)
        # Noise-only: should have very few detections
        assert len(dets) < 10

    def test_cfar_with_strong_target(self):
        rd = np.ones((64, 32), dtype=np.float32) * (-40)
        rd[20, 16] = 10  # Very strong target
        dets = cfar_detector(rd, guard=(2, 2), train=(4, 4), false_alarm=1e-4)
        assert len(dets) >= 1

    def test_nms_removes_duplicates(self):
        dets = [
            RDDetection(range_m=50, velocity_mps=1, snr_db=20, range_bin=10, doppler_bin=5),
            RDDetection(range_m=51, velocity_mps=1, snr_db=15, range_bin=11, doppler_bin=5),
        ]
        kept = _nms(dets, r_tol=2, d_tol=2)
        assert len(kept) == 1
        assert kept[0].snr_db == 20  # Keep stronger

    def test_nms_keeps_separated(self):
        dets = [
            RDDetection(range_m=50, velocity_mps=1, snr_db=20, range_bin=10, doppler_bin=5),
            RDDetection(range_m=100, velocity_mps=-2, snr_db=15, range_bin=30, doppler_bin=20),
        ]
        kept = _nms(dets, r_tol=2, d_tol=2)
        assert len(kept) == 2


# ═══════════════════════════════════════════════════════════════════════
# Kalman RF Tracker Tests
# ═══════════════════════════════════════════════════════════════════════

class TestKalmanRFTracker:

    def test_birth(self):
        tracker = KalmanRFTracker()
        dets = [RDDetection(range_m=50, velocity_mps=1, snr_db=15, range_bin=10, doppler_bin=5)]
        tracks = tracker.update(dets)
        assert len(tracker.tracks) >= 1

    def test_confirm(self):
        tracker = KalmanRFTracker(confirm_hits=2)
        det = RDDetection(range_m=50, velocity_mps=1, snr_db=15, range_bin=10, doppler_bin=5)
        tracker.update([det])
        confirmed = tracker.update([det])
        assert len(confirmed) >= 1

    def test_death(self):
        tracker = KalmanRFTracker(max_missed=2)
        det = RDDetection(range_m=50, velocity_mps=1, snr_db=15, range_bin=10, doppler_bin=5)
        tracker.update([det])
        tracker.update([det])
        tracker.update([])
        tracker.update([])
        tracker.update([])
        # After 3 misses with max_missed=2, track should be deleted
        assert len(tracker.tracks) == 0

    def test_reset(self):
        tracker = KalmanRFTracker()
        det = RDDetection(range_m=50, velocity_mps=1, snr_db=15, range_bin=10, doppler_bin=5)
        tracker.update([det])
        tracker.reset()
        assert len(tracker.tracks) == 0
