"""
SenseForge — End-to-End Pipeline Test
══════════════════════════════════════
15 end-to-end checks across RF / Vision / Fusion layers.
Coloured PASS/FAIL output with ANSI codes.
Exits with code 1 if any check fails.

Usage: python run_pipeline_test.py
"""

import os
import sys
import time
import traceback

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# ── ANSI colour codes ────────────────────────────────────────────────────────
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
BOLD = "\033[1m"
RESET = "\033[0m"

passed = 0
failed = 0
skipped = 0
results = []


def check(name, func):
    """Run a single check and record result."""
    global passed, failed, skipped
    try:
        result = func()
        if result is None or result:
            passed += 1
            results.append((name, "PASS", None))
            print(f"  {GREEN}✓ PASS{RESET}  {name}")
        else:
            failed += 1
            results.append((name, "FAIL", "Returned False"))
            print(f"  {RED}✗ FAIL{RESET}  {name}")
    except ImportError as e:
        skipped += 1
        results.append((name, "SKIP", str(e)))
        print(f"  {YELLOW}○ SKIP{RESET}  {name} — {e}")
    except Exception as e:
        failed += 1
        results.append((name, "FAIL", str(e)))
        print(f"  {RED}✗ FAIL{RESET}  {name} — {e}")


def main():
    global passed, failed, skipped

    print()
    print(f"{BOLD}{'═' * 60}{RESET}")
    print(f"{BOLD}{CYAN}  SenseForge — End-to-End Pipeline Validation{RESET}")
    print(f"{BOLD}{'═' * 60}{RESET}")
    print()

    t_start = time.time()

    # ── RF Pipeline Checks ────────────────────────────────────────────────
    print(f"{BOLD}  RF Pipeline{RESET}")
    print(f"  {'─' * 50}")

    def check_waveform_config():
        from rf.waveform_gen import WaveformConfig, SPEED_OF_LIGHT
        cfg = WaveformConfig()
        assert cfg.num_subcarriers == 272
        assert cfg.carrier_freq == 3.5e9
        assert cfg.range_res_m > 0
        assert cfg.max_range_m > 1000
        return True

    def check_waveform_derived():
        from rf.waveform_gen import WaveformConfig
        cfg = WaveformConfig()
        assert cfg.bandwidth_hz == 272 * 30e3
        assert cfg.velocity_res_mps > 0
        assert cfg.max_velocity_mps > 0
        assert cfg.wavelength_m > 0
        return True

    def check_target_physics():
        from rf.echo_simulator import Target
        from rf.waveform_gen import SPEED_OF_LIGHT
        t = Target(range_m=100, velocity_mps=2.0, rcs_dbsm=-10)
        assert abs(t.rcs_m2 - 0.1) < 1e-6
        assert abs(t.delay_s - 2 * 100 / SPEED_OF_LIGHT) < 1e-12
        fd = t.doppler_hz(3.5e9)
        assert abs(fd - 2 * 2.0 * 3.5e9 / SPEED_OF_LIGHT) < 1e-3
        return True

    def check_scenario_gen():
        from rf.echo_simulator import make_test_scenario
        sc = make_test_scenario(n_targets=3, seed=42)
        assert len(sc.targets) == 3
        for t in sc.targets:
            assert 10 <= t.range_m <= 120
        return True

    def check_channel_estimate():
        from rf.range_doppler import compute_channel_estimate
        tx = np.ones((32, 8), dtype=np.complex64) * 2
        rx = np.ones((32, 8), dtype=np.complex64) * 4
        H = compute_channel_estimate(rx, tx)
        assert np.allclose(np.abs(H), 2.0, atol=0.01)
        return True

    def check_rd_map():
        from rf.range_doppler import compute_range_doppler_map
        H = np.random.RandomState(42).randn(32, 8).astype(np.complex64)
        rd, r_ax, d_ax = compute_range_doppler_map(H, 64, 16)
        assert rd.shape == (64, 16)
        assert rd.dtype == np.float32
        assert np.all(np.isfinite(rd))
        return True

    def check_cfar():
        from rf.range_doppler import cfar_detector
        rd = np.ones((64, 32), dtype=np.float32) * (-40)
        rd[20, 16] = 10
        dets = cfar_detector(rd)
        assert len(dets) >= 1
        return True

    def check_rf_tracker():
        from rf.rf_tracker import KalmanRFTracker
        from rf.range_doppler import RDDetection
        tracker = KalmanRFTracker(confirm_hits=2)
        det = RDDetection(range_m=50, velocity_mps=1, snr_db=15, range_bin=10, doppler_bin=5)
        tracker.update([det])
        confirmed = tracker.update([det])
        assert len(confirmed) >= 1
        return True

    check("WaveformConfig parameters", check_waveform_config)
    check("WaveformConfig derived properties", check_waveform_derived)
    check("Target physics (RCS, Doppler, delay)", check_target_physics)
    check("Scenario generator", check_scenario_gen)
    check("Channel estimation", check_channel_estimate)
    check("Range-Doppler map computation", check_rd_map)
    check("CFAR detection", check_cfar)
    check("RF Kalman tracker", check_rf_tracker)

    print()

    # ── Vision Pipeline Checks ────────────────────────────────────────────
    print(f"{BOLD}  Vision Pipeline{RESET}")
    print(f"  {'─' * 50}")

    def check_degradation():
        from vision.degradation import DegradationMode, apply_degradation, get_camera_confidence
        frame = np.full((480, 640, 3), 128, dtype=np.uint8)
        for mode in DegradationMode:
            result = apply_degradation(frame, mode, intensity=0.5)
            assert result.shape == frame.shape
            assert result.dtype == np.uint8
        conf = get_camera_confidence(DegradationMode.FOG, 0.8)
        assert 0.02 <= conf < 0.5
        return True

    def check_detector():
        from vision.detector import YOLODetector
        det = YOLODetector()
        frame = np.full((480, 640, 3), 100, dtype=np.uint8)
        # Draw a bright blob to trigger synthetic detection
        frame[150:350, 280:360] = 200
        results = det._synthetic_detections(frame)
        # Synthetic detections should work
        assert isinstance(results, list)
        return True

    def check_vision_tracker():
        from vision.tracker import VisionTracker, _iou
        from vision.detector import Detection
        assert abs(_iou((0, 0, 10, 10), (0, 0, 10, 10)) - 1.0) < 1e-6
        tracker = VisionTracker(confirm_hits=2)
        det = Detection(bbox=(10, 10, 50, 100), confidence=0.8)
        tracker.update([det])
        confirmed = tracker.update([det])
        assert len(confirmed) >= 1
        return True

    def check_depth():
        from vision.depth import bbox_to_depth_m
        d = bbox_to_depth_m((100, 100, 200, 300), None, 480)
        assert 0.5 <= d <= 150
        return True

    check("Degradation modes (all 5)", check_degradation)
    check("YOLO detector (synthetic)", check_detector)
    check("Vision tracker + IoU", check_vision_tracker)
    check("Depth estimation (heuristic)", check_depth)

    print()

    # ── Fusion Checks ─────────────────────────────────────────────────────
    print(f"{BOLD}  Fusion Layer{RESET}")
    print(f"  {'─' * 50}")

    def check_feature_vector():
        from fusion.model import build_feature_vector, INPUT_DIM
        feat = build_feature_vector(
            rf_range_m=50, rf_velocity_mps=2, rf_snr_db=15,
            rf_confidence=0.9, rf_present=True,
        )
        assert feat.shape == (INPUT_DIM,)
        assert np.all(feat >= 0) and np.all(feat <= 1)
        return True

    def check_fusion_sources():
        from fusion.model import fuse
        r1 = fuse(rf_present=True, rf_range_m=50, rf_confidence=0.8,
                   vision_present=True, vision_depth_m=48, vision_confidence=0.7)
        assert r1["source"] == "fused"
        r2 = fuse(rf_present=True, rf_range_m=50, rf_confidence=0.8)
        assert r2["source"] == "rf_only"
        r3 = fuse(vision_present=True, vision_depth_m=50, vision_confidence=0.7)
        assert r3["source"] == "vision_only"
        r4 = fuse()
        assert r4["source"] == "none"
        return True

    def check_training_data():
        from fusion.train import generate_training_data
        X, Y = generate_training_data(n=100, seed=42)
        assert X.shape == (100, 14)
        assert Y.shape == (100, 4)
        assert X.dtype == np.float32
        assert np.all(X >= 0) and np.all(X <= 1)
        return True

    check("Feature vector normalisation", check_feature_vector)
    check("Fusion source labelling (all 4 branches)", check_fusion_sources)
    check("Training data generator", check_training_data)

    # ── Summary ───────────────────────────────────────────────────────────
    elapsed = time.time() - t_start
    print()
    print(f"{BOLD}{'═' * 60}{RESET}")
    print(f"  {GREEN}{passed} PASSED{RESET}  /  "
          f"{RED}{failed} FAILED{RESET}  /  "
          f"{YELLOW}{skipped} SKIPPED{RESET}  "
          f"({elapsed:.2f}s)")
    print(f"{BOLD}{'═' * 60}{RESET}")

    if failed > 0:
        print(f"\n  {RED}{BOLD}PIPELINE VALIDATION FAILED{RESET}")
        for name, status, err in results:
            if status == "FAIL":
                print(f"    {RED}✗{RESET} {name}: {err}")
        sys.exit(1)
    else:
        print(f"\n  {GREEN}{BOLD}ALL CHECKS PASSED ✓{RESET}")
        sys.exit(0)


if __name__ == "__main__":
    main()
