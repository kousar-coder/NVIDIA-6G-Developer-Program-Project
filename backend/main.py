"""
SenseForge — FastAPI Backend
═════════════════════════════
Real-time WebSocket streaming of:
  - Radar range-Doppler maps
  - Fused detection tracks
  - Camera video frames

REST endpoints for scenario control and degradation simulation.
"""

from __future__ import annotations

import asyncio
import base64
import io
import json
import os
import sys
import threading
import time
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

# ── Ensure project root is on path ───────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# ── Hard-check pyAerial and Sionna ───────────────────────────────────────────
_sdk_available = True
_sdk_error = None
try:
    import pyaerial  # noqa: F401
    import sionna    # noqa: F401
except ImportError as e:
    _sdk_available = False
    _sdk_error = str(e)
    print(
        "\n"
        "╔══════════════════════════════════════════════════════════════╗\n"
        "║  WARNING: NVIDIA AI Aerial SDK not found.                  ║\n"
        "║  Running in DEMO MODE (synthetic data only).               ║\n"
        "║                                                            ║\n"
        "║  For full functionality, install pyAerial + Sionna.        ║\n"
        "║  See aerial_setup.sh for instructions.                     ║\n"
        "╚══════════════════════════════════════════════════════════════╝\n"
    )

# ── Import SenseForge modules ────────────────────────────────────────────────
from vision.degradation import DegradationMode, apply_degradation, get_camera_confidence
from vision.detector import YOLODetector, Detection
from vision.tracker import VisionTracker
from vision.depth import estimate_depth_map, bbox_to_depth_m
from fusion.model import fuse, load_model

# Conditionally import RF modules
_rf_available = False
try:
    from rf.waveform_gen import WaveformConfig, generate_ofdm_frame
    from rf.echo_simulator import make_test_scenario, simulate_echoes
    from rf.range_doppler import process_slot
    from rf.rf_tracker import KalmanRFTracker
    _rf_available = True
except ImportError:
    pass


# ═════════════════════════════════════════════════════════════════════════════
# GPU / cuPHY verification
# ═════════════════════════════════════════════════════════════════════════════

def _verify_gpu_and_cuphy():
    """Verify GPU and cuPHY PUSCH decoder work correctly."""
    if not _sdk_available:
        return {"status": "sdk_unavailable", "message": _sdk_error}
    try:
        from pyaerial import PuschDecoder, PuschConfig
        cfg = PuschConfig(n_prb=25, mcs_index=14, n_layers=1, snr_db=20, mu=1)
        decoder = PuschDecoder(cfg)
        result = {
            "status": "ok",
            "sionna_version": sionna.__version__,
            "pyaerial_version": getattr(pyaerial, "__version__", "unknown"),
        }
        try:
            import torch
            if torch.cuda.is_available():
                result["gpu_name"] = torch.cuda.get_device_name(0)
        except Exception:
            pass
        return result
    except Exception as e:
        return {"status": "error", "message": str(e)}


# ═════════════════════════════════════════════════════════════════════════════
# Application state
# ═════════════════════════════════════════════════════════════════════════════

app = FastAPI(title="SenseForge", version="1.0.0")

# CORS
frontend_url = os.environ.get("FRONTEND_URL", "http://localhost:3000")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[frontend_url, "http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state
STATE: Dict[str, Any] = {
    "degrade_mode": DegradationMode.CLEAR,
    "degrade_intensity": 0.7,
    "n_targets": 2,
    "scenario_seed": 42,
    "video_source": None,
    # Pipeline outputs
    "rd_matrix": None,
    "range_axis": None,
    "doppler_axis": None,
    "rf_detections": [],
    "rf_tracks": [],
    "vision_tracks": [],
    "fused_detections": [],
    "frame_b64": None,
    "frame_count": 0,
    # Status
    "start_time": time.time(),
    "rf_running": False,
    "vision_running": False,
}

# Pipeline objects
_detector = YOLODetector()
_vision_tracker = VisionTracker()
_rf_tracker = None
if _rf_available:
    _rf_tracker = KalmanRFTracker()


# ═════════════════════════════════════════════════════════════════════════════
# Pydantic models
# ═════════════════════════════════════════════════════════════════════════════

class DegradeRequest(BaseModel):
    mode: str
    intensity: float = 0.7


class ScenarioRequest(BaseModel):
    n_targets: int = 2
    seed: int = 42


# ═════════════════════════════════════════════════════════════════════════════
# REST endpoints
# ═════════════════════════════════════════════════════════════════════════════

@app.get("/health")
async def health():
    uptime = time.time() - STATE["start_time"]
    return {
        "status": "online",
        "uptime_s": round(uptime, 1),
        "sdk_available": _sdk_available,
        "rf_running": STATE["rf_running"],
        "vision_running": STATE["vision_running"],
        "n_targets": STATE["n_targets"],
        "degrade_mode": STATE["degrade_mode"].value,
        "degrade_intensity": STATE["degrade_intensity"],
    }


@app.get("/sdk-info")
async def sdk_info():
    info = {
        "sdk_available": _sdk_available,
        "waveform_params": {
            "num_subcarriers": 272,
            "num_symbols": 14,
            "fft_size": 512,
            "scs_hz": 30000,
            "carrier_freq_hz": 3.5e9,
            "bandwidth_hz": 272 * 30000,
            "mcs_index": 14,
            "band": "n78",
        },
    }
    if _sdk_available:
        info["sionna_version"] = sionna.__version__
        info["pyaerial_version"] = getattr(pyaerial, "__version__", "unknown")
    return info


@app.post("/degrade")
async def set_degradation(req: DegradeRequest):
    try:
        mode = DegradationMode(req.mode)
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid mode: {req.mode}. "
            f"Options: {[m.value for m in DegradationMode]}",
        )
    STATE["degrade_mode"] = mode
    STATE["degrade_intensity"] = max(0.0, min(1.0, req.intensity))
    return {
        "mode": mode.value,
        "intensity": STATE["degrade_intensity"],
        "camera_confidence": get_camera_confidence(mode, STATE["degrade_intensity"]),
    }


@app.post("/scenario")
async def set_scenario(req: ScenarioRequest):
    STATE["n_targets"] = max(1, min(8, req.n_targets))
    STATE["scenario_seed"] = req.seed
    if _rf_tracker is not None:
        _rf_tracker.reset()
    _vision_tracker.reset()
    return {"n_targets": STATE["n_targets"], "seed": STATE["scenario_seed"]}


@app.get("/rd-snapshot")
async def rd_snapshot():
    rd = STATE.get("rd_matrix")
    if rd is None:
        raise HTTPException(status_code=404, detail="No RD map available yet.")

    try:
        from PIL import Image

        # Normalise to [0, 255]
        rd_norm = rd - rd.min()
        rd_max = rd_norm.max()
        if rd_max > 0:
            rd_norm = (rd_norm / rd_max * 255).astype(np.uint8)
        else:
            rd_norm = np.zeros_like(rd, dtype=np.uint8)

        # Apply INFERNO colormap
        colored = cv2.applyColorMap(rd_norm, cv2.COLORMAP_INFERNO)
        _, buf = cv2.imencode(".png", colored)
        b64 = base64.b64encode(buf.tobytes()).decode("utf-8")
        return {"image_b64": b64, "width": rd.shape[1], "height": rd.shape[0]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ═════════════════════════════════════════════════════════════════════════════
# WebSocket endpoints
# ═════════════════════════════════════════════════════════════════════════════

@app.websocket("/ws/radar")
async def ws_radar(ws: WebSocket):
    await ws.accept()
    try:
        while True:
            rd = STATE.get("rd_matrix")
            if rd is not None:
                # Downsample [::2, ::4]
                rd_small = rd[::2, ::4].tolist()
                dets = [
                    {
                        "range_m": d.get("range_m", 0),
                        "velocity_mps": d.get("velocity_mps", 0),
                        "snr_db": d.get("snr_db", 0),
                    }
                    for d in STATE.get("rf_detections", [])
                ]
                await ws.send_json({
                    "rd_matrix": rd_small,
                    "detections": dets,
                    "n_targets": STATE["n_targets"],
                })
            await asyncio.sleep(0.1)  # 10 Hz
    except WebSocketDisconnect:
        pass
    except Exception:
        pass


@app.websocket("/ws/detections")
async def ws_detections(ws: WebSocket):
    await ws.accept()
    try:
        while True:
            fused = STATE.get("fused_detections", [])
            cam_conf = get_camera_confidence(
                STATE["degrade_mode"], STATE["degrade_intensity"]
            )
            await ws.send_json({
                "detections": fused,
                "mode": STATE["degrade_mode"].value,
                "intensity": STATE["degrade_intensity"],
                "camera_confidence": cam_conf,
                "rf_confidence": 1.0,
                "n_targets": STATE["n_targets"],
            })
            await asyncio.sleep(0.05)  # 20 Hz
    except WebSocketDisconnect:
        pass
    except Exception:
        pass


@app.websocket("/ws/video")
async def ws_video(ws: WebSocket):
    await ws.accept()
    try:
        while True:
            frame_b64 = STATE.get("frame_b64")
            if frame_b64:
                await ws.send_json({"frame": frame_b64})
            await asyncio.sleep(1.0 / 15.0)  # 15 Hz
    except WebSocketDisconnect:
        pass
    except Exception:
        pass


# ═════════════════════════════════════════════════════════════════════════════
# Background pipelines
# ═════════════════════════════════════════════════════════════════════════════

def _generate_synthetic_frame(
    width: int = 640,
    height: int = 480,
    frame_count: int = 0,
    n_targets: int = 2,
) -> np.ndarray:
    """Generate a synthetic camera frame with animated person blobs."""
    frame = np.zeros((height, width, 3), dtype=np.uint8)

    # Sky gradient
    for y in range(height // 2):
        t = y / (height // 2)
        b = int(15 + t * 20)
        g = int(10 + t * 15)
        r = int(5 + t * 10)
        frame[y, :] = [b, g, r]

    # Ground
    frame[height // 2 :, :] = [20, 25, 15]

    # Grid lines
    for x in range(0, width, 40):
        cv2.line(frame, (x, height // 2), (x, height), (30, 35, 25), 1)
    for y in range(height // 2, height, 20):
        cv2.line(frame, (0, y), (width, y), (30, 35, 25), 1)

    # Building silhouettes
    buildings = [(50, 180), (180, 220), (350, 160), (500, 200)]
    for bx, bh in buildings:
        y_top = height // 2 - bh
        cv2.rectangle(frame, (bx, y_top), (bx + 60, height // 2), (15, 18, 12), -1)
        # Windows
        for wy in range(y_top + 10, height // 2, 25):
            for wx in range(bx + 10, bx + 55, 20):
                cv2.rectangle(frame, (wx, wy), (wx + 8, wy + 12), (40, 60, 80), -1)

    # Animated person blobs
    for i in range(min(n_targets, 4)):
        px = int(100 + i * 150 + 30 * np.sin(frame_count * 0.03 + i * 1.5))
        py = height // 2 + 30 + i * 20

        # Walking pendulum animation
        leg_offset = int(8 * np.sin(frame_count * 0.15 + i))

        # Body
        cv2.ellipse(frame, (px, py - 25), (12, 30), 0, 0, 360, (60, 80, 100), -1)
        # Head
        cv2.circle(frame, (px, py - 60), 10, (80, 100, 120), -1)
        # Legs
        cv2.line(frame, (px, py + 5), (px - 8 + leg_offset, py + 35), (50, 70, 90), 3)
        cv2.line(frame, (px, py + 5), (px + 8 - leg_offset, py + 35), (50, 70, 90), 3)

    return frame


def _draw_overlays(
    frame: np.ndarray,
    fused_dets: List[Dict],
) -> np.ndarray:
    """Draw corner-bracket bounding boxes on the frame."""
    result = frame.copy()

    for det in fused_dets:
        bbox = det.get("bbox")
        if bbox is None:
            continue
        x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
        source = det.get("source", "none")

        # Color by source
        if source == "fused":
            color = (136, 255, 0)      # #00ff88 → BGR
        elif source == "rf_only":
            color = (255, 182, 56)     # #38b6ff → BGR
        elif source == "vision_only":
            color = (24, 197, 245)     # #f5c518 → BGR
        else:
            color = (100, 100, 100)

        # Corner bracket length
        L = min(20, (x2 - x1) // 3, (y2 - y1) // 3)

        # 4 L-shaped corners
        t = 2
        # Top-left
        cv2.line(result, (x1, y1), (x1 + L, y1), color, t)
        cv2.line(result, (x1, y1), (x1, y1 + L), color, t)
        # Top-right
        cv2.line(result, (x2, y1), (x2 - L, y1), color, t)
        cv2.line(result, (x2, y1), (x2, y1 + L), color, t)
        # Bottom-left
        cv2.line(result, (x1, y2), (x1 + L, y2), color, t)
        cv2.line(result, (x1, y2), (x1, y2 - L), color, t)
        # Bottom-right
        cv2.line(result, (x2, y2), (x2 - L, y2), color, t)
        cv2.line(result, (x2, y2), (x2, y2 - L), color, t)

        # Label pill
        label = f"{source.upper()} {det.get('range_m', 0):.0f}m {det.get('confidence', 0):.0%}"
        label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
        cv2.rectangle(
            result,
            (x1, y1 - label_size[1] - 8),
            (x1 + label_size[0] + 8, y1 - 2),
            color,
            -1,
        )
        cv2.putText(
            result, label, (x1 + 4, y1 - 6),
            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1,
        )

    return result


def _run_rf_pipeline():
    """Background thread: RF radar pipeline at ~10 Hz."""
    STATE["rf_running"] = True
    frame_idx = 0

    if not _rf_available:
        # Synthetic RF mode
        while True:
            try:
                n = STATE["n_targets"]
                rng = np.random.RandomState(STATE["scenario_seed"] + frame_idx)

                # Generate synthetic RD map
                rd = rng.randn(64, 32).astype(np.float32) * 5 - 40

                # Add synthetic targets
                rf_dets = []
                for i in range(n):
                    r_bin = 10 + i * 12 + int(3 * np.sin(frame_idx * 0.05 + i))
                    d_bin = 16 + int(4 * np.sin(frame_idx * 0.08 + i * 2))
                    r_bin = min(r_bin, 63)
                    d_bin = min(d_bin, 31)
                    rd[r_bin, d_bin] += 30
                    rd[max(0, r_bin - 1) : r_bin + 2, max(0, d_bin - 1) : d_bin + 2] += 15

                    rf_dets.append({
                        "range_m": float(r_bin * 200.0 / 64),
                        "velocity_mps": float((d_bin - 16) * 0.5),
                        "snr_db": float(rng.uniform(10, 25)),
                        "range_bin": r_bin,
                        "doppler_bin": d_bin,
                    })

                STATE["rd_matrix"] = rd
                STATE["rf_detections"] = rf_dets
                STATE["rf_tracks"] = rf_dets

                frame_idx += 1
                time.sleep(0.1)  # 10 Hz
            except Exception as e:
                print(f"[RF Pipeline] Error: {e}")
                time.sleep(1)
        return

    # Full RF pipeline with SDK
    cfg = WaveformConfig()
    tracker = KalmanRFTracker()

    while True:
        try:
            n = STATE["n_targets"]
            scenario = make_test_scenario(n_targets=n, seed=STATE["scenario_seed"])

            # Drift target ranges slightly
            for t in scenario.targets:
                t.range_m += 2 * np.sin(frame_idx * 0.05)

            tx_signal, freq_grid, rg, _ = generate_ofdm_frame(cfg, seed=frame_idx, validate=False)
            rx_grid, channel = simulate_echoes(freq_grid, scenario, cfg, seed=frame_idx)
            rd_map, r_axis, d_axis, detections = process_slot(rx_grid, freq_grid, cfg)

            tracks = tracker.update(detections)

            STATE["rd_matrix"] = rd_map[:64, :32] if rd_map.shape[0] > 64 else rd_map
            STATE["range_axis"] = r_axis
            STATE["doppler_axis"] = d_axis
            STATE["rf_detections"] = [
                {
                    "range_m": d.range_m,
                    "velocity_mps": d.velocity_mps,
                    "snr_db": d.snr_db,
                    "range_bin": d.range_bin,
                    "doppler_bin": d.doppler_bin,
                }
                for d in detections
            ]
            STATE["rf_tracks"] = [
                {
                    "track_id": t.track_id,
                    "range_m": t.range_m,
                    "velocity_mps": t.velocity_mps,
                    "snr_db": t.snr_db,
                    "confirmed": t.confirmed,
                }
                for t in tracks
            ]

            frame_idx += 1
            time.sleep(0.1)  # 10 Hz
        except Exception as e:
            print(f"[RF Pipeline] Error: {e}")
            time.sleep(1)


def _run_vision_pipeline():
    """Background thread: Vision pipeline at ~15 Hz."""
    STATE["vision_running"] = True
    cap = None
    video_source = STATE.get("video_source")

    if video_source and os.path.exists(video_source):
        cap = cv2.VideoCapture(video_source)

    frame_count = 0

    while True:
        try:
            # Get frame
            if cap is not None and cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ret, frame = cap.read()
                if not ret:
                    frame = _generate_synthetic_frame(
                        frame_count=frame_count, n_targets=STATE["n_targets"]
                    )
            else:
                frame = _generate_synthetic_frame(
                    frame_count=frame_count, n_targets=STATE["n_targets"]
                )

            # Apply degradation
            mode = STATE["degrade_mode"]
            intensity = STATE["degrade_intensity"]
            degraded = apply_degradation(frame, mode, intensity, seed=frame_count)

            # Detect
            cam_conf = get_camera_confidence(mode, intensity)
            detections = _detector.detect(degraded)

            # Depth estimation
            depth_map = estimate_depth_map(degraded)
            for det in detections:
                det.depth_m = bbox_to_depth_m(
                    det.bbox, depth_map, degraded.shape[0]
                )

            # Track
            v_tracks = _vision_tracker.update(detections)

            # Fuse with RF tracks
            fused_dets = []
            rf_tracks = STATE.get("rf_tracks", [])
            h, w = degraded.shape[:2]

            for vt in v_tracks:
                # Find nearest RF track by range
                best_rf = None
                best_dist = float("inf")
                if vt.depth_m is not None:
                    for rt in rf_tracks:
                        dist = abs(vt.depth_m - rt.get("range_m", 0))
                        if dist < best_dist and dist < 30:
                            best_dist = dist
                            best_rf = rt

                result = fuse(
                    rf_range_m=best_rf.get("range_m", 0) if best_rf else 0,
                    rf_velocity_mps=best_rf.get("velocity_mps", 0) if best_rf else 0,
                    rf_snr_db=best_rf.get("snr_db", 0) if best_rf else 0,
                    rf_confidence=0.9 if best_rf else 0,
                    rf_present=best_rf is not None,
                    vision_depth_m=vt.depth_m or 0,
                    vision_cx_norm=vt.cx / w,
                    vision_cy_norm=vt.cy / h,
                    vision_w_norm=vt.width / w,
                    vision_h_norm=vt.height / h,
                    vision_confidence=vt.confidence,
                    vision_present=True,
                    camera_weight=cam_conf,
                    rf_weight=1.0,
                )
                result["track_id"] = vt.track_id
                result["bbox"] = vt.bbox
                result["depth_m"] = vt.depth_m
                fused_dets.append(result)

            # Add RF-only tracks (no vision match)
            used_rf = {d.get("track_id") for d in fused_dets if d.get("source") == "fused"}
            for rt in rf_tracks:
                rf_tid = rt.get("track_id", -1)
                if rf_tid not in used_rf:
                    # RF-only detection — place bbox based on range
                    cx = w * 0.5  # Centre of frame
                    cy = h * 0.6
                    bw, bh_box = 50, 100
                    result = fuse(
                        rf_range_m=rt.get("range_m", 0),
                        rf_velocity_mps=rt.get("velocity_mps", 0),
                        rf_snr_db=rt.get("snr_db", 0),
                        rf_confidence=0.9,
                        rf_present=True,
                        camera_weight=cam_conf,
                        rf_weight=1.0,
                    )
                    result["track_id"] = rf_tid
                    result["bbox"] = (cx - bw // 2, cy - bh_box // 2, cx + bw // 2, cy + bh_box // 2)
                    fused_dets.append(result)

            STATE["fused_detections"] = fused_dets
            STATE["vision_tracks"] = [
                {
                    "track_id": t.track_id,
                    "bbox": t.bbox,
                    "confidence": t.confidence,
                    "depth_m": t.depth_m,
                }
                for t in v_tracks
            ]

            # Draw overlays
            annotated = _draw_overlays(degraded, fused_dets)

            # Encode JPEG
            _, buf = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 75])
            STATE["frame_b64"] = base64.b64encode(buf.tobytes()).decode("utf-8")
            STATE["frame_count"] = frame_count

            frame_count += 1
            time.sleep(1.0 / 15.0)  # 15 Hz

        except Exception as e:
            print(f"[Vision Pipeline] Error: {e}")
            time.sleep(1)


# ═════════════════════════════════════════════════════════════════════════════
# Startup
# ═════════════════════════════════════════════════════════════════════════════

@app.on_event("startup")
async def startup():
    # Start background threads
    rf_thread = threading.Thread(target=_run_rf_pipeline, daemon=True)
    vision_thread = threading.Thread(target=_run_vision_pipeline, daemon=True)
    rf_thread.start()
    vision_thread.start()
    print("\n✅ SenseForge backend started.")
    print(f"   SDK available: {_sdk_available}")
    print(f"   RF pipeline: {'Sionna + cuPHY' if _rf_available else 'Synthetic'}")
    print(f"   Vision pipeline: Starting...")
    print()


# ═════════════════════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    uvicorn.run(
        "backend.main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
    )
