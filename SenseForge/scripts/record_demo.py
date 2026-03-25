"""
SenseForge — Demo Recorder
═══════════════════════════
Connects to all 3 WebSockets while backend runs.
Cycles through 5 degradation stages and records a composite output.

Stages:
  clear(8s) → fog(8s) → night(8s) → occlusion(8s) → clear(4s)

Output: 1280x480 composite (camera 640 + radar 320 + info 320)
"""

import argparse
import base64
import json
import os
import sys
import time
import threading

import cv2
import numpy as np
import requests

try:
    import websocket
except ImportError:
    print("Install websocket-client: pip install websocket-client")
    sys.exit(1)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Global data stores
latest_frame = None
latest_rd = None
latest_detections = []
frame_lock = threading.Lock()


def ws_video_thread(url):
    global latest_frame
    def on_message(ws, message):
        global latest_frame
        try:
            data = json.loads(message)
            if data.get("frame"):
                img_bytes = base64.b64decode(data["frame"])
                arr = np.frombuffer(img_bytes, dtype=np.uint8)
                with frame_lock:
                    latest_frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        except Exception:
            pass
    def on_error(ws, error):
        pass
    def on_close(ws, *args):
        pass

    ws = websocket.WebSocketApp(
        f"{url}/ws/video",
        on_message=on_message,
        on_error=on_error,
        on_close=on_close,
    )
    ws.run_forever()


def ws_radar_thread(url):
    global latest_rd
    def on_message(ws, message):
        global latest_rd
        try:
            data = json.loads(message)
            if data.get("rd_matrix"):
                with frame_lock:
                    latest_rd = data
        except Exception:
            pass
    ws = websocket.WebSocketApp(
        f"{url}/ws/radar",
        on_message=lambda ws, msg: on_message(ws, msg),
    )
    ws.run_forever()


def ws_detections_thread(url):
    global latest_detections
    def on_message(ws, message):
        global latest_detections
        try:
            data = json.loads(message)
            if data.get("detections"):
                with frame_lock:
                    latest_detections = data["detections"]
        except Exception:
            pass
    ws = websocket.WebSocketApp(
        f"{url}/ws/detections",
        on_message=lambda ws, msg: on_message(ws, msg),
    )
    ws.run_forever()


def draw_info_panel(width, height, stage, mode, n_detections):
    """Draw info panel with stage label, mode badge, detection count."""
    panel = np.zeros((height, width, 3), dtype=np.uint8)
    panel[:] = (13, 7, 5)  # Dark background

    # Stage label
    cv2.putText(panel, "SenseForge", (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 182, 56), 2)
    cv2.putText(panel, "ISAC Demo Recording", (20, 55),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (128, 144, 168), 1)

    # Mode badge
    mode_color = (136, 255, 0) if mode == "clear" else (68, 68, 255)
    cv2.putText(panel, f"MODE: {mode.upper()}", (20, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, mode_color, 1)

    # Stage indicator
    cv2.putText(panel, f"STAGE: {stage}", (20, 120),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (192, 200, 216), 1)

    # Detection count bar
    cv2.putText(panel, f"DETECTIONS: {n_detections}", (20, 155),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 136), 1)
    bar_w = min(260, n_detections * 40)
    cv2.rectangle(panel, (20, 165), (20 + bar_w, 175), (0, 255, 136), -1)

    # NVIDIA badge
    cv2.putText(panel, "NVIDIA AI AERIAL", (20, height - 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (118, 185, 0), 1)
    cv2.putText(panel, "Sionna + cuPHY + YOLOv8", (20, height - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (74, 90, 112), 1)

    return panel


def render_rd_panel(rd_data, width, height):
    """Render simplified range-Doppler heatmap."""
    panel = np.zeros((height, width, 3), dtype=np.uint8)

    if rd_data and rd_data.get("rd_matrix"):
        matrix = np.array(rd_data["rd_matrix"], dtype=np.float32)
        # Normalize
        mn, mx = matrix.min(), matrix.max()
        if mx > mn:
            norm = ((matrix - mn) / (mx - mn) * 255).astype(np.uint8)
        else:
            norm = np.zeros_like(matrix, dtype=np.uint8)

        # Apply colormap and resize
        colored = cv2.applyColorMap(norm, cv2.COLORMAP_INFERNO)
        colored = cv2.resize(colored, (width, height), interpolation=cv2.INTER_NEAREST)
        return colored

    cv2.putText(panel, "Awaiting RF...", (width // 4, height // 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (74, 90, 112), 1)
    return panel


def record_demo(
    backend_url: str = "http://localhost:8000",
    ws_url: str = "ws://localhost:8000",
    output_path: str = None,
    fps: int = 15,
):
    """Record a demo cycling through degradation stages."""
    if output_path is None:
        os.makedirs(os.path.join(PROJECT_ROOT, "data", "videos"), exist_ok=True)
        output_path = os.path.join(PROJECT_ROOT, "data", "videos", "demo_recording.mp4")

    stages = [
        ("clear", 8),
        ("fog", 8),
        ("night", 8),
        ("occlusion", 8),
        ("clear", 4),
    ]

    out_w, out_h = 1280, 480
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (out_w, out_h))

    if not writer.isOpened():
        print(f"ERROR: Cannot open video writer for {output_path}")
        return

    # Start WebSocket threads
    t1 = threading.Thread(target=ws_video_thread, args=(ws_url,), daemon=True)
    t2 = threading.Thread(target=ws_radar_thread, args=(ws_url,), daemon=True)
    t3 = threading.Thread(target=ws_detections_thread, args=(ws_url,), daemon=True)
    t1.start()
    t2.start()
    t3.start()

    print(f"Recording demo to {output_path}")
    print(f"Output: {out_w}x{out_h} @ {fps}fps")

    # Wait for first frame
    print("Waiting for data...")
    time.sleep(2)

    for stage_mode, stage_duration in stages:
        print(f"\n  Stage: {stage_mode.upper()} ({stage_duration}s)")

        # POST degradation mode
        try:
            requests.post(f"{backend_url}/degrade", json={
                "mode": stage_mode,
                "intensity": 0.7,
            })
        except Exception as e:
            print(f"  Warning: Could not set mode: {e}")

        n_frames = stage_duration * fps
        for fi in range(n_frames):
            with frame_lock:
                cam_frame = latest_frame.copy() if latest_frame is not None else np.zeros((480, 640, 3), dtype=np.uint8)
                rd_data = latest_rd
                dets = latest_detections.copy() if latest_detections else []

            # Camera (640x480)
            cam_resized = cv2.resize(cam_frame, (640, 480))

            # Radar (320x480)
            rd_panel = render_rd_panel(rd_data, 320, 480)

            # Info (320x480)
            info_panel = draw_info_panel(320, 480, stage_mode.upper(), stage_mode, len(dets))

            # Composite
            composite = np.hstack([cam_resized, rd_panel, info_panel])
            writer.write(composite)

            time.sleep(1.0 / fps)

    writer.release()
    print(f"\n  Done! Recording saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SenseForge Demo Recorder")
    parser.add_argument("--backend", default="http://localhost:8000")
    parser.add_argument("--ws", default="ws://localhost:8000")
    parser.add_argument("--out", default=None)
    parser.add_argument("--fps", type=int, default=15)
    args = parser.parse_args()

    record_demo(
        backend_url=args.backend,
        ws_url=args.ws,
        output_path=args.out,
        fps=args.fps,
    )
