"""
SenseForge — Demo Video Generator
══════════════════════════════════
Generates synthetic 640x480 MP4 video using OpenCV.
Scene: dark grid background, sky gradient, building silhouettes,
animated pedestrian blobs with pendulum walking animation.

Usage:
  python scripts/demo_generator.py --seconds 30 --fps 30 --variant day --n_peds 3 --out demo.mp4
"""

import argparse
import math
import os
import sys

import cv2
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def generate_sky(width, height, variant="day"):
    """Generate sky gradient based on time-of-day variant."""
    sky = np.zeros((height // 2, width, 3), dtype=np.uint8)
    for y in range(sky.shape[0]):
        t = y / sky.shape[0]
        if variant == "day":
            b = int(180 - t * 80)
            g = int(140 - t * 60)
            r = int(80 - t * 40)
        elif variant == "dusk":
            b = int(40 + t * 30)
            g = int(30 + t * 50)
            r = int(120 - t * 60)
        else:  # night
            b = int(15 + t * 10)
            g = int(10 + t * 8)
            r = int(5 + t * 5)
        sky[y, :] = [max(0, b), max(0, g), max(0, r)]
    return sky


def draw_buildings(frame, width, height, variant="day"):
    """Draw building silhouettes on the horizon."""
    buildings = [
        (30, 50, 190),
        (100, 60, 210),
        (200, 70, 170),
        (310, 55, 230),
        (400, 45, 160),
        (470, 80, 250),
        (560, 60, 200),
    ]
    horizon = height // 2

    for bx, bw, bh in buildings:
        if bx + bw > width:
            bw = width - bx
        y_top = max(0, horizon - bh)
        if variant == "night":
            color = (12, 15, 8)
        elif variant == "dusk":
            color = (25, 30, 20)
        else:
            color = (60, 65, 55)
        cv2.rectangle(frame, (bx, y_top), (bx + bw, horizon), color, -1)

        # Windows
        win_color = (50, 70, 100) if variant != "night" else (35, 55, 80)
        for wy in range(y_top + 8, horizon - 5, 18):
            for wx in range(bx + 8, bx + bw - 8, 16):
                if np.random.random() > 0.3:
                    cv2.rectangle(frame, (wx, wy), (wx + 8, wy + 10), win_color, -1)


def draw_ground(frame, width, height, variant="day"):
    """Draw ground with grid lines."""
    horizon = height // 2
    if variant == "night":
        ground_color = (12, 15, 8)
        grid_color = (20, 25, 15)
    elif variant == "dusk":
        ground_color = (30, 35, 25)
        grid_color = (40, 45, 35)
    else:
        ground_color = (45, 55, 35)
        grid_color = (55, 65, 45)

    frame[horizon:, :] = ground_color

    for x in range(0, width, 40):
        cv2.line(frame, (x, horizon), (x, height), grid_color, 1)
    for y in range(horizon, height, 20):
        cv2.line(frame, (0, y), (width, y), grid_color, 1)


def draw_pedestrian(frame, cx, cy, frame_idx, ped_id, variant="day"):
    """Draw animated pedestrian with pendulum walking."""
    if variant == "night":
        body_color = (50, 60, 75)
        head_color = (65, 75, 90)
    elif variant == "dusk":
        body_color = (60, 75, 90)
        head_color = (75, 90, 105)
    else:
        body_color = (80, 100, 130)
        head_color = (100, 120, 150)

    leg_offset = int(8 * math.sin(frame_idx * 0.15 + ped_id * 0.7))
    arm_offset = int(6 * math.sin(frame_idx * 0.15 + ped_id * 0.7 + math.pi))

    # Body (ellipse)
    cv2.ellipse(frame, (cx, cy - 20), (10, 25), 0, 0, 360, body_color, -1)

    # Head
    cv2.circle(frame, (cx, cy - 50), 9, head_color, -1)

    # Legs
    cv2.line(frame, (cx, cy + 5), (cx - 7 + leg_offset, cy + 35), body_color, 3)
    cv2.line(frame, (cx, cy + 5), (cx + 7 - leg_offset, cy + 35), body_color, 3)

    # Arms
    cv2.line(frame, (cx, cy - 15), (cx - 12 + arm_offset, cy + 10), body_color, 2)
    cv2.line(frame, (cx, cy - 15), (cx + 12 - arm_offset, cy + 10), body_color, 2)


def generate_video(
    output_path: str,
    seconds: float = 30.0,
    fps: int = 30,
    width: int = 640,
    height: int = 480,
    variant: str = "day",
    n_peds: int = 3,
):
    """Generate the demo video."""
    total_frames = int(seconds * fps)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    if not writer.isOpened():
        print(f"ERROR: Cannot open video writer for {output_path}")
        return

    sky = generate_sky(width, height, variant)

    print(f"Generating {variant} video: {width}x{height} @ {fps}fps, {seconds}s, {n_peds} pedestrians")
    print(f"Output: {output_path}")

    for fi in range(total_frames):
        frame = np.zeros((height, width, 3), dtype=np.uint8)

        # Sky
        frame[: height // 2] = sky

        # Ground
        draw_ground(frame, width, height, variant)

        # Buildings
        draw_buildings(frame, width, height, variant)

        # Pedestrians
        for i in range(n_peds):
            px = int(80 + i * (width - 160) / max(n_peds - 1, 1) + 25 * math.sin(fi * 0.025 + i * 1.3))
            py = height // 2 + 40 + i * 15
            draw_pedestrian(frame, px, py, fi, i, variant)

        # Camera shake
        shake_dx = np.random.uniform(-0.5, 0.5)
        shake_dy = np.random.uniform(-0.3, 0.3)
        M = np.float32([[1, 0, shake_dx], [0, 1, shake_dy]])
        frame = cv2.warpAffine(frame, M, (width, height))

        writer.write(frame)

        if fi % (fps * 5) == 0:
            pct = fi / total_frames * 100
            print(f"  Progress: {pct:.0f}%")

    writer.release()
    print(f"  Done! {total_frames} frames written.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SenseForge Demo Video Generator")
    parser.add_argument("--seconds", type=float, default=30.0, help="Video duration")
    parser.add_argument("--fps", type=int, default=30, help="Frames per second")
    parser.add_argument("--variant", choices=["day", "dusk", "night"], default="day")
    parser.add_argument("--n_peds", type=int, default=3, help="Number of pedestrians")
    parser.add_argument("--out", type=str, default=None, help="Output path")
    args = parser.parse_args()

    if args.out is None:
        os.makedirs(os.path.join(PROJECT_ROOT, "data", "videos"), exist_ok=True)
        args.out = os.path.join(PROJECT_ROOT, "data", "videos", f"demo_{args.variant}.mp4")

    generate_video(
        output_path=args.out,
        seconds=args.seconds,
        fps=args.fps,
        variant=args.variant,
        n_peds=args.n_peds,
    )
