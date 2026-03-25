"""
SenseForge — Camera Degradation Simulator
═════════════════════════════════════════
Simulates adverse weather and visibility conditions:
fog, night, occlusion, rain.

This is the KEY demo mechanism — when camera degrades,
RF radar continues unaffected, proving ISAC value.
"""

from __future__ import annotations

import enum
from typing import Tuple

import cv2
import numpy as np


class DegradationMode(str, enum.Enum):
    """Camera degradation conditions."""

    CLEAR = "clear"
    FOG = "fog"
    NIGHT = "night"
    OCCLUSION = "occlusion"
    RAIN = "rain"


def apply_degradation(
    frame: np.ndarray,
    mode: DegradationMode,
    intensity: float = 0.5,
    seed: int = 42,
) -> np.ndarray:
    """
    Apply visual degradation to a camera frame.

    Parameters
    ----------
    frame : np.ndarray
        BGR image, uint8.
    mode : DegradationMode
        Type of degradation.
    intensity : float
        Severity in [0, 1]. 0 = none, 1 = maximum.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    degraded : np.ndarray
        Degraded BGR image, uint8.
    """
    intensity = max(0.0, min(1.0, intensity))
    rng = np.random.RandomState(seed)

    if mode == DegradationMode.CLEAR:
        return frame.copy()

    elif mode == DegradationMode.FOG:
        return _apply_fog(frame, intensity, rng)

    elif mode == DegradationMode.NIGHT:
        return _apply_night(frame, intensity, rng)

    elif mode == DegradationMode.OCCLUSION:
        return _apply_occlusion(frame, intensity, rng)

    elif mode == DegradationMode.RAIN:
        return _apply_rain(frame, intensity, rng)

    return frame.copy()


def _apply_fog(frame: np.ndarray, intensity: float, rng: np.random.RandomState) -> np.ndarray:
    """
    Koschmieder fog model: I_fog = I * (1 - t) + 255 * t
    Plus Gaussian blur for reduced sharpness.
    """
    t = intensity * 0.85  # Transmission factor
    fog_frame = frame.astype(np.float32) * (1.0 - t) + 255.0 * t

    # Gaussian blur for fog scatter
    ksize = max(3, int(intensity * 15) | 1)  # Odd kernel
    fog_frame = cv2.GaussianBlur(fog_frame.astype(np.uint8), (ksize, ksize), 0)

    return np.clip(fog_frame, 0, 255).astype(np.uint8)


def _apply_night(frame: np.ndarray, intensity: float, rng: np.random.RandomState) -> np.ndarray:
    """
    Night simulation: gamma darkening + Gaussian noise.
    """
    # Gamma correction → darken
    gamma = 1.0 + intensity * 4.0  # gamma up to 5.0
    inv_gamma = 1.0 / gamma
    table = np.array([
        ((i / 255.0) ** inv_gamma) * 255 for i in range(256)
    ]).astype(np.uint8)
    dark = cv2.LUT(frame, table)

    # Add sensor noise (Gaussian)
    noise_sigma = intensity * 25.0
    noise = rng.randn(*dark.shape).astype(np.float32) * noise_sigma
    noisy = dark.astype(np.float32) + noise

    return np.clip(noisy, 0, 255).astype(np.uint8)


def _apply_occlusion(frame: np.ndarray, intensity: float, rng: np.random.RandomState) -> np.ndarray:
    """
    Random dark semi-transparent rectangles simulating occlusions.
    """
    result = frame.copy()
    h, w = result.shape[:2]

    n_rects = max(1, int(intensity * 6))

    for _ in range(n_rects):
        rw = rng.randint(int(w * 0.1), int(w * 0.4))
        rh = rng.randint(int(h * 0.1), int(h * 0.5))
        rx = rng.randint(0, max(1, w - rw))
        ry = rng.randint(0, max(1, h - rh))

        alpha = 0.3 + intensity * 0.5  # Semi-transparent → near-opaque
        overlay = result.copy()
        cv2.rectangle(overlay, (rx, ry), (rx + rw, ry + rh), (10, 10, 10), -1)
        result = cv2.addWeighted(overlay, alpha, result, 1 - alpha, 0)

    return result


def _apply_rain(frame: np.ndarray, intensity: float, rng: np.random.RandomState) -> np.ndarray:
    """
    Rain simulation: diagonal motion-blur streaks + brightness reduction.
    """
    result = frame.copy()
    h, w = result.shape[:2]

    # Create rain streak layer
    rain = np.zeros((h, w), dtype=np.uint8)
    n_drops = int(intensity * 500) + 50

    for _ in range(n_drops):
        x = rng.randint(0, w)
        y = rng.randint(0, h)
        length = rng.randint(10, int(30 * intensity + 15))
        thickness = rng.choice([1, 1, 1, 2])
        # Diagonal streaks
        x2 = x + length // 3
        y2 = y + length
        cv2.line(rain, (x, y), (x2, min(y2, h - 1)), 200, thickness)

    # Motion blur on rain layer
    ksize = max(3, int(intensity * 7) | 1)
    kernel = np.zeros((ksize, ksize), dtype=np.float32)
    # Diagonal kernel
    for i in range(ksize):
        kernel[i, min(i, ksize - 1)] = 1.0 / ksize
    rain = cv2.filter2D(rain, -1, kernel)

    # Blend rain onto frame
    rain_3ch = cv2.cvtColor(rain, cv2.COLOR_GRAY2BGR)
    alpha = intensity * 0.4
    result = cv2.addWeighted(result, 1.0, rain_3ch, alpha, 0)

    # Reduce brightness
    brightness_factor = 1.0 - intensity * 0.3
    result = (result.astype(np.float32) * brightness_factor).astype(np.uint8)

    return np.clip(result, 0, 255).astype(np.uint8)


def get_camera_confidence(
    mode: DegradationMode,
    intensity: float = 0.5,
) -> float:
    """
    Get estimated camera detection confidence under degradation.

    Returns a value in [0.02, 1.0].
    """
    intensity = max(0.0, min(1.0, intensity))

    if mode == DegradationMode.CLEAR:
        return 1.0
    elif mode == DegradationMode.FOG:
        return max(0.02, 1.0 - intensity * 0.9)
    elif mode == DegradationMode.NIGHT:
        return max(0.02, 1.0 - intensity * 0.95)
    elif mode == DegradationMode.OCCLUSION:
        return max(0.02, 1.0 - intensity * 0.7)
    elif mode == DegradationMode.RAIN:
        return max(0.02, 1.0 - intensity * 0.6)
    return 1.0
