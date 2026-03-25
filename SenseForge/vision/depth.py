"""
SenseForge — Monocular Depth Estimation
════════════════════════════════════════
MiDaS-based monocular depth estimation.
Maps relative inverse-depth to metric depth using heuristics.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np


# ── Global state for lazy-loaded model ────────────────────────────────────────
_midas_model = None
_midas_transform = None
_midas_device = None


def _load_midas():
    """Load MiDaS small model via torch.hub."""
    global _midas_model, _midas_transform, _midas_device

    if _midas_model is not None:
        return

    try:
        import torch

        _midas_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        _midas_model = torch.hub.load(
            "intel-isl/MiDaS",
            "MiDaS_small",
            trust_repo=True,
        )
        _midas_model.eval()
        _midas_model.to(_midas_device)

        midas_transforms = torch.hub.load(
            "intel-isl/MiDaS",
            "transforms",
            trust_repo=True,
        )
        _midas_transform = midas_transforms.small_transform

    except Exception as e:
        print(f"[Depth] MiDaS load failed: {e}")
        _midas_model = None


def estimate_depth_map(frame_bgr: np.ndarray) -> Optional[np.ndarray]:
    """
    Estimate relative depth map from a BGR frame.

    Parameters
    ----------
    frame_bgr : np.ndarray
        BGR image, shape (H, W, 3), uint8.

    Returns
    -------
    depth_map : np.ndarray or None
        HxW float32 inverse-depth map (higher = closer).
        Returns None if MiDaS is unavailable.
    """
    _load_midas()

    if _midas_model is None or _midas_transform is None:
        return None

    try:
        import torch
        import cv2

        # Convert BGR → RGB
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        # Apply MiDaS transform
        input_batch = _midas_transform(rgb).to(_midas_device)

        with torch.no_grad():
            prediction = _midas_model(input_batch)

        # Resize to original dimensions
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=frame_bgr.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()

        depth_map = prediction.cpu().numpy().astype(np.float32)
        return depth_map

    except Exception as e:
        print(f"[Depth] Estimation failed: {e}")
        return None


def bbox_to_depth_m(
    bbox: Tuple[float, float, float, float],
    depth_map: Optional[np.ndarray],
    frame_h: int,
    max_range: float = 150.0,
) -> float:
    """
    Estimate metric depth for a bounding box.

    Strategy:
    1. If depth_map available: sample torso-region median,
       map inverse-depth to metres.
    2. Fallback heuristic: focal_proxy / bbox_height_fraction.

    Parameters
    ----------
    bbox : tuple of float
        (x1, y1, x2, y2) in pixel coordinates.
    depth_map : np.ndarray or None
        HxW float32 inverse-depth from MiDaS.
    frame_h : int
        Frame height in pixels.
    max_range : float
        Maximum range clamp.

    Returns
    -------
    depth_m : float
        Estimated depth in metres.
    """
    x1, y1, x2, y2 = bbox
    bw = x2 - x1
    bh = y2 - y1

    if depth_map is not None and bh > 5 and bw > 3:
        h_map, w_map = depth_map.shape

        # Sample torso region (middle 40-70% of bbox height)
        torso_y1 = int(max(0, y1 + bh * 0.40))
        torso_y2 = int(min(h_map, y1 + bh * 0.70))
        torso_x1 = int(max(0, x1 + bw * 0.25))
        torso_x2 = int(min(w_map, x2 - bw * 0.25))

        if torso_y2 > torso_y1 and torso_x2 > torso_x1:
            region = depth_map[torso_y1:torso_y2, torso_x1:torso_x2]
            if region.size > 0:
                inv_depth = float(np.median(region))
                if inv_depth > 0:
                    # Map inverse-depth to metres
                    # MiDaS outputs relative values; scale heuristically
                    depth_m = min(max_range, 1000.0 / (inv_depth + 1e-6))
                    return max(0.5, depth_m)

    # ── Heuristic fallback: person height ≈ 1.7m ─────────────────────
    if bh < 5:
        return max_range

    bbox_h_frac = bh / frame_h
    focal_proxy = 0.5  # Focal length proxy (normalised)
    person_height_m = 1.7

    depth_m = person_height_m * focal_proxy / max(bbox_h_frac, 0.01)
    depth_m = min(max_range, max(0.5, depth_m))
    return depth_m
