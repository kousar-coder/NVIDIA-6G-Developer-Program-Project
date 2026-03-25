"""
SenseForge — Fusion MLP Model
══════════════════════════════
Late-fusion MLP that combines RF radar and camera detections.
14-dimensional input feature vector, 4-dimensional output
(fused_range, fused_velocity, fused_confidence, source_label).
"""

from __future__ import annotations

import os
from typing import Dict, Optional, Tuple

import numpy as np

INPUT_DIM = 14
"""
Feature vector layout (14 dims):
  RF Branch (5):
    [0] range_m / 150       — normalised range
    [1] (velocity + 5) / 10 — normalised velocity
    [2] (snr + 10) / 40     — normalised SNR
    [3] rf_confidence        — detection confidence [0,1]
    [4] rf_present           — 1.0 if RF detection exists, else 0.0

  Vision Branch (7):
    [5] depth_m / 150        — normalised depth
    [6] cx_norm              — bbox centre-x / frame_width
    [7] cy_norm              — bbox centre-y / frame_height
    [8] w_norm               — bbox width / frame_width
    [9] h_norm               — bbox height / frame_height
    [10] vision_confidence    — detection confidence [0,1]
    [11] vision_present       — 1.0 if vision detection exists

  Context (2):
    [12] camera_weight        — camera reliability weight [0,1]
    [13] rf_weight            — RF reliability weight [0,1]
"""

# ── Try to import PyTorch ─────────────────────────────────────────────────────
_HAS_TORCH = False
try:
    import torch
    import torch.nn as nn
    _HAS_TORCH = True
except ImportError:
    pass


# ═════════════════════════════════════════════════════════════════════════════
# PyTorch model
# ═════════════════════════════════════════════════════════════════════════════

if _HAS_TORCH:
    class FusionMLP(nn.Module):
        """
        Fusion MLP: 14 → 64 → 32 → 4
        With LayerNorm, GELU activations, and dropout.
        """

        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(INPUT_DIM, 64),
                nn.LayerNorm(64),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(64, 32),
                nn.GELU(),
                nn.Linear(32, 4),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.net(x)
else:
    # ── NumPy fallback ────────────────────────────────────────────────────
    class FusionMLP:  # type: ignore[no-redef]
        """NumPy fallback for FusionMLP when PyTorch is not available."""

        def __init__(self):
            rng = np.random.RandomState(0)
            self.w1 = rng.randn(INPUT_DIM, 64).astype(np.float32) * 0.1
            self.b1 = np.zeros(64, dtype=np.float32)
            self.w2 = rng.randn(64, 32).astype(np.float32) * 0.1
            self.b2 = np.zeros(32, dtype=np.float32)
            self.w3 = rng.randn(32, 4).astype(np.float32) * 0.1
            self.b3 = np.zeros(4, dtype=np.float32)

        def __call__(self, x: np.ndarray) -> np.ndarray:
            # Layer 1 + LayerNorm + GELU
            h = x @ self.w1 + self.b1
            h = (h - h.mean(-1, keepdims=True)) / (h.std(-1, keepdims=True) + 1e-5)
            h = h * 0.5 * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (h + 0.044715 * h ** 3)))
            # Layer 2 + GELU
            h = h @ self.w2 + self.b2
            h = h * 0.5 * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (h + 0.044715 * h ** 3)))
            # Layer 3
            out = h @ self.w3 + self.b3
            return out

        def eval(self):
            return self

        def parameters(self):
            return []

        def state_dict(self):
            return {}


# ═════════════════════════════════════════════════════════════════════════════
# Feature vector construction
# ═════════════════════════════════════════════════════════════════════════════

def build_feature_vector(
    # RF branch
    rf_range_m: float = 0.0,
    rf_velocity_mps: float = 0.0,
    rf_snr_db: float = 0.0,
    rf_confidence: float = 0.0,
    rf_present: bool = False,
    # Vision branch
    vision_depth_m: float = 0.0,
    vision_cx_norm: float = 0.0,
    vision_cy_norm: float = 0.0,
    vision_w_norm: float = 0.0,
    vision_h_norm: float = 0.0,
    vision_confidence: float = 0.0,
    vision_present: bool = False,
    # Context
    camera_weight: float = 1.0,
    rf_weight: float = 1.0,
) -> np.ndarray:
    """
    Build a normalised 14-dim feature vector for fusion.

    All values are clamped to [0, 1].
    """
    feat = np.array([
        # RF (5)
        np.clip(rf_range_m / 150.0, 0.0, 1.0),
        np.clip((rf_velocity_mps + 5.0) / 10.0, 0.0, 1.0),
        np.clip((rf_snr_db + 10.0) / 40.0, 0.0, 1.0),
        np.clip(rf_confidence, 0.0, 1.0),
        1.0 if rf_present else 0.0,
        # Vision (7)
        np.clip(vision_depth_m / 150.0, 0.0, 1.0),
        np.clip(vision_cx_norm, 0.0, 1.0),
        np.clip(vision_cy_norm, 0.0, 1.0),
        np.clip(vision_w_norm, 0.0, 1.0),
        np.clip(vision_h_norm, 0.0, 1.0),
        np.clip(vision_confidence, 0.0, 1.0),
        1.0 if vision_present else 0.0,
        # Context (2)
        np.clip(camera_weight, 0.0, 1.0),
        np.clip(rf_weight, 0.0, 1.0),
    ], dtype=np.float32)

    return feat


# ═════════════════════════════════════════════════════════════════════════════
# Model loading
# ═════════════════════════════════════════════════════════════════════════════

_model_instance: Optional[FusionMLP] = None


def load_model(model_path: Optional[str] = None) -> FusionMLP:
    """
    Load or create the fusion model.

    If model_path exists, load trained weights. Otherwise return
    a fresh model instance.
    """
    global _model_instance

    if _model_instance is not None:
        return _model_instance

    model = FusionMLP()

    if model_path is None:
        model_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "models",
            "fusion_model.pt",
        )

    if _HAS_TORCH and os.path.exists(model_path):
        try:
            state = torch.load(model_path, map_location="cpu", weights_only=True)
            model.load_state_dict(state)
            print(f"[Fusion] Loaded model from {model_path}")
        except Exception as e:
            print(f"[Fusion] Failed to load model: {e}")

    model.eval()
    _model_instance = model
    return model


# ═════════════════════════════════════════════════════════════════════════════
# Fusion function
# ═════════════════════════════════════════════════════════════════════════════

def fuse(
    rf_range_m: float = 0.0,
    rf_velocity_mps: float = 0.0,
    rf_snr_db: float = 0.0,
    rf_confidence: float = 0.0,
    rf_present: bool = False,
    vision_depth_m: float = 0.0,
    vision_cx_norm: float = 0.0,
    vision_cy_norm: float = 0.0,
    vision_w_norm: float = 0.0,
    vision_h_norm: float = 0.0,
    vision_confidence: float = 0.0,
    vision_present: bool = False,
    camera_weight: float = 1.0,
    rf_weight: float = 1.0,
) -> Dict:
    """
    Run weighted-average fusion with source labelling.

    Returns
    -------
    result : dict
        Keys: fused_range_m, fused_velocity_mps, fused_confidence,
              source ("fused", "rf_only", "vision_only", "none")
    """
    # Determine source
    if rf_present and vision_present:
        source = "fused"
    elif rf_present:
        source = "rf_only"
    elif vision_present:
        source = "vision_only"
    else:
        source = "none"

    # Weighted average fusion
    total_weight = 0.0
    fused_range = 0.0
    fused_velocity = 0.0
    fused_conf = 0.0

    if rf_present:
        w = rf_weight * rf_confidence
        fused_range += w * rf_range_m
        fused_velocity += w * rf_velocity_mps
        fused_conf += w * rf_confidence
        total_weight += w

    if vision_present:
        w = camera_weight * vision_confidence
        fused_range += w * vision_depth_m
        fused_velocity += w * 0.0  # Vision doesn't directly measure velocity
        fused_conf += w * vision_confidence
        total_weight += w

    if total_weight > 0:
        fused_range /= total_weight
        fused_velocity /= total_weight
        fused_conf /= total_weight
    else:
        fused_conf = 0.0

    # Clamp confidence
    fused_conf = max(0.0, min(1.0, fused_conf))

    return {
        "fused_range_m": float(fused_range),
        "fused_velocity_mps": float(fused_velocity),
        "fused_confidence": float(fused_conf),
        "source": source,
    }
