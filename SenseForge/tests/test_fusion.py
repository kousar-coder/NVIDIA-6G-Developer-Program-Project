"""
SenseForge — Fusion Test Suite
══════════════════════════════
28 tests covering feature vector normalisation, source label branches,
FusionMLP structure, and training data generator.
"""

import sys
import os
import pytest
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fusion.model import (
    FusionMLP,
    INPUT_DIM,
    build_feature_vector,
    fuse,
)
from fusion.train import generate_training_data


# ═══════════════════════════════════════════════════════════════════════
# Feature Vector Tests
# ═══════════════════════════════════════════════════════════════════════

class TestFeatureVector:

    def test_shape(self):
        feat = build_feature_vector()
        assert feat.shape == (INPUT_DIM,)
        assert INPUT_DIM == 14

    def test_dtype(self):
        feat = build_feature_vector()
        assert feat.dtype == np.float32

    def test_all_zeros_default(self):
        feat = build_feature_vector()
        # Without any inputs: rf_present=False, vision_present=False
        assert feat[4] == 0.0   # rf_present
        assert feat[11] == 0.0  # vision_present

    def test_rf_present_flag(self):
        feat = build_feature_vector(rf_present=True)
        assert feat[4] == 1.0

    def test_vision_present_flag(self):
        feat = build_feature_vector(vision_present=True)
        assert feat[11] == 1.0

    def test_range_normalisation(self):
        feat = build_feature_vector(rf_range_m=75, rf_present=True)
        assert abs(feat[0] - 0.5) < 1e-6

    def test_velocity_normalisation(self):
        feat = build_feature_vector(rf_velocity_mps=0.0, rf_present=True)
        assert abs(feat[1] - 0.5) < 1e-6  # (0+5)/10 = 0.5

    def test_snr_normalisation(self):
        feat = build_feature_vector(rf_snr_db=10.0, rf_present=True)
        assert abs(feat[2] - 0.5) < 1e-6  # (10+10)/40 = 0.5

    def test_clamp_high(self):
        feat = build_feature_vector(rf_range_m=300, rf_present=True)
        assert feat[0] == 1.0  # Clamped

    def test_clamp_low(self):
        feat = build_feature_vector(rf_range_m=-50, rf_present=True)
        assert feat[0] == 0.0  # Clamped

    def test_all_values_in_range(self):
        feat = build_feature_vector(
            rf_range_m=50, rf_velocity_mps=2, rf_snr_db=15,
            rf_confidence=0.9, rf_present=True,
            vision_depth_m=40, vision_cx_norm=0.5, vision_cy_norm=0.5,
            vision_w_norm=0.1, vision_h_norm=0.3, vision_confidence=0.8,
            vision_present=True,
            camera_weight=0.5, rf_weight=1.0,
        )
        assert np.all(feat >= 0.0)
        assert np.all(feat <= 1.0)


# ═══════════════════════════════════════════════════════════════════════
# Fusion Function Tests
# ═══════════════════════════════════════════════════════════════════════

class TestFuse:

    def test_both_present_fused(self):
        result = fuse(
            rf_range_m=50, rf_confidence=0.8, rf_present=True,
            vision_depth_m=48, vision_confidence=0.7, vision_present=True,
        )
        assert result["source"] == "fused"

    def test_rf_only(self):
        result = fuse(rf_range_m=50, rf_confidence=0.8, rf_present=True)
        assert result["source"] == "rf_only"

    def test_vision_only(self):
        result = fuse(vision_depth_m=50, vision_confidence=0.8, vision_present=True)
        assert result["source"] == "vision_only"

    def test_none_source(self):
        result = fuse()
        assert result["source"] == "none"

    def test_fused_range_between(self):
        result = fuse(
            rf_range_m=50, rf_confidence=0.8, rf_present=True,
            vision_depth_m=60, vision_confidence=0.8, vision_present=True,
        )
        assert 50 <= result["fused_range_m"] <= 60

    def test_confidence_positive(self):
        result = fuse(rf_range_m=50, rf_confidence=0.8, rf_present=True)
        assert result["fused_confidence"] > 0

    def test_confidence_clamped(self):
        result = fuse(
            rf_range_m=50, rf_confidence=2.0, rf_present=True,
        )
        assert result["fused_confidence"] <= 1.0

    def test_rf_only_uses_rf_range(self):
        result = fuse(rf_range_m=75, rf_confidence=0.9, rf_present=True)
        assert abs(result["fused_range_m"] - 75) < 1e-6

    def test_camera_weight_affects_fusion(self):
        r1 = fuse(
            rf_range_m=50, rf_confidence=0.5, rf_present=True,
            vision_depth_m=80, vision_confidence=0.5, vision_present=True,
            camera_weight=1.0,
        )
        r2 = fuse(
            rf_range_m=50, rf_confidence=0.5, rf_present=True,
            vision_depth_m=80, vision_confidence=0.5, vision_present=True,
            camera_weight=0.1,
        )
        # Lower camera weight → fused range closer to RF
        assert abs(r2["fused_range_m"] - 50) < abs(r1["fused_range_m"] - 50)


# ═══════════════════════════════════════════════════════════════════════
# FusionMLP Tests
# ═══════════════════════════════════════════════════════════════════════

class TestFusionMLP:

    def test_instantiation(self):
        model = FusionMLP()
        assert model is not None

    def test_output_dim(self):
        model = FusionMLP()
        try:
            import torch
            x = torch.randn(1, INPUT_DIM)
            out = model(x)
            assert out.shape == (1, 4)
        except ImportError:
            x = np.random.randn(1, INPUT_DIM).astype(np.float32)
            out = model(x)
            assert out.shape == (1, 4)

    def test_batch(self):
        model = FusionMLP()
        try:
            import torch
            x = torch.randn(16, INPUT_DIM)
            out = model(x)
            assert out.shape == (16, 4)
        except ImportError:
            x = np.random.randn(16, INPUT_DIM).astype(np.float32)
            out = model(x)
            assert out.shape == (16, 4)


# ═══════════════════════════════════════════════════════════════════════
# Training Data Tests
# ═══════════════════════════════════════════════════════════════════════

class TestTrainingData:

    def test_shape(self):
        X, Y = generate_training_data(n=100, seed=42)
        assert X.shape == (100, 14)
        assert Y.shape == (100, 4)

    def test_dtype(self):
        X, Y = generate_training_data(n=50)
        assert X.dtype == np.float32
        assert Y.dtype == np.float32

    def test_deterministic(self):
        X1, Y1 = generate_training_data(n=50, seed=42)
        X2, Y2 = generate_training_data(n=50, seed=42)
        assert np.array_equal(X1, X2)
        assert np.array_equal(Y1, Y2)

    def test_values_in_range(self):
        X, Y = generate_training_data(n=200, seed=42)
        assert np.all(X >= 0.0)
        assert np.all(X <= 1.0)
        assert np.all(Y[:, 0] >= 0.0)  # range/150
        assert np.all(Y[:, 0] <= 1.0)

    def test_source_variety(self):
        X, Y = generate_training_data(n=1000, seed=42)
        # Should have multiple source types
        source_values = set(np.round(Y[:, 3], 2))
        assert len(source_values) >= 2
