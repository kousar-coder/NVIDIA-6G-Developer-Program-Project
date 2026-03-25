"""
SenseForge — Fusion Training Script
═════════════════════════════════════
Generates synthetic training data and trains the FusionMLP model.
Can be run standalone: python -m fusion.train
"""

from __future__ import annotations

import os
import sys
import time
from typing import Tuple

import numpy as np

# Ensure project root on path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from fusion.model import FusionMLP, INPUT_DIM, build_feature_vector  # noqa: E402


def generate_training_data(
    n: int = 8000,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic fusion training data.

    Ground truth:
        range: [5, 150] m
        velocity: [-5, 5] m/s

    RF branch: add Gaussian noise scaled by SNR.
    Camera branch: randomly degraded (fog/night/occlusion) in 60% of samples.

    Labels: (fused_range/150, (fused_vel+5)/10, fused_conf, source_idx)

    Returns
    -------
    X : np.ndarray, shape (n, 14)
    Y : np.ndarray, shape (n, 4)
    """
    rng = np.random.RandomState(seed)

    X = np.zeros((n, INPUT_DIM), dtype=np.float32)
    Y = np.zeros((n, 4), dtype=np.float32)

    for i in range(n):
        # Ground truth
        gt_range = rng.uniform(5.0, 150.0)
        gt_velocity = rng.uniform(-5.0, 5.0)

        # ── RF branch ─────────────────────────────────────────────────────
        snr = rng.uniform(-5.0, 30.0)
        snr_factor = max(0.1, snr / 30.0)
        rf_present = rng.random() > 0.1  # 90% RF detection rate

        if rf_present:
            rf_range = gt_range + rng.randn() * (5.0 / max(snr_factor, 0.1))
            rf_velocity = gt_velocity + rng.randn() * (0.5 / max(snr_factor, 0.1))
            rf_conf = min(1.0, 0.5 + snr_factor * 0.5)
        else:
            rf_range = 0.0
            rf_velocity = 0.0
            rf_conf = 0.0

        # ── Camera branch ─────────────────────────────────────────────────
        degraded = rng.random() < 0.6  # 60% degraded
        cam_weight = 1.0

        if degraded:
            deg_type = rng.choice(["fog", "night", "occlusion"])
            deg_intensity = rng.uniform(0.3, 1.0)
            if deg_type == "fog":
                cam_weight = max(0.02, 1.0 - deg_intensity * 0.9)
            elif deg_type == "night":
                cam_weight = max(0.02, 1.0 - deg_intensity * 0.95)
            else:
                cam_weight = max(0.02, 1.0 - deg_intensity * 0.7)

        vision_present = rng.random() < cam_weight  # Detection dependent on visibility

        if vision_present:
            depth_noise = rng.randn() * (10.0 if degraded else 3.0)
            vision_depth = max(0.5, gt_range + depth_noise)
            vision_conf = cam_weight * rng.uniform(0.4, 1.0)
            cx_norm = rng.uniform(0.1, 0.9)
            cy_norm = rng.uniform(0.2, 0.8)
            w_norm = rng.uniform(0.03, 0.2)
            h_norm = rng.uniform(0.1, 0.5)
        else:
            vision_depth = 0.0
            vision_conf = 0.0
            cx_norm = 0.0
            cy_norm = 0.0
            w_norm = 0.0
            h_norm = 0.0

        # Build feature vector
        feat = build_feature_vector(
            rf_range_m=rf_range if rf_present else 0.0,
            rf_velocity_mps=rf_velocity if rf_present else 0.0,
            rf_snr_db=snr if rf_present else 0.0,
            rf_confidence=rf_conf,
            rf_present=rf_present,
            vision_depth_m=vision_depth,
            vision_cx_norm=cx_norm,
            vision_cy_norm=cy_norm,
            vision_w_norm=w_norm,
            vision_h_norm=h_norm,
            vision_confidence=vision_conf,
            vision_present=vision_present,
            camera_weight=cam_weight,
            rf_weight=1.0,
        )
        X[i] = feat

        # ── Labels (fused ground truth) ───────────────────────────────────
        # Weighted fusion of GT
        if rf_present and vision_present:
            w_rf = rf_conf
            w_cam = vision_conf * cam_weight
            total_w = w_rf + w_cam + 1e-8
            fused_range = (w_rf * gt_range + w_cam * gt_range) / total_w
            fused_vel = gt_velocity  # Camera doesn't measure velocity
            fused_conf = min(1.0, (w_rf + w_cam) / 2.0)
            source_idx = 0.0  # fused
        elif rf_present:
            fused_range = gt_range
            fused_vel = gt_velocity
            fused_conf = rf_conf
            source_idx = 0.33  # rf_only
        elif vision_present:
            fused_range = gt_range
            fused_vel = 0.0
            fused_conf = vision_conf
            source_idx = 0.67  # vision_only
        else:
            fused_range = 0.0
            fused_vel = 0.0
            fused_conf = 0.0
            source_idx = 1.0  # none

        Y[i] = np.array([
            fused_range / 150.0,
            (fused_vel + 5.0) / 10.0,
            fused_conf,
            source_idx,
        ], dtype=np.float32)

    return X, Y


def train_model(
    n_samples: int = 8000,
    epochs: int = 40,
    batch_size: int = 128,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    seed: int = 42,
    save_path: str = None,
):
    """
    Train the FusionMLP model.

    Uses Adam optimizer with CosineAnnealingLR schedule and MSE loss.
    """
    try:
        import torch
        import torch.nn as nn
        from torch.utils.data import TensorDataset, DataLoader
    except ImportError:
        print("[Fusion Train] PyTorch not available. Cannot train.")
        print("[Fusion Train] Generating numpy-only placeholder model.")
        return None

    print("═" * 60)
    print("  SenseForge — Fusion Model Training")
    print("═" * 60)

    if save_path is None:
        save_path = os.path.join(PROJECT_ROOT, "models", "fusion_model.pt")

    # Generate data
    print(f"\n[1/4] Generating {n_samples} training samples...")
    X, Y = generate_training_data(n=n_samples, seed=seed)

    # Split 80/20
    split = int(0.8 * n_samples)
    X_train, X_val = X[:split], X[split:]
    Y_train, Y_val = Y[:split], Y[split:]

    # Convert to tensors
    train_ds = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(Y_train, dtype=torch.float32),
    )
    val_ds = TensorDataset(
        torch.tensor(X_val, dtype=torch.float32),
        torch.tensor(Y_val, dtype=torch.float32),
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    # Model
    print("[2/4] Building FusionMLP (14 → 64 → 32 → 4)...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FusionMLP().to(device)

    # Optimizer + scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.MSELoss()

    # Training loop
    print(f"[3/4] Training for {epochs} epochs on {device}...")
    print(f"       Train: {len(X_train)} samples | Val: {len(X_val)} samples")
    print()

    best_val_loss = float("inf")
    t_start = time.time()

    for epoch in range(1, epochs + 1):
        # ── Train ─────────────────────────────────────────────────────────
        model.train()
        train_loss = 0.0
        n_batches = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb)
            loss = criterion(pred, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            n_batches += 1
        train_loss /= max(n_batches, 1)

        # ── Validate ──────────────────────────────────────────────────────
        model.eval()
        val_loss = 0.0
        n_val = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                pred = model(xb)
                val_loss += criterion(pred, yb).item()
                n_val += 1
        val_loss /= max(n_val, 1)

        scheduler.step()

        # Print progress
        if epoch % 5 == 0 or epoch == 1:
            lr_now = scheduler.get_last_lr()[0]
            print(
                f"  Epoch {epoch:3d}/{epochs} | "
                f"Train Loss: {train_loss:.6f} | "
                f"Val Loss: {val_loss:.6f} | "
                f"LR: {lr_now:.6f}"
            )

        if val_loss < best_val_loss:
            best_val_loss = val_loss

    elapsed = time.time() - t_start

    # Save model
    print(f"\n[4/4] Saving model to {save_path}...")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(model.state_dict(), save_path)

    print(f"\n{'═' * 60}")
    print(f"  Training complete in {elapsed:.1f}s")
    print(f"  Best val loss: {best_val_loss:.6f}")
    print(f"  Model saved: {save_path}")
    print(f"{'═' * 60}")

    return model


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    train_model()
