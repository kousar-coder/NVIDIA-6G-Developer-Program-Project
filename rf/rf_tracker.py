"""
SenseForge — Kalman-based RF Tracker
═════════════════════════════════════
Linear Kalman filter tracker for range-Doppler detections.
Uses greedy nearest-neighbour association with gating.
"""

from __future__ import annotations

import dataclasses
from typing import List, Optional

import numpy as np

from .range_doppler import RDDetection


@dataclasses.dataclass
class RFTrack:
    """A confirmed or tentative RF track."""

    track_id: int
    range_m: float
    velocity_mps: float
    snr_db: float
    age: int = 0
    missed: int = 0
    confirmed: bool = False


class KalmanRFTracker:
    """
    Kalman filter tracker for RF detections in range-Doppler space.

    State vector: [range, velocity]
    Measurement:  [range, velocity]

    Uses greedy nearest-neighbour data association with rectangular gating.
    """

    def __init__(
        self,
        confirm_hits: int = 2,
        max_missed: int = 5,
        gate_range: float = 15.0,    # metres
        gate_vel: float = 2.0,       # m/s
        dt: float = 0.1,             # 10 Hz update
    ):
        self.confirm_hits = confirm_hits
        self.max_missed = max_missed
        self.gate_range = gate_range
        self.gate_vel = gate_vel
        self.dt = dt

        self._next_id = 1
        self._tracks: List[_KalmanTrackState] = []

    @property
    def tracks(self) -> List[RFTrack]:
        """Return list of current tracks (confirmed + tentative)."""
        return [t.to_rf_track() for t in self._tracks]

    @property
    def confirmed_tracks(self) -> List[RFTrack]:
        """Return only confirmed tracks."""
        return [t.to_rf_track() for t in self._tracks if t.confirmed]

    def update(self, detections: List[RDDetection]) -> List[RFTrack]:
        """
        Run one tracker cycle: predict → associate → update → manage.

        Parameters
        ----------
        detections : list of RDDetection
            Current frame detections.

        Returns
        -------
        confirmed_tracks : list of RFTrack
        """
        # ── Predict ───────────────────────────────────────────────────────
        for t in self._tracks:
            t.predict(self.dt)

        # ── Associate (greedy nearest-neighbour) ──────────────────────────
        used_dets = set()
        for t in self._tracks:
            best_idx = None
            best_dist = float("inf")
            for i, det in enumerate(detections):
                if i in used_dets:
                    continue
                dr = abs(det.range_m - t.x[0])
                dv = abs(det.velocity_mps - t.x[1])
                if dr <= self.gate_range and dv <= self.gate_vel:
                    dist = (dr / self.gate_range) ** 2 + (dv / self.gate_vel) ** 2
                    if dist < best_dist:
                        best_dist = dist
                        best_idx = i
            if best_idx is not None:
                t.correct(detections[best_idx])
                used_dets.add(best_idx)
            else:
                t.missed += 1

        # ── Birth: unassociated detections → new tracks ──────────────────
        for i, det in enumerate(detections):
            if i not in used_dets:
                t = _KalmanTrackState(
                    track_id=self._next_id,
                    x=np.array([det.range_m, det.velocity_mps], dtype=np.float64),
                    snr_db=det.snr_db,
                )
                self._next_id += 1
                self._tracks.append(t)

        # ── Track management ─────────────────────────────────────────────
        alive = []
        for t in self._tracks:
            t.age += 1
            if t.hits >= self.confirm_hits:
                t.confirmed = True
            if t.missed > self.max_missed:
                continue  # Delete track
            alive.append(t)
        self._tracks = alive

        return self.confirmed_tracks

    def reset(self):
        """Clear all tracks."""
        self._tracks.clear()
        self._next_id = 1


class _KalmanTrackState:
    """Internal Kalman filter state for a single track."""

    def __init__(self, track_id: int, x: np.ndarray, snr_db: float = 0.0):
        self.track_id = track_id
        self.x = x.copy()  # [range, velocity]
        self.snr_db = snr_db
        self.age = 0
        self.hits = 1
        self.missed = 0
        self.confirmed = False

        # ── Kalman matrices ──────────────────────────────────────────
        # State transition: range += velocity * dt
        self.F = np.eye(2, dtype=np.float64)
        # (F[0,1] is set in predict)

        # Observation matrix (direct measurement)
        self.H = np.eye(2, dtype=np.float64)

        # Process noise
        self.Q = np.diag([1.0, 0.5]).astype(np.float64)

        # Measurement noise
        self.R = np.diag([5.0, 0.3]).astype(np.float64)

        # State covariance
        self.P = np.diag([25.0, 1.0]).astype(np.float64)

    def predict(self, dt: float):
        """Kalman predict step."""
        self.F[0, 1] = dt
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q

    def correct(self, det: RDDetection):
        """Kalman correct step."""
        z = np.array([det.range_m, det.velocity_mps], dtype=np.float64)
        y = z - self.H @ self.x               # Innovation
        S = self.H @ self.P @ self.H.T + self.R  # Innovation covariance
        K = self.P @ self.H.T @ np.linalg.inv(S)  # Kalman gain

        self.x = self.x + K @ y
        self.P = (np.eye(2) - K @ self.H) @ self.P

        self.snr_db = det.snr_db
        self.hits += 1
        self.missed = 0

    def to_rf_track(self) -> RFTrack:
        return RFTrack(
            track_id=self.track_id,
            range_m=float(self.x[0]),
            velocity_mps=float(self.x[1]),
            snr_db=float(self.snr_db),
            age=self.age,
            missed=self.missed,
            confirmed=self.confirmed,
        )
