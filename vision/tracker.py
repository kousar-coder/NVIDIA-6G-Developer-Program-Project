"""
SenseForge — Vision Tracker (ByteTrack-style)
═════════════════════════════════════════════
Multi-object tracker using Hungarian matching (scipy).
Tracks persons across frames with IoU-based assignment.
"""

from __future__ import annotations

import dataclasses
from typing import List, Optional, Tuple

import numpy as np

try:
    from scipy.optimize import linear_sum_assignment
except ImportError:
    linear_sum_assignment = None  # Fallback below

from .detector import Detection


@dataclasses.dataclass
class VisionTrack:
    """A tracked person in camera space."""

    track_id: int
    bbox: Tuple[float, float, float, float]  # (x1, y1, x2, y2)
    confidence: float
    age: int = 0
    missed: int = 0
    velocity_px: Tuple[float, float] = (0.0, 0.0)
    confirmed: bool = False
    depth_m: Optional[float] = None

    @property
    def cx(self) -> float:
        return (self.bbox[0] + self.bbox[2]) / 2.0

    @property
    def cy(self) -> float:
        return (self.bbox[1] + self.bbox[3]) / 2.0

    @property
    def width(self) -> float:
        return self.bbox[2] - self.bbox[0]

    @property
    def height(self) -> float:
        return self.bbox[3] - self.bbox[1]


def _iou(a: Tuple[float, ...], b: Tuple[float, ...]) -> float:
    """Compute Intersection over Union of two bounding boxes."""
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])

    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if inter <= 0:
        return 0.0

    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    union = area_a + area_b - inter

    return inter / max(union, 1e-6)


class VisionTracker:
    """
    Multi-object tracker using Hungarian assignment with IoU cost.

    Similar to ByteTrack but simplified for the ISAC demo.
    """

    def __init__(
        self,
        iou_threshold: float = 0.3,
        max_missed: int = 10,
        confirm_hits: int = 2,
    ):
        self.iou_threshold = iou_threshold
        self.max_missed = max_missed
        self.confirm_hits = confirm_hits

        self._next_id = 1
        self._tracks: List[_TrackState] = []

    @property
    def tracks(self) -> List[VisionTrack]:
        return [t.to_vision_track() for t in self._tracks]

    @property
    def confirmed_tracks(self) -> List[VisionTrack]:
        return [t.to_vision_track() for t in self._tracks if t.confirmed]

    def update(self, detections: List[Detection]) -> List[VisionTrack]:
        """
        Update tracker with new detections.

        Parameters
        ----------
        detections : list of Detection

        Returns
        -------
        confirmed_tracks : list of VisionTrack
        """
        # ── Predict positions ─────────────────────────────────────────────
        for t in self._tracks:
            t.predict()

        if not detections and not self._tracks:
            return []

        # ── Build cost matrix (1 - IoU) ──────────────────────────────────
        n_tracks = len(self._tracks)
        n_dets = len(detections)

        if n_tracks > 0 and n_dets > 0:
            cost = np.zeros((n_tracks, n_dets), dtype=np.float64)
            for i, t in enumerate(self._tracks):
                for j, d in enumerate(detections):
                    cost[i, j] = 1.0 - _iou(t.bbox, d.bbox)

            # ── Hungarian assignment ──────────────────────────────────────
            if linear_sum_assignment is not None:
                row_ind, col_ind = linear_sum_assignment(cost)
            else:
                # Greedy fallback
                row_ind, col_ind = _greedy_assignment(cost)

            matched_tracks = set()
            matched_dets = set()

            for r, c in zip(row_ind, col_ind):
                if cost[r, c] <= (1.0 - self.iou_threshold):
                    self._tracks[r].correct(detections[c])
                    matched_tracks.add(r)
                    matched_dets.add(c)

            # ── Unmatched tracks → increment missed ──────────────────────
            for i in range(n_tracks):
                if i not in matched_tracks:
                    self._tracks[i].missed += 1

            # ── Unmatched detections → new tracks ────────────────────────
            for j in range(n_dets):
                if j not in matched_dets:
                    self._birth(detections[j])

        elif n_dets > 0:
            for d in detections:
                self._birth(d)
        else:
            for t in self._tracks:
                t.missed += 1

        # ── Track management ─────────────────────────────────────────────
        alive = []
        for t in self._tracks:
            t.age += 1
            if t.hits >= self.confirm_hits:
                t.confirmed = True
            if t.missed > self.max_missed:
                continue  # Delete
            alive.append(t)
        self._tracks = alive

        return self.confirmed_tracks

    def _birth(self, det: Detection):
        t = _TrackState(
            track_id=self._next_id,
            bbox=det.bbox,
            confidence=det.confidence,
            depth_m=det.depth_m,
        )
        self._next_id += 1
        self._tracks.append(t)

    def reset(self):
        self._tracks.clear()
        self._next_id = 1


def _greedy_assignment(cost: np.ndarray):
    """Greedy fallback if scipy is not available."""
    n, m = cost.shape
    rows, cols = [], []
    used_rows, used_cols = set(), set()

    flat_idx = np.argsort(cost.ravel())
    for idx in flat_idx:
        r, c = divmod(int(idx), m)
        if r not in used_rows and c not in used_cols:
            rows.append(r)
            cols.append(c)
            used_rows.add(r)
            used_cols.add(c)
            if len(rows) == min(n, m):
                break

    return np.array(rows), np.array(cols)


class _TrackState:
    """Internal track state."""

    def __init__(
        self,
        track_id: int,
        bbox: Tuple[float, float, float, float],
        confidence: float = 0.0,
        depth_m: Optional[float] = None,
    ):
        self.track_id = track_id
        self.bbox = bbox
        self.confidence = confidence
        self.depth_m = depth_m
        self.age = 0
        self.hits = 1
        self.missed = 0
        self.confirmed = False
        self.prev_cx = (bbox[0] + bbox[2]) / 2.0
        self.prev_cy = (bbox[1] + bbox[3]) / 2.0
        self.vx = 0.0
        self.vy = 0.0

    def predict(self):
        """Simple constant-velocity prediction."""
        cx = (self.bbox[0] + self.bbox[2]) / 2.0 + self.vx
        cy = (self.bbox[1] + self.bbox[3]) / 2.0 + self.vy
        w = self.bbox[2] - self.bbox[0]
        h = self.bbox[3] - self.bbox[1]
        self.bbox = (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)

    def correct(self, det: Detection):
        cx_new = (det.bbox[0] + det.bbox[2]) / 2.0
        cy_new = (det.bbox[1] + det.bbox[3]) / 2.0
        cx_old = (self.bbox[0] + self.bbox[2]) / 2.0
        cy_old = (self.bbox[1] + self.bbox[3]) / 2.0
        self.vx = 0.7 * (cx_new - cx_old) + 0.3 * self.vx
        self.vy = 0.7 * (cy_new - cy_old) + 0.3 * self.vy
        self.bbox = det.bbox
        self.confidence = det.confidence
        self.depth_m = det.depth_m
        self.hits += 1
        self.missed = 0

    def to_vision_track(self) -> VisionTrack:
        return VisionTrack(
            track_id=self.track_id,
            bbox=self.bbox,
            confidence=self.confidence,
            age=self.age,
            missed=self.missed,
            velocity_px=(self.vx, self.vy),
            confirmed=self.confirmed,
            depth_m=self.depth_m,
        )
