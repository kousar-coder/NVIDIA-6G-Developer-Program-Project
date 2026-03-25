"""
SenseForge — YOLOv8 Person Detector
════════════════════════════════════
Wraps ultralytics YOLOv8 for person detection.
Lazy-loads model on first inference.
"""

from __future__ import annotations

import dataclasses
from typing import List, Optional, Tuple

import numpy as np


@dataclasses.dataclass
class Detection:
    """A single object detection."""

    bbox: Tuple[float, float, float, float]   # (x1, y1, x2, y2)
    confidence: float
    label: str = "person"
    depth_m: Optional[float] = None
    track_id: Optional[int] = None

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


class YOLODetector:
    """
    YOLOv8 person detector.

    Lazy-loads the model on first call to detect().
    Filters for class_id == 0 (person) and conf > threshold.
    Falls back to synthetic detections if model unavailable.
    """

    def __init__(
        self,
        model_name: str = "yolov8n.pt",
        conf_threshold: float = 0.4,
        device: Optional[str] = None,
    ):
        self.model_name = model_name
        self.conf_threshold = conf_threshold
        self.device = device
        self._model = None
        self._model_loaded = False
        self._model_failed = False

    def _load_model(self):
        """Lazy-load YOLOv8 model."""
        if self._model_loaded or self._model_failed:
            return
        try:
            from ultralytics import YOLO
            self._model = YOLO(self.model_name)
            if self.device:
                self._model.to(self.device)
            self._model_loaded = True
        except Exception as e:
            print(f"[YOLODetector] Failed to load model: {e}")
            self._model_failed = True

    def detect(self, frame: np.ndarray) -> List[Detection]:
        """
        Run person detection on a BGR frame.

        Parameters
        ----------
        frame : np.ndarray
            BGR image, shape (H, W, 3), uint8.

        Returns
        -------
        detections : list of Detection
        """
        self._load_model()

        if self._model is None:
            return self._synthetic_detections(frame)

        try:
            results = self._model(frame, verbose=False, conf=self.conf_threshold)
            detections = []
            for r in results:
                boxes = r.boxes
                if boxes is None:
                    continue
                for i in range(len(boxes)):
                    cls_id = int(boxes.cls[i].item())
                    if cls_id != 0:  # Only persons
                        continue
                    conf = float(boxes.conf[i].item())
                    x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy().tolist()
                    det = Detection(
                        bbox=(x1, y1, x2, y2),
                        confidence=conf,
                        label="person",
                    )
                    detections.append(det)
            return detections
        except Exception:
            return self._synthetic_detections(frame)

    def _synthetic_detections(self, frame: np.ndarray) -> List[Detection]:
        """
        Generate synthetic person detections for testing.

        Creates plausible bounding boxes based on frame dimensions.
        """
        h, w = frame.shape[:2]
        detections = []

        # Analyse frame to find bright blobs (person-like regions)
        if len(frame.shape) == 3:
            gray = np.mean(frame, axis=2)
        else:
            gray = frame.astype(np.float32)

        # Simple blob detection: find bright columns
        col_mean = np.mean(gray, axis=0)
        threshold = np.mean(col_mean) + 0.5 * np.std(col_mean)

        # Find connected regions above threshold
        above = col_mean > threshold
        regions = []
        start = None
        for i, v in enumerate(above):
            if v and start is None:
                start = i
            elif not v and start is not None:
                if i - start > w * 0.03:  # Minimum width
                    regions.append((start, i))
                start = None
        if start is not None and len(col_mean) - start > w * 0.03:
            regions.append((start, len(col_mean)))

        for idx, (rx1, rx2) in enumerate(regions[:4]):
            # Estimate person bounding box
            cx = (rx1 + rx2) / 2.0
            bw = max(rx2 - rx1, w * 0.08)
            bh = bw * 2.5  # Person aspect ratio
            x1 = max(0, cx - bw / 2)
            y1 = max(0, h * 0.3 - bh * 0.1)
            x2 = min(w, cx + bw / 2)
            y2 = min(h, y1 + bh)

            det = Detection(
                bbox=(float(x1), float(y1), float(x2), float(y2)),
                confidence=0.7 - idx * 0.05,
                label="person",
            )
            detections.append(det)

        return detections
