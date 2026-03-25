# SenseForge Vision Pipeline — Camera-based Detection and Tracking
"""
Computer vision pipeline: YOLOv8 detection, ByteTrack-style tracking,
MiDaS monocular depth estimation, and weather degradation simulation.
"""

try:
    from .detector import YOLODetector, Detection
    from .tracker import VisionTracker, VisionTrack
    from .depth import estimate_depth_map, bbox_to_depth_m
    from .degradation import DegradationMode, apply_degradation, get_camera_confidence
except ImportError as e:
    import sys
    print(f"[Vision] Partial import: {e}", file=sys.stderr)
