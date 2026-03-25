# SenseForge Fusion Layer — Multimodal Sensor Fusion
"""
MLP-based late fusion of RF radar and camera detections.
"""

try:
    from .model import FusionMLP, build_feature_vector, fuse, load_model, INPUT_DIM
except ImportError as e:
    import sys
    print(f"[Fusion] Partial import: {e}", file=sys.stderr)
