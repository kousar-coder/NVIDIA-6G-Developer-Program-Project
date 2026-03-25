# SenseForge RF Pipeline — 5G NR OFDM Radar Sensing
"""
RF sensing pipeline for ISAC (Integrated Sensing and Communications).
Requires: NVIDIA AI Aerial SDK (pyAerial) + Sionna
"""

try:
    from .waveform_gen import WaveformConfig, generate_ofdm_frame
    from .echo_simulator import Target, ScenarioConfig, simulate_echoes, make_test_scenario
    from .range_doppler import compute_range_doppler_map, cfar_detector, process_slot, RDDetection
    from .rf_tracker import KalmanRFTracker, RFTrack
except ImportError:
    pass  # SDK not installed — modules imported directly where needed
