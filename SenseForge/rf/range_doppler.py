"""
SenseForge — Range-Doppler Processing & CFAR Detection
═══════════════════════════════════════════════════════
Computes range-Doppler maps from OFDM channel estimates.
Implements 2D CA-CFAR detection with non-maximum suppression.
"""

from __future__ import annotations

import dataclasses
from typing import List, Optional, Tuple

import numpy as np

from .waveform_gen import SPEED_OF_LIGHT, WaveformConfig


# ═════════════════════════════════════════════════════════════════════════════
# Channel estimation
# ═════════════════════════════════════════════════════════════════════════════

def compute_channel_estimate(
    rx_grid: np.ndarray,
    tx_grid: np.ndarray,
) -> np.ndarray:
    """
    Element-wise LS channel estimate: H = RX / TX.

    Parameters
    ----------
    rx_grid : np.ndarray, shape (N_sc, N_sym), complex
    tx_grid : np.ndarray, shape (N_sc, N_sym), complex

    Returns
    -------
    H : np.ndarray, shape (N_sc, N_sym), complex64
    """
    # Avoid division by zero
    eps = 1e-12
    denom = np.where(np.abs(tx_grid) > eps, tx_grid, eps)
    H = (rx_grid / denom).astype(np.complex64)
    return H


def clutter_removal_eca(H: np.ndarray) -> np.ndarray:
    """
    Extensive Cancellation Algorithm (ECA): subtract mean across symbol axis.

    This removes static clutter (zero-Doppler components) from the
    channel estimate.

    Parameters
    ----------
    H : np.ndarray, shape (N_sc, N_sym), complex

    Returns
    -------
    H_clean : np.ndarray, same shape, complex64
    """
    mean_across_symbols = np.mean(H, axis=1, keepdims=True)
    H_clean = (H - mean_across_symbols).astype(np.complex64)
    return H_clean


# ═════════════════════════════════════════════════════════════════════════════
# Range-Doppler map computation
# ═════════════════════════════════════════════════════════════════════════════

def compute_range_doppler_map(
    H: np.ndarray,
    range_fft_size: int = 512,
    doppler_fft_size: int = 64,
    window: bool = True,
    cfg: Optional[WaveformConfig] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute 2D Range-Doppler map from channel estimate.

    1. Apply Hanning window
    2. IFFT across subcarriers (range dimension)
    3. FFT across symbols (Doppler dimension)
    4. fftshift on Doppler axis

    Parameters
    ----------
    H : np.ndarray, shape (N_sc, N_sym), complex
    range_fft_size : int
        Zero-padded FFT size for range.
    doppler_fft_size : int
        Zero-padded FFT size for Doppler.
    window : bool
        Apply Hanning window before FFT.
    cfg : WaveformConfig, optional
        If provided, compute physical-unit axes.

    Returns
    -------
    rd_map : np.ndarray, shape (range_fft_size, doppler_fft_size), float32
        Range-Doppler power map in dB.
    range_axis : np.ndarray
        Range axis in metres (or bins if no cfg).
    doppler_axis : np.ndarray
        Doppler axis in m/s (or bins if no cfg).
    """
    n_sc, n_sym = H.shape

    # Pad / truncate
    H_padded = np.zeros((range_fft_size, doppler_fft_size), dtype=np.complex64)
    r_end = min(n_sc, range_fft_size)
    d_end = min(n_sym, doppler_fft_size)
    H_padded[:r_end, :d_end] = H[:r_end, :d_end]

    # Hanning window
    if window:
        win_range = np.hanning(r_end).astype(np.float32)
        win_doppler = np.hanning(d_end).astype(np.float32)
        H_padded[:r_end, :d_end] *= win_range[:, np.newaxis] * win_doppler[np.newaxis, :]

    # IFFT along range (subcarrier) axis
    rd = np.fft.ifft(H_padded, n=range_fft_size, axis=0)

    # FFT along Doppler (symbol) axis
    rd = np.fft.fft(rd, n=doppler_fft_size, axis=1)

    # Shift Doppler axis so 0 velocity is centred
    rd = np.fft.fftshift(rd, axes=1)

    # Power in dB
    rd_power = np.abs(rd) ** 2
    rd_db = 10.0 * np.log10(rd_power + 1e-20)
    rd_map = rd_db.astype(np.float32)

    # Compute axes
    if cfg is not None:
        # Range axis: bin index → metres
        range_axis = (
            np.arange(range_fft_size)
            * SPEED_OF_LIGHT
            / (2.0 * cfg.scs * range_fft_size)
        )
        # Doppler axis: bin index → m/s
        t_sym = 1.0 / cfg.scs + cfg.cp_length / cfg.fs
        wavelength = SPEED_OF_LIGHT / cfg.carrier_freq
        doppler_freqs = (
            np.arange(doppler_fft_size) - doppler_fft_size // 2
        ) / (doppler_fft_size * t_sym)
        doppler_axis = doppler_freqs * wavelength / 2.0
    else:
        range_axis = np.arange(range_fft_size, dtype=np.float32)
        doppler_axis = np.arange(doppler_fft_size, dtype=np.float32) - doppler_fft_size // 2

    return rd_map, range_axis.astype(np.float32), doppler_axis.astype(np.float32)


# ═════════════════════════════════════════════════════════════════════════════
# CFAR Detection
# ═════════════════════════════════════════════════════════════════════════════

@dataclasses.dataclass
class RDDetection:
    """A detection in range-Doppler space."""

    range_m: float
    velocity_mps: float
    snr_db: float
    range_bin: int
    doppler_bin: int


def cfar_detector(
    rd_map: np.ndarray,
    guard: Tuple[int, int] = (2, 2),
    train: Tuple[int, int] = (8, 8),
    false_alarm: float = 1e-4,
    min_range_bin: int = 3,
) -> List[RDDetection]:
    """
    2D Cell-Averaging CFAR detector.

    Parameters
    ----------
    rd_map : np.ndarray, shape (N_range, N_doppler), float32 (dB)
    guard : (int, int)
        Guard cells (range, Doppler).
    train : (int, int)
        Training cells (range, Doppler).
    false_alarm : float
        Desired false alarm probability.
    min_range_bin : int
        Skip first N range bins (near-field).

    Returns
    -------
    detections : list of RDDetection
    """
    n_range, n_doppler = rd_map.shape
    # Convert to linear for CFAR
    rd_linear = 10.0 ** (rd_map / 10.0)

    g_r, g_d = guard
    t_r, t_d = train

    # Number of training cells
    n_train = (2 * (t_r + g_r) + 1) * (2 * (t_d + g_d) + 1) - (
        (2 * g_r + 1) * (2 * g_d + 1)
    )
    n_train = max(n_train, 1)

    # CA-CFAR threshold factor
    alpha = n_train * (false_alarm ** (-1.0 / n_train) - 1.0)

    detections: List[RDDetection] = []

    margin_r = t_r + g_r
    margin_d = t_d + g_d

    for r in range(max(margin_r, min_range_bin), n_range - margin_r):
        for d in range(margin_d, n_doppler - margin_d):
            # Training window
            window = rd_linear[
                r - margin_r : r + margin_r + 1,
                d - margin_d : d + margin_d + 1,
            ]
            # Guard window to subtract
            guard_win = rd_linear[
                r - g_r : r + g_r + 1,
                d - g_d : d + g_d + 1,
            ]
            noise_sum = np.sum(window) - np.sum(guard_win)
            noise_level = noise_sum / n_train

            threshold = alpha * noise_level
            cell_power = rd_linear[r, d]

            if cell_power > threshold and noise_level > 0:
                snr_db = 10.0 * np.log10(cell_power / noise_level)
                det = RDDetection(
                    range_m=float(r),  # Placeholder — remap to physical later
                    velocity_mps=float(d - n_doppler // 2),
                    snr_db=float(snr_db),
                    range_bin=int(r),
                    doppler_bin=int(d),
                )
                detections.append(det)

    # NMS
    detections = _nms(detections)
    return detections


def _nms(
    detections: List[RDDetection],
    r_tol: int = 2,
    d_tol: int = 2,
) -> List[RDDetection]:
    """Non-maximum suppression on range-Doppler detections."""
    if not detections:
        return []

    # Sort by SNR descending
    detections = sorted(detections, key=lambda d: d.snr_db, reverse=True)
    kept: List[RDDetection] = []

    for det in detections:
        suppress = False
        for k in kept:
            if (
                abs(det.range_bin - k.range_bin) <= r_tol
                and abs(det.doppler_bin - k.doppler_bin) <= d_tol
            ):
                suppress = True
                break
        if not suppress:
            kept.append(det)

    return kept


# ═════════════════════════════════════════════════════════════════════════════
# Full processing chain
# ═════════════════════════════════════════════════════════════════════════════

def process_slot(
    rx_grid: np.ndarray,
    tx_grid: np.ndarray,
    cfg: Optional[WaveformConfig] = None,
    range_fft_size: int = 512,
    doppler_fft_size: int = 64,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[RDDetection]]:
    """
    Full range-Doppler processing chain for one slot.

    1. Channel estimate
    2. Clutter removal (ECA)
    3. Range-Doppler map
    4. CFAR detection

    Returns
    -------
    rd_map, range_axis, doppler_axis, detections
    """
    # Channel estimate
    H = compute_channel_estimate(rx_grid, tx_grid)

    # Clutter removal
    H = clutter_removal_eca(H)

    # Range-Doppler map
    rd_map, range_axis, doppler_axis = compute_range_doppler_map(
        H,
        range_fft_size=range_fft_size,
        doppler_fft_size=doppler_fft_size,
        window=True,
        cfg=cfg,
    )

    # CFAR detection
    detections = cfar_detector(rd_map)

    # Remap detection coordinates to physical units
    for det in detections:
        if det.range_bin < len(range_axis):
            det.range_m = float(range_axis[det.range_bin])
        if det.doppler_bin < len(doppler_axis):
            det.velocity_mps = float(doppler_axis[det.doppler_bin])

    return rd_map, range_axis, doppler_axis, detections
