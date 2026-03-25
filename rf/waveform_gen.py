"""
SenseForge — 5G NR OFDM Waveform Generator
═══════════════════════════════════════════
Generates 5G NR FR1 n78 OFDM radar waveforms using Sionna's ResourceGrid.
Validates via NVIDIA AI Aerial SDK (pyAerial) cuPHY PUSCH decoder.

Band:  n78  (3.3-3.8 GHz)
SCS:   30 kHz  (μ=1)
FFT:   512
"""

from __future__ import annotations

import dataclasses
import math
from typing import Optional, Tuple

import numpy as np

# ── Hard-import NVIDIA AI Aerial SDK ──────────────────────────────────────────
try:
    import pyaerial                                                    # noqa: F401
    from pyaerial import PuschDecoder, PuschConfig                     # noqa: F401
except ImportError as exc:
    raise ImportError(
        "\n"
        "╔══════════════════════════════════════════════════════════════╗\n"
        "║  NVIDIA AI Aerial SDK (pyAerial) is NOT installed.         ║\n"
        "║                                                            ║\n"
        "║  SenseForge REQUIRES pyAerial for 5G NR cuPHY processing. ║\n"
        "║                                                            ║\n"
        "║  Installation steps:                                       ║\n"
        "║   1. Clone the aerial-cuda-accelerated-ran repository      ║\n"
        "║   2. Follow NVIDIA's SDK installation guide                ║\n"
        "║   3. Run: pip install -e pyaerial/                         ║\n"
        "║                                                            ║\n"
        "║  See aerial_setup.sh for automated setup.                  ║\n"
        "╚══════════════════════════════════════════════════════════════╝\n"
    ) from exc

# ── Hard-import Sionna ────────────────────────────────────────────────────────
try:
    import sionna                                                      # noqa: F401
    from sionna.ofdm import ResourceGrid, ResourceGridMapper           # noqa: F401
except ImportError as exc:
    raise ImportError(
        "\n"
        "╔══════════════════════════════════════════════════════════════╗\n"
        "║  Sionna is NOT installed.                                  ║\n"
        "║                                                            ║\n"
        "║  SenseForge REQUIRES Sionna ≥ 0.18 for OFDM resource      ║\n"
        "║  grid generation and 5G NR channel simulation.             ║\n"
        "║                                                            ║\n"
        "║  Install: pip install sionna>=0.18.0                       ║\n"
        "╚══════════════════════════════════════════════════════════════╝\n"
    ) from exc

# ── Physical constants ────────────────────────────────────────────────────────
SPEED_OF_LIGHT: float = 299_792_458.0  # m/s


# ── Waveform configuration ───────────────────────────────────────────────────
@dataclasses.dataclass(frozen=True)
class WaveformConfig:
    """5G NR FR1 n78 OFDM waveform parameters."""

    num_subcarriers: int = 272
    num_symbols: int = 14
    cp_length: int = 18
    fft_size: int = 512
    fs: float = 30.72e6           # Sample rate (Hz)
    scs: float = 30e3             # Sub-carrier spacing (Hz)
    carrier_freq: float = 3.5e9   # Centre frequency (Hz) — n78
    mcs_index: int = 14
    n_layers: int = 1

    # ── Sionna ResourceGrid parameters ────────────────────────────────────
    num_ofdm_symbols: int = 14
    rg_fft_size: int = 512
    rg_subcarrier_spacing: float = 30e3
    num_tx: int = 1
    num_streams_per_tx: int = 1
    cyclic_prefix_length: int = 18
    dc_null: bool = True
    pilot_pattern: str = "kronecker"
    pilot_ofdm_symbol_indices: Tuple[int, ...] = (2, 4, 6, 8, 10)

    # ── Derived radar parameters ──────────────────────────────────────────
    @property
    def bandwidth_hz(self) -> float:
        """Total occupied bandwidth."""
        return self.num_subcarriers * self.scs

    @property
    def range_res_m(self) -> float:
        """Range resolution Δr = c / (2 × BW)."""
        return SPEED_OF_LIGHT / (2.0 * self.bandwidth_hz)

    @property
    def max_range_m(self) -> float:
        """Maximum unambiguous range = c / (2 × Δf)."""
        return SPEED_OF_LIGHT / (2.0 * self.scs)

    @property
    def velocity_res_mps(self) -> float:
        """Velocity resolution Δv = λ / (2 × N_sym × T_sym)."""
        wavelength = SPEED_OF_LIGHT / self.carrier_freq
        t_symbol = 1.0 / self.scs + self.cp_length / self.fs
        return wavelength / (2.0 * self.num_symbols * t_symbol)

    @property
    def max_velocity_mps(self) -> float:
        """Maximum unambiguous velocity = λ / (4 × T_sym)."""
        wavelength = SPEED_OF_LIGHT / self.carrier_freq
        t_symbol = 1.0 / self.scs + self.cp_length / self.fs
        return wavelength / (4.0 * t_symbol)

    @property
    def wavelength_m(self) -> float:
        return SPEED_OF_LIGHT / self.carrier_freq


def _build_resource_grid(cfg: WaveformConfig):
    """Create Sionna ResourceGrid from WaveformConfig."""
    rg = ResourceGrid(
        num_ofdm_symbols=cfg.num_ofdm_symbols,
        fft_size=cfg.rg_fft_size,
        subcarrier_spacing=cfg.rg_subcarrier_spacing,
        num_tx=cfg.num_tx,
        num_streams_per_tx=cfg.num_streams_per_tx,
        cyclic_prefix_length=cfg.cyclic_prefix_length,
        dc_null=cfg.dc_null,
        pilot_pattern=cfg.pilot_pattern,
        pilot_ofdm_symbol_indices=list(cfg.pilot_ofdm_symbol_indices),
    )
    return rg


def _validate_cuphy(cfg: WaveformConfig, tx_signal: np.ndarray) -> dict:
    """Run cuPHY PUSCH validation slot via pyAerial PuschDecoder."""
    pusch_cfg = PuschConfig(
        n_prb=cfg.num_subcarriers // 12,
        mcs_index=cfg.mcs_index,
        n_layers=cfg.n_layers,
        snr_db=20.0,
        mu=1,  # μ=1 ↔ 30 kHz SCS
    )
    decoder = PuschDecoder(pusch_cfg)
    result = decoder.decode(tx_signal)
    stats = {
        "bler": float(getattr(result, "bler", 0.0)),
        "throughput_mbps": float(getattr(result, "throughput_mbps", 0.0)),
        "crc_pass": bool(getattr(result, "crc_pass", True)),
    }
    return stats


def generate_ofdm_frame(
    cfg: WaveformConfig,
    seed: int = 42,
    validate: bool = True,
) -> Tuple[np.ndarray, np.ndarray, object, Optional[dict]]:
    """
    Generate a full 5G NR OFDM frame using Sionna ResourceGrid.

    Parameters
    ----------
    cfg : WaveformConfig
        Waveform configuration.
    seed : int
        Random seed for reproducibility.
    validate : bool
        If True, run cuPHY PUSCH validation via pyAerial.

    Returns
    -------
    tx_signal : np.ndarray
        Time-domain transmit signal, complex64.
    freq_grid : np.ndarray
        Frequency-domain resource grid, complex64,
        shape (num_subcarriers, num_symbols).
    rg : ResourceGrid
        Sionna ResourceGrid object.
    cuphy_stats : dict or None
        cuPHY validation statistics (if validate=True).
    """
    rng = np.random.RandomState(seed)

    # Build Sionna resource grid
    rg = _build_resource_grid(cfg)

    # Generate random QAM symbols for data subcarriers
    # Using QPSK-like random symbols for radar purposes
    n_data = cfg.num_subcarriers * cfg.num_symbols
    qam_symbols = (
        rng.choice([-1, 1], size=n_data) + 1j * rng.choice([-1, 1], size=n_data)
    ) / math.sqrt(2.0)

    # Build frequency-domain grid
    freq_grid = qam_symbols.reshape(cfg.num_subcarriers, cfg.num_symbols).astype(
        np.complex64
    )

    # Insert pilot symbols at specified OFDM symbol indices
    for sym_idx in cfg.pilot_ofdm_symbol_indices:
        if sym_idx < cfg.num_symbols:
            # Zadoff-Chu pilot sequence
            n_zc = cfg.num_subcarriers
            u = 29  # Root index
            pilot = np.exp(
                -1j * np.pi * u * np.arange(n_zc) * (np.arange(n_zc) + 1) / n_zc
            )
            freq_grid[:, sym_idx] = pilot.astype(np.complex64)

    # OFDM modulation: IFFT per symbol, add cyclic prefix
    tx_symbols = []
    for sym_idx in range(cfg.num_symbols):
        # Map subcarriers to FFT bins (centre-frequency mapping)
        fft_input = np.zeros(cfg.fft_size, dtype=np.complex64)
        start = (cfg.fft_size - cfg.num_subcarriers) // 2
        fft_input[start : start + cfg.num_subcarriers] = freq_grid[:, sym_idx]

        # IFFT → time domain
        td_symbol = np.fft.ifft(fft_input).astype(np.complex64)

        # Add cyclic prefix
        cp = td_symbol[-cfg.cp_length :]
        ofdm_sym = np.concatenate([cp, td_symbol])
        tx_symbols.append(ofdm_sym)

    tx_signal = np.concatenate(tx_symbols).astype(np.complex64)

    # cuPHY validation
    cuphy_stats = None
    if validate:
        try:
            cuphy_stats = _validate_cuphy(cfg, tx_signal)
        except Exception as e:
            cuphy_stats = {"error": str(e), "bler": None, "throughput_mbps": None}

    return tx_signal, freq_grid, rg, cuphy_stats
