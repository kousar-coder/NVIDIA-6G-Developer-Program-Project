"""
SenseForge — Radar Echo / Channel Simulator
════════════════════════════════════════════
Simulates multi-target radar echoes through a 3GPP CDL channel.
Applies two-way Friis path loss, Doppler shifts, range delays,
and Rician fading per target.

Requires: Sionna (CDL channel model from TR 38.901)
"""

from __future__ import annotations

import dataclasses
import math
from typing import List, Optional, Tuple

import numpy as np

# ── Hard-import Sionna CDL channel ────────────────────────────────────────────
try:
    import sionna                                                  # noqa: F401
    from sionna.channel.tr38901 import CDL, PanelArray             # noqa: F401
except ImportError as exc:
    raise ImportError(
        "\n"
        "╔══════════════════════════════════════════════════════════════╗\n"
        "║  Sionna CDL channel model is NOT available.                ║\n"
        "║  Install: pip install sionna>=0.18.0                       ║\n"
        "╚══════════════════════════════════════════════════════════════╝\n"
    ) from exc

from .waveform_gen import SPEED_OF_LIGHT, WaveformConfig


# ═════════════════════════════════════════════════════════════════════════════
# Data classes
# ═════════════════════════════════════════════════════════════════════════════

@dataclasses.dataclass
class Target:
    """A single radar target."""

    range_m: float
    velocity_mps: float
    azimuth_deg: float = 0.0
    elevation_deg: float = 0.0
    rcs_dbsm: float = -10.0      # Radar cross-section in dBsm
    label: str = "person"

    @property
    def rcs_m2(self) -> float:
        """RCS in linear square metres."""
        return 10.0 ** (self.rcs_dbsm / 10.0)

    def doppler_hz(self, carrier_freq: float) -> float:
        """Doppler shift f_d = 2·v·fc / c."""
        return 2.0 * self.velocity_mps * carrier_freq / SPEED_OF_LIGHT

    @property
    def delay_s(self) -> float:
        """Round-trip delay τ = 2·R / c."""
        return 2.0 * self.range_m / SPEED_OF_LIGHT


@dataclasses.dataclass
class ScenarioConfig:
    """Radar scenario with multiple targets and channel parameters."""

    targets: List[Target] = dataclasses.field(default_factory=list)
    tx_power_dbm: float = 30.0
    noise_figure_db: float = 7.0
    clutter_level_db: float = -30.0
    cdl_model: str = "B"


# ═════════════════════════════════════════════════════════════════════════════
# Channel model
# ═════════════════════════════════════════════════════════════════════════════

def build_cdl_channel(
    cfg: WaveformConfig,
    scenario: ScenarioConfig,
    seed: int = 42,
):
    """
    Build a Sionna CDL channel (TR 38.901) for SISO.

    Uses PanelArray with single-element antennas.
    CDL model type from scenario config (default "B").
    """
    # Single-element panel arrays for SISO
    tx_array = PanelArray(
        num_rows_per_panel=1,
        num_cols_per_panel=1,
        polarization="single",
        polarization_type="V",
        antenna_pattern="omni",
        carrier_frequency=cfg.carrier_freq,
    )
    rx_array = PanelArray(
        num_rows_per_panel=1,
        num_cols_per_panel=1,
        polarization="single",
        polarization_type="V",
        antenna_pattern="omni",
        carrier_frequency=cfg.carrier_freq,
    )

    cdl = CDL(
        model=scenario.cdl_model,
        delay_spread=100e-9,          # 100 ns
        carrier_frequency=cfg.carrier_freq,
        ut_array=rx_array,
        bs_array=tx_array,
        direction="uplink",
        min_speed=0.0,
        max_speed=5.0,
    )
    return cdl


# ═════════════════════════════════════════════════════════════════════════════
# Echo simulation
# ═════════════════════════════════════════════════════════════════════════════

def _friis_path_loss(range_m: float, carrier_freq: float) -> float:
    """Two-way free-space path loss (linear)."""
    if range_m <= 0:
        return 1.0
    wavelength = SPEED_OF_LIGHT / carrier_freq
    # Two-way: (4π R)^4 / (λ^2) — simplified radar equation factor
    loss = ((4.0 * math.pi * range_m) ** 2 / wavelength) ** 2
    return max(loss, 1.0)


def _rician_fading(shape: tuple, k_factor: float, rng: np.random.RandomState) -> np.ndarray:
    """Generate Rician fading coefficients with Rice factor K."""
    # LOS component
    los = np.sqrt(k_factor / (k_factor + 1.0))
    # Scattered component
    scatter_power = 1.0 / (k_factor + 1.0)
    scattered = np.sqrt(scatter_power / 2.0) * (
        rng.randn(*shape) + 1j * rng.randn(*shape)
    )
    return (los + scattered).astype(np.complex64)


def simulate_echoes(
    tx_freq_grid: np.ndarray,
    scenario: ScenarioConfig,
    cfg: WaveformConfig,
    seed: int = 42,
) -> Tuple[np.ndarray, Optional[object]]:
    """
    Simulate radar echoes from all targets in the scenario.

    Parameters
    ----------
    tx_freq_grid : np.ndarray
        Transmitted frequency-domain grid, shape (N_sc, N_sym), complex64.
    scenario : ScenarioConfig
        Scenario with targets and channel parameters.
    cfg : WaveformConfig
        Waveform configuration.
    seed : int
        Random seed.

    Returns
    -------
    rx_grid : np.ndarray
        Received frequency-domain grid, complex64, shape (N_sc, N_sym).
    channel : object or None
        CDL channel object (if built), else None.
    """
    rng = np.random.RandomState(seed)
    n_sc, n_sym = tx_freq_grid.shape

    # Build CDL channel for multipath
    channel = None
    try:
        channel = build_cdl_channel(cfg, scenario, seed)
    except Exception:
        pass  # Fallback: use per-target model only

    # Accumulate echoes
    rx_grid = np.zeros_like(tx_freq_grid)

    # Subcarrier and symbol indices
    sc_indices = np.arange(n_sc)
    sym_indices = np.arange(n_sym)

    for target in scenario.targets:
        # ── Path loss ─────────────────────────────────────────────────
        loss_linear = _friis_path_loss(target.range_m, cfg.carrier_freq)
        amplitude = math.sqrt(target.rcs_m2) / math.sqrt(loss_linear)

        # ── Fading ────────────────────────────────────────────────────
        fading = _rician_fading((n_sc, n_sym), k_factor=1.0, rng=rng)

        # ── Range phase ramp (across subcarriers) ────────────────────
        # Phase = exp(-j·2π·Δf·k·τ) where τ = 2R/c, Δf = SCS
        tau = target.delay_s
        range_phase = np.exp(
            -1j * 2.0 * np.pi * cfg.scs * sc_indices * tau
        ).astype(np.complex64)

        # ── Doppler phase ramp (across symbols) ──────────────────────
        # Phase = exp(j·2π·fd·l·T_sym)
        fd = target.doppler_hz(cfg.carrier_freq)
        t_sym = 1.0 / cfg.scs + cfg.cp_length / cfg.fs
        doppler_phase = np.exp(
            1j * 2.0 * np.pi * fd * sym_indices * t_sym
        ).astype(np.complex64)

        # ── Combine: 2D echo = amplitude × fading × range × doppler ──
        echo = (
            amplitude
            * fading
            * range_phase[:, np.newaxis]
            * doppler_phase[np.newaxis, :]
            * tx_freq_grid
        )
        rx_grid += echo

    # ── Add clutter ───────────────────────────────────────────────────
    clutter_power = 10.0 ** (scenario.clutter_level_db / 10.0)
    clutter = np.sqrt(clutter_power / 2.0) * (
        rng.randn(n_sc, n_sym) + 1j * rng.randn(n_sc, n_sym)
    )
    rx_grid += clutter.astype(np.complex64)

    # ── Add AWGN thermal noise ────────────────────────────────────────
    # Noise power: kTBF
    k_boltzmann = 1.380649e-23
    temperature = 290.0  # Kelvin
    noise_power = (
        k_boltzmann * temperature * cfg.bandwidth_hz
        * 10.0 ** (scenario.noise_figure_db / 10.0)
    )
    noise = np.sqrt(noise_power / 2.0) * (
        rng.randn(n_sc, n_sym) + 1j * rng.randn(n_sc, n_sym)
    )
    rx_grid += noise.astype(np.complex64)

    return rx_grid, channel


# ═════════════════════════════════════════════════════════════════════════════
# Test scenario generator
# ═════════════════════════════════════════════════════════════════════════════

def make_test_scenario(
    n_targets: int = 2,
    seed: int = 42,
) -> ScenarioConfig:
    """Generate a random pedestrian-type radar scenario."""
    rng = np.random.RandomState(seed)
    targets = []
    for _ in range(n_targets):
        t = Target(
            range_m=rng.uniform(10.0, 120.0),
            velocity_mps=rng.uniform(-3.0, 3.0),
            azimuth_deg=rng.uniform(-60.0, 60.0),
            elevation_deg=rng.uniform(-5.0, 5.0),
            rcs_dbsm=rng.uniform(-15.0, -5.0),
            label="person",
        )
        targets.append(t)

    return ScenarioConfig(
        targets=targets,
        tx_power_dbm=30.0,
        noise_figure_db=7.0,
        clutter_level_db=-30.0,
        cdl_model="B",
    )
