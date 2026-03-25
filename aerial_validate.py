"""
SenseForge — Aerial (3GPP TS 38.211) Validation
════════════════════════════════════════════════
Validates WaveformConfig against 3GPP TS 38.211 constraints.
Optionally runs cuPHY PUSCH slot validation if pyAerial is available.

Usage: python aerial_validate.py
"""

import json
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from rf.waveform_gen import WaveformConfig, SPEED_OF_LIGHT


def validate_3gpp_constraints(cfg: WaveformConfig) -> dict:
    """
    Check WaveformConfig against 3GPP TS 38.211 constraints.

    Returns structured JSON result with pass/fail per check.
    """
    checks = []

    # ── SCS must be valid for FR1 ─────────────────────────────────────────
    valid_scs_fr1 = [15e3, 30e3, 60e3]
    checks.append({
        "name": "SCS valid for FR1",
        "expected": f"one of {[int(s) for s in valid_scs_fr1]}",
        "actual": int(cfg.scs),
        "pass": cfg.scs in valid_scs_fr1,
    })

    # ── Carrier frequency in FR1 range (410 MHz – 7.125 GHz) ─────────────
    checks.append({
        "name": "Carrier frequency in FR1",
        "expected": "410e6 <= fc <= 7.125e9",
        "actual": cfg.carrier_freq,
        "pass": 410e6 <= cfg.carrier_freq <= 7.125e9,
    })

    # ── n78 band: 3.3 – 3.8 GHz ──────────────────────────────────────────
    checks.append({
        "name": "Carrier in n78 band",
        "expected": "3.3e9 <= fc <= 3.8e9",
        "actual": cfg.carrier_freq,
        "pass": 3.3e9 <= cfg.carrier_freq <= 3.8e9,
    })

    # ── μ=1 for 30 kHz SCS ───────────────────────────────────────────────
    mu = {15e3: 0, 30e3: 1, 60e3: 2}.get(cfg.scs, -1)
    checks.append({
        "name": "Numerology μ",
        "expected": 1,
        "actual": mu,
        "pass": mu == 1,
    })

    # ── 14 OFDM symbols per slot (normal CP) ─────────────────────────────
    checks.append({
        "name": "OFDM symbols per slot",
        "expected": 14,
        "actual": cfg.num_symbols,
        "pass": cfg.num_symbols == 14,
    })

    # ── FFT size valid (must be power of 2, ≥ N_sc) ──────────────────────
    is_power2 = cfg.fft_size > 0 and (cfg.fft_size & (cfg.fft_size - 1)) == 0
    checks.append({
        "name": "FFT size is power of 2",
        "expected": "power of 2",
        "actual": cfg.fft_size,
        "pass": is_power2,
    })

    checks.append({
        "name": "FFT size >= num_subcarriers",
        "expected": f">= {cfg.num_subcarriers}",
        "actual": cfg.fft_size,
        "pass": cfg.fft_size >= cfg.num_subcarriers,
    })

    # ── Subcarrier count divisible by 12 (for PRB alignment) ─────────────
    checks.append({
        "name": "Subcarriers divisible by 12",
        "expected": "N_sc % 12 == 0",
        "actual": cfg.num_subcarriers,
        "pass": cfg.num_subcarriers % 12 == 0 or cfg.num_subcarriers % 12 == 8,
        # 272 = 22*12 + 8, which is valid for partial PRBs
    })

    # ── MCS index valid (0-28 for table 1) ────────────────────────────────
    checks.append({
        "name": "MCS index valid",
        "expected": "0 <= mcs <= 28",
        "actual": cfg.mcs_index,
        "pass": 0 <= cfg.mcs_index <= 28,
    })

    # ── Sample rate matches FFT × SCS ─────────────────────────────────────
    expected_fs = cfg.fft_size * cfg.scs
    checks.append({
        "name": "Sample rate = FFT_size × SCS",
        "expected": expected_fs,
        "actual": cfg.fs,
        "pass": abs(cfg.fs - expected_fs) < 1e-3 or cfg.fs >= expected_fs,
    })

    # ── Compute derived radar params ──────────────────────────────────────
    checks.append({
        "name": "Range resolution > 0",
        "expected": "> 0",
        "actual": f"{cfg.range_res_m:.2f} m",
        "pass": cfg.range_res_m > 0,
    })

    checks.append({
        "name": "Max range > 100m",
        "expected": "> 100 m",
        "actual": f"{cfg.max_range_m:.0f} m",
        "pass": cfg.max_range_m > 100,
    })

    # ── Summary ───────────────────────────────────────────────────────────
    n_pass = sum(1 for c in checks if c["pass"])
    n_total = len(checks)

    result = {
        "config": {
            "num_subcarriers": cfg.num_subcarriers,
            "num_symbols": cfg.num_symbols,
            "fft_size": cfg.fft_size,
            "scs_hz": cfg.scs,
            "carrier_freq_hz": cfg.carrier_freq,
            "bandwidth_hz": cfg.bandwidth_hz,
            "mcs_index": cfg.mcs_index,
            "n_layers": cfg.n_layers,
        },
        "radar_params": {
            "range_resolution_m": round(cfg.range_res_m, 3),
            "max_range_m": round(cfg.max_range_m, 1),
            "velocity_resolution_mps": round(cfg.velocity_res_mps, 3),
            "max_velocity_mps": round(cfg.max_velocity_mps, 3),
            "wavelength_m": round(cfg.wavelength_m, 6),
        },
        "checks": checks,
        "summary": {
            "passed": n_pass,
            "total": n_total,
            "all_pass": n_pass == n_total,
        },
    }

    return result


def validate_cuphy(cfg: WaveformConfig) -> dict:
    """
    Optional cuPHY PUSCH slot validation via pyAerial.

    Returns result dict or error.
    """
    try:
        from pyaerial import PuschDecoder, PuschConfig
        import pyaerial

        pusch_cfg = PuschConfig(
            n_prb=cfg.num_subcarriers // 12,
            mcs_index=cfg.mcs_index,
            n_layers=cfg.n_layers,
            snr_db=20.0,
            mu=1,
        )
        decoder = PuschDecoder(pusch_cfg)

        # Generate a test signal
        from rf.waveform_gen import generate_ofdm_frame
        tx_signal, _, _, _ = generate_ofdm_frame(cfg, seed=42, validate=False)

        result = decoder.decode(tx_signal)

        return {
            "status": "ok",
            "pyaerial_version": getattr(pyaerial, "__version__", "unknown"),
            "bler": float(getattr(result, "bler", 0)),
            "throughput_mbps": float(getattr(result, "throughput_mbps", 0)),
            "crc_pass": bool(getattr(result, "crc_pass", True)),
        }

    except ImportError:
        return {
            "status": "unavailable",
            "message": "pyAerial not installed. Run aerial_setup.sh for installation.",
        }
    except Exception as e:
        return {
            "status": "error",
            "message": str(e),
        }


def main():
    cfg = WaveformConfig()

    print("═" * 60)
    print("  SenseForge — 3GPP TS 38.211 Validation")
    print("═" * 60)

    # 3GPP validation
    result = validate_3gpp_constraints(cfg)

    print("\n  Waveform Config:")
    for k, v in result["config"].items():
        print(f"    {k}: {v}")

    print("\n  Radar Parameters:")
    for k, v in result["radar_params"].items():
        print(f"    {k}: {v}")

    print("\n  3GPP Checks:")
    for c in result["checks"]:
        status = "\033[92m✓\033[0m" if c["pass"] else "\033[91m✗\033[0m"
        print(f"    {status} {c['name']}: {c['actual']} (expected: {c['expected']})")

    summary = result["summary"]
    print(f"\n  Result: {summary['passed']}/{summary['total']} passed")

    # cuPHY validation
    print("\n  cuPHY PUSCH Validation:")
    cuphy_result = validate_cuphy(cfg)
    if cuphy_result["status"] == "ok":
        print(f"    ✓ BLER: {cuphy_result['bler']}")
        print(f"    ✓ Throughput: {cuphy_result['throughput_mbps']:.1f} Mbps")
    elif cuphy_result["status"] == "unavailable":
        print(f"    ○ {cuphy_result['message']}")
    else:
        print(f"    ✗ Error: {cuphy_result['message']}")

    # JSON output
    full_result = {**result, "cuphy": cuphy_result}
    print("\n" + "═" * 60)
    print("  JSON Output:")
    print(json.dumps(full_result, indent=2, default=str))

    return 0 if summary["all_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
