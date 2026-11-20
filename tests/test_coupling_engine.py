"""Unit / stress tests for HELIOS_01 -> ATMOS_02 coupling."""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import pandas as pd
import pytest

from src.coupling_engine import HeliosAtmosCoupler
from src.kinematic_core import CovariantSentinelAuditor

DAY = 86400.0
HOUR = 3600.0


def _coherent_smooth_noise(n: int, rng: np.random.Generator, scale: float = 1.0) -> np.ndarray:
    """Mild temporally-correlated noise shared across channels (laminar-safe)."""
    white = rng.normal(size=n)
    smooth = np.cumsum(white)
    smooth = (smooth - smooth[0]) / max(n, 1)
    return scale * smooth


def make_synthetic_telemetry(
    scenario: str = "quiet",
    seed: int = 7,
) -> Dict[str, np.ndarray]:
    """Generate 30-day multi-rate HELIOS_01 / ATMOS_02 telemetry.

    HELIOS_01: hourly (720 steps), channels B_z, V_sw, P_dyn.
    ATMOS_02: 6-hourly (120 steps), channels z500_anom, jet_lat_disp.

    Scenarios
    ---------
    quiet:
        Laminar baselines + mild coherent Gaussian background noise.
    cme_omega_block:
        Sustained CME shock in HELIOS at Day 15 (hours 360–420) and a building
        Omega Block ramp/instability in ATMOS starting at Day 16.5.
    """
    rng = np.random.default_rng(seed)

    n_helios = 720
    n_atmos = 120
    helios_ts = np.arange(n_helios, dtype=np.float64) * HOUR
    atmos_ts = np.arange(n_atmos, dtype=np.float64) * (6.0 * HOUR)

    t_h = np.linspace(0.0, 1.0, n_helios)
    t_a = np.linspace(0.0, 1.0, n_atmos)

    h_noise = _coherent_smooth_noise(n_helios, rng, scale=1.0)
    a_noise = _coherent_smooth_noise(n_atmos, rng, scale=1.0)

    bz = -2.0 + 0.05 * t_h + 0.01 * h_noise
    v_sw = 400.0 + 2.0 * t_h + 0.40 * h_noise
    p_dyn = 2.0 + 0.02 * t_h + 0.004 * h_noise

    z500 = 0.01 * t_a + 0.002 * a_noise
    jet = 0.005 * t_a + 0.001 * a_noise

    if scenario == "cme_omega_block":
        # Sustained Day-15 CME (hours 360–420) so EWMA excess stays elevated
        # under the 24h half-life / 72h window (not a 1-hour spike).
        cme_a = slice(360, 372)
        cme_b = slice(372, 420)
        n_a = cme_a.stop - cme_a.start
        n_b = cme_b.stop - cme_b.start

        bz[cme_a] = np.linspace(bz[cme_a.start - 1], -18.0, n_a)
        v_sw[cme_a] = np.linspace(v_sw[cme_a.start - 1], 750.0, n_a)
        p_dyn[cme_a] = p_dyn[cme_a.start - 1]

        # Repeated orthogonal shear pulses (P_dyn flips) through the CME window
        bz[cme_b] = -18.0
        v_sw[cme_b] = 750.0
        for j, idx in enumerate(range(cme_b.start, cme_b.stop)):
            if j % 3 == 0:
                p_dyn[idx] = 20.0
            elif j % 3 == 1:
                p_dyn[idx] = 4.0
            else:
                p_dyn[idx] = 16.0

        post = slice(420, n_helios)
        n_post = post.stop - post.start
        k = np.arange(n_post, dtype=np.float64)
        bz[post] = -18.0 + 0.01 * k / max(n_post, 1)
        v_sw[post] = 750.0 + 0.5 * k / max(n_post, 1)
        p_dyn[post] = 8.0 - 2.0 * k / max(n_post, 1)

        # Omega Block from Day 16.5 — sustained orthogonal instability
        # (≥ persistence_steps=6 consecutive 6-hourly samples).
        block_start = 66
        for i, idx in enumerate(range(block_start, n_atmos)):
            z500[idx] = 0.08 * i + 0.5
            jet[idx] = 0.0
            if i % 2 == 1:
                jet[idx] = -0.8 * (1.0 + 0.25 * i)
                z500[idx] = z500[idx - 1]
    elif scenario != "quiet":
        raise ValueError(f"unknown scenario: {scenario}")

    return {
        "helios_ts": helios_ts,
        "helios_raw": np.column_stack([bz, v_sw, p_dyn]),
        "atmos_ts": atmos_ts,
        "atmos_raw": np.column_stack([z500, jet]),
    }


def _audit_pair(
    telemetry: Dict[str, np.ndarray],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Audit HELIOS (hourly) and ATMOS (6-hourly) with domain gap gates."""
    helios_df = CovariantSentinelAuditor(
        persistence_steps=3, max_gap_seconds=2.0 * HOUR
    ).process_stream(telemetry["helios_ts"], telemetry["helios_raw"])
    atmos_df = CovariantSentinelAuditor(
        persistence_steps=6, max_gap_seconds=8.0 * HOUR
    ).process_stream(telemetry["atmos_ts"], telemetry["atmos_raw"])
    return helios_df, atmos_df


def _day_window_mask(
    timestamps: np.ndarray, day_lo: float, day_hi: float
) -> np.ndarray:
    return (timestamps >= day_lo * DAY) & (timestamps <= day_hi * DAY)


def test_quiet_control_no_false_alarms():
    """Quiet solar wind + quiet atmosphere → no post-calibration coupled VETO."""
    tel = make_synthetic_telemetry("quiet", seed=11)
    helios_df, atmos_df = _audit_pair(tel)
    coupled = HeliosAtmosCoupler().couple(helios_df, atmos_df)

    post_cal = coupled["timestamp"] >= (30 * 6.0 * HOUR)
    post = coupled.loc[post_cal]
    assert len(post) > 0
    assert post["omega_coupled"].notna().any()
    assert (post["coupled_phase"] == "NOMINAL").all()
    assert "VETO" not in set(post["coupled_phase"].unique())


def test_cme_accelerates_omega_block_warning():
    """Day-15 CME + Day-16.5 Omega Block: coupling raises omega and escalates."""
    tel = make_synthetic_telemetry("cme_omega_block", seed=21)
    helios_df, atmos_df = _audit_pair(tel)
    coupler = HeliosAtmosCoupler()
    coupled = coupler.couple(helios_df, atmos_df)

    # Wider window so 6-step persistence can complete under 6-hourly sampling
    win = _day_window_mask(coupled["timestamp"].to_numpy(), 16.0, 20.0)
    region = coupled.loc[win].copy()
    assert region["omega_cal"].notna().any()

    finite = region["omega_cal"].notna() & region["omega_coupled"].notna()
    assert (
        region.loc[finite, "omega_coupled"]
        >= region.loc[finite, "omega_cal"] - 1e-12
    ).all()
    assert float(region.loc[finite, "omega_coupled"].max()) > float(
        region.loc[finite, "omega_cal"].max()
    )

    coupled_alert = region["coupled_phase"].isin(["THINNING", "VETO"])
    assert coupled_alert.any(), (
        f"expected coupled escalation in Day 16–20, got {set(region['coupled_phase'])}"
    )

    atmos_alert = atmos_df.loc[win, "phase"].isin(["THINNING", "VETO"])
    if atmos_alert.any():
        t_unc = float(atmos_df.loc[win].loc[atmos_alert, "timestamp"].iloc[0])
        t_cpl = float(region.loc[coupled_alert, "timestamp"].iloc[0])
        assert t_cpl <= t_unc + 1e-9
    else:
        assert coupled_alert.any()


def test_time_shifted_null_removes_coupling_gain():
    """+10-day HELIOS shift breaks causal CME alignment → no coupling gain."""
    tel = make_synthetic_telemetry("cme_omega_block", seed=21)
    helios_df, atmos_df = _audit_pair(tel)

    coupler = HeliosAtmosCoupler()
    aligned = coupler.couple(helios_df, atmos_df)

    shifted_helios = helios_df.copy()
    shifted_helios["timestamp"] = shifted_helios["timestamp"] + 10.0 * DAY
    null = coupler.couple(shifted_helios, atmos_df)

    win = _day_window_mask(atmos_df["timestamp"].to_numpy(), 16.0, 20.0)
    a_win = aligned.loc[win]
    n_win = null.loc[win]

    aligned_gain = (
        a_win["omega_coupled"].to_numpy(dtype=np.float64)
        - a_win["omega_cal"].to_numpy(dtype=np.float64)
    )
    assert np.nanmax(aligned_gain) > 1e-4

    null_gain = (
        n_win["omega_coupled"].to_numpy(dtype=np.float64)
        - n_win["omega_cal"].to_numpy(dtype=np.float64)
    )
    assert np.nanmax(np.abs(null_gain)) < 1e-6
    assert np.nanmax(np.abs(null_gain)) < 0.1 * np.nanmax(aligned_gain)
