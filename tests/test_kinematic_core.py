"""Unit tests for CovariantSentinelAuditor."""

from __future__ import annotations

import numpy as np
import pytest

from src.kinematic_core import CovariantSentinelAuditor


def _uniform_timestamps(n: int, dt: float = 60.0) -> np.ndarray:
    return np.arange(n, dtype=np.float64) * dt


def test_laminar_flow_zero_shear():
    """Smooth linear ramp across 3 channels → near-zero shear, stays NOMINAL."""
    n = 120
    timestamps = _uniform_timestamps(n)
    ramp = np.linspace(0.0, 10.0, n, dtype=np.float64)
    raw = np.column_stack([ramp, ramp * 0.5, ramp * 1.5])

    auditor = CovariantSentinelAuditor()
    df = auditor.process_stream(timestamps, raw)

    valid = df["a_perp_mag"].notna()
    assert valid.sum() > 50
    # Laminar co-linear motion should produce negligible transverse shear
    assert float(df.loc[valid, "a_perp_mag"].max()) < 1e-6
    assert set(df.loc[valid, "phase"].unique()) == {"NOMINAL"}


def test_orthogonal_shock_triggers_veto():
    """Sudden orthogonal direction flip spikes shear and escalates past persistence."""
    n = 100
    timestamps = _uniform_timestamps(n)
    raw = np.zeros((n, 3), dtype=np.float64)

    # Laminar motion along channel-0 for the first half
    raw[:50, 0] = np.linspace(0.0, 5.0, 50)
    # Abrupt orthogonal flip: freeze ch0, ramp ch1 (perpendicular in feature space)
    raw[50:, 0] = 5.0
    raw[50:, 1] = np.linspace(0.0, 8.0, 50)

    auditor = CovariantSentinelAuditor(persistence_steps=3)
    df = auditor.process_stream(timestamps, raw)

    # Transverse magnitude should spike around the direction change
    shock_region = df.iloc[48:60]
    assert float(shock_region["a_perp_mag"].max()) > 0.1

    # Once sustained past persistence_steps, phase must reach THINNING or VETO
    late = df.iloc[50:]
    phases = set(late["phase"].unique())
    assert phases & {"THINNING", "VETO"}, f"expected escalation, got {phases}"


def test_gap_masking_prevents_false_alarm():
    """A 10-hour gap with a step jump is masked and must not false-trigger VETO."""
    n = 80
    timestamps = _uniform_timestamps(n, dt=60.0)
    # Inject a 10-hour gap between index 39 and 40
    timestamps[40:] = timestamps[40:] + (10.0 * 3600.0 - 60.0)

    raw = np.zeros((n, 2), dtype=np.float64)
    raw[:40, 0] = np.linspace(0.0, 2.0, 40)
    # Large discontinuous jump across the gap (would look like a shock without masking)
    raw[40:, 0] = np.linspace(50.0, 52.0, 40)
    raw[40:, 1] = np.linspace(0.0, 1.0, 40)

    auditor = CovariantSentinelAuditor()
    df = auditor.process_stream(timestamps, raw)

    assert bool(df.loc[40, "gap_masked"]) is True
    assert df["gap_masked"].sum() >= 1

    # No VETO anywhere: the gap jump must not be treated as kinematic shear
    assert "VETO" not in set(df["phase"].unique())
    # Kinematic fields on the gap row itself should be undefined
    assert not np.isfinite(df.loc[40, "a_perp_mag"])


def test_anisotropy_whitening():
    """Large-scale + tiny-scale channels: orthogonal shock in tiny channel still fires."""
    n = 100
    timestamps = _uniform_timestamps(n)
    raw = np.zeros((n, 2), dtype=np.float64)

    # Channel 0: huge laminar scale (~1e4)
    raw[:, 0] = np.linspace(0.0, 1.0e4, n)
    # Channel 1: tiny scale (~1e-2); orthogonal shock mid-stream
    raw[:50, 1] = np.linspace(0.0, 1.0e-2, 50)
    raw[50:, 1] = np.linspace(1.0e-2, -2.0e-2, 50)  # reverse / orthogonal kink

    # Without whitening the huge channel would drown the tiny shock; with whitening
    # the auditor must still escalate.
    auditor = CovariantSentinelAuditor(persistence_steps=3)
    df = auditor.process_stream(timestamps, raw)

    late = df.iloc[50:]
    assert float(late["a_perp_mag"].max()) > 0.05
    phases = set(late["phase"].unique())
    assert phases & {"THINNING", "VETO"}, (
        f"whitening failed to surface small-channel shock; phases={phases}"
    )
