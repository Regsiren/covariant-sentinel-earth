"""Offline unit tests for the benchmark / null-hypothesis validation runner."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.benchmark_runner import (
    detect_persistent_blocks,
    remove_diurnal_cycle,
    run_permutation_test,
    run_validation_experiment,
)
from src.kinematic_core import CovariantSentinelAuditor

HOUR = 3600.0
DAY = 86400.0


def _make_synthetic_raw(n_days: int = 40, seed: int = 42):
    """40-day hourly HELIOS + ATMOS frames with one injected atmos anomaly."""
    rng = np.random.default_rng(seed)
    n = n_days * 24
    ts = np.arange(n, dtype=np.float64) * HOUR

    t = np.linspace(0.0, 1.0, n)
    bz = -2.0 + 0.05 * t
    v_sw = 400.0 + 2.0 * t
    p_dyn = 2.0 + 0.02 * t
    cme = slice(15 * 24, 15 * 24 + 12)
    bz[cme] = np.linspace(-2.0, -18.0, 12)
    v_sw[cme] = np.linspace(400.0, 750.0, 12)
    p_dyn[cme.start : cme.start + 6] = 2.05
    p_dyn[cme.start + 6 : cme.stop] = np.linspace(2.05, 18.0, 6)

    helios_raw = pd.DataFrame(
        {
            "timestamp": ts,
            "bz_gsm": bz,
            "v_sw": v_sw,
            "p_dyn": p_dyn,
        }
    )

    pressure = 1013.0 + 0.2 * np.sin(2 * np.pi * t) + 0.05 * rng.normal(size=n)
    temp = 12.0 + 1.5 * np.sin(2 * np.pi * t / 2.0) + 0.05 * rng.normal(size=n)
    wind = 5.0 + 0.3 * t + 0.02 * rng.normal(size=n)

    anom = slice(20 * 24, 20 * 24 + 36)  # 36 hours
    pressure[anom] = pressure[anom] - np.linspace(0.0, 25.0, 36)
    temp[anom] = temp[anom] + np.linspace(0.0, 12.0, 36)

    atmos_raw = pd.DataFrame(
        {
            "timestamp": ts,
            "pressure_msl": pressure,
            "temperature_2m": temp,
            "wind_speed_100m": wind,
        }
    )
    return helios_raw, atmos_raw


def test_run_validation_experiment_synthetic():
    """End-to-end synthetic run: detects event, null controls, non-empty report."""
    helios_raw, atmos_raw = _make_synthetic_raw()
    result = run_validation_experiment(
        helios_raw, atmos_raw, shift_days=[7, -7], mode="all"
    )

    assert "event_onsets" in result
    assert "scores" in result
    assert "null_control_summary" in result
    assert "coupled_df" in result
    assert "report_md" in result
    assert result["mode"] == "all"
    assert result["permutation"] is None

    assert len(result["event_onsets"]) >= 1

    required_detectors = {
        "legacy_trailing_z",
        "uncoupled_atmos",
        "coupled_helios_atmos",
        "coupled_veto_only",
    }
    assert required_detectors.issubset(result["scores"].keys())

    for name, score in result["scores"].items():
        assert score["events_total"] == len(result["event_onsets"])
        assert 0.0 <= score["recall"] <= 1.0
        assert "false_alert_hours" in score
        assert "f1" in score

    ns = result["null_control_summary"]
    assert ns["n_shifts"] == 2
    assert "mean_recall" in ns
    assert "mean_lead_hours" in ns
    assert "mean_false_alert_hours" in ns
    assert len(ns["per_shift"]) == 2

    assert isinstance(result["report_md"], str)
    assert len(result["report_md"].strip()) > 0
    assert "HELIOS→ATMOS Validation Report" in result["report_md"]
    assert "coupled_helios_atmos" in result["report_md"]
    assert "Precision" in result["report_md"]

    assert len(result["coupled_df"]) == len(atmos_raw) - 24
    assert "coupled_phase" in result["coupled_df"].columns
    for score in result["scores"].values():
        assert "precision" in score
        assert score["precision"] >= 0.0


def test_persistent_blocks_and_permutation_test():
    """72h positive pressure block is isolated; permutation p-values in (0, 1]."""
    rng = np.random.default_rng(99)
    n_days = 80
    n = n_days * 24
    ts = np.arange(n, dtype=np.float64) * HOUR
    t = np.linspace(0.0, 1.0, n)

    # Quiet pressure background
    pressure = 1015.0 + 0.3 * np.sin(2 * np.pi * t) + 0.05 * rng.normal(size=n)
    temp = 10.0 + 0.5 * np.sin(2 * np.pi * t) + 0.05 * rng.normal(size=n)
    wind = 4.0 + 0.1 * t

    # Short negative-pressure storm (~18h) — must be ignored by block detector
    storm = slice(25 * 24, 25 * 24 + 18)
    pressure[storm] = pressure[storm] - np.linspace(0.0, 20.0, 18)

    # Sustained 72-hour high-pressure blocking ridge
    block = slice(45 * 24, 45 * 24 + 72)
    pressure[block] = pressure[block] + np.linspace(8.0, 18.0, 72)

    atmos_raw = pd.DataFrame(
        {
            "timestamp": ts,
            "pressure_msl": pressure,
            "temperature_2m": temp,
            "wind_speed_100m": wind,
        }
    )

    onsets = detect_persistent_blocks(atmos_raw)
    assert len(onsets) >= 1
    # First onset should land near the block start (day 45), not the storm (day 25)
    block_start = float(ts[block.start])
    storm_start = float(ts[storm.start])
    assert abs(onsets[0] - block_start) < 3 * DAY
    assert all(abs(o - storm_start) > 5 * DAY for o in onsets)

    # Build auditor streams for a compact permutation test
    bz = -2.0 + 0.05 * t
    v_sw = 400.0 + 2.0 * t
    p_dyn = 2.0 + 0.02 * t
    # Mild HELIOS shear pulse before the block for non-trivial coupling
    pulse = slice(43 * 24, 44 * 24)
    p_dyn[pulse] = np.linspace(2.0, 12.0, pulse.stop - pulse.start)
    bz[pulse] = -10.0

    helios_raw = pd.DataFrame(
        {
            "timestamp": ts,
            "bz_gsm": bz,
            "v_sw": v_sw,
            "p_dyn": p_dyn,
        }
    )
    atmos_for_audit = remove_diurnal_cycle(atmos_raw)
    helios_aud = CovariantSentinelAuditor(
        persistence_steps=3, max_gap_seconds=2.0 * HOUR
    ).process_stream(
        helios_raw["timestamp"].to_numpy(dtype=np.float64),
        helios_raw[["bz_gsm", "v_sw", "p_dyn"]].to_numpy(dtype=np.float64),
    )
    atmos_aud = CovariantSentinelAuditor(
        persistence_steps=6, max_gap_seconds=2.0 * HOUR
    ).process_stream(
        atmos_for_audit["timestamp"].to_numpy(dtype=np.float64),
        atmos_for_audit[
            ["pressure_msl", "temperature_2m", "wind_speed_100m"]
        ].to_numpy(dtype=np.float64),
    )

    perm = run_permutation_test(
        helios_aud, atmos_aud, onsets, n_shifts=12, horizon_seconds=259200.0
    )
    assert 0.0 < perm["p_precision"] <= 1.0
    assert 0.0 < perm["p_f1"] <= 1.0
    assert 0.0 < perm["p_lead"] <= 1.0
    assert "report_md" in perm
    assert "Persistent-Block Permutation Test" in perm["report_md"]
    assert perm["n_shifts"] >= 1
