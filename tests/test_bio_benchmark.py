"""Offline unit tests for BIO_05 flash-drought benchmark."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.bio_benchmark import (
    bootstrap_site_year_metrics,
    detect_flash_drought_onsets,
    run_bio_validation,
)

DAY = 86400.0

DETECTORS = [
    "legacy_30d_soil_z",
    "bio05_thinning_or_veto",
    "bio05_veto_only",
    "bio05_drying_thinning_or_veto",
    "bio05_drying_veto_only",
    "bio05_twostage_confirm",
]


def _make_synthetic_agro(n_days: int = 365, seed: int = 7) -> pd.DataFrame:
    """365-day agro frame with an injected rapid soil-moisture crash + VPD/ET0 spike."""
    rng = np.random.default_rng(seed)
    ts = np.arange(n_days, dtype=np.float64) * DAY
    dates = pd.to_datetime(ts, unit="s", utc=True).strftime("%Y-%m-%d")

    sm0 = 0.28 + 0.01 * np.sin(2 * np.pi * np.arange(n_days) / 365.0)
    sm0 = sm0 + 0.002 * rng.normal(size=n_days)
    sm7 = 0.30 + 0.01 * np.sin(2 * np.pi * np.arange(n_days) / 365.0)
    sm7 = sm7 + 0.002 * rng.normal(size=n_days)
    et0 = 2.5 + 0.3 * np.sin(2 * np.pi * np.arange(n_days) / 365.0)
    et0 = et0 + 0.05 * rng.normal(size=n_days)
    vpd = 0.8 + 0.1 * np.sin(2 * np.pi * np.arange(n_days) / 365.0)
    vpd = vpd + 0.02 * rng.normal(size=n_days)

    onset = 200
    crash = slice(onset, onset + 14)
    dry = slice(onset + 14, onset + 30)
    sm7[crash] = np.linspace(float(sm7[onset - 1]), 0.12, 14)
    sm0[crash] = np.linspace(float(sm0[onset - 1]), 0.10, 14)
    et0[crash] = et0[crash] + np.linspace(0.5, 3.0, 14)
    vpd[crash] = vpd[crash] + np.linspace(0.2, 1.5, 14)
    vpd[onset + 7 : onset + 14] = vpd[onset + 7 : onset + 14] + 1.0
    et0[onset + 7 : onset + 14] = et0[onset + 7 : onset + 14] * 1.4

    sm7[dry] = 0.11 + 0.005 * rng.normal(size=dry.stop - dry.start)
    sm0[dry] = 0.09 + 0.005 * rng.normal(size=dry.stop - dry.start)
    et0[dry] = et0[dry] + 1.5
    vpd[dry] = vpd[dry] + 0.8

    return pd.DataFrame(
        {
            "timestamp": ts,
            "date": dates,
            "soil_moisture_0_to_7cm": sm0,
            "soil_moisture_7_to_28cm": sm7,
            "et0_fao_evapotranspiration": et0,
            "vapour_pressure_deficit": vpd,
        }
    )


def test_run_bio_validation_synthetic():
    """Synthetic flash drought is detected with valid BIO_05 metrics + report."""
    agro = _make_synthetic_agro()
    onsets = detect_flash_drought_onsets(agro)
    assert len(onsets) >= 1

    result = run_bio_validation(agro, horizon_days=21)
    assert "event_onsets" in result
    assert "event_dates" in result
    assert "scores" in result
    assert "audited_df" in result
    assert "detector_masks" in result
    assert "report_md" in result

    assert len(result["event_onsets"]) >= 1
    assert len(result["event_dates"]) == len(result["event_onsets"])

    required = set(DETECTORS)
    assert required.issubset(result["scores"].keys())
    assert required.issubset(result["detector_masks"].keys())

    for name, score in result["scores"].items():
        assert score["events_total"] == len(result["event_onsets"])
        assert 0.0 <= score["recall"] <= 1.0
        assert score["precision"] >= 0.0
        assert "f1" in score
        assert "mean_lead_days" in score
        assert "false_alert_days" in score
        assert "alert_rate_pct" in score
        assert "lead_per_100_fa" in score

    for name, mask in result["detector_masks"].items():
        assert isinstance(mask, np.ndarray)
        assert mask.dtype == bool
        assert len(mask) == len(agro)

    assert isinstance(result["report_md"], str)
    assert "BIO_05 Flash Drought Validation Report" in result["report_md"]
    assert "bio05_thinning_or_veto" in result["report_md"]
    assert "bio05_drying_thinning_or_veto" in result["report_md"]
    assert "bio05_twostage_confirm" in result["report_md"]
    assert "Lead/100 FA" in result["report_md"]
    assert len(result["audited_df"]) == len(agro)
    assert "phase" in result["audited_df"].columns


def test_bootstrap_site_year_metrics_deterministic():
    """Bootstrap CIs are deterministic under a fixed seed with ordered bounds."""
    rng = np.random.default_rng(0)
    blocks = []
    for i in range(4):
        n = 40
        ts = (np.arange(n, dtype=np.float64) + i * 100) * DAY
        # One onset mid-block for odd blocks; empty for even
        onsets = [float(ts[20])] if i % 2 == 1 else []
        legacy = rng.random(n) > 0.85
        bio = rng.random(n) > 0.80
        # Ensure at least one alert near onset for odd blocks
        if onsets:
            legacy[18:23] = True
            bio[15:22] = True
        blocks.append(
            {
                "region": f"R{i}",
                "year": 2015 + i,
                "timestamps": ts,
                "event_onsets": onsets,
                "masks": {
                    "legacy_30d_soil_z": legacy.astype(bool),
                    "bio05_twostage_confirm": bio.astype(bool),
                },
            }
        )

    names = ["legacy_30d_soil_z", "bio05_twostage_confirm"]
    a = bootstrap_site_year_metrics(
        blocks, detector_names=names, n_boot=50, seed=42
    )
    b = bootstrap_site_year_metrics(
        blocks, detector_names=names, n_boot=50, seed=42
    )

    assert a["n_boot"] == 50
    assert a["n_blocks"] == 4
    assert a["seed"] == 42
    assert "ci_report_md" in a
    assert "95% Site-Year Block Bootstrap" in a["ci_report_md"]

    for name in names:
        for metric in ("recall", "precision", "f1", "mean_lead_days"):
            ca = a["ci_metrics"][name][metric]
            cb = b["ci_metrics"][name][metric]
            assert ca["ci95_low"] == cb["ci95_low"]
            assert ca["ci95_high"] == cb["ci95_high"]
            assert ca["boot_mean"] == cb["boot_mean"]
            if np.isfinite(ca["ci95_low"]) and np.isfinite(ca["ci95_high"]):
                assert ca["ci95_low"] <= ca["ci95_high"]

    delta_a = a["ci_deltas"]["bio05_twostage_confirm"]
    delta_b = b["ci_deltas"]["bio05_twostage_confirm"]
    for key in ("delta_recall", "delta_f1", "delta_lead_days"):
        assert delta_a[key]["ci95_low"] == delta_b[key]["ci95_low"]
        assert delta_a[key]["ci95_high"] == delta_b[key]["ci95_high"]
        if np.isfinite(delta_a[key]["ci95_low"]) and np.isfinite(
            delta_a[key]["ci95_high"]
        ):
            assert delta_a[key]["ci95_low"] <= delta_a[key]["ci95_high"]
