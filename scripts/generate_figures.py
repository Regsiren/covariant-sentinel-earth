#!/usr/bin/env python3
"""Generate publication-grade BIO_05 / multi-region benchmark figures."""

from __future__ import annotations

import re
import sys
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.bio_benchmark import run_bio_validation
from src.data_loaders import fetch_open_meteo_agro

FIG_DIR = ROOT / "outputs" / "figures"
REPORT_MD = ROOT / "outputs" / "multi_region_bio_report.md"

# Fallback pooled metrics (30 site-year run) if the Markdown report is absent
POOLED_FALLBACK = {
    "legacy_30d_soil_z": {
        "recall": 0.711,
        "f1": 0.245,
        "mean_lead_days": 12.56,
        "alert_rate_pct": 15.40,
    },
    "bio05_thinning_or_veto": {
        "recall": 0.789,
        "f1": 0.287,
        "mean_lead_days": 16.00,
        "alert_rate_pct": 28.05,
    },
    "bio05_veto_only": {
        "recall": 0.395,
        "f1": 0.153,
        "mean_lead_days": 16.33,
        "alert_rate_pct": 16.20,
    },
    "bio05_drying_thinning_or_veto": {
        "recall": 0.579,
        "f1": 0.181,
        "mean_lead_days": 13.77,
        "alert_rate_pct": 8.12,
    },
    "bio05_drying_veto_only": {
        "recall": 0.316,
        "f1": 0.146,
        "mean_lead_days": 15.33,
        "alert_rate_pct": 4.54,
    },
    "bio05_twostage_confirm": {
        "recall": 0.789,
        "f1": 0.248,
        "mean_lead_days": 13.60,
        "alert_rate_pct": 18.06,
    },
}

DETECTOR_LABELS = {
    "legacy_30d_soil_z": "legacy_30d_soil_z",
    "bio05_thinning_or_veto": "bio05_thinning_or_veto",
    "bio05_twostage_confirm": "bio05_twostage_confirm",
    "bio05_drying_thinning_or_veto": "bio05_drying_thinning_or_veto",
    "bio05_drying_veto_only": "bio05_drying_veto_only",
    "bio05_veto_only": "bio05_veto_only",
}


def _apply_pub_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "#333333",
            "axes.labelcolor": "#222222",
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "font.family": "DejaVu Sans",
            "axes.grid": True,
            "grid.color": "#dddddd",
            "grid.linewidth": 0.7,
            "grid.alpha": 0.9,
            "legend.fontsize": 8,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.facecolor": "white",
        }
    )


def _parse_pooled_metrics(report_path: Path) -> dict:
    """Parse pooled detector metrics from multi_region_bio_report.md."""
    if not report_path.exists():
        return {k: v.copy() for k, v in POOLED_FALLBACK.items()}

    text = report_path.read_text(encoding="utf-8")
    # Prefer the pooled summary table section
    marker = "### Pooled 30 Site-Year Summary Table"
    if marker in text:
        text = text.split(marker, 1)[1]

    metrics: dict = {}
    row_re = re.compile(
        r"\| `(?P<name>[^`]+)` \| "
        r"(?P<recall>[-0-9.]+) \| "
        r"(?P<precision>[-0-9.]+) \| "
        r"(?P<f1>[-0-9.]+) \| "
        r"(?P<lead>[-0-9.]+) \| "
        r"(?P<alert>[-0-9.]+) \| "
        r"(?P<fa>[-0-9.]+) \| "
        r"(?P<lp>[-0-9.]+) \|"
    )
    for m in row_re.finditer(text):
        name = m.group("name")
        if name not in DETECTOR_LABELS:
            continue
        metrics[name] = {
            "recall": float(m.group("recall")),
            "f1": float(m.group("f1")),
            "mean_lead_days": float(m.group("lead")),
            "alert_rate_pct": float(m.group("alert")),
        }

    if len(metrics) < len(DETECTOR_LABELS):
        # Fill any missing detectors from fallback
        for k, v in POOLED_FALLBACK.items():
            metrics.setdefault(k, v.copy())
    return metrics


def generate_figure1() -> Path:
    """2022 East Anglia flash drought case study (2-panel)."""
    print("Fetching East Anglia agro data (2020–2024) ...", file=sys.stderr)
    agro = fetch_open_meteo_agro(
        start_date="2020-01-01",
        end_date="2024-12-31",
        lat=52.6309,
        lon=1.2974,
    )
    print(f"  Daily rows: {len(agro)}", file=sys.stderr)
    result = run_bio_validation(agro)
    audited = result["audited_df"].copy()
    detector_masks = result["detector_masks"]

    agro = agro.sort_values("timestamp").reset_index(drop=True)
    dates = pd.to_datetime(agro["date"])
    agro = agro.assign(plot_date=dates)
    audited = audited.assign(
        plot_date=pd.to_datetime(audited["timestamp"], unit="s", utc=True).dt.tz_localize(
            None
        )
    )

    # Single source of truth from run_bio_validation
    legacy_alert = np.asarray(
        detector_masks["legacy_30d_soil_z"], dtype=bool
    )
    twostage = np.asarray(
        detector_masks["bio05_twostage_confirm"], dtype=bool
    )

    t0 = pd.Timestamp("2022-03-01")
    t1 = pd.Timestamp("2022-08-31")
    mask = (agro["plot_date"] >= t0) & (agro["plot_date"] <= t1)
    a_mask = (audited["plot_date"] >= t0) & (audited["plot_date"] <= t1)

    fig, (ax0, ax1) = plt.subplots(
        2, 1, figsize=(10.5, 7.2), sharex=True, constrained_layout=True
    )

    # --- Top: root-zone soil moisture ---
    x = agro.loc[mask, "plot_date"]
    y = agro.loc[mask, "soil_moisture_7_to_28cm"]
    ax0.plot(x, y, color="#1f4e79", lw=1.6, label="Root-zone SM (7–28 cm)")

    # Shade legacy soil-Z drought alerts (from detector_masks)
    leg = legacy_alert[mask.to_numpy()]
    xd = x.to_numpy()
    if len(xd) > 1:
        padded = np.concatenate([[False], leg, [False]])
        edges = np.diff(padded.astype(int))
        starts = np.where(edges == 1)[0]
        ends = np.where(edges == -1)[0] - 1
        for s, e in zip(starts, ends):
            ax0.axvspan(
                xd[s],
                xd[e],
                color="#f4a261",
                alpha=0.28,
                lw=0,
                label="legacy_30d_soil_z ≤ −1.5" if s == starts[0] else None,
            )

    onsets_2022 = [
        pd.Timestamp(d)
        for d in result["event_dates"]
        if d.startswith("2022-")
    ]
    for i, onset in enumerate(onsets_2022):
        if t0 <= onset <= t1:
            ax0.axvline(
                onset,
                color="#c1121f",
                ls="--",
                lw=1.2,
                alpha=0.9,
                label="Flash-drought onset" if i == 0 else None,
            )

    ax0.set_ylabel("Soil moisture (m³/m³)")
    ax0.set_title(
        "2022 East Anglia Flash Drought: BIO_05 Kinematic Early Warning "
        "vs. Root-Zone Soil Moisture",
        pad=8,
    )
    ax0.legend(loc="upper right", frameon=True, fancybox=False, edgecolor="#cccccc")

    # --- Bottom: omega_cal + thresholds + twostage markers ---
    xb = audited.loc[a_mask, "plot_date"]
    yb = audited.loc[a_mask, "omega_cal"]
    ax1.plot(xb, yb, color="#2a9d8f", lw=1.5, label=r"BIO_05 $\omega_{\mathrm{cal}}$")
    ax1.axhline(0.07, color="#e9c46a", ls="--", lw=1.1, label="Thinning (0.07)")
    ax1.axhline(0.19, color="#e76f51", ls="--", lw=1.1, label="Veto (0.19)")

    fire = twostage[a_mask.to_numpy()]
    xf = xb.to_numpy()[fire]
    yf = np.asarray(yb, dtype=float)[fire]
    # Prefer finite omega for markers
    finite = np.isfinite(yf)
    ax1.scatter(
        xf[finite],
        yf[finite],
        s=28,
        c="#9b2226",
        marker="o",
        zorder=5,
        label="bio05_twostage_confirm",
        edgecolors="white",
        linewidths=0.4,
    )

    ax1.set_ylabel(r"$\omega_{\mathrm{cal}}$")
    ax1.set_xlabel("Date (2022)")
    ax1.legend(loc="upper right", frameon=True, fancybox=False, edgecolor="#cccccc")
    ax1.xaxis.set_major_locator(mdates.MonthLocator())
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    ax1.set_xlim(t0, t1)

    out = FIG_DIR / "fig1_2022_east_anglia_drought.png"
    fig.savefig(out, dpi=300)
    plt.close(fig)
    print(f"Wrote {out}", file=sys.stderr)
    return out


def generate_figure2() -> Path:
    """30 site-year pooled operational frontier (lead vs alert rate)."""
    metrics = _parse_pooled_metrics(REPORT_MD)

    # Distinct colours / markers for each detector
    style = {
        "legacy_30d_soil_z": ("#1f4e79", "s"),
        "bio05_thinning_or_veto": ("#e76f51", "o"),
        "bio05_twostage_confirm": ("#9b2226", "D"),
        "bio05_drying_thinning_or_veto": ("#2a9d8f", "^"),
        "bio05_drying_veto_only": ("#457b9d", "v"),
        "bio05_veto_only": ("#6c757d", "P"),
    }
    # Annotation offsets (points) to reduce overlap
    offsets = {
        "legacy_30d_soil_z": (8, 10),
        "bio05_thinning_or_veto": (8, -18),
        "bio05_twostage_confirm": (-120, 12),
        "bio05_drying_thinning_or_veto": (8, 8),
        "bio05_drying_veto_only": (8, -16),
        "bio05_veto_only": (-5, 14),
    }

    fig, ax = plt.subplots(figsize=(10.0, 6.8), constrained_layout=True)
    for name in DETECTOR_LABELS:
        m = metrics[name]
        color, marker = style[name]
        ax.scatter(
            m["alert_rate_pct"],
            m["mean_lead_days"],
            s=90,
            c=color,
            marker=marker,
            zorder=3,
            edgecolors="white",
            linewidths=0.6,
            label=name,
        )
        dx, dy = offsets.get(name, (8, 8))
        ax.annotate(
            f"{name}\n(R={m['recall']:.2f}, F1={m['f1']:.2f})",
            xy=(m["alert_rate_pct"], m["mean_lead_days"]),
            xytext=(dx, dy),
            textcoords="offset points",
            fontsize=7.5,
            color="#222222",
            bbox=dict(
                boxstyle="round,pad=0.25",
                facecolor="white",
                edgecolor="#dddddd",
                alpha=0.92,
            ),
            arrowprops=dict(arrowstyle="-", color="#bbbbbb", lw=0.6),
        )

    ax.set_xlabel("Alert Rate (%)")
    ax.set_ylabel("Mean Lead Time (days)")
    ax.set_title(
        "30 Site-Year Multi-Region Flash Drought Benchmark "
        "(UK, US, Spain): Lead Time vs. Alert Load",
        pad=10,
    )
    ax.set_xlim(0, max(m["alert_rate_pct"] for m in metrics.values()) * 1.18)
    ymin = min(m["mean_lead_days"] for m in metrics.values()) - 1.5
    ymax = max(m["mean_lead_days"] for m in metrics.values()) + 2.5
    ax.set_ylim(ymin, ymax)
    ax.legend(
        loc="lower right",
        frameon=True,
        fancybox=False,
        edgecolor="#cccccc",
        title="Detector",
    )

    out = FIG_DIR / "fig2_multi_region_pareto.png"
    fig.savefig(out, dpi=300)
    plt.close(fig)
    print(f"Wrote {out}", file=sys.stderr)
    return out


def main() -> int:
    _apply_pub_style()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    f1 = generate_figure1()
    f2 = generate_figure2()
    print(f"Generated:\n  {f1}\n  {f2}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
