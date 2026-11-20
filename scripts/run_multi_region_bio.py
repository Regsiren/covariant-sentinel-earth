#!/usr/bin/env python3
"""30 site-year BIO_05 multi-region flash-drought benchmark (2015–2024)."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.bio_benchmark import (
    _f1_score,
    bootstrap_site_year_metrics,
    run_bio_validation,
)
from src.data_loaders import fetch_open_meteo_agro

REGIONS = [
    {
        "name": "East_Anglia_UK",
        "climate": "Temperate Maritime",
        "lat": 52.6309,
        "lon": 1.2974,
    },
    {
        "name": "Iowa_US_CornBelt",
        "climate": "Continental",
        "lat": 42.0308,
        "lon": -93.6319,
    },
    {
        "name": "Castile_Spain",
        "climate": "Semi-Arid Mediterranean",
        "lat": 39.2796,
        "lon": -3.0977,
    },
]

START = "2015-01-01"
END = "2024-12-31"
YEARS = list(range(2015, 2025))

DETECTOR_ORDER = [
    "legacy_30d_soil_z",
    "bio05_thinning_or_veto",
    "bio05_veto_only",
    "bio05_drying_thinning_or_veto",
    "bio05_drying_veto_only",
    "bio05_twostage_confirm",
]


def _format_score_table(scores: Dict[str, dict], title: str, n_events: int) -> str:
    lines = [
        f"### {title}",
        "",
        f"- **Flash drought events:** {n_events}",
        "",
        "| Detector | Recall | Precision | F1 | Mean Lead (d) | Alert Rate (%) | False Alert Days | Lead/100 FA |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name in DETECTOR_ORDER:
        s = scores[name]
        lead = s["mean_lead_days"]
        lead_s = f"{lead:.2f}" if np.isfinite(lead) else "n/a"
        lp = s.get("lead_per_100_fa", float("nan"))
        lp_s = f"{lp:.2f}" if np.isfinite(lp) else "n/a"
        lines.append(
            f"| `{name}` | {s['recall']:.3f} | {s['precision']:.3f} | {s['f1']:.3f} | "
            f"{lead_s} | {s['alert_rate_pct']:.2f} | {s['false_alert_days']} | {lp_s} |"
        )
    lines.append("")
    return "\n".join(lines)


def _pool_scores(region_results: List[dict]) -> Dict[str, dict]:
    """Aggregate detector scores across regions into pooled site-year metrics."""
    pooled: Dict[str, dict] = {}
    for name in DETECTOR_ORDER:
        ev_tot = 0
        ev_det = 0
        episodes = 0
        fa_days = 0
        lead_num = 0.0
        lead_den = 0
        alert_rates: List[float] = []

        for rr in region_results:
            s = rr["scores"][name]
            ev_tot += int(s["events_total"])
            ev_det += int(s["events_detected"])
            episodes += int(s["alert_episodes"])
            fa_days += int(s["false_alert_days"])
            alert_rates.append(float(s["alert_rate_pct"]))
            if np.isfinite(s["mean_lead_days"]) and int(s["events_detected"]) > 0:
                lead_num += float(s["mean_lead_days"]) * int(s["events_detected"])
                lead_den += int(s["events_detected"])

        recall = float(ev_det) / float(ev_tot) if ev_tot > 0 else 0.0
        precision = float(ev_det) / float(episodes) if episodes > 0 else 0.0
        mean_lead = float(lead_num) / float(lead_den) if lead_den > 0 else float("nan")
        lead_per_100_fa = (
            (mean_lead / max(float(fa_days), 1.0)) * 100.0
            if np.isfinite(mean_lead)
            else float("nan")
        )
        pooled[name] = {
            "events_total": ev_tot,
            "events_detected": ev_det,
            "recall": recall,
            "precision": precision,
            "f1": _f1_score(precision, recall),
            "alert_episodes": episodes,
            "mean_lead_days": mean_lead,
            "alert_rate_pct": float(np.mean(alert_rates)) if alert_rates else 0.0,
            "false_alert_days": fa_days,
            "lead_per_100_fa": float(lead_per_100_fa),
        }
    return pooled


def _slice_site_year_blocks(
    region_name: str,
    timestamps: np.ndarray,
    event_onsets: list[float],
    detector_masks: Dict[str, np.ndarray],
    years: List[int] | None = None,
) -> List[dict]:
    """Slice a regional validation into annual (region, year) bootstrap blocks."""
    if years is None:
        years = YEARS
    ts = np.asarray(timestamps, dtype=np.float64)
    years_arr = pd.to_datetime(ts, unit="s", utc=True).year.to_numpy()
    blocks: List[dict] = []
    for year in years:
        sel = years_arr == int(year)
        if not np.any(sel):
            continue
        block_ts = ts[sel]
        t_lo = float(block_ts.min())
        t_hi = float(block_ts.max())
        block_onsets = [e for e in event_onsets if t_lo <= float(e) <= t_hi]
        masks = {
            name: np.asarray(mask, dtype=bool)[sel]
            for name, mask in detector_masks.items()
        }
        blocks.append(
            {
                "region": region_name,
                "year": int(year),
                "timestamps": block_ts,
                "event_onsets": block_onsets,
                "masks": masks,
            }
        )
    return blocks


def main() -> int:
    region_results: List[dict] = []
    site_year_blocks: List[dict] = []
    sections: List[str] = [
        "# BIO_05 Multi-Region Flash Drought Benchmark (2015–2024)",
        "",
        f"- **Window:** {START} → {END} (10 years × 3 regions = 30 site-years)",
        "",
    ]

    for region in REGIONS:
        name = region["name"]
        print(
            f"Fetching {name} ({region['climate']}) "
            f"@ ({region['lat']}, {region['lon']}) ...",
            file=sys.stderr,
        )
        agro = fetch_open_meteo_agro(
            start_date=START,
            end_date=END,
            lat=region["lat"],
            lon=region["lon"],
        )
        print(f"  Daily rows: {len(agro)}", file=sys.stderr)
        if agro.empty:
            print(f"ERROR: empty agro frame for {name}", file=sys.stderr)
            return 1

        print(f"  Running BIO_05 validation for {name} ...", file=sys.stderr)
        result = run_bio_validation(agro)
        n_events = len(result["event_onsets"])
        region_results.append(
            {
                "name": name,
                "climate": region["climate"],
                "n_events": n_events,
                "event_dates": result["event_dates"],
                "scores": result["scores"],
            }
        )
        blocks = _slice_site_year_blocks(
            region_name=name,
            timestamps=agro["timestamp"].to_numpy(dtype=np.float64),
            event_onsets=result["event_onsets"],
            detector_masks=result["detector_masks"],
        )
        site_year_blocks.extend(blocks)
        print(f"  Site-year blocks: {len(blocks)}", file=sys.stderr)

        sections.append(
            _format_score_table(
                result["scores"],
                title=f"{name} ({region['climate']})",
                n_events=n_events,
            )
        )
        sections.append(
            f"- **Event dates:** "
            f"{', '.join(result['event_dates']) if result['event_dates'] else 'none'}"
        )
        sections.append("")

    pooled = _pool_scores(region_results)
    total_events = sum(r["n_events"] for r in region_results)
    sections.extend(
        [
            "## Pooled 30 Site-Year Summary",
            "",
            f"- **Total flash drought events:** {total_events}",
            "- **Events per region:** "
            + ", ".join(f"{r['name']}={r['n_events']}" for r in region_results),
            "",
            _format_score_table(
                pooled,
                title="Pooled 30 Site-Year Summary Table",
                n_events=total_events,
            ),
        ]
    )

    print(
        f"Running site-year block bootstrap "
        f"(n_blocks={len(site_year_blocks)}, n_boot=1000, seed=42) ...",
        file=sys.stderr,
    )
    boot = bootstrap_site_year_metrics(
        site_year_blocks,
        detector_names=DETECTOR_ORDER,
        baseline_name="legacy_30d_soil_z",
        n_boot=1000,
        seed=42,
    )
    sections.append(boot["ci_report_md"])

    report_md = "\n".join(sections).rstrip() + "\n"
    print(report_md)

    out_path = ROOT / "outputs" / "multi_region_bio_report.md"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report_md, encoding="utf-8")
    print(f"Wrote pooled report → {out_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
