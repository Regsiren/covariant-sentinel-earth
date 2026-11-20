"""BIO_05 Flash Drought early-warning benchmark on agro-hydrological streams."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd

from .kinematic_core import CovariantSentinelAuditor

DAY = 86400.0


def detect_flash_drought_onsets(
    agro_df: pd.DataFrame,
    percentile_thresh: float = 0.20,
    min_drop_14d: float = 0.03,
    min_duration_days: int = 10,
    merge_gap_days: int = 30,
) -> list[float]:
    """Detect ground-truth flash drought onsets from root-zone soil moisture.

    A day is flagged when root-zone moisture is at/below the empirical
    ``percentile_thresh`` quantile and a rapid 14-day drying pulse has
    occurred. Contiguous episodes meeting ``min_duration_days`` are merged
    across gaps shorter than ``merge_gap_days``.
    """
    required = {"timestamp", "soil_moisture_7_to_28cm"}
    if not required.issubset(agro_df.columns):
        raise ValueError(f"agro_df must contain {sorted(required)}")

    df = agro_df.sort_values("timestamp", kind="mergesort").reset_index(drop=True)
    soil = df["soil_moisture_7_to_28cm"].astype(np.float64)
    ts = df["timestamp"].to_numpy(dtype=np.float64)

    q20 = float(soil.quantile(float(percentile_thresh)))
    drop_14d = soil.shift(14) - soil
    rapid = drop_14d.rolling(14, min_periods=1).max() >= float(min_drop_14d)
    anom = ((soil <= q20) & rapid).to_numpy(dtype=bool)

    runs: List[tuple[int, int]] = []
    in_run = False
    start = 0
    for i, flag in enumerate(anom):
        if flag and not in_run:
            start = i
            in_run = True
        elif not flag and in_run:
            runs.append((start, i - 1))
            in_run = False
    if in_run:
        runs.append((start, len(anom) - 1))

    kept: List[tuple[int, int]] = []
    for a, b in runs:
        duration_d = (ts[b] - ts[a]) / DAY + 1.0
        if duration_d >= float(min_duration_days):
            kept.append((a, b))

    if not kept:
        return []

    merged: List[tuple[int, int]] = [kept[0]]
    for a, b in kept[1:]:
        prev_a, prev_b = merged[-1]
        gap_d = (ts[a] - ts[prev_b]) / DAY
        if gap_d < float(merge_gap_days):
            merged[-1] = (prev_a, b)
        else:
            merged.append((a, b))

    return sorted(float(ts[a]) for a, _ in merged)


def _count_alert_episodes_days(
    timestamps: np.ndarray,
    alert_mask: np.ndarray,
    merge_gap_days: float = 14.0,
) -> int:
    """Count distinct alert episodes, merging runs separated by < merge_gap_days."""
    timestamps = np.asarray(timestamps, dtype=np.float64)
    alert_mask = np.asarray(alert_mask, dtype=bool)
    if not np.any(alert_mask):
        return 0

    runs: List[tuple[float, float]] = []
    in_run = False
    start_ts = 0.0
    last_ts = 0.0
    for t, flag in zip(timestamps, alert_mask):
        if flag and not in_run:
            start_ts = float(t)
            last_ts = float(t)
            in_run = True
        elif flag and in_run:
            last_ts = float(t)
        elif not flag and in_run:
            runs.append((start_ts, last_ts))
            in_run = False
    if in_run:
        runs.append((start_ts, last_ts))

    if not runs:
        return 0

    merged = [runs[0]]
    for a, b in runs[1:]:
        prev_a, prev_b = merged[-1]
        gap_d = (a - prev_b) / DAY
        if gap_d < float(merge_gap_days):
            merged[-1] = (prev_a, b)
        else:
            merged.append((a, b))
    return len(merged)


def _f1_score(precision: float, recall: float) -> float:
    denom = float(precision) + float(recall)
    if denom <= 0.0:
        return 0.0
    return float(2.0 * precision * recall / denom)


def _score_bio_detector(
    timestamps: np.ndarray,
    alert_mask: np.ndarray,
    event_onsets: list[float],
    horizon_days: int = 21,
) -> dict:
    """Score a daily alert stream against flash-drought onsets."""
    timestamps = np.asarray(timestamps, dtype=np.float64)
    alert_mask = np.asarray(alert_mask, dtype=bool)
    horizon_seconds = float(horizon_days) * DAY

    events_total = len(event_onsets)
    lead_days: List[float] = []
    events_detected = 0

    for t_e in event_onsets:
        window = (timestamps >= (t_e - horizon_seconds)) & (timestamps <= t_e)
        hits = alert_mask & window
        if np.any(hits):
            events_detected += 1
            first_alert_ts = float(timestamps[hits].min())
            lead_days.append((t_e - first_alert_ts) / DAY)

    recall = (
        float(events_detected) / float(events_total) if events_total > 0 else 0.0
    )
    mean_lead = float(np.mean(lead_days)) if lead_days else float("nan")
    n = max(len(timestamps), 1)
    alert_rate_pct = 100.0 * float(np.sum(alert_mask)) / float(n)

    post_buffer = 14.0 * DAY
    in_event_window = np.zeros(len(timestamps), dtype=bool)
    for t_e in event_onsets:
        in_event_window |= (timestamps >= (t_e - horizon_seconds)) & (
            timestamps <= (t_e + post_buffer)
        )
    false_alert_days = int(np.sum(alert_mask & ~in_event_window))

    n_episodes = _count_alert_episodes_days(
        timestamps, alert_mask, merge_gap_days=14.0
    )
    precision = (
        float(events_detected) / float(n_episodes) if n_episodes > 0 else 0.0
    )
    lead_per_100_fa = (
        (mean_lead / max(float(false_alert_days), 1.0)) * 100.0
        if np.isfinite(mean_lead)
        else float("nan")
    )

    return {
        "events_total": int(events_total),
        "events_detected": int(events_detected),
        "recall": float(recall),
        "precision": float(precision),
        "f1": _f1_score(precision, recall),
        "alert_episodes": int(n_episodes),
        "mean_lead_days": mean_lead,
        "alert_rate_pct": float(alert_rate_pct),
        "false_alert_days": int(false_alert_days),
        "lead_per_100_fa": float(lead_per_100_fa),
    }


def run_bio_validation(agro_df: pd.DataFrame, horizon_days: int = 21) -> dict:
    """Validate BIO_05 flash-drought early warning against legacy soil-Z baseline."""
    required = {
        "timestamp",
        "soil_moisture_0_to_7cm",
        "soil_moisture_7_to_28cm",
        "et0_fao_evapotranspiration",
        "vapour_pressure_deficit",
    }
    missing = sorted(required - set(agro_df.columns))
    if missing:
        raise ValueError(f"agro_df missing columns: {missing}")

    agro = agro_df.sort_values("timestamp", kind="mergesort").reset_index(drop=True)
    event_onsets = detect_flash_drought_onsets(agro)
    event_dates = [
        datetime.fromtimestamp(t, tz=timezone.utc).strftime("%Y-%m-%d")
        for t in event_onsets
    ]

    auditor_cols = [
        "soil_moisture_0_to_7cm",
        "et0_fao_evapotranspiration",
        "vapour_pressure_deficit",
    ]
    auditor = CovariantSentinelAuditor(
        max_gap_seconds=172800.0,
        persistence_steps=3,
        calibration_window=30,
    )
    audited_df = auditor.process_stream(
        agro["timestamp"].to_numpy(dtype=np.float64),
        agro[auditor_cols].to_numpy(dtype=np.float64),
    )

    # Legacy trailing 30-day Z of root-zone soil moisture
    soil = agro["soil_moisture_7_to_28cm"].astype(np.float64)
    roll = soil.rolling(window=30, min_periods=14)
    z_soil = (soil - roll.mean()) / roll.std(ddof=0).replace(0.0, np.nan)
    legacy_alert = (z_soil <= -1.5).fillna(False).to_numpy(dtype=bool)

    # Strictly causal, zero-parameter physical drying-direction gate
    # (suppresses unsigned shear false alarms from sudden wetting/rain shocks)
    sm_surf = agro["soil_moisture_0_to_7cm"].astype(np.float64)
    vpd = agro["vapour_pressure_deficit"].astype(np.float64)
    sm_mean_30 = sm_surf.rolling(30, min_periods=7).mean()
    vpd_mean_30 = vpd.rolling(30, min_periods=7).mean()
    is_drying = (
        (sm_surf < sm_mean_30) & (vpd > vpd_mean_30)
    ).fillna(False).to_numpy(dtype=bool)

    cue_active = audited_df["phase"].isin(["THINNING", "VETO"])
    phase_thin_veto = cue_active.to_numpy(dtype=bool)
    phase_veto = (audited_df["phase"] == "VETO").to_numpy(dtype=bool)
    # Trailing 6-day window = today + prior 5 days (strictly causal, no look-ahead)
    cue_recent_5d = (
        cue_active.astype(np.float64)
        .rolling(6, min_periods=1)
        .max()
        .fillna(0.0)
        .astype(bool)
        .to_numpy(dtype=bool)
    )
    twostage_alert = cue_recent_5d & is_drying

    ts_agro = agro["timestamp"].to_numpy(dtype=np.float64)

    # Single source of truth: boolean masks aligned with agro_df rows
    detector_masks: Dict[str, np.ndarray] = {
        "legacy_30d_soil_z": legacy_alert.astype(bool),
        "bio05_thinning_or_veto": phase_thin_veto.astype(bool),
        "bio05_veto_only": phase_veto.astype(bool),
        "bio05_drying_thinning_or_veto": (phase_thin_veto & is_drying).astype(bool),
        "bio05_drying_veto_only": (phase_veto & is_drying).astype(bool),
        "bio05_twostage_confirm": twostage_alert.astype(bool),
    }

    scores: Dict[str, dict] = {
        name: _score_bio_detector(
            ts_agro,
            mask,
            event_onsets,
            horizon_days=horizon_days,
        )
        for name, mask in detector_masks.items()
    }

    report_md = _format_bio_report_md(
        event_onsets=event_onsets,
        event_dates=event_dates,
        scores=scores,
        horizon_days=horizon_days,
    )

    return {
        "event_onsets": event_onsets,
        "event_dates": event_dates,
        "scores": scores,
        "audited_df": audited_df,
        "detector_masks": detector_masks,
        "report_md": report_md,
    }


def _pool_blocks_and_score(
    blocks: Sequence[dict],
    detector_names: Sequence[str],
    horizon_days: int = 21,
) -> Dict[str, dict]:
    """Score each site-year block independently, then aggregate pooled metrics.

    Blocks are scored separately (not timestamp-concatenated) so overlapping
    calendars across regions cannot cross-contaminate alert horizons.
    """
    empty = {
        "events_total": 0,
        "events_detected": 0,
        "recall": 0.0,
        "precision": 0.0,
        "f1": 0.0,
        "alert_episodes": 0,
        "mean_lead_days": float("nan"),
        "alert_rate_pct": 0.0,
        "false_alert_days": 0,
        "lead_per_100_fa": float("nan"),
    }
    if not blocks:
        return {name: empty.copy() for name in detector_names}

    pooled: Dict[str, dict] = {}
    for name in detector_names:
        ev_tot = 0
        ev_det = 0
        episodes = 0
        fa_days = 0
        n_days = 0
        alert_days = 0
        lead_num = 0.0
        lead_den = 0

        for b in blocks:
            ts = np.asarray(b["timestamps"], dtype=np.float64)
            mask = np.asarray(b["masks"][name], dtype=bool)
            onsets = [float(x) for x in b["event_onsets"]]
            s = _score_bio_detector(
                ts, mask, onsets, horizon_days=horizon_days
            )
            ev_tot += int(s["events_total"])
            ev_det += int(s["events_detected"])
            episodes += int(s["alert_episodes"])
            fa_days += int(s["false_alert_days"])
            n_days += max(len(ts), 0)
            alert_days += int(np.sum(mask))
            if np.isfinite(s["mean_lead_days"]) and int(s["events_detected"]) > 0:
                lead_num += float(s["mean_lead_days"]) * int(s["events_detected"])
                lead_den += int(s["events_detected"])

        recall = float(ev_det) / float(ev_tot) if ev_tot > 0 else 0.0
        precision = float(ev_det) / float(episodes) if episodes > 0 else 0.0
        mean_lead = (
            float(lead_num) / float(lead_den) if lead_den > 0 else float("nan")
        )
        alert_rate = (
            100.0 * float(alert_days) / float(n_days) if n_days > 0 else 0.0
        )
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
            "alert_rate_pct": alert_rate,
            "false_alert_days": fa_days,
            "lead_per_100_fa": float(lead_per_100_fa),
        }
    return pooled


def bootstrap_site_year_metrics(
    site_year_blocks: list[dict],
    detector_names: list[str],
    baseline_name: str = "legacy_30d_soil_z",
    n_boot: int = 1000,
    seed: int = 42,
    horizon_days: int = 21,
) -> dict:
    """Deterministic site-year block bootstrap with 95% CIs and paired deltas.

    Each block is a ``(region, year)`` dict with ``timestamps``, ``event_onsets``,
    and ``masks`` (detector → boolean array aligned with timestamps).
    """
    if baseline_name not in detector_names:
        raise ValueError(f"baseline_name '{baseline_name}' must be in detector_names")
    if not site_year_blocks:
        raise ValueError("site_year_blocks must be non-empty")

    n_blocks = len(site_year_blocks)
    rng = np.random.default_rng(int(seed))

    metric_keys = ("recall", "precision", "f1", "mean_lead_days")
    boot: Dict[str, Dict[str, List[float]]] = {
        name: {k: [] for k in metric_keys} for name in detector_names
    }
    deltas: Dict[str, Dict[str, List[float]]] = {
        name: {f"delta_{k}": [] for k in ("recall", "f1", "lead_days")}
        for name in detector_names
        if name != baseline_name
    }

    for _ in range(int(n_boot)):
        idx = rng.integers(0, n_blocks, size=n_blocks)
        sample = [site_year_blocks[int(i)] for i in idx]
        scores = _pool_blocks_and_score(
            sample, detector_names, horizon_days=horizon_days
        )
        base = scores[baseline_name]
        for name in detector_names:
            s = scores[name]
            for k in metric_keys:
                val = float(s[k]) if np.isfinite(s[k]) else np.nan
                boot[name][k].append(val)
            if name == baseline_name:
                continue
            deltas[name]["delta_recall"].append(
                float(s["recall"]) - float(base["recall"])
            )
            deltas[name]["delta_f1"].append(float(s["f1"]) - float(base["f1"]))
            lead_d = s["mean_lead_days"]
            lead_b = base["mean_lead_days"]
            if np.isfinite(lead_d) and np.isfinite(lead_b):
                deltas[name]["delta_lead_days"].append(float(lead_d) - float(lead_b))
            else:
                deltas[name]["delta_lead_days"].append(np.nan)

    def _ci(values: Sequence[float]) -> tuple[float, float, float]:
        arr = np.asarray(values, dtype=np.float64)
        finite = arr[np.isfinite(arr)]
        if finite.size == 0:
            return float("nan"), float("nan"), float("nan")
        lo, hi = np.percentile(finite, [2.5, 97.5])
        return float(np.mean(finite)), float(lo), float(hi)

    point = _pool_blocks_and_score(
        site_year_blocks, detector_names, horizon_days=horizon_days
    )

    ci_metrics: Dict[str, dict] = {}
    for name in detector_names:
        ci_metrics[name] = {}
        for k in metric_keys:
            mean, lo, hi = _ci(boot[name][k])
            ci_metrics[name][k] = {
                "point": float(point[name][k]),
                "boot_mean": mean,
                "ci95_low": lo,
                "ci95_high": hi,
            }

    ci_deltas: Dict[str, dict] = {}
    for name, dmap in deltas.items():
        ci_deltas[name] = {}
        for k, vals in dmap.items():
            mean, lo, hi = _ci(vals)
            # Point delta from full-sample pooled scores
            if k == "delta_lead_days":
                p_d = point[name]["mean_lead_days"]
                p_b = point[baseline_name]["mean_lead_days"]
                point_delta = (
                    float(p_d) - float(p_b)
                    if np.isfinite(p_d) and np.isfinite(p_b)
                    else float("nan")
                )
            elif k == "delta_recall":
                point_delta = float(point[name]["recall"]) - float(
                    point[baseline_name]["recall"]
                )
            else:  # delta_f1
                point_delta = float(point[name]["f1"]) - float(
                    point[baseline_name]["f1"]
                )
            ci_deltas[name][k] = {
                "point": point_delta,
                "boot_mean": mean,
                "ci95_low": lo,
                "ci95_high": hi,
            }

    ci_report_md = _format_bootstrap_ci_md(
        detector_names=detector_names,
        baseline_name=baseline_name,
        ci_metrics=ci_metrics,
        ci_deltas=ci_deltas,
        n_boot=n_boot,
        n_blocks=n_blocks,
        seed=seed,
    )

    return {
        "n_boot": int(n_boot),
        "n_blocks": int(n_blocks),
        "seed": int(seed),
        "baseline_name": baseline_name,
        "point_scores": point,
        "ci_metrics": ci_metrics,
        "ci_deltas": ci_deltas,
        "ci_report_md": ci_report_md,
    }


def _format_bootstrap_ci_md(
    detector_names: Sequence[str],
    baseline_name: str,
    ci_metrics: dict,
    ci_deltas: dict,
    n_boot: int,
    n_blocks: int,
    seed: int,
) -> str:
    """Markdown tables for 95% bootstrap CIs and paired deltas vs baseline."""

    def _fmt(v: float, digits: int = 3) -> str:
        return f"{v:.{digits}f}" if np.isfinite(v) else "n/a"

    def _band(entry: dict, digits: int = 3) -> str:
        return (
            f"{_fmt(entry['point'], digits)} "
            f"[{_fmt(entry['ci95_low'], digits)}, {_fmt(entry['ci95_high'], digits)}]"
        )

    lines = [
        "## 95% Site-Year Block Bootstrap CI & Paired Delta Table",
        "",
        f"- **Blocks:** {n_blocks} site-years",
        f"- **Bootstrap replicates:** {n_boot} (seed={seed})",
        f"- **Baseline:** `{baseline_name}`",
        "",
        "### Detector metrics (point [95% CI])",
        "",
        "| Detector | Recall | Precision | F1 | Mean Lead (d) |",
        "|---|---|---|---|---|",
    ]
    for name in detector_names:
        m = ci_metrics[name]
        lines.append(
            f"| `{name}` | {_band(m['recall'])} | {_band(m['precision'])} | "
            f"{_band(m['f1'])} | {_band(m['mean_lead_days'], 2)} |"
        )

    lines.extend(
        [
            "",
            f"### Paired deltas vs `{baseline_name}` (point [95% CI])",
            "",
            "| Detector | Δ Recall | Δ F1 | Δ Mean Lead (d) |",
            "|---|---|---|---|",
        ]
    )
    for name in detector_names:
        if name == baseline_name:
            continue
        d = ci_deltas[name]
        lines.append(
            f"| `{name}` | {_band(d['delta_recall'])} | {_band(d['delta_f1'])} | "
            f"{_band(d['delta_lead_days'], 2)} |"
        )
    lines.append("")
    return "\n".join(lines)


def _format_bio_report_md(
    event_onsets: Sequence[float],
    event_dates: Sequence[str],
    scores: Dict[str, dict],
    horizon_days: int,
) -> str:
    """Build Markdown summary for the BIO_05 flash-drought validation."""
    lines = [
        "# BIO_05 Flash Drought Validation Report",
        "",
        f"- **Flash drought events:** {len(event_onsets)}",
        f"- **Pre-onset horizon:** {horizon_days} days",
        f"- **Event dates:** {', '.join(event_dates) if event_dates else 'none'}",
        "",
        "| Detector | Recall | Precision | F1 | Mean Lead (d) | Alert Rate (%) | False Alert Days | Lead/100 FA |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    order = [
        "legacy_30d_soil_z",
        "bio05_thinning_or_veto",
        "bio05_veto_only",
        "bio05_drying_thinning_or_veto",
        "bio05_drying_veto_only",
        "bio05_twostage_confirm",
    ]
    for name in order:
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
