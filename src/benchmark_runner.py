"""Benchmark & null-hypothesis validation for HELIOS→ATMOS coupling."""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from .coupling_engine import HeliosAtmosCoupler
from .kinematic_core import CovariantSentinelAuditor

HOUR = 3600.0
DAY = 86400.0


def remove_diurnal_cycle(
    atmos_raw: pd.DataFrame,
    feature_cols: Sequence[str] | None = None,
    period_steps: int = 24,
) -> pd.DataFrame:
    """Strictly causal 24-hour differencing to remove predictable diurnal cycles.

    Computes ``x[t] - x[t-period_steps]`` for each feature and drops the first
    ``period_steps`` rows (insufficient history). Genuine synoptic shifts remain.
    """
    if feature_cols is None:
        feature_cols = ["pressure_msl", "temperature_2m", "wind_speed_100m"]

    df = atmos_raw.sort_values("timestamp", kind="mergesort").reset_index(drop=True)
    out = df.copy()
    for col in feature_cols:
        if col not in out.columns:
            raise ValueError(f"atmos frame missing column '{col}'")
        out[col] = out[col].astype(np.float64) - out[col].astype(np.float64).shift(
            int(period_steps)
        )
    return out.iloc[int(period_steps) :].reset_index(drop=True)


def detect_atmos_events(
    atmos_raw: pd.DataFrame,
    window_hours: int = 336,
    z_thresh: float = 2.0,
    min_duration_hours: int = 12,
    merge_gap_hours: int = 48,
) -> list[float]:
    """Detect atmospheric anomaly event onsets from pressure/temperature.

    Uses a strictly causal trailing Z-score on ``pressure_msl`` and
    ``temperature_2m``. Contiguous anomalous runs meeting the duration
    threshold are merged across short gaps; onset timestamps are returned.
    """
    required = {"timestamp", "pressure_msl", "temperature_2m"}
    if not required.issubset(atmos_raw.columns):
        raise ValueError(f"atmos_raw must contain {sorted(required)}")

    df = atmos_raw.sort_values("timestamp", kind="mergesort").reset_index(drop=True)
    ts = df["timestamp"].to_numpy(dtype=np.float64)

    z_p = _causal_rolling_z(df["pressure_msl"], window_hours)
    z_t = _causal_rolling_z(df["temperature_2m"], window_hours)
    anom = (np.maximum(np.abs(z_p), np.abs(z_t)) >= float(z_thresh)).astype(bool)

    return _onsets_from_mask(
        ts, anom, min_duration_hours=min_duration_hours, merge_gap_hours=merge_gap_hours
    )


def detect_persistent_blocks(
    atmos_raw: pd.DataFrame,
    window_hours: int = 336,
    z_press_thresh: float = 1.5,
    min_duration_hours: int = 48,
    merge_gap_hours: int = 72,
) -> list[float]:
    """Detect multi-day high-pressure blocking ridge onsets.

    Flags hours where the causal pressure Z-score is strictly positive and
    at least ``z_press_thresh``, requiring contiguous runs of
    ``min_duration_hours`` (default 48h+) so routine low-pressure storms are
    excluded.
    """
    required = {"timestamp", "pressure_msl"}
    if not required.issubset(atmos_raw.columns):
        raise ValueError(f"atmos_raw must contain {sorted(required)}")

    df = atmos_raw.sort_values("timestamp", kind="mergesort").reset_index(drop=True)
    ts = df["timestamp"].to_numpy(dtype=np.float64)
    z_p = _causal_rolling_z(df["pressure_msl"], window_hours)
    anom = (z_p >= float(z_press_thresh)).astype(bool)

    return _onsets_from_mask(
        ts,
        anom,
        min_duration_hours=min_duration_hours,
        merge_gap_hours=merge_gap_hours,
    )


def _onsets_from_mask(
    ts: np.ndarray,
    anom: np.ndarray,
    min_duration_hours: float,
    merge_gap_hours: float,
) -> list[float]:
    """Convert a boolean anomaly mask into merged event onset timestamps."""
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
        duration_h = (ts[b] - ts[a]) / HOUR + 1.0
        if duration_h >= float(min_duration_hours):
            kept.append((a, b))

    if not kept:
        return []

    merged: List[tuple[int, int]] = [kept[0]]
    for a, b in kept[1:]:
        prev_a, prev_b = merged[-1]
        gap_h = (ts[a] - ts[prev_b]) / HOUR
        if gap_h < float(merge_gap_hours):
            merged[-1] = (prev_a, b)
        else:
            merged.append((a, b))

    return sorted(float(ts[a]) for a, _ in merged)


def _causal_rolling_z(series: pd.Series, window: int, min_periods: int = 48) -> np.ndarray:
    """Strictly causal trailing Z-score (no look-ahead)."""
    roll = series.astype(np.float64).rolling(
        window=int(window), min_periods=int(min_periods)
    )
    mean = roll.mean()
    std = roll.std(ddof=0)
    z = (series.astype(np.float64) - mean) / std.replace(0.0, np.nan)
    return z.to_numpy(dtype=np.float64)


def _count_alert_episodes(
    timestamps: np.ndarray,
    alert_mask: np.ndarray,
    merge_gap_hours: float = 24.0,
) -> int:
    """Count distinct alert episodes, merging runs separated by < merge_gap_hours."""
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
        gap_h = (a - prev_b) / HOUR
        if gap_h < float(merge_gap_hours):
            merged[-1] = (prev_a, b)
        else:
            merged.append((a, b))
    return len(merged)


def _f1_score(precision: float, recall: float) -> float:
    denom = float(precision) + float(recall)
    if denom <= 0.0:
        return 0.0
    return float(2.0 * precision * recall / denom)


def score_detector(
    timestamps: np.ndarray,
    alert_mask: np.ndarray,
    event_onsets: list[float],
    horizon_seconds: float = 259200.0,
) -> dict:
    """Score a boolean alert stream against event onsets under a pre-onset horizon."""
    timestamps = np.asarray(timestamps, dtype=np.float64)
    alert_mask = np.asarray(alert_mask, dtype=bool)
    if timestamps.shape[0] != alert_mask.shape[0]:
        raise ValueError("timestamps and alert_mask must have the same length")

    events_total = len(event_onsets)
    lead_hours: List[float] = []
    events_detected = 0

    for t_e in event_onsets:
        window = (timestamps >= (t_e - horizon_seconds)) & (timestamps <= t_e)
        hits = alert_mask & window
        if np.any(hits):
            events_detected += 1
            first_alert_ts = float(timestamps[hits].min())
            lead_hours.append((t_e - first_alert_ts) / HOUR)

    recall = (
        float(events_detected) / float(events_total) if events_total > 0 else 0.0
    )
    mean_lead = float(np.mean(lead_hours)) if lead_hours else float("nan")
    n = max(len(timestamps), 1)
    alert_rate_pct = 100.0 * float(np.sum(alert_mask)) / float(n)

    post_buffer = 43200.0
    in_event_window = np.zeros(len(timestamps), dtype=bool)
    for t_e in event_onsets:
        in_event_window |= (timestamps >= (t_e - horizon_seconds)) & (
            timestamps <= (t_e + post_buffer)
        )
    false_alert_hours = int(np.sum(alert_mask & ~in_event_window))

    n_episodes = _count_alert_episodes(timestamps, alert_mask, merge_gap_hours=24.0)
    precision = (
        float(events_detected) / float(n_episodes) if n_episodes > 0 else 0.0
    )

    return {
        "events_total": int(events_total),
        "events_detected": int(events_detected),
        "recall": float(recall),
        "precision": float(precision),
        "f1": _f1_score(precision, recall),
        "alert_episodes": int(n_episodes),
        "mean_lead_hours": mean_lead,
        "alert_rate_pct": float(alert_rate_pct),
        "false_alert_hours": int(false_alert_hours),
    }


def _even_circular_day_shifts(total_days: int, n_shifts: int = 50) -> np.ndarray:
    """Evenly spaced day-shifts in [10, total_days-10], excluding near-zero overlap."""
    total_days = int(total_days)
    n_shifts = int(n_shifts)
    lo, hi = 10, total_days - 10
    if total_days < 25 or hi <= lo:
        # Short series fallback: any |d| >= 10 within the circle
        candidates = [
            d
            for d in range(-(total_days - 1), total_days)
            if abs(d) >= 10 and abs(d) <= total_days - 10
        ]
        if not candidates:
            candidates = list(range(1, max(total_days, 2)))
        idx = np.linspace(0, len(candidates) - 1, num=min(n_shifts, len(candidates)))
        return np.unique(np.array(candidates, dtype=int)[np.rint(idx).astype(int)])

    raw = np.linspace(lo, hi, num=n_shifts)
    shifts = np.unique(np.rint(raw).astype(int))
    # Top up if rounding collapsed duplicates
    if len(shifts) < n_shifts:
        pool = np.arange(lo, hi + 1, dtype=int)
        extra = [d for d in pool if d not in set(shifts)]
        need = n_shifts - len(shifts)
        shifts = np.unique(
            np.concatenate([shifts, np.array(extra[:need], dtype=int)])
        )
    return shifts[:n_shifts]


def run_permutation_test(
    helios_aud: pd.DataFrame,
    atmos_aud: pd.DataFrame,
    event_onsets: list[float],
    n_shifts: int = 50,
    horizon_seconds: float = 259200.0,
) -> dict:
    """50-shift circular permutation test for coupled HELIOS→ATMOS skill."""
    coupler = HeliosAtmosCoupler()
    aligned = coupler.couple(helios_aud, atmos_aud)
    aligned_alert = aligned["coupled_phase"].isin(["THINNING", "VETO"]).to_numpy()
    aligned_score = score_detector(
        aligned["timestamp"].to_numpy(dtype=np.float64),
        aligned_alert,
        event_onsets,
        horizon_seconds=horizon_seconds,
    )
    aligned_p = float(aligned_score["precision"])
    aligned_r = float(aligned_score["recall"])
    aligned_f1 = float(aligned_score["f1"])
    aligned_lead = float(aligned_score["mean_lead_hours"])

    ts = helios_aud["timestamp"].to_numpy(dtype=np.float64)
    span_days = max(int(np.floor((float(ts.max()) - float(ts.min())) / DAY)) + 1, 1)
    # Also respect sample count for hourly streams
    span_days = max(span_days, max(len(helios_aud) // 24, 1))
    shift_days = _even_circular_day_shifts(span_days, n_shifts=n_shifts)

    omega = helios_aud["omega_cal"].to_numpy(dtype=np.float64)
    null_precision: List[float] = []
    null_recall: List[float] = []
    null_f1: List[float] = []
    null_lead: List[float] = []

    for d in shift_days:
        shifted = helios_aud.copy()
        shifted["omega_cal"] = np.roll(omega, int(d) * 24)
        null_coupled = coupler.couple(shifted, atmos_aud)
        null_alert = null_coupled["coupled_phase"].isin(["THINNING", "VETO"]).to_numpy()
        ns = score_detector(
            null_coupled["timestamp"].to_numpy(dtype=np.float64),
            null_alert,
            event_onsets,
            horizon_seconds=horizon_seconds,
        )
        null_precision.append(float(ns["precision"]))
        null_recall.append(float(ns["recall"]))
        null_f1.append(float(ns["f1"]))
        null_lead.append(float(ns["mean_lead_hours"]))

    n_eff = len(shift_days)
    p_precision = (sum(1 for x in null_precision if x >= aligned_p) + 1) / (
        n_eff + 1
    )
    p_f1 = (sum(1 for x in null_f1 if x >= aligned_f1) + 1) / (n_eff + 1)
    if np.isfinite(aligned_lead):
        p_lead = (
            sum(
                1
                for x in null_lead
                if np.isfinite(x) and x >= aligned_lead
            )
            + 1
        ) / (n_eff + 1)
    else:
        p_lead = 1.0

    report_md = _format_permutation_md(
        aligned_score=aligned_score,
        null_precision=null_precision,
        null_recall=null_recall,
        null_f1=null_f1,
        null_lead=null_lead,
        p_precision=p_precision,
        p_f1=p_f1,
        p_lead=p_lead,
        n_shifts=n_eff,
        shift_days=shift_days.tolist(),
        n_events=len(event_onsets),
    )

    return {
        "aligned": aligned_score,
        "aligned_precision": aligned_p,
        "aligned_recall": aligned_r,
        "aligned_f1": aligned_f1,
        "aligned_mean_lead_hours": aligned_lead,
        "null_precision": null_precision,
        "null_recall": null_recall,
        "null_f1": null_f1,
        "null_mean_lead_hours": null_lead,
        "null_mean_precision": float(np.mean(null_precision)) if null_precision else float("nan"),
        "null_mean_recall": float(np.mean(null_recall)) if null_recall else float("nan"),
        "null_mean_f1": float(np.mean(null_f1)) if null_f1 else float("nan"),
        "null_mean_lead": (
            float(np.nanmean([x for x in null_lead if np.isfinite(x)]))
            if any(np.isfinite(x) for x in null_lead)
            else float("nan")
        ),
        "p_precision": float(p_precision),
        "p_f1": float(p_f1),
        "p_lead": float(p_lead),
        "n_shifts": int(n_eff),
        "shift_days": [int(x) for x in shift_days],
        "report_md": report_md,
    }


def _format_permutation_md(
    aligned_score: dict,
    null_precision: Sequence[float],
    null_recall: Sequence[float],
    null_f1: Sequence[float],
    null_lead: Sequence[float],
    p_precision: float,
    p_f1: float,
    p_lead: float,
    n_shifts: int,
    shift_days: Sequence[int],
    n_events: int,
) -> str:
    """Markdown section summarizing the circular permutation test."""
    lead_a = aligned_score["mean_lead_hours"]
    lead_a_s = f"{lead_a:.2f}" if np.isfinite(lead_a) else "n/a"
    null_lead_mean = (
        float(np.nanmean([x for x in null_lead if np.isfinite(x)]))
        if any(np.isfinite(x) for x in null_lead)
        else float("nan")
    )
    null_lead_s = f"{null_lead_mean:.2f}" if np.isfinite(null_lead_mean) else "n/a"

    lines = [
        "## Persistent-Block Permutation Test (Circular Shift)",
        "",
        f"- **Block events:** {n_events}",
        f"- **Null shifts (n={n_shifts}):** {list(shift_days)[:8]}{' ...' if n_shifts > 8 else ''}",
        "",
        "| Metric | Aligned | Null Mean | p-value |",
        "|---|---:|---:|---:|",
        f"| Precision | {aligned_score['precision']:.3f} | "
        f"{float(np.mean(null_precision)):.3f} | {p_precision:.4f} |",
        f"| Recall | {aligned_score['recall']:.3f} | "
        f"{float(np.mean(null_recall)):.3f} | — |",
        f"| F1 | {aligned_score['f1']:.3f} | "
        f"{float(np.mean(null_f1)):.3f} | {p_f1:.4f} |",
        f"| Mean Lead (h) | {lead_a_s} | {null_lead_s} | {p_lead:.4f} |",
        "",
    ]
    return "\n".join(lines)


def run_validation_experiment(
    helios_raw: pd.DataFrame,
    atmos_raw: pd.DataFrame,
    shift_days: list[int] | None = None,
    mode: str = "all",
) -> dict:
    """Run aligned + null-shifted HELIOS→ATMOS validation experiment.

    Parameters
    ----------
    mode:
        ``"all"`` uses ``detect_atmos_events``.
        ``"blocks"`` uses ``detect_persistent_blocks`` and appends a 50-shift
        permutation p-value table.
    """
    if shift_days is None:
        shift_days = [7, 14, 21, -7, -14]
    if mode not in {"all", "blocks"}:
        raise ValueError("mode must be 'all' or 'blocks'")

    helios_cols = ["bz_gsm", "v_sw", "p_dyn"]
    atmos_cols = ["pressure_msl", "temperature_2m", "wind_speed_100m"]
    for col in helios_cols:
        if col not in helios_raw.columns:
            raise ValueError(f"helios_raw missing column '{col}'")
    for col in atmos_cols:
        if col not in atmos_raw.columns:
            raise ValueError(f"atmos_raw missing column '{col}'")
    if "timestamp" not in helios_raw.columns or "timestamp" not in atmos_raw.columns:
        raise ValueError("both raw frames require a timestamp column")

    helios = helios_raw.sort_values("timestamp", kind="mergesort").reset_index(drop=True)
    atmos = atmos_raw.sort_values("timestamp", kind="mergesort").reset_index(drop=True)

    if mode == "blocks":
        event_onsets = detect_persistent_blocks(atmos)
        event_label = "persistent high-pressure blocks"
    else:
        event_onsets = detect_atmos_events(atmos)
        event_label = "atmospheric anomaly events"

    atmos_for_audit = remove_diurnal_cycle(atmos, feature_cols=atmos_cols)

    helios_auditor = CovariantSentinelAuditor(
        persistence_steps=3, max_gap_seconds=2.0 * HOUR
    )
    atmos_auditor = CovariantSentinelAuditor(
        persistence_steps=6, max_gap_seconds=2.0 * HOUR
    )
    helios_aud = helios_auditor.process_stream(
        helios["timestamp"].to_numpy(dtype=np.float64),
        helios[helios_cols].to_numpy(dtype=np.float64),
    )
    atmos_aud = atmos_auditor.process_stream(
        atmos_for_audit["timestamp"].to_numpy(dtype=np.float64),
        atmos_for_audit[atmos_cols].to_numpy(dtype=np.float64),
    )

    coupler = HeliosAtmosCoupler()
    coupled_df = coupler.couple(helios_aud, atmos_aud)

    scores: Dict[str, dict] = {}

    z_p = _causal_rolling_z(atmos["pressure_msl"], window=336)
    z_t = _causal_rolling_z(atmos["temperature_2m"], window=336)
    legacy_alert = np.maximum(np.abs(z_p), np.abs(z_t)) >= 2.0
    legacy_alert = np.nan_to_num(legacy_alert.astype(np.float64), nan=0.0).astype(bool)
    scores["legacy_trailing_z"] = score_detector(
        atmos["timestamp"].to_numpy(dtype=np.float64),
        legacy_alert,
        event_onsets,
    )

    uncoupled_alert = atmos_aud["phase"].isin(["THINNING", "VETO"]).to_numpy()
    scores["uncoupled_atmos"] = score_detector(
        atmos_aud["timestamp"].to_numpy(dtype=np.float64),
        uncoupled_alert,
        event_onsets,
    )

    coupled_alert = coupled_df["coupled_phase"].isin(["THINNING", "VETO"]).to_numpy()
    scores["coupled_helios_atmos"] = score_detector(
        coupled_df["timestamp"].to_numpy(dtype=np.float64),
        coupled_alert,
        event_onsets,
    )

    veto_alert = (coupled_df["coupled_phase"] == "VETO").to_numpy()
    scores["coupled_veto_only"] = score_detector(
        coupled_df["timestamp"].to_numpy(dtype=np.float64),
        veto_alert,
        event_onsets,
    )

    null_scores: List[dict] = []
    omega = helios_aud["omega_cal"].to_numpy(dtype=np.float64)
    for d in shift_days:
        shifted = helios_aud.copy()
        shifted["omega_cal"] = np.roll(omega, int(d) * 24)
        null_coupled = coupler.couple(shifted, atmos_aud)
        null_alert = null_coupled["coupled_phase"].isin(["THINNING", "VETO"]).to_numpy()
        null_scores.append(
            score_detector(
                null_coupled["timestamp"].to_numpy(dtype=np.float64),
                null_alert,
                event_onsets,
            )
        )

    null_control_summary = {
        "n_shifts": len(null_scores),
        "shift_days": list(shift_days),
        "mean_recall": float(np.mean([s["recall"] for s in null_scores]))
        if null_scores
        else float("nan"),
        "mean_precision": float(np.mean([s["precision"] for s in null_scores]))
        if null_scores
        else float("nan"),
        "mean_lead_hours": (
            float(
                np.nanmean(
                    [
                        s["mean_lead_hours"]
                        for s in null_scores
                        if np.isfinite(s["mean_lead_hours"])
                    ]
                )
            )
            if null_scores
            and any(np.isfinite(s["mean_lead_hours"]) for s in null_scores)
            else float("nan")
        ),
        "mean_false_alert_hours": float(
            np.mean([s["false_alert_hours"] for s in null_scores])
        )
        if null_scores
        else float("nan"),
        "per_shift": null_scores,
    }

    report_md = _format_report_md(
        event_onsets=event_onsets,
        scores=scores,
        null_control_summary=null_control_summary,
        event_label=event_label,
        mode=mode,
    )

    permutation = None
    if mode == "blocks":
        permutation = run_permutation_test(
            helios_aud, atmos_aud, event_onsets, n_shifts=50
        )
        report_md = report_md.rstrip() + "\n\n" + permutation["report_md"]

    return {
        "mode": mode,
        "event_onsets": event_onsets,
        "scores": scores,
        "null_control_summary": null_control_summary,
        "coupled_df": coupled_df,
        "permutation": permutation,
        "report_md": report_md,
    }


def _format_report_md(
    event_onsets: Sequence[float],
    scores: Dict[str, dict],
    null_control_summary: dict,
    event_label: str = "atmospheric anomaly events",
    mode: str = "all",
) -> str:
    """Build a clean Markdown summary table for the validation experiment."""
    lines = [
        "# HELIOS→ATMOS Validation Report",
        "",
        f"- **Mode:** `{mode}`",
        f"- **Events detected (ground truth {event_label}):** {len(event_onsets)}",
        f"- **Null shifts:** {null_control_summary.get('shift_days', [])}",
        "",
        "| Detector | Recall | Precision | F1 | Mean Lead (h) | Alert Rate (%) | False Alert Hours |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]

    order = [
        "legacy_trailing_z",
        "uncoupled_atmos",
        "coupled_helios_atmos",
        "coupled_veto_only",
    ]
    for name in order:
        s = scores[name]
        lead = s["mean_lead_hours"]
        lead_s = f"{lead:.2f}" if np.isfinite(lead) else "n/a"
        lines.append(
            f"| `{name}` | {s['recall']:.3f} | {s['precision']:.3f} | {s['f1']:.3f} | "
            f"{lead_s} | {s['alert_rate_pct']:.2f} | {s['false_alert_hours']} |"
        )

    ns = null_control_summary
    lead_n = ns.get("mean_lead_hours", float("nan"))
    lead_n_s = f"{lead_n:.2f}" if np.isfinite(lead_n) else "n/a"
    lines.extend(
        [
            "",
            "## Null-Hypothesis Control (Circular Shift)",
            "",
            f"- **Mean recall:** {ns.get('mean_recall', float('nan')):.3f}",
            f"- **Mean precision:** {ns.get('mean_precision', float('nan')):.3f}",
            f"- **Mean lead hours:** {lead_n_s}",
            f"- **Mean false alert hours:** "
            f"{ns.get('mean_false_alert_hours', float('nan')):.1f}",
            "",
        ]
    )
    return "\n".join(lines)
