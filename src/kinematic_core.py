"""Covariant Sentinel Auditor: scale-invariant kinematic trajectory auditor."""

from __future__ import annotations

from collections import deque
from typing import Deque, List, Optional

import numpy as np
import pandas as pd


class CovariantSentinelAuditor:
    """Streaming kinematic auditor with gap-aware masking and persistence gasket.

    Implements causal expanding Z-score whitening, tangential/transverse
    acceleration decomposition, fatigue-buffered omega scoring, session
    re-anchoring, and a tactical persistence gasket for phase escalation.
    """

    def __init__(
        self,
        omega_0: float = 0.05,
        alpha: float = 0.12,
        beta: float = 0.08,
        gamma: float = 0.02,
        phase_thinning: float = 0.07,
        phase_veto: float = 0.19,
        delta_min: float = 0.01,
        delta_max: float = 5.0,
        kappa: float = 10.0,
        history_window: int = 30,
        calibration_window: int = 30,
        persistence_steps: int = 3,
        max_gap_seconds: float = 7200.0,
    ) -> None:
        self.omega_0 = float(omega_0)
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.gamma = float(gamma)
        self.phase_thinning = float(phase_thinning)
        self.phase_veto = float(phase_veto)
        self.delta_min = float(delta_min)
        self.delta_max = float(delta_max)
        self.kappa = float(kappa)
        self.eps = self.kappa * np.finfo(np.float64).eps
        self.history_window = int(history_window)
        self.calibration_window = int(calibration_window)
        self.persistence_steps = int(persistence_steps)
        self.max_gap_seconds = float(max_gap_seconds)
        self._whitening_floor = 1e-8

    def process_stream(
        self, timestamps: np.ndarray, raw_matrix: np.ndarray
    ) -> pd.DataFrame:
        """Process a multivariate time series and return kinematic audit rows.

        Parameters
        ----------
        timestamps:
            1-D array of length N (seconds or epoch floats).
        raw_matrix:
            Array of shape (N, D); cast internally to float64.

        Returns
        -------
        pd.DataFrame
            Columns: timestamp, gap_masked, s_t, a_perp_mag, rolling_fc,
            omega_raw, omega_cal, phase.
        """
        timestamps = np.asarray(timestamps, dtype=np.float64).reshape(-1)
        raw = np.asarray(raw_matrix, dtype=np.float64)
        if raw.ndim != 2:
            raise ValueError("raw_matrix must have shape (N, D)")
        n, d = raw.shape
        if timestamps.shape[0] != n:
            raise ValueError("timestamps length must match raw_matrix rows")

        # Causal expanding whitening state (Welford); reset on gaps.
        w_count = 0
        w_mean = np.zeros(d, dtype=np.float64)
        w_m2 = np.zeros(d, dtype=np.float64)

        prev_theta: Optional[np.ndarray] = None
        prev_velocity = np.zeros(d, dtype=np.float64)
        prev_delta_tau: Optional[float] = None
        have_anchor = False

        fatigue_buf: Deque[float] = deque(maxlen=self.history_window)
        omega_cal_samples: List[float] = []
        delta_omega_base: Optional[float] = None

        thinning_streak = 0
        veto_streak = 0

        rows: List[dict] = []

        for t in range(n):
            x_t = raw[t].astype(np.float64, copy=False)

            gap_masked = False
            if t > 0:
                dt = float(timestamps[t] - timestamps[t - 1])
                if dt > self.max_gap_seconds:
                    # Gap-Aware Masking Gate: pause derivatives, reset velocity,
                    # and resume cleanly (whitening + fatigue re-seeded).
                    gap_masked = True
                    prev_velocity = np.zeros(d, dtype=np.float64)
                    prev_delta_tau = None
                    have_anchor = False
                    thinning_streak = 0
                    veto_streak = 0
                    fatigue_buf.clear()
                    w_count = 0
                    w_mean = np.zeros(d, dtype=np.float64)
                    w_m2 = np.zeros(d, dtype=np.float64)

            # --- Dimensional Anisotropy Fix: causal expanding Z-score ---
            w_count += 1
            if w_count == 1:
                w_mean = x_t.copy()
                w_m2 = np.zeros(d, dtype=np.float64)
            else:
                delta = x_t - w_mean
                w_mean = w_mean + delta / w_count
                w_m2 = w_m2 + delta * (x_t - w_mean)

            if w_count < 2:
                w_std = np.full(d, self._whitening_floor, dtype=np.float64)
            else:
                w_std = np.sqrt(np.maximum(w_m2 / w_count, 0.0))
                w_std = np.maximum(w_std, self._whitening_floor)

            theta_t = (x_t - w_mean) / w_std

            s_t = np.nan
            a_perp_mag = np.nan
            rolling_fc = np.nan
            omega_raw = np.nan
            omega_cal = np.nan
            phase = "NOMINAL"

            if gap_masked:
                # Current sample becomes the post-gap anchor; no derivatives.
                prev_theta = theta_t.copy()
                have_anchor = True
            elif not have_anchor:
                prev_theta = theta_t.copy()
                prev_velocity = np.zeros(d, dtype=np.float64)
                prev_delta_tau = None
                have_anchor = True
            else:
                assert prev_theta is not None
                delta_theta = theta_t - prev_theta
                s_t = float(np.linalg.norm(delta_theta))
                delta_tau_t = float(np.clip(s_t, self.delta_min, self.delta_max))

                v_t = delta_theta / delta_tau_t

                if prev_delta_tau is None:
                    denom = delta_tau_t + delta_tau_t
                else:
                    denom = delta_tau_t + prev_delta_tau
                a_t = 2.0 * (v_t - prev_velocity) / denom

                v_dot_v = float(np.dot(v_t, v_t))
                a_parallel = (np.dot(a_t, v_t) / (v_dot_v + self.eps)) * v_t
                a_perp = a_t - a_parallel
                a_perp_mag = float(np.linalg.norm(a_perp))

                fatigue_buf.append(a_perp_mag)
                rolling_fc = float(np.mean(fatigue_buf))

                omega_raw = (
                    self.omega_0
                    + self.alpha * s_t
                    + self.beta * (a_perp_mag ** 2)
                    + self.gamma * rolling_fc
                )

                if delta_omega_base is None:
                    omega_cal_samples.append(omega_raw)
                    if len(omega_cal_samples) >= self.calibration_window:
                        med = float(np.median(omega_cal_samples))
                        delta_omega_base = med - self.omega_0
                    # Hold NOMINAL while the session baseline is still forming.
                    omega_cal = omega_raw
                    phase = "NOMINAL"
                    thinning_streak = 0
                    veto_streak = 0
                else:
                    omega_cal = omega_raw - delta_omega_base
                    if omega_cal >= self.phase_veto:
                        veto_streak += 1
                        thinning_streak += 1
                    elif omega_cal >= self.phase_thinning:
                        veto_streak = 0
                        thinning_streak += 1
                    else:
                        veto_streak = 0
                        thinning_streak = 0

                    if veto_streak >= self.persistence_steps:
                        phase = "VETO"
                    elif thinning_streak >= self.persistence_steps:
                        phase = "THINNING"
                    else:
                        phase = "NOMINAL"

                prev_theta = theta_t.copy()
                prev_velocity = v_t.copy()
                prev_delta_tau = delta_tau_t

            rows.append(
                {
                    "timestamp": float(timestamps[t]),
                    "gap_masked": bool(gap_masked),
                    "s_t": s_t,
                    "a_perp_mag": a_perp_mag,
                    "rolling_fc": rolling_fc,
                    "omega_raw": omega_raw,
                    "omega_cal": omega_cal,
                    "phase": phase,
                }
            )

        df = pd.DataFrame(rows)

        # Back-fill omega_cal once the session baseline is known; re-apply gasket
        # only on post-calibration valid steps (calibration window stays NOMINAL).
        if delta_omega_base is not None:
            valid = df["omega_raw"].notna()
            df.loc[valid, "omega_cal"] = (
                df.loc[valid, "omega_raw"].to_numpy(dtype=np.float64) - delta_omega_base
            )
            df = self._reapply_persistence(df)

        return df

    def _reapply_persistence(self, df: pd.DataFrame) -> pd.DataFrame:
        """Recompute phase with persistence; calibration window stays NOMINAL."""
        thinning_streak = 0
        veto_streak = 0
        phases: List[str] = []
        valid_seen = 0

        for _, row in df.iterrows():
            if row["gap_masked"] or not np.isfinite(row["omega_cal"]):
                thinning_streak = 0
                veto_streak = 0
                phases.append("NOMINAL")
                continue

            valid_seen += 1
            if valid_seen <= self.calibration_window:
                thinning_streak = 0
                veto_streak = 0
                phases.append("NOMINAL")
                continue

            omega_cal = float(row["omega_cal"])
            if omega_cal >= self.phase_veto:
                veto_streak += 1
                thinning_streak += 1
            elif omega_cal >= self.phase_thinning:
                veto_streak = 0
                thinning_streak += 1
            else:
                veto_streak = 0
                thinning_streak = 0

            if veto_streak >= self.persistence_steps:
                phases.append("VETO")
            elif thinning_streak >= self.persistence_steps:
                phases.append("THINNING")
            else:
                phases.append("NOMINAL")

        df = df.copy()
        df["phase"] = phases
        return df
