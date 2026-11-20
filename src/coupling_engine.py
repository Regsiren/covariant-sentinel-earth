"""HELIOS_01 -> ATMOS_02 cross-domain coupling via entropy transfer."""

from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd


class HeliosAtmosCoupler:
    """Causal HELIOS→ATMOS coupler using an Entropy Transfer Transition Matrix.

    For each atmospheric timestamp, a strictly trailing exponentially weighted
    lag kernel aggregates excess heliospheric uncertainty, builds a dynamic
    coupling coefficient ``chi_t``, an escalation probability ``p_23``, and a
    coupled uncertainty ``omega_coupled``. A persistence gasket assigns
    ``coupled_phase``.
    """

    def __init__(
        self,
        lag_window_seconds: float = 259200.0,
        lag_half_life_seconds: float = 86400.0,
        chi_0: float = 0.08,
        gamma_fluid: float = 0.25,
        zeta: float = 0.20,
        omega_0: float = 0.05,
        phase_thinning: float = 0.07,
        phase_veto: float = 0.19,
        persistence_steps: int = 6,
    ) -> None:
        self.lag_window_seconds = float(lag_window_seconds)
        self.lag_half_life_seconds = float(lag_half_life_seconds)
        self.chi_0 = float(chi_0)
        self.gamma_fluid = float(gamma_fluid)
        self.zeta = float(zeta)
        self.omega_0 = float(omega_0)
        self.phase_thinning = float(phase_thinning)
        self.phase_veto = float(phase_veto)
        self.persistence_steps = int(persistence_steps)

    def couple(
        self, helios_df: pd.DataFrame, atmos_df: pd.DataFrame
    ) -> pd.DataFrame:
        """Couple heliospheric and atmospheric auditor streams.

        Parameters
        ----------
        helios_df, atmos_df:
            DataFrames with at least ``timestamp`` (float seconds, ascending)
            and ``omega_cal``.

        Returns
        -------
        pd.DataFrame
            Copy of ``atmos_df`` with columns
            ``omega_helios_lag``, ``chi_t``, ``p_23``, ``omega_coupled``,
            ``coupled_phase``.
        """
        required = {"timestamp", "omega_cal"}
        if not required.issubset(helios_df.columns):
            raise ValueError("helios_df must contain timestamp and omega_cal")
        if not required.issubset(atmos_df.columns):
            raise ValueError("atmos_df must contain timestamp and omega_cal")

        h_ts = helios_df["timestamp"].to_numpy(dtype=np.float64)
        h_omega = helios_df["omega_cal"].to_numpy(dtype=np.float64)
        a_ts = atmos_df["timestamp"].to_numpy(dtype=np.float64)
        a_omega = atmos_df["omega_cal"].to_numpy(dtype=np.float64)

        if not np.all(np.diff(h_ts) >= 0):
            order = np.argsort(h_ts, kind="mergesort")
            h_ts = h_ts[order]
            h_omega = h_omega[order]
        if not np.all(np.diff(a_ts) >= 0):
            raise ValueError("atmos_df timestamps must be sorted ascending")

        n = len(atmos_df)
        omega_helios_lag = np.zeros(n, dtype=np.float64)
        chi_t = np.empty(n, dtype=np.float64)
        p_23 = np.empty(n, dtype=np.float64)
        omega_coupled = np.empty(n, dtype=np.float64)

        left_bounds = np.searchsorted(
            h_ts, a_ts - self.lag_window_seconds, side="left"
        )
        right_bounds = np.searchsorted(h_ts, a_ts, side="right")
        log2 = np.log(2.0)

        for i in range(n):
            t = float(a_ts[i])
            omega_atmos = (
                float(a_omega[i]) if np.isfinite(a_omega[i]) else self.omega_0
            )

            lo = int(left_bounds[i])
            hi = int(right_bounds[i])
            # Strictly causal EWMA of excess solar uncertainty above thinning
            omega_h = self._ewma_excess(
                t=t,
                h_ts=h_ts[lo:hi],
                h_omega=h_omega[lo:hi],
                log2=log2,
            )
            omega_helios_lag[i] = omega_h

            denom_h = max(1.0 - omega_h, 1e-3)
            chi = float(self.chi_0 * np.exp(min(omega_h / denom_h, 10.0)))
            chi_t[i] = chi

            if omega_atmos >= self.phase_veto:
                p = 1.0
            else:
                p = float(
                    np.clip(
                        self.gamma_fluid
                        * chi
                        * (
                            omega_atmos
                            / max(self.phase_veto - omega_atmos, 1e-3)
                        ),
                        0.0,
                        1.0,
                    )
                )
            p_23[i] = p

            # omega_helios_lag is already an excess EWMA; transfer that mass
            excess_helios = max(0.0, omega_h)
            if not np.isfinite(a_omega[i]):
                omega_coupled[i] = np.nan
            else:
                omega_coupled[i] = omega_atmos + (self.zeta * p * excess_helios)

        out = atmos_df.copy()
        out["omega_helios_lag"] = omega_helios_lag
        out["chi_t"] = chi_t
        out["p_23"] = p_23
        out["omega_coupled"] = omega_coupled
        out["coupled_phase"] = self._apply_persistence(omega_coupled)
        return out

    def _ewma_excess(
        self,
        t: float,
        h_ts: np.ndarray,
        h_omega: np.ndarray,
        log2: float,
    ) -> float:
        """Exponentially weighted mean of max(0, omega_cal - phase_thinning)."""
        if h_ts.size == 0:
            return 0.0

        ages = t - h_ts.astype(np.float64)
        excess = np.maximum(
            0.0, h_omega.astype(np.float64) - self.phase_thinning
        )
        finite = np.isfinite(excess) & np.isfinite(ages) & (ages >= 0.0)
        if not np.any(finite):
            return 0.0

        weights = np.exp(-log2 * ages[finite] / self.lag_half_life_seconds)
        w_sum = float(np.sum(weights))
        if w_sum <= 0.0:
            return 0.0
        return float(np.sum(weights * excess[finite]) / w_sum)

    def _apply_persistence(self, omega_coupled: np.ndarray) -> List[str]:
        """Assign coupled_phase with tactical persistence gasket."""
        thinning_streak = 0
        veto_streak = 0
        phases: List[str] = []

        for val in omega_coupled:
            if not np.isfinite(val):
                thinning_streak = 0
                veto_streak = 0
                phases.append("NOMINAL")
                continue

            if val >= self.phase_veto:
                veto_streak += 1
                thinning_streak += 1
            elif val >= self.phase_thinning:
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

        return phases
