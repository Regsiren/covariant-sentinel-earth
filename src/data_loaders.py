"""Open-access telemetry loaders for HELIOS (OMNI2) and ATMOS (Open-Meteo)."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import List

import numpy as np
import pandas as pd
import requests

OMNI2_URL = "https://spdf.gsfc.nasa.gov/pub/data/omni/low_res_omni/omni2_{year}.dat"
OPEN_METEO_URL = "https://archive-api.open-meteo.com/v1/archive"

# OMNI2 fill thresholds (values at or above are missing)
_BZ_FILL = 999.0
_VSW_FILL = 9990.0
_PDYN_FILL = 99.0


def _utc_unix_from_year_doy_hour(year: int, doy: int, hour: int) -> float:
    """Convert Year + 1-based DOY + Hour to UTC Unix timestamp (seconds)."""
    dt = datetime(int(year), 1, 1, tzinfo=timezone.utc) + timedelta(
        days=int(doy) - 1, hours=int(hour)
    )
    return float(dt.timestamp())


def _parse_iso_date_utc(date_str: str) -> datetime:
    """Parse YYYY-MM-DD as UTC midnight."""
    return datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)


def parse_omni2_lines(
    lines: list[str],
    start_date: str | None = None,
    end_date: str | None = None,
) -> pd.DataFrame:
    """Parse NASA SPDF OMNI2 hourly whitespace-delimited text lines.

    Column positions (0-indexed): year, DOY, hour, bz_gsm(16), v_sw(24), p_dyn(28).
    Fill values are replaced with NaN and rows with any NaN feature are dropped
    (no interpolation) so gap-masking can handle blackouts natively.
    """
    rows: List[dict] = []
    for raw in lines:
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) <= 28:
            continue
        try:
            year = int(float(parts[0]))
            doy = int(float(parts[1]))
            hour = int(float(parts[2]))
            bz = float(parts[16])
            v_sw = float(parts[24])
            p_dyn = float(parts[28])
        except (ValueError, IndexError):
            continue

        if bz >= _BZ_FILL:
            bz = np.nan
        if v_sw >= _VSW_FILL:
            v_sw = np.nan
        if p_dyn >= _PDYN_FILL:
            p_dyn = np.nan

        rows.append(
            {
                "timestamp": _utc_unix_from_year_doy_hour(year, doy, hour),
                "bz_gsm": bz,
                "v_sw": v_sw,
                "p_dyn": p_dyn,
            }
        )

    df = pd.DataFrame(rows, columns=["timestamp", "bz_gsm", "v_sw", "p_dyn"])
    if df.empty:
        return df

    df = df.dropna(subset=["bz_gsm", "v_sw", "p_dyn"]).copy()

    if start_date is not None or end_date is not None:
        ts = pd.to_datetime(df["timestamp"], unit="s", utc=True)
        if start_date is not None:
            start = _parse_iso_date_utc(start_date)
            df = df.loc[ts >= start]
            ts = pd.to_datetime(df["timestamp"], unit="s", utc=True)
        if end_date is not None:
            # Inclusive on UTC calendar date → keep through end of end_date
            end = _parse_iso_date_utc(end_date) + timedelta(days=1)
            df = df.loc[ts < end]

    df = df.sort_values("timestamp", kind="mergesort").reset_index(drop=True)
    return df[["timestamp", "bz_gsm", "v_sw", "p_dyn"]]


def fetch_nasa_omni2(
    year: int,
    start_date: str | None = None,
    end_date: str | None = None,
    timeout: int = 30,
) -> pd.DataFrame:
    """Download and parse a yearly NASA OMNI2 low-resolution file."""
    url = OMNI2_URL.format(year=int(year))
    response = requests.get(url, timeout=timeout)
    response.raise_for_status()
    lines = response.text.splitlines()
    return parse_omni2_lines(lines, start_date=start_date, end_date=end_date)


def fetch_nasa_omni2_range(
    start_date: str,
    end_date: str,
    timeout: int = 30,
) -> pd.DataFrame:
    """Fetch and concatenate NASA OMNI2 across one or more calendar years.

    Loops ``fetch_nasa_omni2`` from ``start_year`` through ``end_year``
    (inclusive), drops empty frames and duplicate timestamps, then sorts
    ascending by ``timestamp``.
    """
    start_year = int(start_date[:4])
    end_year = int(end_date[:4])
    if end_year < start_year:
        raise ValueError("end_date year must be >= start_date year")

    frames: List[pd.DataFrame] = []
    for year in range(start_year, end_year + 1):
        df = fetch_nasa_omni2(
            year=year,
            start_date=start_date,
            end_date=end_date,
            timeout=timeout,
        )
        if df is not None and not df.empty:
            frames.append(df)

    if not frames:
        return pd.DataFrame(columns=["timestamp", "bz_gsm", "v_sw", "p_dyn"])

    out = pd.concat(frames, ignore_index=True)
    out = out.drop_duplicates(subset=["timestamp"], keep="first")
    out = out.sort_values("timestamp", kind="mergesort").reset_index(drop=True)
    return out[["timestamp", "bz_gsm", "v_sw", "p_dyn"]]


def fetch_open_meteo_atmos(
    start_date: str,
    end_date: str,
    lat: float = 51.5074,
    lon: float = -0.1278,
    timeout: int = 30,
) -> pd.DataFrame:
    """Fetch ERA5-backed hourly atmospheric telemetry via Open-Meteo Archive API."""
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": "pressure_msl,temperature_2m,wind_speed_100m",
        "timezone": "UTC",
    }
    response = requests.get(OPEN_METEO_URL, params=params, timeout=timeout)
    response.raise_for_status()
    payload = response.json()
    hourly = payload.get("hourly") or {}

    times = hourly.get("time") or []
    pressure = hourly.get("pressure_msl") or []
    temperature = hourly.get("temperature_2m") or []
    wind = hourly.get("wind_speed_100m") or []

    n = min(len(times), len(pressure), len(temperature), len(wind))
    rows: List[dict] = []
    for i in range(n):
        t_raw = times[i]
        # Open-Meteo returns "YYYY-MM-DDTHH:MM" (UTC when timezone=UTC)
        if isinstance(t_raw, (int, float)):
            ts = float(t_raw)
        else:
            dt = datetime.fromisoformat(str(t_raw))
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            else:
                dt = dt.astimezone(timezone.utc)
            ts = float(dt.timestamp())

        def _as_float(val: object) -> float:
            if val is None:
                return np.nan
            try:
                return float(val)
            except (TypeError, ValueError):
                return np.nan

        rows.append(
            {
                "timestamp": ts,
                "pressure_msl": _as_float(pressure[i]),
                "temperature_2m": _as_float(temperature[i]),
                "wind_speed_100m": _as_float(wind[i]),
            }
        )

    df = pd.DataFrame(
        rows,
        columns=[
            "timestamp",
            "pressure_msl",
            "temperature_2m",
            "wind_speed_100m",
        ],
    )
    if df.empty:
        return df

    df = df.dropna(
        subset=["pressure_msl", "temperature_2m", "wind_speed_100m"]
    ).copy()
    df = df.sort_values("timestamp", kind="mergesort").reset_index(drop=True)
    return df[
        ["timestamp", "pressure_msl", "temperature_2m", "wind_speed_100m"]
    ]


def fetch_open_meteo_agro(
    start_date: str,
    end_date: str,
    lat: float = 52.6309,
    lon: float = 1.2974,
    timeout: int = 30,
) -> pd.DataFrame:
    """Fetch ERA5-backed agro-hydrological telemetry and resample to daily UTC.

    Hourly Open-Meteo archive fields are aggregated to daily means for soil
    moisture / VPD and daily sum for ET0, then returned with Unix timestamps.
    Multi-year ranges are fetched year-by-year then concatenated for API safety.
    """
    feature_hourly = (
        "soil_moisture_0_to_7cm,soil_moisture_7_to_28cm,"
        "et0_fao_evapotranspiration,vapour_pressure_deficit"
    )
    start_year = int(start_date[:4])
    end_year = int(end_date[:4])
    hourly_rows: List[dict] = []

    def _as_float(val: object) -> float:
        if val is None:
            return np.nan
        try:
            return float(val)
        except (TypeError, ValueError):
            return np.nan

    for year in range(start_year, end_year + 1):
        y_start = start_date if year == start_year else f"{year}-01-01"
        y_end = end_date if year == end_year else f"{year}-12-31"
        params = {
            "latitude": lat,
            "longitude": lon,
            "start_date": y_start,
            "end_date": y_end,
            "hourly": feature_hourly,
            "timezone": "UTC",
        }
        response = requests.get(OPEN_METEO_URL, params=params, timeout=timeout)
        response.raise_for_status()
        payload = response.json()
        hourly = payload.get("hourly") or {}

        times = hourly.get("time") or []
        sm0 = hourly.get("soil_moisture_0_to_7cm") or []
        sm7 = hourly.get("soil_moisture_7_to_28cm") or []
        et0 = hourly.get("et0_fao_evapotranspiration") or []
        vpd = hourly.get("vapour_pressure_deficit") or []
        n = min(len(times), len(sm0), len(sm7), len(et0), len(vpd))
        for i in range(n):
            t_raw = times[i]
            if isinstance(t_raw, (int, float)):
                ts = float(t_raw)
            else:
                dt = datetime.fromisoformat(str(t_raw))
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                else:
                    dt = dt.astimezone(timezone.utc)
                ts = float(dt.timestamp())
            hourly_rows.append(
                {
                    "timestamp": ts,
                    "soil_moisture_0_to_7cm": _as_float(sm0[i]),
                    "soil_moisture_7_to_28cm": _as_float(sm7[i]),
                    "et0_fao_evapotranspiration": _as_float(et0[i]),
                    "vapour_pressure_deficit": _as_float(vpd[i]),
                }
            )

    if not hourly_rows:
        return pd.DataFrame(
            columns=[
                "timestamp",
                "date",
                "soil_moisture_0_to_7cm",
                "soil_moisture_7_to_28cm",
                "et0_fao_evapotranspiration",
                "vapour_pressure_deficit",
            ]
        )

    hourly_df = pd.DataFrame(hourly_rows)
    hourly_df = hourly_df.sort_values("timestamp", kind="mergesort")
    hourly_df["datetime"] = pd.to_datetime(
        hourly_df["timestamp"], unit="s", utc=True
    )
    hourly_df = hourly_df.set_index("datetime")

    daily = pd.DataFrame(
        {
            "soil_moisture_0_to_7cm": hourly_df[
                "soil_moisture_0_to_7cm"
            ].resample("1D").mean(),
            "soil_moisture_7_to_28cm": hourly_df[
                "soil_moisture_7_to_28cm"
            ].resample("1D").mean(),
            "vapour_pressure_deficit": hourly_df[
                "vapour_pressure_deficit"
            ].resample("1D").mean(),
            "et0_fao_evapotranspiration": hourly_df[
                "et0_fao_evapotranspiration"
            ].resample("1D").sum(),
        }
    ).dropna(how="any")

    daily = daily.reset_index()
    daily["date"] = daily["datetime"].dt.strftime("%Y-%m-%d")
    # UTC midnight Unix seconds for each calendar day
    daily["timestamp"] = (
        daily["datetime"].astype("int64") // 10**9
    ).astype(np.float64)

    out = daily[
        [
            "timestamp",
            "date",
            "soil_moisture_0_to_7cm",
            "soil_moisture_7_to_28cm",
            "et0_fao_evapotranspiration",
            "vapour_pressure_deficit",
        ]
    ].copy()
    out = out.sort_values("timestamp", kind="mergesort").reset_index(drop=True)
    return out


def load_generic_csv(
    filepath: str, time_col: str, feature_cols: list[str]
) -> pd.DataFrame:
    """Load a generic CSV into auditor-ready ``timestamp`` + feature columns."""
    raw = pd.read_csv(filepath)
    if time_col not in raw.columns:
        raise ValueError(f"time_col '{time_col}' not found in CSV")
    missing = [c for c in feature_cols if c not in raw.columns]
    if missing:
        raise ValueError(f"feature_cols missing from CSV: {missing}")

    time_series = raw[time_col]
    if pd.api.types.is_numeric_dtype(time_series):
        timestamps = time_series.astype(np.float64)
    else:
        dt = pd.to_datetime(time_series, utc=True)
        timestamps = (dt.astype("int64") // 10**9).astype(np.float64)

    out = pd.DataFrame({"timestamp": timestamps})
    for col in feature_cols:
        out[col] = pd.to_numeric(raw[col], errors="coerce")

    out = out.dropna(subset=feature_cols).copy()
    out = out.sort_values("timestamp", kind="mergesort").reset_index(drop=True)
    return out[["timestamp"] + list(feature_cols)]
