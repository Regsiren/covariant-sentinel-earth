"""Offline unit tests for open-access telemetry loaders."""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from src.data_loaders import (
    fetch_nasa_omni2_range,
    fetch_open_meteo_atmos,
    load_generic_csv,
    parse_omni2_lines,
)


def _omni2_line(
    year: int,
    doy: int,
    hour: int,
    bz_gsm: float,
    v_sw: float,
    p_dyn: float,
) -> str:
    """Build a whitespace-delimited OMNI2-like line with required columns."""
    cols = ["0.0"] * 29
    cols[0] = str(year)
    cols[1] = str(doy)
    cols[2] = str(hour)
    cols[16] = str(bz_gsm)
    cols[24] = str(v_sw)
    cols[28] = str(p_dyn)
    return " ".join(cols)


def test_parse_omni2_strips_fill_values_and_preserves_gaps():
    """Fill-value rows are dropped; the 2-hour gap between valid rows remains."""
    # 2024-05-01 = DOY 122
    year, doy = 2024, 122
    lines = [
        _omni2_line(year, doy, 0, bz_gsm=-3.2, v_sw=410.0, p_dyn=2.1),  # valid
        _omni2_line(year, doy, 1, bz_gsm=999.9, v_sw=420.0, p_dyn=2.2),  # bz fill
        _omni2_line(year, doy, 3, bz_gsm=-4.0, v_sw=9999.0, p_dyn=2.3),  # v_sw fill
        _omni2_line(year, doy, 2, bz_gsm=-5.1, v_sw=430.0, p_dyn=2.4),  # valid
    ]

    df = parse_omni2_lines(lines)

    assert len(df) == 2
    assert list(df.columns) == ["timestamp", "bz_gsm", "v_sw", "p_dyn"]

    t0 = datetime(2024, 5, 1, 0, 0, tzinfo=timezone.utc).timestamp()
    t2 = datetime(2024, 5, 1, 2, 0, tzinfo=timezone.utc).timestamp()
    assert df["timestamp"].tolist() == pytest.approx([t0, t2])
    assert df["timestamp"].iloc[1] - df["timestamp"].iloc[0] == 2.0 * 3600.0

    assert df["bz_gsm"].tolist() == pytest.approx([-3.2, -5.1])
    assert df["v_sw"].tolist() == pytest.approx([410.0, 430.0])
    assert df["p_dyn"].tolist() == pytest.approx([2.1, 2.4])
    assert df["bz_gsm"].notna().all()
    assert df["v_sw"].notna().all()
    assert df["p_dyn"].notna().all()


def test_fetch_open_meteo_atmos_mocked():
    """Mocked Open-Meteo payload: drop None feature rows, keep clean floats."""
    payload = {
        "hourly": {
            "time": [
                "2024-05-01T00:00",
                "2024-05-01T01:00",
                "2024-05-01T02:00",
            ],
            "pressure_msl": [1012.5, None, 1013.0],
            "temperature_2m": [10.0, 11.0, 12.5],
            "wind_speed_100m": [5.0, 6.0, 7.5],
        }
    }
    mock_resp = MagicMock()
    mock_resp.json.return_value = payload
    mock_resp.raise_for_status.return_value = None

    with patch("src.data_loaders.requests.get", return_value=mock_resp) as mock_get:
        df = fetch_open_meteo_atmos("2024-05-01", "2024-05-01")

    mock_get.assert_called_once()
    kwargs = mock_get.call_args.kwargs
    assert kwargs["params"]["hourly"] == "pressure_msl,temperature_2m,wind_speed_100m"
    assert kwargs["params"]["timezone"] == "UTC"

    assert len(df) == 2
    assert list(df.columns) == [
        "timestamp",
        "pressure_msl",
        "temperature_2m",
        "wind_speed_100m",
    ]

    t0 = datetime(2024, 5, 1, 0, 0, tzinfo=timezone.utc).timestamp()
    t2 = datetime(2024, 5, 1, 2, 0, tzinfo=timezone.utc).timestamp()
    assert df["timestamp"].tolist() == pytest.approx([t0, t2])
    assert df["pressure_msl"].tolist() == pytest.approx([1012.5, 1013.0])
    assert df["temperature_2m"].tolist() == pytest.approx([10.0, 12.5])
    assert df["wind_speed_100m"].tolist() == pytest.approx([5.0, 7.5])
    assert df.select_dtypes(include=[np.floating]).shape[1] == 4


def test_fetch_nasa_omni2_range_mocked():
    """Multi-year OMNI2 range: concat years, dedupe timestamps, sort ascending."""
    t_2023 = datetime(2023, 12, 30, 12, 0, tzinfo=timezone.utc).timestamp()
    t_dup = datetime(2024, 1, 1, 0, 0, tzinfo=timezone.utc).timestamp()
    t_2024 = datetime(2024, 1, 2, 6, 0, tzinfo=timezone.utc).timestamp()

    df_2023 = pd.DataFrame(
        {
            "timestamp": [t_dup, t_2023],  # deliberately unsorted + shared edge
            "bz_gsm": [-1.0, -2.0],
            "v_sw": [400.0, 410.0],
            "p_dyn": [2.0, 2.1],
        }
    )
    df_2024 = pd.DataFrame(
        {
            "timestamp": [t_2024, t_dup],  # duplicate of 2024-01-01T00:00
            "bz_gsm": [-4.0, -3.0],
            "v_sw": [430.0, 420.0],
            "p_dyn": [2.3, 2.2],
        }
    )

    def _fake_fetch(year, start_date=None, end_date=None, timeout=30):
        if year == 2023:
            return df_2023.copy()
        if year == 2024:
            return df_2024.copy()
        return pd.DataFrame(columns=["timestamp", "bz_gsm", "v_sw", "p_dyn"])

    with patch(
        "src.data_loaders.fetch_nasa_omni2", side_effect=_fake_fetch
    ) as mock_fetch:
        out = fetch_nasa_omni2_range("2023-12-30", "2024-01-02")

    assert mock_fetch.call_count == 2
    years_called = sorted(
        (c.kwargs["year"] if "year" in c.kwargs else c.args[0])
        for c in mock_fetch.call_args_list
    )
    assert years_called == [2023, 2024]

    assert len(out) == 3  # duplicate timestamp dropped
    assert out["timestamp"].is_monotonic_increasing
    assert out["timestamp"].tolist() == pytest.approx([t_2023, t_dup, t_2024])
    # keep="first" retains the 2023 copy of the duplicated edge timestamp
    assert float(out.loc[out["timestamp"] == t_dup, "bz_gsm"].iloc[0]) == -1.0


def test_load_generic_csv(tmp_path):
    """Temporary CSV loads with sorted timestamps and selected features."""
    csv_path = tmp_path / "sample_telemetry.csv"
    csv_path.write_text(
        "\n".join(
            [
                "time,feat_a,feat_b,ignored",
                "2024-05-01T02:00:00Z,3.0,30.0,x",
                "2024-05-01T00:00:00Z,1.0,10.0,y",
                "2024-05-01T01:00:00Z,,20.0,z",
                "2024-05-01T01:30:00Z,2.0,20.0,w",
            ]
        ),
        encoding="utf-8",
    )

    df = load_generic_csv(
        str(csv_path), time_col="time", feature_cols=["feat_a", "feat_b"]
    )

    assert list(df.columns) == ["timestamp", "feat_a", "feat_b"]
    assert len(df) == 3  # NaN feat_a row dropped

    expected_ts = [
        datetime(2024, 5, 1, 0, 0, tzinfo=timezone.utc).timestamp(),
        datetime(2024, 5, 1, 1, 30, tzinfo=timezone.utc).timestamp(),
        datetime(2024, 5, 1, 2, 0, tzinfo=timezone.utc).timestamp(),
    ]
    assert df["timestamp"].tolist() == pytest.approx(expected_ts)
    assert df["timestamp"].is_monotonic_increasing
    assert df["feat_a"].tolist() == pytest.approx([1.0, 2.0, 3.0])
    assert df["feat_b"].tolist() == pytest.approx([10.0, 20.0, 30.0])
