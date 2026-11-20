#!/usr/bin/env python3
"""Live CLI: validate HELIOS→ATMOS coupling on NASA OMNI2 + Open-Meteo ERA5."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running as `python scripts/run_live_validation.py` from repo root
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.benchmark_runner import run_validation_experiment
from src.data_loaders import fetch_nasa_omni2_range, fetch_open_meteo_atmos


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Fetch NASA OMNI2 + Open-Meteo ERA5 and run the HELIOS→ATMOS "
            "benchmark / null-hypothesis validation."
        )
    )
    p.add_argument(
        "--start",
        type=str,
        default="2022-01-01",
        help="ISO start date (multi-year SC25 default)",
    )
    p.add_argument(
        "--end",
        type=str,
        default="2024-12-31",
        help="ISO end date (multi-year SC25 default)",
    )
    p.add_argument("--lat", type=float, default=51.5074, help="Latitude (London)")
    p.add_argument("--lon", type=float, default=-0.1278, help="Longitude (London)")
    p.add_argument(
        "--mode",
        type=str,
        choices=["all", "blocks"],
        default="blocks",
        help="Event definition: all anomalies or persistent high-pressure blocks",
    )
    p.add_argument(
        "--out",
        type=str,
        default=str(ROOT / "outputs" / "validation_results.csv"),
        help="Path for coupled_df CSV output",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()

    print(
        f"Fetching NASA OMNI2 range [{args.start} → {args.end}] ...",
        file=sys.stderr,
    )
    helios_raw = fetch_nasa_omni2_range(args.start, args.end)
    print(f"  HELIOS rows: {len(helios_raw)}", file=sys.stderr)

    print(
        f"Fetching Open-Meteo ERA5 atmos @ ({args.lat}, {args.lon}) ...",
        file=sys.stderr,
    )
    atmos_raw = fetch_open_meteo_atmos(
        start_date=args.start,
        end_date=args.end,
        lat=args.lat,
        lon=args.lon,
    )
    print(f"  ATMOS rows: {len(atmos_raw)}", file=sys.stderr)

    if helios_raw.empty or atmos_raw.empty:
        print("ERROR: empty HELIOS or ATMOS frame; aborting.", file=sys.stderr)
        return 1

    print(f"Running validation experiment (mode={args.mode}) ...", file=sys.stderr)
    result = run_validation_experiment(helios_raw, atmos_raw, mode=args.mode)

    print(result["report_md"])

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    result["coupled_df"].to_csv(out_path, index=False)
    print(f"Wrote coupled_df → {out_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
