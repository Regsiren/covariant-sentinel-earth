#!/usr/bin/env python3
"""Live CLI: BIO_05 Flash Drought validation on Copernicus ERA5 agro data."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.bio_benchmark import run_bio_validation
from src.data_loaders import fetch_open_meteo_agro


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Fetch East Anglia ERA5 agro-hydrology via Open-Meteo and validate "
            "the BIO_05 flash-drought early-warning engine."
        )
    )
    p.add_argument("--start", type=str, default="2020-01-01")
    p.add_argument("--end", type=str, default="2024-12-31")
    p.add_argument("--lat", type=float, default=52.6309, help="East Anglia / Norfolk")
    p.add_argument("--lon", type=float, default=1.2974, help="East Anglia / Norfolk")
    p.add_argument(
        "--out",
        type=str,
        default=str(ROOT / "outputs" / "bio_validation_results.csv"),
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()

    print(
        f"Fetching Open-Meteo agro ERA5 @ ({args.lat}, {args.lon}) "
        f"[{args.start} → {args.end}] ...",
        file=sys.stderr,
    )
    agro_df = fetch_open_meteo_agro(
        start_date=args.start,
        end_date=args.end,
        lat=args.lat,
        lon=args.lon,
    )
    print(f"  Daily agro rows: {len(agro_df)}", file=sys.stderr)

    if agro_df.empty:
        print("ERROR: empty agro frame; aborting.", file=sys.stderr)
        return 1

    print("Running BIO_05 flash-drought validation ...", file=sys.stderr)
    result = run_bio_validation(agro_df)

    print(result["report_md"])
    print("Detected flash drought onset dates:")
    if result["event_dates"]:
        for d in result["event_dates"]:
            print(f"  - {d}")
    else:
        print("  (none)")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    result["audited_df"].to_csv(out_path, index=False)
    print(f"Wrote audited_df → {out_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
