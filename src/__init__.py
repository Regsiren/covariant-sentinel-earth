"""sentinel-helios-atmos: scale-invariant kinematic trajectory auditor."""

from .benchmark_runner import (
    detect_atmos_events,
    detect_persistent_blocks,
    run_permutation_test,
    run_validation_experiment,
    score_detector,
)
from .bio_benchmark import (
    bootstrap_site_year_metrics,
    detect_flash_drought_onsets,
    run_bio_validation,
)
from .coupling_engine import HeliosAtmosCoupler
from .data_loaders import (
    fetch_nasa_omni2,
    fetch_nasa_omni2_range,
    fetch_open_meteo_agro,
    fetch_open_meteo_atmos,
    load_generic_csv,
    parse_omni2_lines,
)
from .kinematic_core import CovariantSentinelAuditor

__all__ = [
    "CovariantSentinelAuditor",
    "HeliosAtmosCoupler",
    "detect_atmos_events",
    "bootstrap_site_year_metrics",
    "detect_flash_drought_onsets",
    "detect_persistent_blocks",
    "fetch_nasa_omni2",
    "fetch_nasa_omni2_range",
    "fetch_open_meteo_agro",
    "fetch_open_meteo_atmos",
    "load_generic_csv",
    "parse_omni2_lines",
    "run_bio_validation",
    "run_permutation_test",
    "run_validation_experiment",
    "score_detector",
]
