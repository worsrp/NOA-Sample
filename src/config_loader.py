"""Configuration and environment loader for scripts and notebooks."""
import os
from pathlib import Path
from typing import Any, Dict, Optional

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

try:
    import yaml
except ImportError:
    yaml = None


DEFAULT_CONFIG: Dict[str, Any] = {
    "paths": {
        "data_dir": "data",
        "raw_beacon": "data/raw_beacon.csv",
        "m_place": "data/m_place.csv",
        "beacon_distance": "data/beacon_distance.csv",
        "m_shift": "data/m_shift.csv",
        "t_daily_shift": "data/t_daily_shift.csv",
        "m_user": "data/m_user.csv",
        "t_daily_busyness": "data/t_daily_busyness.csv",
        "t_daily_summary": "data/t_daily_summary.csv",
        "staff_patients_care": "data/staff_patients_care.csv",
        "staff_patients_assignments": "data/staff_patients_assignments.csv",
        "interpolated_beacon": "data/01_interpolated_beacon.csv",
        "cleaned_beacon": "data/02_cleaned_beacon.csv",
        "hourly_features": "data/t_staff_hourly_beacon_features.csv",
        "staff_patient_care_level": "data/staff_patient_care_level.csv",
        "floor_daily_care_level_summary": "data/floor_daily_care_level_summary.csv",
        "floor_daily_staffing_summary": "data/floor_daily_staffing_summary.csv",
        "figures_dir": "notebooks/figures",
    },
    "pipeline": {
        "interval_freq": "30s",
        "interval_seconds": 30,
        "shift_buffer_minutes": 0.0,
        "max_missing_threshold": 1,
        "wall_max_minutes": 5.0,
        "stay_threshold_minutes": 1.0,
        "long_stay_threshold_minutes": 10.0,
        "revisit_threshold_minutes": 30.0,
        "min_shift_seconds": 10800,
    },
    "station_groups": {
        "1010": "F1_NURSE_STATION",
        "1011": "F1_NURSE_STATION",
        "2010": "F2_NURSE_STATION",
        "2011": "F2_NURSE_STATION",
        "3010": "F3_NURSE_STATION",
        "3011": "F3_NURSE_STATION",
    },
    "type_mapping": {
        "patient_room": "patient_room",
        "nurse_station": "nurse_station",
        "hallway": "hallway",
    },
    "analysis": {
        "day_shift_hours": [8, 9, 10, 11, 12, 13, 14, 15, 16, 17],
        "day_shift_type": "day_shift",
        "floors": ["1", "2", "3"],
        "floor_colors": {"1": "#e41a1c", "2": "#377eb8", "3": "#4daf4a"},
        "hospital_vars": [
            "num_inpatients",
            "num_admitted_patients",
            "num_nursing_procedures",
        ],
        "all_event_vars": [
            "num_admitted_patients",
            "num_discharged_patients",
            "num_nursing_procedures",
            "num_surgeries",
            "num_inpatients",
        ],
        "beacon_feature_cols": [
            "f_total_time_patient_rooms",
            "f_total_time_nurse_station_areas",
            "f_total_time_hallway",
            "f_total_time_other_areas",
            "f_travel_distance",
            "f_num_location_changes",
            "f_num_patient_room_visits",
            "f_num_patient_room_long_visits",
        ],
        # --- [REDACTED: PROPRIETARY LOGIC] ---
        # Uniform baseline weights replacing proprietary tier/composite scoring weights
        "ladder_to_tier_weight": {0: 1, 1: 1, 2: 1, 3: 1, 4: 1},
        "skill_map": {0: "Novice", 1: "Mid", 2: "Mid", 3: "Veteran", 4: "Veteran"},
        "skill_order": ["Novice", "Mid", "Veteran"],
        "skill_colors": {"Novice": "#fee08b", "Mid": "#fc8d59", "Veteran": "#d73027"},
        "composite_index_weights": {
            "room_ratio": 0.3333,
            "ns_ratio": 0.3333,
            "other_ratio": 0.3334,
        },
    },
}


def find_project_root(start: Optional[Path] = None) -> Path:
    """Locate the repository root containing config.yaml."""
    cur = (start or Path.cwd()).resolve()
    for parent in [cur] + list(cur.parents):
        if (parent / "config.yaml").exists():
            return parent
    return Path(__file__).resolve().parent.parent


def load_config(config_path: Optional[Path] = None) -> Dict[str, Any]:
    """Load configuration from config.yaml and apply .env overrides."""
    root = find_project_root()
    if config_path is None:
        env_cfg = os.environ.get("PORTFOLIO_CONFIG_PATH")
        config_path = (root / env_cfg) if env_cfg else (root / "config.yaml")

    cfg = dict(DEFAULT_CONFIG)
    if yaml is not None and config_path.exists():
        with open(config_path, "r", encoding="utf-8") as f:
            loaded = yaml.safe_load(f) or {}
        cfg.update(loaded)

    # Resolve relative paths against project root
    resolved_paths = {}
    for k, v in cfg.get("paths", {}).items():
        resolved_paths[k] = root / str(v)
    cfg["resolved_paths"] = resolved_paths
    cfg["project_root"] = root
    return cfg
