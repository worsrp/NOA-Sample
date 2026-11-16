"""Step 3: Hourly Workload & Mobility Feature Engineering Script.

Segments cleaned beacon time grids into stays, filters transient noise stays,
and aggregates 14 hourly nursing workflow and mobility features combined with floor busyness.
Configuration and mappings are loaded from config.yaml / .env.
"""
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import argparse
import json
import logging
import os
import sys
import unicodedata
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config_loader import load_config

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# Constants
# ============================================================================
ALLOWED_CATEGORIES = ('patient_room', 'nurse_station', 'hallway')
MIN_SHIFT_SECONDS = 3 * 3600
UNKNOWN_KEY = '__unknown__'
NULL_TEXTS = ['', 'NULL', 'null', 'Null', 'NaN', 'nan', 'None']

OUTPUT_COLUMNS = [
    'user_id', 'shift_date', 'shift_id', 'hour', 'floor', 'busyness',
    'f_total_time_patient_rooms', 'f_total_time_nurse_station_areas',
    'f_total_time_hallway', 'f_total_time_other_areas',
    'f_travel_distance', 'f_average_speed', 'f_num_location_changes',
    'f_num_unique_location_visits', 'f_num_floor_changes',
    'f_num_patient_room_visits', 'f_num_unique_patient_room_visits',
    'f_num_patient_room_long_visits', 'f_transition_entropy',
    'f_num_patient_room_revisits',
]
PATIENT_COLUMNS = [
    'f_num_patient_room_visits', 'f_num_unique_patient_room_visits',
    'f_num_patient_room_long_visits', 'f_num_patient_room_revisits'
]
TIME_COLUMNS = {
    'patient_room': 'f_total_time_patient_rooms',
    'nurse_station': 'f_total_time_nurse_station_areas',
    'hallway': 'f_total_time_hallway',
    'other': 'f_total_time_other_areas',
}


def get_default_type_mapping() -> Dict[str, str]:
    cfg = load_config()
    return {str(k): str(v) for k, v in cfg.get('type_mapping', {}).items()}


def get_default_station_groups() -> Dict[str, str]:
    cfg = load_config()
    return {str(k).strip(): str(v) for k, v in cfg.get('station_groups', {}).items()}


# ============================================================================
# Small helpers
# ============================================================================
def to_ns(series: pd.Series) -> pd.Series:
    """Force nanosecond resolution (pandas 3 may infer a coarser unit)."""
    return series.astype('datetime64[ns]')


def norm_text(value) -> Optional[str]:
    """Trim and unify full-width/half-width characters."""
    if value is None or value is pd.NA or (isinstance(value, float) and np.isnan(value)):
        return None
    return unicodedata.normalize('NFKC', str(value)).strip()


def to_nullable_int(series: pd.Series, name: str) -> pd.Series:
    num = pd.to_numeric(series, errors='coerce')
    bad = series.notna() & num.isna()
    if bad.any():
        raise ValueError(f"Column '{name}' has non-numeric values, e.g. {series[bad].iloc[0]!r}")
    fractional = num.notna() & ((num % 1) != 0)
    if fractional.any():
        raise ValueError(f"Column '{name}' must hold whole numbers, e.g. {num[fractional].iloc[0]!r}")
    return num.astype('Int64')


def load_json_dict(path: Path, label: str) -> Dict[str, str]:
    if not path.exists():
        raise FileNotFoundError(f"{label} file not found: {path}")
    with open(path, encoding='utf-8-sig') as f:
        data = json.load(f)
    if not isinstance(data, dict) or not all(isinstance(v, str) for v in data.values()):
        raise ValueError(f"{label} file must be a JSON object with text values, "
                         f"for example {{\"1010\": \"F1_NURSE_STATION\"}}")
    return {str(k): v for k, v in data.items()}


# ============================================================================
# Configuration
# ============================================================================
@dataclass
class Config:
    """Settings for the hourly feature step."""

    beacon_data_path: Path
    beacon_distance_path: Path
    beacon_place_path: Path
    daily_busyness_path: Path
    output_path: Path

    stay_duration_threshold_minutes: Optional[float] = 1.0
    long_stay_duration_threshold_minutes: Optional[float] = 10.0
    revisit_time_threshold_minutes: Optional[float] = 30.0

    beacon_interval_seconds: int = 30

    type_mapping: Dict[str, str] = field(default_factory=get_default_type_mapping)
    station_groups: Dict[str, str] = field(default_factory=get_default_station_groups)

    def __post_init__(self):
        for name in ('stay_duration_threshold_minutes', 'long_stay_duration_threshold_minutes',
                     'revisit_time_threshold_minutes'):
            value = getattr(self, name)
            if value is not None and value < 0:
                raise ValueError(f"{name} must be 0 or larger")
            if value == 0:
                setattr(self, name, None)
        if self.beacon_interval_seconds <= 0:
            raise ValueError("--interval-seconds must be positive")

        bad = {k: v for k, v in self.type_mapping.items() if v not in ALLOWED_CATEGORIES}
        if bad:
            raise ValueError(f"Type mapping has unknown categories {sorted(set(bad.values()))}. "
                             f"Allowed: {list(ALLOWED_CATEGORIES)}")
        self.type_mapping = {norm_text(k): v for k, v in self.type_mapping.items()}
        self.station_groups = {str(k).strip(): v for k, v in self.station_groups.items()}

    @classmethod
    def from_args(cls) -> 'Config':
        cfg = load_config()
        p = cfg['resolved_paths']
        pipe_cfg = cfg.get('pipeline', {})

        parser = argparse.ArgumentParser(description='Beacon Location Feature Engineering Script (Hourly)')
        parser.add_argument('--beacon-data', type=Path,
                            default=Path(os.environ.get('CLEANED_BEACON_PATH', p['cleaned_beacon'])),
                            help='Cleaned beacon CSV from beacon_cleaning.py')
        parser.add_argument('--beacon-distance', type=Path,
                            default=Path(os.environ.get('BEACON_DISTANCE_PATH', p['beacon_distance'])),
                            help='Distance matrix CSV (start_place_id, end_place_id, distance)')
        parser.add_argument('--beacon-place', type=Path,
                            default=Path(os.environ.get('M_PLACE_PATH', p['m_place'])),
                            help='Place master CSV (place_id, type, and floor)')
        parser.add_argument('--daily-busyness', type=Path,
                            default=Path(os.environ.get('DAILY_BUSYNESS_PATH', p['t_daily_busyness'])),
                            help='Daily busyness CSV (date, floor, busyness)')
        parser.add_argument('--output', type=Path,
                            default=Path(os.environ.get('HOURLY_FEATURES_OUTPUT_PATH', p['hourly_features'])),
                            help='Output features CSV')
        parser.add_argument('--stay-threshold', type=float,
                            default=float(pipe_cfg.get('stay_threshold_minutes', 1.0)),
                            help='Minutes. Shorter stays are treated as noise (0 = no threshold)')
        parser.add_argument('--long-stay-threshold', type=float,
                            default=float(pipe_cfg.get('long_stay_threshold_minutes', 10.0)),
                            help='Minutes. Patient room stays at least this long are long visits (0 = none)')
        parser.add_argument('--revisit-threshold', type=float,
                            default=float(pipe_cfg.get('revisit_threshold_minutes', 30.0)),
                            help='Minutes. A return to a patient room counts as a revisit if it happens '
                                 'within this time after leaving it (0 = no time cap)')
        parser.add_argument('--interval-seconds', type=int,
                            default=int(pipe_cfg.get('interval_seconds', 30)),
                            help='Seconds per row; must match the interpolation step (default: 30)')
        parser.add_argument('--type-mapping', type=Path, default=None,
                            help='Optional JSON {"place master type": "patient_room|nurse_station|hallway"}')
        parser.add_argument('--station-groups', type=Path, default=None,
                            help='Optional JSON {"place_id": "group label"}')
        args = parser.parse_args()

        kwargs = {}
        if args.type_mapping:
            kwargs['type_mapping'] = load_json_dict(args.type_mapping, 'Type mapping')
        if args.station_groups:
            kwargs['station_groups'] = load_json_dict(args.station_groups, 'Station groups')

        return cls(
            beacon_data_path=args.beacon_data,
            beacon_distance_path=args.beacon_distance,
            beacon_place_path=args.beacon_place,
            daily_busyness_path=args.daily_busyness,
            output_path=args.output,
            stay_duration_threshold_minutes=args.stay_threshold,
            long_stay_duration_threshold_minutes=args.long_stay_threshold,
            revisit_time_threshold_minutes=args.revisit_threshold,
            beacon_interval_seconds=args.interval_seconds,
            **kwargs,
        )


# ============================================================================
# Data loading
# ============================================================================
class DataLoader:
    """Reads the input files as text and checks their columns."""

    def __init__(self, config: Config):
        self.config = config

    @staticmethod
    def read_csv(path: Path, required_columns: List[str], label: str) -> pd.DataFrame:
        if not path.exists():
            raise FileNotFoundError(f"{label} file not found: {path}")
        logger.info(f"Loading {label}: {path}")
        df = pd.read_csv(path, dtype=str, keep_default_na=False, na_values=NULL_TEXTS,
                         encoding='utf-8-sig')
        df.columns = [str(c).strip() for c in df.columns]
        missing = [c for c in required_columns if c not in df.columns]
        if missing:
            raise ValueError(f"{label}: missing required columns {missing} (found: {list(df.columns)})")
        return df

    def load_all(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        c = self.config
        beacon = self.read_csv(c.beacon_data_path,
                               ['user_id', 'shift_date', 'shift_id', 'timestamp', 'place_id'],
                               'beacon data')
        distance = self.read_csv(c.beacon_distance_path,
                                 ['start_place_id', 'end_place_id', 'distance'], 'distance matrix')
        place = self.read_csv(c.beacon_place_path, ['place_id', 'type'], 'place master')
        busyness = self.read_csv(c.daily_busyness_path, ['date', 'floor', 'busyness'], 'daily busyness')
        return beacon, distance, place, busyness


# ============================================================================
# Preparation
# ============================================================================
class Preparer:
    """Turns the loaded tables into clean row, place and distance tables."""

    def __init__(self, config: Config):
        self.config = config

    def prepare_places(self, place_df: pd.DataFrame) -> pd.DataFrame:
        """Returns a table indexed by place_id with `category` and `floor`."""
        tbl = pd.DataFrame({
            'place_id': to_nullable_int(place_df['place_id'], 'place_id'),
            'type_text': place_df['type'].map(norm_text),
        })
        tbl['floor'] = (to_nullable_int(place_df['floor'], 'floor')
                        if 'floor' in place_df.columns else pd.array([pd.NA] * len(tbl), dtype='Int64'))
        tbl = tbl.dropna(subset=['place_id']).drop_duplicates()
        if tbl['place_id'].duplicated().any():
            raise ValueError("Place master has the same place_id with different type or floor, "
                             f"e.g. {int(tbl.loc[tbl['place_id'].duplicated(), 'place_id'].iloc[0])}")

        mapping = self.config.type_mapping
        tbl['category'] = tbl['type_text'].map(mapping).fillna('other')

        present = set(tbl['type_text'].dropna())
        unmapped = tbl[tbl['category'] == 'other']['type_text'].value_counts(dropna=False)
        if len(unmapped):
            shown = {('(empty)' if k is None or (isinstance(k, float) and np.isnan(k)) else k): int(v)
                     for k, v in unmapped.items()}
            logger.info(f"Place types counted as other areas (type: number of places): {shown}")
        unused = [k for k in mapping if k not in present]
        if unused:
            logger.warning(f"Type mapping entries not found in the place master: {unused}")
        for cat in ALLOWED_CATEGORIES:
            if cat not in mapping.values():
                logger.warning(f"The type mapping has no entry for '{cat}', so that time feature will be 0.")
        return tbl.set_index('place_id')[['category', 'floor']]

    @staticmethod
    def create_distance_lookup(distance_df: pd.DataFrame) -> Dict[Tuple[int, int], float]:
        start = to_nullable_int(distance_df['start_place_id'], 'start_place_id')
        end = to_nullable_int(distance_df['end_place_id'], 'end_place_id')
        dist = pd.to_numeric(distance_df['distance'], errors='coerce')
        ok = start.notna() & end.notna() & dist.notna()
        if (~ok).any():
            logger.warning(f"Ignoring {int((~ok).sum())} distance rows with an empty or non-numeric value.")
        explicit = dict(zip(zip(start[ok].astype('int64'), end[ok].astype('int64')), dist[ok].astype(float)))
        lookup = dict(explicit)
        for (a, b), d in explicit.items():
            lookup.setdefault((b, a), d)
        return lookup

    def prepare_rows(self, beacon_df: pd.DataFrame, places: pd.DataFrame) -> pd.DataFrame:
        """Clean beacon rows with timestamp, shift keys, place key, category and floor."""
        ts = pd.to_datetime(beacon_df['timestamp'], format='ISO8601', errors='coerce')
        if ts.isna().any():
            raise ValueError(f"Beacon data has {int(ts.isna().sum())} unreadable timestamps.")
        sd = pd.to_datetime(beacon_df['shift_date'], format='ISO8601', errors='coerce')
        if sd.isna().any():
            raise ValueError(f"Beacon data has {int(sd.isna().sum())} empty or unreadable shift_date values.")

        df = pd.DataFrame({
            'user_id': beacon_df['user_id'].str.strip(),
            'timestamp': to_ns(ts),
            'shift_date': to_ns(sd).dt.normalize(),
            'shift_id': to_nullable_int(beacon_df['shift_id'], 'shift_id'),
            'place_id': to_nullable_int(beacon_df['place_id'], 'place_id'),
        })
        if df['user_id'].isna().any():
            raise ValueError("Beacon data has rows with an empty user_id.")

        no_shift = df['shift_id'].isna()
        if no_shift.any():
            logger.info(f"Dropping {int(no_shift.sum())} rows without a shift_id.")
            beacon_floor = beacon_df.loc[~no_shift]
            df = df[~no_shift]
        else:
            beacon_floor = beacon_df

        before = len(df)
        df = df.drop_duplicates()
        dup = df.duplicated(['user_id', 'timestamp'], keep='first')
        if dup.any():
            logger.warning(f"{int(dup.sum())} rows share a user_id and timestamp; keeping the first of each.")
            df = df[~dup]
        if len(df) < before:
            logger.info(f"Removed {before - len(df)} duplicate rows.")

        groups = self.config.station_groups
        known_ids = df['place_id'].dropna().unique()
        key_map = {int(p): groups.get(str(int(p)), str(int(p))) for p in known_ids}
        df['loc_key'] = df['place_id'].map(key_map).fillna(UNKNOWN_KEY).astype(object)

        cat_map = places['category'].to_dict()
        df['category'] = df['place_id'].map(cat_map).fillna('other').astype(object)
        floor_map = places['floor'].dropna().astype('int64').to_dict()
        df['floor'] = df['place_id'].map(floor_map).astype('Int64')
        if df['floor'].notna().sum() == 0 and 'floor' in beacon_floor.columns:
            logger.info("Place master has no floor; using the floor column of the beacon data.")
            fl = to_nullable_int(beacon_floor['floor'], 'floor')
            df['floor'] = fl.reindex(df.index)
        if df['floor'].notna().sum() == 0:
            raise ValueError("No floor information found (place master and beacon data have no usable floor).")

        missing_places = df['place_id'].notna() & ~df['place_id'].isin(places.index)
        if missing_places.any():
            ids = sorted(df.loc[missing_places, 'place_id'].unique())[:10]
            logger.warning(f"{int(missing_places.sum())} rows use a place_id that is not in the place "
                           f"master (e.g. {[int(i) for i in ids]}); they count as other areas.")

        df = df.sort_values(['user_id', 'shift_date', 'shift_id', 'timestamp']).reset_index(drop=True)
        df['shift_key'] = df.groupby(['user_id', 'shift_date', 'shift_id'], sort=True).ngroup()
        return df

    def filter_short_shifts(self, rows: pd.DataFrame) -> pd.DataFrame:
        # --- [REDACTED: PROPRIETARY LOGIC] ---
        # Conceptually, this step filters out partial or interrupted clinical shifts whose
        # observed beacon span falls below a domain-calibrated minimum duration threshold.
        # Below is a simplified baseline check retaining all shifts with a positive time span.
        span = rows.groupby('shift_key')['timestamp'].agg(['min', 'max'])
        seconds = (span['max'] - span['min']).dt.total_seconds()
        keep = seconds[seconds > 0].index
        logger.info(f"Shifts before duration filter: {len(span)}")
        logger.info(f"Shifts after baseline duration filter (> 0s): {len(keep)}")
        out = rows[rows['shift_key'].isin(keep)].reset_index(drop=True)
        if out.empty:
            raise ValueError("No shift is longer than the minimum shift length, so no features can be made.")
        return out


# ============================================================================
# Stays
# ============================================================================
class StayBuilder:
    """Rows -> stays -> valid stays (S) -> known moves (M), all per shift."""

    def __init__(self, config: Config):
        self.config = config
        self.interval_s = config.beacon_interval_seconds
        self.step = pd.Timedelta(seconds=self.interval_s)
        thr = config.stay_duration_threshold_minutes
        self.stay_threshold_s = thr * 60 if thr else None

    def check_interval(self, rows: pd.DataFrame):
        diffs = rows.groupby('shift_key')['timestamp'].diff().dropna()
        if diffs.empty:
            return
        median = diffs.median()
        if abs((median - self.step).total_seconds()) > 1:
            logger.warning(f"The usual spacing between rows is {median.total_seconds():g} s, but "
                           f"--interval-seconds is {self.interval_s}. Time features would be wrong; "
                           f"use the same interval as the interpolation step.")

    def build(self, rows: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        # --- [REDACTED: PROPRIETARY LOGIC] ---
        # Conceptually, this step implements a 3-stage trajectory segmentation state machine
        # that detects temporal holes, filters transient pass-by noise stays below a clinical
        # duration threshold, and iteratively re-merges contiguous same-place stays across
        # filtered noise gaps using cumulative hole counters.
        # Below is a simplified single-pass contiguous run-length grouping baseline.
        n = len(rows)
        sk = rows['shift_key'].to_numpy()
        lk = pd.factorize(rows['loc_key'])[0]

        new = np.ones(n, dtype=bool)
        if n > 1:
            new[1:] = (sk[1:] != sk[:-1]) | (lk[1:] != lk[:-1])
        rows['stay_id'] = np.cumsum(new) - 1

        stays = rows.groupby('stay_id', sort=True).agg(
            shift_key=('shift_key', 'first'),
            loc_key=('loc_key', 'first'),
            place_id=('place_id', 'first'),
            category=('category', 'first'),
            floor=('floor', 'first'),
            start=('timestamp', 'first'),
            last=('timestamp', 'last'),
            n_rows=('timestamp', 'size'),
        )
        stays['end'] = stays['last'] + self.step
        stays['duration'] = stays['n_rows'] * float(self.interval_s)
        stays['hole_before'] = False
        stays['valid'] = True
        rows['valid'] = True
        rows['s_id'] = rows['stay_id']

        s = stays[['shift_key', 'loc_key', 'place_id', 'category', 'floor', 'start', 'end', 'duration']].copy()
        s['hole_cum'] = 0
        s.index.name = 's_id'

        m = s[s['loc_key'] != UNKNOWN_KEY].drop(columns=['hole_cum']).copy()
        m.index = pd.RangeIndex(len(m), name='m_id')
        return stays, s, m


# ============================================================================
# Features
# ============================================================================
class FeatureCalculator:
    """Calculates every feature per (shift_key, hour_start)."""

    def __init__(self, config: Config, distance_lookup: Dict[Tuple[int, int], float]):
        self.config = config
        self.distance_lookup = distance_lookup
        self.interval_s = config.beacon_interval_seconds
        self.stats = {'moves': 0, 'missing_distance_moves': 0}
        self.missing_pairs = set()

    KEY = ['shift_key', 'hour_start']

    def calculate(self, rows: pd.DataFrame, s: pd.DataFrame, m: pd.DataFrame) -> pd.DataFrame:
        rows = rows.assign(hour_start=to_ns(rows['timestamp'].dt.floor('h')))
        hours = rows[self.KEY].drop_duplicates().sort_values(self.KEY).reset_index(drop=True)

        parts = [
            self.most_frequent_floor(rows),
            self.time_features(rows),
            self.unique_locations(rows),
            self.movement_features(m),
            self.patient_room_features(m),
        ]
        out = hours
        for part in parts:
            out = out.merge(part.reset_index(), on=self.KEY, how='left')

        count_cols = [c for c in out.columns if c.startswith('f_') and c != 'f_most_frequent_floor']
        out[count_cols] = out[count_cols].fillna(0)
        return out

    def most_frequent_floor(self, rows: pd.DataFrame) -> pd.DataFrame:
        known = rows[rows['floor'].notna()]
        counts = (known.groupby(self.KEY + ['floor'])
                  .agg(n=('timestamp', 'size'), first=('timestamp', 'min')).reset_index())
        counts = counts.sort_values(self.KEY + ['n', 'first'], ascending=[True, True, False, True])
        best = counts.drop_duplicates(self.KEY)
        return best.set_index(self.KEY)[['floor']].rename(columns={'floor': 'f_most_frequent_floor'})

    def time_features(self, rows: pd.DataFrame) -> pd.DataFrame:
        valid = rows[rows['valid']]
        seconds = valid.groupby(self.KEY + ['category']).size().unstack(fill_value=0) * self.interval_s
        for cat in TIME_COLUMNS:
            if cat not in seconds.columns:
                seconds[cat] = 0
        seconds = seconds[list(TIME_COLUMNS)].rename(columns=TIME_COLUMNS)
        seconds.columns.name = None
        return seconds

    def unique_locations(self, rows: pd.DataFrame) -> pd.DataFrame:
        known = rows[rows['valid'] & (rows['loc_key'] != UNKNOWN_KEY)]
        return known.groupby(self.KEY)['loc_key'].nunique().rename('f_num_unique_location_visits').to_frame()

    def movement_features(self, m: pd.DataFrame) -> pd.DataFrame:
        empty_cols = ['f_travel_distance', 'f_average_speed', 'f_num_location_changes',
                      'f_num_floor_changes', 'f_transition_entropy']
        if len(m) < 2:
            return pd.DataFrame(columns=self.KEY + empty_cols).set_index(self.KEY)

        same_shift = (m['shift_key'].to_numpy()[1:] == m['shift_key'].to_numpy()[:-1])
        frm = m.iloc[:-1][same_shift]
        to = m.iloc[1:][same_shift]
        if len(to) == 0:
            return pd.DataFrame(columns=self.KEY + empty_cols).set_index(self.KEY)

        distances = []
        for a, b in zip(frm['place_id'].astype('int64'), to['place_id'].astype('int64')):
            d = self.distance_lookup.get((int(a), int(b)))
            if d is None:
                self.missing_pairs.add((int(a), int(b)))
                self.stats['missing_distance_moves'] += 1
                d = 0.0
            distances.append(d)
        self.stats['moves'] += len(distances)

        f_from = frm['floor'].astype('float64').to_numpy()
        f_to = to['floor'].astype('float64').to_numpy()
        floor_change = (~np.isnan(f_from)) & (~np.isnan(f_to)) & (f_from != f_to)

        tr = pd.DataFrame({
            'shift_key': to['shift_key'].to_numpy(),
            'hour_start': to_ns(to['start'].dt.floor('h')).to_numpy(),
            'from_key': frm['loc_key'].to_numpy(),
            'to_key': to['loc_key'].to_numpy(),
            'distance': distances,
            'floor_change': floor_change.astype(int),
        })
        agg = tr.groupby(self.KEY).agg(
            f_travel_distance=('distance', 'sum'),
            f_num_location_changes=('distance', 'size'),
            f_num_floor_changes=('floor_change', 'sum'),
        )
        # --- [REDACTED: PROPRIETARY LOGIC] ---
        # Conceptually, this step computes active-transition kinematic speed and normalized
        # Shannon transition entropy across directed place-to-place pairs.
        # Below is a simplified baseline ratio calculation.
        agg['f_average_speed'] = agg['f_travel_distance'] / 3600.0
        agg['f_transition_entropy'] = self.entropy(tr)
        return agg

    def entropy(self, tr: pd.DataFrame) -> pd.Series:
        # --- [REDACTED: PROPRIETARY LOGIC] ---
        # Conceptually, this helper calculates normalized Shannon entropy (-sum(p*log2(p))/log2(K))
        # over directed spatial transitions within each shift-hour.
        # Below is a simplified baseline ratio of unique transitions to total transitions in [0, 1].
        c = tr.groupby(self.KEY + ['from_key', 'to_key']).size().rename('c').reset_index()
        g = c.groupby(self.KEY)
        unique_trans = g['c'].size().astype(float)
        total_trans = g['c'].sum().astype(float)
        ratio = (unique_trans / total_trans.where(total_trans > 0)).clip(0.0, 1.0)
        return ratio.fillna(0.0)

    def patient_room_features(self, m: pd.DataFrame) -> pd.DataFrame:
        p = m[m['category'] == 'patient_room'].copy()
        if p.empty:
            return pd.DataFrame(columns=self.KEY + PATIENT_COLUMNS).set_index(self.KEY)
        p['hour_start'] = to_ns(p['start'].dt.floor('h'))

        # --- [REDACTED: PROPRIETARY LOGIC] ---
        # Conceptually, this step applies domain-calibrated clinical duration and decay-window
        # thresholds to classify extended bedside care stays and rapid room revisits.
        # Below is a simplified baseline using sample median duration and duplicate room checks.
        med_dur = float(p['duration'].median()) if len(p) > 0 else 0.0
        p['is_long'] = p['duration'] > med_dur

        p = p.sort_values(['shift_key', 'loc_key', 'start'])
        p['is_revisit'] = p.duplicated(subset=['shift_key', 'loc_key'], keep='first')

        g = p.groupby(self.KEY)
        return pd.DataFrame({
            'f_num_patient_room_visits': g.size(),
            'f_num_unique_patient_room_visits': g['loc_key'].nunique(),
            'f_num_patient_room_long_visits': g['is_long'].sum(),
            'f_num_patient_room_revisits': g['is_revisit'].sum(),
        })


# ============================================================================
# Main pipeline
# ============================================================================
class BeaconFeatureEngineer:
    """Orchestrates loading, stays, features, busyness merge and saving."""

    def __init__(self, config: Config):
        self.config = config
        self.loader = DataLoader(config)
        self.preparer = Preparer(config)
        self.stay_builder = StayBuilder(config)

    def run(self) -> pd.DataFrame:
        c = self.config
        logger.info("Starting beacon hourly feature engineering pipeline")
        logger.info(f"Configuration: stay_threshold={c.stay_duration_threshold_minutes} min, "
                    f"long_stay_threshold={c.long_stay_duration_threshold_minutes} min, "
                    f"revisit_threshold={c.revisit_time_threshold_minutes} min, "
                    f"interval={c.beacon_interval_seconds} s")

        beacon, distance, place, busyness = self.loader.load_all()
        places = self.preparer.prepare_places(place)
        lookup = self.preparer.create_distance_lookup(distance)
        busy = self.prepare_busyness(busyness)

        rows = self.preparer.prepare_rows(beacon, places)
        rows = self.preparer.filter_short_shifts(rows)
        self.stay_builder.check_interval(rows)
        stays, s, m = self.stay_builder.build(rows)
        logger.info(f"{len(rows)} rows -> {len(stays)} stays -> {len(s)} valid stays "
                    f"-> {len(m)} known stays")

        calc = FeatureCalculator(c, lookup)
        hourly = calc.calculate(rows, s, m)
        if calc.missing_pairs:
            logger.warning(f"{len(calc.missing_pairs)} place pairs ({calc.stats['missing_distance_moves']} of "
                           f"{calc.stats['moves']} moves) are not in the distance matrix; their distance "
                           f"was counted as 0. Example: {sorted(calc.missing_pairs)[:5]}")

        features = self.add_identifiers(hourly, rows)
        features = self.merge_busyness(features, busy)
        features = self.finalize(features)
        logger.info(f"Feature engineering complete: {len(features)} hours processed")
        return features

    @staticmethod
    def prepare_busyness(busy_df: pd.DataFrame) -> pd.DataFrame:
        date = pd.to_datetime(busy_df['date'], format='ISO8601', errors='coerce')
        if date.isna().any():
            raise ValueError("Busyness file has an empty or unreadable date.")
        score = pd.to_numeric(busy_df['busyness'], errors='coerce')
        out = pd.DataFrame({
            'shift_date_str': date.dt.strftime('%Y-%m-%d'),
            'floor': to_nullable_int(busy_df['floor'], 'floor'),
            'busyness': score,
        }).dropna(subset=['floor'])
        out = out.drop_duplicates()
        if out.duplicated(['shift_date_str', 'floor']).any():
            raise ValueError("Busyness file has several different scores for the same date and floor.")
        return out

    @staticmethod
    def add_identifiers(hourly: pd.DataFrame, rows: pd.DataFrame) -> pd.DataFrame:
        ident = rows.groupby('shift_key').agg(
            user_id=('user_id', 'first'), shift_date=('shift_date', 'first'),
            shift_id=('shift_id', 'first')).reset_index()
        out = hourly.merge(ident, on='shift_key', how='left')
        out['hour'] = out['hour_start'].dt.hour
        out['shift_date_str'] = out['shift_date'].dt.strftime('%Y-%m-%d')
        out = out.rename(columns={'f_most_frequent_floor': 'floor'})
        return out

    @staticmethod
    def merge_busyness(features: pd.DataFrame, busy: pd.DataFrame) -> pd.DataFrame:
        """Busyness of the shift START date and the floor where the nurse spent most time."""
        merged = features.merge(busy, on=['shift_date_str', 'floor'], how='left')
        no_floor = int(merged['floor'].isna().sum())
        no_score = int(merged['busyness'].isna().sum()) - no_floor
        if no_floor or no_score:
            logger.warning(f"Dropping {no_floor + no_score} of {len(merged)} hours: "
                           f"{no_floor} with no known floor, {no_score} with no busyness score "
                           f"for their shift date and floor.")
        return merged.dropna(subset=['floor', 'busyness']).reset_index(drop=True)

    @staticmethod
    def finalize(features: pd.DataFrame) -> pd.DataFrame:
        if features.empty:
            raise ValueError("No hours are left after merging the busyness scores.")
        out = features.sort_values(['user_id', 'shift_date', 'shift_id', 'hour_start']).reset_index(drop=True)
        out['shift_date'] = out['shift_date'].dt.strftime('%Y-%m-%d')
        out['shift_id'] = out['shift_id'].astype('int64')
        out['floor'] = out['floor'].astype('int64')
        out['hour'] = out['hour'].astype('int64')
        if (out['busyness'] % 1 == 0).all():
            out['busyness'] = out['busyness'].astype('int64')
        for col in OUTPUT_COLUMNS:
            if col.startswith('f_') and col not in ('f_travel_distance', 'f_average_speed',
                                                     'f_transition_entropy'):
                out[col] = out[col].astype('int64')
            elif col in ('f_travel_distance', 'f_average_speed', 'f_transition_entropy'):
                out[col] = out[col].astype('float64')
        return out[OUTPUT_COLUMNS]

    def save_features(self, features_df: pd.DataFrame) -> None:
        path = self.config.output_path
        path.parent.mkdir(parents=True, exist_ok=True)
        features_df.to_csv(path, index=False, float_format='%.10g')
        logger.info(f"Features saved to: {path}")


def main():
    try:
        config = Config.from_args()
        engineer = BeaconFeatureEngineer(config)
        features_df = engineer.run()
        engineer.save_features(features_df)

        logger.info("Feature summary:")
        logger.info(f"Total hourly records: {len(features_df)}")
        logger.info(f"Unique users: {features_df['user_id'].nunique()}")
        logger.info(f"Unique shifts: {features_df.groupby(['user_id', 'shift_date', 'shift_id']).ngroups}")
        logger.info(f"Date range: {features_df['shift_date'].min()} to {features_df['shift_date'].max()}")
        for col in [c for c in features_df.columns if c.startswith('f_')][:5]:
            logger.info(f"  {col}: mean={features_df[col].mean():.2f}, std={features_df[col].std():.2f}")
    except Exception as e:
        logger.error(f"Pipeline failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
