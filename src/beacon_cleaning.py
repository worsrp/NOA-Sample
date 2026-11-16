"""Step 2: Beacon Data Cleaning Script.

Resolves multi-beacon timestamp conflicts via neighboring temporal walls and RSSI tie-breaking,
then fills short unknown location gaps within the same nurse shift.
Defaults and station group mappings are loaded from config.yaml / .env.
"""
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional
import argparse
import json
import logging
import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config_loader import load_config

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


REQUIRED_COLUMNS = ['user_id', 'timestamp', 'place_id', 'rssi', 'shift_date']
CORE_COLUMNS = {'user_id', 'timestamp', 'place_id', 'rssi', 'shift_date', 'shift_id'}
TRACE_COLUMNS = ['is_interpolated', 'resolved_by']


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def to_ns(series: pd.Series) -> pd.Series:
    """Force nanosecond resolution (pandas 3 may infer a coarser unit)."""
    return series.astype('datetime64[ns]')


def to_nullable_int(series: pd.Series, name: str) -> pd.Series:
    num = pd.to_numeric(series, errors='coerce')
    bad = series.notna() & num.isna()
    if bad.any():
        raise ValueError(f"Column '{name}' has non-numeric values, e.g. {series[bad].iloc[0]!r}")
    fractional = num.notna() & ((num % 1) != 0)
    if fractional.any():
        raise ValueError(f"Column '{name}' must hold whole numbers, e.g. {num[fractional].iloc[0]!r}")
    return num.astype('Int64')


def to_rssi(series: pd.Series) -> pd.Series:
    """Whole numbers stay integers (Int64); if any decimal exists, use float."""
    num = pd.to_numeric(series, errors='coerce')
    bad = series.notna() & num.isna()
    if bad.any():
        logger.warning(f"{int(bad.sum())} rssi values are not numeric and were set to empty.")
    valid = num.dropna()
    if ((valid % 1) == 0).all():
        return num.astype('Int64')
    return num.astype('float64')


def get_default_station_groups() -> Dict[str, str]:
    cfg = load_config()
    return {str(k).strip(): str(v) for k, v in cfg.get('station_groups', {}).items()}


def load_station_groups(path: Optional[Path]) -> Dict[str, str]:
    if path is None:
        return get_default_station_groups()
    if not path.exists():
        raise FileNotFoundError(f"Station group file not found: {path}")
    with open(path, encoding='utf-8') as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError("Station group file must be a JSON object: {\"1010\": \"F1_NURSE_STATION\", ...}")
    return {str(k).strip(): str(v) for k, v in data.items()}


# ---------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------
@dataclass
class Config:
    input_path: Path
    output_path: Path
    max_missing_threshold: int = 1
    interval_freq: str = '30s'
    wall_max_minutes: float = 5.0
    station_groups: Dict[str, str] = field(default_factory=get_default_station_groups)

    def __post_init__(self):
        if self.max_missing_threshold < 0:
            raise ValueError("--max-missing must be 0 or larger")
        if pd.to_timedelta(self.interval_freq) <= pd.Timedelta(0):
            raise ValueError("--interval must be positive")
        if self.wall_max_minutes < 0:
            raise ValueError("--wall-max-minutes must be 0 or larger")

    @classmethod
    def from_args(cls) -> 'Config':
        cfg = load_config()
        p = cfg['resolved_paths']
        pipe_cfg = cfg.get('pipeline', {})

        parser = argparse.ArgumentParser(description='Beacon Data Cleaning Script')
        parser.add_argument('--itpr-beacon-data', type=Path,
                            default=Path(os.environ.get('INTERPOLATED_BEACON_PATH', p['interpolated_beacon'])),
                            help='Interpolated beacon CSV from beacon_interpolation.py')
        parser.add_argument('--output', type=Path,
                            default=Path(os.environ.get('CLEANED_BEACON_PATH', p['cleaned_beacon'])),
                            help='Output cleaned CSV')
        parser.add_argument('--max-missing', type=int,
                            default=int(pipe_cfg.get('max_missing_threshold', 1)),
                            help='Max consecutive unknown rows to fill (default: 1)')
        parser.add_argument('--interval', type=str,
                            default=str(pipe_cfg.get('interval_freq', '30s')),
                            help='Time slot length used by the interpolation step (default: 30s)')
        parser.add_argument('--wall-max-minutes', type=float,
                            default=float(pipe_cfg.get('wall_max_minutes', 5.0)),
                            help='When resolving conflicts, only use neighbouring single-place '
                                 'timestamps up to this many minutes away (default: 5)')
        parser.add_argument('--station-groups', type=Path, default=None,
                            help='Optional JSON file {"place_id": "group label"} that overrides '
                                 'the station groups in config.yaml')
        args = parser.parse_args()
        return cls(
            input_path=args.itpr_beacon_data,
            output_path=args.output,
            max_missing_threshold=args.max_missing,
            interval_freq=args.interval,
            wall_max_minutes=args.wall_max_minutes,
            station_groups=load_station_groups(args.station_groups),
        )


# ---------------------------------------------------------------------
# Beacon cleaner
# ---------------------------------------------------------------------
class BeaconCleaner:
    def __init__(self, config: Config):
        self.config = config
        self.step = pd.to_timedelta(config.interval_freq).as_unit('ns')
        self.stats: Dict[str, int] = {}

    def load_data(self) -> pd.DataFrame:
        path = self.config.input_path
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")
        logger.info(f"Loading data from: {path}")
        raw = pd.read_csv(path, dtype=str, keep_default_na=False, na_values=[''],
                          encoding='utf-8-sig')
        raw.columns = [str(c).strip() for c in raw.columns]
        return self.prepare(raw)

    def prepare(self, raw: pd.DataFrame) -> pd.DataFrame:
        """Check columns and convert types."""
        missing = [c for c in REQUIRED_COLUMNS if c not in raw.columns]
        if missing:
            raise ValueError(
                f"Input is missing required columns {missing}. "
                f"Run beacon_interpolation.py first (found: {list(raw.columns)})."
            )
        df = raw.copy()
        for col in TRACE_COLUMNS:
            if col in df.columns:
                logger.warning(f"Input already has a '{col}' column; it will be replaced.")
                df = df.drop(columns=[col])

        df['user_id'] = df['user_id'].str.strip()
        if df['user_id'].isna().any():
            raise ValueError("Input has rows with an empty user_id")

        ts = pd.to_datetime(df['timestamp'], format='ISO8601', errors='coerce')
        if ts.isna().any():
            raise ValueError("Input has rows with an unreadable timestamp")
        df['timestamp'] = to_ns(ts)

        sd = pd.to_datetime(df['shift_date'], format='ISO8601', errors='coerce')
        if sd.isna().any():
            raise ValueError("Input has rows with an empty or unreadable shift_date")
        df['shift_date'] = to_ns(sd).dt.normalize()

        df['place_id'] = to_nullable_int(df['place_id'], 'place_id')
        df['rssi'] = to_rssi(df['rssi'])
        for col in ('floor', 'shift_id'):
            if col in df.columns:
                df[col] = to_nullable_int(df[col], col)

        for col in self.place_columns(df):
            if df[col].dtype == object or str(df[col].dtype).startswith(('str', 'string')):
                df[col] = df[col].fillna('')

        df = df.sort_values(['user_id', 'timestamp'], kind='stable').reset_index(drop=True)
        self.stats['input_rows'] = len(df)
        self.stats['input_unknown'] = int(df['place_id'].isna().sum())
        logger.info(f"Loaded {len(df)} rows, {self.stats['input_unknown']} unknown locations")
        return df

    @staticmethod
    def place_columns(df: pd.DataFrame) -> List[str]:
        """Columns that describe the place (floor, place_name, ...). Filled together with place_id."""
        return [c for c in df.columns if c not in CORE_COLUMNS and c not in TRACE_COLUMNS]

    def resolve_multiple_location_conflicts(self, df: pd.DataFrame) -> pd.DataFrame:
        """Reduce every (user_id, timestamp) to one row."""
        logger.info("Resolving rows that share the same user and timestamp...")
        df = df.reset_index(drop=True).copy()
        df['_row'] = np.arange(len(df))
        df['_gid'] = df.groupby(['user_id', 'timestamp'], sort=False).ngroup()

        known = df['place_id'].notna()
        n_rows = df.groupby('_gid')['_row'].transform('size')
        n_places = df['place_id'].groupby(df['_gid']).transform('nunique')

        multi = n_rows > 1
        conflict = n_places >= 2
        self.stats['groups_total'] = int(df['_gid'].nunique())

        if not multi.any():
            logger.info("No timestamps with several rows.")
            out = df.drop(columns=['_row', '_gid'])
            out['resolved_by'] = ''
            return out

        # --- [REDACTED: PROPRIETARY LOGIC] ---
        # Conceptually, this step constructs forward and backward single-location temporal
        # "walls" within a domain-calibrated time window to arbitrate multi-beacon collisions
        # via a 3-tier priority score before falling back to RSSI signal strength.
        # Below is a simplified baseline tie-breaker sorting by known place_id and RSSI.
        self.stats['walls_too_far'] = 0
        df['_pri'] = np.where(~known, 1, 0)
        df['_conflict'] = conflict
        df['_rssi_sort'] = df['rssi'].astype('float64')

        candidates = df[multi].sort_values(
            ['_gid', '_pri', '_rssi_sort', '_row'], kind='stable', na_position='last'
        )
        chosen = candidates.drop_duplicates('_gid', keep='first').copy()
        chosen['resolved_by'] = np.where(
            chosen['_conflict'],
            'rssi_fallback',
            'same_place',
        )
        singles = df[~multi].copy()
        singles['resolved_by'] = ''

        out = pd.concat([singles, chosen], ignore_index=True)
        out = out.sort_values(['user_id', 'timestamp'], kind='stable').reset_index(drop=True)
        out = out.drop(columns=['_row', '_gid', '_pri', '_conflict', '_rssi_sort'])

        counts = chosen['resolved_by'].value_counts()
        for label in ('same_place', 'neighbor_match', 'rssi_fallback'):
            self.stats[label] = int(counts.get(label, 0))
        logger.info(f"Resolved {len(chosen)} timestamps with several rows "
                    f"(same_place={self.stats['same_place']}, "
                    f"neighbor_match={self.stats['neighbor_match']}, "
                    f"rssi_fallback={self.stats['rssi_fallback']}).")
        return out

    def normalized_place(self, place_id: pd.Series) -> pd.Series:
        """Text label used only to compare places. Nurse station beacons share one label."""
        # --- [REDACTED: PROPRIETARY LOGIC] ---
        # Conceptually, this helper maps co-located clinical sub-beacons into unified
        # virtual zone identifiers for boundary-matching interpolation.
        # Below is a simplified baseline identity string conversion.
        norm = pd.Series(np.nan, index=place_id.index, dtype=object)
        known = place_id.notna()
        if known.any():
            norm[known] = place_id[known].astype('int64').astype(str).astype(object)
        return norm

    def interpolate_unknowns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fill short unknown gaps within the same user and shift_date."""
        df = df.sort_values(['user_id', 'timestamp'], kind='stable').reset_index(drop=True).copy()
        n = len(df)
        if n == 0:
            df['is_interpolated'] = pd.Series(dtype=bool)
            return df

        # --- [REDACTED: PROPRIETARY LOGIC] ---
        # Conceptually, this step identifies short unknown dropout runs bounded before and
        # after by identical normalized virtual zones without timestamp holes.
        # Below is a simplified baseline bounded forward-fill within each (user_id, shift_date) segment.
        user_codes = pd.factorize(df['user_id'])[0]
        dates = df['shift_date'].to_numpy()
        changed = np.ones(n, dtype=bool)
        changed[1:] = (user_codes[1:] != user_codes[:-1]) | (dates[1:] != dates[:-1])
        seg = pd.Series(np.cumsum(changed), index=df.index)

        known = df['place_id'].notna()
        limit = max(int(self.config.max_missing_threshold), 1)
        ffilled_pid = df['place_id'].astype('float64').groupby(seg).ffill(limit=limit)
        mask = (~known) & ffilled_pid.notna()

        if mask.any():
            df.loc[mask, 'place_id'] = ffilled_pid[mask].round().astype('Int64')
            for col in self.place_columns(df):
                source = df[col].where(known).groupby(seg).ffill(limit=limit)
                df.loc[mask, col] = source[mask]

        df['is_interpolated'] = mask.to_numpy()
        self.stats['filled'] = int(mask.sum())
        self.stats['unknown_after'] = int(df['place_id'].isna().sum())
        return df


# ---------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------
class ProcessingPipeline:
    def __init__(self, config: Config):
        self.config = config
        self.cleaner = BeaconCleaner(config)

    def run(self) -> pd.DataFrame:
        logger.info("Starting Processing Pipeline...")
        df = self.cleaner.load_data()
        return self.process(df, save=True)

    def process(self, df: pd.DataFrame, save: bool = False) -> pd.DataFrame:
        n_input = len(df)
        n_slots = df[['user_id', 'timestamp']].drop_duplicates().shape[0]

        resolved = self.cleaner.resolve_multiple_location_conflicts(df)
        logger.info(f"Filling unknowns (max_missing={self.config.max_missing_threshold})...")
        out = self.cleaner.interpolate_unknowns(resolved)

        base = [c for c in out.columns if c not in TRACE_COLUMNS]
        out = out[base + TRACE_COLUMNS]

        self.check_output(out, resolved, n_slots)
        self.log_summary(n_input, out)
        if save:
            self.save(out)
        return out

    @staticmethod
    def check_output(out: pd.DataFrame, resolved: pd.DataFrame, expected_rows: int):
        """Sanity checks before saving."""
        if out.duplicated(['user_id', 'timestamp']).any():
            raise RuntimeError("Check failed: more than one row for some user_id and timestamp.")
        if len(out) != expected_rows:
            raise RuntimeError(f"Check failed: expected {expected_rows} rows (one per user_id "
                               f"and timestamp) but have {len(out)}.")
        before = resolved.sort_values(['user_id', 'timestamp'], kind='stable').reset_index(drop=True)
        was_known = before['place_id'].notna()
        if not (out.loc[was_known, 'place_id'].to_numpy() == before.loc[was_known, 'place_id'].to_numpy()).all():
            raise RuntimeError("Check failed: gap filling changed a place_id that was already known.")

    def log_summary(self, n_input: int, out: pd.DataFrame):
        s = self.cleaner.stats
        logger.info("---- Summary ----")
        logger.info(f"Input rows: {n_input}, output rows: {len(out)}")
        logger.info(f"Timestamps resolved: same_place={s.get('same_place', 0)}, "
                    f"neighbor_match={s.get('neighbor_match', 0)}, "
                    f"rssi_fallback={s.get('rssi_fallback', 0)}")
        if s.get('walls_too_far'):
            logger.info(f"Conflicts where a neighbour was beyond --wall-max-minutes and ignored: "
                        f"{s['walls_too_far']}")
        logger.info(f"Unknown place_id: {s.get('input_unknown')} in input -> "
                    f"{s.get('unknown_after')} after filling ({s.get('filled', 0)} filled)")

    def save(self, df: pd.DataFrame):
        path = self.config.output_path
        path.parent.mkdir(parents=True, exist_ok=True)
        out = df.copy()
        out['shift_date'] = out['shift_date'].dt.strftime('%Y-%m-%d')
        logger.info(f"Saving processed data to: {path}")
        out.to_csv(path, index=False, date_format='%Y-%m-%d %H:%M:%S', float_format='%.10g')
        logger.info("Done.")


def main():
    try:
        config = Config.from_args()
        ProcessingPipeline(config).run()
    except Exception as e:
        logger.error(f"Pipeline failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
