"""Step 1: Beacon Data Interpolation Script.

Maps raw indoor BLE beacon logs onto regular per-shift time grids (default 30s slots).
Paths and default parameters are loaded from config.yaml / .env and can be overridden via CLI flags.
"""
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple
import argparse
import logging
import os
import re
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


FINAL_COLUMNS = [
    'user_id', 'timestamp', 'floor', 'place_id', 'place_name',
    'place_name_en', 'rssi', 'shift_date', 'shift_id',
]

_TIME_RE = re.compile(r'(\d{1,2}):(\d{2})(?::(\d{2}))?\s*$')

NULL_TEXTS = ['', 'NULL', 'null', 'Null', 'NaN', 'nan', 'None']


# ============================================================================
# Small helpers
# ============================================================================
def to_ns(series: pd.Series) -> pd.Series:
    """Force nanosecond resolution (pandas 3 may infer a coarser unit)."""
    return series.astype('datetime64[ns]')


def to_nullable_int(series: pd.Series, name: str) -> pd.Series:
    """Convert text to a nullable integer column; reject non-whole numbers."""
    num = pd.to_numeric(series, errors='coerce')
    bad = series.notna() & num.isna()
    if bad.any():
        raise ValueError(
            f"Column '{name}' has non-numeric values, e.g. {series[bad].iloc[0]!r}"
        )
    fractional = num.notna() & ((num % 1) != 0)
    if fractional.any():
        raise ValueError(
            f"Column '{name}' must hold whole numbers, e.g. {num[fractional].iloc[0]!r}"
        )
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


def parse_time_of_day(series: pd.Series, label: str) -> pd.Series:
    """
    Read the time of day from values like '20:00:00' or '1900-01-01 20:00:00'.
    Any date part is ignored. Returns a timedelta (time since midnight).
    """
    text = series.astype(str).str.strip()
    parts = text.str.extract(_TIME_RE)
    hours = pd.to_numeric(parts[0], errors='coerce').astype('float64')
    minutes = pd.to_numeric(parts[1], errors='coerce').astype('float64')
    seconds = pd.to_numeric(parts[2], errors='coerce').astype('float64').fillna(0)
    bad = hours.isna() | minutes.isna() | (hours > 24) | (minutes > 59) | (seconds > 59)
    if bad.any():
        raise ValueError(f"{label}: cannot read a time of day from {text[bad].iloc[0]!r}")
    return pd.to_timedelta(hours * 3600 + minutes * 60 + seconds, unit='s').astype('timedelta64[ns]')


def time_text_to_seconds(text: str, label: str) -> int:
    match = _TIME_RE.fullmatch(text.strip())
    if not match:
        raise ValueError(f"{label} must look like HH:MM:SS, got {text!r}")
    h, m, s = int(match.group(1)), int(match.group(2)), int(match.group(3) or 0)
    return h * 3600 + m * 60 + s


# ============================================================================
# Configuration
# ============================================================================
@dataclass
class Config:
    """Settings for the interpolation step."""

    beacon_data_path: Path
    m_place_path: Path
    daily_shift_path: Path
    m_shift_path: Path
    output_path: Path

    interval_freq: str = '30s'

    # Period filter on shift_date (inclusive), format YYYY-MM-DD
    start_date: Optional[str] = None
    end_date: Optional[str] = None

    # Clock-time filter (HH:MM:SS), applied after shifts are assigned
    shift_start_time: Optional[str] = None
    shift_end_time: Optional[str] = None

    # Extra minutes around each planned shift when matching logs to shifts
    shift_buffer_minutes: float = 0.0

    def __post_init__(self):
        step = pd.to_timedelta(self.interval_freq)
        if step <= pd.Timedelta(0):
            raise ValueError("--interval must be positive")
        if self.shift_buffer_minutes < 0:
            raise ValueError("--shift-buffer-minutes must be 0 or larger")
        if self.start_date:
            pd.to_datetime(self.start_date, format='%Y-%m-%d')
        if self.end_date:
            pd.to_datetime(self.end_date, format='%Y-%m-%d')
        if self.start_date and self.end_date:
            if pd.to_datetime(self.start_date) > pd.to_datetime(self.end_date):
                raise ValueError("--start-date is later than --end-date")
        if bool(self.shift_start_time) != bool(self.shift_end_time):
            raise ValueError("--shift-start-time and --shift-end-time must be given together")
        if self.shift_start_time:
            time_text_to_seconds(self.shift_start_time, '--shift-start-time')
            time_text_to_seconds(self.shift_end_time, '--shift-end-time')

    @classmethod
    def from_args(cls) -> 'Config':
        cfg = load_config()
        p = cfg['resolved_paths']
        pipe_cfg = cfg.get('pipeline', {})

        parser = argparse.ArgumentParser(description='Beacon Data Interpolation Script')
        parser.add_argument('--beacon-data', type=Path,
                            default=Path(os.environ.get('RAW_BEACON_PATH', p['raw_beacon'])),
                            help='Raw beacon logs CSV (timestamp, receiver, user_name, rssi)')
        parser.add_argument('--m-place', type=Path,
                            default=Path(os.environ.get('M_PLACE_PATH', p['m_place'])),
                            help='Place master CSV (receiver, place_id, place_name, place_name_en, floor)')
        parser.add_argument('--daily-shift', type=Path,
                            default=Path(os.environ.get('T_DAILY_SHIFT_PATH', p['t_daily_shift'])),
                            help='Daily shift assignment CSV (user_id, shift_date, shift_id)')
        parser.add_argument('--m-shift', type=Path,
                            default=Path(os.environ.get('M_SHIFT_PATH', p['m_shift'])),
                            help='Shift master CSV (shift_id, start_time, end_time)')
        parser.add_argument('--output', type=Path,
                            default=Path(os.environ.get('INTERPOLATED_BEACON_PATH', p['interpolated_beacon'])),
                            help='Output interpolated CSV')
        parser.add_argument('--interval', type=str,
                            default=str(pipe_cfg.get('interval_freq', '30s')),
                            help='Time slot length (default: 30s)')
        parser.add_argument('--start-date', type=str, default=None,
                            help='First shift_date to keep, YYYY-MM-DD')
        parser.add_argument('--end-date', type=str, default=None,
                            help='Last shift_date to keep, YYYY-MM-DD')
        parser.add_argument('--shift-start-time', type=str, default=None,
                            help='Keep only logs at or after this clock time, HH:MM:SS')
        parser.add_argument('--shift-end-time', type=str, default=None,
                            help='Keep only logs at or before this clock time, HH:MM:SS '
                                 '(if earlier than the start time, the range wraps past midnight)')
        parser.add_argument('--shift-buffer-minutes', type=float,
                            default=float(pipe_cfg.get('shift_buffer_minutes', 0.0)),
                            help='Minutes added around each planned shift when matching logs '
                                 'to shifts (default: 0 = strict plan)')
        args = parser.parse_args()
        return cls(
            beacon_data_path=args.beacon_data,
            m_place_path=args.m_place,
            daily_shift_path=args.daily_shift,
            m_shift_path=args.m_shift,
            output_path=args.output,
            interval_freq=args.interval,
            start_date=args.start_date,
            end_date=args.end_date,
            shift_start_time=args.shift_start_time,
            shift_end_time=args.shift_end_time,
            shift_buffer_minutes=args.shift_buffer_minutes,
        )


# ============================================================================
# Data loading
# ============================================================================
class DataLoader:
    """Reads the four input files as text and checks their columns."""

    def __init__(self, config: Config):
        self.config = config

    @staticmethod
    def read_csv(path: Path, required_columns, label: str) -> pd.DataFrame:
        if not path.exists():
            raise FileNotFoundError(f"{label} file not found: {path}")
        logger.info(f"Loading {label}: {path}")
        df = pd.read_csv(path, dtype=str, keep_default_na=False, na_values=NULL_TEXTS,
                         encoding='utf-8-sig')
        df.columns = [str(c).strip() for c in df.columns]
        missing = [c for c in required_columns if c not in df.columns]
        if missing:
            raise ValueError(
                f"{label}: missing required columns {missing} (found: {list(df.columns)})"
            )
        return df

    def load_all(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        c = self.config
        beacon = self.read_csv(c.beacon_data_path,
                               ['timestamp', 'receiver', 'user_name', 'rssi'], 'beacon logs')
        m_place = self.read_csv(c.m_place_path,
                                ['receiver', 'place_id', 'place_name', 'place_name_en', 'floor'],
                                'place master')
        daily = self.read_csv(c.daily_shift_path,
                              ['user_id', 'shift_date', 'shift_id'], 'daily shift')
        m_shift = self.read_csv(c.m_shift_path,
                                ['shift_id', 'start_time', 'end_time'], 'shift master')
        return beacon, m_place, daily, m_shift


# ============================================================================
# Processing
# ============================================================================
class BeaconInterpolator:
    """Prepares logs, assigns shifts and builds the per-shift time grid."""

    def __init__(self, config: Config):
        self.config = config
        self.step = pd.to_timedelta(config.interval_freq).as_unit('ns')
        self.stats = {}

    def prepare_logs(self, beacon_df: pd.DataFrame) -> pd.DataFrame:
        """Clean types, drop unusable rows, apply a wide date margin."""
        logger.info("Preparing raw beacon logs...")
        df = pd.DataFrame({
            'receiver': beacon_df['receiver'].str.strip(),
            'user_id': beacon_df['user_name'].str.strip(),
            'rssi_raw': beacon_df['rssi'],
        })
        ts = pd.to_datetime(beacon_df['timestamp'], format='ISO8601', errors='coerce')
        if getattr(ts.dt, 'tz', None) is not None:
            ts = ts.dt.tz_localize(None)
        df['timestamp'] = to_ns(ts)
        self.stats['raw_rows'] = len(df)

        invalid = df['timestamp'].isna() | df['user_id'].isna() | df['receiver'].isna()
        if invalid.any():
            logger.warning(f"Dropping {int(invalid.sum())} rows with an unreadable timestamp, "
                           f"user_name or receiver.")
            df = df.loc[~invalid].copy()

        df['rssi'] = to_rssi(df['rssi_raw'])
        df = df.drop(columns=['rssi_raw'])

        if self.config.start_date:
            lo = pd.to_datetime(self.config.start_date, format='%Y-%m-%d') - pd.Timedelta(days=1)
            df = df[df['timestamp'] >= lo]
        if self.config.end_date:
            hi = pd.to_datetime(self.config.end_date, format='%Y-%m-%d') + pd.Timedelta(days=2)
            df = df[df['timestamp'] < hi]

        df = df.reset_index(drop=True)
        df['_order'] = np.arange(len(df), dtype='int64')
        return df

    def attach_places(self, logs: pd.DataFrame, m_place_df: pd.DataFrame) -> pd.DataFrame:
        master = pd.DataFrame({
            'receiver': m_place_df['receiver'].str.strip(),
            'floor': to_nullable_int(m_place_df['floor'], 'floor'),
            'place_id': to_nullable_int(m_place_df['place_id'], 'place_id'),
            'place_name': m_place_df['place_name'].fillna(''),
            'place_name_en': m_place_df['place_name_en'].fillna(''),
        })
        dup = master['receiver'].duplicated(keep=False)
        if dup.any():
            examples = sorted(master.loc[dup, 'receiver'].unique())[:10]
            raise ValueError(f"Place master has duplicate receivers, e.g. {examples}. "
                             f"Each receiver must appear once.")

        merged = logs.merge(master, on='receiver', how='left', indicator=True,
                            validate='many_to_one')
        unknown = merged['_merge'] == 'left_only'
        self.stats['unknown_receiver_rows'] = int(unknown.sum())
        if unknown.any():
            names = sorted(merged.loc[unknown, 'receiver'].unique())
            logger.warning(f"{int(unknown.sum())} log rows have a receiver that is not in the place "
                           f"master ({len(names)} receivers, e.g. {names[:10]}). "
                           f"They are kept with an empty place_id.")
        merged = merged.drop(columns=['_merge', 'receiver'])
        merged['place_name'] = merged['place_name'].fillna('')
        merged['place_name_en'] = merged['place_name_en'].fillna('')
        return merged

    def build_shift_windows(self, daily_df: pd.DataFrame, m_shift_df: pd.DataFrame) -> pd.DataFrame:
        """One planned window (start/end datetime) per assigned (user, shift_date)."""
        usable = (m_shift_df['shift_id'].notna() & m_shift_df['start_time'].notna()
                  & m_shift_df['end_time'].notna())
        if not usable.all():
            logger.info(f"Skipping {int((~usable).sum())} shift master rows with an empty "
                        f"shift_id, start_time or end_time.")
        m_shift_df = m_shift_df[usable]
        if m_shift_df.empty:
            raise ValueError("Shift master has no usable rows (shift_id, start_time, end_time).")
        master = pd.DataFrame({
            'shift_id': to_nullable_int(m_shift_df['shift_id'], 'shift_id'),
            'start_td': parse_time_of_day(m_shift_df['start_time'], 'shift master start_time'),
            'end_td': parse_time_of_day(m_shift_df['end_time'], 'shift master end_time'),
        })
        if master['shift_id'].duplicated().any():
            raise ValueError("Shift master has duplicate shift_id values")

        shift_date = pd.to_datetime(daily_df['shift_date'], format='ISO8601', errors='coerce')
        bad_date = shift_date.isna() & daily_df['shift_date'].notna()
        if bad_date.any():
            raise ValueError(f"Daily shift file has an unreadable shift_date, "
                             f"e.g. {daily_df.loc[bad_date, 'shift_date'].iloc[0]!r}")
        daily = pd.DataFrame({
            'user_id': daily_df['user_id'].str.strip(),
            'shift_date': to_ns(shift_date).dt.normalize(),
            'shift_id': to_nullable_int(daily_df['shift_id'], 'shift_id'),
        })
        before = len(daily)
        daily = daily.dropna().drop_duplicates()
        if len(daily) < before:
            logger.info(f"Ignored {before - len(daily)} daily shift rows that are empty or duplicated.")

        unknown_shift = ~daily['shift_id'].isin(master['shift_id'])
        if unknown_shift.any():
            logger.warning(f"{int(unknown_shift.sum())} daily assignments use a shift_id that is not "
                           f"in the shift master; they are ignored.")
            daily = daily[~unknown_shift]
        if daily.empty:
            raise ValueError("No usable shift assignments after reading the daily shift file.")

        win = daily.merge(master, on='shift_id', how='left', validate='many_to_one')
        overnight = (win['end_td'] <= win['start_td']).astype('int64')
        win['win_start'] = to_ns(win['shift_date'] + win['start_td'])
        win['win_end'] = to_ns(win['shift_date'] + win['end_td'] + pd.to_timedelta(overnight, unit='D'))
        win = win[['user_id', 'shift_date', 'shift_id', 'win_start', 'win_end']]
        win = win.sort_values(['user_id', 'win_start'], kind='stable').reset_index(drop=True)

        prev_end = win.groupby('user_id')['win_end'].shift()
        overlaps = int((win['win_start'] < prev_end).sum())
        if overlaps:
            logger.warning(f"{overlaps} planned shift windows overlap the previous shift of the "
                           f"same user. The later-starting shift wins where they overlap.")
        return win

    def assign_shifts(self, logs: pd.DataFrame, windows: pd.DataFrame) -> pd.DataFrame:
        """
        Give every log a shift_date (shift START date) and shift_id.
        1) Log inside a planned window -> that shift.
        2) Otherwise, if within the buffer of the nearest window -> that shift.
        3) Otherwise -> calendar date and empty shift_id.
        """
        # --- [REDACTED: PROPRIETARY LOGIC] ---
        # Conceptually, this step resolves off-schedule BLE pings near shift boundaries
        # using a bidirectional temporal buffer competition heuristic between adjacent
        # shift windows before falling back to calendar dates.
        # Below is a simplified baseline interval join using backward merge_asof.
        left = logs[['_order', 'user_id', 'timestamp']].sort_values('timestamp', kind='stable')
        left = left.reset_index(drop=True)
        right = windows.sort_values('win_start', kind='stable').reset_index(drop=True)

        prev = pd.merge_asof(left, right, left_on='timestamp', right_on='win_start',
                             by='user_id', direction='backward')

        ts = left['timestamp']
        inside = prev['win_end'].notna() & (ts <= prev['win_end'])

        shift_date = prev['shift_date'].where(inside, ts.dt.normalize())
        shift_id = prev['shift_id'].where(inside)

        result = pd.DataFrame({
            '_order': left['_order'],
            'shift_date': to_ns(shift_date),
            'shift_id': shift_id.astype('Int64'),
        })
        self.stats['shift_inside'] = int(inside.sum())
        self.stats['shift_by_buffer'] = 0
        self.stats['shift_fallback'] = int((~inside).sum())
        if self.stats['shift_fallback']:
            logger.warning(f"{self.stats['shift_fallback']} log rows fit no shift; they use their "
                           f"calendar date as shift_date and an empty shift_id.")
        return logs.merge(result, on='_order', how='left', validate='one_to_one')

    def apply_filters(self, logs: pd.DataFrame) -> pd.DataFrame:
        c = self.config
        df = logs
        if c.start_date:
            df = df[df['shift_date'] >= pd.to_datetime(c.start_date, format='%Y-%m-%d')]
        if c.end_date:
            df = df[df['shift_date'] <= pd.to_datetime(c.end_date, format='%Y-%m-%d')]

        if c.shift_start_time and c.shift_end_time:
            start = time_text_to_seconds(c.shift_start_time, '--shift-start-time')
            end = time_text_to_seconds(c.shift_end_time, '--shift-end-time')
            t = df['timestamp'].dt
            seconds = t.hour * 3600 + t.minute * 60 + t.second
            if start <= end:
                df = df[(seconds >= start) & (seconds <= end)]
            else:
                df = df[(seconds >= start) | (seconds <= end)]
        return df.reset_index(drop=True)

    def build_output(self, logs: pd.DataFrame) -> pd.DataFrame:
        """Round to slots, build the per-shift grid, merge logs onto it."""
        if logs.empty:
            raise ValueError("No log rows are left after filtering. Check the dates, "
                             "the shift files and the clock-time filter.")
        freq = self.config.interval_freq
        step_ns = int(self.step.value)

        df = logs.copy()
        df['slot'] = df['timestamp'].dt.floor(freq)
        df['shift_key'] = df['shift_id'].fillna(-1).astype('int64')
        keys = ['user_id', 'shift_date', 'shift_key']

        bounds = df.groupby(keys, sort=True)['slot'].agg(['min', 'max']).reset_index()
        first = to_ns(bounds['min']).to_numpy()
        n_slots = ((bounds['max'] - bounds['min']) // self.step).to_numpy().astype('int64') + 1
        total = int(n_slots.sum())
        logger.info(f"Building a grid of {total} slots for {len(bounds)} user-shift groups "
                    f"({freq} interval).")

        rep = np.repeat(np.arange(len(bounds)), n_slots)
        offset = np.arange(total, dtype='int64') - np.repeat(np.cumsum(n_slots) - n_slots, n_slots)
        slots = first[rep] + (offset * step_ns).astype('timedelta64[ns]')
        grid = bounds[keys].iloc[rep].reset_index(drop=True)
        grid['timestamp'] = slots

        data = df.drop(columns=['timestamp']).rename(columns={'slot': 'timestamp'})
        merged = grid.merge(data, on=keys + ['timestamp'], how='left')
        has_log = merged['_order'].notna()

        logged = (merged.loc[has_log, ['user_id', 'timestamp']].drop_duplicates()
                  .assign(_logged=True))
        merged = merged.merge(logged, on=['user_id', 'timestamp'], how='left')
        redundant = ~has_log & merged['_logged'].notna()
        if redundant.any():
            merged = merged[~redundant]

        merged['shift_id'] = merged['shift_key'].where(merged['shift_key'] >= 0).astype('Int64')
        merged['place_name'] = merged['place_name'].fillna('')
        merged['place_name_en'] = merged['place_name_en'].fillna('')

        merged = merged.sort_values(['user_id', 'timestamp', '_order'], kind='stable',
                                    na_position='first')
        out = merged[FINAL_COLUMNS]
        before = len(out)
        out = out.drop_duplicates().reset_index(drop=True)
        self.stats['exact_duplicates_dropped'] = before - len(out)
        self.stats['empty_slot_rows'] = int((out['place_id'].isna() & out['rssi'].isna()).sum())
        return out


# ============================================================================
# Pipeline
# ============================================================================
class InterpolationPipeline:
    def __init__(self, config: Config):
        self.config = config
        self.loader = DataLoader(config)
        self.interpolator = BeaconInterpolator(config)

    def run(self) -> pd.DataFrame:
        logger.info("Starting beacon data interpolation pipeline")
        beacon, m_place, daily, m_shift = self.loader.load_all()

        it = self.interpolator
        logs = it.prepare_logs(beacon)
        logs = it.attach_places(logs, m_place)
        windows = it.build_shift_windows(daily, m_shift)
        logs = it.assign_shifts(logs, windows)
        logs = it.apply_filters(logs)
        final_df = it.build_output(logs)

        self.log_summary(final_df)
        self.save_data(final_df)
        return final_df

    def log_summary(self, df: pd.DataFrame):
        s = self.interpolator.stats
        logger.info("---- Summary ----")
        logger.info(f"Raw log rows read: {s.get('raw_rows')}")
        logger.info(f"Rows with unknown receiver: {s.get('unknown_receiver_rows')}")
        logger.info(f"Log rows inside a planned shift: {s.get('shift_inside')}, "
                    f"matched by buffer: {s.get('shift_by_buffer')}, "
                    f"no shift (calendar-date fallback): {s.get('shift_fallback')}")
        logger.info(f"Exact duplicate rows dropped: {s.get('exact_duplicates_dropped')}")
        logger.info(f"Empty slot rows created: {s.get('empty_slot_rows')}")
        logger.info(f"Output rows: {len(df)} "
                    f"({df['user_id'].nunique()} users, "
                    f"{df[['user_id', 'shift_date']].drop_duplicates().shape[0]} user-shift-dates)")

    def save_data(self, df: pd.DataFrame):
        path = self.config.output_path
        path.parent.mkdir(parents=True, exist_ok=True)
        out = df.copy()
        out['shift_date'] = out['shift_date'].dt.strftime('%Y-%m-%d')
        logger.info(f"Saving {len(out)} rows to {path}")
        out.to_csv(path, index=False, date_format='%Y-%m-%d %H:%M:%S', float_format='%.10g')
        logger.info("Save complete.")


def main():
    try:
        config = Config.from_args()
        InterpolationPipeline(config).run()
    except Exception as e:
        logger.error(f"Pipeline failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
