"""
run_history_reader.py
Loads one month's unfiltered game CSV, applies domain and output filters, and computes full
per-lag past-performance history for past-perf-metric EDA.

Latest changes: 17/08/26:
- Initial commit

Run:
    !python run_history_reader.py --date 2026-01
CLI:
    --date       Dump month, YYYY-MM. Required.
    --in-dir     Folder holding game_unfiltered_{date}.csv. Default: GAME_DIR.
    --out-dir    Folder to write game_history_{date}.csv into. Default: GAME_DIR.
"""

import os
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from scripts.config import GAME_DIR

import argparse
import re
from collections import deque
from dataclasses import dataclass

import numpy as np
import pandas as pd

####################
# CONSTANTS
####################

DATE_RE = re.compile(r'^\d{4}-\d{2}$')

VALID_SPEEDS = {'Bullet', 'Blitz', 'Rapid', 'Classical'}

# Domain filters
RATED_ONLY = True
TIME_CONTROLS = ['Rapid']
REMOVE_BOTS = True

# Output-quality filters
MIN_PLIES = 12
TERMINATIONS = ['Normal', 'Time forfeit']
VALID_CLOCK = True
TAILS = True
MIN_ELO = 800
MAX_ELO = 2200

# Past-performance / history window
HISTORY_MAX_GAMES = 10
HISTORY_MAX_DAYS = 10

UNFILTERED_CSV_DTYPES = {
    'game_id': 'string',
    'white': 'string',
    'black': 'string',
    'white_elo': 'int32',
    'black_elo': 'int32',
    'white_title': 'category',
    'black_title': 'category',
    'white_rating_diff': 'Int64',
    'black_rating_diff': 'Int64',
    'result': 'float32',
    'speed': 'category',
    'time_control': 'category',
    'termination': 'category',
    'eco': 'category',
    'ply_count': 'int16',
}
# load 'datetime' with parse_dates=['datetime']
# load 'rated'/'clock_ok' with true_values=['True'], false_values=['False']

HISTORY_BASE_FIELDNAMES = [
    'game_id', 'datetime', 'white', 'black', 'white_elo', 'black_elo',
    'white_title', 'black_title', 'white_rating_diff', 'black_rating_diff',
    'result', 'speed', 'time_control', 'termination', 'eco', 'ply_count',
    'prev_white', 'prev_black', 'hours_since_white', 'hours_since_black',
    'has_history_white', 'has_history_black', 'rematch',
]
HISTORY_LAG_FIELDNAMES = [
    f'{color}_past_{field}_{lag}'
    for color in ('white', 'black')
    for field in ('result', 'hours_since', 'elo_gain')
    for lag in range(1, HISTORY_MAX_GAMES + 1)
]
HISTORY_CSV_FIELDNAMES = HISTORY_BASE_FIELDNAMES + HISTORY_LAG_FIELDNAMES

_HISTORY_BASE_DTYPES = {
    'game_id': 'string',
    'white': 'string',
    'black': 'string',
    'white_elo': 'int32',
    'black_elo': 'int32',
    'white_title': 'category',
    'black_title': 'category',
    'white_rating_diff': 'Int64',
    'black_rating_diff': 'Int64',
    'result': 'float32',
    'speed': 'category',
    'time_control': 'category',
    'termination': 'category',
    'eco': 'category',
    'ply_count': 'int16',
    'prev_white': 'float32',
    'prev_black': 'float32',
    'hours_since_white': 'float32',
    'hours_since_black': 'float32',
}
HISTORY_CSV_DTYPES = _HISTORY_BASE_DTYPES | {name: 'float32' for name in HISTORY_LAG_FIELDNAMES}
# load 'datetime' with parse_dates=['datetime']
HISTORY_CSV_BOOL_COLS = ['has_history_white', 'has_history_black', 'rematch']
# load with true_values=['True'], false_values=['False']

####################
# FUNCTIONS
####################

# (a) DOMAIN / OUTPUT FILTERING

def build_domain_mask(df: pd.DataFrame, config: 'DomainFilterConfig') -> np.ndarray:
    """Returns a boolean mask selecting rows eligible to contribute to history."""
    mask = np.ones(len(df), dtype=bool)
    if config.rated_only:
        mask = mask & df['rated'].to_numpy()
    mask = mask & df['speed'].isin(config.time_controls).to_numpy()
    if config.remove_bots:
        mask = mask & (df['white_title'].str.upper() != 'BOT').to_numpy()
        mask = mask & (df['black_title'].str.upper() != 'BOT').to_numpy()
    mask = mask & (df['white'] != df['black']).to_numpy()
    return mask

def build_output_mask(df: pd.DataFrame, config: 'OutputFilterConfig') -> np.ndarray:
    """Returns a boolean mask selecting rows to keep in the output CSV."""
    mask = df['termination'].isin(config.terminations).to_numpy()
    mask = mask & (df['ply_count'] >= config.min_plies).to_numpy()
    if config.valid_clock:
        mask = mask & df['clock_ok'].to_numpy()
    if not config.tails:
        mask = mask & (df['white_elo'] >= config.min_elo).to_numpy()
        mask = mask & (df['black_elo'] >= config.min_elo).to_numpy()
        if config.max_elo is not None:
            mask = mask & (df['white_elo'] < config.max_elo).to_numpy()
            mask = mask & (df['black_elo'] < config.max_elo).to_numpy()
    return mask

# (b) HISTORY COMPUTATION

def _check_sorted(df: pd.DataFrame, datetime_col: str = 'datetime') -> None:
    """Raises if df isn't sorted ascending by datetime_col."""
    if not df[datetime_col].is_monotonic_increasing:
        raise ValueError(f"df must be sorted ascending by '{datetime_col}' before computing history.")

def compute_history(df: pd.DataFrame, max_games: int, max_days: int) -> dict[str, np.ndarray]:
    """Computes per-lag result/hours-since/elo-gain history (1..max_games, day-windowed to max_days)
    plus prev/hours_since/has_history/rematch, using the same per-player-deque method as
    HistoryTracker in run_game_reader.py."""
    _check_sorted(df)
    n = len(df)
    dt = df['datetime'].to_numpy()
    white = df['white'].to_numpy()
    black = df['black'].to_numpy()
    result = df['result'].to_numpy(dtype='float64')
    white_elo_gain = df['white_rating_diff'].to_numpy(dtype='float64', na_value=np.nan)
    black_elo_gain = df['black_rating_diff'].to_numpy(dtype='float64', na_value=np.nan)

    by_player: dict[str, deque] = {}

    out = {name: np.full(n, np.nan, dtype='float32') for name in HISTORY_LAG_FIELDNAMES}
    prev_white = np.full(n, np.nan, dtype='float32')
    prev_black = np.full(n, np.nan, dtype='float32')
    hours_since_white = np.full(n, np.nan, dtype='float32')
    hours_since_black = np.full(n, np.nan, dtype='float32')
    has_history_white = np.zeros(n, dtype=bool)
    has_history_black = np.zeros(n, dtype=bool)
    rematch = np.zeros(n, dtype=bool)

    cutoff_delta = np.timedelta64(max_days, 'D')

    def prune(dq: deque, now: np.datetime64) -> None:
        cutoff = now - cutoff_delta
        while dq and dq[0][0] < cutoff:
            dq.popleft()

    for i in range(n):
        now = dt[i]
        w, b = white[i], black[i]

        w_dq = by_player.get(w)
        if w_dq is not None:
            prune(w_dq, now)
        w_last_opp = None
        if w_dq:
            entries = list(w_dq)[::-1]
            for lag, (past_dt, past_result, past_opp, past_gain) in enumerate(entries, start=1):
                out[f'white_past_result_{lag}'][i] = past_result
                out[f'white_past_hours_since_{lag}'][i] = (now - past_dt) / np.timedelta64(1, 'h')
                out[f'white_past_elo_gain_{lag}'][i] = past_gain
            prev_white[i] = entries[0][1]
            hours_since_white[i] = (now - entries[0][0]) / np.timedelta64(1, 'h')
            has_history_white[i] = True
            w_last_opp = entries[0][2]

        b_dq = by_player.get(b)
        if b_dq is not None:
            prune(b_dq, now)
        b_last_opp = None
        if b_dq:
            entries = list(b_dq)[::-1]
            for lag, (past_dt, past_result, past_opp, past_gain) in enumerate(entries, start=1):
                out[f'black_past_result_{lag}'][i] = past_result
                out[f'black_past_hours_since_{lag}'][i] = (now - past_dt) / np.timedelta64(1, 'h')
                out[f'black_past_elo_gain_{lag}'][i] = past_gain
            prev_black[i] = entries[0][1]
            hours_since_black[i] = (now - entries[0][0]) / np.timedelta64(1, 'h')
            has_history_black[i] = True
            b_last_opp = entries[0][2]

        rematch[i] = (has_history_white[i] and w_last_opp == b
                       and has_history_black[i] and b_last_opp == w)

        w_result = result[i]
        b_result = 1.0 - result[i] if result[i] != 0.5 else 0.5
        w_dq = by_player.setdefault(w, deque(maxlen=max_games))
        prune(w_dq, now)
        w_dq.append((now, w_result, b, white_elo_gain[i]))
        b_dq = by_player.setdefault(b, deque(maxlen=max_games))
        prune(b_dq, now)
        b_dq.append((now, b_result, w, black_elo_gain[i]))

    out.update({
        'prev_white': prev_white, 'prev_black': prev_black,
        'hours_since_white': hours_since_white, 'hours_since_black': hours_since_black,
        'has_history_white': has_history_white, 'has_history_black': has_history_black,
        'rematch': rematch,
    })
    return out

# (c) CLI

def _validate_config() -> None:
    """Validates filter/history constants, exiting with a clear message if any are invalid."""
    if HISTORY_MAX_GAMES < 1:
        sys.exit('HISTORY_MAX_GAMES must be >= 1.')
    if HISTORY_MAX_DAYS < 1:
        sys.exit('HISTORY_MAX_DAYS must be >= 1.')
    unknown = set(TIME_CONTROLS) - VALID_SPEEDS
    if unknown:
        sys.exit(f'TIME_CONTROLS has unrecognised value(s) {sorted(unknown)}. '
                 f'Valid options: {sorted(VALID_SPEEDS)}.')
    if MIN_PLIES < 0:
        sys.exit('MIN_PLIES must be >= 0.')
    if MIN_ELO < 0:
        sys.exit('MIN_ELO must be >= 0.')
    if MAX_ELO is not None and MAX_ELO <= MIN_ELO:
        sys.exit('MAX_ELO must be > MIN_ELO (MAX_ELO is an exclusive upper bound), or None.')

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parses --date, --in-dir, --out-dir CLI arguments."""
    parser = argparse.ArgumentParser(
        description='Loads one month\'s unfiltered game CSV and computes full per-lag past-'
                     'performance history for past-perf-metric EDA.'
    )
    parser.add_argument('--date', required=True, help='Dump month, YYYY-MM.')
    parser.add_argument('--in-dir', default=GAME_DIR, help=f'Default: {GAME_DIR}')
    parser.add_argument('--out-dir', default=GAME_DIR, help=f'Default: {GAME_DIR}')
    args = parser.parse_args(argv)
    if not DATE_RE.match(args.date):
        parser.error(f'--date must be YYYY-MM, got: {args.date!r}')
    return args

def main(argv: list[str] | None = None) -> None:
    """Reads args, loads the unfiltered CSV, applies domain filtering, computes history,
    applies output filtering, writes the result."""
    _validate_config()
    args = parse_args(argv)
    os.makedirs(args.out_dir, exist_ok=True)

    input_path = os.path.join(args.in_dir, f'game_unfiltered_{args.date}.csv')
    output_final = os.path.join(args.out_dir, f'game_history_{args.date}.csv')
    output_tmp = output_final + '.tmp'

    if not os.path.exists(input_path):
        sys.exit(f'No input file at {input_path} -- run run_reader_unfiltered.py --date {args.date} first.')

    print(f'Reading: {input_path}')
    df = pd.read_csv(input_path, dtype=UNFILTERED_CSV_DTYPES, parse_dates=['datetime'],
                      true_values=['True'], false_values=['False'])
    df = df.sort_values('datetime', kind='mergesort').reset_index(drop=True)
    n_loaded = len(df)

    domain_config = DomainFilterConfig(
        time_controls=set(TIME_CONTROLS), rated_only=RATED_ONLY, remove_bots=REMOVE_BOTS,
    )
    domain_mask = build_domain_mask(df, domain_config)
    df = df[domain_mask].reset_index(drop=True)
    n_domain_eligible = len(df)

    print('Computing history...')
    history_cols = compute_history(df, HISTORY_MAX_GAMES, HISTORY_MAX_DAYS)
    for col, arr in history_cols.items():
        df[col] = arr

    output_config = OutputFilterConfig(
        min_plies=MIN_PLIES, terminations=set(TERMINATIONS), valid_clock=VALID_CLOCK,
        tails=TAILS, min_elo=MIN_ELO, max_elo=MAX_ELO,
    )
    output_mask = build_output_mask(df, output_config)
    df_out = df.loc[output_mask, HISTORY_CSV_FIELDNAMES]

    print(f'Loaded: {n_loaded:,}, domain-eligible: {n_domain_eligible:,}, kept: {len(df_out):,}')
    df_out.to_csv(output_tmp, index=False)
    os.replace(output_tmp, output_final)
    print(f'Done -> {output_final}')

####################
# CLASSES
####################

@dataclass
class DomainFilterConfig:
    """Domain filters: rated-only, time controls, bot removal."""
    time_controls: set
    rated_only: bool
    remove_bots: bool

@dataclass
class OutputFilterConfig:
    """Output filters: min plies, termination, clock, elo range."""
    min_plies: int
    terminations: set
    valid_clock: bool
    tails: bool
    min_elo: int
    max_elo: int | None


if __name__ == '__main__':
    main()