"""
run_history_reader.py
Streams one month's unfiltered game CSV row by row, applying domain and output filters and
computing full per-lag past-performance history for past-perf-metric EDA.

Latest changes: 17/08/26:
- Rewritten to stream row-by-row to avoid RAM crashes

Run:
    !python run_history_reader.py --date 2026-01
CLI:
    --date       Dump month, YYYY-MM. Required.
    --in-dir     Folder holding game_unfiltered_{date}.csv. Default: GAME_DIR (config.py).
    --out-dir    Folder to write game_history_{date}.csv into. Default: GAME_DIR (config.py).
"""

import os
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from scripts.config import GAME_DIR

import argparse
import csv
import re
import shutil
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timedelta

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

MIN_FREE_DISK_GB = 20.0

PROGRESS_EVERY_N_ROWS = 1_000_000

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

# (a) PARSING

def parse_row(row: dict[str, str]) -> dict:
    """Converts one unfiltered-CSV row's string fields to typed values."""
    return {
        'game_id': row['game_id'],
        'datetime': datetime.strptime(row['datetime'], '%Y-%m-%d %H:%M:%S'),
        'white': row['white'], 'black': row['black'],
        'white_elo': int(row['white_elo']), 'black_elo': int(row['black_elo']),
        'white_title': row['white_title'], 'black_title': row['black_title'],
        'white_rating_diff': int(row['white_rating_diff']) if row['white_rating_diff'] else None,
        'black_rating_diff': int(row['black_rating_diff']) if row['black_rating_diff'] else None,
        'result': float(row['result']), 'speed': row['speed'], 'time_control': row['time_control'],
        'termination': row['termination'], 'rated': row['rated'] == 'True',
        'eco': row['eco'], 'ply_count': int(row['ply_count']), 'clock_ok': row['clock_ok'] == 'True',
    }

# (b) DOMAIN / OUTPUT FILTERING

def passes_domain_filter(core: dict, config: 'DomainFilterConfig') -> bool:
    """Returns whether a parsed row is eligible to contribute to history."""
    if config.rated_only and not core['rated']:
        return False
    if core['speed'] not in config.time_controls:
        return False
    if config.remove_bots:
        if core['white_title'].upper() == 'BOT' or core['black_title'].upper() == 'BOT':
            return False
    if core['white'] == core['black']:
        return False
    return True

def passes_output_filter(core: dict, config: 'OutputFilterConfig') -> bool:
    """Returns whether a parsed row passes the output-quality filters."""
    if core['ply_count'] < config.min_plies:
        return False
    if core['termination'] not in config.terminations:
        return False
    if config.valid_clock and not core['clock_ok']:
        return False
    if not config.tails:
        if core['white_elo'] < config.min_elo or core['black_elo'] < config.min_elo:
            return False
        if config.max_elo is not None:
            if core['white_elo'] >= config.max_elo or core['black_elo'] >= config.max_elo:
                return False
    return True

# (c) HISTORY COMPUTATION

def _prune(dq: deque, now: datetime) -> None:
    """Evicts entries older than HISTORY_MAX_DAYS from dq, in place."""
    cutoff = now - timedelta(days=HISTORY_MAX_DAYS)
    while dq and dq[0][0] < cutoff:
        dq.popleft()

def _lag_features(dq: deque | None, now: datetime) -> dict:
    """Returns per-lag result/hours-since/elo-gain (1..HISTORY_MAX_GAMES) plus prev/hours_since/
    has_history/last_opponent, for an already-pruned deque."""
    feats: dict = {}
    for lag in range(1, HISTORY_MAX_GAMES + 1):
        feats[f'past_result_{lag}'] = None
        feats[f'past_hours_since_{lag}'] = None
        feats[f'past_elo_gain_{lag}'] = None
    feats['prev'] = None
    feats['hours_since'] = None
    feats['has_history'] = False
    feats['last_opponent'] = None

    if not dq:
        return feats

    entries = list(dq)[::-1]
    for lag, (past_dt, past_result, past_opp, past_gain) in enumerate(entries, start=1):
        feats[f'past_result_{lag}'] = past_result
        feats[f'past_hours_since_{lag}'] = (now - past_dt).total_seconds() / 3600.0
        feats[f'past_elo_gain_{lag}'] = past_gain

    feats['prev'] = entries[0][1]
    feats['hours_since'] = (now - entries[0][0]).total_seconds() / 3600.0
    feats['has_history'] = True
    feats['last_opponent'] = entries[0][2]
    return feats

# (d) ROW ASSEMBLY

def _build_row(core: dict, white_feats: dict, black_feats: dict, rematch: bool) -> dict:
    """Assembles one output CSV row from parsed core fields and both players' lag features."""
    row = {
        'game_id': core['game_id'],
        'datetime': core['datetime'].strftime('%Y-%m-%d %H:%M:%S'),
        'white': core['white'], 'black': core['black'],
        'white_elo': core['white_elo'], 'black_elo': core['black_elo'],
        'white_title': core['white_title'], 'black_title': core['black_title'],
        'white_rating_diff': core['white_rating_diff'] if core['white_rating_diff'] is not None else '',
        'black_rating_diff': core['black_rating_diff'] if core['black_rating_diff'] is not None else '',
        'result': core['result'], 'speed': core['speed'], 'time_control': core['time_control'],
        'termination': core['termination'], 'eco': core['eco'], 'ply_count': core['ply_count'],
        'prev_white': white_feats['prev'] if white_feats['prev'] is not None else '',
        'prev_black': black_feats['prev'] if black_feats['prev'] is not None else '',
        'hours_since_white': white_feats['hours_since'] if white_feats['hours_since'] is not None else '',
        'hours_since_black': black_feats['hours_since'] if black_feats['hours_since'] is not None else '',
        'has_history_white': white_feats['has_history'], 'has_history_black': black_feats['has_history'],
        'rematch': rematch,
    }
    for lag in range(1, HISTORY_MAX_GAMES + 1):
        for color, feats in (('white', white_feats), ('black', black_feats)):
            for field in ('past_result', 'past_hours_since', 'past_elo_gain'):
                value = feats[f'{field}_{lag}']
                row[f'{color}_{field}_{lag}'] = value if value is not None else ''
    return row

# (e) OUTPUT I/O

def _check_disk_space(path: str, required_gb: float) -> None:
    """Raises if path's filesystem has less than required_gb free."""
    os.makedirs(path, exist_ok=True)
    free_gb = shutil.disk_usage(path).free / (1024 ** 3)
    if free_gb < required_gb:
        raise RuntimeError(
            f'Not enough free disk space at {path}: {free_gb:.1f}GB free, need at least {required_gb:.1f}GB.'
        )

# (f) STREAMED EXTRACTION

def run(input_path: str, output_path: str) -> dict:
    """Streams input_path row by row, applying domain/output filters and computing windowed
    per-lag history, writing kept rows to a tmp file that's atomically renamed once done."""
    domain_config = DomainFilterConfig(
        time_controls=set(TIME_CONTROLS), rated_only=RATED_ONLY, remove_bots=REMOVE_BOTS,
    )
    output_config = OutputFilterConfig(
        min_plies=MIN_PLIES, terminations=set(TERMINATIONS), valid_clock=VALID_CLOCK,
        tails=TAILS, min_elo=MIN_ELO, max_elo=MAX_ELO,
    )

    by_player: dict[str, deque] = {}
    n_loaded = 0
    n_domain_eligible = 0
    n_written = 0
    order_violations = 0
    last_seen_dt = None

    output_tmp = output_path + '.tmp'
    with open(input_path, newline='', encoding='utf-8') as in_f, \
            open(output_tmp, 'w', newline='', encoding='utf-8') as out_f:
        reader = csv.DictReader(in_f)
        writer = csv.DictWriter(out_f, fieldnames=HISTORY_CSV_FIELDNAMES)
        writer.writeheader()

        for row in reader:
            n_loaded += 1
            core = parse_row(row)

            if not passes_domain_filter(core, domain_config):
                continue
            n_domain_eligible += 1

            dt = core['datetime']
            if last_seen_dt is not None and dt < last_seen_dt:
                order_violations += 1
            else:
                last_seen_dt = dt

            w, b = core['white'], core['black']

            w_dq = by_player.get(w)
            if w_dq is not None:
                _prune(w_dq, dt)
            white_feats = _lag_features(w_dq, dt)

            b_dq = by_player.get(b)
            if b_dq is not None:
                _prune(b_dq, dt)
            black_feats = _lag_features(b_dq, dt)

            rematch = (white_feats['has_history'] and white_feats['last_opponent'] == b
                       and black_feats['has_history'] and black_feats['last_opponent'] == w)

            if passes_output_filter(core, output_config):
                writer.writerow(_build_row(core, white_feats, black_feats, rematch))
                n_written += 1

            w_result = core['result']
            b_result = 1.0 - core['result'] if core['result'] != 0.5 else 0.5
            w_dq = by_player.setdefault(w, deque(maxlen=HISTORY_MAX_GAMES))
            _prune(w_dq, dt)
            w_dq.append((dt, w_result, b, core['white_rating_diff']))
            b_dq = by_player.setdefault(b, deque(maxlen=HISTORY_MAX_GAMES))
            _prune(b_dq, dt)
            b_dq.append((dt, b_result, w, core['black_rating_diff']))

            if n_loaded % PROGRESS_EVERY_N_ROWS == 0:
                print(f'Loaded: {n_loaded:,}, domain-eligible: {n_domain_eligible:,}, kept: {n_written:,}')

    os.replace(output_tmp, output_path)

    print(f'Loaded: {n_loaded:,}, domain-eligible: {n_domain_eligible:,}, kept: {n_written:,}')
    if order_violations:
        print(f'WARNING: {order_violations:,} row(s) had a datetime earlier than a preceding row. '
              f'History features assume ascending order; investigate if this count is large.')

    return {'n_loaded': n_loaded, 'n_domain_eligible': n_domain_eligible,
            'n_written': n_written, 'order_violations': order_violations}

# (g) CLI

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
        description='Streams one month\'s unfiltered game CSV and computes full per-lag past-'
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
    """Reads args, validates config, streams the unfiltered CSV, writing the history CSV."""
    _validate_config()
    args = parse_args(argv)
    os.makedirs(args.out_dir, exist_ok=True)
    _check_disk_space(args.out_dir, MIN_FREE_DISK_GB)

    input_path = os.path.join(args.in_dir, f'game_unfiltered_{args.date}.csv')
    output_path = os.path.join(args.out_dir, f'game_history_{args.date}.csv')

    if not os.path.exists(input_path):
        sys.exit(f'No input file at {input_path} -- run run_reader_unfiltered.py --date {args.date} first.')
    if os.path.getsize(input_path) == 0:
        sys.exit(f"{input_path} is empty (0 bytes) -- run_reader_unfiltered.py likely didn't complete "
                  f'for {args.date}. Check its output for errors and rerun it for this month.')

    print(f'Reading: {input_path}')
    run(input_path, output_path)
    print(f'Done -> {output_path}')

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