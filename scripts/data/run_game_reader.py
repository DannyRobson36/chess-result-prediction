"""
run_game_reader.py
Streams lichess dump months into windowed-history eda/train/val/test game CSVs, each with a matching PGN-only cache.

Latest changes: 17/08/26:
- Include insufficient material in termination types

Run:
    !python run_game_reader.py
CLI:
    --in-dir      Folder holding lichess_{YYYY-MM}.pgn.zst files. Default: RAW_DUMP_DIR (config.py).
    --out-dir     Folder to write game_{split}_{ddmm}_{ddmm}.csv / pgn_{split}_{ddmm}_{ddmm}.pgn.zst into.
                  Default: GAME_DIR (config.py).
    --max-games   Cap on games scanned per dump file. Default: None.
"""

import os
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from scripts.config import RAW_DUMP_DIR, GAME_DIR

import argparse
import csv
import io
import re
import shutil
from collections import deque
from dataclasses import dataclass
from datetime import date, datetime, timedelta

import zstandard as zstd
from tqdm import tqdm

####################
# CONSTANTS
####################

DATE_RE = re.compile(r'^\d{4}-\d{2}-\d{2}$')
HEADER_RE = re.compile(r'\[(\w+)\s+"(.*)"\]')
TIME_RE = re.compile(r'^\d{2}:\d{2}:\d{2}$')
MOVE_NUM_RE = re.compile(r'^\d+\.+$')
CLK_RE = re.compile(r'\[%clk\s+[\d:]+\]')

RESULT_MAP = {'1-0': 1.0, '0-1': 0.0, '1/2-1/2': 0.5}
RESULT_TOKENS = set(RESULT_MAP) | {'*'}

MIN_ELO_BOUND = 0
MAX_ELO_BOUND = 4000

BULLET_PRESETS = ['60+0', '120+1']
BLITZ_PRESETS = ['180+0', '180+2', '300+0', '300+3']
RAPID_PRESETS = ['600+0', '600+5', '900+10']
CLASSICAL_PRESETS = ['1800+0', '1800+20']

SPEED_PRESETS = {
    'Bullet': BULLET_PRESETS,
    'Blitz': BLITZ_PRESETS,
    'Rapid': RAPID_PRESETS,
    'Classical': CLASSICAL_PRESETS,
}
PRESET_TO_SPEED = {tc: speed for speed, presets in SPEED_PRESETS.items() for tc in presets}

SPLIT_ORDER = ['eda', 'train', 'val', 'test']

# Domain / parse filters
RATED_ONLY = True
TIME_CONTROLS = ['Rapid']
REMOVE_BOTS = True

# Output-quality filters
MIN_PLIES = 12
TERMINATIONS = ['Normal', 'Time forfeit', 'Insufficient material']
VALID_CLOCK = True
TAILS = True
MIN_ELO = 800
MAX_ELO = 2200

# Past-performance / history window
HISTORY_MAX_GAMES = 10
HISTORY_MAX_DAYS = 10
PAST_PERF_WEIGHTS = list(range(HISTORY_MAX_GAMES, 0, -1))
PAST_PERF_K = 0.0
PRUNE_EVERY_N_GAMES = 500_000

# PGN cache
PGN_CACHE_COMPRESSION_LEVEL = 9

# Date ranges, 'YYYY-MM-DD', inclusive both ends. None: skips that split.
EDA_FROM, EDA_TO = '2026-01-01', '2026-01-31'
TRAIN_FROM, TRAIN_TO = '2026-02-01', '2026-04-30'
VAL_FROM, VAL_TO = '2026-05-01', '2026-05-15'
TEST_FROM, TEST_TO = '2026-05-16', '2026-05-31'

# Cap on games scanned per dump file. None reads the whole file.
MAX_GAMES = None

MIN_FREE_DISK_GB = 20.0

GAME_CSV_FIELDNAMES = [
    'game_id', 'datetime', 'white', 'black', 'white_elo', 'black_elo',
    'white_title', 'black_title', 'white_rating_diff', 'black_rating_diff',
    'result', 'speed', 'time_control', 'termination', 'eco', 'ply_count',
    'past_white', 'past_black', 'prev_white', 'prev_black',
    'hours_since_white', 'hours_since_black',
    'has_history_white', 'has_history_black', 'rematch',
]

GAME_CSV_DTYPES = {
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
    'past_white': 'float32',
    'past_black': 'float32',
    'prev_white': 'float32',
    'prev_black': 'float32',
    'hours_since_white': 'float32',
    'hours_since_black': 'float32',
}

# Boolean columns only computable using historical data
GAME_CSV_BOOL_COLS = ['has_history_white', 'has_history_black', 'rematch']

####################
# FUNCTIONS
####################

# (a) DATE / MONTH HELPERS

def _parse_date(date_str: str, label: str) -> date:
    """Parses a YYYY-MM-DD string to a date, exiting with a clear message if malformed."""
    if not DATE_RE.match(date_str):
        sys.exit(f'{label} must be YYYY-MM-DD, got: {date_str!r}')
    return date.fromisoformat(date_str)

def _months_needed(active: dict[str, tuple[date, date]]) -> list[str]:
    """Returns YYYY-MM months spanning (earliest active start - HISTORY_MAX_DAYS) to (latest active end)."""
    earliest = min(f for f, _ in active.values()) - timedelta(days=HISTORY_MAX_DAYS)
    latest = max(t for _, t in active.values())
    months = []
    cur = earliest.replace(day=1)
    while cur <= latest:
        months.append(cur.strftime('%Y-%m'))
        cur = cur.replace(year=cur.year + 1, month=1) if cur.month == 12 else cur.replace(month=cur.month + 1)
    return months

def _bucket_bounds(active: dict[str, tuple[date, date]]) -> dict[str, tuple[datetime, datetime]]:
    """Converts each active split's inclusive date range into a half-open [start, end) datetime interval."""
    return {
        split: (datetime.combine(f, datetime.min.time()),
                datetime.combine(t + timedelta(days=1), datetime.min.time()))
        for split, (f, t) in active.items()
    }

def _ddmm(d: date) -> str:
    """Formats a date as DD-MM for output filenames."""
    return d.strftime('%d-%m')

# (b) PARSING

def _count_plies(movetext: str) -> int:
    """Counts plies in movetext, stripping comments, NAG codes, move numbers, and result tokens."""
    text = re.sub(r'\{[^}]*\}', '', movetext)
    text = re.sub(r'\$\d+', '', text)
    plies = 0
    for token in text.split():
        if token in RESULT_TOKENS or MOVE_NUM_RE.match(token):
            continue
        plies += 1
    return plies

def _parse_rating_diff(raw: str) -> int | None:
    """Parses a WhiteRatingDiff/BlackRatingDiff header value to int, or None if missing/malformed."""
    try:
        return int(raw)
    except ValueError:
        return None

def parse_core(header_lines: str, movetext: str, config: 'ParseFilterConfig') -> dict | None:
    """Parses one game's headers/movetext, applying every structural sanity check and domain filter;
    returns None on any failure."""
    headers = dict(HEADER_RE.findall(header_lines))

    if config.rated_only and not headers.get('Event', '').strip().startswith('Rated'):
        return None

    if config.remove_bots:
        if headers.get('WhiteTitle', '').strip().upper() == 'BOT':
            return None
        if headers.get('BlackTitle', '').strip().upper() == 'BOT':
            return None

    time_control_raw = headers.get('TimeControl', '').strip()
    speed = PRESET_TO_SPEED.get(time_control_raw)
    if speed is None or speed not in config.time_controls:
        return None

    white_elo_raw = headers.get('WhiteElo', '')
    black_elo_raw = headers.get('BlackElo', '')
    if not (white_elo_raw.isdigit() and black_elo_raw.isdigit()):
        return None
    white_elo = int(white_elo_raw)
    black_elo = int(black_elo_raw)
    if not (MIN_ELO_BOUND < white_elo < MAX_ELO_BOUND and MIN_ELO_BOUND < black_elo < MAX_ELO_BOUND):
        return None

    eco = headers.get('ECO', '').strip()
    if len(eco) != 3:
        return None

    white_title = headers.get('WhiteTitle', '').strip() or 'None'
    black_title = headers.get('BlackTitle', '').strip() or 'None'

    white = headers.get('White', '').strip()
    black = headers.get('Black', '').strip()
    if not white or not black or white == black:
        return None

    result_raw = headers.get('Result', '').strip()
    if result_raw not in RESULT_MAP:
        return None
    result = RESULT_MAP[result_raw]

    utc_date = headers.get('UTCDate', '').strip()
    utc_time = headers.get('UTCTime', '').strip()
    date_parts = utc_date.split('.')
    if len(date_parts) != 3 or not all(p.isdigit() for p in date_parts) or not TIME_RE.match(utc_time):
        return None
    try:
        game_dt = datetime.strptime(f"{'-'.join(date_parts)} {utc_time}", '%Y-%m-%d %H:%M:%S')
    except ValueError:
        return None

    site = headers.get('Site', '').strip()
    game_id = site.rsplit('/', 1)[-1] if site else ''
    if not game_id:
        return None

    if not movetext.strip():
        return None

    white_rating_diff = _parse_rating_diff(headers.get('WhiteRatingDiff', ''))
    black_rating_diff = _parse_rating_diff(headers.get('BlackRatingDiff', ''))

    termination = headers.get('Termination', '').strip()
    ply_count = _count_plies(movetext)
    clock_ok = len(CLK_RE.findall(movetext)) == ply_count

    return {
        'game_id': game_id, 'datetime': game_dt, 'white': white, 'black': black,
        'white_elo': white_elo, 'black_elo': black_elo,
        'white_title': white_title, 'black_title': black_title,
        'white_rating_diff': white_rating_diff, 'black_rating_diff': black_rating_diff,
        'result': result, 'speed': speed, 'time_control': time_control_raw,
        'termination': termination, 'eco': eco, 'ply_count': ply_count, 'clock_ok': clock_ok,
    }

# (c) OUTPUT FILTERING

def passes_output_filter(core: dict, config: 'OutputFilterConfig') -> bool:
    """Returns whether a parsed game passes the output-quality filters."""
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

# (d) ROW ASSEMBLY

def _build_row(core: dict, white_feats: dict, black_feats: dict, rematch: bool) -> dict:
    """Assembles one output CSV row from parsed core fields and both players' history features."""
    return {
        'game_id': core['game_id'],
        'datetime': core['datetime'].strftime('%Y-%m-%d %H:%M:%S'),
        'white': core['white'], 'black': core['black'],
        'white_elo': core['white_elo'], 'black_elo': core['black_elo'],
        'white_title': core['white_title'], 'black_title': core['black_title'],
        'white_rating_diff': core['white_rating_diff'] if core['white_rating_diff'] is not None else '',
        'black_rating_diff': core['black_rating_diff'] if core['black_rating_diff'] is not None else '',
        'result': core['result'], 'speed': core['speed'], 'time_control': core['time_control'],
        'termination': core['termination'], 'eco': core['eco'], 'ply_count': core['ply_count'],
        'past_white': white_feats['past'], 'past_black': black_feats['past'],
        'prev_white': white_feats['prev'], 'prev_black': black_feats['prev'],
        'hours_since_white': white_feats['hours_since'] if white_feats['hours_since'] is not None else '',
        'hours_since_black': black_feats['hours_since'] if black_feats['hours_since'] is not None else '',
        'has_history_white': white_feats['has_history'], 'has_history_black': black_feats['has_history'],
        'rematch': rematch,
    }

# (e) OUTPUT I/O

def _check_disk_space(path: str, required_gb: float) -> None:
    """Raises if path's filesystem has less than required_gb free."""
    os.makedirs(path, exist_ok=True)
    free_gb = shutil.disk_usage(path).free / (1024 ** 3)
    if free_gb < required_gb:
        raise RuntimeError(
            f'Not enough free disk space at {path}: {free_gb:.1f}GB free, need at least {required_gb:.1f}GB.'
        )

def _open_split_outputs(split: str, out_dir: str, active: dict) -> dict:
    """Opens tmp-path CSV and PGN-cache writers for one split; returns a bundle of handles/paths."""
    f_date, t_date = active[split]
    tag = f'{_ddmm(f_date)}_{_ddmm(t_date)}'

    csv_final = os.path.join(out_dir, f'game_{split}_{tag}.csv')
    csv_tmp = csv_final + '.tmp'
    csv_file = open(csv_tmp, 'w', newline='', encoding='utf-8')
    csv_writer = csv.DictWriter(csv_file, fieldnames=GAME_CSV_FIELDNAMES)
    csv_writer.writeheader()

    pgn_final = os.path.join(out_dir, f'pgn_{split}_{tag}.pgn.zst')
    pgn_tmp = pgn_final + '.tmp'
    pgn_file = open(pgn_tmp, 'wb')
    cctx = zstd.ZstdCompressor(level=PGN_CACHE_COMPRESSION_LEVEL)
    pgn_stream = cctx.stream_writer(pgn_file)

    print(f'Output ({split}): {csv_final}')
    print(f'PGN cache ({split}): {pgn_final}')

    return {
        'csv_writer': csv_writer, 'csv_file': csv_file, 'csv_tmp': csv_tmp, 'csv_final': csv_final,
        'pgn_stream': pgn_stream, 'pgn_file': pgn_file, 'pgn_tmp': pgn_tmp, 'pgn_final': pgn_final,
    }

def _verify_pgn_cache(path: str, expected_games: int) -> bool:
    """Reopens a written PGN cache and confirms it decompresses cleanly with the expected game count."""
    try:
        dctx = zstd.ZstdDecompressor()
        with open(path, 'rb') as f, dctx.stream_reader(f) as reader:
            text_stream = io.TextIOWrapper(reader, encoding='utf-8')
            n_games = sum(1 for line in text_stream if line.startswith('[GameId '))
    except (zstd.ZstdError, OSError, UnicodeDecodeError):
        return False
    return n_games == expected_games

def _finalize_split(split: str, handles: dict, n_written: int) -> None:
    """Closes a split's writers, verifies the PGN cache decompresses cleanly, then atomically renames
    both outputs to their final paths. Raises rather than renaming if verification fails."""
    handles['csv_file'].close()
    os.replace(handles['csv_tmp'], handles['csv_final'])

    handles['pgn_stream'].close()
    handles['pgn_file'].close()

    if not _verify_pgn_cache(handles['pgn_tmp'], n_written):
        raise RuntimeError(
            f"PGN cache for split '{split}' failed integrity verification "
            f"({handles['pgn_tmp']}); not renaming to final path. Investigate before rerunning."
        )
    os.replace(handles['pgn_tmp'], handles['pgn_final'])
    print(f'Finalized {split}: {n_written:,} games -> {handles["csv_final"]}, {handles["pgn_final"]}')

# (f) STREAMED EXTRACTION

def run(input_paths: list[str], months: list[str], out_dir: str, parse_config: 'ParseFilterConfig',
        output_config: 'OutputFilterConfig', tracker: 'HistoryTracker', active: dict,
        max_games: int | None) -> dict:
    """Streams every needed dump month once, routing domain-eligible games to whichever active split
    their datetime falls in, finalizing each split's outputs as soon as the stream passes its end date."""
    bounds = _bucket_bounds(active)
    open_splits = set(active)

    handles = {split: _open_split_outputs(split, out_dir, active) for split in SPLIT_ORDER if split in active}

    games_seen = 0
    games_domain_eligible = 0
    games_written = {split: 0 for split in active}
    games_dropped_late = 0
    order_violations = 0
    last_seen_dt = None

    for input_path, dump_month in zip(input_paths, months):
        if not open_splits:
            break
        print(f'Reading from: {input_path}')
        total_size = os.path.getsize(input_path)

        with open(input_path, 'rb') as raw_f:
            progress_reader = _ProgressReader(raw_f, total_size, desc=dump_month)
            try:
                dctx = zstd.ZstdDecompressor()
                with dctx.stream_reader(progress_reader) as reader:
                    text_stream = io.TextIOWrapper(reader, encoding='utf-8')

                    header_lines = ''
                    movetext = ''
                    reading_moves = False
                    stop_file = False
                    file_seen = 0

                    def handle_block():
                        nonlocal games_seen, games_domain_eligible, games_dropped_late
                        nonlocal order_violations, last_seen_dt, stop_file, file_seen

                        games_seen += 1
                        file_seen += 1

                        core = parse_core(header_lines, movetext, parse_config)
                        if core is not None:
                            dt = core['datetime']
                            if last_seen_dt is not None and dt < last_seen_dt:
                                order_violations += 1
                            else:
                                last_seen_dt = dt

                            games_domain_eligible += 1

                            white_feats = tracker.features(core['white'], dt)
                            black_feats = tracker.features(core['black'], dt)
                            rematch = (
                                white_feats['has_history'] and white_feats['last_opponent'] == core['black']
                                and black_feats['has_history'] and black_feats['last_opponent'] == core['white']
                            )

                            true_bucket = None
                            for split, (f_dt, t_dt) in bounds.items():
                                if f_dt <= dt < t_dt:
                                    true_bucket = split
                                    break

                            if true_bucket is not None:
                                if true_bucket in open_splits and passes_output_filter(core, output_config):
                                    row = _build_row(core, white_feats, black_feats, rematch)
                                    handles[true_bucket]['csv_writer'].writerow(row)
                                    handles[true_bucket]['pgn_stream'].write(
                                        f'[GameId "{core["game_id"]}"]\n'.encode('utf-8'))
                                    handles[true_bucket]['pgn_stream'].write(movetext.encode('utf-8'))
                                    handles[true_bucket]['pgn_stream'].write(b'\n\n')
                                    games_written[true_bucket] += 1
                                elif true_bucket not in open_splits:
                                    games_dropped_late += 1

                            white_result = core['result']
                            black_result = 1.0 - core['result'] if core['result'] != 0.5 else 0.5
                            tracker.update(core['white'], dt, white_result, core['black'])
                            tracker.update(core['black'], dt, black_result, core['white'])

                            if games_seen % PRUNE_EVERY_N_GAMES == 0:
                                tracker.prune_stale(dt)

                            for split in list(open_splits):
                                if dt >= bounds[split][1]:
                                    _finalize_split(split, handles[split], games_written[split])
                                    open_splits.discard(split)

                        if games_seen % 5_000_000 == 0:
                            print(f'Seen: {games_seen:,}, domain-eligible: {games_domain_eligible:,}, '
                                  f'written: {games_written}')
                        if max_games is not None and file_seen >= max_games:
                            stop_file = True
                        if not open_splits:
                            stop_file = True

                    try:
                        for line in text_stream:
                            if line.startswith('['):
                                if reading_moves and movetext:
                                    handle_block()
                                    header_lines = ''
                                    movetext = ''
                                    reading_moves = False
                                    if stop_file:
                                        print(f'Stopping {dump_month} early.')
                                        break
                                header_lines += line
                            elif line.strip() == '':
                                reading_moves = True
                            else:
                                movetext += line
                        else:
                            if movetext:
                                handle_block()
                    except (zstd.ZstdError, OSError, UnicodeDecodeError) as e:
                        if header_lines or movetext:
                            handle_block()
                        print(f'{dump_month} stream ended early (likely truncated dump): {e} '
                              f'-- keeping games parsed before cut-off.')
            finally:
                progress_reader.close()

    for split in list(open_splits):
        _finalize_split(split, handles[split], games_written[split])
        open_splits.discard(split)

    print(f'Done. Seen: {games_seen:,}, domain-eligible: {games_domain_eligible:,}, written: {games_written}')
    if order_violations:
        print(f'WARNING: {order_violations:,} game(s) had a datetime earlier than a preceding game in the '
              f'stream. History features assume ascending order; investigate if this count is large.')
    if games_dropped_late:
        print(f'WARNING: {games_dropped_late:,} game(s) matched a split that had already been finalized '
              f'(a symptom of the same ordering issue) and were dropped from output.')

    return {'games_seen': games_seen, 'games_domain_eligible': games_domain_eligible,
            'games_written': games_written, 'order_violations': order_violations,
            'games_dropped_late': games_dropped_late}

# (g) VALIDATION & CLI

def _validate_config() -> dict[str, tuple[date, date]]:
    """Validates date-range/history/filter constants; returns parsed, non-overlapping active split date ranges."""
    if HISTORY_MAX_GAMES < 1:
        sys.exit('HISTORY_MAX_GAMES must be >= 1.')
    if HISTORY_MAX_DAYS < 1:
        sys.exit('HISTORY_MAX_DAYS must be >= 1.')
    if len(PAST_PERF_WEIGHTS) != HISTORY_MAX_GAMES:
        sys.exit(f'PAST_PERF_WEIGHTS must have length HISTORY_MAX_GAMES ({HISTORY_MAX_GAMES}), '
                 f'got {len(PAST_PERF_WEIGHTS)}.')
    if PRUNE_EVERY_N_GAMES < 1:
        sys.exit('PRUNE_EVERY_N_GAMES must be >= 1.')
    unknown = set(TIME_CONTROLS) - set(SPEED_PRESETS)
    if unknown:
        sys.exit(f'TIME_CONTROLS has unrecognised value(s) {sorted(unknown)}. '
                 f'Valid options: {sorted(SPEED_PRESETS)}.')
    if MIN_PLIES < 0:
        sys.exit('MIN_PLIES must be >= 0.')
    if MIN_ELO < 0:
        sys.exit('MIN_ELO must be >= 0.')
    if MAX_ELO is not None and MAX_ELO <= MIN_ELO:
        sys.exit('MAX_ELO must be > MIN_ELO (MAX_ELO is an exclusive upper bound), or None.')

    raw_ranges = {'eda': (EDA_FROM, EDA_TO), 'train': (TRAIN_FROM, TRAIN_TO),
                  'val': (VAL_FROM, VAL_TO), 'test': (TEST_FROM, TEST_TO)}
    active = {}
    for split, (f, t) in raw_ranges.items():
        if f is None and t is None:
            continue
        if f is None or t is None:
            sys.exit(f'{split.upper()}_FROM/{split.upper()}_TO must both be set or both be None.')
        f_date = _parse_date(f, f'{split.upper()}_FROM')
        t_date = _parse_date(t, f'{split.upper()}_TO')
        if f_date > t_date:
            sys.exit(f'{split.upper()}_FROM ({f}) is after {split.upper()}_TO ({t}).')
        active[split] = (f_date, t_date)

    if not active:
        sys.exit('At least one of EDA/TRAIN/VAL/TEST date ranges must be set.')

    ordered = sorted(active.items(), key=lambda kv: kv[1][0])
    for (n1, (f1, t1)), (n2, (f2, t2)) in zip(ordered, ordered[1:]):
        if f2 <= t1:
            sys.exit(f'Date ranges for {n1} and {n2} overlap.')

    return active

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parses --in-dir, --out-dir, --max-games CLI arguments."""
    parser = argparse.ArgumentParser(
        description='Streams lichess dump months, applies domain/output filters, computes windowed '
                     'past-performance history, and writes eda/train/val/test game CSVs plus PGN caches.'
    )
    parser.add_argument('--in-dir', default=RAW_DUMP_DIR, help=f'Default: {RAW_DUMP_DIR}')
    parser.add_argument('--out-dir', default=GAME_DIR, help=f'Default: {GAME_DIR}')
    parser.add_argument('--max-games', type=int, default=MAX_GAMES,
                         help='Cap on games scanned per dump file. Default: None.')
    return parser.parse_args(argv)

def main(argv: list[str] | None = None) -> None:
    """Reads args, validates config, streams all needed dump months, writing eda/train/val/test outputs."""
    active = _validate_config()
    args = parse_args(argv)
    if args.max_games is not None and args.max_games <= 0:
        sys.exit('--max-games must be > 0 or omitted.')

    os.makedirs(args.out_dir, exist_ok=True)
    _check_disk_space(args.out_dir, MIN_FREE_DISK_GB)

    months = _months_needed(active)
    input_paths = [os.path.join(args.in_dir, f'lichess_{m}.pgn.zst') for m in months]
    missing = [(p, m) for p, m in zip(input_paths, months) if not os.path.exists(p)]
    if missing:
        for p, m in missing:
            print(f'No local dump at {p} -- run: python run_download_dumps.py --date {m}')
        sys.exit(f'Missing {len(missing)} required dump(s), see above.')

    print(f'Active splits: {list(active)}')
    print(f'Months needed (history lookback -> latest split end): {months}')

    parse_config = ParseFilterConfig(
        time_controls=set(TIME_CONTROLS), rated_only=RATED_ONLY, remove_bots=REMOVE_BOTS,
    )
    output_config = OutputFilterConfig(
        min_plies=MIN_PLIES, terminations=set(TERMINATIONS), valid_clock=VALID_CLOCK,
        tails=TAILS, min_elo=MIN_ELO, max_elo=MAX_ELO,
    )
    tracker = HistoryTracker(
        max_games=HISTORY_MAX_GAMES, max_days=HISTORY_MAX_DAYS,
        weights=PAST_PERF_WEIGHTS, smoothing_k=PAST_PERF_K,
    )

    run(input_paths, months, args.out_dir, parse_config, output_config, tracker, active, args.max_games)

####################
# CLASSES
####################

@dataclass
class ParseFilterConfig:
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

class HistoryTracker:
    """Per-player rolling window of past games, bounded by count (HISTORY_MAX_GAMES) and day-based
    recency (HISTORY_MAX_DAYS); computes the lag-weighted past-performance metric plus lag-1 fields."""

    def __init__(self, max_games: int, max_days: int, weights: list[float], smoothing_k: float):
        self._by_player: dict[str, deque] = {}
        self._max_games = max_games
        self._max_days = max_days
        self._weights = weights
        self._smoothing_k = smoothing_k

    def _prune(self, dq: deque, now: datetime) -> None:
        cutoff = now - timedelta(days=self._max_days)
        while dq and dq[0][0] < cutoff:
            dq.popleft()

    def features(self, player: str, now: datetime) -> dict:
        """Returns past/prev/hours_since/has_history/last_opponent for player, as of just before now."""
        dq = self._by_player.get(player)
        if dq is not None:
            self._prune(dq, now)
        if not dq:
            return {'past': 0.0, 'prev': 0.0, 'hours_since': None, 'has_history': False, 'last_opponent': None}

        entries = list(dq)[::-1]
        weighted_sum, weight_total = 0.0, 0.0
        for i, (_, past_result, _) in enumerate(entries):
            w = self._weights[i]
            weighted_sum += past_result * w
            weight_total += w
        denom = weight_total + self._smoothing_k
        past = weighted_sum / denom if denom > 0 else 0.0

        last_dt, last_result, last_opponent = entries[0]
        hours_since = (now - last_dt).total_seconds() / 3600.0

        return {'past': past, 'prev': last_result, 'hours_since': hours_since,
                'has_history': True, 'last_opponent': last_opponent}

    def update(self, player: str, dt: datetime, result_for_player: float, opponent: str) -> None:
        """Records one played game for player, pruning stale entries first."""
        dq = self._by_player.setdefault(player, deque(maxlen=self._max_games))
        self._prune(dq, dt)
        dq.append((dt, result_for_player, opponent))

    def prune_stale(self, now: datetime) -> int:
        """Evicts every tracked player whose most recent game is older than max_days; returns count evicted."""
        cutoff = now - timedelta(days=self._max_days)
        stale = [p for p, dq in self._by_player.items() if not dq or dq[-1][0] < cutoff]
        for p in stale:
            del self._by_player[p]
        return len(stale)

    def __len__(self) -> int:
        return len(self._by_player)

class _ProgressReader:
    """Wraps a binary file object, updating a tqdm bar as bytes are read."""

    def __init__(self, raw, total: int, desc: str):
        self.raw = raw
        self.bar = tqdm(total=total, unit='B', unit_scale=True, desc=desc)

    def read(self, size: int = -1) -> bytes:
        chunk = self.raw.read(size)
        self.bar.update(len(chunk))
        return chunk

    def close(self) -> None:
        self.bar.close()


if __name__ == '__main__':
    main()