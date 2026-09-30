"""
run_reader_unfiltered.py
Unfiltered pass over a lichess .pgn.zst dump. Applies only structural sanity checks,
aim is to apply substantial filtering decisions to this dataset, including data
analysis for choice-justification.

Latest changes: 17/08/26:
- Fixed tmp writing to avoid overwrites

Run:
    !python run_reader_unfiltered.py --date 2026-01
CLI:
    --date       Dump month, YYYY-MM. Required.
    --in-dir     Folder holding lichess_{date}.pgn.zst. Default: RAW_DUMP_DIR (config.py).
    --out-dir    Folder to write game_unfiltered_{date}.csv into. Default: GAME_DIR (config.py).
"""

import os
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from src.config import RAW_DUMP_DIR, GAME_DIR

import argparse
import csv
import io
import re
import shutil
from dataclasses import dataclass

import zstandard as zstd
from tqdm import tqdm

####################
# CONSTANTS
####################

DATE_RE = re.compile(r'^\d{4}-\d{2}$')
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

FIELDNAMES = ['game_id', 'datetime', 'white', 'black', 'white_elo', 'black_elo',
              'white_title', 'black_title', 'white_rating_diff', 'black_rating_diff',
              'result', 'speed', 'time_control', 'termination', 'rated',
              'eco', 'ply_count', 'clock_ok']

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

MAX_GAMES = None
# Cap on games processed. None reads the whole dump.

MIN_FREE_DISK_GB = 20.0

####################
# FUNCTIONS
####################

# (a) PARSING

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

def parse_core(header_lines: str, movetext: str, dump_date: str, seen_game_ids: set[str]) -> dict | None:
    """Parses one game's headers/movetext into a metadata row, applying every SanityChecks check;
    returns None on any failure."""
    headers = dict(HEADER_RE.findall(header_lines))

    termination = headers.get('Termination', '').strip()
    rated = headers.get('Event', '').strip().startswith('Rated')

    white_title = headers.get('WhiteTitle', '').strip() or 'None'
    black_title = headers.get('BlackTitle', '').strip() or 'None'

    time_control_raw = headers.get('TimeControl', '').strip()
    speed = PRESET_TO_SPEED.get(time_control_raw, 'Custom')

    white_elo_raw = headers.get('WhiteElo', '')
    black_elo_raw = headers.get('BlackElo', '')
    if not (white_elo_raw.isdigit() and black_elo_raw.isdigit()):
        return None
    white_elo = int(white_elo_raw)
    black_elo = int(black_elo_raw)
    if not (MIN_ELO_BOUND < white_elo < MAX_ELO_BOUND and MIN_ELO_BOUND < black_elo < MAX_ELO_BOUND):
        return None

    white_rating_diff = _parse_rating_diff(headers.get('WhiteRatingDiff', ''))
    black_rating_diff = _parse_rating_diff(headers.get('BlackRatingDiff', ''))

    eco = headers.get('ECO', '').strip()
    if len(eco) != 3:
        return None

    white = headers.get('White', '').strip()
    black = headers.get('Black', '').strip()
    if not white or not black:
        return None

    result_raw = headers.get('Result', '').strip()
    if result_raw not in RESULT_MAP:
        return None
    result = RESULT_MAP[result_raw]

    utc_date = headers.get('UTCDate', '').strip()
    utc_time = headers.get('UTCTime', '').strip()
    date_parts = utc_date.split('.')
    if len(date_parts) != 3 or not all(p.isdigit() for p in date_parts):
        return None
    if not TIME_RE.match(utc_time):
        return None
    if f'{date_parts[0]}-{date_parts[1]}' != dump_date:
        return None
    datetime_str = f"{'-'.join(date_parts)} {utc_time}"

    site = headers.get('Site', '').strip()
    game_id = site.rsplit('/', 1)[-1] if site else ''
    if not game_id or game_id in seen_game_ids:
        return None
    seen_game_ids.add(game_id)

    if not movetext.strip():
        return None
    ply_count = _count_plies(movetext)
    clock_ok = len(CLK_RE.findall(movetext)) == ply_count

    return {
        'game_id': game_id,
        'datetime': datetime_str,
        'white': white,
        'black': black,
        'white_elo': white_elo,
        'black_elo': black_elo,
        'white_title': white_title,
        'black_title': black_title,
        'white_rating_diff': white_rating_diff,
        'black_rating_diff': black_rating_diff,
        'result': result,
        'speed': speed,
        'time_control': time_control_raw,
        'termination': termination,
        'rated': rated,
        'eco': eco,
        'ply_count': ply_count,
        'clock_ok': clock_ok,
    }

# (b) OUTPUT I/O

def _check_disk_space(path: str, required_gb: float) -> None:
    """Raises if path's filesystem has less than required_gb free."""
    os.makedirs(path, exist_ok=True)
    free_gb = shutil.disk_usage(path).free / (1024 ** 3)
    if free_gb < required_gb:
        raise RuntimeError(
            f'Not enough free disk space at {path}: {free_gb:.1f}GB free, need at least {required_gb:.1f}GB.'
        )

# (c) STREAMED EXTRACTION

def run(input_path: str, output_path: str, dump_date: str, max_games: int | None) -> tuple[int, int]:
    """Streams input_path through parse_core, writing kept rows to a tmp file that's atomically
    renamed to output_path once the stream finishes; returns (games_seen, games_kept)."""
    games_seen = 0
    games_kept = 0
    seen_game_ids: set[str] = set()

    print(f'Reading from: {input_path}')
    print(f'Output:       {output_path}')

    total_size = os.path.getsize(input_path)

    output_tmp = output_path + '.tmp'
    out_f = open(output_tmp, 'w', newline='', encoding='utf-8')
    writer = csv.DictWriter(out_f, fieldnames=FIELDNAMES)
    writer.writeheader()

    try:
        with open(input_path, 'rb') as raw_f:
            progress_reader = _ProgressReader(raw_f, total_size, desc=dump_date)
            try:
                dctx = zstd.ZstdDecompressor()
                with dctx.stream_reader(progress_reader) as reader:
                    text_stream = io.TextIOWrapper(reader, encoding='utf-8')

                    header_lines = ''
                    movetext = ''
                    reading_moves = False
                    stop = False

                    def handle_block():
                        nonlocal games_seen, games_kept, header_lines, movetext, stop
                        row = parse_core(header_lines, movetext, dump_date, seen_game_ids)
                        games_seen += 1
                        if row is not None:
                            writer.writerow(row)
                            games_kept += 1
                        if games_seen % 5_000_000 == 0:
                            print(f'Seen: {games_seen:,}, Kept: {games_kept:,}')
                        if max_games is not None and games_seen >= max_games:
                            stop = True

                    try:
                        for line in text_stream:
                            if line.startswith('['):
                                if reading_moves and movetext:
                                    handle_block()
                                    header_lines = ''
                                    movetext = ''
                                    reading_moves = False
                                    if stop:
                                        print(f'Reached MAX_GAMES cap ({max_games}), stopping.')
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
                        print(f'Stream ended early (likely truncated dump): {e} -- keeping games parsed before cut-off.')
            finally:
                progress_reader.close()
    finally:
        out_f.close()

    os.replace(output_tmp, output_path)

    print('Done.')
    print(f'Games parsed (pre validity checks): {games_seen:,}')
    print(f'Games kept (in output CSV):         {games_kept:,}')
    return games_seen, games_kept

# (d) CLI

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parses --date, --in-dir, --out-dir CLI arguments."""
    parser = argparse.ArgumentParser(description='Unfiltered baseline metadata pass over a local lichess dump.')
    parser.add_argument('--date', required=True, help='Dump month, YYYY-MM.')
    parser.add_argument('--in-dir', default=RAW_DUMP_DIR, help=f'Default: {RAW_DUMP_DIR}')
    parser.add_argument('--out-dir', default=GAME_DIR, help=f'Default: {GAME_DIR}')
    args = parser.parse_args(argv)
    if not DATE_RE.match(args.date):
        parser.error(f'--date must be YYYY-MM, got: {args.date!r}')
    return args

def main(argv: list[str] | None = None) -> None:
    """Reads args, runs the extraction, exits early if the dump or MAX_GAMES is invalid."""
    if MAX_GAMES is not None and MAX_GAMES <= 0:
        sys.exit('MAX_GAMES must be > 0 or None.')

    args = parse_args(argv)
    os.makedirs(args.out_dir, exist_ok=True)
    _check_disk_space(args.out_dir, MIN_FREE_DISK_GB)

    input_path = os.path.join(args.in_dir, f'lichess_{args.date}.pgn.zst')
    output_path = os.path.join(args.out_dir, f'game_unfiltered_{args.date}.csv')

    if not os.path.exists(input_path):
        sys.exit(f'No local dump at {input_path} -- run run_download_dumps.py --date {args.date} first.')

    run(input_path, output_path, args.date, MAX_GAMES)

####################
# CLASSES
####################

@dataclass(frozen=True)
class SanityChecks:
    """Always-applied structural checks a parsed game must pass; not configurable."""
    valid_elo: bool = True
    valid_eco: bool = True
    valid_username: bool = True
    valid_result: bool = True
    valid_datetime: bool = True
    valid_month: bool = True
    valid_movetext: bool = True
    valid_unique_game_id: bool = True

SANITY_CHECKS = SanityChecks()


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