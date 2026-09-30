"""
run_pos_reader.py
Samples positions for one train/val/test/eda split, reading that split's game CSV and matching
PGN cache from run_game_reader.py in lockstep.

Sampling approach:
  For each game, sample floor((ply_count - MIN_PLIES_PLAYED + 1) / PLIES_PER_SAMPLE)
  positions, drawn uniformly without replacement from plies
  [MIN_PLIES_PLAYED, ply_count - 1].

Run:
    !python run_pos_reader.py train
CLI:
    split        One of: eda, train, val, test. Required.
    --game-dir   Folder holding game_{split}_*.csv / pgn_{split}_*.pgn.zst. Default: GAME_DIR (config.py).
    --out-dir    Folder to write pos_{split}_{tag}.csv into. Default: POS_DIR (config.py).

Latest changes: 18/08/26:
- Raised min plies in games used for sampling to match truth, reduced printing
"""

import os
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from src.config import GAME_DIR, POS_DIR

import argparse
import csv
import gc
import glob
import io
import multiprocessing as mp
import random
import re
import shutil
from itertools import zip_longest

import chess
import chess.pgn
import zstandard as zstd
from tqdm import tqdm

####################
# CONSTANTS
####################

SPLITS = ['eda', 'train', 'val', 'test']

GAME_ID_RE = re.compile(r'^\[GameId "(.*)"\]$')
CLOCK_RE = re.compile(r'\[%clk\s+(\d+):(\d+):(\d+)\]')

MIN_PLIES_PLAYED = 11
# Smallest ply eligible for sampling, inclusive.

PLIES_PER_SAMPLE = 10
# One position sampled per this-many eligible plies in a game, rounded down.

RANDOM_SEED = 0
# Seed for position sampling.

MIN_GAME_PLIES = MIN_PLIES_PLAYED + PLIES_PER_SAMPLE - 1
# Smallest ply_count that can produce at least one sampled position.

POSITION_BATCH_SIZE = 20_000
# Games buffered before handing a batch to the worker pool.

SAMPLE_CHUNKSIZE = 64
# imap chunksize for the parallel extraction pool.

GC_EVERY_N_BATCHES = 50

PROGRESS_EVERY_N_POSITIONS = 10_000_000

NUM_WORKERS = os.cpu_count() or 1

MIN_FREE_DISK_GB = 20.0

META_PASSTHROUGH_FIELDS = ['game_id', 'datetime', 'speed', 'time_control',
                            'termination', 'eco', 'ply_count']

GAME_CSV_REQUIRED_FIELDS = [
    'game_id', 'datetime', 'white', 'black', 'white_elo', 'black_elo',
    'white_title', 'black_title', 'white_rating_diff', 'black_rating_diff',
    'result', 'speed', 'time_control', 'termination', 'eco', 'ply_count',
    'past_white', 'past_black', 'prev_white', 'prev_black',
    'hours_since_white', 'hours_since_black',
    'has_history_white', 'has_history_black', 'rematch',
]

POS_CSV_FIELDNAMES = [
    'game_id', 'mover', 'opponent', 'mover_elo', 'opponent_elo',
    'mover_title', 'opponent_title', 'mover_rating_diff', 'opponent_rating_diff',
    'mover_is_white',
    'datetime', 'speed', 'time_control', 'termination', 'eco', 'ply_count',
    'ply_played', 'fen', 'next_move',
    'mover_clock', 'opponent_clock', 'mover_result',
    'past_mover', 'past_opponent', 'prev_mover', 'prev_opponent',
    'hours_since_mover', 'hours_since_opponent',
    'has_history_mover', 'has_history_opponent', 'rematch',
]

POS_CSV_DTYPES = {
    'game_id': 'string',
    'mover': 'string',
    'opponent': 'string',
    'mover_elo': 'int32',
    'opponent_elo': 'int32',
    'mover_title': 'category',
    'opponent_title': 'category',
    'mover_rating_diff': 'Int64',
    'opponent_rating_diff': 'Int64',
    'speed': 'category',
    'time_control': 'category',
    'termination': 'category',
    'eco': 'category',
    'ply_count': 'int16',
    'ply_played': 'int16',
    'fen': 'string',
    'next_move': 'string',
    'mover_clock': 'int32',
    'opponent_clock': 'int32',
    'mover_result': 'float32',
    'past_mover': 'float32',
    'past_opponent': 'float32',
    'prev_mover': 'float32',
    'prev_opponent': 'float32',
    'hours_since_mover': 'float32',
    'hours_since_opponent': 'float32',
}
# load 'datetime' with parse_dates=['datetime']
POS_CSV_BOOL_COLS = ['mover_is_white', 'has_history_mover', 'has_history_opponent', 'rematch']
# load with true_values=['True'], false_values=['False']

_MISSING = object()

####################
# FUNCTIONS
####################

# (a) FILE LOCATION

def locate_split_files(split: str, game_dir: str) -> tuple[str, str]:
    """Finds the single game_{split}_*.csv and its matching pgn_{split}_*.pgn.zst in game_dir."""
    csv_matches = sorted(glob.glob(os.path.join(game_dir, f'game_{split}_*.csv')))
    if len(csv_matches) == 0:
        sys.exit(f'No game_{split}_*.csv found in {game_dir} -- run run_game_reader.py first.')
    if len(csv_matches) > 1:
        sys.exit(f'Expected exactly one game_{split}_*.csv in {game_dir}, found {len(csv_matches)}: '
                  f'{csv_matches}')
    csv_path = csv_matches[0]

    basename = os.path.basename(csv_path)
    tag = basename[len(f'game_{split}_'):-len('.csv')]
    pgn_path = os.path.join(game_dir, f'pgn_{split}_{tag}.pgn.zst')
    if not os.path.exists(pgn_path):
        sys.exit(f'No matching PGN cache at {pgn_path} for {csv_path}.')

    return csv_path, pgn_path

def _validate_csv_schema(csv_path: str) -> None:
    """Checks the game CSV has every column this script depends on."""
    with open(csv_path, newline='', encoding='utf-8') as f:
        header = next(csv.reader(f))
    missing = set(GAME_CSV_REQUIRED_FIELDS) - set(header)
    if missing:
        sys.exit(f'{csv_path} is missing expected column(s) {sorted(missing)} -- '
                  f'check it matches run_game_reader.py\'s output schema.')

# (b) PGN CACHE READING

def iter_pgn_cache(pgn_path: str, progress_reader: '_ProgressReader'):
    """Yields (game_id, pgn_text) for each game block in a PGN cache written by run_game_reader.py."""
    dctx = zstd.ZstdDecompressor()
    with dctx.stream_reader(progress_reader) as reader:
        text_stream = io.TextIOWrapper(reader, encoding='utf-8')

        game_id = None
        movetext_lines: list[str] = []

        for line in text_stream:
            match = GAME_ID_RE.match(line.rstrip('\n'))
            if match:
                if game_id is not None:
                    yield game_id, f'[GameId "{game_id}"]\n' + ''.join(movetext_lines)
                game_id = match.group(1)
                movetext_lines = []
            else:
                movetext_lines.append(line)
        if game_id is not None:
            yield game_id, f'[GameId "{game_id}"]\n' + ''.join(movetext_lines)

# (c) META ROW PARSING

def _parse_meta_row(row: dict) -> dict:
    """Converts one game-CSV row's string fields to typed values."""
    return {
        'game_id': row['game_id'], 'datetime': row['datetime'],
        'white': row['white'], 'black': row['black'],
        'white_elo': int(row['white_elo']), 'black_elo': int(row['black_elo']),
        'white_title': row['white_title'], 'black_title': row['black_title'],
        'white_rating_diff': int(row['white_rating_diff']) if row['white_rating_diff'] else None,
        'black_rating_diff': int(row['black_rating_diff']) if row['black_rating_diff'] else None,
        'result': float(row['result']),
        'speed': row['speed'], 'time_control': row['time_control'],
        'termination': row['termination'], 'eco': row['eco'],
        'ply_count': int(row['ply_count']),
        'past_white': float(row['past_white']), 'past_black': float(row['past_black']),
        'prev_white': float(row['prev_white']), 'prev_black': float(row['prev_black']),
        'hours_since_white': float(row['hours_since_white']) if row['hours_since_white'] else None,
        'hours_since_black': float(row['hours_since_black']) if row['hours_since_black'] else None,
        'has_history_white': row['has_history_white'] == 'True',
        'has_history_black': row['has_history_black'] == 'True',
        'rematch': row['rematch'] == 'True',
    }

# (d) SAMPLING

def _positions_to_sample(ply_count: int) -> int:
    """Returns how many positions to sample from a game, based on its eligible ply range."""
    n_eligible = ply_count - MIN_PLIES_PLAYED + 1
    n_extractable = ply_count - MIN_PLIES_PLAYED
    return max(0, min(n_eligible // PLIES_PER_SAMPLE, n_extractable))

def _sample_plies_for_game(ply_count: int, seed: int) -> list[int]:
    """Uniformly samples _positions_to_sample(ply_count) plies without replacement."""
    n = _positions_to_sample(ply_count)
    if n == 0:
        return []
    rng = random.Random(seed)
    return sorted(rng.sample(range(MIN_PLIES_PLAYED, ply_count), n))

# (e) EXTRACTION

class _MoveListVisitor(chess.pgn.BaseVisitor):
    def __init__(self):
        self.moves = []
        self.clocks = []

    def visit_move(self, board, move):
        self.moves.append(move)
        self.clocks.append(None)

    def visit_comment(self, comment):
        match = CLOCK_RE.search(comment)
        if match and self.clocks:
            h, m, s = (int(part) for part in match.groups())
            self.clocks[-1] = h * 3600 + m * 60 + s

    def result(self):
        return self.moves, self.clocks

def _parse_base_seconds(time_control_raw: str) -> int:
    base_str, _, _ = time_control_raw.partition('+')
    try:
        return int(base_str)
    except ValueError:
        return 0

def _mover_clock_at_ply(clocks_list: list, k: int, base_seconds: int) -> int:
    idx = k - 2
    return clocks_list[idx] if idx >= 0 else base_seconds

def _opponent_clock_at_ply(clocks_list: list, k: int, base_seconds: int) -> int:
    idx = k - 1
    return clocks_list[idx] if idx >= 0 else base_seconds

def _extract_positions_for_game(args: tuple) -> list[dict] | None:
    """Parses one game's PGN text, samples plies, and returns one output row per sampled ply."""
    game_idx, pgn_text, meta_row = args

    try:
        parsed = chess.pgn.read_game(io.StringIO(pgn_text), Visitor=_MoveListVisitor)
    except Exception:
        return None
    if parsed is None:
        return None
    moves_list, clocks_list = parsed

    actual_ply_count = len(moves_list)
    if actual_ply_count != meta_row['ply_count']:
        return None
    if any(c is None for c in clocks_list):
        return None

    assigned_plies = _sample_plies_for_game(actual_ply_count, RANDOM_SEED + game_idx)
    if not assigned_plies:
        return None

    base_seconds = _parse_base_seconds(meta_row['time_control'])
    base_row = {field: meta_row[field] for field in META_PASSTHROUGH_FIELDS}

    board = chess.Board()
    moves_pushed = 0
    rows = []
    for k in assigned_plies:
        mover_clock = _mover_clock_at_ply(clocks_list, k, base_seconds)
        opponent_clock = _opponent_clock_at_ply(clocks_list, k, base_seconds)
        if mover_clock <= 0 or opponent_clock <= 0:
            continue

        while moves_pushed < k:
            board.push(moves_list[moves_pushed])
            moves_pushed += 1
        mover_is_white = board.turn

        if mover_is_white:
            mover, opponent = meta_row['white'], meta_row['black']
            mover_elo, opponent_elo = meta_row['white_elo'], meta_row['black_elo']
            mover_title, opponent_title = meta_row['white_title'], meta_row['black_title']
            mover_rating_diff = meta_row['white_rating_diff']
            opponent_rating_diff = meta_row['black_rating_diff']
            mover_result = meta_row['result']
            past_mover, past_opponent = meta_row['past_white'], meta_row['past_black']
            prev_mover, prev_opponent = meta_row['prev_white'], meta_row['prev_black']
            hours_since_mover = meta_row['hours_since_white']
            hours_since_opponent = meta_row['hours_since_black']
            has_history_mover = meta_row['has_history_white']
            has_history_opponent = meta_row['has_history_black']
        else:
            mover, opponent = meta_row['black'], meta_row['white']
            mover_elo, opponent_elo = meta_row['black_elo'], meta_row['white_elo']
            mover_title, opponent_title = meta_row['black_title'], meta_row['white_title']
            mover_rating_diff = meta_row['black_rating_diff']
            opponent_rating_diff = meta_row['white_rating_diff']
            mover_result = 1.0 - meta_row['result']
            past_mover, past_opponent = meta_row['past_black'], meta_row['past_white']
            prev_mover, prev_opponent = meta_row['prev_black'], meta_row['prev_white']
            hours_since_mover = meta_row['hours_since_black']
            hours_since_opponent = meta_row['hours_since_white']
            has_history_mover = meta_row['has_history_black']
            has_history_opponent = meta_row['has_history_white']

        row = dict(base_row)
        row['ply_played'] = k
        row['fen'] = board.fen()
        row['next_move'] = moves_list[k].uci()
        row['mover_is_white'] = mover_is_white
        row['mover'] = mover
        row['opponent'] = opponent
        row['mover_elo'] = mover_elo
        row['opponent_elo'] = opponent_elo
        row['mover_title'] = mover_title
        row['opponent_title'] = opponent_title
        row['mover_rating_diff'] = mover_rating_diff if mover_rating_diff is not None else ''
        row['opponent_rating_diff'] = opponent_rating_diff if opponent_rating_diff is not None else ''
        row['mover_result'] = mover_result
        row['mover_clock'] = mover_clock
        row['opponent_clock'] = opponent_clock
        row['past_mover'] = past_mover
        row['past_opponent'] = past_opponent
        row['prev_mover'] = prev_mover
        row['prev_opponent'] = prev_opponent
        row['hours_since_mover'] = hours_since_mover if hours_since_mover is not None else ''
        row['hours_since_opponent'] = hours_since_opponent if hours_since_opponent is not None else ''
        row['has_history_mover'] = has_history_mover
        row['has_history_opponent'] = has_history_opponent
        row['rematch'] = meta_row['rematch']
        rows.append(row)

    return rows

# (f) MEMORY / DISK

def _release_memory() -> None:
    gc.collect()
    try:
        import ctypes
        ctypes.CDLL('libc.so.6').malloc_trim(0)
    except Exception:
        pass

def _check_disk_space(path: str, required_gb: float) -> None:
    """Raises if path's filesystem has less than required_gb free."""
    os.makedirs(path, exist_ok=True)
    free_gb = shutil.disk_usage(path).free / (1024 ** 3)
    if free_gb < required_gb:
        raise RuntimeError(
            f'Not enough free disk space at {path}: {free_gb:.1f}GB free, need at least {required_gb:.1f}GB.'
        )

# (g) STREAMED EXTRACTION

def run(csv_path: str, pgn_path: str, out_path: str) -> dict:
    """Streams the game CSV and PGN cache in lockstep, batches games through the worker pool,
    and writes sampled positions to a tmp file that's atomically renamed once done."""
    out_tmp = out_path + '.tmp'
    ctx = mp.get_context('spawn')

    n_games = 0
    n_eligible = 0
    n_written = 0
    n_skipped = 0

    with open(csv_path, newline='', encoding='utf-8') as csv_f, \
            open(out_tmp, 'w', newline='', encoding='utf-8') as out_f, \
            ctx.Pool(processes=NUM_WORKERS) as pool:

        writer = csv.DictWriter(out_f, fieldnames=POS_CSV_FIELDNAMES)
        writer.writeheader()

        total_size = os.path.getsize(pgn_path)
        with open(pgn_path, 'rb') as pgn_raw_f:
            progress_reader = _ProgressReader(pgn_raw_f, total_size, desc=os.path.basename(pgn_path))
            try:
                csv_reader = csv.DictReader(csv_f)
                pgn_iter = iter_pgn_cache(pgn_path, progress_reader)

                def flush_batch(batch: list) -> None:
                    nonlocal n_written, n_skipped
                    for res in pool.imap(_extract_positions_for_game, batch, chunksize=SAMPLE_CHUNKSIZE):
                        if not res:
                            n_skipped += 1
                            continue
                        writer.writerows(res)
                        n_written += len(res)

                batch = []
                n_batches = 0
                last_progress_at = 0
                for row, pgn_block in zip_longest(csv_reader, pgn_iter, fillvalue=_MISSING):
                    if row is _MISSING or pgn_block is _MISSING:
                        raise RuntimeError(
                            f'{csv_path} and {pgn_path} have different row counts -- mismatched inputs.'
                        )
                    pgn_game_id, pgn_text = pgn_block
                    if row['game_id'] != pgn_game_id:
                        raise RuntimeError(
                            f"Order mismatch at game {n_games}: csv game_id={row['game_id']!r} "
                            f'vs pgn game_id={pgn_game_id!r}.'
                        )

                    n_games += 1
                    meta_row = _parse_meta_row(row)
                    if meta_row['ply_count'] < MIN_GAME_PLIES:
                        continue
                    n_eligible += 1

                    batch.append((n_games, pgn_text, meta_row))
                    if len(batch) >= POSITION_BATCH_SIZE:
                        flush_batch(batch)
                        batch = []
                        n_batches += 1
                        if n_written - last_progress_at >= PROGRESS_EVERY_N_POSITIONS:
                            print(f'Games seen: {n_games:,}, eligible: {n_eligible:,}, '
                                  f'positions written: {n_written:,}, games skipped: {n_skipped:,}')
                            last_progress_at = n_written
                        if n_batches % GC_EVERY_N_BATCHES == 0:
                            _release_memory()

                if batch:
                    flush_batch(batch)
            finally:
                progress_reader.close()

    os.replace(out_tmp, out_path)

    print(f'Done. Games seen: {n_games:,}, eligible: {n_eligible:,}, '
          f'positions written: {n_written:,}, games skipped: {n_skipped:,}')
    return {'n_games': n_games, 'n_eligible': n_eligible, 'n_written': n_written, 'n_skipped': n_skipped}

# (h) CLI

def _validate_config() -> None:
    """Validates sampling/batch constants, exiting with a clear message if any are invalid."""
    if MIN_PLIES_PLAYED < 0:
        sys.exit('MIN_PLIES_PLAYED must be >= 0.')
    if PLIES_PER_SAMPLE <= 0:
        sys.exit('PLIES_PER_SAMPLE must be > 0.')
    if POSITION_BATCH_SIZE <= 0:
        sys.exit('POSITION_BATCH_SIZE must be > 0.')

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parses split, --game-dir, --out-dir CLI arguments."""
    parser = argparse.ArgumentParser(
        description='Samples positions for one game-data split, reading its CSV and matching '
                     'PGN cache from run_game_reader.py in lockstep.'
    )
    parser.add_argument('split', choices=SPLITS, help='Which split to process: eda, train, val, or test.')
    parser.add_argument('--game-dir', default=GAME_DIR, help=f'Default: {GAME_DIR}')
    parser.add_argument('--out-dir', default=POS_DIR, help=f'Default: {POS_DIR}')
    return parser.parse_args(argv)

def main(argv: list[str] | None = None) -> None:
    """Reads args, validates config, locates the split's files, and runs extraction."""
    _validate_config()
    args = parse_args(argv)
    os.makedirs(args.out_dir, exist_ok=True)
    _check_disk_space(args.out_dir, MIN_FREE_DISK_GB)

    csv_path, pgn_path = locate_split_files(args.split, args.game_dir)
    _validate_csv_schema(csv_path)

    basename = os.path.basename(csv_path)
    tag = basename[len(f'game_{args.split}_'):-len('.csv')]
    out_path = os.path.join(args.out_dir, f'pos_{args.split}_{tag}.csv')

    print(f'Split: {args.split}')
    print(f'Game data: {csv_path}')
    print(f'PGN cache: {pgn_path}')
    print(f'Output: {out_path}')

    run(csv_path, pgn_path, out_path)

####################
# CLASSES
####################

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