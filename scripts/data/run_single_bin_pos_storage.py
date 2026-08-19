"""
run_single_bin_pos_storage.py
Builds one fixed-size training CSV sampled from a single elo bin, using the same elo-gap
threshold filtering as run_pos_storage.py's 'res' variant before sampling.

    python run_single_bin_pos_storage.py --bin 3                  normal run, bin 3 (1000-1199)
    python run_single_bin_pos_storage.py --bin 3 --limit 200000   dry run on a row slice
    python run_single_bin_pos_storage.py --bin 3 --force          rebuild even if output exists

Latest changes: 19/08/26:
- Initial commit
"""

import os
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from scripts.config import POS_DIR

import argparse
import functools
import gc
import glob
import logging
import shutil
import time
import traceback
from datetime import datetime

import duckdb
import numpy as np
import pandas as pd
import psutil

####################
# CONSTANTS
####################

LOCAL_DATA_DIR = os.environ.get('LOCAL_DATA_DIR', '/content/local_data')

# Training set size this script builds.
SINGLE_BIN_TRAIN_SIZE = 2_500_000

ELO_LOWER = 800
ELO_UPPER = 2200
ELO_STEP = 200
TAILS = True

# Bin id -> label (--bin):
#   1  <800        4  1200-1399    7  1800-1999
#   2  800-999     5  1400-1599    8  2000-2199
#   3  1000-1199   6  1600-1799    9  >=2200

GAP_BIN_WIDTH = 10
ELO_GAP_TARGET_SCORE = 0.55

RANDOM_STATE = 0

_TOTAL_RAM_GB = psutil.virtual_memory().total / (1024 ** 3)
DUCKDB_MEMORY_LIMIT_GB = float(os.environ.get('DUCKDB_MEMORY_LIMIT_GB', max(2.0, _TOTAL_RAM_GB * 0.7)))
MIN_FREE_DISK_GB = float(os.environ.get('MIN_FREE_DISK_GB', 60))

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

POS_DUCKDB_TYPES = {
    'game_id': 'VARCHAR', 'mover': 'VARCHAR', 'opponent': 'VARCHAR',
    'mover_elo': 'INTEGER', 'opponent_elo': 'INTEGER',
    'mover_title': 'VARCHAR', 'opponent_title': 'VARCHAR',
    'mover_rating_diff': 'INTEGER', 'opponent_rating_diff': 'INTEGER',
    'mover_is_white': 'BOOLEAN',
    'datetime': 'TIMESTAMP', 'speed': 'VARCHAR', 'time_control': 'VARCHAR',
    'termination': 'VARCHAR', 'eco': 'VARCHAR', 'ply_count': 'INTEGER',
    'ply_played': 'INTEGER', 'fen': 'VARCHAR', 'next_move': 'VARCHAR',
    'mover_clock': 'INTEGER', 'opponent_clock': 'INTEGER', 'mover_result': 'FLOAT',
    'past_mover': 'FLOAT', 'past_opponent': 'FLOAT', 'prev_mover': 'FLOAT', 'prev_opponent': 'FLOAT',
    'hours_since_mover': 'FLOAT', 'hours_since_opponent': 'FLOAT',
    'has_history_mover': 'BOOLEAN', 'has_history_opponent': 'BOOLEAN', 'rematch': 'BOOLEAN',
}

# Title columns needing the no-title sentinel relabeled (see _select_fragment).
TITLE_COLS = ('mover_title', 'opponent_title')
# Raw sentinel run_game_reader.py/run_reader_unfiltered.py write for "no title".
TITLE_NO_TITLE_RAW = 'None'
# Relabeled value: avoids colliding with the default NA-string list most CSV/DataFrame readers use.
TITLE_NO_TITLE_VALUE = 'no_title'

SINGLE_BIN_TRAIN_OUT = os.path.join(POS_DIR, 'train', 'single_bin')

####################
# FUNCTIONS
####################

# (a) DISK I/O

def get_connection(tmp_dir: str, memory_limit_gb: float) -> duckdb.DuckDBPyConnection:
    """One DuckDB connection for the run. tmp_dir must be local disk, not Drive."""
    os.makedirs(tmp_dir, exist_ok=True)
    con = duckdb.connect()
    con.execute(f"PRAGMA temp_directory='{tmp_dir}'")
    con.execute(f"PRAGMA memory_limit='{memory_limit_gb:.1f}GB'")
    return con

def locate_split_csv(split: str, pos_dir: str) -> str:
    """Finds the single pos_{split}_*.csv written by run_pos_reader.py."""
    matches = sorted(glob.glob(os.path.join(pos_dir, f'pos_{split}_*.csv')))
    if len(matches) == 0:
        sys.exit(f'No pos_{split}_*.csv found in {pos_dir} -- run run_pos_reader.py {split} first.')
    if len(matches) > 1:
        sys.exit(f'Expected exactly one pos_{split}_*.csv in {pos_dir}, found {len(matches)}: {matches}')
    return matches[0]

def _select_fragment(col: str) -> str:
    """Returns one column's SELECT fragment for csv_to_parquet: a plain CAST, except for
    TITLE_COLS, which first relabel TITLE_NO_TITLE_RAW to TITLE_NO_TITLE_VALUE."""
    if col in TITLE_COLS:
        return (f"CAST(CASE WHEN \"{col}\" = '{TITLE_NO_TITLE_RAW}' THEN '{TITLE_NO_TITLE_VALUE}' "
                f"ELSE \"{col}\" END AS {POS_DUCKDB_TYPES[col]}) AS \"{col}\"")
    return f'CAST("{col}" AS {POS_DUCKDB_TYPES[col]}) AS "{col}"'

def csv_to_parquet(con: duckdb.DuckDBPyConnection, csv_path: str, parquet_path: str, force: bool = False,
                    limit_rows: int | None = None) -> str:
    """Converts a CSV to Parquet on local disk once, skipping if it's already there."""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f'csv_to_parquet: source CSV not found: {csv_path}')

    if force or not os.path.exists(parquet_path):
        os.makedirs(os.path.dirname(parquet_path), exist_ok=True)
        tmp_path = parquet_path + '.tmp'
        select_expr = ', '.join(_select_fragment(c) for c in POS_CSV_FIELDNAMES)
        limit_clause = f' LIMIT {int(limit_rows)}' if limit_rows else ''
        con.execute(f"""
            COPY (SELECT {select_expr} FROM read_csv_auto('{csv_path}', SAMPLE_SIZE=-1){limit_clause})
            TO '{tmp_path}' (FORMAT PARQUET, COMPRESSION ZSTD)
        """)
        os.replace(tmp_path, parquet_path)
    return parquet_path

def _check_disk_space(path: str, required_gb: float, label: str = '') -> None:
    """Raises if path's filesystem has less than required_gb free."""
    os.makedirs(path, exist_ok=True)
    free_gb = shutil.disk_usage(path).free / (1024 ** 3)
    if free_gb < required_gb:
        raise RuntimeError(
            f"Not enough free disk space at {path}{f' ({label})' if label else ''}: "
            f'{free_gb:.1f}GB free, need at least {required_gb:.1f}GB.'
        )

def _files_expr(parquet_paths: list[str]) -> str:
    paths = ', '.join(f"'{p}'" for p in parquet_paths)
    return f'read_parquet([{paths}])'

def game_level_frame(con: duckdb.DuckDBPyConnection, parquet_paths: list[str],
                      game_id_col: str = 'game_id', agg_cols: list[str] | None = None) -> pd.DataFrame:
    """One row per game_id via DuckDB; pass agg_cols to limit columns kept."""
    if agg_cols is None:
        cols = con.execute(f'DESCRIBE SELECT * FROM {_files_expr(parquet_paths)} LIMIT 0').df()['column_name']
        agg_cols = [c for c in cols if c != game_id_col]
    select_cols = ', '.join(f'any_value({c}) AS {c}' for c in agg_cols)
    return con.execute(f"""
        SELECT {game_id_col}, {select_cols}, count(*) AS n_positions
        FROM {_files_expr(parquet_paths)}
        GROUP BY {game_id_col}
    """).df()

def rows_for_game_ids(con: duckdb.DuckDBPyConnection, parquet_paths: list[str], game_ids,
                       game_id_col: str = 'game_id') -> pd.DataFrame:
    """Pulls the full rows for the given game_ids back out of the Parquet files."""
    con.register('_wanted_game_ids', pd.DataFrame({game_id_col: list(game_ids)}))
    try:
        result = con.execute(f"""
            SELECT t.* FROM {_files_expr(parquet_paths)} t
            JOIN _wanted_game_ids w USING ({game_id_col})
        """).df()
    finally:
        con.unregister('_wanted_game_ids')
    return result

# (b) ELO BINNING / GAP-THRESHOLD FITTING

def _compute_bin_edges(lower: int, upper: int, step: int, tails: bool) -> list:
    """Builds the elo bin edge list, open-ended on both sides when tails is True."""
    edges = list(range(lower, upper + 1, step))
    return [-np.inf] + edges + [np.inf] if tails else edges

def elo_bin_labels(bin_edges: list) -> list[str]:
    labels = []
    for i in range(len(bin_edges) - 1):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        if lo == -np.inf:
            labels.append(f'<{int(hi)}')
        elif hi == np.inf:
            labels.append(f'>={int(lo)}')
        else:
            labels.append(f'{int(lo)}-{int(hi) - 1}')
    return labels

BIN_EDGES = _compute_bin_edges(ELO_LOWER, ELO_UPPER, ELO_STEP, TAILS)
BIN_LABELS = elo_bin_labels(BIN_EDGES)
BIN_ID_TO_LABEL = {i + 1: label for i, label in enumerate(BIN_LABELS)}

def _dedup_to_game_level(df: pd.DataFrame, game_id_col: str = 'game_id') -> pd.DataFrame:
    n_before = len(df)
    game_df = df.drop_duplicates(subset=game_id_col).sort_values(game_id_col).reset_index(drop=True)
    if len(game_df) != n_before:
        print(f'  [_dedup_to_game_level] collapsed {n_before:,} rows -> {len(game_df):,} games')
    return game_df

def _mean_bin_and_gap(df: pd.DataFrame, lower: int, upper: int, step: int, tails: bool) -> tuple:
    bin_edges = _compute_bin_edges(lower, upper, step, tails)

    df = df.copy()
    df['elo_gap'] = (df['mover_elo'] - df['opponent_elo']).abs()
    mean_elo = (df['mover_elo'] + df['opponent_elo']) / 2
    mean_bin = pd.cut(mean_elo, bins=bin_edges, labels=False, right=False)

    n_before = len(df)
    valid = mean_bin.notna()
    df = df[valid].copy()
    df['mean_bin'] = mean_bin[valid].astype(int)
    n_dropped = n_before - len(df)
    if n_dropped > 0:
        print(f'  [_mean_bin_and_gap] dropped {n_dropped:,} games outside elo range '
              f'[{lower}, {upper}) (tails={tails})')

    return df, bin_edges

def _res_better(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['res_better'] = np.where(df['mover_elo'] >= df['opponent_elo'],
                                 df['mover_result'], 1 - df['mover_result'])
    return df

def _find_threshold_crossing(bin_centres: list, values: list, target: float) -> float | None:
    pairs = [(c, v) for c, v in zip(bin_centres, values) if v is not None]
    if not pairs:
        return None
    centres, vals = zip(*pairs)
    above = [v >= target for v in vals]
    if not any(above) or all(above):
        return None
    idx = above.index(True)
    if idx == 0:
        return centres[0]
    x0, x1 = centres[idx - 1], centres[idx]
    y0, y1 = vals[idx - 1], vals[idx]
    return x0 + (target - y0) * (x1 - x0) / (y1 - y0)

def _compute_gap_threshold(group: pd.DataFrame, target_score: float, gap_bin_width: int) -> float | None:
    nonzero = group[group['elo_gap'] > 0]
    if nonzero.empty:
        return None
    gap_bin_idx = np.ceil(nonzero['elo_gap'] / gap_bin_width).astype(int) - 1
    stats = nonzero.groupby(gap_bin_idx)['res_better'].mean()
    max_idx = stats.index.max()
    bin_centres = [idx * gap_bin_width + gap_bin_width / 2 for idx in range(max_idx + 1)]
    expected = [stats.get(idx) for idx in range(max_idx + 1)]
    threshold = _find_threshold_crossing(bin_centres, expected, target_score)
    return round(threshold) if threshold is not None else None

def fit_elo_gap_thresholds(df_train: pd.DataFrame, target_score: float, game_id_col: str = 'game_id',
                            lower_elo: int = ELO_LOWER, upper_elo: int = ELO_UPPER, bin_size: int = ELO_STEP,
                            tails: bool = TAILS, gap_bin_width: int = GAP_BIN_WIDTH,
                            on_missing: str = 'keep_all') -> dict:
    """Fits a per-elo-bin gap threshold on train only; apply_elo_gap_thresholds reuses this
    on every other split without refitting."""
    if on_missing not in ('keep_all', 'drop_all'):
        raise ValueError(f"on_missing must be 'keep_all' or 'drop_all', got {on_missing!r}")

    game_df = _dedup_to_game_level(df_train, game_id_col)
    binned, bin_edges = _mean_bin_and_gap(game_df, lower_elo, upper_elo, bin_size, tails)
    binned = _res_better(binned)
    labels = elo_bin_labels(bin_edges)
    n_bins = len(bin_edges) - 1

    global_threshold = _compute_gap_threshold(binned, target_score, gap_bin_width)
    print(f'Global elo-gap threshold (all brackets pooled): {global_threshold}')

    fallback_value = np.inf if on_missing == 'keep_all' else -1
    thresholds = {}
    used_global_fallback = []
    used_hardcoded_fallback = []

    for i in range(n_bins):
        group = binned[binned['mean_bin'] == i]
        n_games = len(group)
        t = _compute_gap_threshold(group, target_score, gap_bin_width)
        if t is None:
            if global_threshold is not None:
                t = global_threshold
                used_global_fallback.append((i, labels[i], n_games))
            else:
                t = fallback_value
                used_hardcoded_fallback.append((i, labels[i], n_games))
        thresholds[i] = t

    if used_global_fallback:
        print(f'\n{len(used_global_fallback)} bracket(s) had no own threshold; fell back to global '
              f'({global_threshold}):')
        for i, label, n in used_global_fallback:
            print(f'  bin {i} ({label}): n={n:,} games in bracket')

    if used_hardcoded_fallback:
        print(f"\n{len(used_hardcoded_fallback)} bracket(s) had NO own threshold AND no usable global "
              f"fallback; used on_missing='{on_missing}' (threshold={fallback_value}):")
        for i, label, n in used_hardcoded_fallback:
            print(f'  bin {i} ({label}): n={n:,} games in bracket')

    print('\nPer-bracket thresholds:')
    for i in range(n_bins):
        print(f'  {labels[i]:>12}: gap <= {thresholds[i]}')

    return {
        'thresholds': thresholds, 'bin_edges': bin_edges, 'labels': labels,
        'config': dict(target_score=target_score, lower_elo=lower_elo, upper_elo=upper_elo,
                        bin_size=bin_size, tails=tails, gap_bin_width=gap_bin_width, game_id_col=game_id_col),
    }

def apply_elo_gap_thresholds(df: pd.DataFrame, fit_result: dict, verbose: bool = True) -> list:
    """Applies an already-fitted set of thresholds to a (possibly different) split."""
    cfg = fit_result['config']
    thresholds = fit_result['thresholds']
    game_id_col = cfg.get('game_id_col', 'game_id')

    game_df = _dedup_to_game_level(df, game_id_col)
    binned, bin_edges = _mean_bin_and_gap(game_df, cfg['lower_elo'], cfg['upper_elo'],
                                           cfg['bin_size'], cfg['tails'])
    assert bin_edges == fit_result['bin_edges'], \
        'bin_edges mismatch between fit and apply -- should be impossible; investigate.'

    row_thresholds = binned['mean_bin'].map(thresholds)
    keep_mask = binned['elo_gap'] <= row_thresholds
    kept_game_ids = binned.loc[keep_mask, game_id_col].tolist()

    if verbose:
        n_games_in = len(game_df)
        n_games_out_of_range = n_games_in - len(binned)
        n_games_over_gap = int((~keep_mask).sum())
        n_games_kept = len(kept_game_ids)
        n_rows_in = len(df)
        n_rows_kept = int(df[game_id_col].isin(kept_game_ids).sum())
        print(f'apply_elo_gap_thresholds (game level): {n_games_in:,} games in -> '
              f'{n_games_out_of_range:,} outside elo range, {n_games_over_gap:,} over gap threshold, '
              f'{n_games_kept:,} games kept')
        print(f'  input rows: {n_rows_in:,} -> {n_rows_kept:,} rows would be kept '
              f'({n_rows_in - n_rows_kept:,} dropped) if filtered by this game_id list')

    return kept_game_ids

def select_bin_pool(df: pd.DataFrame, bin_id: int, game_id_col: str = 'game_id') -> pd.DataFrame:
    """Restricts a game-level frame to the single elo bin identified by bin_id."""
    binned, bin_edges = _mean_bin_and_gap(df, ELO_LOWER, ELO_UPPER, ELO_STEP, TAILS)
    assert bin_edges == BIN_EDGES, 'bin_edges mismatch against BIN_EDGES; this should be impossible, investigate.'
    return binned[binned['mean_bin'] == bin_id - 1].reset_index(drop=True)

# (c) SINGLE-BIN GAME-LEVEL SAMPLING

def _cumulative_take_head(game_ids, pos_counts, target: float, rng: np.random.Generator) -> tuple:
    ids = list(game_ids)
    rng.shuffle(ids)
    kept, total = [], 0
    for gid in ids:
        kept.append(gid)
        total += pos_counts[gid]
        if total >= target:
            break
    return kept, total

def sample_single_bin(pool_df: pd.DataFrame, row_fetcher, target: int, game_id_col: str = 'game_id',
                       random_state: int = RANDOM_STATE) -> pd.DataFrame:
    """Takes random whole games from pool_df until target positions are reached, trims the one
    boundary game to hit target exactly, fetches and shuffles the resulting rows."""
    pos_lookup = pool_df.set_index(game_id_col)['n_positions'].to_dict()
    available = sum(pos_lookup.values())
    if available < target:
        raise RuntimeError(
            f'Bin has only {available:,} positions available across {len(pool_df):,} games; '
            f'needs {target:,} to build this training size from this bin.'
        )

    rng = np.random.default_rng(random_state)
    kept, total = _cumulative_take_head(pool_df[game_id_col].tolist(), pos_lookup, target, rng)

    rows = row_fetcher(kept)
    trim = total - target
    if trim > 0:
        boundary_game = kept[-1]
        boundary_rows = rows[rows[game_id_col] == boundary_game]
        drop_idx = boundary_rows.sample(n=min(trim, len(boundary_rows)), random_state=random_state).index
        rows = rows.drop(index=drop_idx)

    return rows.sample(frac=1, random_state=random_state).reset_index(drop=True)

# (d) LOGGING / PIPELINE HELPERS

def setup_logging() -> logging.Logger:
    log_dir = os.path.join(LOCAL_DATA_DIR, 'logs')
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f'run_single_bin_pos_storage_{datetime.now():%Y%m%d_%H%M%S}.log')

    logger = logging.getLogger('run_single_bin_pos_storage')
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter('%(asctime)s  %(levelname)-7s  %(message)s', '%H:%M:%S')
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(fmt)
    logger.addHandler(console)
    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(fmt)
    logger.addHandler(file_handler)

    logger.info(f'Logging to {log_path} (kept even if this session dies)')
    return logger

log = logging.getLogger('run_single_bin_pos_storage')

class Stage:
    """Times a stage and logs start/end/failure with the stage name."""

    def __init__(self, name: str):
        self.name = name

    def __enter__(self):
        log.info(f'START  {self.name}')
        self._t0 = time.time()
        return self

    def __exit__(self, exc_type, exc, tb):
        elapsed = time.time() - self._t0
        if exc_type is not None:
            log.error(f'FAILED {self.name} after {elapsed:.1f}s: {exc}')
            log.error(''.join(traceback.format_exception(exc_type, exc, tb)))
            return False
        log.info(f'DONE   {self.name} ({elapsed:.1f}s)')
        return False

def write_csv_safely(df: pd.DataFrame, path: str, force: bool) -> bool:
    """Writes df to path, skipping if it exists unless force=True."""
    if os.path.exists(path) and not force:
        log.info(f'  skip (exists): {path}')
        return False
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp_path = path + '.tmp'
    df.to_csv(tmp_path, index=False)
    os.replace(tmp_path, path)
    log.info(f'  wrote {len(df):,} rows -> {path}')
    return True

def _size_tag(n: int) -> str:
    if n >= 1_000_000:
        value, unit = n / 1_000_000, 'm'
    else:
        value, unit = n / 1_000, 'k'
    if value == int(value):
        return f'{int(value)}{unit}'
    return f'{value:g}'.replace('.', 'p') + unit

# (e) PIPELINE

def run_pipeline(bin_id: int, limit_rows: int | None, force: bool) -> dict:
    _check_disk_space(LOCAL_DATA_DIR, MIN_FREE_DISK_GB, 'LOCAL_DATA_DIR')

    train_csv = locate_split_csv('train', POS_DIR)
    con = get_connection(tmp_dir=os.path.join(LOCAL_DATA_DIR, 'duckdb_tmp'),
                          memory_limit_gb=DUCKDB_MEMORY_LIMIT_GB)
    if limit_rows:
        log.warning(f'--limit {limit_rows} set: this is a DRY RUN on a row slice, not a real build')

    parquet_dir = os.path.join(LOCAL_DATA_DIR, 'parquet')
    with Stage('csv_to_parquet: train'):
        train_pq = csv_to_parquet(con, train_csv, os.path.join(parquet_dir, 'train.parquet'),
                                   force=force, limit_rows=limit_rows)
    train_fetch = functools.partial(rows_for_game_ids, con, [train_pq])

    with Stage('game_level_frame: train'):
        gl_train = _dedup_to_game_level(game_level_frame(con, [train_pq]))
        log.info(f'  {len(gl_train):,} unique games')

    with Stage('fit_elo_gap_thresholds'):
        fit_result = fit_elo_gap_thresholds(gl_train, target_score=ELO_GAP_TARGET_SCORE, on_missing='keep_all')

    with Stage('apply_elo_gap_thresholds: train'):
        train_kept = apply_elo_gap_thresholds(gl_train, fit_result)
    gl_train_res = gl_train[gl_train['game_id'].isin(train_kept)].reset_index(drop=True)

    with Stage(f'select_bin_pool: bin {bin_id} ({BIN_ID_TO_LABEL[bin_id]})'):
        pool = select_bin_pool(gl_train_res, bin_id)
        log.info(f'  {len(pool):,} games, {pool["n_positions"].sum():,} positions available '
                 f'in bin {bin_id} ({BIN_ID_TO_LABEL[bin_id]}) after gap filtering')

    with Stage(f'sample_single_bin: target {SINGLE_BIN_TRAIN_SIZE:,} positions'):
        rows = sample_single_bin(pool, train_fetch, SINGLE_BIN_TRAIN_SIZE)

    n_rows = len(rows)
    size_tag = _size_tag(SINGLE_BIN_TRAIN_SIZE)
    out_path = os.path.join(SINGLE_BIN_TRAIN_OUT, f'train_bin{bin_id}_{size_tag}.csv')
    wrote = write_csv_safely(rows, out_path, force)

    con.close()
    del rows
    gc.collect()

    return {'bin_id': bin_id, 'bin_label': BIN_ID_TO_LABEL[bin_id], 'out_path': out_path,
            'wrote': wrote, 'n_rows': n_rows}

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parses --bin, --force, --limit CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--bin', type=int, required=True, choices=sorted(BIN_ID_TO_LABEL),
                         help='Elo bin id to build training data from (1-9). No default.')
    parser.add_argument('--force', action='store_true',
                         help='Rebuild the output CSV even if it already exists.')
    parser.add_argument('--limit', type=int, default=None,
                         help='Only read this many rows from the source CSV. Default: None (all).')
    return parser.parse_args(argv)

def main(argv: list[str] | None = None) -> None:
    """Reads args and runs the single-bin pipeline, logging progress and failures throughout."""
    args = parse_args(argv)

    global log
    log = setup_logging()

    log.info(f'POS_DIR (source/output CSVs) = {POS_DIR}')
    log.info(f'LOCAL_DATA_DIR (parquet/tmp)  = {LOCAL_DATA_DIR}')
    log.info(f'DuckDB memory limit           = {DUCKDB_MEMORY_LIMIT_GB:.1f}GB '
             f'(of {_TOTAL_RAM_GB:.1f}GB detected total RAM)')
    log.info(f'bin={args.bin} ({BIN_ID_TO_LABEL[args.bin]})  target_size={SINGLE_BIN_TRAIN_SIZE:,}  '
             f'force={args.force}  limit={args.limit}')

    t0 = time.time()
    try:
        result = run_pipeline(bin_id=args.bin, limit_rows=args.limit, force=args.force)
    except Exception:
        log.error('Pipeline aborted -- see traceback above for the failing stage.')
        raise

    elapsed = time.time() - t0
    status = 'wrote' if result['wrote'] else 'skipped (already existed)'
    log.info('=' * 60)
    log.info(f"DONE in {elapsed / 60:.1f} min. {status}: {result['out_path']} "
             f"({result['n_rows']:,} rows, bin {result['bin_id']} = {result['bin_label']})")

####################
# CLASSES
####################

if __name__ == '__main__':
    main()