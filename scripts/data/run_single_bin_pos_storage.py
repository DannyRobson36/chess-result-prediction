"""
run_single_bin_pos_storage.py
Builds one single-elo-bin training CSV per elo bin from run_pos_reader.py's train output,
using the same elo-gap threshold filtering as run_pos_storage.py's 'res' variant (fit and
applied to train itself) before a random, game-level sample within each bin.

    python run_single_bin_pos_storage.py

Latest changes: 20/08/26:
- Altered to 500k size due to >=2200 bin limitations
"""

import os
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from scripts.config import POS_DIR

import argparse
import gc
import glob
import shutil
from collections.abc import Iterator
from typing import IO

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

####################
# CONSTANTS
####################

LOCAL_DATA_DIR = os.environ.get('LOCAL_DATA_DIR', '/content/local_data')

# Target position count for each elo bin's single-bin training CSV.
SINGLE_BIN_SIZE = 500_000

ELO_LOWER = 800
ELO_UPPER = 2200
ELO_STEP = 200
TAILS = True
GAP_BIN_WIDTH = 10
ELO_GAP_TARGET_SCORE = 0.55

RANDOM_STATE = 0

# Cap on games used to fit elo-gap thresholds, always a random subsample of train.
ELO_GAP_FIT_SAMPLE_SIZE = 3_000_000

# Row count per streamed read/write batch, for CSV, Parquet, and final-write passes.
STREAM_BATCH_SIZE = 500_000

MIN_FREE_DISK_GB = 60.0

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

# Dtypes applied at CSV-read time. Title/speed/time_control/termination/eco are read as plain
# strings, not pandas 'category' -- categorical dtype is for in-memory feature prep, not Parquet
# storage, and Parquet's own dictionary encoding already handles repeated strings efficiently.
POS_READ_DTYPES = {
    'game_id': 'string', 'mover': 'string', 'opponent': 'string',
    'mover_elo': 'int32', 'opponent_elo': 'int32',
    'mover_title': 'string', 'opponent_title': 'string',
    'mover_rating_diff': 'Int64', 'opponent_rating_diff': 'Int64',
    'speed': 'string', 'time_control': 'string', 'termination': 'string',
    'eco': 'string', 'ply_count': 'int32', 'ply_played': 'int32',
    'fen': 'string', 'next_move': 'string',
    'mover_clock': 'int32', 'opponent_clock': 'int32', 'mover_result': 'float32',
    'past_mover': 'float32', 'past_opponent': 'float32',
    'prev_mover': 'float32', 'prev_opponent': 'float32',
    'hours_since_mover': 'float32', 'hours_since_opponent': 'float32',
}
POS_BOOL_COLS = ['mover_is_white', 'has_history_mover', 'has_history_opponent', 'rematch']
POS_PARSE_DATES = ['datetime']

# Title columns needing the no-title sentinel relabeled.
TITLE_COLS = ('mover_title', 'opponent_title')
# Raw sentinel run_game_reader.py/run_reader_unfiltered.py write for "no title".
TITLE_NO_TITLE_RAW = 'None'
# Relabeled value: avoids colliding with the default NA-string list most CSV/DataFrame readers use.
TITLE_NO_TITLE_VALUE = 'no_title'

# Columns needed for game-level allocation/threshold decisions.
GAME_LEVEL_COLS = ['game_id', 'mover_elo', 'opponent_elo', 'mover_result']

SINGLE_BIN_TRAIN_OUT = os.path.join(POS_DIR, 'train_data', 'single_bin')

####################
# FUNCTIONS
####################

# (a) DISK I/O

def locate_split_csv(split: str, pos_dir: str) -> str:
    """Finds the single pos_{split}_*.csv written by run_pos_reader.py."""
    matches = sorted(glob.glob(os.path.join(pos_dir, f'pos_{split}_*.csv')))
    if len(matches) == 0:
        sys.exit(f'No pos_{split}_*.csv found in {pos_dir} -- run run_pos_reader.py {split} first.')
    if len(matches) > 1:
        sys.exit(f'Expected exactly one pos_{split}_*.csv in {pos_dir}, found {len(matches)}: {matches}')
    return matches[0]

def _check_disk_space(path: str, required_gb: float, label: str = '') -> None:
    """Raises if path's filesystem has less than required_gb free."""
    os.makedirs(path, exist_ok=True)
    free_gb = shutil.disk_usage(path).free / (1024 ** 3)
    if free_gb < required_gb:
        raise RuntimeError(
            f"Not enough free disk space at {path}{f' ({label})' if label else ''}: "
            f'{free_gb:.1f}GB free, need at least {required_gb:.1f}GB.'
        )

def _read_csv_chunks(csv_source: IO[bytes]) -> Iterator[pd.DataFrame]:
    """Yields correctly-typed, title-relabeled chunks of csv_source, streamed."""
    reader = pd.read_csv(
        csv_source, dtype=POS_READ_DTYPES, parse_dates=POS_PARSE_DATES,
        true_values=['True'], false_values=['False'],
        keep_default_na=False, na_values=[''],
        chunksize=STREAM_BATCH_SIZE, low_memory=False,
    )
    for chunk in reader:
        chunk = chunk[POS_CSV_FIELDNAMES]
        for col in TITLE_COLS:
            chunk[col] = chunk[col].replace(TITLE_NO_TITLE_RAW, TITLE_NO_TITLE_VALUE)
        for col in POS_BOOL_COLS:
            chunk[col] = chunk[col].astype('bool')
        yield chunk

def csv_to_parquet(csv_path: str, parquet_path: str) -> str:
    """Converts a CSV to Parquet on local disk, streaming in bounded-size chunks, always overwriting."""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f'csv_to_parquet: source CSV not found: {csv_path}')

    os.makedirs(os.path.dirname(parquet_path), exist_ok=True)
    tmp_path = parquet_path + '.tmp'
    file_size = os.path.getsize(csv_path)

    writer = None
    try:
        with open(csv_path, 'rb') as raw_file:
            with tqdm(total=file_size, desc=os.path.basename(csv_path),
                      unit='B', unit_scale=True, unit_divisor=1024) as pbar:
                last_pos = 0
                for chunk in _read_csv_chunks(raw_file):
                    table = pa.Table.from_pandas(chunk, preserve_index=False)
                    if writer is None:
                        writer = pq.ParquetWriter(tmp_path, table.schema, compression='zstd')
                    writer.write_table(table)

                    current_pos = raw_file.tell()
                    if current_pos > last_pos:
                        pbar.update(current_pos - last_pos)
                        last_pos = current_pos

                if last_pos < file_size:
                    pbar.update(file_size - last_pos)
    finally:
        if writer is not None:
            writer.close()
    os.replace(tmp_path, parquet_path)
    return parquet_path

# (b) GAME-LEVEL AGGREGATION

def _aggregate_game_level(batches: Iterator[pd.DataFrame], game_id_col: str = 'game_id') -> pd.DataFrame:
    """Incrementally aggregates streamed batches into one row per game_id with a running n_positions count."""
    agg: dict[str, dict] = {}
    for chunk in batches:
        grouped = chunk.groupby(game_id_col, sort=False)
        first = grouped.first()
        counts = grouped.size()
        for gid, row in first.iterrows():
            entry = agg.get(gid)
            if entry is None:
                agg[gid] = {**row.to_dict(), 'n_positions': int(counts[gid])}
            else:
                entry['n_positions'] += int(counts[gid])

    game_df = pd.DataFrame.from_dict(agg, orient='index')
    game_df.index.name = game_id_col
    return game_df.reset_index()

def game_level_frame_from_parquet(parquet_path: str, columns: list = GAME_LEVEL_COLS,
                                   game_id_col: str = 'game_id') -> pd.DataFrame:
    """Streams parquet_path in row-group batches, returning one row per game_id."""
    pf = pq.ParquetFile(parquet_path)

    def batches() -> Iterator[pd.DataFrame]:
        with tqdm(total=pf.metadata.num_rows, desc=os.path.basename(parquet_path),
                  unit='rows', unit_scale=True) as pbar:
            for batch in pf.iter_batches(columns=columns, batch_size=STREAM_BATCH_SIZE):
                chunk = batch.to_pandas()
                pbar.update(len(chunk))
                yield chunk

    return _aggregate_game_level(batches(), game_id_col)

# (c) ELO BINNING / GAP-THRESHOLD FITTING

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

def _dedup_to_game_level(df: pd.DataFrame, game_id_col: str = 'game_id') -> pd.DataFrame:
    """Returns df unchanged if game_id_col is already unique, else deduplicates and sorts."""
    if df[game_id_col].is_unique:
        return df
    return df.drop_duplicates(subset=game_id_col).sort_values(game_id_col).reset_index(drop=True)

def _mean_bin_gap_and_res_better(df: pd.DataFrame, lower: int, upper: int, step: int, tails: bool) -> tuple:
    """Adds mean_bin, elo_gap, and res_better in one pass, one copy."""
    edges = list(range(lower, upper + 1, step))
    bin_edges = [-np.inf] + edges + [np.inf] if tails else edges

    df = df.copy()
    df['elo_gap'] = (df['mover_elo'] - df['opponent_elo']).abs()
    df['res_better'] = np.where(df['mover_elo'] >= df['opponent_elo'],
                                 df['mover_result'], 1 - df['mover_result'])
    mean_elo = (df['mover_elo'] + df['opponent_elo']) / 2
    mean_bin = pd.cut(mean_elo, bins=bin_edges, labels=False, right=False)

    valid = mean_bin.notna()
    df = df[valid].copy()
    df['mean_bin'] = mean_bin[valid].astype(int)

    return df, bin_edges

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
                            on_missing: str = 'keep_all', fit_sample_size: int = ELO_GAP_FIT_SAMPLE_SIZE,
                            random_state: int = RANDOM_STATE) -> dict:
    """Fits a per-elo-bin gap threshold on a random sample of up to fit_sample_size games from df_train."""
    if on_missing not in ('keep_all', 'drop_all'):
        raise ValueError(f"on_missing must be 'keep_all' or 'drop_all', got {on_missing!r}")

    game_df = _dedup_to_game_level(df_train, game_id_col)
    if len(game_df) > fit_sample_size:
        game_df = game_df.sample(n=fit_sample_size, random_state=random_state).reset_index(drop=True)

    binned, bin_edges = _mean_bin_gap_and_res_better(game_df, lower_elo, upper_elo, bin_size, tails)
    labels = elo_bin_labels(bin_edges)
    n_bins = len(bin_edges) - 1

    global_threshold = _compute_gap_threshold(binned, target_score, gap_bin_width)
    fallback_value = np.inf if on_missing == 'keep_all' else -1

    thresholds = {}
    for i in range(n_bins):
        group = binned[binned['mean_bin'] == i]
        t = _compute_gap_threshold(group, target_score, gap_bin_width)
        if t is None:
            t = global_threshold if global_threshold is not None else fallback_value
        thresholds[i] = t

    return {
        'thresholds': thresholds, 'bin_edges': bin_edges, 'labels': labels,
        'config': dict(target_score=target_score, lower_elo=lower_elo, upper_elo=upper_elo,
                        bin_size=bin_size, tails=tails, gap_bin_width=gap_bin_width, game_id_col=game_id_col),
    }

def apply_elo_gap_thresholds(df: pd.DataFrame, fit_result: dict) -> list:
    """Applies an already-fitted set of thresholds to a (possibly different) split."""
    cfg = fit_result['config']
    thresholds = fit_result['thresholds']
    game_id_col = cfg.get('game_id_col', 'game_id')

    game_df = _dedup_to_game_level(df, game_id_col)
    binned, bin_edges = _mean_bin_gap_and_res_better(game_df, cfg['lower_elo'], cfg['upper_elo'],
                                                       cfg['bin_size'], cfg['tails'])
    assert bin_edges == fit_result['bin_edges'], \
        'bin_edges mismatch between fit and apply -- should be impossible; investigate.'

    row_thresholds = binned['mean_bin'].map(thresholds)
    keep_mask = binned['elo_gap'] <= row_thresholds
    return binned.loc[keep_mask, game_id_col].tolist()

# (d) SINGLE-BIN GAME-LEVEL SAMPLING

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

def sample_bin_games(bin_pool: pd.DataFrame, target: int, random_state: int,
                      game_id_col: str = 'game_id') -> tuple[list, int]:
    """Randomly takes whole games from bin_pool until target positions are reached, or all
    games are exhausted if the bin is too sparse to fill target."""
    if bin_pool.empty:
        return [], 0
    pos_lookup = bin_pool.set_index(game_id_col)['n_positions'].to_dict()
    rng = np.random.default_rng(random_state)
    return _cumulative_take_head(bin_pool[game_id_col].tolist(), pos_lookup, target, rng)

# (e) POSITION POOL LOADING & WRITING

def load_pool_positions(parquet_path: str, pool_game_ids: set, game_id_col: str = 'game_id') -> pd.DataFrame:
    """Streams parquet_path, keeping only rows whose game_id is in pool_game_ids, into one in-memory frame."""
    pf = pq.ParquetFile(parquet_path)
    parts = []
    with tqdm(total=pf.metadata.num_rows, desc='Loading pool positions', unit='rows', unit_scale=True) as pbar:
        for batch in pf.iter_batches(columns=POS_CSV_FIELDNAMES, batch_size=STREAM_BATCH_SIZE):
            chunk = batch.to_pandas()
            pbar.update(len(chunk))
            mask = chunk[game_id_col].isin(pool_game_ids)
            if mask.any():
                parts.append(chunk[mask])
    return pd.concat(parts, ignore_index=True)

def write_position_csv_with_ids(df: pd.DataFrame, out_dir: str, filename: str) -> None:
    """Writes df's positions to out_dir/filename and its unique game_id list to out_dir/ids, both atomically."""
    os.makedirs(out_dir, exist_ok=True)
    ids_dir = os.path.join(out_dir, 'ids')
    os.makedirs(ids_dir, exist_ok=True)

    path = os.path.join(out_dir, filename)
    tmp_path = path + '.tmp'
    with tqdm(total=len(df), desc=filename, unit='rows', unit_scale=True) as pbar:
        with open(tmp_path, 'w', newline='', encoding='utf-8') as f:
            df.iloc[:0].to_csv(f, index=False)
            for start in range(0, len(df), STREAM_BATCH_SIZE):
                chunk = df.iloc[start:start + STREAM_BATCH_SIZE]
                chunk.to_csv(f, index=False, header=False)
                pbar.update(len(chunk))
    os.replace(tmp_path, path)

    ids_path = os.path.join(ids_dir, filename.replace('.csv', '_ids.csv'))
    ids_tmp_path = ids_path + '.tmp'
    df[['game_id']].drop_duplicates().to_csv(ids_tmp_path, index=False)
    os.replace(ids_tmp_path, ids_path)

def _size_tag(n: int) -> str:
    if n >= 1_000_000:
        value, unit = n / 1_000_000, 'm'
    else:
        value, unit = n / 1_000, 'k'
    if value == int(value):
        return f'{int(value)}{unit}'
    return f'{value:g}'.replace('.', 'p') + unit

# (f) PIPELINE

def run_pipeline() -> dict[str, int]:
    """Builds one single-bin training CSV per elo bin, fitting and applying elo-gap
    restriction to train itself. Returns {bin_label: actual_size_written}."""
    _check_disk_space(LOCAL_DATA_DIR, MIN_FREE_DISK_GB, 'LOCAL_DATA_DIR')

    train_csv = locate_split_csv('train', POS_DIR)
    parquet_dir = os.path.join(LOCAL_DATA_DIR, 'parquet')
    train_pq = csv_to_parquet(train_csv, os.path.join(parquet_dir, 'train.parquet'))

    gl_train = game_level_frame_from_parquet(train_pq)

    fit_result = fit_elo_gap_thresholds(gl_train, target_score=ELO_GAP_TARGET_SCORE,
                                         fit_sample_size=ELO_GAP_FIT_SAMPLE_SIZE, on_missing='keep_all')

    res_ids = apply_elo_gap_thresholds(gl_train, fit_result)
    gl_train_res = gl_train[gl_train['game_id'].isin(res_ids)].reset_index(drop=True)

    binned, bin_edges = _mean_bin_gap_and_res_better(gl_train_res, ELO_LOWER, ELO_UPPER, ELO_STEP, TAILS)
    labels = elo_bin_labels(bin_edges)

    summary = {}
    for bin_idx, label in enumerate(tqdm(labels, desc='Building single-bin datasets', unit='bin')):
        bin_pool = binned[binned['mean_bin'] == bin_idx].reset_index(drop=True)
        kept_ids, total_positions = sample_bin_games(bin_pool, SINGLE_BIN_SIZE, random_state=RANDOM_STATE + bin_idx)

        if not kept_ids:
            summary[label] = 0
            continue

        rows = load_pool_positions(train_pq, set(kept_ids))

        trim = total_positions - min(total_positions, SINGLE_BIN_SIZE)
        if trim > 0:
            boundary_game = kept_ids[-1]
            boundary_idx = rows.index[rows['game_id'] == boundary_game]
            keep_n = max(len(boundary_idx) - trim, 0)
            rng = np.random.default_rng(RANDOM_STATE + bin_idx + 1000)
            keep_idx = rng.choice(boundary_idx.to_numpy(), size=min(keep_n, len(boundary_idx)), replace=False)
            drop_idx = boundary_idx.difference(pd.Index(keep_idx))
            rows = rows.drop(index=drop_idx)

        rows = rows.sample(frac=1, random_state=RANDOM_STATE + bin_idx).reset_index(drop=True)

        fname = f'train_bin{bin_idx + 1}_{_size_tag(SINGLE_BIN_SIZE)}.csv'
        write_position_csv_with_ids(rows, SINGLE_BIN_TRAIN_OUT, fname)
        summary[label] = len(rows)
        del rows
        gc.collect()

    return summary

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """No CLI arguments; parses only for --help consistency with other CLI scripts."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    return parser.parse_args(argv)

def main(argv: list[str] | None = None) -> None:
    """Reads args and runs the single-bin pipeline for every elo bin."""
    parse_args(argv)
    summary = run_pipeline()

    print(f'target size: {SINGLE_BIN_SIZE:,}')
    for label, size in summary.items():
        print(f'bin {label}: {size:,}')

####################
# CLASSES
####################

if __name__ == '__main__':
    main()