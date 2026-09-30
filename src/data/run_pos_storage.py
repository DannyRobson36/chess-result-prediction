"""
run_pos_storage.py
Builds one split's (train, val, or test) model-ready position CSVs from run_pos_reader.py's
output, sampling a random game-level pool into memory, then sizing it down with elo-bin-
proportional, game-level selection so smaller datasets are strict subsets of larger ones.

    python run_pos_storage.py train
    python run_pos_storage.py val
    python run_pos_storage.py test

Latest changes: 20/08/26:
- One split (t/v/t) per-run and revert to using System-RAM
"""

import os
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from src.config import POS_DIR

import argparse
import gc
import glob
import heapq
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

SPLITS = ['train', 'val', 'test']

LOCAL_DATA_DIR = os.environ.get('LOCAL_DATA_DIR', '/content/local_data')

TRAIN_SIZES = [100_000, 500_000, 2_500_000, 12_500_000, 62_500_000]
VAL_TEST_SIZE = 900_000

# Target position count for each split's randomly-sampled in-memory game pool.
POOL_POSITIONS = {'train': 100_000_000, 'val': 50_000_000, 'test': 50_000_000}

ELO_LOWER = 800
ELO_UPPER = 2200
ELO_STEP = 200
TAILS = True
GAP_BIN_WIDTH = 10
ELO_GAP_TARGET_SCORE = 0.55

CONFIDENCE_Z = 3.0
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

TRAIN_OUT = os.path.join(POS_DIR, 'train_data')
VAL_OUT = os.path.join(POS_DIR, 'val_data')
VAL_OUT_WL = os.path.join(VAL_OUT, 'wl')
TEST_OUT = os.path.join(POS_DIR, 'test_data')
TEST_OUT_WL = os.path.join(TEST_OUT, 'wl')

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

def game_level_frame_from_csv(csv_path: str, columns: list = GAME_LEVEL_COLS,
                               game_id_col: str = 'game_id') -> pd.DataFrame:
    """Streams csv_path directly, without a Parquet conversion, returning one row per game_id."""
    dtype = {c: POS_READ_DTYPES[c] for c in columns if c in POS_READ_DTYPES}
    file_size = os.path.getsize(csv_path)

    def batches() -> Iterator[pd.DataFrame]:
        with open(csv_path, 'rb') as f:
            reader = pd.read_csv(f, usecols=columns, dtype=dtype, keep_default_na=False,
                                  na_values=[''], chunksize=STREAM_BATCH_SIZE, low_memory=False)
            with tqdm(total=file_size, desc=os.path.basename(csv_path),
                      unit='B', unit_scale=True, unit_divisor=1024) as pbar:
                last_pos = 0
                for chunk in reader:
                    yield chunk
                    current_pos = f.tell()
                    if current_pos > last_pos:
                        pbar.update(current_pos - last_pos)
                        last_pos = current_pos

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

# (d) ELO-BIN BALANCING (val/test only)

def balance_positions_by_lowest_bin(game_level_df: pd.DataFrame, lower: int = ELO_LOWER,
                                     upper: int = ELO_UPPER, step: int = ELO_STEP, tails: bool = TAILS,
                                     game_id_col: str = 'game_id') -> pd.DataFrame:
    """Restricts every elo bin to the sparsest bin's position count, keeping that bin in full."""
    if 'n_positions' not in game_level_df.columns:
        raise ValueError("game_level_df must have an 'n_positions' column")
    game_df = game_level_df.copy()

    if len(game_df) == 0:
        return game_df

    game_df = game_df.sort_values(game_id_col).reset_index(drop=True)
    edges = list(range(lower, upper + 1, step))
    bin_edges = [-np.inf] + edges + [np.inf] if tails else edges
    mean_elo = (game_df['mover_elo'] + game_df['opponent_elo']) / 2
    mean_bin = pd.cut(mean_elo, bins=bin_edges, labels=False, right=False)
    game_df = game_df[mean_bin.notna()].copy()
    game_df['mean_bin'] = mean_bin[mean_bin.notna()].astype(int)

    bin_position_counts = game_df.groupby('mean_bin')['n_positions'].sum()
    sparsest_bin = bin_position_counts.idxmin()
    target_per_bin = int(bin_position_counts.min())

    pooled_var = game_df['n_positions'].var(ddof=1)
    if pd.isna(pooled_var):
        pooled_var = 0.0
    bin_stats = game_df.groupby('mean_bin')['n_positions'].agg(mean='mean', var='var')
    bin_stats['var'] = bin_stats['var'].fillna(pooled_var)

    caps = game_df['mean_bin'].value_counts().to_dict()
    rng = np.random.default_rng(RANDOM_STATE)

    kept_game_ids = []
    for b in sorted(bin_stats.index):
        bin_games = game_df.loc[game_df['mean_bin'] == b, game_id_col]
        if b == sparsest_bin:
            kept_game_ids.extend(bin_games.tolist())
            continue

        pos_lookup = game_df.set_index(game_id_col).loc[bin_games, 'n_positions']
        n_estimate = _games_needed_for_target(bin_stats.loc[b, 'mean'], bin_stats.loc[b, 'var'],
                                               target_per_bin, CONFIDENCE_Z)
        candidate_n = min(n_estimate, caps[b])

        candidate_ids = rng.choice(bin_games.to_numpy(), size=candidate_n, replace=False)
        selected, total = _cumulative_take_head(candidate_ids, pos_lookup, target_per_bin, rng)
        if total < target_per_bin:
            remaining = bin_games[~bin_games.isin(selected)]
            topped_up, total = _cumulative_take_head(remaining.to_numpy(), pos_lookup,
                                                       target_per_bin - total, rng)
            selected = selected + topped_up
        kept_game_ids.extend(selected)

    return game_level_df[game_level_df[game_id_col].isin(kept_game_ids)].reset_index(drop=True)

def _games_needed_for_target(mean: float, var: float, target: float, confidence_z: float) -> int:
    a, b, c = mean, -confidence_z * np.sqrt(max(var, 0.0)), -target
    x = (-b + np.sqrt(b ** 2 - 4 * a * c)) / (2 * a)
    return int(np.ceil(x ** 2))

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

def _allocate_with_caps(total: int, props: dict, caps: dict) -> tuple:
    remaining_total = total
    remaining_props = dict(props)
    allocation = {k: 0.0 for k in props}
    active = set(props.keys())
    capped_bins = set()

    while active and remaining_total > 0:
        prop_sum = sum(remaining_props[k] for k in active)
        if prop_sum <= 0:
            break
        shares = {k: remaining_total * (remaining_props[k] / prop_sum) for k in active}
        newly_capped = [k for k in active if allocation[k] + shares[k] > caps[k]]
        if not newly_capped:
            for k in active:
                allocation[k] += shares[k]
            remaining_total = 0
        else:
            for k in newly_capped:
                allocation[k] = caps[k]
                active.discard(k)
                capped_bins.add(k)
            remaining_total = total - sum(allocation.values())

    return {k: int(round(v)) for k, v in allocation.items()}, capped_bins

# (e) POSITION POOL SAMPLING

def sample_position_pool(game_level_df: pd.DataFrame, target_positions: int,
                          random_state: int = RANDOM_STATE) -> pd.DataFrame:
    """Randomly shuffles game_level_df and returns a prefix whose cumulative n_positions reaches target_positions."""
    shuffled = game_level_df.sample(frac=1, random_state=random_state).reset_index(drop=True)
    cum = shuffled['n_positions'].cumsum()
    cutoff = int(cum.searchsorted(target_positions)) + 1
    cutoff = min(cutoff, len(shuffled))
    return shuffled.iloc[:cutoff].reset_index(drop=True)

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

# (f) GAME ORDERING FOR NESTED SIZES

def _proportional_interleave(bin_game_lists: dict, allocation: dict) -> list:
    """Merges each bin's own randomly-ordered game list into one order where any prefix stays
    close to each bin's target proportion (allocation[bin] / total), not just the full list."""
    heap = []
    iters = {}
    for b, games in bin_game_lists.items():
        if not games:
            continue
        iters[b] = iter(games)
        heapq.heappush(heap, (1.0 / allocation[b], b, 0))

    order = []
    while heap:
        _, b, taken = heapq.heappop(heap)
        game = next(iters[b], None)
        if game is None:
            continue
        order.append(game)
        taken += 1
        if taken < allocation[b]:
            heapq.heappush(heap, ((taken + 1) / allocation[b], b, taken))
    return order

def _nested_selections(ordered_games: list, pos_counts: dict, sizes: list) -> dict:
    """For sizes ascending, returns {size: (kept_game_ids, trim_count)} using one shared game
    order, so a smaller size's kept games are always a prefix of a larger size's."""
    results = {}
    running_games = []
    running_total = 0
    idx = 0
    for n in sorted(sizes):
        while running_total < n and idx < len(ordered_games):
            g = ordered_games[idx]
            running_games.append(g)
            running_total += pos_counts[g]
            idx += 1
        trim = max(0, running_total - n)
        results[n] = (list(running_games), trim)
    return results

def _compute_ordered_games(game_level_df: pd.DataFrame, sizes: list, lower: int = ELO_LOWER,
                            upper: int = ELO_UPPER, step: int = ELO_STEP, tails: bool = TAILS,
                            game_id_col: str = 'game_id', confidence_z: float = CONFIDENCE_Z,
                            random_state: int | None = RANDOM_STATE) -> tuple[list, dict]:
    """Selects an elo-bin-proportional, game-level pool sized for max(sizes), then orders it so
    any prefix approximates the same proportions. Returns (ordered_games, pos_counts)."""
    if not sizes:
        raise ValueError('sizes must be non-empty')
    max_n = max(sizes)

    game_df = game_level_df.sort_values(game_id_col).reset_index(drop=True)
    edges = list(range(lower, upper + 1, step))
    bin_edges = [-np.inf] + edges + [np.inf] if tails else edges
    mean_elo = (game_df['mover_elo'] + game_df['opponent_elo']) / 2
    mean_bin = pd.cut(mean_elo, bins=bin_edges, labels=False, right=False)
    game_df = game_df[mean_bin.notna()].copy()
    game_df['mean_bin'] = mean_bin[mean_bin.notna()].astype(int)

    bin_counts = game_df['mean_bin'].value_counts()
    total_games = int(bin_counts.sum())
    props = (bin_counts / total_games).to_dict()
    caps = bin_counts.to_dict()

    pooled_var = game_df['n_positions'].var(ddof=1)
    if pd.isna(pooled_var):
        pooled_var = 0.0
    bin_stats = game_df.groupby('mean_bin')['n_positions'].agg(mean='mean', var='var')
    bin_stats['var'] = bin_stats['var'].fillna(pooled_var)

    weighted_mean = sum(props[b] * bin_stats.loc[b, 'mean'] for b in props)
    weighted_var = sum(props[b] * bin_stats.loc[b, 'var'] for b in props)
    n_total_games = _games_needed_for_target(weighted_mean, weighted_var, max_n, confidence_z)
    allocation, capped_bins = _allocate_with_caps(n_total_games, props, caps)

    rng = np.random.default_rng(random_state)
    bin_game_lists = {}
    for b, n in allocation.items():
        if n <= 0:
            continue
        pool = game_df.loc[game_df['mean_bin'] == b, game_id_col].to_numpy()
        chosen = rng.choice(pool, size=min(n, len(pool)), replace=False)
        rng.shuffle(chosen)
        bin_game_lists[b] = list(chosen)
        allocation[b] = len(chosen)

    ordered_games = _proportional_interleave(bin_game_lists, allocation)
    pos_counts = game_df.set_index(game_id_col)['n_positions'].to_dict()
    return ordered_games, pos_counts

def _build_tier_lookup(ordered_games: list, pos_counts: dict, sizes: list) -> dict:
    """Resolves per-tier cutoffs, boundary games, trim counts, and a game_id -> smallest-tier-
    index map, from one shared nested selection."""
    nested = _nested_selections(ordered_games, pos_counts, sizes)
    sizes_sorted = sorted(sizes)
    tier_cutoffs = [len(nested[n][0]) for n in sizes_sorted]
    boundary_game = [nested[n][0][-1] if nested[n][0] else None for n in sizes_sorted]
    trim_count = [nested[n][1] for n in sizes_sorted]

    game_to_min_tier: dict[str, int] = {}
    tier_ptr = 0
    for i, g in enumerate(ordered_games):
        while tier_ptr < len(tier_cutoffs) and i >= tier_cutoffs[tier_ptr]:
            tier_ptr += 1
        if tier_ptr >= len(tier_cutoffs):
            break
        if g not in game_to_min_tier:
            game_to_min_tier[g] = tier_ptr

    return {
        'sizes_sorted': sizes_sorted, 'tier_cutoffs': tier_cutoffs,
        'boundary_game': boundary_game, 'trim_count': trim_count,
        'game_to_min_tier': game_to_min_tier,
    }

# (g) MATERIALIZATION & WRITING

def materialize_nested_sizes(pool_df: pd.DataFrame, game_level_df: pd.DataFrame, sizes: list[int],
                              random_state: int = RANDOM_STATE, game_id_col: str = 'game_id') -> dict[int, pd.DataFrame]:
    """Slices pool_df into one DataFrame per size in sizes, using game_level_df's elo-bin-
    proportional game ordering so smaller sizes' rows are subsets of larger sizes' rows."""
    ordered_games, pos_counts = _compute_ordered_games(game_level_df, sizes, game_id_col=game_id_col,
                                                         random_state=random_state)
    lookup = _build_tier_lookup(ordered_games, pos_counts, sizes)
    sizes_sorted = lookup['sizes_sorted']
    boundary_game = lookup['boundary_game']
    trim_count = lookup['trim_count']
    game_to_min_tier = lookup['game_to_min_tier']

    tier_series = pool_df[game_id_col].map(game_to_min_tier)

    results = {}
    for t, size in enumerate(sizes_sorted):
        mask = tier_series.notna() & (tier_series <= t)
        subset = pool_df[mask]
        trim = trim_count[t]
        if trim > 0 and boundary_game[t] is not None:
            boundary_idx = subset.index[subset[game_id_col] == boundary_game[t]]
            keep_n = max(len(boundary_idx) - trim, 0)
            rng = np.random.default_rng(random_state + t)
            keep_idx = rng.choice(boundary_idx.to_numpy(), size=min(keep_n, len(boundary_idx)), replace=False)
            drop_idx = boundary_idx.difference(pd.Index(keep_idx))
            subset = subset.drop(index=drop_idx)
        results[size] = subset.reset_index(drop=True)
    return results

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

# (h) PIPELINE

def _write_train_outputs(pool_df: pd.DataFrame, gl_pool: pd.DataFrame,
                          gl_pool_res: pd.DataFrame) -> list[tuple[str, str]]:
    """Writes train's raw and elo-gap-restricted variants at every nested size in TRAIN_SIZES."""
    failures = []
    for game_level_df, variant in [(gl_pool, ''), (gl_pool_res, 'res')]:
        sized = materialize_nested_sizes(pool_df, game_level_df, TRAIN_SIZES, random_state=RANDOM_STATE)
        for size, df in sized.items():
            tag = _size_tag(size)
            fname = f'train_{tag}_{variant}.csv' if variant else f'train_{tag}.csv'
            try:
                write_position_csv_with_ids(df, TRAIN_OUT, fname)
            except Exception as e:  # noqa: BLE001
                failures.append((fname, str(e)))
            gc.collect()
    return failures

def _write_val_test_outputs(split: str, pool_df: pd.DataFrame, gl_pool: pd.DataFrame,
                             gl_pool_res: pd.DataFrame) -> list[tuple[str, str]]:
    """Writes val or test's raw/restricted/balanced/wl variant combinations at VAL_TEST_SIZE."""
    base_dir = VAL_OUT if split == 'val' else TEST_OUT
    wl_dir = VAL_OUT_WL if split == 'val' else TEST_OUT_WL

    gl_pool_wl = gl_pool[gl_pool['mover_result'] != 0.5].reset_index(drop=True)
    gl_pool_res_wl = gl_pool_res[gl_pool_res['mover_result'] != 0.5].reset_index(drop=True)
    gl_pool_bal = balance_positions_by_lowest_bin(gl_pool)
    gl_pool_res_bal = balance_positions_by_lowest_bin(gl_pool_res)
    gl_pool_bal_wl = balance_positions_by_lowest_bin(gl_pool_wl)
    gl_pool_res_bal_wl = balance_positions_by_lowest_bin(gl_pool_res_wl)

    variants = [
        (gl_pool, base_dir, ''),
        (gl_pool_res, base_dir, 'res'),
        (gl_pool_bal, base_dir, 'bal'),
        (gl_pool_res_bal, base_dir, 'res_bal'),
        (gl_pool_wl, wl_dir, 'wl'),
        (gl_pool_res_wl, wl_dir, 'res_wl'),
        (gl_pool_bal_wl, wl_dir, 'bal_wl'),
        (gl_pool_res_bal_wl, wl_dir, 'res_bal_wl'),
    ]

    failures = []
    for game_level_df, out_dir, variant in variants:
        sized = materialize_nested_sizes(pool_df, game_level_df, [VAL_TEST_SIZE], random_state=RANDOM_STATE)
        df = sized[VAL_TEST_SIZE]
        tag = _size_tag(VAL_TEST_SIZE)
        fname = f'{split}_{tag}_{variant}.csv' if variant else f'{split}_{tag}.csv'
        try:
            write_position_csv_with_ids(df, out_dir, fname)
        except Exception as e:  # noqa: BLE001
            failures.append((fname, str(e)))
        gc.collect()
    return failures

def run_pipeline(split: str) -> list[tuple[str, str]]:
    """Builds every output CSV and its ids sidecar for one split: train, val, or test."""
    _check_disk_space(LOCAL_DATA_DIR, MIN_FREE_DISK_GB, 'LOCAL_DATA_DIR')

    split_csv = locate_split_csv(split, POS_DIR)
    parquet_dir = os.path.join(LOCAL_DATA_DIR, 'parquet')
    split_pq = csv_to_parquet(split_csv, os.path.join(parquet_dir, f'{split}.parquet'))

    gl_split = game_level_frame_from_parquet(split_pq)

    if split == 'train':
        gl_train_full = gl_split
    else:
        train_csv = locate_split_csv('train', POS_DIR)
        gl_train_full = game_level_frame_from_csv(train_csv)

    fit_result = fit_elo_gap_thresholds(gl_train_full, target_score=ELO_GAP_TARGET_SCORE,
                                         fit_sample_size=ELO_GAP_FIT_SAMPLE_SIZE, on_missing='keep_all')

    gl_pool = sample_position_pool(gl_split, POOL_POSITIONS[split], random_state=RANDOM_STATE)
    pool_ids = set(gl_pool['game_id'])
    pool_df = load_pool_positions(split_pq, pool_ids)

    res_ids = apply_elo_gap_thresholds(gl_pool, fit_result)
    gl_pool_res = gl_pool[gl_pool['game_id'].isin(res_ids)].reset_index(drop=True)

    if split == 'train':
        return _write_train_outputs(pool_df, gl_pool, gl_pool_res)
    return _write_val_test_outputs(split, pool_df, gl_pool, gl_pool_res)

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parses the required split positional argument."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('split', choices=SPLITS, help='Which split to build: train, val, or test.')
    return parser.parse_args(argv)

def main(argv: list[str] | None = None) -> None:
    """Reads args and runs the pipeline for one split."""
    args = parse_args(argv)
    failures = run_pipeline(args.split)
    if failures:
        print('Failed outputs:')
        for fname, err in failures:
            print(f'  - {fname}: {err}')
        sys.exit(1)

####################
# CLASSES
####################

if __name__ == '__main__':
    main()