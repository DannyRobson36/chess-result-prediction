"""
run_pos_storage.py
Builds train/val/test model-ready position CSVs from run_pos_reader.py's output, sampling
each to a set of target sizes with elo-bin-proportional, game-level (not position-level)
selection so smaller datasets are strict subsets of larger ones and long games keep their
natural over-representation.

    python run_pos_storage.py                 normal run
    python run_pos_storage.py --limit 200000   dry run on a row slice
    python run_pos_storage.py --force          rebuild everything

Latest changes: 19/08/26:
- Removed DuckDB 
"""

import os
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from scripts.config import POS_DIR

import argparse
import csv
import gc
import glob
import heapq
import logging
import shutil
import time
import traceback
from datetime import datetime

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

####################
# CONSTANTS
####################

SPLITS = ['train', 'val', 'test']

LOCAL_DATA_DIR = os.environ.get('LOCAL_DATA_DIR', '/content/local_data')

TRAIN_SIZES = [100_000, 500_000, 2_500_000, 12_500_000, 62_500_000]
VAL_SIZES = [100_000, 900_000]
TEST_SIZES = [900_000]

ELO_LOWER = 800
ELO_UPPER = 2200
ELO_STEP = 200
TAILS = True
GAP_BIN_WIDTH = 10
ELO_GAP_TARGET_SCORE = 0.55

CONFIDENCE_Z = 3.0
RANDOM_STATE = 0

# Cap on games used to fit elo-gap thresholds; train is too large to fit on in full.
ELO_GAP_FIT_SAMPLE_SIZE = 3_000_000

# Row count per streamed read/write batch, for both CSV and Parquet passes.
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

# Columns needed for game-level allocation/threshold decisions; everything else is only ever
# read again from the position-level Parquet at final-write time.
GAME_LEVEL_COLS = ['game_id', 'mover_elo', 'opponent_elo', 'mover_result']

IDS_PATH = os.path.join(POS_DIR, 'ids')
TRAIN_OUT = os.path.join(POS_DIR, 'train')
VAL_OUT, VAL_OUT_WL = os.path.join(POS_DIR, 'val'), os.path.join(POS_DIR, 'val', 'wl')
TEST_OUT, TEST_OUT_WL = os.path.join(POS_DIR, 'test'), os.path.join(POS_DIR, 'test', 'wl')

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

def _read_csv_chunks(csv_path: str, limit_rows: int | None):
    """Yields correctly-typed, title-relabeled chunks of csv_path, streamed."""
    rows_read = 0
    reader = pd.read_csv(
        csv_path, dtype=POS_READ_DTYPES, parse_dates=POS_PARSE_DATES,
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
        rows_read += len(chunk)
        if limit_rows is not None and rows_read >= limit_rows:
            return

def csv_to_parquet(csv_path: str, parquet_path: str, force: bool = False,
                    limit_rows: int | None = None) -> str:
    """Converts a CSV to Parquet on local disk once, streaming in bounded-size chunks."""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f'csv_to_parquet: source CSV not found: {csv_path}')
    if not force and os.path.exists(parquet_path):
        return parquet_path

    os.makedirs(os.path.dirname(parquet_path), exist_ok=True)
    tmp_path = parquet_path + '.tmp'

    writer = None
    try:
        for chunk in _read_csv_chunks(csv_path, limit_rows):
            table = pa.Table.from_pandas(chunk, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(tmp_path, table.schema, compression='zstd')
            writer.write_table(table)
    finally:
        if writer is not None:
            writer.close()
    os.replace(tmp_path, parquet_path)
    return parquet_path

# (b) GAME-LEVEL AGGREGATION

def game_level_frame(parquet_path: str, game_id_col: str = 'game_id') -> pd.DataFrame:
    """Streams parquet_path in batches, aggregating to one row per game_id: elo/result (from
    the first row seen) plus n_positions (a running count), never materializing the full
    position-level table."""
    pf = pq.ParquetFile(parquet_path)
    agg: dict[str, dict] = {}

    for batch in pf.iter_batches(columns=GAME_LEVEL_COLS, batch_size=STREAM_BATCH_SIZE):
        chunk = batch.to_pandas()
        grouped = chunk.groupby(game_id_col, sort=False)
        first = grouped.first()
        counts = grouped.size()
        for gid, row in first.iterrows():
            entry = agg.get(gid)
            if entry is None:
                agg[gid] = {'mover_elo': row['mover_elo'], 'opponent_elo': row['opponent_elo'],
                            'mover_result': row['mover_result'], 'n_positions': int(counts[gid])}
            else:
                entry['n_positions'] += int(counts[gid])

    game_df = pd.DataFrame.from_dict(agg, orient='index')
    game_df.index.name = game_id_col
    game_df = game_df.reset_index()
    return game_df

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
    n_before = len(df)
    game_df = df.drop_duplicates(subset=game_id_col).sort_values(game_id_col).reset_index(drop=True)
    print(f'  [_dedup_to_game_level] collapsed {n_before:,} rows -> {len(game_df):,} games')
    return game_df

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

    n_before = len(df)
    valid = mean_bin.notna()
    df = df[valid].copy()
    df['mean_bin'] = mean_bin[valid].astype(int)
    n_dropped = n_before - len(df)
    if n_dropped > 0:
        print(f'  [_mean_bin_gap_and_res_better] dropped {n_dropped:,} games outside elo range '
              f'[{lower}, {upper}) (tails={tails})')

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
    """Fits a per-elo-bin gap threshold on a random sample of up to fit_sample_size games from
    df_train; apply_elo_gap_thresholds reuses this on every other split without refitting."""
    if on_missing not in ('keep_all', 'drop_all'):
        raise ValueError(f"on_missing must be 'keep_all' or 'drop_all', got {on_missing!r}")

    game_df = _dedup_to_game_level(df_train, game_id_col)
    if len(game_df) > fit_sample_size:
        game_df = game_df.sample(n=fit_sample_size, random_state=random_state).reset_index(drop=True)
        print(f'  [fit_elo_gap_thresholds] subsampled to {fit_sample_size:,} games for fitting')

    binned, bin_edges = _mean_bin_gap_and_res_better(game_df, lower_elo, upper_elo, bin_size, tails)
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
    binned, bin_edges = _mean_bin_gap_and_res_better(game_df, cfg['lower_elo'], cfg['upper_elo'],
                                                       cfg['bin_size'], cfg['tails'])
    assert bin_edges == fit_result['bin_edges'], \
        'bin_edges mismatch between fit and apply -- should be impossible; investigate.'

    row_thresholds = binned['mean_bin'].map(thresholds)
    keep_mask = binned['elo_gap'] <= row_thresholds
    kept_game_ids = binned.loc[keep_mask, game_id_col].tolist()

    if verbose:
        n_games_in = len(game_df)
        n_games_over_gap = int((~keep_mask).sum())
        n_games_kept = len(kept_game_ids)
        n_rows_in = len(df)
        n_rows_kept = int(df[game_id_col].isin(kept_game_ids).sum())
        print(f'apply_elo_gap_thresholds (game level): {n_games_in:,} games in -> '
              f'{n_games_over_gap:,} over gap threshold, {n_games_kept:,} games kept')
        print(f'  input rows: {n_rows_in:,} -> {n_rows_kept:,} rows would be kept '
              f'({n_rows_in - n_rows_kept:,} dropped) if filtered by this game_id list')

    return kept_game_ids

# (d) ELO-BIN BALANCING (val/test only)

def balance_positions_by_lowest_bin(game_level_df: pd.DataFrame, lower: int = ELO_LOWER,
                                     upper: int = ELO_UPPER, step: int = ELO_STEP, tails: bool = TAILS,
                                     game_id_col: str = 'game_id', verbose: bool = True) -> pd.DataFrame:
    """Restricts every elo bin to the sparsest bin's position count, keeping that bin in full."""
    if 'n_positions' not in game_level_df.columns:
        raise ValueError("game_level_df must have an 'n_positions' column")
    game_df = game_level_df.copy()

    if len(game_df) == 0:
        if verbose:
            print('NOTE: input has 0 games -- nothing to balance, returning empty result')
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
    notes = []
    for b in sorted(bin_stats.index):
        bin_games = game_df.loc[game_df['mean_bin'] == b, game_id_col]
        if b == sparsest_bin:
            kept_game_ids.extend(bin_games.tolist())
            continue

        pos_lookup = game_df.set_index(game_id_col).loc[bin_games, 'n_positions']
        n_estimate = _games_needed_for_target(bin_stats.loc[b, 'mean'], bin_stats.loc[b, 'var'],
                                               target_per_bin, CONFIDENCE_Z)
        candidate_n = min(n_estimate, caps[b])
        if n_estimate > caps[b]:
            notes.append(f'bin {b}: estimated {n_estimate} games needed but only {caps[b]} '
                         f'available -- using all of them as the initial pool')

        candidate_ids = rng.choice(bin_games.to_numpy(), size=candidate_n, replace=False)
        selected, total = _cumulative_take_head(candidate_ids, pos_lookup, target_per_bin, rng)
        if total < target_per_bin:
            remaining = bin_games[~bin_games.isin(selected)]
            topped_up, total = _cumulative_take_head(remaining.to_numpy(), pos_lookup,
                                                       target_per_bin - total, rng)
            selected = selected + topped_up
            notes.append(f'bin {b}: candidate pool fell short, topped up with '
                         f'{len(topped_up)} more games to reach target')
        kept_game_ids.extend(selected)

    if verbose and notes:
        print('NOTE: some bins needed extra handling to hit the shared target:')
        for n in notes:
            print(f'  {n}')

    final_df = game_level_df[game_level_df[game_id_col].isin(kept_game_ids)].reset_index(drop=True)
    if verbose:
        print(f'Target per bin: {target_per_bin:,} positions (sparsest bin, kept in full) '
              f'x {len(bin_stats)} bins')
        print(f'Kept {len(kept_game_ids):,} games (~{target_per_bin * len(bin_stats):,} positions '
              f'across them, not yet fetched)')
    return final_df

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

# (e) GAME ORDERING FOR NESTED SIZES

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
                            random_state: int | None = RANDOM_STATE, verbose: bool = True) -> tuple[list, dict]:
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

    if verbose and capped_bins:
        labels = elo_bin_labels(bin_edges)
        capped_str = ', '.join(f'{labels[b]} ({caps[b]} avail)' for b in sorted(capped_bins))
        print(f'NOTE: {len(capped_bins)} bin(s) hit their game cap: {capped_str}')

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

# (f) STREAMED, MULTI-SIZE POSITION WRITING

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

def stream_write_nested_sizes(source_parquet_path: str, game_level_df: pd.DataFrame, sizes: list,
                               out_dir: str, name_prefix: str, variant: str, force: bool,
                               random_state: int = RANDOM_STATE) -> dict:
    """Streams source_parquet_path once, writing every size in sizes simultaneously: a game
    selected for a smaller size is also written to every larger size's file (nested subsets),
    with a per-tier random position trim on each tier's single boundary game to hit exact
    target sizes. Returns {size: 'written' | 'skipped'}."""
    size_tags = {n: _size_tag(n) for n in sizes}
    out_paths = {}
    for n in sizes:
        fname = f'{name_prefix}_{size_tags[n]}_{variant}.csv' if variant else f'{name_prefix}_{size_tags[n]}.csv'
        out_paths[n] = os.path.join(out_dir, fname)

    pending_sizes = [n for n in sizes if force or not os.path.exists(out_paths[n])]
    status = {n: ('written' if n in pending_sizes else 'skipped') for n in sizes}
    if not pending_sizes:
        for n in sizes:
            print(f'  skip (exists): {out_paths[n]}')
        return status

    ordered_games, pos_counts = _compute_ordered_games(game_level_df, pending_sizes, random_state=random_state)
    lookup = _build_tier_lookup(ordered_games, pos_counts, pending_sizes)
    sizes_sorted = lookup['sizes_sorted']
    boundary_game = lookup['boundary_game']
    trim_count = lookup['trim_count']
    game_to_min_tier = lookup['game_to_min_tier']

    os.makedirs(out_dir, exist_ok=True)
    tmp_paths = {n: out_paths[n] + '.tmp' for n in pending_sizes}
    files = {n: open(tmp_paths[n], 'w', newline='', encoding='utf-8') for n in pending_sizes}
    writers = {n: csv.DictWriter(files[n], fieldnames=POS_CSV_FIELDNAMES) for n in pending_sizes}
    for w in writers.values():
        w.writeheader()

    n_written = {n: 0 for n in pending_sizes}
    trim_rng_seed = {n: random_state + i for i, n in enumerate(sizes_sorted)}

    def flush_game(gid: str, rows_for_game: list[dict]) -> None:
        if gid not in game_to_min_tier:
            return
        min_tier = game_to_min_tier[gid]
        for t in range(min_tier, len(sizes_sorted)):
            size = sizes_sorted[t]
            if size not in writers:
                continue
            rows_to_write = rows_for_game
            if gid == boundary_game[t] and trim_count[t] > 0:
                keep_n = max(len(rows_for_game) - trim_count[t], 0)
                rng = np.random.default_rng(trim_rng_seed[size])
                idx = rng.choice(len(rows_for_game), size=min(keep_n, len(rows_for_game)), replace=False)
                rows_to_write = [rows_for_game[i] for i in idx]
            for row in rows_to_write:
                writers[size].writerow(row)
            n_written[size] += len(rows_to_write)

    pf = pq.ParquetFile(source_parquet_path)
    carry_game_id = None
    carry_rows: list[dict] = []
    try:
        for batch in pf.iter_batches(columns=POS_CSV_FIELDNAMES, batch_size=STREAM_BATCH_SIZE):
            chunk = batch.to_pandas()
            for row in chunk.to_dict('records'):
                gid = row['game_id']
                if gid != carry_game_id:
                    if carry_game_id is not None:
                        flush_game(carry_game_id, carry_rows)
                    carry_game_id = gid
                    carry_rows = []
                carry_rows.append(row)
        if carry_game_id is not None:
            flush_game(carry_game_id, carry_rows)
    finally:
        for f in files.values():
            f.close()

    for n in pending_sizes:
        os.replace(tmp_paths[n], out_paths[n])
        print(f'  N={n:,}: wrote {n_written[n]:,} positions -> {out_paths[n]}')

    return status

def _size_tag(n: int) -> str:
    if n >= 1_000_000:
        value, unit = n / 1_000_000, 'm'
    else:
        value, unit = n / 1_000, 'k'
    if value == int(value):
        return f'{int(value)}{unit}'
    return f'{value:g}'.replace('.', 'p') + unit

# (g) LOGGING / PIPELINE HELPERS

def setup_logging() -> logging.Logger:
    log_dir = os.path.join(LOCAL_DATA_DIR, 'logs')
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f'run_pos_storage_{datetime.now():%Y%m%d_%H%M%S}.log')

    logger = logging.getLogger('run_pos_storage')
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

log = logging.getLogger('run_pos_storage')

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

# (h) PIPELINE

def run_pipeline(limit_rows: int | None, force: bool) -> dict:
    _check_disk_space(LOCAL_DATA_DIR, MIN_FREE_DISK_GB, 'LOCAL_DATA_DIR')

    train_csv = locate_split_csv('train', POS_DIR)
    val_csv = locate_split_csv('val', POS_DIR)
    test_csv = locate_split_csv('test', POS_DIR)

    if limit_rows:
        log.warning(f'--limit {limit_rows} set: this is a DRY RUN on a row slice, not a real build')

    parquet_dir = os.path.join(LOCAL_DATA_DIR, 'parquet')

    with Stage('csv_to_parquet: train'):
        train_pq = csv_to_parquet(train_csv, os.path.join(parquet_dir, 'train.parquet'),
                                   force=force, limit_rows=limit_rows)
    with Stage('csv_to_parquet: val'):
        val_pq = csv_to_parquet(val_csv, os.path.join(parquet_dir, 'val.parquet'),
                                 force=force, limit_rows=limit_rows)
    with Stage('csv_to_parquet: test'):
        test_pq = csv_to_parquet(test_csv, os.path.join(parquet_dir, 'test.parquet'),
                                  force=force, limit_rows=limit_rows)

    with Stage('game_level_frame: train'):
        gl_train = game_level_frame(train_pq)
        log.info(f'  {len(gl_train):,} unique games')
    with Stage('game_level_frame: val'):
        gl_val = game_level_frame(val_pq)
        log.info(f'  {len(gl_val):,} unique games')
    with Stage('game_level_frame: test'):
        gl_test = game_level_frame(test_pq)
        log.info(f'  {len(gl_test):,} unique games')

    with Stage('write untouched game_id lists'):
        write_csv_safely(gl_train[['game_id']], os.path.join(IDS_PATH, 'unf_train_ids.csv'), force)
        write_csv_safely(gl_val[['game_id']], os.path.join(IDS_PATH, 'unf_val_ids.csv'), force)
        write_csv_safely(gl_test[['game_id']], os.path.join(IDS_PATH, 'unf_test_ids.csv'), force)

    gl_val_wl = gl_val[gl_val['mover_result'] != 0.5].reset_index(drop=True)
    gl_test_wl = gl_test[gl_test['mover_result'] != 0.5].reset_index(drop=True)

    with Stage('fit_elo_gap_thresholds'):
        fit_result = fit_elo_gap_thresholds(gl_train, target_score=ELO_GAP_TARGET_SCORE, on_missing='keep_all')

    with Stage('apply_elo_gap_thresholds: all splits'):
        train_kept = apply_elo_gap_thresholds(gl_train, fit_result)
        val_kept = apply_elo_gap_thresholds(gl_val, fit_result)
        test_kept = apply_elo_gap_thresholds(gl_test, fit_result)
        val_kept_wl = apply_elo_gap_thresholds(gl_val_wl, fit_result)
        test_kept_wl = apply_elo_gap_thresholds(gl_test_wl, fit_result)

    gl_train_res = gl_train[gl_train['game_id'].isin(train_kept)].reset_index(drop=True)
    gl_val_res = gl_val[gl_val['game_id'].isin(val_kept)].reset_index(drop=True)
    gl_test_res = gl_test[gl_test['game_id'].isin(test_kept)].reset_index(drop=True)
    gl_val_res_wl = gl_val_wl[gl_val_wl['game_id'].isin(val_kept_wl)].reset_index(drop=True)
    gl_test_res_wl = gl_test_wl[gl_test_wl['game_id'].isin(test_kept_wl)].reset_index(drop=True)

    with Stage('write elo-gap-restricted game_id lists'):
        write_csv_safely(gl_train_res[['game_id']], os.path.join(IDS_PATH, 'res_train_ids.csv'), force)
        write_csv_safely(gl_val_res[['game_id']], os.path.join(IDS_PATH, 'res_val_ids.csv'), force)
        write_csv_safely(gl_test_res[['game_id']], os.path.join(IDS_PATH, 'res_test_ids.csv'), force)

    with Stage('balance_positions_by_lowest_bin: val/test'):
        gl_val_bal = balance_positions_by_lowest_bin(gl_val)
        gl_val_bal_wl = balance_positions_by_lowest_bin(gl_val_wl)
        gl_val_res_bal = balance_positions_by_lowest_bin(gl_val_res)
        gl_val_res_bal_wl = balance_positions_by_lowest_bin(gl_val_res_wl)
        gl_test_bal = balance_positions_by_lowest_bin(gl_test)
        gl_test_bal_wl = balance_positions_by_lowest_bin(gl_test_wl)
        gl_test_res_bal = balance_positions_by_lowest_bin(gl_test_res)
        gl_test_res_bal_wl = balance_positions_by_lowest_bin(gl_test_res_wl)

    results = {'written': [], 'skipped': [], 'failed': []}

    def sample_and_write(game_level_df: pd.DataFrame, source_pq: str, sizes: list, out_dir: str,
                          name_prefix: str, variant: str) -> None:
        tag = f'{name_prefix}_{variant}' if variant else name_prefix
        try:
            with Stage(f'stream_write_nested_sizes: {tag}'):
                status = stream_write_nested_sizes(source_pq, game_level_df, sizes, out_dir,
                                                    name_prefix, variant, force)
                for n, s in status.items():
                    label = f'{tag}_{_size_tag(n)}'
                    results[s].append(label)
                gc.collect()
        except Exception as e:  # noqa: BLE001
            results['failed'].append((tag, str(e)))
            log.error(f'  giving up on {tag}, moving to the next variant: {e}')

    sample_and_write(gl_train, train_pq, TRAIN_SIZES, TRAIN_OUT, 'train', '')
    sample_and_write(gl_train_res, train_pq, TRAIN_SIZES, TRAIN_OUT, 'train', 'res')

    sample_and_write(gl_val, val_pq, VAL_SIZES, VAL_OUT, 'val', '')
    sample_and_write(gl_val_res, val_pq, VAL_SIZES, VAL_OUT, 'val', 'res')
    sample_and_write(gl_val_bal, val_pq, VAL_SIZES, VAL_OUT, 'val', 'bal')
    sample_and_write(gl_val_res_bal, val_pq, VAL_SIZES, VAL_OUT, 'val', 'res_bal')
    sample_and_write(gl_val_wl, val_pq, VAL_SIZES, VAL_OUT_WL, 'val', 'wl')
    sample_and_write(gl_val_res_wl, val_pq, VAL_SIZES, VAL_OUT_WL, 'val', 'res_wl')
    sample_and_write(gl_val_bal_wl, val_pq, VAL_SIZES, VAL_OUT_WL, 'val', 'bal_wl')
    sample_and_write(gl_val_res_bal_wl, val_pq, VAL_SIZES, VAL_OUT_WL, 'val', 'res_bal_wl')

    sample_and_write(gl_test, test_pq, TEST_SIZES, TEST_OUT, 'test', '')
    sample_and_write(gl_test_res, test_pq, TEST_SIZES, TEST_OUT, 'test', 'res')
    sample_and_write(gl_test_bal, test_pq, TEST_SIZES, TEST_OUT, 'test', 'bal')
    sample_and_write(gl_test_res_bal, test_pq, TEST_SIZES, TEST_OUT, 'test', 'res_bal')
    sample_and_write(gl_test_wl, test_pq, TEST_SIZES, TEST_OUT_WL, 'test', 'wl')
    sample_and_write(gl_test_res_wl, test_pq, TEST_SIZES, TEST_OUT_WL, 'test', 'res_wl')
    sample_and_write(gl_test_bal_wl, test_pq, TEST_SIZES, TEST_OUT_WL, 'test', 'bal_wl')
    sample_and_write(gl_test_res_bal_wl, test_pq, TEST_SIZES, TEST_OUT_WL, 'test', 'res_bal_wl')

    return results

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parses --force, --limit CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--force', action='store_true',
                         help='Rebuild every Parquet conversion and output CSV, even if it already exists.')
    parser.add_argument('--limit', type=int, default=None,
                         help='Only read this many rows from each source CSV. Default: None (all).')
    return parser.parse_args(argv)

def main(argv: list[str] | None = None) -> None:
    """Reads args and runs the pipeline, logging progress and failures throughout."""
    args = parse_args(argv)

    global log
    log = setup_logging()

    log.info(f'POS_DIR (source/output CSVs) = {POS_DIR}')
    log.info(f'LOCAL_DATA_DIR (parquet/tmp)  = {LOCAL_DATA_DIR}')
    log.info(f'force={args.force}  limit={args.limit}')

    t0 = time.time()
    try:
        results = run_pipeline(limit_rows=args.limit, force=args.force)
    except Exception:
        log.error('Pipeline aborted -- see traceback above for the failing stage.')
        raise

    elapsed = time.time() - t0
    log.info('=' * 60)
    log.info(f"DONE in {elapsed / 60:.1f} min. {len(results['written'])} written, "
             f"{len(results['skipped'])} skipped (already existed), {len(results['failed'])} failed.")
    if results['failed']:
        log.error('Failed outputs (see log above for each traceback):')
        for tag, err in results['failed']:
            log.error(f'  - {tag}: {err}')
        sys.exit(1)

####################
# CLASSES
####################

if __name__ == '__main__':
    main()