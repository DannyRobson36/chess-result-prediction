"""
run_pos_storage.py
Builds train/val/test model-ready position CSVs from run_pos_reader.py's output, sampling
each to a set of target sizes with elo-bin-proportional, game-level (not position-level)
selection so smaller datasets are strict subsets of larger ones and long games keep their
natural over-representation.

    python run_pos_storage.py                 normal run
    python run_pos_storage.py --limit 200000   dry run on a row slice
    python run_pos_storage.py --force          rebuild everything

Latest changes: 20/08/26:
- tqdm progress bars
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
import shutil
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

def _read_csv_chunks(csv_source: IO[bytes], limit_rows: int | None):
    """Yields correctly-typed, title-relabeled chunks of csv_source, streamed."""
    rows_read = 0
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
    file_size = os.path.getsize(csv_path)

    writer = None
    try:
        with open(csv_path, 'rb') as raw_file:
            with tqdm.wrapattr(raw_file, 'read', total=file_size,
                                desc=os.path.basename(csv_path)) as tracked_file:
                for chunk in _read_csv_chunks(tracked_file, limit_rows):
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
    """Fits a per-elo-bin gap threshold on a random sample of up to fit_sample_size games from
    df_train; apply_elo_gap_thresholds reuses this on every other split without refitting."""
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
    desc = f'{name_prefix}_{variant}' if variant else name_prefix
    carry_game_id = None
    carry_rows: list[dict] = []
    try:
        with tqdm(total=pf.metadata.num_rows, desc=desc, unit='rows', unit_scale=True) as pbar:
            for batch in pf.iter_batches(columns=POS_CSV_FIELDNAMES, batch_size=STREAM_BATCH_SIZE):
                chunk = batch.to_pandas()
                pbar.update(len(chunk))
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

    return status

def _size_tag(n: int) -> str:
    if n >= 1_000_000:
        value, unit = n / 1_000_000, 'm'
    else:
        value, unit = n / 1_000, 'k'
    if value == int(value):
        return f'{int(value)}{unit}'
    return f'{value:g}'.replace('.', 'p') + unit

# (g) PIPELINE HELPERS

def write_csv_safely(df: pd.DataFrame, path: str, force: bool) -> bool:
    """Writes df to path, skipping if it exists unless force=True."""
    if os.path.exists(path) and not force:
        return False
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp_path = path + '.tmp'
    df.to_csv(tmp_path, index=False)
    os.replace(tmp_path, path)
    return True

# (h) PIPELINE

def run_pipeline(limit_rows: int | None, force: bool) -> dict:
    _check_disk_space(LOCAL_DATA_DIR, MIN_FREE_DISK_GB, 'LOCAL_DATA_DIR')

    train_csv = locate_split_csv('train', POS_DIR)
    val_csv = locate_split_csv('val', POS_DIR)
    test_csv = locate_split_csv('test', POS_DIR)

    if limit_rows:
        print(f'--limit {limit_rows} set: this is a DRY RUN on a row slice, not a real build')

    parquet_dir = os.path.join(LOCAL_DATA_DIR, 'parquet')

    train_pq = csv_to_parquet(train_csv, os.path.join(parquet_dir, 'train.parquet'),
                               force=force, limit_rows=limit_rows)
    val_pq = csv_to_parquet(val_csv, os.path.join(parquet_dir, 'val.parquet'),
                             force=force, limit_rows=limit_rows)
    test_pq = csv_to_parquet(test_csv, os.path.join(parquet_dir, 'test.parquet'),
                              force=force, limit_rows=limit_rows)

    gl_train = game_level_frame(train_pq)
    gl_val = game_level_frame(val_pq)
    gl_test = game_level_frame(test_pq)

    write_csv_safely(gl_train[['game_id']], os.path.join(IDS_PATH, 'unf_train_ids.csv'), force)
    write_csv_safely(gl_val[['game_id']], os.path.join(IDS_PATH, 'unf_val_ids.csv'), force)
    write_csv_safely(gl_test[['game_id']], os.path.join(IDS_PATH, 'unf_test_ids.csv'), force)

    gl_val_wl = gl_val[gl_val['mover_result'] != 0.5].reset_index(drop=True)
    gl_test_wl = gl_test[gl_test['mover_result'] != 0.5].reset_index(drop=True)

    fit_result = fit_elo_gap_thresholds(gl_train, target_score=ELO_GAP_TARGET_SCORE, on_missing='keep_all')

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

    write_csv_safely(gl_train_res[['game_id']], os.path.join(IDS_PATH, 'res_train_ids.csv'), force)
    write_csv_safely(gl_val_res[['game_id']], os.path.join(IDS_PATH, 'res_val_ids.csv'), force)
    write_csv_safely(gl_test_res[['game_id']], os.path.join(IDS_PATH, 'res_test_ids.csv'), force)

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
            status = stream_write_nested_sizes(source_pq, game_level_df, sizes, out_dir,
                                                name_prefix, variant, force)
            for n, s in status.items():
                results[s].append(f'{tag}_{_size_tag(n)}')
            gc.collect()
        except Exception as e:  # noqa: BLE001
            results['failed'].append((tag, str(e)))
            print(f'FAILED {tag}: {e} -- continuing with next variant')

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
    """Reads args and runs the pipeline."""
    args = parse_args(argv)

    print(f'POS_DIR        = {POS_DIR}')
    print(f'LOCAL_DATA_DIR = {LOCAL_DATA_DIR}')
    print(f'force={args.force}  limit={args.limit}')

    results = run_pipeline(limit_rows=args.limit, force=args.force)

    print(f"\nDone. {len(results['written'])} written, {len(results['skipped'])} skipped, "
          f"{len(results['failed'])} failed.")
    if results['failed']:
        print('Failed outputs:')
        for tag, err in results['failed']:
            print(f'  - {tag}: {err}')
        sys.exit(1)

####################
# CLASSES
####################

if __name__ == '__main__':
    main()