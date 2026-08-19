"""
run_pos_storage.py
Builds train/val/test model-ready position CSVs from run_pos_reader.py's output, sampling
each to a set of target sizes with elo-bin-proportional, game-level selection.

    python run_pos_storage.py                 normal run
    python run_pos_storage.py --limit 200000   dry run on a row slice
    python run_pos_storage.py --force          rebuild everything

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
import heapq
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

IDS_PATH = os.path.join(POS_DIR, 'ids')
TRAIN_OUT = os.path.join(POS_DIR, 'train')
VAL_OUT, VAL_OUT_WL = os.path.join(POS_DIR, 'val'), os.path.join(POS_DIR, 'val', 'wl')
TEST_OUT, TEST_OUT_WL = os.path.join(POS_DIR, 'test'), os.path.join(POS_DIR, 'test', 'wl')

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
    n_before = len(df)
    game_df = df.drop_duplicates(subset=game_id_col).sort_values(game_id_col).reset_index(drop=True)
    if len(game_df) != n_before:
        print(f'  [_dedup_to_game_level] collapsed {n_before:,} rows -> {len(game_df):,} games')
    return game_df

def _mean_bin_and_gap(df: pd.DataFrame, lower: int, upper: int, step: int, tails: bool) -> tuple:
    edges = list(range(lower, upper + 1, step))
    bin_edges = [-np.inf] + edges + [np.inf] if tails else edges

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

# (c) ELO-BIN BALANCING (val/test only)

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

# (d) NESTED, GAME-LEVEL POSITION SAMPLING

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

def sample_nested_targets(game_level_df: pd.DataFrame, row_fetcher, sizes: list, lower: int = ELO_LOWER,
                           upper: int = ELO_UPPER, step: int = ELO_STEP, tails: bool = TAILS,
                           game_id_col: str = 'game_id', confidence_z: float = CONFIDENCE_Z,
                           random_state: int | None = RANDOM_STATE, verbose: bool = True) -> dict:
    """Selects nested position samples for every size in sizes from one elo-bin-proportional,
    game-level-trimmed pool sized for max(sizes). Returns {size: dataframe}; each is a strict
    row-level subset of every larger size's dataframe. Games are kept whole wherever possible --
    only the one boundary game per size gets a (small) random position-level trim."""
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
    nested = _nested_selections(ordered_games, pos_counts, sizes)

    results = {}
    for n in sizes:
        kept, trim = nested[n]
        rows = row_fetcher(kept)
        if trim > 0 and kept:
            boundary_game = kept[-1]
            boundary_rows = rows[rows[game_id_col] == boundary_game]
            drop_idx = boundary_rows.sample(n=min(trim, len(boundary_rows)), random_state=random_state).index
            rows = rows.drop(index=drop_idx)
        rows = rows.sample(frac=1, random_state=random_state).reset_index(drop=True)
        results[n] = rows
        if verbose:
            print(f'  N={n:,}: {len(kept):,} games -> {len(rows):,} positions')

    return results

# (e) LOGGING / PIPELINE HELPERS

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

# (f) PIPELINE

def _size_tag(n: int) -> str:
    if n >= 1_000_000:
        value, unit = n / 1_000_000, 'm'
    else:
        value, unit = n / 1_000, 'k'
    if value == int(value):
        return f'{int(value)}{unit}'
    return f'{value:g}'.replace('.', 'p') + unit

def run_pipeline(limit_rows: int | None, force: bool) -> dict:
    _check_disk_space(LOCAL_DATA_DIR, MIN_FREE_DISK_GB, 'LOCAL_DATA_DIR')

    train_csv = locate_split_csv('train', POS_DIR)
    val_csv = locate_split_csv('val', POS_DIR)
    test_csv = locate_split_csv('test', POS_DIR)

    con = get_connection(tmp_dir=os.path.join(LOCAL_DATA_DIR, 'duckdb_tmp'),
                          memory_limit_gb=DUCKDB_MEMORY_LIMIT_GB)
    if limit_rows:
        log.warning(f'--limit {limit_rows} set: this is a DRY RUN on a row slice, not a real build')

    parquet_dir = os.path.join(LOCAL_DATA_DIR, 'parquet')

    with Stage('csv_to_parquet: train'):
        train_pq = csv_to_parquet(con, train_csv, os.path.join(parquet_dir, 'train.parquet'),
                                   force=force, limit_rows=limit_rows)
    with Stage('csv_to_parquet: val'):
        val_pq = csv_to_parquet(con, val_csv, os.path.join(parquet_dir, 'val.parquet'),
                                 force=force, limit_rows=limit_rows)
    with Stage('csv_to_parquet: test'):
        test_pq = csv_to_parquet(con, test_csv, os.path.join(parquet_dir, 'test.parquet'),
                                  force=force, limit_rows=limit_rows)

    train_fetch = functools.partial(rows_for_game_ids, con, [train_pq])
    val_fetch = functools.partial(rows_for_game_ids, con, [val_pq])
    test_fetch = functools.partial(rows_for_game_ids, con, [test_pq])

    with Stage('game_level_frame: train'):
        gl_train = _dedup_to_game_level(game_level_frame(con, [train_pq]))
        log.info(f'  {len(gl_train):,} unique games')
    with Stage('game_level_frame: val'):
        gl_val = _dedup_to_game_level(game_level_frame(con, [val_pq]))
        log.info(f'  {len(gl_val):,} unique games')
    with Stage('game_level_frame: test'):
        gl_test = _dedup_to_game_level(game_level_frame(con, [test_pq]))
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

    def sample_and_write(game_level_df: pd.DataFrame, row_fetcher, sizes: list, out_dir: str,
                          name_prefix: str, variant: str) -> None:
        tag = f'{name_prefix}_{variant}' if variant else name_prefix
        try:
            with Stage(f'sample_nested_targets: {tag}'):
                sampled = sample_nested_targets(game_level_df, row_fetcher, sizes)
                for n, df in sampled.items():
                    size_tag = _size_tag(n)
                    fname = f'{name_prefix}_{size_tag}_{variant}.csv' if variant else f'{name_prefix}_{size_tag}.csv'
                    out_path = os.path.join(out_dir, fname)
                    wrote = write_csv_safely(df, out_path, force)
                    label = f'{tag}_{size_tag}'
                    (results['written'] if wrote else results['skipped']).append(label)
                    del df
                gc.collect()
        except Exception as e:  # noqa: BLE001
            results['failed'].append((tag, str(e)))
            log.error(f'  giving up on {tag}, moving to the next variant: {e}')

    sample_and_write(gl_train, train_fetch, TRAIN_SIZES, TRAIN_OUT, 'train', '')
    sample_and_write(gl_train_res, train_fetch, TRAIN_SIZES, TRAIN_OUT, 'train', 'res')

    sample_and_write(gl_val, val_fetch, VAL_SIZES, VAL_OUT, 'val', '')
    sample_and_write(gl_val_res, val_fetch, VAL_SIZES, VAL_OUT, 'val', 'res')
    sample_and_write(gl_val_bal, val_fetch, VAL_SIZES, VAL_OUT, 'val', 'bal')
    sample_and_write(gl_val_res_bal, val_fetch, VAL_SIZES, VAL_OUT, 'val', 'res_bal')
    sample_and_write(gl_val_wl, val_fetch, VAL_SIZES, VAL_OUT_WL, 'val', 'wl')
    sample_and_write(gl_val_res_wl, val_fetch, VAL_SIZES, VAL_OUT_WL, 'val', 'res_wl')
    sample_and_write(gl_val_bal_wl, val_fetch, VAL_SIZES, VAL_OUT_WL, 'val', 'bal_wl')
    sample_and_write(gl_val_res_bal_wl, val_fetch, VAL_SIZES, VAL_OUT_WL, 'val', 'res_bal_wl')

    sample_and_write(gl_test, test_fetch, TEST_SIZES, TEST_OUT, 'test', '')
    sample_and_write(gl_test_res, test_fetch, TEST_SIZES, TEST_OUT, 'test', 'res')
    sample_and_write(gl_test_bal, test_fetch, TEST_SIZES, TEST_OUT, 'test', 'bal')
    sample_and_write(gl_test_res_bal, test_fetch, TEST_SIZES, TEST_OUT, 'test', 'res_bal')
    sample_and_write(gl_test_wl, test_fetch, TEST_SIZES, TEST_OUT_WL, 'test', 'wl')
    sample_and_write(gl_test_res_wl, test_fetch, TEST_SIZES, TEST_OUT_WL, 'test', 'res_wl')
    sample_and_write(gl_test_bal_wl, test_fetch, TEST_SIZES, TEST_OUT_WL, 'test', 'bal_wl')
    sample_and_write(gl_test_res_bal_wl, test_fetch, TEST_SIZES, TEST_OUT_WL, 'test', 'res_bal_wl')

    con.close()
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
    log.info(f'DuckDB memory limit           = {DUCKDB_MEMORY_LIMIT_GB:.1f}GB '
             f'(of {_TOTAL_RAM_GB:.1f}GB detected total RAM)')
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