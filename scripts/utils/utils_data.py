"""
utils_data.py
Data-process helpers, including reading/outputting csvs

Latest changes: 19/08/26:
- read_csv takes inclusive start_date/end_date, added df-length printer
"""

import os
import time

import numpy as np
import pandas as pd

from scripts.utils.utils_chess import EloBinConfig, ELO_BINS, elo_bin_by_mover, elo_bin_edges, elo_bin_labels

####################
# FUNCTIONS
####################

# (a) DATA READING/SAVING

def read_csv(path: str, max_size_gb: float, nrows: int | None = None, dtype: dict | None = None,
             parse_dates: list[str] | None = ['datetime'], usecols: list[str] | None = None,
             start_date: str | None = None, end_date: str | None = None,
             datetime_col: str = 'datetime', chunksize: int = 500_000) -> pd.DataFrame:
    """Reads a csv with a file-size guard, optionally filtered to an inclusive datetime_col range via chunked reads."""
    if not os.path.exists(path):
        raise FileNotFoundError(f'No file found at {path}')
    size_gb = os.path.getsize(path) / (1024 ** 3)
    if size_gb > max_size_gb:
        raise MemoryError(f'File is {size_gb:.2f} GB, exceeds max_size_gb={max_size_gb}.')

    if start_date is None and end_date is None:
        df = pd.read_csv(path, nrows=nrows, dtype=dtype, parse_dates=parse_dates,
                          usecols=usecols, low_memory=False)
        mem_mb = df.memory_usage(deep=True).sum() / (1024 ** 2)
        print(f'Loaded {len(df):,} rows x {len(df.columns)} cols, ~{mem_mb:.1f} MB in memory.')
        return df

    if parse_dates is None or datetime_col not in parse_dates:
        raise ValueError(f"datetime_col='{datetime_col}' must be in parse_dates to filter by date.")
    if usecols is not None and datetime_col not in usecols:
        raise ValueError(f"datetime_col='{datetime_col}' must be in usecols to filter by date.")

    start_ts = pd.Timestamp(start_date) if start_date is not None else None
    end_ts = pd.Timestamp(end_date) if end_date is not None else None

    chunks = []
    rows_read = 0
    for chunk in pd.read_csv(path, dtype=dtype, parse_dates=parse_dates, usecols=usecols,
                              low_memory=False, chunksize=chunksize):
        rows_read += len(chunk)
        mask = pd.Series(True, index=chunk.index)
        if start_ts is not None:
            mask &= chunk[datetime_col] >= start_ts
        if end_ts is not None:
            mask &= chunk[datetime_col] <= end_ts
        if mask.any():
            chunks.append(chunk[mask])
        if nrows is not None and rows_read >= nrows:
            break

    df = pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame(columns=usecols)
    if nrows is not None:
        df = df.iloc[:nrows]

    mem_mb = df.memory_usage(deep=True).sum() / (1024 ** 2)
    start_str = start_date or '-inf'
    end_str = end_date or '+inf'
    range_str = f'[{start_str}, {end_str}]'
    print(f'Loaded {len(df):,} rows x {len(df.columns)} cols, ~{mem_mb:.1f} MB in memory '
          f'(datetime in {range_str}, scanned {rows_read:,} rows).')
    return df


def save_csv(df: pd.DataFrame, path: str, index: bool = False) -> None:
    """Writes df to a csv at path, creating parent directories if needed."""
    if not path.endswith('.csv'):
        path = path + '.csv'
    dirname = os.path.dirname(path)
    if dirname:
        os.makedirs(dirname, exist_ok=True)

    df.to_csv(path, index=index)
    mem_mb = df.memory_usage(deep=True).sum() / (1024 ** 2)
    print(f'Saved {len(df):,} rows x {len(df.columns)} cols, ~{mem_mb:.1f} MB, to {path}')


def wait_for_file(path: str, expected_size: int | None = None, timeout: int = 300,
                   poll_interval: int = 1, stable_checks: int = 3) -> bool:
    """Blocks until path exists and is fully written, by size match or size stability."""
    start = time.time()
    last_size = -1
    stable_count = 0

    while time.time() - start < timeout:
        if os.path.exists(path):
            current_size = os.path.getsize(path)

            if expected_size:
                if current_size == expected_size:
                    return True
            else:
                if current_size > 0 and current_size == last_size:
                    stable_count += 1
                    if stable_count >= stable_checks:
                        return True
                else:
                    stable_count = 0
                last_size = current_size

        time.sleep(poll_interval)

    raise TimeoutError(f'Gave up waiting for {path} after {timeout}s')


def count_rows(path: str, by_elo_bin: bool = False, mover_elo_col: str = 'mover_elo',
                oppo_elo_col: str = 'opponent_elo', cfg: EloBinConfig = ELO_BINS,
                chunksize: int = 500_000) -> None:
    """Streams path in chunks, printing total row count and optionally counts per mean-elo bin."""
    if not os.path.exists(path):
        raise FileNotFoundError(f'No file found at {path}')

    usecols = [mover_elo_col, oppo_elo_col] if by_elo_bin else None
    edges = elo_bin_edges(cfg)
    labels = elo_bin_labels(edges)
    bin_counts = np.zeros(len(labels), dtype='int64') if by_elo_bin else None

    total_rows = 0
    for chunk in pd.read_csv(path, usecols=usecols, chunksize=chunksize, low_memory=False):
        total_rows += len(chunk)
        if by_elo_bin:
            mean_elo = (chunk[mover_elo_col] + chunk[oppo_elo_col]) / 2
            elo_bin = pd.cut(mean_elo, bins=edges, labels=False, right=False)
            if elo_bin.isna().any():
                raise ValueError(f'{elo_bin.isna().sum()} rows handled incorrectly - null.')
            bin_counts += np.bincount(elo_bin.astype('int64'), minlength=len(labels))

    print(f'{total_rows:,} rows in {path}')
    if by_elo_bin:
        print()
        print('Rows by mean-elo bin:')
        for label, count in zip(labels, bin_counts):
            print(f'  {label}: {int(count):,}')


# (b) DATA SAMPLING/ALTERING 

def subsample_df(df: pd.DataFrame, n: int | None, shuffle: bool = False, seed: int = 0) -> pd.DataFrame:
    """Returns df cut to n rows: first n if shuffle=False, else a uniform sample."""
    if n is None or n >= len(df):
        return df.reset_index(drop=True)
    if shuffle:
        return df.sample(n=n, random_state=seed).reset_index(drop=True)
    return df.head(n).reset_index(drop=True)


def attach_game_level_columns(df_game: pd.DataFrame, df_positions: pd.DataFrame,
                               columns: str | list[str], game_id_col: str = 'game_id') -> pd.DataFrame:
    """Joins game-level columns onto position-level rows via game_id_col."""
    columns = [columns] if isinstance(columns, str) else list(columns)
    lookup = df_game.set_index(game_id_col)[columns]
    return df_positions.join(lookup, on=game_id_col)


def sample_one_position_per_game(df_positions: pd.DataFrame, game_id_col: str = 'game_id',
                                  random_state: int | None = None) -> pd.DataFrame:
    """Randomly samples one row per game_id."""
    shuffled = df_positions.sample(frac=1, random_state=random_state)
    return shuffled.drop_duplicates(subset=game_id_col, keep='first').reset_index(drop=True)


def balance_by_lowest(df: pd.DataFrame, cfg: EloBinConfig = ELO_BINS,
                       random_state: int | None = None) -> pd.DataFrame:
    """Downsamples df so each mean-elo bin has the same row count as the smallest bin."""
    df, _ = elo_bin_by_mover(df, cfg, method='mean')
    min_n = df['elo_bin'].value_counts().min()

    balanced = (
        df.groupby('elo_bin', sort=False, group_keys=False)
        .sample(n=min_n, random_state=random_state)
    )

    return balanced.drop(columns='elo_bin').reset_index(drop=True)