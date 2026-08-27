"""
features.py
Builds model-ready SplitData from raw dataframes: scaling, elo binning.
Also builds SplitData for new (inference) data from an already-fitted PrepConfig.

Latest changes: 27/08/26:
- Corrected to full version
"""

import json
import multiprocessing as mp
import os
import signal
import subprocess
import time
import warnings
from collections.abc import Iterable
from dataclasses import dataclass, field

import chess
import chess.engine
import numpy as np
import pandas as pd
import psutil
import torch
from tqdm import tqdm

from scripts.config import STOCKFISH_PATH
from scripts.utils.utils_chess import (
    EloBinConfig, ELO_BINS, elo_bin_edges, elo_bin_labels,
    fen_to_tensor, fen_to_token_ids, fen_to_legal_dest, fen_to_attacked_squares,
    encode_result_class, encode_result_continuous, encode_title_idx, BOARD_SEQ_LEN,
)

####################
# CONSTANTS
####################

_ELO_TRANS = {'raw': 'raw', 'normal': 'normal'}
_PLY_TRANS = {'raw': 'raw', 'normal': 'normal'}
_PAST_TRANS = {'raw': 'raw', 'normal': 'normal', 'centered': 'centered'}
_LAST_RESULT_TRANS = {'raw': 'raw', 'normal': 'normal'}
_HOURS_TRANS = {
    'raw': 'raw', 'symmetric_prop': 'centered', 'symmetric_prop_log': 'log_centered',
    'hours_since_pow': 'pow_hours',
}
_CLOCK_TRANS = {
    'raw': 'raw',
    'clock_prop_norm': 'normal',
    'clock_prop_centred': 'centered',
    'clock_prop_log_norm': 'log_norm',
    'clock_prop_log_centred': 'log_centered',
    'clock_prop_pow': 'pow_clock',
}
_FLAG_TRANS = {'raw': 'raw'}
_SF_TRANS = {'raw': 'raw', 'normal': 'normal', 'centered': 'centered'}

_TRANS_NAME_TO_BASE: dict[str, str] = {}
for _trans_map in (_ELO_TRANS, _PLY_TRANS, _PAST_TRANS, _LAST_RESULT_TRANS, _HOURS_TRANS,
                   _CLOCK_TRANS, _FLAG_TRANS, _SF_TRANS):
    _TRANS_NAME_TO_BASE.update(_trans_map)

SEC_MAPPING = {'600+0': 600, '600+5': 600, '900+10': 900}
INC_FLAG_MAPPING = {'600+0': 0, '600+5': 1, '900+10': 1}
TOTAL_LENGTH_FLAG_MAPPING = {'600+0': 0, '600+5': 0, '900+10': 1}

CHUNK_SIZE = 1_000_000
BOARD_TENSOR_SHAPE = (18, 8, 8)

# Board/token-tensor writing mode -> (write_boards, write_board_token_ids).
BOARD_MODE_REGISTRY: dict[str, tuple[bool, bool]] = {
    'none': (False, False),
    'boards_only': (True, False),
    'tokens_only': (False, True),
    'both': (True, True),
}

# Centipawn -> win-probability sigmoid constant (the Lichess win% curve).
WIN_PCT_CONST = 0.00368208

# Sentinel mover-perspective eval for a forced mate (positive if mate favors the mover).
MATE_SCORE_MOVER = 5_000

# Per-engine hash size in MB for stockfish evaluation workers.
HASH_MB_PER_ENGINE = 16

# Timeout in seconds for one position's stockfish analysis before a same-depth retry.
ATTEMPT_TIMEOUT_SECONDS = 120

# Starting Elo rating for a new Lichess account.
NEW_USER_STARTING_ELO = 1500

# Exponent for the clip-then-power clock/hours-since transforms.
POW_EXPONENT = 0.3

####################
# FUNCTIONS
####################

# (a) SIMPLE FEATURE TRANSFORMS

def cp_to_mover_winprob(cp: pd.Series | np.ndarray) -> pd.Series | np.ndarray:
    """Cp to mover win probability, lichess logistic curve."""
    return 0.5 + 0.5 * (2 / (1 + np.exp(-WIN_PCT_CONST * cp)) - 1)


# (b) SCALING TRANSFORMS

def _fit_raw(pooled_train: np.ndarray) -> dict:
    """Empty stats for the raw transform."""
    return {}


def _apply_raw(vals: np.ndarray, stats: dict) -> np.ndarray:
    """Casts vals to float32."""
    return vals.astype('float32')


def _fit_normal(pooled_train: np.ndarray) -> dict:
    """Mean and std for z-score scaling."""
    return {'mu': float(pooled_train.mean()), 'sigma': float(pooled_train.std())}


def _apply_normal(vals: np.ndarray, stats: dict) -> np.ndarray:
    """Z-score scales vals using fitted mu/sigma."""
    return ((vals - stats['mu']) / stats['sigma']).astype('float32')


def _fit_log_norm(pooled_train: np.ndarray) -> dict:
    """Mean and std of log1p(vals) for log scaling."""
    log_vals = np.log1p(pooled_train)
    return {'mu': float(log_vals.mean()), 'sigma': float(log_vals.std())}


def _apply_log_norm(vals: np.ndarray, stats: dict) -> np.ndarray:
    """Log1p then z-score scales vals using fitted mu/sigma."""
    return ((np.log1p(vals) - stats['mu']) / stats['sigma']).astype('float32')


def _fit_centered(pooled_train: np.ndarray) -> dict:
    """Min and max for scaling to [-1, 1]."""
    return {'lo': float(pooled_train.min()), 'hi': float(pooled_train.max())}


def _apply_centered(vals: np.ndarray, stats: dict) -> np.ndarray:
    """Rescales vals to [-1, 1] using fitted lo/hi."""
    span = stats['hi'] - stats['lo'] if stats['hi'] > stats['lo'] else 1.0
    return (2 * ((vals - stats['lo']) / span) - 1).astype('float32')


def _fit_log_centered(pooled_train: np.ndarray) -> dict:
    """Min and max of log1p(vals) for scaling to [-1, 1]."""
    log_vals = np.log1p(pooled_train)
    return {'lo': float(log_vals.min()), 'hi': float(log_vals.max())}


def _apply_log_centered(vals: np.ndarray, stats: dict) -> np.ndarray:
    """Log1p then rescales vals to [-1, 1] using fitted lo/hi."""
    span = stats['hi'] - stats['lo'] if stats['hi'] > stats['lo'] else 1.0
    return (2 * ((np.log1p(vals) - stats['lo']) / span) - 1).astype('float32')


def _fit_pow_clock(pooled_train: np.ndarray) -> dict:
    """Empty stats for the clip-then-power clock transform."""
    return {}


def _apply_pow_clock(vals: np.ndarray, stats: dict) -> np.ndarray:
    """Clips vals to [0, 1], then raises to POW_EXPONENT."""
    return (np.clip(vals, 0.0, 1.0) ** POW_EXPONENT).astype('float32')


def _fit_pow_hours(pooled_train: np.ndarray) -> dict:
    """Empty stats for the clip-then-power hours-since transform."""
    return {}


def _apply_pow_hours(vals: np.ndarray, stats: dict) -> np.ndarray:
    """Clips vals to [0, 1], then raises to POW_EXPONENT."""
    return (np.clip(vals, 0.0, 1.0) ** POW_EXPONENT).astype('float32')


_TRANS_FIT_APPLY = {
    'raw': (_fit_raw, _apply_raw),
    'normal': (_fit_normal, _apply_normal),
    'log_norm': (_fit_log_norm, _apply_log_norm),
    'centered': (_fit_centered, _apply_centered),
    'log_centered': (_fit_log_centered, _apply_log_centered),
    'pow_clock': (_fit_pow_clock, _apply_pow_clock),
    'pow_hours': (_fit_pow_hours, _apply_pow_hours),
}


def _resolve_trans_name(scale_dict: dict | None, key: str, trans_map: dict[str, str], group_label: str) -> tuple[str, str]:
    """Looks up and validates the transform name for key."""
    trans_name = (scale_dict or {}).get(key, 'raw')
    if trans_name not in trans_map:
        raise ValueError(f'{group_label}[{key!r}]: unknown trans_name {trans_name!r}, '
                          f'choose from {sorted(trans_map)}.')
    return trans_name, trans_map[trans_name]


def _validate_scale_keys(scale_dict: dict | None, expected_keys: set[str], group_label: str) -> None:
    """Raises on any scale_dict key outside expected_keys."""
    if scale_dict is None:
        return
    unknown = set(scale_dict) - set(expected_keys)
    if unknown:
        raise ValueError(f'{group_label} has unrecognized key(s) {sorted(unknown)}; '
                          f'expected only {sorted(expected_keys)}.')


def _fit_apply_pooled(train_mover_raw: np.ndarray, train_opponent_raw: np.ndarray, per_split_raw: dict,
                       trans_name: str, base_trans: str, group_label: str) -> tuple[dict, dict]:
    """Fits base_trans on pooled train arrays, applies to every split/side array."""
    fit_fn, apply_fn = _TRANS_FIT_APPLY[base_trans]
    pooled = np.concatenate([train_mover_raw, train_opponent_raw])
    stats = fit_fn(pooled)
    scaled = {
        split_name: {side: apply_fn(arr, stats) for side, arr in sides.items()}
        for split_name, sides in per_split_raw.items()
    }
    return scaled, {'trans_name': trans_name, **stats}


def _apply_stored_stats(raw: np.ndarray, stats: dict) -> np.ndarray:
    """Applies a previously-fitted transform's apply function using its stored trans_name/stats."""
    base_trans = _TRANS_NAME_TO_BASE[stats['trans_name']]
    apply_fn = _TRANS_FIT_APPLY[base_trans][1]
    return apply_fn(raw, stats)


# (c) COLUMN GROUP RESOLUTION

def _match_one(cols: list[str], keyword: str, group_label: str) -> str:
    """Returns the one column containing keyword, or raises."""
    matches = [c for c in cols if keyword in c]
    if len(matches) != 1:
        raise ValueError(f'{group_label}: expected exactly one column containing {keyword!r} '
                          f'in {cols}, found {matches}.')
    return matches[0]


def _resolve_elo_cols(elo_cols: list[str] | None) -> tuple[str, str]:
    """Resolves elo_cols to (mover_col, opponent_col)."""
    if elo_cols is None or len(elo_cols) != 2:
        raise ValueError(f'elo_cols must have exactly 2 entries, got {elo_cols}.')
    return _match_one(elo_cols, 'mover', 'elo_cols'), _match_one(elo_cols, 'opponent', 'elo_cols')


def _resolve_clock_cols(clock_cols: list[str]) -> tuple[str, str, str]:
    """Resolves clock_cols to (mover_col, opponent_col, time_control_col)."""
    if len(clock_cols) != 3:
        raise ValueError(f'clock_cols must have exactly 3 entries, got {clock_cols}.')
    mover_col = _match_one(clock_cols, 'mover', 'clock_cols')
    opponent_col = _match_one(clock_cols, 'opponent', 'clock_cols')
    remaining = [c for c in clock_cols if c not in (mover_col, opponent_col)]
    if len(remaining) != 1:
        raise ValueError(f'clock_cols: could not identify a single time_control column in {clock_cols}.')
    return mover_col, opponent_col, remaining[0]


def _clock_props(df: pd.DataFrame, mover_col: str, opponent_col: str, time_control_col: str) -> tuple[np.ndarray, np.ndarray]:
    """Returns (mover, opponent) clock time remaining as a proportion of total game time."""
    game_time = df[time_control_col].map(SEC_MAPPING).to_numpy(dtype='float64')
    mover = df[mover_col].to_numpy(dtype='float64') / game_time
    opponent = df[opponent_col].to_numpy(dtype='float64') / game_time
    return mover, opponent


def _resolve_past_cols(past_cols: list[str]) -> dict[str, str]:
    """Resolves past_cols to a dict of named columns."""
    if len(past_cols) not in (2, 4, 6):
        raise ValueError(f'past_cols must have 2, 4, or 6 entries, got {len(past_cols)}: {past_cols}.')
    past_group = [c for c in past_cols if 'past' in c]
    hours_group = [c for c in past_cols if 'hours' in c]
    result_group = [c for c in past_cols if 'result' in c]
    if len(past_group) != 2:
        raise ValueError(f"past_cols: expected exactly 2 'past' columns, found {past_group} in {past_cols}.")
    out = {
        'past_mover': _match_one(past_group, 'mover', 'past_cols (past)'),
        'past_opponent': _match_one(past_group, 'opponent', 'past_cols (past)'),
    }
    if len(past_cols) >= 4:
        if len(hours_group) != 2:
            raise ValueError(f"past_cols: expected exactly 2 'hours' columns, found {hours_group} in {past_cols}.")
        out['hours_since_mover'] = _match_one(hours_group, 'mover', 'past_cols (hours)')
        out['hours_since_opponent'] = _match_one(hours_group, 'opponent', 'past_cols (hours)')
    if len(past_cols) == 6:
        if len(result_group) != 2:
            raise ValueError(f"past_cols: expected exactly 2 'result' columns, found {result_group} in {past_cols}.")
        out['last_result_mover'] = _match_one(result_group, 'mover', 'past_cols (result)')
        out['last_result_opponent'] = _match_one(result_group, 'opponent', 'past_cols (result)')
    return out


def _has_history_mask(df: pd.DataFrame, hours_since_col: str) -> np.ndarray:
    """Boolean mask of rows with a non-missing hours-since value."""
    return pd.to_numeric(df[hours_since_col], errors='coerce').notna().to_numpy()


def _hours_since_props(df: pd.DataFrame, mover_col: str, opponent_col: str, max_hours: float) -> tuple[np.ndarray, np.ndarray]:
    """Returns (mover, opponent) hours-since-last-game as a proportion of max_hours, missing values filled to max_hours."""
    mover = pd.to_numeric(df[mover_col], errors='coerce').fillna(max_hours).to_numpy(dtype='float64') / max_hours
    opponent = pd.to_numeric(df[opponent_col], errors='coerce').fillna(max_hours).to_numpy(dtype='float64') / max_hours
    return mover, opponent


def _resolve_color_cols(color_cols: list[str]) -> str:
    """Resolves color_cols to its column name."""
    if len(color_cols) != 1:
        raise ValueError(f'color_cols must have exactly 1 entry, got {color_cols}.')
    return color_cols[0]


def _resolve_ply_cols(ply_cols: list[str]) -> str:
    """Resolves ply_cols to its column name."""
    if len(ply_cols) != 1:
        raise ValueError(f'ply_cols must have exactly 1 entry, got {ply_cols}.')
    return ply_cols[0]


def _resolve_rematch_cols(rematch_cols: list[str]) -> tuple[str, str]:
    """Resolves rematch_cols to (rematch_flag_col, prev_result_col)."""
    if len(rematch_cols) != 2:
        raise ValueError(f'rematch_cols must have exactly 2 entries, got {rematch_cols}.')
    return _match_one(rematch_cols, 'rematch', 'rematch_cols'), _match_one(rematch_cols, 'prev', 'rematch_cols')


def _resolve_title_cols(title_cols: list[str]) -> tuple[str, str]:
    """Resolves title_cols to (mover_col, opponent_col)."""
    if len(title_cols) != 2:
        raise ValueError(f'title_cols must have exactly 2 entries, got {title_cols}.')
    return _match_one(title_cols, 'mover', 'title_cols'), _match_one(title_cols, 'opponent', 'title_cols')


def _validate_time_control(values: pd.Series, valid_keys: Iterable[str]) -> None:
    """Checks time_control values are all in valid_keys."""
    unknown = set(values.unique()) - set(valid_keys)
    if unknown:
        raise ValueError(f'time_control contains unrecognized value(s) {sorted(unknown)}; '
                          f'expected only {sorted(valid_keys)}.')


# (d) ELO BINNING FOR SPLITS

def _bin_elo_splits(mover_elo: pd.Series, opponent_elo: pd.Series,
                     cfg: EloBinConfig = ELO_BINS) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, list[str]]:
    """Bins mover, opponent, and mean elo into cfg's edges."""
    edges = elo_bin_edges(cfg)
    n_bins = len(edges) - 1
    labels = elo_bin_labels(edges)

    game_elo = (mover_elo + opponent_elo) / 2.0
    elo_mean_bin = pd.cut(game_elo, bins=edges, labels=False, right=False).to_numpy().astype(np.int64)
    elo_self_bin = pd.cut(mover_elo, bins=edges, labels=False, right=False).to_numpy().astype(np.int64)
    elo_oppo_bin = pd.cut(opponent_elo, bins=edges, labels=False, right=False).to_numpy().astype(np.int64)

    return elo_mean_bin, elo_self_bin, elo_oppo_bin, n_bins, labels


# (e) STOCKFISH EVALUATION

_engine = None
# Per-worker persistent engine handle, set by _stockfish_init_worker inside each pool process.


def _ensure_stockfish_installed(engine_path: str) -> None:
    """Installs stockfish via apt-get if engine_path doesn't already exist, raising if that fails."""
    if os.path.exists(engine_path):
        return
    install = subprocess.run(['apt-get', 'install', '-y', 'stockfish'], capture_output=True, text=True)
    if install.returncode != 0 or not os.path.exists(engine_path):
        raise RuntimeError(f'Stockfish install failed or binary not found at {engine_path}.\n{install.stderr}')


def _stockfish_start_engine() -> chess.engine.SimpleEngine:
    """Starts one stockfish engine process for this worker."""
    engine = chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH)
    engine.configure({'Threads': 1, 'Hash': HASH_MB_PER_ENGINE})
    return engine


def _stockfish_init_worker() -> None:
    """Starts this worker's persistent engine, stored in the module-level _engine global."""
    global _engine
    _engine = _stockfish_start_engine()


def _stockfish_restart_engine() -> None:
    """Closes and restarts this worker's engine after a timeout or error."""
    global _engine
    try:
        _engine.close()
    except Exception:
        try:
            _engine.transport.kill()
        except Exception:
            pass
    _engine = _stockfish_start_engine()


def _stockfish_alarm_handler(signum, frame) -> None:
    """SIGALRM handler that converts a timeout into a _StockfishTimeout."""
    raise _StockfishTimeout()


def _stockfish_analyse_one(args: tuple[str, int]) -> float:
    """Evaluates one fen at depth via this worker's persistent engine, mover-perspective centipawns."""
    fen, depth = args
    board = chess.Board(fen)
    mover_is_white = board.turn

    if board.is_game_over():
        return float(-MATE_SCORE_MOVER) if board.is_checkmate() else 0.0

    for _ in range(2):
        signal.signal(signal.SIGALRM, _stockfish_alarm_handler)
        signal.alarm(ATTEMPT_TIMEOUT_SECONDS)
        try:
            _engine.configure({'Clear Hash': None})
            info = _engine.analyse(board, chess.engine.Limit(depth=depth))
            signal.alarm(0)
            white_score = info['score'].white()
            if white_score.is_mate():
                mate_for_white = white_score.mate() > 0
                mate_for_mover = mate_for_white == mover_is_white
                return float(MATE_SCORE_MOVER if mate_for_mover else -MATE_SCORE_MOVER)
            raw_cp = white_score.score()
            return float(raw_cp if mover_is_white else -raw_cp)
        except Exception:
            signal.alarm(0)
            _stockfish_restart_engine()

    print(f'Stockfish failed twice on one position, falling back to eval=0.0: {fen}')
    return 0.0


def _stockfish_eval_fens(fens: np.ndarray, depth: int) -> np.ndarray:
    """Evaluates every fen at depth via a multiprocessing pool of persistent engines, mover-perspective cp."""
    _ensure_stockfish_installed(STOCKFISH_PATH)

    tasks = [(fen, depth) for fen in fens]
    n_workers = os.cpu_count() or 1

    results = []
    with mp.Pool(processes=n_workers, initializer=_stockfish_init_worker) as pool:
        for cp in tqdm(pool.imap(_stockfish_analyse_one, tasks), total=len(tasks), desc='stockfish eval', unit='pos'):
            results.append(cp)

    return np.array(results, dtype='float64')


# (f) DISK PERSISTENCE

def _rss_gb() -> float:
    """Returns the current process's resident memory usage in GB."""
    return psutil.Process().memory_info().rss / (1024 ** 3)


def _resolve_board_mode(board_mode: str) -> tuple[bool, bool]:
    """Returns (write_boards, write_board_token_ids) for board_mode, from BOARD_MODE_REGISTRY."""
    if board_mode not in BOARD_MODE_REGISTRY:
        raise ValueError(f"Unknown board_mode '{board_mode}', choose from {list(BOARD_MODE_REGISTRY)}")
    return BOARD_MODE_REGISTRY[board_mode]


def _preallocate_npy(path: str, shape: tuple, dtype: type) -> np.memmap:
    """Creates a disk-backed .npy array at path, preallocated to shape/dtype."""
    return np.lib.format.open_memmap(path, mode='w+', dtype=dtype, shape=shape)


def _write_boards_tokens_and_aux(split_dir: str, fens: np.ndarray, desc: str, chunk_size: int,
                                  write_boards: bool, write_tokens: bool, aux_targets: bool) -> None:
    """Encodes fens row by row directly into preallocated boards/board_token_ids/aux-target .npy
    files, flushing silently every chunk_size rows."""
    if not write_boards and not write_tokens and not aux_targets:
        return

    n_rows = len(fens)
    boards_mm = (_preallocate_npy(os.path.join(split_dir, 'boards.npy'), (n_rows, *BOARD_TENSOR_SHAPE), np.bool_)
                 if write_boards else None)
    token_ids_mm = (_preallocate_npy(os.path.join(split_dir, 'board_token_ids.npy'), (n_rows, BOARD_SEQ_LEN), np.uint8)
                    if write_tokens else None)
    legal_dest_mm = (_preallocate_npy(os.path.join(split_dir, 'legal_dest.npy'), (n_rows, 64), np.bool_)
                      if aux_targets else None)
    attacked_mover_mm = (_preallocate_npy(os.path.join(split_dir, 'attacked_mover.npy'), (n_rows, 64), np.bool_)
                          if aux_targets else None)
    attacked_opponent_mm = (_preallocate_npy(os.path.join(split_dir, 'attacked_opponent.npy'), (n_rows, 64), np.bool_)
                             if aux_targets else None)

    for i in tqdm(range(n_rows), desc=desc, unit='rows'):
        if write_boards:
            boards_mm[i] = fen_to_tensor(fens[i]).numpy().astype(np.bool_)
        if write_tokens:
            token_ids_mm[i] = fen_to_token_ids(fens[i]).numpy().astype(np.uint8)
        if aux_targets:
            legal_dest_mm[i] = fen_to_legal_dest(fens[i]).numpy().astype(np.bool_)
            attacked = fen_to_attacked_squares(fens[i]).numpy().astype(np.bool_)
            attacked_mover_mm[i] = attacked[0]
            attacked_opponent_mm[i] = attacked[1]

        if (i + 1) % chunk_size == 0 or i + 1 == n_rows:
            if write_boards:
                boards_mm.flush()
            if write_tokens:
                token_ids_mm.flush()
            if aux_targets:
                legal_dest_mm.flush()
                attacked_mover_mm.flush()
                attacked_opponent_mm.flush()

    if write_boards:
        del boards_mm
    if write_tokens:
        del token_ids_mm
    if aux_targets:
        del legal_dest_mm, attacked_mover_mm, attacked_opponent_mm


def _write_split_arrays(split_dir: str, elo_mean_bin: np.ndarray, elo_self_bin: np.ndarray, elo_oppo_bin: np.ndarray,
                         features: dict, result_class: torch.Tensor, result_cont: torch.Tensor,
                         n_elo_bins: int, bin_labels: list, game_id: list, fen: list,
                         has_boards: bool, has_board_token_ids: bool, has_aux_targets: bool) -> None:
    """Writes elo bins, features, targets, meta.json, and ids.json (game_id/fen) to split_dir."""
    np.save(os.path.join(split_dir, 'elo_mean_bin.npy'), elo_mean_bin.astype(np.uint8))
    np.save(os.path.join(split_dir, 'elo_self_bin.npy'), elo_self_bin.astype(np.uint8))
    np.save(os.path.join(split_dir, 'elo_oppo_bin.npy'), elo_oppo_bin.astype(np.uint8))
    np.save(os.path.join(split_dir, 'result_class.npy'), result_class.numpy())
    np.save(os.path.join(split_dir, 'result_cont.npy'), result_cont.numpy())

    feature_dir = os.path.join(split_dir, 'features')
    os.makedirs(feature_dir, exist_ok=True)
    for name, tensor in features.items():
        np.save(os.path.join(feature_dir, f'{name}.npy'), tensor.numpy())

    with open(os.path.join(split_dir, 'meta.json'), 'w') as f:
        json.dump({'n_elo_bins': n_elo_bins, 'bin_labels': bin_labels,
                   'has_boards': has_boards, 'has_board_token_ids': has_board_token_ids,
                   'has_aux_targets': has_aux_targets}, f)

    with open(os.path.join(split_dir, 'ids.json'), 'w') as f:
        json.dump({'game_id': game_id, 'fen': fen}, f)


def load_split(split_dir: str) -> 'SplitData':
    """Loads SplitData from split_dir, memory-mapped (game_id/fen loaded fully, not memory-mapped)."""
    def _mmap_path(path: str) -> torch.Tensor:
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', message='.*not writable.*', category=UserWarning)
            return torch.from_numpy(np.load(path, mmap_mode='r'))

    def _mmap(name: str) -> torch.Tensor:
        return _mmap_path(os.path.join(split_dir, f'{name}.npy'))

    feature_dir = os.path.join(split_dir, 'features')
    features = {
        fname.removesuffix('.npy'): _mmap_path(os.path.join(feature_dir, fname))
        for fname in sorted(os.listdir(feature_dir))
    }

    with open(os.path.join(split_dir, 'meta.json')) as f:
        meta = json.load(f)

    with open(os.path.join(split_dir, 'ids.json')) as f:
        ids = json.load(f)

    has_aux = meta.get('has_aux_targets', False)
    return SplitData(
        boards=_mmap('boards') if meta.get('has_boards', True) else None,
        board_token_ids=_mmap('board_token_ids') if meta.get('has_board_token_ids', True) else None,
        legal_dest=_mmap('legal_dest') if has_aux else None,
        attacked_mover=_mmap('attacked_mover') if has_aux else None,
        attacked_opponent=_mmap('attacked_opponent') if has_aux else None,
        elo_mean_bin=_mmap('elo_mean_bin'),
        elo_self_bin=_mmap('elo_self_bin'),
        elo_oppo_bin=_mmap('elo_oppo_bin'),
        features=features,
        result_class=_mmap('result_class'),
        result_cont=_mmap('result_cont'),
        n_elo_bins=meta['n_elo_bins'],
        bin_labels=meta['bin_labels'],
        game_id=ids['game_id'],
        fen=ids['fen'],
    )


# (g) DATA PREP PIPELINE

def prepare_splits(df_train: pd.DataFrame, df_val: pd.DataFrame, df_val_wl: pd.DataFrame, *,
                    mover_result_col: str,
                    out_dir: str,
                    fen_col: str = 'fen',
                    game_id_col: str = 'game_id',
                    elo_cols: list[str] | None = None, elo_scale: dict | None = None,
                    clock_cols: list[str] | None = None, clock_scale: dict | None = None,
                    past_cols: list[str] | None = None, past_scale: dict | None = None,
                    color_cols: list[str] | None = None, color_scale: dict | None = None,
                    ply_cols: list[str] | None = None, ply_scale: dict | None = None,
                    rematch_cols: list[str] | None = None, rematch_scale: dict | None = None,
                    title_cols: list[str] | None = None, title_scale: dict | None = None,
                    cfg: EloBinConfig = ELO_BINS, chunk_size: int = CHUNK_SIZE,
                    board_mode: str = 'both', aux_targets: bool = False,
                    sf_depth: int | None = None, sf_scale: dict | None = None,
                    ) -> tuple['SplitData', 'SplitData', 'SplitData', 'PrepConfig']:
    """Builds, saves, and memory-maps train/val/val_wl SplitData; returns a PrepConfig for reapplying
    to new data. df_val drives training/early-stopping and temperature fitting; df_val_wl is a
    separate, decisive-only (no-draw) sample used only for probit calibration fitting."""
    if elo_cols is None or elo_scale is None:
        raise ValueError('elo_cols and elo_scale are mandatory.')

    write_boards, write_tokens = _resolve_board_mode(board_mode)

    dfs = {'train': df_train, 'val': df_val, 'val_wl': df_val_wl}

    features = {name: {} for name in dfs}
    scaling_stats = {}

    # Elo
    _validate_scale_keys(elo_scale, {'pooled_elo'}, 'elo_scale')
    mover_elo_col, opponent_elo_col = _resolve_elo_cols(elo_cols)
    trans_name, base_trans = _resolve_trans_name(elo_scale, 'pooled_elo', _ELO_TRANS, 'elo_scale')

    train_mover_raw = df_train[mover_elo_col].to_numpy(dtype='float64')
    train_opponent_raw = df_train[opponent_elo_col].to_numpy(dtype='float64')
    per_split_raw = {name: {'mover': df[mover_elo_col].to_numpy(dtype='float64'),
                             'opponent': df[opponent_elo_col].to_numpy(dtype='float64')}
                      for name, df in dfs.items()}
    scaled, stats = _fit_apply_pooled(train_mover_raw, train_opponent_raw, per_split_raw,
                                       trans_name, base_trans, 'elo_scale')
    scaling_stats['pooled_elo'] = stats
    for name in dfs:
        features[name]['mover_elo_unscaled'] = per_split_raw[name]['mover'].astype('float32')
        features[name]['opponent_elo_unscaled'] = per_split_raw[name]['opponent'].astype('float32')
        features[name]['mover_elo_scaled'] = scaled[name]['mover']
        features[name]['opponent_elo_scaled'] = scaled[name]['opponent']

    elo_bins = {name: _bin_elo_splits(df[mover_elo_col], df[opponent_elo_col], cfg) for name, df in dfs.items()}

    # Clock
    mover_clock_col = opponent_clock_col = time_control_col = None
    if clock_cols is not None:
        if clock_scale is None:
            raise ValueError('clock_scale is required whenever clock_cols is given.')
        _validate_scale_keys(clock_scale, {'pooled_clock', 'inc_flag', 'total_length_flag'}, 'clock_scale')
        mover_clock_col, opponent_clock_col, time_control_col = _resolve_clock_cols(clock_cols)

        for df in dfs.values():
            _validate_time_control(df[time_control_col], SEC_MAPPING.keys())

        per_split_prop = {}
        for name, df in dfs.items():
            m, o = _clock_props(df, mover_clock_col, opponent_clock_col, time_control_col)
            per_split_prop[name] = {'mover': m, 'opponent': o}
        train_mover_prop = per_split_prop['train']['mover']
        train_opponent_prop = per_split_prop['train']['opponent']

        trans_name, base_trans = _resolve_trans_name(clock_scale, 'pooled_clock', _CLOCK_TRANS, 'clock_scale')
        scaled, stats = _fit_apply_pooled(train_mover_prop, train_opponent_prop, per_split_prop,
                                           trans_name, base_trans, 'clock_scale')
        scaling_stats['pooled_clock'] = stats
        for name in dfs:
            features[name]['mover_clock_unscaled'] = per_split_prop[name]['mover'].astype('float32')
            features[name]['opponent_clock_unscaled'] = per_split_prop[name]['opponent'].astype('float32')
            features[name]['mover_clock_scaled'] = scaled[name]['mover']
            features[name]['opponent_clock_scaled'] = scaled[name]['opponent']

        for flag_name, mapping in (('inc_flag', INC_FLAG_MAPPING), ('total_length_flag', TOTAL_LENGTH_FLAG_MAPPING)):
            trans_name, base_trans = _resolve_trans_name(clock_scale, flag_name, _FLAG_TRANS, 'clock_scale')
            apply_fn = _TRANS_FIT_APPLY[base_trans][1]
            for name, df in dfs.items():
                raw = df[time_control_col].map(mapping).to_numpy(dtype='float64')
                features[name][f'{flag_name}_unscaled'] = raw.astype('float32')
                features[name][f'{flag_name}_scaled'] = apply_fn(raw, {})
            scaling_stats[flag_name] = {'trans_name': trans_name}

    # Past
    past_cols_resolved = None
    if past_cols is not None:
        if past_scale is None:
            raise ValueError('past_scale is required whenever past_cols is given.')
        expected_scale_keys = {'pooled_past'}
        if len(past_cols) >= 4:
            expected_scale_keys.add('pooled_hours_since')
            expected_scale_keys.add('pooled_has_history')
            expected_scale_keys.add('pooled_new_player')
        if len(past_cols) == 6:
            expected_scale_keys.add('pooled_last_result')
        _validate_scale_keys(past_scale, expected_scale_keys, 'past_scale')
        resolved = _resolve_past_cols(past_cols)
        past_cols_resolved = resolved
        has_hours = len(past_cols) >= 4

        has_history = None
        if has_hours:
            has_history = {name: {'mover': _has_history_mask(df, resolved['hours_since_mover']),
                                   'opponent': _has_history_mask(df, resolved['hours_since_opponent'])}
                            for name, df in dfs.items()}

        trans_name, base_trans = _resolve_trans_name(past_scale, 'pooled_past', _PAST_TRANS, 'past_scale')
        per_split_raw = {name: {'mover': df[resolved['past_mover']].to_numpy(dtype='float64'),
                                 'opponent': df[resolved['past_opponent']].to_numpy(dtype='float64')}
                          for name, df in dfs.items()}

        no_history_fill = None
        fit_raw = per_split_raw
        if has_hours:
            train_has_history_vals = np.concatenate([
                per_split_raw['train']['mover'][has_history['train']['mover']],
                per_split_raw['train']['opponent'][has_history['train']['opponent']],
            ])
            no_history_fill = float(train_has_history_vals.mean())
            fit_raw = {
                name: {'mover': np.where(has_history[name]['mover'], vals['mover'], no_history_fill),
                       'opponent': np.where(has_history[name]['opponent'], vals['opponent'], no_history_fill)}
                for name, vals in per_split_raw.items()
            }

        scaled, stats = _fit_apply_pooled(fit_raw['train']['mover'], fit_raw['train']['opponent'], fit_raw,
                                           trans_name, base_trans, 'past_scale')
        if no_history_fill is not None:
            stats['no_history_fill'] = no_history_fill
        scaling_stats['pooled_past'] = stats
        for name in dfs:
            features[name]['past_mover_unscaled'] = per_split_raw[name]['mover'].astype('float32')
            features[name]['past_opponent_unscaled'] = per_split_raw[name]['opponent'].astype('float32')
            features[name]['past_mover_scaled'] = scaled[name]['mover']
            features[name]['past_opponent_scaled'] = scaled[name]['opponent']

        if has_hours:
            hist_trans_name, hist_base_trans = _resolve_trans_name(
                past_scale, 'pooled_has_history', _FLAG_TRANS, 'past_scale')
            new_trans_name, new_base_trans = _resolve_trans_name(
                past_scale, 'pooled_new_player', _FLAG_TRANS, 'past_scale')
            hist_apply_fn = _TRANS_FIT_APPLY[hist_base_trans][1]
            new_apply_fn = _TRANS_FIT_APPLY[new_base_trans][1]

            for name, df in dfs.items():
                m_elo = df[mover_elo_col].to_numpy(dtype='float64')
                o_elo = df[opponent_elo_col].to_numpy(dtype='float64')

                for side, notna, elo in (('mover', has_history[name]['mover'], m_elo),
                                          ('opponent', has_history[name]['opponent'], o_elo)):
                    hist_raw = notna.astype('float64')
                    features[name][f'has_history_{side}_unscaled'] = hist_raw.astype('float32')
                    features[name][f'has_history_{side}_scaled'] = hist_apply_fn(hist_raw, {})

                    new_player_raw = ((~notna) & (elo == NEW_USER_STARTING_ELO)).astype('float64')
                    features[name][f'new_player_{side}_unscaled'] = new_player_raw.astype('float32')
                    features[name][f'new_player_{side}_scaled'] = new_apply_fn(new_player_raw, {})

            scaling_stats['pooled_has_history'] = {'trans_name': hist_trans_name}
            scaling_stats['pooled_new_player'] = {'trans_name': new_trans_name}

            train_max_hours = max(
                pd.to_numeric(df_train[resolved['hours_since_mover']], errors='coerce').max(skipna=True),
                pd.to_numeric(df_train[resolved['hours_since_opponent']], errors='coerce').max(skipna=True),
            )

            per_split_prop = {}
            for name, df in dfs.items():
                m, o = _hours_since_props(df, resolved['hours_since_mover'], resolved['hours_since_opponent'],
                                           train_max_hours)
                per_split_prop[name] = {'mover': m, 'opponent': o}

            trans_name, base_trans = _resolve_trans_name(past_scale, 'pooled_hours_since', _HOURS_TRANS, 'past_scale')
            scaled, stats = _fit_apply_pooled(per_split_prop['train']['mover'], per_split_prop['train']['opponent'],
                                               per_split_prop, trans_name, base_trans, 'past_scale')
            stats['max_hours'] = float(train_max_hours)
            scaling_stats['pooled_hours_since'] = stats
            for name in dfs:
                features[name]['hours_since_mover_unscaled'] = per_split_prop[name]['mover'].astype('float32')
                features[name]['hours_since_opponent_unscaled'] = per_split_prop[name]['opponent'].astype('float32')
                features[name]['hours_since_mover_scaled'] = scaled[name]['mover']
                features[name]['hours_since_opponent_scaled'] = scaled[name]['opponent']
        else:
            print("past_cols has 2 entries, has_history_mover/opponent not created "
                  "(no NaN signal survives in past_mover/past_opponent to derive it from); "
                  "past no-history correction skipped for the same reason.")

        if len(past_cols) == 6:
            trans_name, base_trans = _resolve_trans_name(past_scale, 'pooled_last_result',
                                                           _LAST_RESULT_TRANS, 'past_scale')

            def _last_result_filled(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
                m = pd.to_numeric(df[resolved['last_result_mover']], errors='coerce').fillna(0.5).to_numpy(dtype='float64')
                o = pd.to_numeric(df[resolved['last_result_opponent']], errors='coerce').fillna(0.5).to_numpy(dtype='float64')
                return m, o

            train_mover_raw, train_opponent_raw = _last_result_filled(df_train)
            per_split_raw = {}
            for name, df in dfs.items():
                m, o = _last_result_filled(df)
                per_split_raw[name] = {'mover': m, 'opponent': o}

            scaled, stats = _fit_apply_pooled(train_mover_raw, train_opponent_raw, per_split_raw,
                                               trans_name, base_trans, 'past_scale')
            scaling_stats['pooled_last_result'] = stats
            for name in dfs:
                features[name]['last_result_mover_unscaled'] = per_split_raw[name]['mover'].astype('float32')
                features[name]['last_result_opponent_unscaled'] = per_split_raw[name]['opponent'].astype('float32')
                features[name]['last_result_mover_scaled'] = scaled[name]['mover']
                features[name]['last_result_opponent_scaled'] = scaled[name]['opponent']

    # Color
    color_col = None
    if color_cols is not None:
        if color_scale is None:
            raise ValueError('color_scale is required whenever color_cols is given.')
        _validate_scale_keys(color_scale, {'mover_is_white'}, 'color_scale')
        color_col = _resolve_color_cols(color_cols)
        trans_name, base_trans = _resolve_trans_name(color_scale, 'mover_is_white', _FLAG_TRANS, 'color_scale')
        apply_fn = _TRANS_FIT_APPLY[base_trans][1]
        for name, df in dfs.items():
            raw = df[color_col].to_numpy(dtype='float64')
            features[name]['mover_is_white_unscaled'] = raw.astype('float32')
            features[name]['mover_is_white_scaled'] = apply_fn(raw, {})
        scaling_stats['mover_is_white'] = {'trans_name': trans_name}

    # Ply
    ply_col = None
    if ply_cols is not None:
        if ply_scale is None:
            raise ValueError('ply_scale is required whenever ply_cols is given.')
        _validate_scale_keys(ply_scale, {'ply_played'}, 'ply_scale')
        ply_col = _resolve_ply_cols(ply_cols)
        trans_name, base_trans = _resolve_trans_name(ply_scale, 'ply_played', _PLY_TRANS, 'ply_scale')
        fit_fn, apply_fn = _TRANS_FIT_APPLY[base_trans]
        train_raw = df_train[ply_col].to_numpy(dtype='float64')
        stats = fit_fn(train_raw)
        scaling_stats['ply_played'] = {'trans_name': trans_name, **stats}
        for name, df in dfs.items():
            raw = df[ply_col].to_numpy(dtype='float64')
            features[name]['ply_played_unscaled'] = raw.astype('float32')
            features[name]['ply_played_scaled'] = apply_fn(raw, stats)

    # Rematch
    rematch_flag_col = rematch_prev_col = None
    if rematch_cols is not None:
        if rematch_scale is None:
            raise ValueError('rematch_scale is required whenever rematch_cols is given.')
        _validate_scale_keys(rematch_scale, {'rematch_prev_result'}, 'rematch_scale')
        rematch_flag_col, rematch_prev_col = _resolve_rematch_cols(rematch_cols)
        trans_name, base_trans = _resolve_trans_name(rematch_scale, 'rematch_prev_result', _FLAG_TRANS, 'rematch_scale')
        apply_fn = _TRANS_FIT_APPLY[base_trans][1]
        for name, df in dfs.items():
            is_rematch = df[rematch_flag_col].to_numpy()
            prev = df[rematch_prev_col].to_numpy(dtype='float64')
            raw = np.where(is_rematch, (prev - 0.5) * 2.0, 0.0)
            features[name]['rematch_prev_result_unscaled'] = raw.astype('float32')
            features[name]['rematch_prev_result_scaled'] = apply_fn(raw, {})
        scaling_stats['rematch_prev_result'] = {'trans_name': trans_name}

    # Title
    title_mover_col = title_opponent_col = None
    if title_cols is not None:
        if title_scale is None:
            raise ValueError('title_scale is required whenever title_cols is given.')
        _validate_scale_keys(title_scale, {'pooled_title'}, 'title_scale')
        if 'pooled_title' not in title_scale:
            raise ValueError("title_scale must include 'pooled_title': 'cat'.")
        if title_scale['pooled_title'] != 'cat':
            raise ValueError(f"title_scale['pooled_title'] must be 'cat', got {title_scale['pooled_title']!r}.")
        title_mover_col, title_opponent_col = _resolve_title_cols(title_cols)
        for name, df in dfs.items():
            m_idx = encode_title_idx(df[title_mover_col]).astype('float32')
            o_idx = encode_title_idx(df[title_opponent_col]).astype('float32')
            features[name]['mover_title_unscaled'] = m_idx
            features[name]['opponent_title_unscaled'] = o_idx
            features[name]['mover_title_scaled'] = m_idx
            features[name]['opponent_title_scaled'] = o_idx
        scaling_stats['pooled_title'] = {'trans_name': 'cat'}

    # Stockfish
    if sf_depth is not None and sf_scale is None:
        raise ValueError('sf_scale is required whenever sf_depth is given.')
    if sf_scale is not None and sf_depth is None:
        raise ValueError('sf_depth is required whenever sf_scale is given.')

    if sf_depth is not None:
        _validate_scale_keys(sf_scale, {'stockfish_eval'}, 'sf_scale')
        trans_name, base_trans = _resolve_trans_name(sf_scale, 'stockfish_eval', _SF_TRANS, 'sf_scale')
        fit_fn, apply_fn = _TRANS_FIT_APPLY[base_trans]

        sf_raw = {name: _stockfish_eval_fens(df[fen_col].to_numpy(), sf_depth) for name, df in dfs.items()}
        stats = fit_fn(sf_raw['train'])
        scaling_stats['stockfish_eval'] = {'trans_name': trans_name, **stats}
        for name in dfs:
            winprob = cp_to_mover_winprob(sf_raw[name])
            features[name]['stockfish_eval_unscaled'] = sf_raw[name].astype('float32')
            features[name]['stockfish_eval_scaled'] = apply_fn(sf_raw[name], stats)
            features[name]['stockfish_winprob_unscaled'] = winprob.astype('float32')
            features[name]['stockfish_winprob_scaled'] = winprob.astype('float32')

    print(f'Feature prep done, RSS: {_rss_gb():.2f} GB')

    # Assemble, save, memory-map
    splits = {}
    for name, df in dfs.items():
        t0 = time.time()
        n_rows = len(df)
        split_dir = os.path.join(out_dir, name)
        os.makedirs(split_dir, exist_ok=True)

        fens = df[fen_col].to_numpy()
        _write_boards_tokens_and_aux(split_dir, fens, desc=f'{name} boards/tokens/aux', chunk_size=chunk_size,
                                      write_boards=write_boards, write_tokens=write_tokens,
                                      aux_targets=aux_targets)

        elo_mean_bin, elo_self_bin, elo_oppo_bin, n_elo_bins, bin_labels = elo_bins[name]
        _write_split_arrays(
            split_dir, elo_mean_bin, elo_self_bin, elo_oppo_bin,
            {k: torch.tensor(v, dtype=torch.float32) for k, v in features[name].items()},
            encode_result_class(df[mover_result_col]).to(torch.uint8),
            encode_result_continuous(df[mover_result_col]),
            n_elo_bins, bin_labels,
            df[game_id_col].astype(str).tolist(), df[fen_col].tolist(),
            has_boards=write_boards, has_board_token_ids=write_tokens, has_aux_targets=aux_targets,
        )
        print(f'{name:>7}: {n_rows:>9,} rows prepared in {time.time() - t0:.2f}s')

        splits[name] = load_split(split_dir)
        print(f'{name:>7}: saved and memory-mapped from {split_dir}. RSS: {_rss_gb():.2f} GB')

    train_out = splits['train']

    base_names = sorted({k.rsplit('_', 1)[0] for k in train_out.features})
    print('Features:')
    for i in range(0, len(base_names), 6):
        print('  ' + ', '.join(base_names[i:i + 6]))
    print(f'Elo bins: {train_out.n_elo_bins} ({train_out.bin_labels[0]} ... {train_out.bin_labels[-1]})')

    class_counts = torch.bincount(train_out.result_class).tolist()
    print(f'Target distribution (train, by class index): {class_counts}')
    print(f'prepare_splits done. RSS: {_rss_gb():.2f} GB')

    prep_cfg = PrepConfig(
        mover_result_col=mover_result_col,
        fen_col=fen_col,
        game_id_col=game_id_col,
        cfg=cfg,
        mover_elo_col=mover_elo_col,
        opponent_elo_col=opponent_elo_col,
        mover_clock_col=mover_clock_col,
        opponent_clock_col=opponent_clock_col,
        time_control_col=time_control_col,
        past_cols_resolved=past_cols_resolved,
        color_col=color_col,
        ply_col=ply_col,
        rematch_flag_col=rematch_flag_col,
        rematch_prev_col=rematch_prev_col,
        title_mover_col=title_mover_col,
        title_opponent_col=title_opponent_col,
        board_mode=board_mode,
        aux_targets=aux_targets,
        sf_depth=sf_depth,
        scaling_stats=scaling_stats,
    )

    return train_out, splits['val'], splits['val_wl'], prep_cfg


def apply_prepared_splits(df: pd.DataFrame, prep_cfg: 'PrepConfig', out_dir: str,
                           chunk_size: int = CHUNK_SIZE) -> 'SplitData':
    """Applies an already-fitted PrepConfig to new raw data, returning a memory-mapped SplitData."""
    required = [prep_cfg.mover_result_col, prep_cfg.fen_col, prep_cfg.game_id_col,
                prep_cfg.mover_elo_col, prep_cfg.opponent_elo_col]
    if prep_cfg.mover_clock_col is not None:
        required += [prep_cfg.mover_clock_col, prep_cfg.opponent_clock_col, prep_cfg.time_control_col]
    if prep_cfg.past_cols_resolved is not None:
        required += list(prep_cfg.past_cols_resolved.values())
    if prep_cfg.color_col is not None:
        required.append(prep_cfg.color_col)
    if prep_cfg.ply_col is not None:
        required.append(prep_cfg.ply_col)
    if prep_cfg.rematch_flag_col is not None:
        required += [prep_cfg.rematch_flag_col, prep_cfg.rematch_prev_col]
    if prep_cfg.title_mover_col is not None:
        required += [prep_cfg.title_mover_col, prep_cfg.title_opponent_col]

    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f'apply_prepared_splits: df is missing required column(s) {missing}.')

    write_boards, write_tokens = _resolve_board_mode(prep_cfg.board_mode)

    stats = prep_cfg.scaling_stats
    features = {}

    # Elo
    mover_raw = df[prep_cfg.mover_elo_col].to_numpy(dtype='float64')
    opponent_raw = df[prep_cfg.opponent_elo_col].to_numpy(dtype='float64')
    features['mover_elo_unscaled'] = mover_raw.astype('float32')
    features['opponent_elo_unscaled'] = opponent_raw.astype('float32')
    features['mover_elo_scaled'] = _apply_stored_stats(mover_raw, stats['pooled_elo'])
    features['opponent_elo_scaled'] = _apply_stored_stats(opponent_raw, stats['pooled_elo'])

    # Clock
    if prep_cfg.mover_clock_col is not None:
        _validate_time_control(df[prep_cfg.time_control_col], SEC_MAPPING.keys())
        mover_prop, opponent_prop = _clock_props(df, prep_cfg.mover_clock_col, prep_cfg.opponent_clock_col,
                                                  prep_cfg.time_control_col)
        features['mover_clock_unscaled'] = mover_prop.astype('float32')
        features['opponent_clock_unscaled'] = opponent_prop.astype('float32')
        features['mover_clock_scaled'] = _apply_stored_stats(mover_prop, stats['pooled_clock'])
        features['opponent_clock_scaled'] = _apply_stored_stats(opponent_prop, stats['pooled_clock'])

        for flag_name, mapping in (('inc_flag', INC_FLAG_MAPPING), ('total_length_flag', TOTAL_LENGTH_FLAG_MAPPING)):
            raw = df[prep_cfg.time_control_col].map(mapping).to_numpy(dtype='float64')
            features[f'{flag_name}_unscaled'] = raw.astype('float32')
            features[f'{flag_name}_scaled'] = _apply_stored_stats(raw, stats[flag_name])

    # Past
    if prep_cfg.past_cols_resolved is not None:
        resolved = prep_cfg.past_cols_resolved
        mover_raw = df[resolved['past_mover']].to_numpy(dtype='float64')
        opponent_raw = df[resolved['past_opponent']].to_numpy(dtype='float64')
        features['past_mover_unscaled'] = mover_raw.astype('float32')
        features['past_opponent_unscaled'] = opponent_raw.astype('float32')

        has_hours = 'hours_since_mover' in resolved
        if has_hours:
            m_has_history = _has_history_mask(df, resolved['hours_since_mover'])
            o_has_history = _has_history_mask(df, resolved['hours_since_opponent'])
            fill = stats['pooled_past']['no_history_fill']
            fit_mover_raw = np.where(m_has_history, mover_raw, fill)
            fit_opponent_raw = np.where(o_has_history, opponent_raw, fill)
        else:
            fit_mover_raw, fit_opponent_raw = mover_raw, opponent_raw

        features['past_mover_scaled'] = _apply_stored_stats(fit_mover_raw, stats['pooled_past'])
        features['past_opponent_scaled'] = _apply_stored_stats(fit_opponent_raw, stats['pooled_past'])

        if has_hours:
            for side, notna in (('mover', m_has_history), ('opponent', o_has_history)):
                hist_raw = notna.astype('float64')
                features[f'has_history_{side}_unscaled'] = hist_raw.astype('float32')
                features[f'has_history_{side}_scaled'] = _apply_stored_stats(hist_raw, stats['pooled_has_history'])

                side_elo_col = prep_cfg.mover_elo_col if side == 'mover' else prep_cfg.opponent_elo_col
                elo = df[side_elo_col].to_numpy(dtype='float64')
                new_player_raw = ((~notna) & (elo == NEW_USER_STARTING_ELO)).astype('float64')
                features[f'new_player_{side}_unscaled'] = new_player_raw.astype('float32')
                features[f'new_player_{side}_scaled'] = _apply_stored_stats(new_player_raw, stats['pooled_new_player'])

            max_hours = stats['pooled_hours_since']['max_hours']
            m_hours, o_hours = _hours_since_props(df, resolved['hours_since_mover'], resolved['hours_since_opponent'],
                                                   max_hours)
            features['hours_since_mover_unscaled'] = m_hours.astype('float32')
            features['hours_since_opponent_unscaled'] = o_hours.astype('float32')
            features['hours_since_mover_scaled'] = _apply_stored_stats(m_hours, stats['pooled_hours_since'])
            features['hours_since_opponent_scaled'] = _apply_stored_stats(o_hours, stats['pooled_hours_since'])

        if 'last_result_mover' in resolved:
            m_res = pd.to_numeric(df[resolved['last_result_mover']], errors='coerce').fillna(0.5).to_numpy(dtype='float64')
            o_res = pd.to_numeric(df[resolved['last_result_opponent']], errors='coerce').fillna(0.5).to_numpy(dtype='float64')
            features['last_result_mover_unscaled'] = m_res.astype('float32')
            features['last_result_opponent_unscaled'] = o_res.astype('float32')
            features['last_result_mover_scaled'] = _apply_stored_stats(m_res, stats['pooled_last_result'])
            features['last_result_opponent_scaled'] = _apply_stored_stats(o_res, stats['pooled_last_result'])

    # Color
    if prep_cfg.color_col is not None:
        raw = df[prep_cfg.color_col].to_numpy(dtype='float64')
        features['mover_is_white_unscaled'] = raw.astype('float32')
        features['mover_is_white_scaled'] = _apply_stored_stats(raw, stats['mover_is_white'])

    # Ply
    if prep_cfg.ply_col is not None:
        raw = df[prep_cfg.ply_col].to_numpy(dtype='float64')
        features['ply_played_unscaled'] = raw.astype('float32')
        features['ply_played_scaled'] = _apply_stored_stats(raw, stats['ply_played'])

    # Rematch
    if prep_cfg.rematch_flag_col is not None:
        is_rematch = df[prep_cfg.rematch_flag_col].to_numpy()
        prev = df[prep_cfg.rematch_prev_col].to_numpy(dtype='float64')
        raw = np.where(is_rematch, (prev - 0.5) * 2.0, 0.0)
        features['rematch_prev_result_unscaled'] = raw.astype('float32')
        features['rematch_prev_result_scaled'] = _apply_stored_stats(raw, stats['rematch_prev_result'])

    # Title
    if prep_cfg.title_mover_col is not None:
        m_idx = encode_title_idx(df[prep_cfg.title_mover_col]).astype('float32')
        o_idx = encode_title_idx(df[prep_cfg.title_opponent_col]).astype('float32')
        features['mover_title_unscaled'] = m_idx
        features['opponent_title_unscaled'] = o_idx
        features['mover_title_scaled'] = m_idx
        features['opponent_title_scaled'] = o_idx

    # Stockfish
    if prep_cfg.sf_depth is not None:
        raw = _stockfish_eval_fens(df[prep_cfg.fen_col].to_numpy(), prep_cfg.sf_depth)
        winprob = cp_to_mover_winprob(raw)
        features['stockfish_eval_unscaled'] = raw.astype('float32')
        features['stockfish_eval_scaled'] = _apply_stored_stats(raw, stats['stockfish_eval'])
        features['stockfish_winprob_unscaled'] = winprob.astype('float32')
        features['stockfish_winprob_scaled'] = winprob.astype('float32')

    print(f'Feature prep done, RSS: {_rss_gb():.2f} GB')

    # Assemble, save, memory-map
    t0 = time.time()
    n_rows = len(df)
    os.makedirs(out_dir, exist_ok=True)

    fens = df[prep_cfg.fen_col].to_numpy()
    _write_boards_tokens_and_aux(out_dir, fens, desc='boards/tokens/aux', chunk_size=chunk_size,
                                  write_boards=write_boards, write_tokens=write_tokens,
                                  aux_targets=prep_cfg.aux_targets)

    elo_mean_bin, elo_self_bin, elo_oppo_bin, n_elo_bins, bin_labels = _bin_elo_splits(
        df[prep_cfg.mover_elo_col], df[prep_cfg.opponent_elo_col], prep_cfg.cfg)

    _write_split_arrays(
        out_dir, elo_mean_bin, elo_self_bin, elo_oppo_bin,
        {k: torch.tensor(v, dtype=torch.float32) for k, v in features.items()},
        encode_result_class(df[prep_cfg.mover_result_col]).to(torch.uint8),
        encode_result_continuous(df[prep_cfg.mover_result_col]),
        n_elo_bins, bin_labels,
        df[prep_cfg.game_id_col].astype(str).tolist(), df[prep_cfg.fen_col].tolist(),
        has_boards=write_boards, has_board_token_ids=write_tokens, has_aux_targets=prep_cfg.aux_targets,
    )
    print(f'{n_rows:>9,} rows prepared in {time.time() - t0:.2f}s')

    split = load_split(out_dir)
    print(f'saved and memory-mapped from {out_dir}. RSS: {_rss_gb():.2f} GB')

    base_names = sorted({k.rsplit('_', 1)[0] for k in split.features})
    print('Features:')
    for i in range(0, len(base_names), 6):
        print('  ' + ', '.join(base_names[i:i + 6]))

    class_counts = torch.bincount(split.result_class).tolist()
    print(f'Target distribution (by class index): {class_counts}')

    return split

####################
# CLASSES
####################

# (a) DATA CONTAINERS

@dataclass
class SplitData:
    """Model inputs and targets for one split."""
    boards: torch.Tensor | None
    board_token_ids: torch.Tensor | None
    legal_dest: torch.Tensor | None
    attacked_mover: torch.Tensor | None
    attacked_opponent: torch.Tensor | None
    elo_mean_bin: torch.Tensor
    elo_self_bin: torch.Tensor
    elo_oppo_bin: torch.Tensor
    features: dict
    result_class: torch.Tensor
    result_cont: torch.Tensor
    n_elo_bins: int
    bin_labels: list
    game_id: list
    fen: list

    def __len__(self) -> int:
        return len(self.result_class)


@dataclass
class PrepConfig:
    """Resolved column names, transform choices, and fitted scaling stats needed to reapply prepare_splits to new data."""
    mover_result_col: str
    fen_col: str
    game_id_col: str
    cfg: EloBinConfig

    mover_elo_col: str
    opponent_elo_col: str

    mover_clock_col: str | None = None
    opponent_clock_col: str | None = None
    time_control_col: str | None = None

    past_cols_resolved: dict[str, str] | None = None

    color_col: str | None = None
    ply_col: str | None = None

    rematch_flag_col: str | None = None
    rematch_prev_col: str | None = None

    title_mover_col: str | None = None
    title_opponent_col: str | None = None

    board_mode: str = 'both'
    aux_targets: bool = False
    sf_depth: int | None = None

    scaling_stats: dict = field(default_factory=dict)


# (b) STOCKFISH INTERNAL CONTROL FLOW

class _StockfishTimeout(Exception):
    """Raised when a single position's stockfish analysis exceeds ATTEMPT_TIMEOUT_SECONDS."""