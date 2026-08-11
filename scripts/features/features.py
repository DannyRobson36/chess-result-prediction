"""
features.py
Builds model-ready SplitData from raw dataframes: scaling, elo binning, task-mode detection.

Latest changes: 08/08/26:
- External helper remap_targets - notation corrected
"""

import json
import os
import time
import warnings
from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch

from scripts.utils.utils_chess import (
    EloBinConfig, ELO_BINS, elo_bin_edges, elo_bin_labels,
    fen_to_tensor, fen_to_token_ids, encode_result_class, encode_result_continuous,
    RESULT_CLASS_NAMES,
)

####################
# CONSTANTS
####################

_ELO_TRANS = {'raw': 'raw', 'normal': 'normal'}
_PLY_TRANS = {'raw': 'raw', 'normal': 'normal'}
_PAST_TRANS = {'raw': 'raw', 'normal': 'normal'}
_LAST_RESULT_TRANS = {'raw': 'raw', 'normal': 'normal'}
_HOURS_TRANS = {'raw': 'raw', 'symmetric_prop': 'centered', 'symmetric_prop_log': 'log_centered'}
_CLOCK_TRANS = {
    'raw': 'raw',
    'clock_prop_norm': 'normal',
    'clock_prop_centred': 'centered',
    'clock_prop_log_norm': 'log_norm',
    'clock_prop_log_centred': 'log_centered',
}
_FLAG_TRANS = {'raw': 'raw'}

SEC_MAPPING = {'600+0': 600, '600+5': 600, '900+10': 900}
INC_FLAG_MAPPING = {'600+0': 0, '600+5': 1, '900+10': 1}
TOTAL_LENGTH_FLAG_MAPPING = {'600+0': 0, '600+5': 0, '900+10': 1}

TWO_WAY_CLASS_NAMES = ['loss', 'win']

####################
# FUNCTIONS
####################

# (a) SIMPLE FEATURE TRANSFORMS

def cp_to_mover_winprob(cp: pd.Series | np.ndarray) -> pd.Series | np.ndarray:
    """Cp to mover win probability, lichess logistic curve."""
    return 0.5 + 0.5 * (2 / (1 + np.exp(-0.00368208 * cp)) - 1)


# (b) SPLIT DATA CONTAINER

@dataclass
class SplitData:
    """Model inputs and targets for one split."""
    boards: torch.Tensor
    board_token_ids: torch.Tensor
    elo_mean_bin: torch.Tensor
    elo_self_bin: torch.Tensor
    elo_oppo_bin: torch.Tensor
    features: dict
    result_class: torch.Tensor
    result_cont: torch.Tensor
    n_elo_bins: int
    bin_labels: list

    def __len__(self) -> int:
        return len(self.boards)


# (c) SCALING TRANSFORMS

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


_TRANS_FIT_APPLY = {
    'raw': (_fit_raw, _apply_raw),
    'normal': (_fit_normal, _apply_normal),
    'log_norm': (_fit_log_norm, _apply_log_norm),
    'centered': (_fit_centered, _apply_centered),
    'log_centered': (_fit_log_centered, _apply_log_centered),
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


# (d) COLUMN GROUP RESOLUTION

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


def _validate_time_control(values: pd.Series, valid_keys: Iterable[str]) -> None:
    """Checks time_control values are all in valid_keys."""
    unknown = set(values.unique()) - set(valid_keys)
    if unknown:
        raise ValueError(f'time_control contains unrecognized value(s) {sorted(unknown)}; '
                          f'expected only {sorted(valid_keys)}.')


# (e) TASK MODE DETECTION - TWO-WAY VS THREE-WAY

def detect_two_way(*splits: 'SplitData | None') -> bool:
    """True if no split contains a draw."""
    for split in splits:
        if split is None:
            continue
        if bool((split.result_class == 1).any()):
            return False
    return True


def class_names_for_mode(two_way: bool) -> list[str]:
    """Class names for the detected task mode."""
    return TWO_WAY_CLASS_NAMES if two_way else RESULT_CLASS_NAMES


def remap_targets(result_class: torch.Tensor, two_way: bool) -> torch.Tensor:
    """Remaps result_class to two-way (loss/win) labels."""
    return result_class // 2 if two_way else result_class


# (f) ELO BINNING FOR SPLITS

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


# (g) DISK STORAGE

def _save_split(split: SplitData, split_dir: str) -> None:
    """Writes SplitData fields to split_dir as .npy files."""
    os.makedirs(split_dir, exist_ok=True)
    np.save(os.path.join(split_dir, 'boards.npy'), split.boards.numpy())
    np.save(os.path.join(split_dir, 'board_token_ids.npy'), split.board_token_ids.numpy())
    np.save(os.path.join(split_dir, 'elo_mean_bin.npy'), split.elo_mean_bin.numpy())
    np.save(os.path.join(split_dir, 'elo_self_bin.npy'), split.elo_self_bin.numpy())
    np.save(os.path.join(split_dir, 'elo_oppo_bin.npy'), split.elo_oppo_bin.numpy())
    np.save(os.path.join(split_dir, 'result_class.npy'), split.result_class.numpy())
    np.save(os.path.join(split_dir, 'result_cont.npy'), split.result_cont.numpy())

    feature_dir = os.path.join(split_dir, 'features')
    os.makedirs(feature_dir, exist_ok=True)
    for name, tensor in split.features.items():
        np.save(os.path.join(feature_dir, f'{name}.npy'), tensor.numpy())

    with open(os.path.join(split_dir, 'meta.json'), 'w') as f:
        json.dump({'n_elo_bins': split.n_elo_bins, 'bin_labels': split.bin_labels}, f)


def load_split(split_dir: str) -> SplitData:
    """Loads SplitData from split_dir, memory-mapped."""
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

    return SplitData(
        boards=_mmap('boards'),
        board_token_ids=_mmap('board_token_ids'),
        elo_mean_bin=_mmap('elo_mean_bin'),
        elo_self_bin=_mmap('elo_self_bin'),
        elo_oppo_bin=_mmap('elo_oppo_bin'),
        features=features,
        result_class=_mmap('result_class'),
        result_cont=_mmap('result_cont'),
        n_elo_bins=meta['n_elo_bins'],
        bin_labels=meta['bin_labels'],
    )


# (h) DATA PREP PIPELINE

def prepare_splits(df_train: pd.DataFrame, df_val: pd.DataFrame, df_test: pd.DataFrame | None = None, *,
                    mover_result_col: str,
                    out_dir: str,
                    fen_col: str = 'fen',
                    elo_cols: list[str] | None = None, elo_scale: dict | None = None,
                    clock_cols: list[str] | None = None, clock_scale: dict | None = None,
                    past_cols: list[str] | None = None, past_scale: dict | None = None,
                    color_cols: list[str] | None = None, color_scale: dict | None = None,
                    ply_cols: list[str] | None = None, ply_scale: dict | None = None,
                    cfg: EloBinConfig = ELO_BINS) -> tuple[SplitData, SplitData, SplitData | None, dict, bool]:
    """Builds, saves, and memory-maps train/val/test SplitData."""
    if elo_cols is None or elo_scale is None:
        raise ValueError('elo_cols and elo_scale are mandatory.')

    dfs = {'train': df_train, 'val': df_val}
    if df_test is not None:
        dfs['test'] = df_test

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
    if clock_cols is not None:
        if clock_scale is None:
            raise ValueError('clock_scale is required whenever clock_cols is given.')
        _validate_scale_keys(clock_scale, {'pooled_clock', 'inc_flag', 'total_length_flag'}, 'clock_scale')
        mover_clock_col, opponent_clock_col, time_control_col = _resolve_clock_cols(clock_cols)

        for df in dfs.values():
            _validate_time_control(df[time_control_col], SEC_MAPPING.keys())

        def _clock_prop(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
            game_time = df[time_control_col].map(SEC_MAPPING).to_numpy(dtype='float64')
            m = df[mover_clock_col].to_numpy(dtype='float64') / game_time
            o = df[opponent_clock_col].to_numpy(dtype='float64') / game_time
            return m, o

        train_mover_prop, train_opponent_prop = _clock_prop(df_train)
        per_split_prop = {}
        for name, df in dfs.items():
            m, o = _clock_prop(df)
            per_split_prop[name] = {'mover': m, 'opponent': o}

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
    if past_cols is not None:
        if past_scale is None:
            raise ValueError('past_scale is required whenever past_cols is given.')
        expected_scale_keys = {'pooled_past'}
        if len(past_cols) >= 4:
            expected_scale_keys.add('pooled_hours_since')
        if len(past_cols) == 6:
            expected_scale_keys.add('pooled_last_result')
        _validate_scale_keys(past_scale, expected_scale_keys, 'past_scale')
        resolved = _resolve_past_cols(past_cols)

        trans_name, base_trans = _resolve_trans_name(past_scale, 'pooled_past', _PAST_TRANS, 'past_scale')
        train_mover_raw = df_train[resolved['past_mover']].to_numpy(dtype='float64')
        train_opponent_raw = df_train[resolved['past_opponent']].to_numpy(dtype='float64')
        per_split_raw = {name: {'mover': df[resolved['past_mover']].to_numpy(dtype='float64'),
                                 'opponent': df[resolved['past_opponent']].to_numpy(dtype='float64')}
                          for name, df in dfs.items()}
        scaled, stats = _fit_apply_pooled(train_mover_raw, train_opponent_raw, per_split_raw,
                                           trans_name, base_trans, 'past_scale')
        scaling_stats['pooled_past'] = stats
        for name in dfs:
            features[name]['past_mover_unscaled'] = per_split_raw[name]['mover'].astype('float32')
            features[name]['past_opponent_unscaled'] = per_split_raw[name]['opponent'].astype('float32')
            features[name]['past_mover_scaled'] = scaled[name]['mover']
            features[name]['past_opponent_scaled'] = scaled[name]['opponent']

        if len(past_cols) >= 4:
            has_history = {}
            for name, df in dfs.items():
                m_notna = pd.to_numeric(df[resolved['hours_since_mover']], errors='coerce').notna().to_numpy()
                o_notna = pd.to_numeric(df[resolved['hours_since_opponent']], errors='coerce').notna().to_numpy()
                has_history[name] = {'mover': m_notna, 'opponent': o_notna}

            for side in ('mover', 'opponent'):
                apply_fn = _TRANS_FIT_APPLY['raw'][1]
                for name in dfs:
                    raw = has_history[name][side].astype('float64')
                    features[name][f'has_history_{side}_unscaled'] = raw.astype('float32')
                    features[name][f'has_history_{side}_scaled'] = apply_fn(raw, {})
                scaling_stats[f'has_history_{side}'] = {'trans_name': 'raw'}

            train_max_hours = max(
                pd.to_numeric(df_train[resolved['hours_since_mover']], errors='coerce').max(skipna=True),
                pd.to_numeric(df_train[resolved['hours_since_opponent']], errors='coerce').max(skipna=True),
            )

            def _hours_prop(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
                m = (pd.to_numeric(df[resolved['hours_since_mover']], errors='coerce')
                     .fillna(train_max_hours).to_numpy(dtype='float64'))
                o = (pd.to_numeric(df[resolved['hours_since_opponent']], errors='coerce')
                     .fillna(train_max_hours).to_numpy(dtype='float64'))
                return m / train_max_hours, o / train_max_hours

            train_mover_prop, train_opponent_prop = _hours_prop(df_train)
            per_split_prop = {}
            for name, df in dfs.items():
                m, o = _hours_prop(df)
                per_split_prop[name] = {'mover': m, 'opponent': o}

            trans_name, base_trans = _resolve_trans_name(past_scale, 'pooled_hours_since', _HOURS_TRANS, 'past_scale')
            scaled, stats = _fit_apply_pooled(train_mover_prop, train_opponent_prop, per_split_prop,
                                               trans_name, base_trans, 'past_scale')
            stats['max_hours'] = float(train_max_hours)
            scaling_stats['pooled_hours_since'] = stats
            for name in dfs:
                features[name]['hours_since_mover_unscaled'] = per_split_prop[name]['mover'].astype('float32')
                features[name]['hours_since_opponent_unscaled'] = per_split_prop[name]['opponent'].astype('float32')
                features[name]['hours_since_mover_scaled'] = scaled[name]['mover']
                features[name]['hours_since_opponent_scaled'] = scaled[name]['opponent']
        else:
            print("past_cols has 2 entries, has_history_mover/opponent not created "
                  "(no NaN signal survives in past_mover/past_opponent to derive it from).")

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

    # Assemble, save, memory-map
    splits = {}
    for name, df in dfs.items():
        t0 = time.time()
        boards = torch.stack([fen_to_tensor(f) for f in df[fen_col]]).to(torch.bool)
        board_token_ids = torch.stack([fen_to_token_ids(f) for f in df[fen_col]]).to(torch.uint8)
        elo_mean_bin, elo_self_bin, elo_oppo_bin, n_elo_bins, bin_labels = elo_bins[name]

        split = SplitData(
            boards=boards,
            board_token_ids=board_token_ids,
            elo_mean_bin=torch.tensor(elo_mean_bin, dtype=torch.uint8),
            elo_self_bin=torch.tensor(elo_self_bin, dtype=torch.uint8),
            elo_oppo_bin=torch.tensor(elo_oppo_bin, dtype=torch.uint8),
            features={k: torch.tensor(v, dtype=torch.float32) for k, v in features[name].items()},
            result_class=encode_result_class(df[mover_result_col]).to(torch.uint8),
            result_cont=encode_result_continuous(df[mover_result_col]),
            n_elo_bins=n_elo_bins,
            bin_labels=bin_labels,
        )
        print(f'{name:>5}: {len(split):>9,} rows prepared in {time.time() - t0:.2f}s')

        split_dir = os.path.join(out_dir, name)
        _save_split(split, split_dir)
        splits[name] = load_split(split_dir)
        print(f'{name:>5}: saved and memory-mapped from {split_dir}')

    train_out = splits['train']
    two_way = detect_two_way(train_out, splits['val'], splits.get('test'))
    mode_str = 'two-way (win/loss only)' if two_way else 'three-way (win/draw/loss)'
    print(f'Detected task mode: {mode_str}')

    base_names = sorted({k.rsplit('_', 1)[0] for k in train_out.features})
    print('Features:')
    for i in range(0, len(base_names), 6):
        print('  ' + ', '.join(base_names[i:i + 6]))
    print(f'Elo bins: {train_out.n_elo_bins} ({train_out.bin_labels[0]} ... {train_out.bin_labels[-1]})')

    class_counts = torch.bincount(train_out.result_class).tolist()
    print(f'Target distribution (train, by class index): {class_counts}')

    return train_out, splits['val'], splits.get('test'), scaling_stats, two_way