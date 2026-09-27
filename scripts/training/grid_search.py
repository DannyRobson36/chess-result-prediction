"""
grid_search.py
Cartesian-searches model and training config grids for one fixed architecture on prepared
splits, via run_training, ranking parameter influence on val loss and on per-epoch runtime.

Latest changes: 27/09/26:
- Docstring tightening, updated reporting to match most recent training.py
"""

import dataclasses
import itertools
import time
from collections import defaultdict
from typing import Any

import numpy as np
import pandas as pd
import torch

from scripts.features.features import SplitData
from scripts.models.model_arch import MODEL_REGISTRY, build_model, get_output_type
from scripts.training.training import TRAIN_PROBE_SIZE, TrainConfig, get_device, probe_idx, run_training
from scripts.utils.utils_eval import metric_mode, is_better
from scripts.utils.utils_setup import set_seed

####################
# FUNCTIONS
####################

# (a) GRID EXPANSION

def _resolve_grid_axes(grid: dict, fixed_groups: list[tuple] | None) -> list[tuple]:
    """Groups grid keys into (keys, value_tuples) axes, pairing fixed_groups."""
    fixed_groups = fixed_groups or []
    axes = []
    grouped_keys = set()

    for group in fixed_groups:
        group = tuple(group)
        present = [k for k in group if k in grid]
        if not present:
            continue
        if len(present) != len(group):
            missing = [k for k in group if k not in grid]
            raise ValueError(f'fixed_group {group} has keys in neither grid: {missing}.')
        lengths = {len(grid[k]) for k in group}
        if len(lengths) != 1:
            raise ValueError(f'fixed_group {group} needs equal-length value lists, '
                             f'got lengths {[len(grid[k]) for k in group]}.')
        n = lengths.pop()
        axes.append((group, [tuple(grid[k][i] for k in group) for i in range(n)]))
        grouped_keys.update(group)

    remaining_keys = [k for k in grid if k not in grouped_keys]
    for k in remaining_keys:
        axes.append(((k,), [(v,) for v in grid[k]]))

    return axes


def _grid_combinations(grid: dict, fixed_groups: list[tuple] | None = None) -> list[dict]:
    """Expands grid into list of param dicts, pairing fixed_groups."""
    if not grid:
        return [{}]
    axes = _resolve_grid_axes(grid, fixed_groups)

    combos = []
    for combo_tuples in itertools.product(*[vals for _, vals in axes]):
        d = {}
        for (keys, _), values in zip(axes, combo_tuples):
            d.update(dict(zip(keys, values)))
        combos.append(d)
    return combos

# (b) PER-COMBO REPORTING

def _format_metrics_line(metrics: dict, prefix: str = 'val') -> str:
    """Formats metrics as labelled summary line."""
    return '  |  '.join(f'{prefix}_{k}: {v:.4f}' for k, v in metrics.items())

# (c) PARAMETER INFLUENCE

def _hashable(val: Any) -> Any:
    """Converts dataclasses, dicts and lists to hashable tuples."""
    if dataclasses.is_dataclass(val) and not isinstance(val, type):
        val = dataclasses.asdict(val)
    if isinstance(val, dict):
        return tuple(sorted((k, _hashable(v)) for k, v in val.items()))
    if isinstance(val, (list, tuple)):
        return tuple(_hashable(v) for v in val)
    return val


def _param_source(keys: tuple, cfg_grid_keys: set, train_grid_keys: set) -> str:
    """Labels axis keys as 'model', 'train' or 'mixed'."""
    in_cfg = bool(set(keys) & cfg_grid_keys)
    in_train = bool(set(keys) & train_grid_keys)
    if in_cfg and in_train:
        return 'mixed'
    return 'model' if in_cfg else 'train'


def _param_influence_table(rows: list[dict], combined_grid: dict, fixed_groups: list[tuple] | None,
                           cfg_grid_keys: set, train_grid_keys: set,
                           metric_key: str, metric_label: str) -> pd.DataFrame:
    """Ranks swept params by count-weighted std of per-level mean metric_key."""
    axes = _resolve_grid_axes(combined_grid, fixed_groups)
    out_rows = []

    for keys, _ in axes:
        level_values = defaultdict(list)
        for row in rows:
            level = tuple(_hashable(row[k]) for k in keys)
            level_values[level].append(row[metric_key])
        if len(level_values) <= 1:
            continue

        level_means = {level: float(np.mean(vals)) for level, vals in level_values.items()}
        counts = {level: len(vals) for level, vals in level_values.items()}
        n_total = sum(counts.values())
        weighted_mean = sum(counts[l] * level_means[l] for l in level_means) / n_total
        weighted_var = sum(counts[l] * (level_means[l] - weighted_mean) ** 2 for l in level_means) / n_total

        display_levels = {
            (level[0] if len(keys) == 1 else level): round(mean_val, 4)
            for level, mean_val in level_means.items()
        }

        out_rows.append({
            'param': '+'.join(keys),
            'source': _param_source(keys, cfg_grid_keys, train_grid_keys),
            'n_levels': len(level_means),
            f'influence_on_{metric_label}': round(float(np.sqrt(weighted_var)), 4),
            f'mean_{metric_label}_by_level': display_levels,
        })

    if not out_rows:
        return pd.DataFrame(columns=['param', 'source', 'n_levels', f'influence_on_{metric_label}',
                                     f'mean_{metric_label}_by_level'])
    return pd.DataFrame(out_rows).sort_values(f'influence_on_{metric_label}', ascending=False).reset_index(drop=True)

# (d) MAIN GRID SEARCH

def grid_search(
    train: SplitData,
    val: SplitData,
    arch_name: str,
    cfg_grid: dict,
    train_grid: dict | None = None,
    fixed_groups: list[tuple] | None = None,
    probe_size: int = TRAIN_PROBE_SIZE,
    probe_val: bool = True,
    device: torch.device | None = None,
    verbose: int = 1,
) -> dict:
    """Grid-searches cfg_grid x train_grid for arch_name, returns best combo, results and influence tables."""
    t_start = time.time()
    device = device or get_device()
    train_grid = train_grid or {}

    if arch_name not in MODEL_REGISTRY:
        raise ValueError(f'Unknown arch_name {arch_name!r}, choose from {list(MODEL_REGISTRY)}.')
    _, cfg_cls = MODEL_REGISTRY[arch_name]
    output_type = get_output_type(arch_name)
    base_metric_keys = ['loss', 'accuracy', 'macro_f1'] if output_type == 'classification' else ['loss']

    overlap = set(cfg_grid) & set(train_grid)
    if overlap:
        raise ValueError(f'cfg_grid and train_grid share key name(s) {overlap}, rename to keep unique.')
    if 'primary_metric' in train_grid:
        raise ValueError('train_grid cannot sweep primary_metric.')

    combined_grid = {**cfg_grid, **train_grid}
    combos = _grid_combinations(combined_grid, fixed_groups)
    primary_metric = TrainConfig().primary_metric
    mode = metric_mode(primary_metric)

    if verbose >= 1:
        print(f'Running on: {device}')
        print(f'Architecture: {arch_name} ({output_type})')
        print(f'Grid search: {len(combos)} combo(s) to run.')

    train_probe_idx = probe_idx(train, probe_size, seed=0)
    val_probe_idx = probe_idx(val, probe_size, seed=0) if probe_val else None

    rows = []
    best_score, best_combo_desc = None, None
    best_result, best_model, best_model_cfg, best_train_cfg = None, None, None, None

    for i, combo in enumerate(combos):
        cfg_overrides = {k: v for k, v in combo.items() if k in cfg_grid}
        train_overrides = {k: v for k, v in combo.items() if k in train_grid}

        model_cfg = cfg_cls(**cfg_overrides)
        train_cfg = TrainConfig(**{**train_overrides, 'verbose': (verbose == 2)})

        if verbose >= 1:
            print(f'\n[combo {i + 1}/{len(combos)}]  cfg={cfg_overrides}  train={train_overrides}')

        set_seed(train_cfg.seed)
        model = build_model(arch_name, model_cfg, n_elo_bins=train.n_elo_bins)

        combo_start = time.time()
        result = run_training(model, arch_name, train, val, train_cfg,
                              train_probe_idx=train_probe_idx, val_probe_idx=val_probe_idx, device=device)
        combo_runtime = time.time() - combo_start

        combo_metric_keys = base_metric_keys + (['aux_loss'] if 'val_aux_loss' in result['history'] else [])
        best_epoch = result['best_epoch']
        best_idx = best_epoch - 1
        combo_metrics = {k: result['history'][f'val_{k}'][best_idx] for k in combo_metric_keys}
        n_epochs_run = len(result['history']['train_loss'])
        time_per_epoch = combo_runtime / n_epochs_run

        if verbose >= 1:
            print(_format_metrics_line(combo_metrics) +
                  f'  (best_epoch={best_epoch}, '
                  f'epochs_run={n_epochs_run}, time={combo_runtime:.2f}s, '
                  f'time_per_epoch={time_per_epoch:.2f}s)')

        rows.append({**combo, **combo_metrics, 'best_epoch': best_epoch,
                     'n_epochs_run': n_epochs_run, 'runtime': combo_runtime,
                     'time_per_epoch': time_per_epoch})

        score = combo_metrics[primary_metric]
        if best_score is None or is_better(score, best_score, mode):
            if best_model is not None:
                del best_model
                if device.type == 'cuda':
                    torch.cuda.empty_cache()
            model.load_state_dict(result['best_state_dict'])
            best_score = score
            best_combo_desc = {'arch_name': arch_name, 'cfg_overrides': cfg_overrides,
                               'train_overrides': train_overrides}
            best_result = result
            best_model = model
            best_model_cfg = model_cfg
            best_train_cfg = train_cfg
        else:
            del model
            if device.type == 'cuda':
                torch.cuda.empty_cache()

    results_df = pd.DataFrame(rows).sort_values(primary_metric, ascending=(mode == 'min')).reset_index(drop=True)

    best_cfg, best_train = best_combo_desc['cfg_overrides'], best_combo_desc['train_overrides']
    print(f'\nBest combo: arch={arch_name}  cfg={best_cfg}  train={best_train}')
    print(f'Best val {primary_metric}: {best_score:.4f}')

    influence_df = _param_influence_table(rows, combined_grid, fixed_groups, set(cfg_grid), set(train_grid),
                                          metric_key='loss', metric_label='val_loss')
    print('\nParameter influence on val loss (most influential first):')
    if influence_df.empty:
        print('No swept parameter had more than one level, nothing to rank.')
    else:
        print(influence_df.to_string(index=False))

    time_influence_df = _param_influence_table(rows, combined_grid, fixed_groups, set(cfg_grid), set(train_grid),
                                               metric_key='time_per_epoch', metric_label='time_per_epoch')
    print('\nParameter influence on per-epoch runtime (most influential first):')
    if time_influence_df.empty:
        print('No swept parameter had more than one level, nothing to rank.')
    else:
        print(time_influence_df.to_string(index=False))

    total_time = time.time() - t_start
    print(f'\nTotal grid search runtime: {total_time:.2f}s')

    return {
        'results_df': results_df,
        'influence_df': influence_df,
        'time_influence_df': time_influence_df,
        'best_combo': best_combo_desc,
        'best_model': best_model,
        'best_model_cfg': best_model_cfg,
        'best_train_cfg': best_train_cfg,
        'best_result': best_result,
        'arch_name': arch_name,
        'total_time': total_time,
        'device': device,
    }