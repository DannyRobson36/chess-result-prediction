"""
grid_search.py
Cartesian-searches model and training config grids for one or more architectures on prepared splits,
via run_training, ranking parameter influence on val loss.

Latest changes: 15/08/26:
- Initial commit
"""

import time
import itertools
from collections import defaultdict

import numpy as np
import pandas as pd
import torch

from scripts.features.features import SplitData, detect_two_way
from scripts.models.model_arch import MODEL_REGISTRY, build_model
from scripts.training.training import TrainConfig, get_device, set_seed, probe_idx, run_training
from scripts.utils.utils_eval import metric_mode, is_better

####################
# CONSTANTS
####################

COMBO_METRIC_KEYS = ("loss", "accuracy", "macro_f1")

####################
# FUNCTIONS
####################

# (a) GRID EXPANSION

def _resolve_grid_axes(grid: dict, fixed_groups: list[tuple] | None) -> list[tuple]:
    """Groups grid keys into (keys, list_of_value_tuples) axes, honoring fixed_groups pairing."""
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
            raise ValueError(
                f"fixed_group {group} has keys missing from this grid: {missing}. "
                f"If the group spans cfg_grid and train_grid, that is fine, but every "
                f"key in the group must appear in exactly one of the two grids."
            )
        lengths = {len(grid[k]) for k in group}
        if len(lengths) != 1:
            raise ValueError(
                f"fixed_group {group} requires equal-length value lists, "
                f"got lengths {[len(grid[k]) for k in group]}"
            )
        n = lengths.pop()
        axes.append((group, [tuple(grid[k][i] for k in group) for i in range(n)]))
        grouped_keys.update(group)

    remaining_keys = [k for k in grid if k not in grouped_keys]
    for k in remaining_keys:
        axes.append(((k,), [(v,) for v in grid[k]]))

    return axes


def _grid_combinations(grid: dict, fixed_groups: list[tuple] | None = None) -> list[dict]:
    """Expands a param -> values dict into a list of param -> value dicts, honoring fixed_groups."""
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

def _format_metrics_line(metrics: dict, prefix: str = "val") -> str:
    """Formats a metrics dict as a labelled summary line."""
    return "  |  ".join(f"{prefix}_{k}: {v:.4f}" for k, v in metrics.items())


# (c) PARAMETER INFLUENCE

def _param_source(keys: tuple, cfg_grid_keys: set, train_grid_keys: set) -> str:
    """Labels a (possibly grouped) axis as 'model', 'train', or 'mixed' by which grid(s) its keys came from."""
    in_cfg = bool(set(keys) & cfg_grid_keys)
    in_train = bool(set(keys) & train_grid_keys)
    if in_cfg and in_train:
        return "mixed"
    return "model" if in_cfg else "train"


def _param_influence_table(rows: list[dict], combined_grid: dict, fixed_groups: list[tuple] | None,
                            cfg_grid_keys: set, train_grid_keys: set, metric: str = "loss") -> pd.DataFrame:
    """Ranks swept params by the count-weighted std of their per-level mean val metric, most influential first."""
    axes = _resolve_grid_axes(combined_grid, fixed_groups)
    out_rows = []

    for keys, _ in axes:
        level_values = defaultdict(list)
        for row in rows:
            level = tuple(row[k] for k in keys)
            level_values[level].append(row[metric])
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
            "param": "+".join(keys),
            "source": _param_source(keys, cfg_grid_keys, train_grid_keys),
            "n_levels": len(level_means),
            f"influence_on_val_{metric}": round(float(np.sqrt(weighted_var)), 4),
            f"mean_val_{metric}_by_level": display_levels,
        })

    if not out_rows:
        return pd.DataFrame(columns=["param", "source", "n_levels", f"influence_on_val_{metric}",
                                      f"mean_val_{metric}_by_level"])
    return pd.DataFrame(out_rows).sort_values(f"influence_on_val_{metric}", ascending=False).reset_index(drop=True)


# (d) MAIN GRID SEARCH

def grid_search(
    train: SplitData,
    val: SplitData,
    cfg_grid: dict,
    train_grid: dict | None = None,
    fixed_groups: list[tuple] | None = None,
    probe_size: int = 100_000,
    probe_val: bool = True,
    device: torch.device | None = None,
    verbose: int = 1,
) -> dict:
    """Cartesian-searches cfg_grid x train_grid via run_training, defaulting anything not swept to its config class's own field default; returns the best combo, results table, and parameter-influence ranking."""
    t_start = time.time()
    device = device or get_device()
    train_grid = train_grid or {}

    if "arch_name" not in cfg_grid:
        raise ValueError("cfg_grid must include 'arch_name' (a list of one or more registered architecture names).")
    unknown_archs = [a for a in cfg_grid["arch_name"] if a not in MODEL_REGISTRY]
    if unknown_archs:
        raise ValueError(f"cfg_grid['arch_name'] contains unregistered architecture(s) {unknown_archs}, "
                          f"choose from {list(MODEL_REGISTRY)}.")

    overlap = set(cfg_grid) & set(train_grid)
    if overlap:
        raise ValueError(f"cfg_grid and train_grid share key name(s) {overlap}, "
                          f"rename so each param name is unique across the two grids.")
    if "primary_metric" in train_grid:
        raise ValueError("train_grid cannot sweep 'primary_metric', since combos would then be "
                          "compared against different, non-comparable target metrics.")

    two_way = detect_two_way(train, val)
    combined_grid = {**cfg_grid, **train_grid}
    combos = _grid_combinations(combined_grid, fixed_groups)
    primary_metric = TrainConfig().primary_metric
    mode = metric_mode(primary_metric)

    if verbose >= 1:
        print(f"Running on: {device}")
        mode_str = "two-way (win/loss only)" if two_way else "three-way (win/draw/loss)"
        print(f"Detected task mode: {mode_str}")
        print(f"Grid search: {len(combos)} combo(s) to run.")

    train_probe_idx = probe_idx(train, probe_size, seed=0)
    val_probe_idx = probe_idx(val, probe_size, seed=0) if probe_val else None

    rows = []
    best_score, best_combo_desc = None, None
    best_result, best_model, best_model_cfg, best_train_cfg = None, None, None, None

    for i, combo in enumerate(combos):
        cfg_overrides = {k: v for k, v in combo.items() if k in cfg_grid}
        train_overrides = {k: v for k, v in combo.items() if k in train_grid}

        arch_name = cfg_overrides.pop("arch_name")
        _, cfg_cls = MODEL_REGISTRY[arch_name]
        model_cfg = cfg_cls(**cfg_overrides)
        train_cfg = TrainConfig(**{**train_overrides, "verbose": (verbose == 2)})

        if verbose >= 1:
            print(f"\n[combo {i + 1}/{len(combos)}] arch={arch_name}  "
                  f"cfg={cfg_overrides}  train={train_overrides}")

        set_seed(train_cfg.seed)
        model = build_model(arch_name, model_cfg, n_elo_bins=train.n_elo_bins, two_way=two_way)

        combo_start = time.time()
        result = run_training(model, arch_name, train, val, train_cfg, two_way=two_way,
                               train_probe_idx=train_probe_idx, val_probe_idx=val_probe_idx, device=device)
        combo_runtime = time.time() - combo_start

        best_idx = result["best_epoch"] - 1
        combo_metrics = {k: result["history"][f"val_{k}"][best_idx] for k in COMBO_METRIC_KEYS}

        if verbose >= 1:
            print(_format_metrics_line(combo_metrics) +
                  f"  (best_epoch={result['best_epoch']}, "
                  f"epochs_run={len(result['history']['train_loss'])}, time={combo_runtime:.2f}s)")

        rows.append({**combo, **combo_metrics, "best_epoch": result["best_epoch"],
                     "n_epochs_run": len(result["history"]["train_loss"]), "runtime": combo_runtime})

        score = combo_metrics[primary_metric]
        if best_score is None or is_better(score, best_score, mode):
            if best_model is not None:
                del best_model
                if device.type == "cuda":
                    torch.cuda.empty_cache()
            best_score = score
            best_combo_desc = {"arch_name": arch_name, "cfg_overrides": cfg_overrides,
                                "train_overrides": train_overrides}
            best_result = result
            best_model = model
            best_model_cfg = model_cfg
            best_train_cfg = train_cfg
        else:
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()

    results_df = pd.DataFrame(rows).sort_values(primary_metric, ascending=(mode == "min")).reset_index(drop=True)

    print(f"\nBest combo: arch={best_combo_desc['arch_name']}  cfg={best_combo_desc['cfg_overrides']}  "
          f"train={best_combo_desc['train_overrides']}")
    print(f"Best val {primary_metric}: {best_score:.4f}")

    influence_df = _param_influence_table(rows, combined_grid, fixed_groups, set(cfg_grid), set(train_grid))
    print("\nParameter influence on val loss (most influential first):")
    if influence_df.empty:
        print("No swept parameter had more than one level; nothing to rank.")
    else:
        print(influence_df.to_string(index=False))

    total_time = time.time() - t_start
    print(f"\nTotal grid search runtime: {total_time:.2f}s")

    return {
        "results_df": results_df,
        "influence_df": influence_df,
        "best_combo": best_combo_desc,
        "best_model": best_model,
        "best_model_cfg": best_model_cfg,
        "best_train_cfg": best_train_cfg,
        "best_result": best_result,
        "arch_name": best_model_cfg.arch_name,
        "two_way": two_way,
        "total_time": total_time,
    }