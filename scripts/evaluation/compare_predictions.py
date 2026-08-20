"""
compare_predictions.py
Reads per-model prediction csvs against a fixed val/test positions df, merges them,
and provides eval-plots across models.

Latest changes: 20/08/26:
- Plot functions take an optional display_names dict, used for legend labels
"""

import glob
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

from scripts.utils.utils_chess import EloBinConfig, ELO_BINS, elo_bin_by_mover, elo_bin_labels, RESULT_TO_CLASS, RESULT_CLASS_NAMES
from scripts.utils.utils_eval import resolve_display_name
from scripts.utils.utils_plotting import colors_for, BASE_FIGSIZE
from scripts.features.features import SEC_MAPPING

####################
# CONSTANTS
####################

REQUIRED_PRED_COLS = ['game_id', 'fen', 'prob_win', 'prob_draw', 'prob_loss', 'predicted_class']

####################
# FUNCTIONS
####################

# (a) DISCOVERY & LOADING

def discover_predictions(predictions_dir: str, split: str) -> dict[str, str]:
    """Finds every {split}_predictions_{name}.csv in predictions_dir, returns {name: path}, sorted by name."""
    prefix = f'{split}_predictions_'
    pattern = os.path.join(predictions_dir, f'{prefix}*.csv')
    discovered = {}
    for path in sorted(glob.glob(pattern)):
        fname = os.path.basename(path)
        name = fname[len(prefix):-len('.csv')]
        discovered[name] = path
    return discovered


def _check_unique_keys(df: pd.DataFrame, label: str, game_id_col: str, fen_col: str) -> None:
    """Raises if df has duplicate (game_id_col, fen_col) combinations."""
    dupes = df.duplicated(subset=[game_id_col, fen_col])
    if dupes.any():
        raise ValueError(
            f'{label}: found {dupes.sum():,} duplicate ({game_id_col}, {fen_col}) combos '
            f'-- keys must be unique.'
        )


def combine_predictions(main_df: pd.DataFrame, predictions_dir: str, split: str,
                         game_id_col: str = 'game_id', fen_col: str = 'fen') -> tuple[pd.DataFrame, list[str]]:
    """Auto-discovers and merges every model's predictions csv in predictions_dir onto main_df."""
    if split not in ('val', 'test'):
        raise ValueError(f"split must be 'val' or 'test', got {split!r}")

    _check_unique_keys(main_df, 'main_df', game_id_col, fen_col)

    discovered = discover_predictions(predictions_dir, split)
    if not discovered:
        raise FileNotFoundError(f"No '{split}_predictions_*.csv' files found in {predictions_dir}")

    df_combined = main_df.copy()
    loaded_names = []

    for name, path in discovered.items():
        df_pred = pd.read_csv(path)

        missing_cols = [c for c in REQUIRED_PRED_COLS if c not in df_pred.columns]
        if missing_cols:
            raise ValueError(f'[{name}] predictions csv at {path} is missing column(s): {missing_cols}')

        df_pred = df_pred[REQUIRED_PRED_COLS].copy()
        _check_unique_keys(df_pred, f'[{name}] predictions df', game_id_col, fen_col)

        if len(df_pred) != len(main_df):
            raise ValueError(
                f'[{name}] predictions df has {len(df_pred):,} rows, expected {len(main_df):,} '
                f'(main_df row count).'
            )

        df_pred = df_pred.rename(columns={
            'prob_win': f'{name}_prob_win',
            'prob_draw': f'{name}_prob_draw',
            'prob_loss': f'{name}_prob_loss',
            'predicted_class': f'{name}_predicted_class',
        })

        n_before = len(df_combined)
        df_combined = df_combined.merge(
            df_pred, on=[game_id_col, fen_col], how='inner', validate='one_to_one',
        )
        if len(df_combined) != n_before:
            raise ValueError(
                f'[{name}] merge changed row count {n_before:,} -> {len(df_combined):,} '
                f'-- ({game_id_col}, {fen_col}) keys do not fully match main_df.'
            )

        loaded_names.append(name)

    print(f'combine_predictions: merged {len(loaded_names)} model(s) into '
          f'{len(df_combined):,} rows x {len(df_combined.columns)} cols -- {loaded_names}')

    return df_combined, loaded_names


# (b) ROW FILTERING

def restrict_to_common_rows(df: pd.DataFrame, cols: list[str], verbose: bool = True) -> pd.DataFrame:
    """Returns df restricted to rows with no NaN in any of cols."""
    mask = df[cols].notna().all(axis=1)
    if verbose and (~mask).any():
        print(f'restrict_to_common_rows: dropping {(~mask).sum():,} of {len(df):,} rows '
              f'-- NaN in one or more of {cols}.')
    return df[mask].copy()


# (c) PLOTS

def plot_accuracy_by_elo_bin(df: pd.DataFrame, names: list[str],
                              mover_result_col: str = 'mover_result',
                              cfg: EloBinConfig = ELO_BINS,
                              ylim: tuple[float, float] = (50, 70),
                              title: str = 'Result prediction accuracy by Elo bin',
                              show_hist: bool = False,
                              figsize: tuple[float, float] | None = None,
                              display_names: dict[str, str] | None = None) -> None:
    """Plots per-model accuracy against true mover_result, binned by mean elo."""
    cols_needed = [f'{name}_predicted_class' for name in names] + [mover_result_col, 'mover_elo', 'opponent_elo']
    df = restrict_to_common_rows(df, cols_needed)

    binned, edges = elo_bin_by_mover(df, cfg, method='mean')
    labels = elo_bin_labels(edges)
    n_elo_bins = len(edges) - 1
    elo_mean_bin = binned['elo_bin'].to_numpy()

    idx_to_name = dict(enumerate(RESULT_CLASS_NAMES))
    true_class = binned[mover_result_col].map(RESULT_TO_CLASS).map(idx_to_name)

    colors = colors_for(names)
    fig, ax = plt.subplots(figsize=figsize or BASE_FIGSIZE)

    for name in names:
        pred_class = binned[f'{name}_predicted_class']
        acc_by_bin = []
        for b in range(n_elo_bins):
            mask = elo_mean_bin == b
            acc_by_bin.append((pred_class[mask] == true_class[mask]).mean() if mask.sum() else float('nan'))

        ax.plot(labels, [a * 100 for a in acc_by_bin], marker='D', markersize=6,
                markeredgecolor='white', markeredgewidth=0.6, linewidth=2, color=colors[name],
                label=resolve_display_name(name, display_names))

    ax.set_xlabel('Elo bin (mean)')
    ax.set_ylabel('Accuracy (%)')
    ax.set_title(title)
    ax.legend()
    ax.set_ylim(*ylim)
    plt.xticks(rotation=45)

    if show_hist:
        counts = np.bincount(elo_mean_bin, minlength=n_elo_bins)
        ax2 = ax.twinx()
        ax2.bar(labels, counts, color='grey', alpha=0.3, zorder=1)
        ax2.set_ylabel('Number of positions')
        ax2.set_ylim(0, counts.max() * 1.2)
        ax.set_zorder(ax2.get_zorder() + 1)
        ax.patch.set_visible(False)

    plt.tight_layout()
    plt.show()


def plot_accuracy_by_ply(df: pd.DataFrame, names: list[str],
                          ply_played_col: str = 'ply_played',
                          mover_result_col: str = 'mover_result',
                          group_size: int = 20, max_ply: int = 200,
                          ylim: tuple[float, float] = (0, 80),
                          title: str = 'Result-prediction accuracy by plies played',
                          show_hist: bool = False,
                          figsize: tuple[float, float] | None = None,
                          display_names: dict[str, str] | None = None) -> None:
    """Plots per-model accuracy against true mover_result, binned by plies played."""
    cols_needed = [f'{name}_predicted_class' for name in names] + [mover_result_col, ply_played_col]
    df = restrict_to_common_rows(df, cols_needed)
    df = df[df[ply_played_col] <= max_ply]

    ply_group = (df[ply_played_col] // group_size) * group_size + group_size // 2
    class_to_val = {'loss': 0.0, 'draw': 0.5, 'win': 1.0}

    colors = colors_for(names)
    fig, ax = plt.subplots(figsize=figsize or BASE_FIGSIZE)

    for name in names:
        pred_val = df[f'{name}_predicted_class'].map(class_to_val)
        correct = pred_val == df[mover_result_col]
        acc_by_group = correct.groupby(ply_group).mean().sort_index()

        ax.plot(acc_by_group.index, acc_by_group.values * 100, marker='D', markersize=6,
                markeredgecolor='white', markeredgewidth=0.6, linewidth=2, color=colors[name],
                label=resolve_display_name(name, display_names))

    ax.set_xlabel(f'Plies played (bins of {group_size})')
    ax.set_ylabel('Accuracy (%)')
    ax.set_title(title)
    ax.legend()
    ax.set_ylim(*ylim)

    if show_hist:
        counts = ply_group.value_counts().sort_index()
        ax2 = ax.twinx()
        ax2.bar(counts.index, counts.values, width=group_size * 0.9, color='grey', alpha=0.3, zorder=1)
        ax2.set_ylabel('Number of positions')
        ax2.set_ylim(0, counts.max() * 1.2)
        ax.set_zorder(ax2.get_zorder() + 1)
        ax.patch.set_visible(False)

    plt.tight_layout()
    plt.show()


def plot_accuracy_by_combined_clock(df: pd.DataFrame, names: list[str],
                                     mover_clock_col: str = 'mover_clock',
                                     opponent_clock_col: str = 'opponent_clock',
                                     time_control_col: str = 'time_control',
                                     mover_result_col: str = 'mover_result',
                                     bin_width: float = 0.05,
                                     ylim: tuple[float, float] = (50, 80),
                                     title: str = 'Result-prediction accuracy by time remaining',
                                     show_hist: bool = False,
                                     figsize: tuple[float, float] | None = None,
                                     display_names: dict[str, str] | None = None) -> None:
    """Plots per-model accuracy against true mover_result, binned by combined proportion of clock time remaining."""
    cols_needed = ([f'{name}_predicted_class' for name in names]
                   + [mover_result_col, mover_clock_col, opponent_clock_col, time_control_col])
    df = restrict_to_common_rows(df, cols_needed)

    game_time = df[time_control_col].map(SEC_MAPPING)

    n_before = len(df)
    df = df[game_time.notna()]
    game_time = game_time[game_time.notna()]
    if len(df) != n_before:
        print(f'plot_accuracy_by_combined_clock: dropped {n_before - len(df):,} rows with unmapped {time_control_col}.')

    time_left_prop = ((df[mover_clock_col] + df[opponent_clock_col]) / (2 * game_time)).clip(0, 1)
    time_bin = ((time_left_prop // bin_width) * bin_width + bin_width / 2).round(4)

    class_to_val = {'loss': 0.0, 'draw': 0.5, 'win': 1.0}

    colors = colors_for(names)
    fig, ax = plt.subplots(figsize=figsize or BASE_FIGSIZE)

    for name in names:
        pred_val = df[f'{name}_predicted_class'].map(class_to_val)
        correct = pred_val == df[mover_result_col]
        acc_by_bin = correct.groupby(time_bin).mean().sort_index()

        ax.plot(acc_by_bin.index, acc_by_bin.values * 100, marker='D', markersize=6,
                markeredgecolor='white', markeredgewidth=0.6, linewidth=2, color=colors[name],
                label=resolve_display_name(name, display_names))

    ax.set_xlabel('Proportion of total time remaining (both players summed)')
    ax.set_ylabel('Accuracy (%)')
    ax.set_title(title)
    ax.legend()
    ax.set_ylim(*ylim)
    ax.invert_xaxis()

    if show_hist:
        counts = time_bin.value_counts().sort_index()
        ax2 = ax.twinx()
        ax2.bar(counts.index, counts.values, width=bin_width * 0.9, color='grey', alpha=0.3, zorder=1)
        ax2.set_ylabel('Number of positions')
        ax2.set_ylim(0, counts.max() * 1.2)
        ax.set_zorder(ax2.get_zorder() + 1)
        ax.patch.set_visible(False)

    plt.tight_layout()
    plt.show()


def plot_roc_auc(df: pd.DataFrame, names: list[str],
                  mover_result_col: str = 'mover_result',
                  positive_class: str | None = None,
                  ylim: tuple[float, float] = (0, 1),
                  title: str | None = None,
                  figsize: tuple[float, float] | None = None,
                  display_names: dict[str, str] | None = None) -> None:
    """Plots ROC curves (one panel per class, or just positive_class if given) across models."""
    class_to_result = {'win': 1.0, 'loss': 0.0, 'draw': 0.5}
    if positive_class is not None and positive_class not in class_to_result:
        raise ValueError(f'positive_class must be one of {list(class_to_result)}, got {positive_class!r}')

    classes_to_plot = [positive_class] if positive_class is not None else ['win', 'draw', 'loss']

    cols_needed = [f'{name}_prob_{cls}' for name in names for cls in classes_to_plot] + [mover_result_col]
    df = restrict_to_common_rows(df, cols_needed)

    if positive_class is None:
        draw_cols = [f'{name}_prob_draw' for name in names]
        if (df[draw_cols] == 0).all(axis=None):
            classes_to_plot = ['win', 'loss']
            print('plot_roc_auc: all draw probabilities are 0 across every model -- skipping draw panel.')

    panel_width, panel_height = figsize or BASE_FIGSIZE
    colors = colors_for(names)
    fig, axes = plt.subplots(1, len(classes_to_plot), figsize=(panel_width * len(classes_to_plot), panel_height),
                              squeeze=False)
    axes = axes[0]
    if title:
        fig.suptitle(title)

    for ax, cls in zip(axes, classes_to_plot):
        y_true = (df[mover_result_col] == class_to_result[cls]).astype(int)

        for name in names:
            y_score = df[f'{name}_prob_{cls}']
            fpr, tpr, _ = roc_curve(y_true, y_score)
            roc_auc = auc(fpr, tpr)
            label = f'{resolve_display_name(name, display_names)} (AUC = {roc_auc:.3f})'
            ax.plot(fpr, tpr, color=colors[name], linewidth=2, label=label)

        ax.plot([0, 1], [0, 1], color='grey', linestyle='--', linewidth=1, label='Chance')
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title(f'{cls.capitalize()} prediction')
        ax.legend(loc='lower right')
        ax.set_xlim(0, 1)
        ax.set_ylim(*ylim)

    plt.tight_layout()
    plt.show()