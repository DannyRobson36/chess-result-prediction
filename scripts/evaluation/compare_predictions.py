"""
compare_predictions.py
Reads per-model prediction csvs against a fixed val/test positions df, merges them, and provides eval-plots across models.

Latest changes: 24/08/26:
- Added Brier score, confidence, termination split, agreement-by-phase/clock, past performance,
  rematch, new-player etc.
"""

import glob
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

from scripts.utils.utils_chess import (
    EloBinConfig, ELO_BINS, elo_bin_by_mover, elo_bin_labels, RESULT_TO_CLASS, RESULT_CLASS_NAMES,
    total_material, material_diff, game_phase, phase_label, title_track, title_strength,
)
from scripts.utils.utils_eval import resolve_display_name
from scripts.utils.utils_plotting import colors_for, BASE_FIGSIZE
from scripts.features.features import SEC_MAPPING, INC_FLAG_MAPPING

####################
# CONSTANTS
####################

REQUIRED_PRED_COLS = ['game_id', 'fen', 'prob_win', 'prob_draw', 'prob_loss', 'predicted_class']
PHASE_LABEL_ORDER = ['opening', 'middlegame', 'endgame']
CLASS_TO_VAL = {'loss': 0.0, 'draw': 0.5, 'win': 1.0}

# Only these termination values are analysed; everything else (Abandoned, Rules infraction,
# Insufficient material, etc.) is excluded wherever termination is used.
TERMINATIONS_KEPT = ['Normal', 'Time forfeit']

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


def add_material_and_phase_cols(df: pd.DataFrame, fen_col: str = 'fen',
                                 material_col: str = 'material', phase_col: str = 'phase') -> pd.DataFrame:
    """Adds material_col (total board material) and phase_col (0-256 game phase score), computed once per fen."""
    df = df.copy()
    df[material_col] = df[fen_col].apply(total_material)
    df[phase_col] = df[fen_col].apply(game_phase)
    print(f'add_material_and_phase_cols: computed {material_col!r} and {phase_col!r} for {len(df):,} rows.')
    return df


# (b) ROW FILTERING

def restrict_to_common_rows(df: pd.DataFrame, cols: list[str], verbose: bool = True) -> pd.DataFrame:
    """Returns df restricted to rows with no NaN in any of cols."""
    mask = df[cols].notna().all(axis=1)
    if verbose and (~mask).any():
        print(f'restrict_to_common_rows: dropping {(~mask).sum():,} of {len(df):,} rows '
              f'-- NaN in one or more of {cols}.')
    return df[mask].copy()


# (c) ELO-BIN / TERMINATION SPLITTING

def _validate_elo_bins(elo_bins) -> tuple[int, int]:
    """Validates elo_bins as exactly two integers, without checking against the available bin count yet."""
    if not isinstance(elo_bins, (tuple, list)):
        raise ValueError(f'elo_bins must be a tuple or list of 2 integers, got {elo_bins!r}.')
    elo_bins = tuple(elo_bins)
    if len(elo_bins) != 2:
        raise ValueError(f'elo_bins must contain exactly 2 bin numbers, got {len(elo_bins)}.')
    for b in elo_bins:
        if not isinstance(b, int) or isinstance(b, bool):
            raise ValueError(f'elo_bins values must be integers, got {b!r}.')
    return elo_bins


def _split_by_elo_bin(df: pd.DataFrame, elo_bins: tuple[int, int],
                       cfg: EloBinConfig) -> tuple[list[pd.DataFrame], list[str]]:
    """Splits df into two subsets by mover mean-elo bin membership, returns (subsets, panel_labels)."""
    b1, b2 = _validate_elo_bins(elo_bins)
    binned, edges = elo_bin_by_mover(df, cfg, method='mean')
    n_available_bins = len(edges) - 1
    for b in (b1, b2):
        if not (1 <= b <= n_available_bins):
            raise ValueError(f'elo_bins values must be in [1, {n_available_bins}], got {b}.')

    labels = elo_bin_labels(edges)
    subsets, panel_labels = [], []
    for b in (b1, b2):
        mask = binned['elo_bin'].to_numpy() == (b - 1)
        subset = binned[mask].copy()
        if len(subset) == 0:
            print(f'_split_by_elo_bin: elo bin {b} has 0 rows in the given data.')
        subsets.append(subset)
        panel_labels.append(f'elo bin {b} ({labels[b - 1]})')
    return subsets, panel_labels


def _split_by_termination(df: pd.DataFrame, termination_col: str) -> tuple[list[pd.DataFrame], list[str]]:
    """Splits df into Normal and Time forfeit subsets, returns (subsets, panel_labels). Other termination values are excluded."""
    n_before = len(df)
    df = df[df[termination_col].isin(TERMINATIONS_KEPT)]
    n_excluded = n_before - len(df)
    if n_excluded:
        print(f'_split_by_termination: excluded {n_excluded:,} of {n_before:,} rows '
              f'({n_excluded / n_before * 100:.1f}%) -- termination not in {TERMINATIONS_KEPT}.')
    subsets = [df[df[termination_col] == term].copy() for term in TERMINATIONS_KEPT]
    panel_labels = list(TERMINATIONS_KEPT)
    return subsets, panel_labels


def _count_active_row_splits(*flags: bool) -> None:
    """Raises if more than one row-split flag is active."""
    if sum(bool(f) for f in flags) > 1:
        raise ValueError('elo_bins, split_by_increment, and split_by_termination are mutually exclusive '
                          '-- choose at most one row split.')


# (d) SHARED PLOT HELPERS

def _true_class_series(df: pd.DataFrame, mover_result_col: str) -> pd.Series:
    """Maps mover_result_col's continuous 0/0.5/1 values to RESULT_CLASS_NAMES strings."""
    idx_to_name = dict(enumerate(RESULT_CLASS_NAMES))
    return df[mover_result_col].map(RESULT_TO_CLASS).map(idx_to_name)


def _agreement_mask(df: pd.DataFrame, name: str, baseline: str) -> pd.Series:
    """Returns a boolean Series, True where name's predicted_class matches baseline's predicted_class."""
    return df[f'{name}_predicted_class'] == df[f'{baseline}_predicted_class']


def _brier_scores(df: pd.DataFrame, name: str, mover_result_col: str) -> pd.Series:
    """Returns each row's 3-class Brier score (sum of squared prob-vs-onehot errors) for name's predictions."""
    true_class = _true_class_series(df, mover_result_col)
    probs = df[[f'{name}_prob_win', f'{name}_prob_draw', f'{name}_prob_loss']].to_numpy(dtype='float64')
    onehot = np.column_stack([(true_class == c).to_numpy() for c in ('win', 'draw', 'loss')]).astype('float64')
    return pd.Series(((probs - onehot) ** 2).sum(axis=1), index=df.index)


def _binned_line_plot(ax: plt.Axes, df: pd.DataFrame, names: list[str], mover_result_col: str | None,
                       bin_series: pd.Series, bin_values: list, x_labels: list,
                       colors: dict[str, str], panel_title: str, display_names: dict[str, str] | None,
                       metric: str = 'accuracy', ylim: tuple[float, float] | None = None) -> None:
    """Draws one line panel (one line per model) of metric ('accuracy', 'brier', or 'confidence') across bin_values of bin_series onto ax. mover_result_col is unused (may be None) for metric='confidence'."""
    for name in names:
        if metric == 'accuracy':
            pred_val = df[f'{name}_predicted_class'].map(CLASS_TO_VAL)
            correct = pred_val == df[mover_result_col]
            y = [correct[bin_series == b].mean() * 100 if (bin_series == b).sum() else float('nan') for b in bin_values]
        elif metric == 'brier':
            brier = _brier_scores(df, name, mover_result_col)
            y = [brier[bin_series == b].mean() if (bin_series == b).sum() else float('nan') for b in bin_values]
        elif metric == 'confidence':
            conf = df[[f'{name}_prob_win', f'{name}_prob_draw', f'{name}_prob_loss']].max(axis=1)
            y = [conf[bin_series == b].mean() if (bin_series == b).sum() else float('nan') for b in bin_values]
        else:
            raise ValueError(f"metric must be 'accuracy', 'brier', or 'confidence', got {metric!r}")

        ax.plot(x_labels, y, marker='D', markersize=6, markeredgecolor='white',
                markeredgewidth=0.6, linewidth=2, color=colors[name],
                label=resolve_display_name(name, display_names))

    ylabels = {'accuracy': 'Accuracy (%)', 'brier': 'Brier score (lower is better)',
               'confidence': 'Mean max predicted probability'}
    ax.set_ylabel(ylabels[metric])
    ax.set_title(panel_title)
    ax.legend()
    if metric == 'accuracy' and ylim is not None:
        ax.set_ylim(*ylim)


def _add_hist_twin(ax: plt.Axes, bin_series: pd.Series, bin_values: list, x_labels: list, bar_width: float) -> None:
    """Adds a secondary-axis histogram of row counts per bin_value behind ax."""
    counts = [(bin_series == b).sum() for b in bin_values]
    ax2 = ax.twinx()
    ax2.bar(x_labels, counts, width=bar_width, color='grey', alpha=0.3, zorder=1)
    ax2.set_ylabel('Number of positions')
    if counts and max(counts) > 0:
        ax2.set_ylim(0, max(counts) * 1.2)
    ax.set_zorder(ax2.get_zorder() + 1)
    ax.patch.set_visible(False)


# (e) EARLY DIAGNOSTICS

def plot_accuracy_by_class(df: pd.DataFrame, names: list[str],
                            mover_result_col: str = 'mover_result',
                            ylim: tuple[float, float] = (0, 100),
                            title: str = 'Result-prediction accuracy by true class',
                            figsize: tuple[float, float] | None = None,
                            display_names: dict[str, str] | None = None) -> None:
    """Plots each model's accuracy (recall) per true result class, dropping the draw class if no draws are present in mover_result_col."""
    cols_needed = [f'{name}_predicted_class' for name in names] + [mover_result_col]
    df = restrict_to_common_rows(df, cols_needed)

    true_class = _true_class_series(df, mover_result_col)
    classes_present = [c for c in RESULT_CLASS_NAMES if (true_class == c).any()]
    if 'draw' not in classes_present:
        print('plot_accuracy_by_class: no draws present in mover_result_col -- dropping the draw class.')

    colors = colors_for(names)
    x = np.arange(len(classes_present))
    width = 0.8 / len(names)

    fig, ax = plt.subplots(figsize=figsize or BASE_FIGSIZE)
    for i, name in enumerate(names):
        pred_class = df[f'{name}_predicted_class']
        recalls = []
        for cls in classes_present:
            mask = true_class == cls
            recalls.append((pred_class[mask] == cls).mean() * 100 if mask.sum() else float('nan'))
        ax.bar(x + i * width - 0.4 + width / 2, recalls, width, color=colors[name],
               label=resolve_display_name(name, display_names))

    ax.set_xticks(x)
    ax.set_xticklabels([c.capitalize() for c in classes_present])
    ax.set_ylabel('Accuracy / recall (%)')
    ax.set_title(title)
    ax.legend()
    ax.set_ylim(*ylim)

    plt.tight_layout()
    plt.show()


def plot_accuracy_by_termination(df: pd.DataFrame, names: list[str],
                                  termination_col: str = 'termination',
                                  mover_result_col: str = 'mover_result',
                                  ylim: tuple[float, float] = (0, 100),
                                  title: str = 'Result-prediction accuracy by termination type',
                                  figsize: tuple[float, float] | None = None,
                                  display_names: dict[str, str] | None = None) -> None:
    """Plots each model's accuracy against true mover_result, grouped by termination type (Normal vs Time forfeit only)."""
    cols_needed = [f'{name}_predicted_class' for name in names] + [mover_result_col, termination_col]
    df = restrict_to_common_rows(df, cols_needed)

    n_before = len(df)
    df = df[df[termination_col].isin(TERMINATIONS_KEPT)]
    n_excluded = n_before - len(df)
    if n_excluded:
        print(f'plot_accuracy_by_termination: excluded {n_excluded:,} of {n_before:,} rows '
              f'({n_excluded / n_before * 100:.1f}%) -- termination not in {TERMINATIONS_KEPT}.')

    true_class = _true_class_series(df, mover_result_col)
    colors = colors_for(names)
    x = np.arange(len(TERMINATIONS_KEPT))
    width = 0.8 / len(names)

    fig, ax = plt.subplots(figsize=figsize or BASE_FIGSIZE)
    for i, name in enumerate(names):
        pred_class = df[f'{name}_predicted_class']
        correct = pred_class == true_class
        accs = []
        for term in TERMINATIONS_KEPT:
            mask = df[termination_col] == term
            accs.append(correct[mask].mean() * 100 if mask.sum() else float('nan'))
        ax.bar(x + i * width - 0.4 + width / 2, accs, width, color=colors[name],
               label=resolve_display_name(name, display_names))

    ax.set_xticks(x)
    ax.set_xticklabels(TERMINATIONS_KEPT)
    ax.set_ylabel('Accuracy (%)')
    ax.set_title(title)
    ax.legend()
    ax.set_ylim(*ylim)

    plt.tight_layout()
    plt.show()