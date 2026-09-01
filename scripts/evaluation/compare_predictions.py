"""
compare_predictions.py
Reads per-model prediction csvs against a fixed val/test positions df, merges them, and provides eval-plots across models.

Latest changes: 01/09/26:
- Selective prediction loading, default display names/colours, colour/linestyle overrides
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
from scripts.utils.utils_plotting import colors_for, linestyles_for, BASE_FIGSIZE
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

# Canonical model name -> display label, in the fixed order used for both legend text and colour
# assignment (see _resolve_plot_style). A name absent from a given plot's df is harmless here; it's
# only looked up if that name also appears in the names list passed to a plot call.
DEFAULT_DISPLAY_NAMES: dict[str, str] = {
    'maia2_value_replica': 'Maia2-Value (Replica)',
    # TODO: confirm and add the remaining arch_name keys from model_arch.py's ARCH_REGISTRY
    # (Maia2ValueBoard / Feature / PureTransformer) -- not guessed here, snake_case unconfirmed.
    'stockfish_d1': 'Stockfish',
    'maia2': 'Maia2',
}

####################
# FUNCTIONS
####################

# (a) DISCOVERY & LOADING

def discover_predictions(predictions_dir: str, split: str, names: list[str] | None = None) -> dict[str, str]:
    """Finds {split}_predictions_{name}.csv in predictions_dir for each of names, or every matching file
    if names is None, returns {name: path}, sorted by name. A requested name with no matching file is
    skipped, not raised."""
    prefix = f'{split}_predictions_'

    if names is None:
        pattern = os.path.join(predictions_dir, f'{prefix}*.csv')
        discovered = {}
        for path in sorted(glob.glob(pattern)):
            fname = os.path.basename(path)
            name = fname[len(prefix):-len('.csv')]
            discovered[name] = path
        return discovered

    discovered = {}
    for name in names:
        path = os.path.join(predictions_dir, f'{prefix}{name}.csv')
        if os.path.exists(path):
            discovered[name] = path
        else:
            print(f"discover_predictions: no '{prefix}{name}.csv' found in {predictions_dir} -- skipping.")
    return dict(sorted(discovered.items()))


def present_model_names(df: pd.DataFrame) -> list[str]:
    """Returns every model name already merged into df, detected via its {name}_predicted_class column."""
    suffix = '_predicted_class'
    return sorted(c[:-len(suffix)] for c in df.columns if c.endswith(suffix))


def _check_unique_keys(df: pd.DataFrame, label: str, game_id_col: str, fen_col: str) -> None:
    """Raises if df has duplicate (game_id_col, fen_col) combinations."""
    dupes = df.duplicated(subset=[game_id_col, fen_col])
    if dupes.any():
        raise ValueError(
            f'{label}: found {dupes.sum():,} duplicate ({game_id_col}, {fen_col}) combos '
            f'-- keys must be unique.'
        )


def combine_predictions(main_df: pd.DataFrame, predictions_dir: str, split: str,
                         names: list[str] | None = None, game_id_col: str = 'game_id',
                         fen_col: str = 'fen') -> tuple[pd.DataFrame, list[str]]:
    """Merges each requested model's predictions csv in predictions_dir onto main_df (every matching
    file if names is None), skipping any name already present in main_df. Returns (df_combined,
    loaded_names), where loaded_names is every model now present in df_combined, old and new."""
    if split not in ('val', 'test'):
        raise ValueError(f"split must be 'val' or 'test', got {split!r}")

    _check_unique_keys(main_df, 'main_df', game_id_col, fen_col)

    already_present = set(present_model_names(main_df))
    discovered = discover_predictions(predictions_dir, split, names=names)

    skipped_existing = [name for name in discovered if name in already_present]
    for name in skipped_existing:
        print(f"combine_predictions: '{name}' already merged into main_df -- skipping.")

    to_load = {name: path for name, path in discovered.items() if name not in already_present}

    if not to_load and not already_present:
        raise FileNotFoundError(
            f"No '{split}_predictions_*.csv' files found in {predictions_dir}"
            + (f' for names={names}' if names is not None else '')
        )

    df_combined = main_df.copy()
    newly_loaded = []

    for name, path in to_load.items():
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

        newly_loaded.append(name)

    loaded_names = present_model_names(df_combined)
    print(f'combine_predictions: merged {len(newly_loaded)} new model(s) -- {newly_loaded}; '
          f'{len(loaded_names)} total in combined df -- {loaded_names}')

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

def _resolve_plot_style(names: list[str], display_names: dict[str, str] | None,
                         color_overrides: list[str | None] | None = None,
                         linestyle_overrides: list[str | None] | None = None
                         ) -> tuple[dict[str, str], dict[str, str], dict[str, str]]:
    """Resolves display_names (falling back to DEFAULT_DISPLAY_NAMES), and per-name colours/linestyles
    for names: colours are fixed by each name's position in DEFAULT_DISPLAY_NAMES, stable across plots
    and subsets, linestyles default to solid. Both are overridable per name, by position in names."""
    display_names = DEFAULT_DISPLAY_NAMES if display_names is None else display_names
    colors = colors_for(names, canonical_order=list(DEFAULT_DISPLAY_NAMES), overrides=color_overrides)
    linestyles = linestyles_for(names, overrides=linestyle_overrides)
    return display_names, colors, linestyles


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
                       colors: dict[str, str], linestyles: dict[str, str], panel_title: str,
                       display_names: dict[str, str] | None,
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
                markeredgewidth=0.6, linewidth=2, linestyle=linestyles[name], color=colors[name],
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
                            display_names: dict[str, str] | None = None,
                            color_overrides: list[str | None] | None = None) -> None:
    """Plots each model's accuracy (recall) per true result class, dropping the draw class if no draws are present in mover_result_col."""
    cols_needed = [f'{name}_predicted_class' for name in names] + [mover_result_col]
    df = restrict_to_common_rows(df, cols_needed)

    true_class = _true_class_series(df, mover_result_col)
    classes_present = [c for c in RESULT_CLASS_NAMES if (true_class == c).any()]
    if 'draw' not in classes_present:
        print('plot_accuracy_by_class: no draws present in mover_result_col -- dropping the draw class.')

    display_names, colors, _ = _resolve_plot_style(names, display_names, color_overrides)
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
                                  display_names: dict[str, str] | None = None,
                                  color_overrides: list[str | None] | None = None) -> None:
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
    display_names, colors, _ = _resolve_plot_style(names, display_names, color_overrides)
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

# (f) ACCURACY BY ELO BIN

def plot_accuracy_by_elo_bin(df: pd.DataFrame, names: list[str],
                              mover_result_col: str = 'mover_result',
                              cfg: EloBinConfig = ELO_BINS,
                              ylim: tuple[float, float] = (50, 70),
                              title: str = 'Result prediction accuracy by Elo bin',
                              show_hist: bool = False,
                              show_brier: bool = False,
                              figsize: tuple[float, float] | None = None,
                              display_names: dict[str, str] | None = None,
                              color_overrides: list[str | None] | None = None,
                              linestyle_overrides: list[str | None] | None = None) -> None:
    """Plots per-model accuracy (and optionally Brier score) against true mover_result, binned by mean elo."""
    cols_needed = [f'{name}_predicted_class' for name in names] + [mover_result_col, 'mover_elo', 'opponent_elo']
    if show_brier:
        cols_needed += [f'{name}_prob_{c}' for name in names for c in ('win', 'draw', 'loss')]
    df = restrict_to_common_rows(df, cols_needed)

    binned, edges = elo_bin_by_mover(df, cfg, method='mean')
    labels = elo_bin_labels(edges)
    n_elo_bins = len(edges) - 1
    elo_mean_bin = binned['elo_bin']
    bin_values = list(range(n_elo_bins))

    display_names, colors, linestyles = _resolve_plot_style(names, display_names, color_overrides, linestyle_overrides)
    n_cols = 2 if show_brier else 1
    panel_width, panel_height = figsize or BASE_FIGSIZE
    fig, axes = plt.subplots(1, n_cols, figsize=(panel_width * n_cols, panel_height), squeeze=False)

    ax_acc = axes[0][0]
    _binned_line_plot(ax_acc, binned, names, mover_result_col, elo_mean_bin, bin_values, labels,
                       colors, linestyles, title, display_names, metric='accuracy', ylim=ylim)
    ax_acc.set_xlabel('Elo bin (mean)')
    ax_acc.tick_params(axis='x', rotation=45)

    if show_hist:
        _add_hist_twin(ax_acc, elo_mean_bin, bin_values, labels, bar_width=0.9)

    if show_brier:
        ax_brier = axes[0][1]
        _binned_line_plot(ax_brier, binned, names, mover_result_col, elo_mean_bin, bin_values, labels,
                           colors, linestyles, f'{title} (Brier)', display_names, metric='brier')
        ax_brier.set_xlabel('Elo bin (mean)')
        ax_brier.tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.show()


def plot_confidence_by_elo_bin(df: pd.DataFrame, names: list[str],
                                mover_result_col: str = 'mover_result',
                                cfg: EloBinConfig = ELO_BINS,
                                title: str = 'Mean predicted confidence by Elo bin',
                                figsize: tuple[float, float] | None = None,
                                display_names: dict[str, str] | None = None,
                                color_overrides: list[str | None] | None = None,
                                linestyle_overrides: list[str | None] | None = None) -> None:
    """Plots each model's mean max predicted probability, binned by mean elo."""
    cols_needed = ([f'{name}_prob_{c}' for name in names for c in ('win', 'draw', 'loss')]
                   + [mover_result_col, 'mover_elo', 'opponent_elo'])
    df = restrict_to_common_rows(df, cols_needed)

    binned, edges = elo_bin_by_mover(df, cfg, method='mean')
    labels = elo_bin_labels(edges)
    bin_values = list(range(len(edges) - 1))

    display_names, colors, linestyles = _resolve_plot_style(names, display_names, color_overrides, linestyle_overrides)
    fig, ax = plt.subplots(figsize=figsize or BASE_FIGSIZE)
    _binned_line_plot(ax, binned, names, mover_result_col, binned['elo_bin'], bin_values, labels,
                       colors, linestyles, title, display_names, metric='confidence')
    ax.set_xlabel('Elo bin (mean)')
    ax.tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.show()


# (g) NEW PLAYER / REMATCH

def plot_accuracy_new_player_comparison(df: pd.DataFrame, names: list[str],
                                         mover_elo_col: str = 'mover_elo',
                                         has_history_mover_col: str = 'has_history_mover',
                                         mover_result_col: str = 'mover_result',
                                         new_player_elo: int = 1500,
                                         ylim: tuple[float, float] = (0, 100),
                                         title: str = 'Accuracy: new-account movers vs everyone else',
                                         figsize: tuple[float, float] | None = None,
                                         display_names: dict[str, str] | None = None) -> None:
    """Plots each model's accuracy split into 'mover is a new account' (no history, elo exactly new_player_elo) vs everyone else."""
    cols_needed = ([f'{name}_predicted_class' for name in names]
                   + [mover_elo_col, has_history_mover_col, mover_result_col])
    df = restrict_to_common_rows(df, cols_needed)
    display_names = DEFAULT_DISPLAY_NAMES if display_names is None else display_names

    is_new = (~df[has_history_mover_col].astype(bool)) & (df[mover_elo_col] == new_player_elo)
    print(f'plot_accuracy_new_player_comparison: {int(is_new.sum()):,} of {len(df):,} rows '
          f'({is_new.mean() * 100:.2f}%) flagged as new-account movers.')

    true_class = _true_class_series(df, mover_result_col)

    new_acc, rest_acc = [], []
    for name in names:
        correct = df[f'{name}_predicted_class'] == true_class
        new_acc.append(correct[is_new].mean() * 100 if is_new.sum() else float('nan'))
        rest_acc.append(correct[~is_new].mean() * 100 if (~is_new).sum() else float('nan'))

    bar_labels = [resolve_display_name(name, display_names) for name in names]
    x = np.arange(len(names))
    width = 0.35

    fig, ax = plt.subplots(figsize=figsize or BASE_FIGSIZE)
    ax.bar(x - width / 2, rest_acc, width, color='tab:blue', label='Everyone else')
    ax.bar(x + width / 2, new_acc, width, color='tab:orange', label='New-account mover')

    ax.set_xticks(x)
    ax.set_xticklabels(bar_labels)
    ax.set_ylabel('Accuracy (%)')
    ax.set_title(title)
    ax.legend()
    ax.set_ylim(*ylim)

    plt.tight_layout()
    plt.show()


def plot_accuracy_by_elo_bin_rematch(df: pd.DataFrame, names: list[str],
                                      rematch_col: str = 'rematch',
                                      mover_result_col: str = 'mover_result',
                                      cfg: EloBinConfig = ELO_BINS,
                                      ylim: tuple[float, float] = (0, 100),
                                      title: str = 'Accuracy by Elo bin, rematch vs non-rematch',
                                      figsize: tuple[float, float] | None = None,
                                      display_names: dict[str, str] | None = None,
                                      color_overrides: list[str | None] | None = None) -> None:
    """Plots per-model accuracy against true mover_result, binned by mean elo, split into rematch vs non-rematch lines. Linestyle already encodes rematch/non-rematch here, so only colour is overridable."""
    cols_needed = ([f'{name}_predicted_class' for name in names]
                   + [mover_result_col, 'mover_elo', 'opponent_elo', rematch_col])
    df = restrict_to_common_rows(df, cols_needed)

    binned, edges = elo_bin_by_mover(df, cfg, method='mean')
    labels = elo_bin_labels(edges)
    n_elo_bins = len(edges) - 1
    elo_mean_bin = binned['elo_bin'].to_numpy()
    true_class = _true_class_series(binned, mover_result_col)
    is_rematch = binned[rematch_col].astype(bool).to_numpy()

    display_names, colors, _ = _resolve_plot_style(names, display_names, color_overrides)
    fig, ax = plt.subplots(figsize=figsize or BASE_FIGSIZE)

    for name in names:
        correct = (binned[f'{name}_predicted_class'] == true_class).to_numpy()
        for condition_mask, condition_label, linestyle in [(is_rematch, 'rematch', '-'), (~is_rematch, 'non-rematch', '--')]:
            acc_by_bin = []
            for b in range(n_elo_bins):
                mask = (elo_mean_bin == b) & condition_mask
                acc_by_bin.append(correct[mask].mean() * 100 if mask.sum() else float('nan'))
            ax.plot(labels, acc_by_bin, marker='D', markersize=6, markeredgecolor='white', markeredgewidth=0.6,
                    linewidth=2, linestyle=linestyle, color=colors[name],
                    label=f'{resolve_display_name(name, display_names)} ({condition_label})')

    ax.set_xlabel('Elo bin (mean)')
    ax.set_ylabel('Accuracy (%)')
    ax.set_title(title)
    ax.legend()
    ax.set_ylim(*ylim)
    plt.xticks(rotation=45)

    plt.tight_layout()
    plt.show()


# (h) ACCURACY BY PLY

def plot_accuracy_by_ply(df: pd.DataFrame, names: list[str],
                          ply_played_col: str = 'ply_played',
                          mover_result_col: str = 'mover_result',
                          group_size: int = 20, max_ply: int = 200,
                          ylim: tuple[float, float] = (0, 80),
                          title: str = 'Result-prediction accuracy by plies played',
                          show_hist: bool = False,
                          show_brier: bool = False,
                          figsize: tuple[float, float] | None = None,
                          display_names: dict[str, str] | None = None,
                          elo_bins: tuple[int, int] | None = None,
                          elo_bin_cfg: EloBinConfig = ELO_BINS,
                          color_overrides: list[str | None] | None = None,
                          linestyle_overrides: list[str | None] | None = None) -> None:
    """Plots per-model accuracy (and optionally Brier score) against true mover_result, binned by plies played, as one row or two elo-bin-restricted rows."""
    cols_needed = [f'{name}_predicted_class' for name in names] + [mover_result_col, ply_played_col]
    if show_brier:
        cols_needed += [f'{name}_prob_{c}' for name in names for c in ('win', 'draw', 'loss')]
    if elo_bins is not None:
        cols_needed += ['mover_elo', 'opponent_elo']
    df = restrict_to_common_rows(df, cols_needed)
    df = df[df[ply_played_col] <= max_ply].copy()
    df['_ply_group'] = (df[ply_played_col] // group_size) * group_size + group_size // 2

    display_names, colors, linestyles = _resolve_plot_style(names, display_names, color_overrides, linestyle_overrides)
    panel_width, panel_height = figsize or BASE_FIGSIZE

    subsets, row_labels = ([df], [None]) if elo_bins is None else _split_by_elo_bin(df, elo_bins, elo_bin_cfg)
    n_rows, n_cols = len(subsets), (2 if show_brier else 1)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(panel_width * n_cols, panel_height * n_rows), squeeze=False)

    for row_idx, (sub_df, row_label) in enumerate(zip(subsets, row_labels)):
        bin_series = sub_df['_ply_group']
        bin_values = sorted(bin_series.dropna().unique())
        row_title = title if row_label is None else f'{title} -- {row_label}'

        ax_acc = axes[row_idx][0]
        _binned_line_plot(ax_acc, sub_df, names, mover_result_col, bin_series, bin_values, bin_values,
                           colors, linestyles, row_title, display_names, metric='accuracy', ylim=ylim)
        ax_acc.set_xlabel(f'Plies played (bins of {group_size})')
        if show_hist:
            _add_hist_twin(ax_acc, bin_series, bin_values, bin_values, bar_width=group_size * 0.9)

        if show_brier:
            ax_brier = axes[row_idx][1]
            _binned_line_plot(ax_brier, sub_df, names, mover_result_col, bin_series, bin_values, bin_values,
                               colors, linestyles, f'{row_title} (Brier)', display_names, metric='brier')
            ax_brier.set_xlabel(f'Plies played (bins of {group_size})')

    plt.tight_layout()
    plt.show()


def plot_accuracy_by_ply_proportion(df: pd.DataFrame, names: list[str],
                                     ply_played_col: str = 'ply_played',
                                     ply_count_col: str = 'ply_count',
                                     mover_result_col: str = 'mover_result',
                                     bin_width: float = 0.05,
                                     ylim: tuple[float, float] = (0, 80),
                                     title: str = 'Result-prediction accuracy by proportion of game completed',
                                     show_hist: bool = False,
                                     show_brier: bool = False,
                                     figsize: tuple[float, float] | None = None,
                                     display_names: dict[str, str] | None = None,
                                     elo_bins: tuple[int, int] | None = None,
                                     elo_bin_cfg: EloBinConfig = ELO_BINS,
                                     color_overrides: list[str | None] | None = None,
                                     linestyle_overrides: list[str | None] | None = None) -> None:
    """Plots per-model accuracy against true mover_result, binned by ply_played / ply_count (proportion of the eventual game length reached), as one row or two elo-bin-restricted rows."""
    cols_needed = [f'{name}_predicted_class' for name in names] + [mover_result_col, ply_played_col, ply_count_col]
    if show_brier:
        cols_needed += [f'{name}_prob_{c}' for name in names for c in ('win', 'draw', 'loss')]
    if elo_bins is not None:
        cols_needed += ['mover_elo', 'opponent_elo']
    df = restrict_to_common_rows(df, cols_needed)
    df = df.copy()

    n_before = len(df)
    df = df[df[ply_count_col] > 0]
    if len(df) != n_before:
        print(f'plot_accuracy_by_ply_proportion: dropped {n_before - len(df):,} rows with {ply_count_col} <= 0.')

    prop = (df[ply_played_col] / df[ply_count_col]).clip(0, 1)
    df['_ply_prop_bin'] = ((prop // bin_width) * bin_width + bin_width / 2).round(4)

    display_names, colors, linestyles = _resolve_plot_style(names, display_names, color_overrides, linestyle_overrides)
    panel_width, panel_height = figsize or BASE_FIGSIZE

    subsets, row_labels = ([df], [None]) if elo_bins is None else _split_by_elo_bin(df, elo_bins, elo_bin_cfg)
    n_rows, n_cols = len(subsets), (2 if show_brier else 1)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(panel_width * n_cols, panel_height * n_rows), squeeze=False)

    for row_idx, (sub_df, row_label) in enumerate(zip(subsets, row_labels)):
        bin_series = sub_df['_ply_prop_bin']
        bin_values = sorted(bin_series.dropna().unique())
        row_title = title if row_label is None else f'{title} -- {row_label}'

        ax_acc = axes[row_idx][0]
        _binned_line_plot(ax_acc, sub_df, names, mover_result_col, bin_series, bin_values, bin_values,
                           colors, linestyles, row_title, display_names, metric='accuracy', ylim=ylim)
        ax_acc.set_xlabel('Proportion of game completed (ply_played / ply_count)')
        if show_hist:
            _add_hist_twin(ax_acc, bin_series, bin_values, bin_values, bar_width=bin_width * 0.9)

        if show_brier:
            ax_brier = axes[row_idx][1]
            _binned_line_plot(ax_brier, sub_df, names, mover_result_col, bin_series, bin_values, bin_values,
                               colors, linestyles, f'{row_title} (Brier)', display_names, metric='brier')
            ax_brier.set_xlabel('Proportion of game completed (ply_played / ply_count)')

    plt.tight_layout()
    plt.show()


def plot_confidence_by_ply(df: pd.DataFrame, names: list[str],
                            ply_played_col: str = 'ply_played',
                            group_size: int = 20, max_ply: int = 200,
                            title: str = 'Mean predicted confidence by plies played',
                            figsize: tuple[float, float] | None = None,
                            display_names: dict[str, str] | None = None,