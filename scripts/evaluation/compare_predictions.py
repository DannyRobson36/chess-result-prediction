"""
compare_predictions.py
Reads per-model prediction csvs against a fixed val/test positions df, merges them, and provides eval-plots across models.

Latest changes: 01/09/26:
- Added table classification report and confusion matrix
"""

import glob
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix as sk_confusion_matrix, classification_report
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
    # (a) BASELINES

    'stockfish_d1': 'Stockfish',
    'maia2': 'Maia2',

    # (b) CNN -- DATASET / MODEL SIZE SCALING
    'maia2_100k_final_small': 'CNN 100k Small',
    'maia2_100k_final_medium': 'CNN 100k Large',
    'maia2_500k_final_small': 'CNN 500k Small',
    'maia2_500k_final_medium': 'CNN 500k Large',
    'maia2_2p5m_final_small': 'CNN 2.5m Small',
    'maia2_2p5m_final_medium': 'CNN 2.5m Large',
    'maia2_12p5m_final_small': 'CNN 12.5m Small',
    'maia2_12p5m_final_medium': 'CNN 12.5m Large',
    'maia2_62p5m_final_small': 'CNN 62.5m Small',
    'maia2_62p5m_final_medium': 'CNN 62.5m Large',

    # (c) CNN -- AUXILIARY LOSS WEIGHT ABLATION
    'maia2_100k_final_small_auxtest_noaux_seed1': 'CNN 100k Small - No Aux (Seed 1)',
    'maia2_100k_final_small_auxtest_noaux_seed2': 'CNN 100k Small - No Aux (Seed 2)',
    'maia2_100k_final_small_auxtest_noaux_seed3': 'CNN 100k Small - No Aux (Seed 3)',
    'maia2_100k_final_small_auxtest_noaux_seed4': 'CNN 100k Small - No Aux (Seed 4)',
    'maia2_100k_final_small_auxtest_noaux_seed5': 'CNN 100k Small - No Aux (Seed 5)',
    'maia2_100k_final_small_auxtest_aux0p5_seed1': 'CNN 100k Small - 0.5 Aux (Seed 1)',
    'maia2_100k_final_small_auxtest_aux0p5_seed2': 'CNN 100k Small - 0.5 Aux (Seed 2)',
    'maia2_100k_final_small_auxtest_aux0p5_seed3': 'CNN 100k Small - 0.5 Aux (Seed 3)',
    'maia2_100k_final_small_auxtest_aux0p5_seed4': 'CNN 100k Small - 0.5 Aux (Seed 4)',
    'maia2_100k_final_small_auxtest_aux0p5_seed5': 'CNN 100k Small - 0.5 Aux (Seed 5)',
    'maia2_100k_final_small_auxtest_aux1p0_seed1': 'CNN 100k Small - 1.0 Aux (Seed 1)',
    'maia2_100k_final_small_auxtest_aux1p0_seed2': 'CNN 100k Small - 1.0 Aux (Seed 2)',
    'maia2_100k_final_small_auxtest_aux1p0_seed3': 'CNN 100k Small - 1.0 Aux (Seed 3)',
    'maia2_100k_final_small_auxtest_aux1p0_seed4': 'CNN 100k Small - 1.0 Aux (Seed 4)',
    'maia2_100k_final_small_auxtest_aux1p0_seed5': 'CNN 100k Small - 1.0 Aux (Seed 5)',
    'maia2_500k_final_small_auxtest_noaux_seed1': 'CNN 500k Small - No Aux (Seed 1)',
    'maia2_500k_final_small_auxtest_noaux_seed2': 'CNN 500k Small - No Aux (Seed 2)',
    'maia2_500k_final_small_auxtest_noaux_seed3': 'CNN 500k Small - No Aux (Seed 3)',
    'maia2_500k_final_small_auxtest_noaux_seed4': 'CNN 500k Small - No Aux (Seed 4)',
    'maia2_500k_final_small_auxtest_noaux_seed5': 'CNN 500k Small - No Aux (Seed 5)',
    'maia2_500k_final_small_auxtest_aux0p5_seed1': 'CNN 500k Small - 0.5 Aux (Seed 1)',
    'maia2_500k_final_small_auxtest_aux0p5_seed2': 'CNN 500k Small - 0.5 Aux (Seed 2)',
    'maia2_500k_final_small_auxtest_aux0p5_seed3': 'CNN 500k Small - 0.5 Aux (Seed 3)',
    'maia2_500k_final_small_auxtest_aux0p5_seed4': 'CNN 500k Small - 0.5 Aux (Seed 4)',
    'maia2_500k_final_small_auxtest_aux0p5_seed5': 'CNN 500k Small - 0.5 Aux (Seed 5)',
    'maia2_500k_final_small_auxtest_aux1p0_seed1': 'CNN 500k Small - 1.0 Aux (Seed 1)',
    'maia2_500k_final_small_auxtest_aux1p0_seed2': 'CNN 500k Small - 1.0 Aux (Seed 2)',
    'maia2_500k_final_small_auxtest_aux1p0_seed3': 'CNN 500k Small - 1.0 Aux (Seed 3)',
    'maia2_500k_final_small_auxtest_aux1p0_seed4': 'CNN 500k Small - 1.0 Aux (Seed 4)',
    'maia2_500k_final_small_auxtest_aux1p0_seed5': 'CNN 500k Small - 1.0 Aux (Seed 5)',

    # (d) CNN -- SINGLE ELO-BIN SPECIALISTS
    'maia2_500k_final_small_singlebin_bin1': 'CNN 500k Small - Bin 1',
    'maia2_500k_final_small_singlebin_bin2': 'CNN 500k Small - Bin 2',
    'maia2_500k_final_small_singlebin_bin3': 'CNN 500k Small - Bin 3',
    'maia2_500k_final_small_singlebin_bin4': 'CNN 500k Small - Bin 4',
    'maia2_500k_final_small_singlebin_bin5': 'CNN 500k Small - Bin 5',
    'maia2_500k_final_small_singlebin_bin6': 'CNN 500k Small - Bin 6',
    'maia2_500k_final_small_singlebin_bin7': 'CNN 500k Small - Bin 7',
    'maia2_500k_final_small_singlebin_bin8': 'CNN 500k Small - Bin 8',
    'maia2_500k_final_small_singlebin_bin9': 'CNN 500k Small - Bin 9',

    # (e) CNN -- FEATURE ABLATION (NO ELO / NO CLOCK / NO INC)
    'maia2_2p5m_final_medium_no_elo': 'CNN 2.5m Large - No Elo',
    'maia2_2p5m_final_medium_no_clock': 'CNN 2.5m Large - No Clock',
    'maia2_2p5m_final_medium_no_inc': 'CNN 2.5m Large - No Inc',

    # (f) CNN -- REPLICAS
    'maia2_value_replica': 'Maia2-Value (Replica)',

    # (g) TRANSFORMER -- AUXILIARY LOSS WEIGHT ABLATION
    'pure_transformer_100k_final_small_2and2_noaux_seed1': 'Transformer 100k - No Aux (Seed 1)',
    'pure_transformer_100k_final_small_2and2_noaux_seed2': 'Transformer 100k - No Aux (Seed 2)',
    'pure_transformer_100k_final_small_2and2_noaux_seed3': 'Transformer 100k - No Aux (Seed 3)',
    'pure_transformer_100k_final_small_2and2_aux0p5_seed1': 'Transformer 100k - 0.5 Aux (Seed 1)',
    'pure_transformer_100k_final_small_2and2_aux0p5_seed2': 'Transformer 100k - 0.5 Aux (Seed 2)',
    'pure_transformer_100k_final_small_2and2_aux0p5_seed3': 'Transformer 100k - 0.5 Aux (Seed 3)',
    'pure_transformer_100k_final_small_2and2_aux1p0_seed1': 'Transformer 100k - 1.0 Aux (Seed 1)',
    'pure_transformer_100k_final_small_2and2_aux1p0_seed2': 'Transformer 100k - 1.0 Aux (Seed 2)',
    'pure_transformer_100k_final_small_2and2_aux1p0_seed3': 'Transformer 100k - 1.0 Aux (Seed 3)',
    'pure_transformer_500k_final_small_2and2_noaux_seed1': 'Transformer 500k - No Aux (Seed 1)',
    'pure_transformer_500k_final_small_2and2_noaux_seed2': 'Transformer 500k - No Aux (Seed 2)',
    'pure_transformer_500k_final_small_2and2_noaux_seed3': 'Transformer 500k - No Aux (Seed 3)',
    'pure_transformer_500k_final_small_2and2_aux0p5_seed1': 'Transformer 500k - 0.5 Aux (Seed 1)',
    'pure_transformer_500k_final_small_2and2_aux0p5_seed2': 'Transformer 500k - 0.5 Aux (Seed 2)',
    'pure_transformer_500k_final_small_2and2_aux0p5_seed3': 'Transformer 500k - 0.5 Aux (Seed 3)',
    'pure_transformer_500k_final_small_2and2_aux1p0_seed1': 'Transformer 500k - 1.0 Aux (Seed 1)',
    'pure_transformer_500k_final_small_2and2_aux1p0_seed2': 'Transformer 500k - 1.0 Aux (Seed 2)',
    'pure_transformer_500k_final_small_2and2_aux1p0_seed3': 'Transformer 500k - 1.0 Aux (Seed 3)',
}

####################
# FUNCTIONS
####################

# (a) DISCOVERY & LOADING

def discover_predictions(predictions_dir: str, split: str, names: list[str] | None = None) -> dict[str, str]:
    """Finds {split}_pred_{name}.csv in predictions_dir for each of names, or every matching file
    if names is None, returns {name: path}, sorted by name. A requested name with no matching file is
    skipped, not raised."""
    prefix = f'{split}_pred_'

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
            f"No '{split}_pred_*.csv' files found in {predictions_dir}"
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
                            color_overrides: list[str | None] | None = None,
                            linestyle_overrides: list[str | None] | None = None) -> None:
    """Plots each model's mean max predicted probability, binned by plies played."""
    cols_needed = [f'{name}_prob_{c}' for name in names for c in ('win', 'draw', 'loss')] + [ply_played_col]
    df = restrict_to_common_rows(df, cols_needed)
    df = df[df[ply_played_col] <= max_ply].copy()
    df['_ply_group'] = (df[ply_played_col] // group_size) * group_size + group_size // 2

    bin_values = sorted(df['_ply_group'].dropna().unique())
    display_names, colors, linestyles = _resolve_plot_style(names, display_names, color_overrides, linestyle_overrides)
    fig, ax = plt.subplots(figsize=figsize or BASE_FIGSIZE)
    _binned_line_plot(ax, df, names, None, df['_ply_group'], bin_values, bin_values,
                       colors, linestyles, title, display_names, metric='confidence')
    ax.set_xlabel(f'Plies played (bins of {group_size})')

    plt.tight_layout()
    plt.show()

# (i) ACCURACY BY CLOCK

def plot_accuracy_by_combined_clock(df: pd.DataFrame, names: list[str],
                                     mover_clock_col: str = 'mover_clock',
                                     opponent_clock_col: str = 'opponent_clock',
                                     time_control_col: str = 'time_control',
                                     mover_result_col: str = 'mover_result',
                                     bin_width: float = 0.05,
                                     ylim: tuple[float, float] = (50, 80),
                                     title: str = 'Result-prediction accuracy by time remaining',
                                     show_hist: bool = False,
                                     show_brier: bool = False,
                                     figsize: tuple[float, float] | None = None,
                                     display_names: dict[str, str] | None = None,
                                     elo_bins: tuple[int, int] | None = None,
                                     elo_bin_cfg: EloBinConfig = ELO_BINS,
                                     split_by_increment: bool = False,
                                     termination_col: str = 'termination',
                                     split_by_termination: bool = False,
                                     color_overrides: list[str | None] | None = None,
                                     linestyle_overrides: list[str | None] | None = None) -> None:
    """Plots per-model accuracy (and optionally Brier score) against true mover_result, binned by proportion of clock time remaining, as one row or two elo-bin/increment/termination-restricted rows."""
    _count_active_row_splits(elo_bins is not None, split_by_increment, split_by_termination)

    cols_needed = ([f'{name}_predicted_class' for name in names]
                   + [mover_result_col, mover_clock_col, opponent_clock_col, time_control_col])
    if show_brier:
        cols_needed += [f'{name}_prob_{c}' for name in names for c in ('win', 'draw', 'loss')]
    if elo_bins is not None:
        cols_needed += ['mover_elo', 'opponent_elo']
    if split_by_termination:
        cols_needed += [termination_col]
    df = restrict_to_common_rows(df, cols_needed)

    game_time = df[time_control_col].map(SEC_MAPPING)
    n_before = len(df)
    df = df[game_time.notna()]
    game_time = game_time[game_time.notna()]
    if len(df) != n_before:
        print(f'plot_accuracy_by_combined_clock: dropped {n_before - len(df):,} rows with unmapped {time_control_col}.')

    df = df.copy()
    time_left_prop = ((df[mover_clock_col] + df[opponent_clock_col]) / (2 * game_time)).clip(0, 1)
    df['_time_bin'] = ((time_left_prop // bin_width) * bin_width + bin_width / 2).round(4)

    display_names, colors, linestyles = _resolve_plot_style(names, display_names, color_overrides, linestyle_overrides)
    panel_width, panel_height = figsize or BASE_FIGSIZE

    if elo_bins is not None:
        subsets, row_labels = _split_by_elo_bin(df, elo_bins, elo_bin_cfg)
    elif split_by_increment:
        inc_flag = df[time_control_col].map(INC_FLAG_MAPPING)
        n_unmapped = int(inc_flag.isna().sum())
        if n_unmapped:
            print(f'plot_accuracy_by_combined_clock: {n_unmapped:,} rows have unmapped {time_control_col} '
                  f'for increment split -- excluded from both panels.')
        subsets = [df[inc_flag == 0], df[inc_flag == 1]]
        row_labels = ['no increment', 'with increment']
    elif split_by_termination:
        subsets, row_labels = _split_by_termination(df, termination_col)
    else:
        subsets, row_labels = [df], [None]

    n_rows, n_cols = len(subsets), (2 if show_brier else 1)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(panel_width * n_cols, panel_height * n_rows), squeeze=False)

    for row_idx, (sub_df, row_label) in enumerate(zip(subsets, row_labels)):
        bin_series = sub_df['_time_bin']
        bin_values = sorted(bin_series.dropna().unique())
        row_title = title if row_label is None else f'{title} -- {row_label}'

        ax_acc = axes[row_idx][0]
        _binned_line_plot(ax_acc, sub_df, names, mover_result_col, bin_series, bin_values, bin_values,
                           colors, linestyles, row_title, display_names, metric='accuracy', ylim=ylim)
        ax_acc.set_xlabel('Proportion of total time remaining (both players summed)')
        ax_acc.invert_xaxis()
        if show_hist:
            _add_hist_twin(ax_acc, bin_series, bin_values, bin_values, bar_width=bin_width * 0.9)

        if show_brier:
            ax_brier = axes[row_idx][1]
            _binned_line_plot(ax_brier, sub_df, names, mover_result_col, bin_series, bin_values, bin_values,
                               colors, linestyles, f'{row_title} (Brier)', display_names, metric='brier')
            ax_brier.set_xlabel('Proportion of total time remaining (both players summed)')
            ax_brier.invert_xaxis()

    plt.tight_layout()
    plt.show()


def plot_confidence_by_combined_clock(df: pd.DataFrame, names: list[str],
                                       mover_clock_col: str = 'mover_clock',
                                       opponent_clock_col: str = 'opponent_clock',
                                       time_control_col: str = 'time_control',
                                       bin_width: float = 0.05,
                                       title: str = 'Mean predicted confidence by time remaining',
                                       figsize: tuple[float, float] | None = None,
                                       display_names: dict[str, str] | None = None,
                                       color_overrides: list[str | None] | None = None,
                                       linestyle_overrides: list[str | None] | None = None) -> None:
    """Plots each model's mean max predicted probability, binned by proportion of clock time remaining."""
    cols_needed = ([f'{name}_prob_{c}' for name in names for c in ('win', 'draw', 'loss')]
                   + [mover_clock_col, opponent_clock_col, time_control_col])
    df = restrict_to_common_rows(df, cols_needed)

    game_time = df[time_control_col].map(SEC_MAPPING)
    df = df[game_time.notna()]
    game_time = game_time[game_time.notna()]
    df = df.copy()
    time_left_prop = ((df[mover_clock_col] + df[opponent_clock_col]) / (2 * game_time)).clip(0, 1)
    df['_time_bin'] = ((time_left_prop // bin_width) * bin_width + bin_width / 2).round(4)

    bin_values = sorted(df['_time_bin'].dropna().unique())
    display_names, colors, linestyles = _resolve_plot_style(names, display_names, color_overrides, linestyle_overrides)
    fig, ax = plt.subplots(figsize=figsize or BASE_FIGSIZE)
    _binned_line_plot(ax, df, names, None, df['_time_bin'], bin_values, bin_values,
                       colors, linestyles, title, display_names, metric='confidence')
    ax.set_xlabel('Proportion of total time remaining (both players summed)')
    ax.invert_xaxis()

    plt.tight_layout()
    plt.show()


def plot_accuracy_by_mover_clock(df: pd.DataFrame, names: list[str],
                                  mover_clock_col: str = 'mover_clock',
                                  time_control_col: str = 'time_control',
                                  mover_result_col: str = 'mover_result',
                                  bin_width: float = 0.05,
                                  ylim: tuple[float, float] = (50, 80),
                                  title: str = "Result-prediction accuracy by mover's own time remaining",
                                  show_hist: bool = False,
                                  show_brier: bool = False,
                                  figsize: tuple[float, float] | None = None,
                                  display_names: dict[str, str] | None = None,
                                  elo_bins: tuple[int, int] | None = None,
                                  elo_bin_cfg: EloBinConfig = ELO_BINS,
                                  termination_col: str = 'termination',
                                  split_by_termination: bool = False,
                                  color_overrides: list[str | None] | None = None,
                                  linestyle_overrides: list[str | None] | None = None) -> None:
    """Plots per-model accuracy (and optionally Brier score) against true mover_result, binned by mover's own proportion of clock time remaining, as one row or two elo-bin/termination-restricted rows."""
    _count_active_row_splits(elo_bins is not None, split_by_termination)

    cols_needed = [f'{name}_predicted_class' for name in names] + [mover_result_col, mover_clock_col, time_control_col]
    if show_brier:
        cols_needed += [f'{name}_prob_{c}' for name in names for c in ('win', 'draw', 'loss')]
    if elo_bins is not None:
        cols_needed += ['mover_elo', 'opponent_elo']
    if split_by_termination:
        cols_needed += [termination_col]
    df = restrict_to_common_rows(df, cols_needed)

    game_time = df[time_control_col].map(SEC_MAPPING)
    n_before = len(df)
    df = df[game_time.notna()]
    game_time = game_time[game_time.notna()]
    if len(df) != n_before:
        print(f'plot_accuracy_by_mover_clock: dropped {n_before - len(df):,} rows with unmapped {time_control_col}.')

    df = df.copy()
    time_left_prop = (df[mover_clock_col] / game_time).clip(0, 1)
    df['_mover_time_bin'] = ((time_left_prop // bin_width) * bin_width + bin_width / 2).round(4)

    display_names, colors, linestyles = _resolve_plot_style(names, display_names, color_overrides, linestyle_overrides)
    panel_width, panel_height = figsize or BASE_FIGSIZE

    if elo_bins is not None:
        subsets, row_labels = _split_by_elo_bin(df, elo_bins, elo_bin_cfg)
    elif split_by_termination:
        subsets, row_labels = _split_by_termination(df, termination_col)
    else:
        subsets, row_labels = [df], [None]

    n_rows, n_cols = len(subsets), (2 if show_brier else 1)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(panel_width * n_cols, panel_height * n_rows), squeeze=False)

    for row_idx, (sub_df, row_label) in enumerate(zip(subsets, row_labels)):
        bin_series = sub_df['_mover_time_bin']
        bin_values = sorted(bin_series.dropna().unique())
        row_title = title if row_label is None else f'{title} -- {row_label}'

        ax_acc = axes[row_idx][0]
        _binned_line_plot(ax_acc, sub_df, names, mover_result_col, bin_series, bin_values, bin_values,
                           colors, linestyles, row_title, display_names, metric='accuracy', ylim=ylim)
        ax_acc.set_xlabel("Proportion of mover's own time remaining")
        ax_acc.invert_xaxis()
        if show_hist:
            _add_hist_twin(ax_acc, bin_series, bin_values, bin_values, bar_width=bin_width * 0.9)

        if show_brier:
            ax_brier = axes[row_idx][1]
            _binned_line_plot(ax_brier, sub_df, names, mover_result_col, bin_series, bin_values, bin_values,
                               colors, linestyles, f'{row_title} (Brier)', display_names, metric='brier')
            ax_brier.set_xlabel("Proportion of mover's own time remaining")
            ax_brier.invert_xaxis()

    plt.tight_layout()
    plt.show()


def plot_confidence_by_mover_clock(df: pd.DataFrame, names: list[str],
                                    mover_clock_col: str = 'mover_clock',
                                    time_control_col: str = 'time_control',
                                    bin_width: float = 0.05,
                                    title: str = "Mean predicted confidence by mover's own time remaining",
                                    figsize: tuple[float, float] | None = None,
                                    display_names: dict[str, str] | None = None,
                                    color_overrides: list[str | None] | None = None,
                                    linestyle_overrides: list[str | None] | None = None) -> None:
    """Plots each model's mean max predicted probability, binned by mover's own proportion of clock time remaining."""
    cols_needed = [f'{name}_prob_{c}' for name in names for c in ('win', 'draw', 'loss')] + [mover_clock_col, time_control_col]
    df = restrict_to_common_rows(df, cols_needed)

    game_time = df[time_control_col].map(SEC_MAPPING)
    df = df[game_time.notna()]
    game_time = game_time[game_time.notna()]
    df = df.copy()
    time_left_prop = (df[mover_clock_col] / game_time).clip(0, 1)
    df['_mover_time_bin'] = ((time_left_prop // bin_width) * bin_width + bin_width / 2).round(4)

    bin_values = sorted(df['_mover_time_bin'].dropna().unique())
    display_names, colors, linestyles = _resolve_plot_style(names, display_names, color_overrides, linestyle_overrides)
    fig, ax = plt.subplots(figsize=figsize or BASE_FIGSIZE)
    _binned_line_plot(ax, df, names, None, df['_mover_time_bin'], bin_values, bin_values,
                       colors, linestyles, title, display_names, metric='confidence')
    ax.set_xlabel("Proportion of mover's own time remaining")
    ax.invert_xaxis()

    plt.tight_layout()
    plt.show()


def plot_accuracy_by_clock_diff(df: pd.DataFrame, names: list[str],
                                 mover_clock_col: str = 'mover_clock',
                                 opponent_clock_col: str = 'opponent_clock',
                                 mover_result_col: str = 'mover_result',
                                 bin_width: float = 10.0,
                                 max_abs_diff: float | None = None,
                                 ylim: tuple[float, float] = (50, 80),
                                 title: str = 'Result-prediction accuracy by mover/opponent clock difference',
                                 show_hist: bool = False,
                                 show_brier: bool = False,
                                 figsize: tuple[float, float] | None = None,
                                 display_names: dict[str, str] | None = None,
                                 elo_bins: tuple[int, int] | None = None,
                                 elo_bin_cfg: EloBinConfig = ELO_BINS,
                                 termination_col: str = 'termination',
                                 split_by_termination: bool = False,
                                 color_overrides: list[str | None] | None = None,
                                 linestyle_overrides: list[str | None] | None = None) -> None:
    """Plots per-model accuracy (and optionally Brier score) against true mover_result, binned by mover-minus-opponent clock seconds, as one row or two elo-bin/termination-restricted rows."""
    _count_active_row_splits(elo_bins is not None, split_by_termination)

    cols_needed = [f'{name}_predicted_class' for name in names] + [mover_result_col, mover_clock_col, opponent_clock_col]
    if show_brier:
        cols_needed += [f'{name}_prob_{c}' for name in names for c in ('win', 'draw', 'loss')]
    if elo_bins is not None:
        cols_needed += ['mover_elo', 'opponent_elo']
    if split_by_termination:
        cols_needed += [termination_col]
    df = restrict_to_common_rows(df, cols_needed)
    df = df.copy()

    clock_diff = df[mover_clock_col] - df[opponent_clock_col]
    if max_abs_diff is not None:
        n_before = len(df)
        keep = clock_diff.abs() <= max_abs_diff
        df, clock_diff = df[keep], clock_diff[keep]
        print(f'plot_accuracy_by_clock_diff: max_abs_diff={max_abs_diff} kept {len(df):,} of {n_before:,} rows.')

    df['_clock_diff_bin'] = (clock_diff // bin_width) * bin_width + bin_width / 2

    display_names, colors, linestyles = _resolve_plot_style(names, display_names, color_overrides, linestyle_overrides)
    panel_width, panel_height = figsize or BASE_FIGSIZE

    if elo_bins is not None:
        subsets, row_labels = _split_by_elo_bin(df, elo_bins, elo_bin_cfg)
    elif split_by_termination:
        subsets, row_labels = _split_by_termination(df, termination_col)
    else:
        subsets, row_labels = [df], [None]

    n_rows, n_cols = len(subsets), (2 if show_brier else 1)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(panel_width * n_cols, panel_height * n_rows), squeeze=False)

    for row_idx, (sub_df, row_label) in enumerate(zip(subsets, row_labels)):
        bin_series = sub_df['_clock_diff_bin']
        bin_values = sorted(bin_series.dropna().unique())
        row_title = title if row_label is None else f'{title} -- {row_label}'

        ax_acc = axes[row_idx][0]
        _binned_line_plot(ax_acc, sub_df, names, mover_result_col, bin_series, bin_values, bin_values,
                           colors, linestyles, row_title, display_names, metric='accuracy', ylim=ylim)
        ax_acc.set_xlabel('Mover clock minus opponent clock (seconds)')
        if show_hist:
            _add_hist_twin(ax_acc, bin_series, bin_values, bin_values, bar_width=bin_width * 0.9)

        if show_brier:
            ax_brier = axes[row_idx][1]
            _binned_line_plot(ax_brier, sub_df, names, mover_result_col, bin_series, bin_values, bin_values,
                               colors, linestyles, f'{row_title} (Brier)', display_names, metric='brier')
            ax_brier.set_xlabel('Mover clock minus opponent clock (seconds)')

    plt.tight_layout()
    plt.show()


def plot_confidence_by_clock_diff(df: pd.DataFrame, names: list[str],
                                   mover_clock_col: str = 'mover_clock',
                                   opponent_clock_col: str = 'opponent_clock',
                                   bin_width: float = 10.0,
                                   max_abs_diff: float | None = None,
                                   title: str = 'Mean predicted confidence by mover/opponent clock difference',
                                   figsize: tuple[float, float] | None = None,
                                   display_names: dict[str, str] | None = None,
                                   color_overrides: list[str | None] | None = None,
                                   linestyle_overrides: list[str | None] | None = None) -> None:
    """Plots each model's mean max predicted probability, binned by mover-minus-opponent clock seconds."""
    cols_needed = [f'{name}_prob_{c}' for name in names for c in ('win', 'draw', 'loss')] + [mover_clock_col, opponent_clock_col]
    df = restrict_to_common_rows(df, cols_needed)
    df = df.copy()

    clock_diff = df[mover_clock_col] - df[opponent_clock_col]
    if max_abs_diff is not None:
        keep = clock_diff.abs() <= max_abs_diff
        df, clock_diff = df[keep], clock_diff[keep]

    df['_clock_diff_bin'] = (clock_diff // bin_width) * bin_width + bin_width / 2
    bin_values = sorted(df['_clock_diff_bin'].dropna().unique())
    display_names, colors, linestyles = _resolve_plot_style(names, display_names, color_overrides, linestyle_overrides)
    fig, ax = plt.subplots(figsize=figsize or BASE_FIGSIZE)
    _binned_line_plot(ax, df, names, None, df['_clock_diff_bin'], bin_values, bin_values,
                       colors, linestyles, title, display_names, metric='confidence')
    ax.set_xlabel('Mover clock minus opponent clock (seconds)')

    plt.tight_layout()
    plt.show()


def _log_ratio_bin_edges(n_bins_per_side: int) -> list[float]:
    """Returns integer-step bin edges from -n_bins_per_side to +n_bins_per_side, with open -inf/+inf tails."""
    if n_bins_per_side < 1:
        raise ValueError(f'n_bins_per_side must be >= 1, got {n_bins_per_side}.')
    edges = list(range(-n_bins_per_side, n_bins_per_side + 1))
    return [-np.inf] + edges + [np.inf]


def _log_ratio_bin_labels(edges: list[float], step_multiplier: float) -> list[str]:
    """Returns a multiplier display label per bin, e.g. 'x0.125-', 'x1-x2', 'x8+'."""
    labels = []
    for i in range(len(edges) - 1):
        lo, hi = edges[i], edges[i + 1]
        if lo == -np.inf:
            labels.append(f'x{step_multiplier ** hi:.3g}-')
        elif hi == np.inf:
            labels.append(f'x{step_multiplier ** lo:.3g}+')
        else:
            labels.append(f'x{step_multiplier ** lo:.3g}-x{step_multiplier ** hi:.3g}')
    return labels


def plot_accuracy_by_clock_ratio(df: pd.DataFrame, names: list[str],
                                  mover_clock_col: str = 'mover_clock',
                                  opponent_clock_col: str = 'opponent_clock',
                                  time_control_col: str = 'time_control',
                                  mover_result_col: str = 'mover_result',
                                  step_multiplier: float = 2.0,
                                  n_bins_per_side: int = 3,
                                  max_clock_seconds: int | None = None,
                                  no_increment_only: bool = False,
                                  ylim: tuple[float, float] = (50, 80),
                                  title: str = 'Result-prediction accuracy by opponent/mover clock ratio',
                                  show_hist: bool = False,
                                  show_brier: bool = False,
                                  figsize: tuple[float, float] | None = None,
                                  display_names: dict[str, str] | None = None,
                                  elo_bins: tuple[int, int] | None = None,
                                  elo_bin_cfg: EloBinConfig = ELO_BINS,
                                  termination_col: str = 'termination',
                                  split_by_termination: bool = False,
                                  color_overrides: list[str | None] | None = None,
                                  linestyle_overrides: list[str | None] | None = None) -> None:
    """Plots per-model accuracy (and optionally Brier score) against true mover_result, binned by opponent/mover clock ratio, as one row or two elo-bin/termination-restricted rows."""
    _count_active_row_splits(elo_bins is not None, split_by_termination)

    cols_needed = [f'{name}_predicted_class' for name in names] + [mover_result_col, mover_clock_col, opponent_clock_col]
    if no_increment_only:
        cols_needed.append(time_control_col)
    if show_brier:
        cols_needed += [f'{name}_prob_{c}' for name in names for c in ('win', 'draw', 'loss')]
    if elo_bins is not None:
        cols_needed += ['mover_elo', 'opponent_elo']
    if split_by_termination:
        cols_needed += [termination_col]
    df = restrict_to_common_rows(df, cols_needed)

    if max_clock_seconds is not None:
        n_before = len(df)
        df = df[(df[mover_clock_col] <= max_clock_seconds) & (df[opponent_clock_col] <= max_clock_seconds)]
        print(f'plot_accuracy_by_clock_ratio: max_clock_seconds={max_clock_seconds} kept {len(df):,} of {n_before:,} rows.')

    if no_increment_only:
        n_before = len(df)
        df = df[df[time_control_col].map(INC_FLAG_MAPPING) == 0]
        print(f'plot_accuracy_by_clock_ratio: no_increment_only kept {len(df):,} of {n_before:,} rows.')

    both_zero = (df[mover_clock_col] == 0) & (df[opponent_clock_col] == 0)
    if both_zero.any():
        print(f'plot_accuracy_by_clock_ratio: dropping {int(both_zero.sum()):,} rows with both clocks at 0.')
        df = df[~both_zero]

    with np.errstate(divide='ignore', invalid='ignore'):
        ratio = df[opponent_clock_col] / df[mover_clock_col]
        log_ratio = np.log(ratio) / np.log(step_multiplier)

    edges = _log_ratio_bin_edges(n_bins_per_side)
    labels = _log_ratio_bin_labels(edges, step_multiplier)
    df = df.copy()
    df['_ratio_bin'] = pd.cut(log_ratio, bins=edges, labels=False, right=True, include_lowest=True)

    display_names, colors, linestyles = _resolve_plot_style(names, display_names, color_overrides, linestyle_overrides)
    panel_width, panel_height = figsize or BASE_FIGSIZE
    bin_values = list(range(len(labels)))

    if elo_bins is not None:
        subsets, row_labels = _split_by_elo_bin(df, elo_bins, elo_bin_cfg)
    elif split_by_termination:
        subsets, row_labels = _split_by_termination(df, termination_col)
    else:
        subsets, row_labels = [df], [None]

    n_rows, n_cols = len(subsets), (2 if show_brier else 1)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(panel_width * n_cols, panel_height * n_rows), squeeze=False)

    for row_idx, (sub_df, row_label) in enumerate(zip(subsets, row_labels)):
        row_title = title if row_label is None else f'{title} -- {row_label}'

        ax_acc = axes[row_idx][0]
        _binned_line_plot(ax_acc, sub_df, names, mover_result_col, sub_df['_ratio_bin'], bin_values, labels,
                           colors, linestyles, row_title, display_names, metric='accuracy', ylim=ylim)
        ax_acc.set_xlabel('Opponent clock / mover clock')
        ax_acc.tick_params(axis='x', rotation=45)
        if show_hist:
            _add_hist_twin(ax_acc, sub_df['_ratio_bin'], bin_values, labels, bar_width=0.9)

        if show_brier:
            ax_brier = axes[row_idx][1]
            _binned_line_plot(ax_brier, sub_df, names, mover_result_col, sub_df['_ratio_bin'], bin_values, labels,
                               colors, linestyles, f'{row_title} (Brier)', display_names, metric='brier')
            ax_brier.set_xlabel('Opponent clock / mover clock')
            ax_brier.tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.show()


def plot_confidence_by_clock_ratio(df: pd.DataFrame, names: list[str],
                                    mover_clock_col: str = 'mover_clock',
                                    opponent_clock_col: str = 'opponent_clock',
                                    step_multiplier: float = 2.0,
                                    n_bins_per_side: int = 3,
                                    max_clock_seconds: int | None = None,
                                    title: str = 'Mean predicted confidence by opponent/mover clock ratio',
                                    figsize: tuple[float, float] | None = None,
                                    display_names: dict[str, str] | None = None,
                                    color_overrides: list[str | None] | None = None,
                                    linestyle_overrides: list[str | None] | None = None) -> None:
    """Plots each model's mean max predicted probability, binned by opponent/mover clock ratio."""
    cols_needed = [f'{name}_prob_{c}' for name in names for c in ('win', 'draw', 'loss')] + [mover_clock_col, opponent_clock_col]
    df = restrict_to_common_rows(df, cols_needed)

    if max_clock_seconds is not None:
        df = df[(df[mover_clock_col] <= max_clock_seconds) & (df[opponent_clock_col] <= max_clock_seconds)]

    both_zero = (df[mover_clock_col] == 0) & (df[opponent_clock_col] == 0)
    df = df[~both_zero]

    with np.errstate(divide='ignore', invalid='ignore'):
        ratio = df[opponent_clock_col] / df[mover_clock_col]
        log_ratio = np.log(ratio) / np.log(step_multiplier)

    edges = _log_ratio_bin_edges(n_bins_per_side)
    labels = _log_ratio_bin_labels(edges, step_multiplier)
    df = df.copy()
    df['_ratio_bin'] = pd.cut(log_ratio, bins=edges, labels=False, right=True, include_lowest=True)
    bin_values = list(range(len(labels)))

    display_names, colors, linestyles = _resolve_plot_style(names, display_names, color_overrides, linestyle_overrides)
    fig, ax = plt.subplots(figsize=figsize or BASE_FIGSIZE)
    _binned_line_plot(ax, df, names, None, df['_ratio_bin'], bin_values, labels,
                       colors, linestyles, title, display_names, metric='confidence')
    ax.set_xlabel('Opponent clock / mover clock')
    ax.tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.show()

# (j) ROC & CALIBRATION

def _draw_roc_panel(ax: plt.Axes, df: pd.DataFrame, names: list[str], mover_result_col: str, cls: str,
                     class_to_result: dict[str, float], colors: dict[str, str], linestyles: dict[str, str],
                     ylim: tuple[float, float], panel_title: str, display_names: dict[str, str] | None) -> None:
    """Draws one ROC-curve panel (one result class, all models) onto ax."""
    y_true = (df[mover_result_col] == class_to_result[cls]).astype(int)
    for name in names:
        y_score = df[f'{name}_prob_{cls}']
        fpr, tpr, _ = roc_curve(y_true, y_score)
        roc_auc = auc(fpr, tpr)
        label = f'{resolve_display_name(name, display_names)} (AUC = {roc_auc:.3f})'
        ax.plot(fpr, tpr, color=colors[name], linestyle=linestyles[name], linewidth=2, label=label)
    ax.plot([0, 1], [0, 1], color='grey', linestyle='--', linewidth=1, label='Chance')
    ax.set_xlabel('False Positive Rate'); ax.set_ylabel('True Positive Rate')
    ax.set_title(panel_title); ax.legend(loc='lower right')
    ax.set_xlim(0, 1); ax.set_ylim(*ylim)


def plot_roc_auc(df: pd.DataFrame, names: list[str], mover_result_col: str = 'mover_result',
                  positive_class: str | None = None, ylim: tuple[float, float] = (0, 1),
                  title: str | None = None, figsize: tuple[float, float] | None = None,
                  display_names: dict[str, str] | None = None,
                  elo_bins: tuple[int, int] | None = None, elo_bin_cfg: EloBinConfig = ELO_BINS,
                  color_overrides: list[str | None] | None = None,
                  linestyle_overrides: list[str | None] | None = None) -> None:
    """Plots ROC curves (one panel per class, or just positive_class if given) across models, as one row or two elo-bin-restricted rows."""
    class_to_result = {'win': 1.0, 'loss': 0.0, 'draw': 0.5}
    if positive_class is not None and positive_class not in class_to_result:
        raise ValueError(f'positive_class must be one of {list(class_to_result)}, got {positive_class!r}')
    classes_to_plot = [positive_class] if positive_class is not None else ['win', 'draw', 'loss']
    cols_needed = [f'{name}_prob_{cls}' for name in names for cls in classes_to_plot] + [mover_result_col]
    if elo_bins is not None:
        cols_needed += ['mover_elo', 'opponent_elo']
    df = restrict_to_common_rows(df, cols_needed)
    if positive_class is None:
        draw_cols = [f'{name}_prob_draw' for name in names]
        if (df[draw_cols] == 0).all(axis=None):
            classes_to_plot = ['win', 'loss']
            print('plot_roc_auc: all draw probabilities are 0 across every model -- skipping draw panel.')
    row_data = [(df, None)] if elo_bins is None else list(zip(*_split_by_elo_bin(df, elo_bins, elo_bin_cfg)))
    panel_width, panel_height = figsize or BASE_FIGSIZE
    display_names, colors, linestyles = _resolve_plot_style(names, display_names, color_overrides, linestyle_overrides)
    fig, axes = plt.subplots(len(row_data), len(classes_to_plot),
                              figsize=(panel_width * len(classes_to_plot), panel_height * len(row_data)), squeeze=False)
    if title:
        fig.suptitle(title)
    for row_idx, (row_df, panel_label) in enumerate(row_data):
        for col_idx, cls in enumerate(classes_to_plot):
            panel_title = f'{cls.capitalize()} prediction' + (f' -- {panel_label}' if panel_label is not None else '')
            _draw_roc_panel(axes[row_idx][col_idx], row_df, names, mover_result_col, cls,
                             class_to_result, colors, linestyles, ylim, panel_title, display_names)
    plt.tight_layout(); plt.show()


def _draw_calibration_panel(ax: plt.Axes, df: pd.DataFrame, names: list[str], mover_result_col: str, cls: str,
                             class_to_result: dict[str, float], bin_edges: np.ndarray, n_bins: int,
                             colors: dict[str, str], linestyles: dict[str, str], ylim: tuple[float, float],
                             panel_title: str, display_names: dict[str, str] | None) -> None:
    """Draws one reliability-diagram panel (one result class, all models) onto ax."""
    y_true = (df[mover_result_col] == class_to_result[cls]).astype(int)
    for name in names:
        y_score = df[f'{name}_prob_{cls}']
        prob_bin = pd.cut(y_score, bins=bin_edges, labels=False, right=True, include_lowest=True)
        mean_pred, observed_rate = [], []
        for b in range(n_bins):
            mask = prob_bin == b
            if mask.sum():
                mean_pred.append(y_score[mask].mean()); observed_rate.append(y_true[mask].mean())
        ax.plot(mean_pred, observed_rate, marker='D', markersize=6, markeredgecolor='white',
                markeredgewidth=0.6, linewidth=2, linestyle=linestyles[name], color=colors[name],
                label=resolve_display_name(name, display_names))
    ax.plot([0, 1], [0, 1], color='grey', linestyle='--', linewidth=1, label='Perfectly calibrated')
    ax.set_xlabel('Mean predicted probability'); ax.set_ylabel('Observed frequency')
    ax.set_title(panel_title); ax.legend(loc='upper left')
    ax.set_xlim(0, 1); ax.set_ylim(*ylim)


def plot_calibration(df: pd.DataFrame, names: list[str], mover_result_col: str = 'mover_result',
                      positive_class: str | None = None, n_bins: int = 10, ylim: tuple[float, float] = (0, 1),
                      title: str | None = None, figsize: tuple[float, float] | None = None,
                      display_names: dict[str, str] | None = None,
                      elo_bins: tuple[int, int] | None = None, elo_bin_cfg: EloBinConfig = ELO_BINS,
                      color_overrides: list[str | None] | None = None,
                      linestyle_overrides: list[str | None] | None = None) -> None:
    """Plots reliability diagrams (mean predicted probability vs observed frequency, equal-width bins) across models, as one row or two elo-bin-restricted rows."""
    class_to_result = {'win': 1.0, 'loss': 0.0, 'draw': 0.5}
    if positive_class is not None and positive_class not in class_to_result:
        raise ValueError(f'positive_class must be one of {list(class_to_result)}, got {positive_class!r}')
    classes_to_plot = [positive_class] if positive_class is not None else ['win', 'draw', 'loss']
    cols_needed = [f'{name}_prob_{cls}' for name in names for cls in classes_to_plot] + [mover_result_col]
    if elo_bins is not None:
        cols_needed += ['mover_elo', 'opponent_elo']
    df = restrict_to_common_rows(df, cols_needed)
    if positive_class is None:
        draw_cols = [f'{name}_prob_draw' for name in names]
        if (df[draw_cols] == 0).all(axis=None):
            classes_to_plot = ['win', 'loss']
            print('plot_calibration: all draw probabilities are 0 across every model -- skipping draw panel.')
    bin_edges = np.linspace(0, 1, n_bins + 1)
    row_data = [(df, None)] if elo_bins is None else list(zip(*_split_by_elo_bin(df, elo_bins, elo_bin_cfg)))
    panel_width, panel_height = figsize or BASE_FIGSIZE
    display_names, colors, linestyles = _resolve_plot_style(names, display_names, color_overrides, linestyle_overrides)
    fig, axes = plt.subplots(len(row_data), len(classes_to_plot),
                              figsize=(panel_width * len(classes_to_plot), panel_height * len(row_data)), squeeze=False)
    if title:
        fig.suptitle(title)
    for row_idx, (row_df, panel_label) in enumerate(row_data):
        for col_idx, cls in enumerate(classes_to_plot):
            panel_title = f'{cls.capitalize()} calibration' + (f' -- {panel_label}' if panel_label is not None else '')
            _draw_calibration_panel(axes[row_idx][col_idx], row_df, names, mover_result_col, cls,
                                     class_to_result, bin_edges, n_bins, colors, linestyles, ylim,
                                     panel_title, display_names)
    plt.tight_layout(); plt.show()

# (k) AGREEMENT

def plot_agreement_rate(df: pd.DataFrame, names: list[str], baseline: str,
                         ylim: tuple[float, float] = (0, 100),
                         title: str | None = None,
                         figsize: tuple[float, float] | None = None,
                         display_names: dict[str, str] | None = None,
                         color_overrides: list[str | None] | None = None) -> None:
    """Plots each model's predicted_class agreement rate (%) with baseline's predicted_class, as one bar per model."""
    cols_needed = [f'{name}_predicted_class' for name in names] + [f'{baseline}_predicted_class']
    df = restrict_to_common_rows(df, cols_needed)

    display_names, colors, _ = _resolve_plot_style(names, display_names, color_overrides)
    agreement_pct = [_agreement_mask(df, name, baseline).mean() * 100 for name in names]
    bar_labels = [resolve_display_name(name, display_names) for name in names]
    bar_colors = [colors[name] for name in names]

    fig, ax = plt.subplots(figsize=figsize or BASE_FIGSIZE)
    ax.bar(bar_labels, agreement_pct, color=bar_colors)
    ax.set_ylabel('Agreement rate (%)')
    ax.set_title(title or f'Agreement rate with {resolve_display_name(baseline, display_names)}')
    ax.set_ylim(*ylim)

    plt.tight_layout()
    plt.show()


def _draw_agreement_rate_panel(ax: plt.Axes, df: pd.DataFrame, names: list[str], baseline: str,
                                bin_series: pd.Series, bin_values: list, x_labels: list,
                                colors: dict[str, str], linestyles: dict[str, str], ylim: tuple[float, float],
                                panel_title: str, display_names: dict[str, str] | None) -> None:
    """Draws one agreement-rate-by-group panel (one line per model) onto ax."""
    for name in names:
        agree = _agreement_mask(df, name, baseline)
        y = [agree[bin_series == b].mean() * 100 if (bin_series == b).sum() else float('nan') for b in bin_values]
        ax.plot(x_labels, y, marker='D', markersize=6, markeredgecolor='white', markeredgewidth=0.6,
                linewidth=2, linestyle=linestyles[name], color=colors[name],
                label=resolve_display_name(name, display_names))
    ax.set_ylabel('Agreement rate (%)')
    ax.set_title(panel_title)
    ax.legend()
    ax.set_ylim(*ylim)


def plot_agreement_rate_by_elo_bin(df: pd.DataFrame, names: list[str], baseline: str,
                                    cfg: EloBinConfig = ELO_BINS,
                                    ylim: tuple[float, float] = (0, 100),
                                    title: str | None = None,
                                    show_hist: bool = False,
                                    figsize: tuple[float, float] | None = None,
                                    display_names: dict[str, str] | None = None,
                                    color_overrides: list[str | None] | None = None,
                                    linestyle_overrides: list[str | None] | None = None) -> None:
    """Plots each model's predicted_class agreement rate (%) with baseline's predicted_class, binned by mean elo."""
    cols_needed = ([f'{name}_predicted_class' for name in names]
                   + [f'{baseline}_predicted_class', 'mover_elo', 'opponent_elo'])
    df = restrict_to_common_rows(df, cols_needed)

    binned, edges = elo_bin_by_mover(df, cfg, method='mean')
    labels = elo_bin_labels(edges)
    bin_values = list(range(len(edges) - 1))
    display_names, colors, linestyles = _resolve_plot_style(names, display_names, color_overrides, linestyle_overrides)

    fig, ax = plt.subplots(figsize=figsize or BASE_FIGSIZE)
    _draw_agreement_rate_panel(ax, binned, names, baseline, binned['elo_bin'], bin_values, labels,
                                colors, linestyles, ylim, title or f'Agreement rate with {resolve_display_name(baseline, display_names)} by Elo bin',
                                display_names)
    ax.set_xlabel('Elo bin (mean)')
    plt.xticks(rotation=45)

    if show_hist:
        _add_hist_twin(ax, binned['elo_bin'], bin_values, labels, bar_width=0.9)

    plt.tight_layout()
    plt.show()


def plot_agreement_rate_by_combined_clock(df: pd.DataFrame, names: list[str], baseline: str,
                                           mover_clock_col: str = 'mover_clock',
                                           opponent_clock_col: str = 'opponent_clock',
                                           time_control_col: str = 'time_control',
                                           bin_width: float = 0.05,
                                           ylim: tuple[float, float] = (0, 100),
                                           title: str | None = None,
                                           figsize: tuple[float, float] | None = None,
                                           display_names: dict[str, str] | None = None,
                                           color_overrides: list[str | None] | None = None,
                                           linestyle_overrides: list[str | None] | None = None) -> None:
    """Plots each model's agreement rate (%) with baseline, binned by proportion of clock time remaining."""
    cols_needed = ([f'{name}_predicted_class' for name in names]
                   + [f'{baseline}_predicted_class', mover_clock_col, opponent_clock_col, time_control_col])
    df = restrict_to_common_rows(df, cols_needed)

    game_time = df[time_control_col].map(SEC_MAPPING)
    df = df[game_time.notna()]
    game_time = game_time[game_time.notna()]
    df = df.copy()
    time_left_prop = ((df[mover_clock_col] + df[opponent_clock_col]) / (2 * game_time)).clip(0, 1)
    df['_time_bin'] = ((time_left_prop // bin_width) * bin_width + bin_width / 2).round(4)

    bin_values = sorted(df['_time_bin'].dropna().unique())
    display_names, colors, linestyles = _resolve_plot_style(names, display_names, color_overrides, linestyle_overrides)

    fig, ax = plt.subplots(figsize=figsize or BASE_FIGSIZE)
    _draw_agreement_rate_panel(ax, df, names, baseline, df['_time_bin'], bin_values, bin_values, colors, linestyles, ylim,
                                title or f'Agreement rate with {resolve_display_name(baseline, display_names)} by time remaining',
                                display_names)
    ax.set_xlabel('Proportion of total time remaining (both players summed)')
    ax.invert_xaxis()

    plt.tight_layout()
    plt.show()


def plot_agreement_rate_by_clock_ratio(df: pd.DataFrame, names: list[str], baseline: str,
                                        mover_clock_col: str = 'mover_clock',
                                        opponent_clock_col: str = 'opponent_clock',
                                        step_multiplier: float = 2.0,
                                        n_bins_per_side: int = 3,
                                        max_clock_seconds: int | None = None,
                                        ylim: tuple[float, float] = (0, 100),
                                        title: str | None = None,
                                        figsize: tuple[float, float] | None = None,
                                        display_names: dict[str, str] | None = None,
                                        color_overrides: list[str | None] | None = None,
                                        linestyle_overrides: list[str | None] | None = None) -> None:
    """Plots each model's agreement rate (%) with baseline, binned by opponent/mover clock ratio."""
    cols_needed = ([f'{name}_predicted_class' for name in names]
                   + [f'{baseline}_predicted_class', mover_clock_col, opponent_clock_col])
    df = restrict_to_common_rows(df, cols_needed)

    if max_clock_seconds is not None:
        df = df[(df[mover_clock_col] <= max_clock_seconds) & (df[opponent_clock_col] <= max_clock_seconds)]
    both_zero = (df[mover_clock_col] == 0) & (df[opponent_clock_col] == 0)
    df = df[~both_zero]

    with np.errstate(divide='ignore', invalid='ignore'):
        ratio = df[opponent_clock_col] / df[mover_clock_col]
        log_ratio = np.log(ratio) / np.log(step_multiplier)

    edges = _log_ratio_bin_edges(n_bins_per_side)
    labels = _log_ratio_bin_labels(edges, step_multiplier)
    df = df.copy()
    df['_ratio_bin'] = pd.cut(log_ratio, bins=edges, labels=False, right=True, include_lowest=True)
    bin_values = list(range(len(labels)))
    display_names, colors, linestyles = _resolve_plot_style(names, display_names, color_overrides, linestyle_overrides)

    fig, ax = plt.subplots(figsize=figsize or BASE_FIGSIZE)
    _draw_agreement_rate_panel(ax, df, names, baseline, df['_ratio_bin'], bin_values, labels, colors, linestyles, ylim,
                                title or f'Agreement rate with {resolve_display_name(baseline, display_names)} by clock ratio',
                                display_names)
    ax.set_xlabel('Opponent clock / mover clock')
    ax.tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.show()


def plot_agreement_rate_by_phase(df: pd.DataFrame, names: list[str], baseline: str,
                                  phase_col: str = 'phase', bin_width: int = 16,
                                  ylim: tuple[float, float] = (0, 100),
                                  title: str | None = None,
                                  figsize: tuple[float, float] | None = None,
                                  display_names: dict[str, str] | None = None,
                                  color_overrides: list[str | None] | None = None,
                                  linestyle_overrides: list[str | None] | None = None) -> None:
    """Plots each model's agreement rate (%) with baseline, binned by continuous game phase score."""
    if phase_col not in df.columns:
        raise ValueError(f'{phase_col!r} not found in df -- run add_material_and_phase_cols(df) first.')

    cols_needed = [f'{name}_predicted_class' for name in names] + [f'{baseline}_predicted_class', phase_col]
    df = restrict_to_common_rows(df, cols_needed)
    df = df.copy()
    df['_phase'] = (df[phase_col] // bin_width) * bin_width + bin_width // 2

    bin_values = sorted(df['_phase'].dropna().unique())
    display_names, colors, linestyles = _resolve_plot_style(names, display_names, color_overrides, linestyle_overrides)

    fig, ax = plt.subplots(figsize=figsize or BASE_FIGSIZE)
    _draw_agreement_rate_panel(ax, df, names, baseline, df['_phase'], bin_values, bin_values, colors, linestyles, ylim,
                                title or f'Agreement rate with {resolve_display_name(baseline, display_names)} by game phase',
                                display_names)
    ax.set_xlabel('Game phase (0 = opening/full material, 256 = bare endgame)')

    plt.tight_layout()
    plt.show()


def plot_agreement_rate_by_phase_simple(df: pd.DataFrame, names: list[str], baseline: str,
                                         phase_col: str = 'phase',
                                         ylim: tuple[float, float] = (0, 100),
                                         title: str | None = None,
                                         figsize: tuple[float, float] | None = None,
                                         display_names: dict[str, str] | None = None,
                                         color_overrides: list[str | None] | None = None,
                                         linestyle_overrides: list[str | None] | None = None) -> None:
    """Plots each model's agreement rate (%) with baseline, grouped into opening/middlegame/endgame."""
    if phase_col not in df.columns:
        raise ValueError(f'{phase_col!r} not found in df -- run add_material_and_phase_cols(df) first.')

    cols_needed = [f'{name}_predicted_class' for name in names] + [f'{baseline}_predicted_class', phase_col]
    df = restrict_to_common_rows(df, cols_needed)
    df = df.copy()
    df['_phase_label'] = df[phase_col].apply(phase_label)

    display_names, colors, linestyles = _resolve_plot_style(names, display_names, color_overrides, linestyle_overrides)
    fig, ax = plt.subplots(figsize=figsize or BASE_FIGSIZE)
    _draw_agreement_rate_panel(ax, df, names, baseline, df['_phase_label'], PHASE_LABEL_ORDER, PHASE_LABEL_ORDER,
                                colors, linestyles, ylim, title or f'Agreement rate with {resolve_display_name(baseline, display_names)} by simplified game phase',
                                display_names)
    ax.set_xlabel('Game phase')

    plt.tight_layout()
    plt.show()


def plot_accuracy_by_agreement(df: pd.DataFrame, names: list[str], baseline: str,
                                mover_result_col: str = 'mover_result',
                                ylim: tuple[float, float] = (0, 100),
                                title: str | None = None,
                                figsize: tuple[float, float] | None = None,
                                display_names: dict[str, str] | None = None) -> None:
    """Plots each model's accuracy against true mover_result, split into agreeing-with-baseline vs disagreeing-with-baseline bars."""
    cols_needed = ([f'{name}_predicted_class' for name in names]
                   + [f'{baseline}_predicted_class', mover_result_col])
    df = restrict_to_common_rows(df, cols_needed)
    display_names = DEFAULT_DISPLAY_NAMES if display_names is None else display_names

    true_class = _true_class_series(df, mover_result_col)

    agree_acc, disagree_acc = [], []
    for name in names:
        agree_mask = _agreement_mask(df, name, baseline)
        correct = df[f'{name}_predicted_class'] == true_class
        agree_acc.append(correct[agree_mask].mean() * 100 if agree_mask.sum() else float('nan'))
        disagree_acc.append(correct[~agree_mask].mean() * 100 if (~agree_mask).sum() else float('nan'))

    bar_labels = [resolve_display_name(name, display_names) for name in names]
    x = np.arange(len(names))
    width = 0.35

    fig, ax = plt.subplots(figsize=figsize or BASE_FIGSIZE)
    ax.bar(x - width / 2, agree_acc, width, color='tab:green',
           label=f'Agrees with {resolve_display_name(baseline, display_names)}')
    ax.bar(x + width / 2, disagree_acc, width, color='tab:red',
           label=f'Disagrees with {resolve_display_name(baseline, display_names)}')

    ax.set_xticks(x)
    ax.set_xticklabels(bar_labels)
    ax.set_ylabel('Accuracy (%)')
    ax.set_title(title or f'Accuracy by agreement with {resolve_display_name(baseline, display_names)}')
    ax.legend()
    ax.set_ylim(*ylim)

    plt.tight_layout()
    plt.show()


def _draw_accuracy_by_agreement_panel(ax: plt.Axes, df: pd.DataFrame, names: list[str], baseline: str,
                                       mover_result_col: str, bin_series: pd.Series, bin_values: list,
                                       x_labels: list, colors: dict[str, str], ylim: tuple[float, float],
                                       panel_title: str, display_names: dict[str, str] | None) -> None:
    """Draws one accuracy-by-agreement-by-group panel (agree solid, disagree dashed, one pair per model) onto ax."""
    true_class = _true_class_series(df, mover_result_col)
    for name in names:
        pred_class = df[f'{name}_predicted_class']
        correct = pred_class == true_class
        agree_mask = _agreement_mask(df, name, baseline)

        for condition_mask, condition_label, linestyle in [(agree_mask, 'agree', '-'), (~agree_mask, 'disagree', '--')]:
            y = []
            for b in bin_values:
                mask = (bin_series == b) & condition_mask
                y.append(correct[mask].mean() * 100 if mask.sum() else float('nan'))
            ax.plot(x_labels, y, marker='D', markersize=6, markeredgecolor='white', markeredgewidth=0.6,
                    linewidth=2, linestyle=linestyle, color=colors[name],
                    label=f'{resolve_display_name(name, display_names)} ({condition_label})')

    ax.set_ylabel('Accuracy (%)')
    ax.set_title(panel_title)
    ax.legend()
    ax.set_ylim(*ylim)


def plot_accuracy_by_agreement_elo_bin(df: pd.DataFrame, names: list[str], baseline: str,
                                        mover_result_col: str = 'mover_result',
                                        cfg: EloBinConfig = ELO_BINS,
                                        ylim: tuple[float, float] = (0, 100),
                                        title: str | None = None,
                                        figsize: tuple[float, float] | None = None,
                                        display_names: dict[str, str] | None = None,
                                        color_overrides: list[str | None] | None = None) -> None:
    """Plots each model's accuracy against true mover_result, split by agreement with baseline, binned by mean elo. Linestyle already encodes agree/disagree here, so only colour is overridable."""
    cols_needed = ([f'{name}_predicted_class' for name in names]
                   + [f'{baseline}_predicted_class', mover_result_col, 'mover_elo', 'opponent_elo'])
    df = restrict_to_common_rows(df, cols_needed)

    binned, edges = elo_bin_by_mover(df, cfg, method='mean')
    labels = elo_bin_labels(edges)
    bin_values = list(range(len(edges) - 1))
    display_names, colors, _ = _resolve_plot_style(names, display_names, color_overrides)

    fig, ax = plt.subplots(figsize=figsize or BASE_FIGSIZE)
    _draw_accuracy_by_agreement_panel(ax, binned, names, baseline, mover_result_col, binned['elo_bin'], bin_values,
                                       labels, colors, ylim,
                                       title or f'Accuracy by agreement with {resolve_display_name(baseline, display_names)}, by Elo bin',
                                       display_names)
    ax.set_xlabel('Elo bin (mean)')
    plt.xticks(rotation=45)

    plt.tight_layout()
    plt.show()


def plot_accuracy_by_agreement_by_combined_clock(df: pd.DataFrame, names: list[str], baseline: str,
                                                  mover_clock_col: str = 'mover_clock',
                                                  opponent_clock_col: str = 'opponent_clock',
                                                  time_control_col: str = 'time_control',
                                                  mover_result_col: str = 'mover_result',
                                                  bin_width: float = 0.05,
                                                  ylim: tuple[float, float] = (0, 100),
                                                  title: str | None = None,
                                                  figsize: tuple[float, float] | None = None,
                                                  display_names: dict[str, str] | None = None,
                                                  color_overrides: list[str | None] | None = None) -> None:
    """Plots each model's accuracy, split by agreement with baseline, binned by proportion of clock time remaining. Linestyle already encodes agree/disagree here, so only colour is overridable."""
    cols_needed = ([f'{name}_predicted_class' for name in names]
                   + [f'{baseline}_predicted_class', mover_result_col, mover_clock_col, opponent_clock_col, time_control_col])
    df = restrict_to_common_rows(df, cols_needed)

    game_time = df[time_control_col].map(SEC_MAPPING)
    df = df[game_time.notna()]
    game_time = game_time[game_time.notna()]
    df = df.copy()
    time_left_prop = ((df[mover_clock_col] + df[opponent_clock_col]) / (2 * game_time)).clip(0, 1)
    df['_time_bin'] = ((time_left_prop // bin_width) * bin_width + bin_width / 2).round(4)

    bin_values = sorted(df['_time_bin'].dropna().unique())
    display_names, colors, _ = _resolve_plot_style(names, display_names, color_overrides)

    fig, ax = plt.subplots(figsize=figsize or BASE_FIGSIZE)
    _draw_accuracy_by_agreement_panel(ax, df, names, baseline, mover_result_col, df['_time_bin'], bin_values,
                                       bin_values, colors, ylim,
                                       title or f'Accuracy by agreement with {resolve_display_name(baseline, display_names)}, by time remaining',
                                       display_names)
    ax.set_xlabel('Proportion of total time remaining (both players summed)')
    ax.invert_xaxis()

    plt.tight_layout()
    plt.show()


def plot_accuracy_by_agreement_by_clock_ratio(df: pd.DataFrame, names: list[str], baseline: str,
                                               mover_clock_col: str = 'mover_clock',
                                               opponent_clock_col: str = 'opponent_clock',
                                               mover_result_col: str = 'mover_result',
                                               step_multiplier: float = 2.0,
                                               n_bins_per_side: int = 3,
                                               max_clock_seconds: int | None = None,
                                               ylim: tuple[float, float] = (0, 100),
                                               title: str | None = None,
                                               figsize: tuple[float, float] | None = None,
                                               display_names: dict[str, str] | None = None,
                                               color_overrides: list[str | None] | None = None) -> None:
    """Plots each model's accuracy, split by agreement with baseline, binned by opponent/mover clock ratio. Linestyle already encodes agree/disagree here, so only colour is overridable."""
    cols_needed = ([f'{name}_predicted_class' for name in names]
                   + [f'{baseline}_predicted_class', mover_result_col, mover_clock_col, opponent_clock_col])
    df = restrict_to_common_rows(df, cols_needed)

    if max_clock_seconds is not None:
        df = df[(df[mover_clock_col] <= max_clock_seconds) & (df[opponent_clock_col] <= max_clock_seconds)]
    both_zero = (df[mover_clock_col] == 0) & (df[opponent_clock_col] == 0)
    df = df[~both_zero]

    with np.errstate(divide='ignore', invalid='ignore'):
        ratio = df[opponent_clock_col] / df[mover_clock_col]
        log_ratio = np.log(ratio) / np.log(step_multiplier)

    edges = _log_ratio_bin_edges(n_bins_per_side)
    labels = _log_ratio_bin_labels(edges, step_multiplier)
    df = df.copy()
    df['_ratio_bin'] = pd.cut(log_ratio, bins=edges, labels=False, right=True, include_lowest=True)
    bin_values = list(range(len(labels)))
    display_names, colors, _ = _resolve_plot_style(names, display_names, color_overrides)

    fig, ax = plt.subplots(figsize=figsize or BASE_FIGSIZE)
    _draw_accuracy_by_agreement_panel(ax, df, names, baseline, mover_result_col, df['_ratio_bin'], bin_values,
                                       labels, colors, ylim,
                                       title or f'Accuracy by agreement with {resolve_display_name(baseline, display_names)}, by clock ratio',
                                       display_names)
    ax.set_xlabel('Opponent clock / mover clock')
    ax.tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.show()


def plot_accuracy_by_agreement_by_phase(df: pd.DataFrame, names: list[str], baseline: str,
                                         phase_col: str = 'phase', bin_width: int = 16,
                                         mover_result_col: str = 'mover_result',
                                         ylim: tuple[float, float] = (0, 100),
                                         title: str | None = None,
                                         figsize: tuple[float, float] | None = None,
                                         display_names: dict[str, str] | None = None,
                                         color_overrides: list[str | None] | None = None) -> None:
    """Plots each model's accuracy, split by agreement with baseline, binned by continuous game phase score. Linestyle already encodes agree/disagree here, so only colour is overridable."""
    if phase_col not in df.columns:
        raise ValueError(f'{phase_col!r} not found in df -- run add_material_and_phase_cols(df) first.')

    cols_needed = ([f'{name}_predicted_class' for name in names]
                   + [f'{baseline}_predicted_class', mover_result_col, phase_col])
    df = restrict_to_common_rows(df, cols_needed)
    df = df.copy()
    df['_phase'] = (df[phase_col] // bin_width) * bin_width + bin_width // 2

    bin_values = sorted(df['_phase'].dropna().unique())
    display_names, colors, _ = _resolve_plot_style(names, display_names, color_overrides)

    fig, ax = plt.subplots(figsize=figsize or BASE_FIGSIZE)
    _draw_accuracy_by_agreement_panel(ax, df, names, baseline, mover_result_col, df['_phase'], bin_values,
                                       bin_values, colors, ylim,
                                       title or f'Accuracy by agreement with {resolve_display_name(baseline, display_names)}, by game phase',
                                       display_names)
    ax.set_xlabel('Game phase (0 = opening/full material, 256 = bare endgame)')

    plt.tight_layout()
    plt.show()


def plot_accuracy_by_agreement_by_phase_simple(df: pd.DataFrame, names: list[str], baseline: str,
                                                phase_col: str = 'phase',
                                                mover_result_col: str = 'mover_result',
                                                ylim: tuple[float, float] = (0, 100),
                                                title: str | None = None,
                                                figsize: tuple[float, float] | None = None,
                                                display_names: dict[str, str] | None = None,
                                                color_overrides: list[str | None] | None = None) -> None:
    """Plots each model's accuracy, split by agreement with baseline, grouped into opening/middlegame/endgame. Linestyle already encodes agree/disagree here, so only colour is overridable."""
    if phase_col not in df.columns:
        raise ValueError(f'{phase_col!r} not found in df -- run add_material_and_phase_cols(df) first.')

    cols_needed = ([f'{name}_predicted_class' for name in names]
                   + [f'{baseline}_predicted_class', mover_result_col, phase_col])
    df = restrict_to_common_rows(df, cols_needed)
    df = df.copy()
    df['_phase_label'] = df[phase_col].apply(phase_label)

    display_names, colors, _ = _resolve_plot_style(names, display_names, color_overrides)
    fig, ax = plt.subplots(figsize=figsize or BASE_FIGSIZE)
    _draw_accuracy_by_agreement_panel(ax, df, names, baseline, mover_result_col, df['_phase_label'],
                                       PHASE_LABEL_ORDER, PHASE_LABEL_ORDER, colors, ylim,
                                       title or f'Accuracy by agreement with {resolve_display_name(baseline, display_names)}, by simplified game phase',
                                       display_names)
    ax.set_xlabel('Game phase')

    plt.tight_layout()
    plt.show()


# (l) ACCURACY BY MATERIAL / PHASE

def plot_accuracy_by_material(df: pd.DataFrame, names: list[str],
                               material_col: str = 'material',
                               mover_result_col: str = 'mover_result',
                               bin_width: int = 4,
                               min_material: int | None = None,
                               ylim: tuple[float, float] = (0, 80),
                               title: str = 'Result-prediction accuracy by total material on board',
                               show_hist: bool = False,
                               show_brier: bool = False,
                               figsize: tuple[float, float] | None = None,
                               display_names: dict[str, str] | None = None,
                               elo_bins: tuple[int, int] | None = None,
                               elo_bin_cfg: EloBinConfig = ELO_BINS,
                               color_overrides: list[str | None] | None = None,
                               linestyle_overrides: list[str | None] | None = None) -> None:
    """Plots per-model accuracy (and optionally Brier score) against true mover_result, binned by total material on board (descending: max material left, 0 right, from a precomputed material_col), as one row or two elo-bin-restricted rows."""
    if material_col not in df.columns:
        raise ValueError(f'{material_col!r} not found in df -- run add_material_and_phase_cols(df) first.')

    cols_needed = [f'{name}_predicted_class' for name in names] + [mover_result_col, material_col]
    if show_brier:
        cols_needed += [f'{name}_prob_{c}' for name in names for c in ('win', 'draw', 'loss')]
    if elo_bins is not None:
        cols_needed += ['mover_elo', 'opponent_elo']
    df = restrict_to_common_rows(df, cols_needed)
    df = df.copy()

    material_raw = df[material_col]
    if min_material is not None:
        n_before = len(df)
        keep = material_raw >= min_material
        df, material_raw = df[keep], material_raw[keep]
        print(f'plot_accuracy_by_material: min_material={min_material} kept {len(df):,} of {n_before:,} rows.')

    df['_material'] = (material_raw // bin_width) * bin_width + bin_width // 2

    display_names, colors, linestyles = _resolve_plot_style(names, display_names, color_overrides, linestyle_overrides)
    panel_width, panel_height = figsize or BASE_FIGSIZE
    subsets, row_labels = ([df], [None]) if elo_bins is None else _split_by_elo_bin(df, elo_bins, elo_bin_cfg)
    n_rows, n_cols = len(subsets), (2 if show_brier else 1)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(panel_width * n_cols, panel_height * n_rows), squeeze=False)

    for row_idx, (sub_df, row_label) in enumerate(zip(subsets, row_labels)):
        bin_series = sub_df['_material']
        bin_values = sorted(bin_series.dropna().unique())
        row_title = title if row_label is None else f'{title} -- {row_label}'

        ax_acc = axes[row_idx][0]
        _binned_line_plot(ax_acc, sub_df, names, mover_result_col, bin_series, bin_values, bin_values,
                           colors, linestyles, row_title, display_names, metric='accuracy', ylim=ylim)
        ax_acc.set_xlabel('Total material on board (both sides)')
        ax_acc.invert_xaxis()
        if show_hist:
            _add_hist_twin(ax_acc, bin_series, bin_values, bin_values, bar_width=bin_width * 0.9)

        if show_brier:
            ax_brier = axes[row_idx][1]
            _binned_line_plot(ax_brier, sub_df, names, mover_result_col, bin_series, bin_values, bin_values,
                               colors, linestyles, f'{row_title} (Brier)', display_names, metric='brier')
            ax_brier.set_xlabel('Total material on board (both sides)')
            ax_brier.invert_xaxis()

    plt.tight_layout()
    plt.show()


def plot_confidence_by_material(df: pd.DataFrame, names: list[str],
                                 material_col: str = 'material',
                                 bin_width: int = 4,
                                 min_material: int | None = None,
                                 title: str = 'Mean predicted confidence by total material on board',
                                 figsize: tuple[float, float] | None = None,
                                 display_names: dict[str, str] | None = None,
                                 color_overrides: list[str | None] | None = None,
                                 linestyle_overrides: list[str | None] | None = None) -> None:
    """Plots each model's mean max predicted probability, binned by total material on board (descending axis)."""
    if material_col not in df.columns:
        raise ValueError(f'{material_col!r} not found in df -- run add_material_and_phase_cols(df) first.')

    cols_needed = [f'{name}_prob_{c}' for name in names for c in ('win', 'draw', 'loss')] + [material_col]
    df = restrict_to_common_rows(df, cols_needed)
    df = df.copy()

    material_raw = df[material_col]
    if min_material is not None:
        df = df[material_raw >= min_material]
        material_raw = material_raw[material_raw >= min_material]

    df['_material'] = (material_raw // bin_width) * bin_width + bin_width // 2
    bin_values = sorted(df['_material'].dropna().unique())
    display_names, colors, linestyles = _resolve_plot_style(names, display_names, color_overrides, linestyle_overrides)

    fig, ax = plt.subplots(figsize=figsize or BASE_FIGSIZE)
    _binned_line_plot(ax, df, names, None, df['_material'], bin_values, bin_values,
                       colors, linestyles, title, display_names, metric='confidence')
    ax.set_xlabel('Total material on board (both sides)')
    ax.invert_xaxis()

    plt.tight_layout()
    plt.show()


def plot_accuracy_by_material_diff(df: pd.DataFrame, names: list[str],
                                    fen_col: str = 'fen',
                                    mover_result_col: str = 'mover_result',
                                    bin_width: int = 2,
                                    max_abs_diff: int | None = None,
                                    ylim: tuple[float, float] = (0, 100),
                                    title: str = 'Result-prediction accuracy by material difference',
                                    show_hist: bool = False,
                                    show_brier: bool = False,
                                    figsize: tuple[float, float] | None = None,
                                    display_names: dict[str, str] | None = None,
                                    elo_bins: tuple[int, int] | None = None,
                                    elo_bin_cfg: EloBinConfig = ELO_BINS,
                                    color_overrides: list[str | None] | None = None,
                                    linestyle_overrides: list[str | None] | None = None) -> None:
    """Plots per-model accuracy (and optionally Brier score) against true mover_result, binned by mover-minus-opponent material, as one row or two elo-bin-restricted rows."""
    cols_needed = [f'{name}_predicted_class' for name in names] + [mover_result_col, fen_col]
    if show_brier:
        cols_needed += [f'{name}_prob_{c}' for name in names for c in ('win', 'draw', 'loss')]
    if elo_bins is not None:
        cols_needed += ['mover_elo', 'opponent_elo']
    df = restrict_to_common_rows(df, cols_needed)
    df = df.copy()

    diff_raw = df[fen_col].apply(material_diff)
    if max_abs_diff is not None:
        n_before = len(df)
        keep = diff_raw.abs() <= max_abs_diff
        df, diff_raw = df[keep], diff_raw[keep]
        print(f'plot_accuracy_by_material_diff: max_abs_diff={max_abs_diff} kept {len(df):,} of {n_before:,} rows.')

    df['_material_diff'] = (diff_raw // bin_width) * bin_width + bin_width // 2

    display_names, colors, linestyles = _resolve_plot_style(names, display_names, color_overrides, linestyle_overrides)
    panel_width, panel_height = figsize or BASE_FIGSIZE
    subsets, row_labels = ([df], [None]) if elo_bins is None else _split_by_elo_bin(df, elo_bins, elo_bin_cfg)
    n_rows, n_cols = len(subsets), (2 if show_brier else 1)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(panel_width * n_cols, panel_height * n_rows), squeeze=False)

    for row_idx, (sub_df, row_label) in enumerate(zip(subsets, row_labels)):
        bin_series = sub_df['_material_diff']
        bin_values = sorted(bin_series.dropna().unique())
        row_title = title if row_label is None else f'{title} -- {row_label}'

        ax_acc = axes[row_idx][0]
        _binned_line_plot(ax_acc, sub_df, names, mover_result_col, bin_series, bin_values, bin_values,
                           colors, linestyles, row_title, display_names, metric='accuracy', ylim=ylim)
        ax_acc.set_xlabel('Material difference, mover minus opponent')
        if show_hist:
            _add_hist_twin(ax_acc, bin_series, bin_values, bin_values, bar_width=bin_width * 0.9)

        if show_brier:
            ax_brier = axes[row_idx][1]
            _binned_line_plot(ax_brier, sub_df, names, mover_result_col, bin_series, bin_values, bin_values,
                               colors, linestyles, f'{row_title} (Brier)', display_names, metric='brier')
            ax_brier.set_xlabel('Material difference, mover minus opponent')

    plt.tight_layout()
    plt.show()


def plot_confidence_by_material_diff(df: pd.DataFrame, names: list[str],
                                      fen_col: str = 'fen',
                                      bin_width: int = 2,
                                      max_abs_diff: int | None = None,
                                      title: str = 'Mean predicted confidence by material difference',
                                      figsize: tuple[float, float] | None = None,
                                      display_names: dict[str, str] | None = None,
                                      color_overrides: list[str | None] | None = None,
                                      linestyle_overrides: list[str | None] | None = None) -> None:
    """Plots each model's mean max predicted probability, binned by mover-minus-opponent material."""
    cols_needed = [f'{name}_prob_{c}' for name in names for c in ('win', 'draw', 'loss')] + [fen_col]
    df = restrict_to_common_rows(df, cols_needed)
    df = df.copy()

    diff_raw = df[fen_col].apply(material_diff)
    if max_abs_diff is not None:
        keep = diff_raw.abs() <= max_abs_diff
        df, diff_raw = df[keep], diff_raw[keep]

    df['_material_diff'] = (diff_raw // bin_width) * bin_width + bin_width // 2
    bin_values = sorted(df['_material_diff'].dropna().unique())
    display_names, colors, linestyles = _resolve_plot_style(names, display_names, color_overrides, linestyle_overrides)

    fig, ax = plt.subplots(figsize=figsize or BASE_FIGSIZE)
    _binned_line_plot(ax, df, names, None, df['_material_diff'], bin_values, bin_values,
                       colors, linestyles, title, display_names, metric='confidence')
    ax.set_xlabel('Material difference, mover minus opponent')

    plt.tight_layout()
    plt.show()


def plot_accuracy_by_phase(df: pd.DataFrame, names: list[str],
                            phase_col: str = 'phase',
                            mover_result_col: str = 'mover_result',
                            bin_width: int = 16,
                            max_phase: int | None = None,
                            ylim: tuple[float, float] = (0, 80),
                            title: str = 'Result-prediction accuracy by game phase',
                            show_hist: bool = False,
                            show_brier: bool = False,
                            figsize: tuple[float, float] | None = None,
                            display_names: dict[str, str] | None = None,
                            elo_bins: tuple[int, int] | None = None,
                            elo_bin_cfg: EloBinConfig = ELO_BINS,
                            color_overrides: list[str | None] | None = None,
                            linestyle_overrides: list[str | None] | None = None) -> None:
    """Plots per-model accuracy (and optionally Brier score) against true mover_result, binned by continuous game phase score, as one row or two elo-bin-restricted rows."""
    if phase_col not in df.columns:
        raise ValueError(f'{phase_col!r} not found in df -- run add_material_and_phase_cols(df) first.')

    cols_needed = [f'{name}_predicted_class' for name in names] + [mover_result_col, phase_col]
    if show_brier:
        cols_needed += [f'{name}_prob_{c}' for name in names for c in ('win', 'draw', 'loss')]
    if elo_bins is not None:
        cols_needed += ['mover_elo', 'opponent_elo']
    df = restrict_to_common_rows(df, cols_needed)
    df = df.copy()

    phase_raw = df[phase_col]
    if max_phase is not None:
        n_before = len(df)
        keep = phase_raw <= max_phase
        df, phase_raw = df[keep], phase_raw[keep]
        print(f'plot_accuracy_by_phase: max_phase={max_phase} kept {len(df):,} of {n_before:,} rows.')

    df['_phase'] = (phase_raw // bin_width) * bin_width + bin_width // 2

    display_names, colors, linestyles = _resolve_plot_style(names, display_names, color_overrides, linestyle_overrides)
    panel_width, panel_height = figsize or BASE_FIGSIZE
    subsets, row_labels = ([df], [None]) if elo_bins is None else _split_by_elo_bin(df, elo_bins, elo_bin_cfg)
    n_rows, n_cols = len(subsets), (2 if show_brier else 1)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(panel_width * n_cols, panel_height * n_rows), squeeze=False)

    for row_idx, (sub_df, row_label) in enumerate(zip(subsets, row_labels)):
        bin_series = sub_df['_phase']
        bin_values = sorted(bin_series.dropna().unique())
        row_title = title if row_label is None else f'{title} -- {row_label}'

        ax_acc = axes[row_idx][0]
        _binned_line_plot(ax_acc, sub_df, names, mover_result_col, bin_series, bin_values, bin_values,
                           colors, linestyles, row_title, display_names, metric='accuracy', ylim=ylim)
        ax_acc.set_xlabel('Game phase (0 = opening/full material, 256 = bare endgame)')
        if show_hist:
            _add_hist_twin(ax_acc, bin_series, bin_values, bin_values, bar_width=bin_width * 0.9)

        if show_brier:
            ax_brier = axes[row_idx][1]
            _binned_line_plot(ax_brier, sub_df, names, mover_result_col, bin_series, bin_values, bin_values,
                               colors, linestyles, f'{row_title} (Brier)', display_names, metric='brier')
            ax_brier.set_xlabel('Game phase (0 = opening/full material, 256 = bare endgame)')

    plt.tight_layout()
    plt.show()


def plot_confidence_by_phase(df: pd.DataFrame, names: list[str],
                              phase_col: str = 'phase',
                              bin_width: int = 16,
                              title: str = 'Mean predicted confidence by game phase',
                              figsize: tuple[float, float] | None = None,
                              display_names: dict[str, str] | None = None,
                              color_overrides: list[str | None] | None = None,
                              linestyle_overrides: list[str | None] | None = None) -> None:
    """Plots each model's mean max predicted probability, binned by continuous game phase score."""
    if phase_col not in df.columns:
        raise ValueError(f'{phase_col!r} not found in df -- run add_material_and_phase_cols(df) first.')

    cols_needed = [f'{name}_prob_{c}' for name in names for c in ('win', 'draw', 'loss')] + [phase_col]
    df = restrict_to_common_rows(df, cols_needed)
    df = df.copy()
    df['_phase'] = (df[phase_col] // bin_width) * bin_width + bin_width // 2
    bin_values = sorted(df['_phase'].dropna().unique())
    display_names, colors, linestyles = _resolve_plot_style(names, display_names, color_overrides, linestyle_overrides)

    fig, ax = plt.subplots(figsize=figsize or BASE_FIGSIZE)
    _binned_line_plot(ax, df, names, None, df['_phase'], bin_values, bin_values,
                       colors, linestyles, title, display_names, metric='confidence')
    ax.set_xlabel('Game phase (0 = opening/full material, 256 = bare endgame)')

    plt.tight_layout()
    plt.show()


def plot_accuracy_by_phase_simple(df: pd.DataFrame, names: list[str],
                                   phase_col: str = 'phase',
                                   mover_result_col: str = 'mover_result',
                                   ylim: tuple[float, float] = (0, 80),
                                   title: str = 'Result-prediction accuracy by simplified game phase',
                                   show_brier: bool = False,
                                   figsize: tuple[float, float] | None = None,
                                   display_names: dict[str, str] | None = None,
                                   color_overrides: list[str | None] | None = None,
                                   linestyle_overrides: list[str | None] | None = None) -> None:
    """Plots per-model accuracy (and optionally Brier score) against true mover_result, grouped into opening/middlegame/endgame (derived from a precomputed phase_col)."""
    if phase_col not in df.columns:
        raise ValueError(f'{phase_col!r} not found in df -- run add_material_and_phase_cols(df) first.')

    cols_needed = [f'{name}_predicted_class' for name in names] + [mover_result_col, phase_col]
    if show_brier:
        cols_needed += [f'{name}_prob_{c}' for name in names for c in ('win', 'draw', 'loss')]
    df = restrict_to_common_rows(df, cols_needed)
    df = df.copy()
    df['_phase_label'] = df[phase_col].apply(phase_label)

    display_names, colors, linestyles = _resolve_plot_style(names, display_names, color_overrides, linestyle_overrides)
    n_cols = 2 if show_brier else 1
    panel_width, panel_height = figsize or BASE_FIGSIZE
    fig, axes = plt.subplots(1, n_cols, figsize=(panel_width * n_cols, panel_height), squeeze=False)

    ax_acc = axes[0][0]
    _binned_line_plot(ax_acc, df, names, mover_result_col, df['_phase_label'], PHASE_LABEL_ORDER, PHASE_LABEL_ORDER,
                       colors, linestyles, title, display_names, metric='accuracy', ylim=ylim)
    ax_acc.set_xlabel('Game phase')

    if show_brier:
        ax_brier = axes[0][1]
        _binned_line_plot(ax_brier, df, names, mover_result_col, df['_phase_label'], PHASE_LABEL_ORDER, PHASE_LABEL_ORDER,
                           colors, linestyles, f'{title} (Brier)', display_names, metric='brier')
        ax_brier.set_xlabel('Game phase')

    plt.tight_layout()
    plt.show()


# (m) ACCURACY BY PAST PERFORMANCE

def plot_accuracy_by_past_performance(df: pd.DataFrame, names: list[str],
                                       past_mover_col: str = 'past_mover',
                                       has_history_mover_col: str = 'has_history_mover',
                                       mover_result_col: str = 'mover_result',
                                       bin_width: float = 0.05,
                                       ylim: tuple[float, float] = (0, 80),
                                       title: str = "Result-prediction accuracy by mover's recent-form score",
                                       show_hist: bool = False,
                                       show_brier: bool = False,
                                       figsize: tuple[float, float] | None = None,
                                       display_names: dict[str, str] | None = None,
                                       elo_bins: tuple[int, int] | None = None,
                                       elo_bin_cfg: EloBinConfig = ELO_BINS,
                                       color_overrides: list[str | None] | None = None,
                                       linestyle_overrides: list[str | None] | None = None) -> None:
    """Plots per-model accuracy (and optionally Brier score) against true mover_result, binned by the mover's recency-weighted past win rate (0 to ~0.965). Rows with no history for the mover are excluded."""
    cols_needed = [f'{name}_predicted_class' for name in names] + [mover_result_col, past_mover_col, has_history_mover_col]
    if show_brier:
        cols_needed += [f'{name}_prob_{c}' for name in names for c in ('win', 'draw', 'loss')]
    if elo_bins is not None:
        cols_needed += ['mover_elo', 'opponent_elo']

    n_before = len(df)
    df = df[df[has_history_mover_col].astype(bool)]
    n_excluded = n_before - len(df)
    if n_excluded:
        print(f'plot_accuracy_by_past_performance: excluded {n_excluded:,} of {n_before:,} rows '
              f'({n_excluded / n_before * 100:.1f}%) -- no prior game history for the mover.')

    df = restrict_to_common_rows(df, cols_needed)
    df = df.copy()
    df['_past_bin'] = ((df[past_mover_col] // bin_width) * bin_width + bin_width / 2).round(4)

    display_names, colors, linestyles = _resolve_plot_style(names, display_names, color_overrides, linestyle_overrides)
    panel_width, panel_height = figsize or BASE_FIGSIZE
    subsets, row_labels = ([df], [None]) if elo_bins is None else _split_by_elo_bin(df, elo_bins, elo_bin_cfg)
    n_rows, n_cols = len(subsets), (2 if show_brier else 1)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(panel_width * n_cols, panel_height * n_rows), squeeze=False)

    for row_idx, (sub_df, row_label) in enumerate(zip(subsets, row_labels)):
        bin_series = sub_df['_past_bin']
        bin_values = sorted(bin_series.dropna().unique())
        row_title = title if row_label is None else f'{title} -- {row_label}'

        ax_acc = axes[row_idx][0]
        _binned_line_plot(ax_acc, sub_df, names, mover_result_col, bin_series, bin_values, bin_values,
                           colors, linestyles, row_title, display_names, metric='accuracy', ylim=ylim)
        ax_acc.set_xlabel("Mover's recency-weighted past win rate")
        if show_hist:
            _add_hist_twin(ax_acc, bin_series, bin_values, bin_values, bar_width=bin_width * 0.9)

        if show_brier:
            ax_brier = axes[row_idx][1]
            _binned_line_plot(ax_brier, sub_df, names, mover_result_col, bin_series, bin_values, bin_values,
                               colors, linestyles, f'{row_title} (Brier)', display_names, metric='brier')
            ax_brier.set_xlabel("Mover's recency-weighted past win rate")

    plt.tight_layout()
    plt.show()


def plot_confidence_by_past_performance(df: pd.DataFrame, names: list[str],
                                         past_mover_col: str = 'past_mover',
                                         has_history_mover_col: str = 'has_history_mover',
                                         bin_width: float = 0.05,
                                         title: str = "Mean predicted confidence by mover's recent-form score",
                                         figsize: tuple[float, float] | None = None,
                                         display_names: dict[str, str] | None = None,
                                         color_overrides: list[str | None] | None = None,
                                         linestyle_overrides: list[str | None] | None = None) -> None:
    """Plots each model's mean max predicted probability, binned by the mover's recency-weighted past win rate."""
    cols_needed = [f'{name}_prob_{c}' for name in names for c in ('win', 'draw', 'loss')] + [past_mover_col, has_history_mover_col]
    df = df[df[has_history_mover_col].astype(bool)]
    df = restrict_to_common_rows(df, cols_needed)
    df = df.copy()
    df['_past_bin'] = ((df[past_mover_col] // bin_width) * bin_width + bin_width / 2).round(4)

    bin_values = sorted(df['_past_bin'].dropna().unique())
    display_names, colors, linestyles = _resolve_plot_style(names, display_names, color_overrides, linestyle_overrides)
    fig, ax = plt.subplots(figsize=figsize or BASE_FIGSIZE)
    _binned_line_plot(ax, df, names, None, df['_past_bin'], bin_values, bin_values,
                       colors, linestyles, title, display_names, metric='confidence')
    ax.set_xlabel("Mover's recency-weighted past win rate")

    plt.tight_layout()
    plt.show()


def plot_accuracy_by_past_performance_diff(df: pd.DataFrame, names: list[str],
                                            past_mover_col: str = 'past_mover',
                                            past_opponent_col: str = 'past_opponent',
                                            has_history_mover_col: str = 'has_history_mover',
                                            has_history_opponent_col: str = 'has_history_opponent',
                                            mover_result_col: str = 'mover_result',
                                            bin_width: float = 0.05,
                                            ylim: tuple[float, float] = (0, 80),
                                            title: str = 'Result-prediction accuracy by past-performance difference',
                                            show_hist: bool = False,
                                            show_brier: bool = False,
                                            figsize: tuple[float, float] | None = None,
                                            display_names: dict[str, str] | None = None,
                                            elo_bins: tuple[int, int] | None = None,
                                            elo_bin_cfg: EloBinConfig = ELO_BINS,
                                            color_overrides: list[str | None] | None = None,
                                            linestyle_overrides: list[str | None] | None = None) -> None:
    """Plots per-model accuracy (and optionally Brier score) against true mover_result, binned by mover-minus-opponent recency-weighted past win rate. Rows with no history for either player are excluded."""
    cols_needed = ([f'{name}_predicted_class' for name in names]
                   + [mover_result_col, past_mover_col, past_opponent_col, has_history_mover_col, has_history_opponent_col])
    if show_brier:
        cols_needed += [f'{name}_prob_{c}' for name in names for c in ('win', 'draw', 'loss')]
    if elo_bins is not None:
        cols_needed += ['mover_elo', 'opponent_elo']

    n_before = len(df)
    df = df[df[has_history_mover_col].astype(bool) & df[has_history_opponent_col].astype(bool)]
    n_excluded = n_before - len(df)
    if n_excluded:
        print(f'plot_accuracy_by_past_performance_diff: excluded {n_excluded:,} of {n_before:,} rows '
              f'({n_excluded / n_before * 100:.1f}%) -- no prior game history for the mover and/or opponent.')

    df = restrict_to_common_rows(df, cols_needed)
    df = df.copy()
    diff = df[past_mover_col] - df[past_opponent_col]
    df['_past_diff_bin'] = ((diff // bin_width) * bin_width + bin_width / 2).round(4)

    display_names, colors, linestyles = _resolve_plot_style(names, display_names, color_overrides, linestyle_overrides)
    panel_width, panel_height = figsize or BASE_FIGSIZE
    subsets, row_labels = ([df], [None]) if elo_bins is None else _split_by_elo_bin(df, elo_bins, elo_bin_cfg)
    n_rows, n_cols = len(subsets), (2 if show_brier else 1)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(panel_width * n_cols, panel_height * n_rows), squeeze=False)

    for row_idx, (sub_df, row_label) in enumerate(zip(subsets, row_labels)):
        bin_series = sub_df['_past_diff_bin']
        bin_values = sorted(bin_series.dropna().unique())
        row_title = title if row_label is None else f'{title} -- {row_label}'

        ax_acc = axes[row_idx][0]
        _binned_line_plot(ax_acc, sub_df, names, mover_result_col, bin_series, bin_values, bin_values,
                           colors, linestyles, row_title, display_names, metric='accuracy', ylim=ylim)
        ax_acc.set_xlabel('Past-performance difference, mover minus opponent')
        if show_hist:
            _add_hist_twin(ax_acc, bin_series, bin_values, bin_values, bar_width=bin_width * 0.9)

        if show_brier:
            ax_brier = axes[row_idx][1]
            _binned_line_plot(ax_brier, sub_df, names, mover_result_col, bin_series, bin_values, bin_values,
                               colors, linestyles, f'{row_title} (Brier)', display_names, metric='brier')
            ax_brier.set_xlabel('Past-performance difference, mover minus opponent')

    plt.tight_layout()
    plt.show()


def plot_confidence_by_past_performance_diff(df: pd.DataFrame, names: list[str],
                                              past_mover_col: str = 'past_mover',
                                              past_opponent_col: str = 'past_opponent',
                                              has_history_mover_col: str = 'has_history_mover',
                                              has_history_opponent_col: str = 'has_history_opponent',
                                              bin_width: float = 0.05,
                                              title: str = 'Mean predicted confidence by past-performance difference',
                                              figsize: tuple[float, float] | None = None,
                                              display_names: dict[str, str] | None = None,
                                              color_overrides: list[str | None] | None = None,
                                              linestyle_overrides: list[str | None] | None = None) -> None:
    """Plots each model's mean max predicted probability, binned by mover-minus-opponent recency-weighted past win rate."""
    cols_needed = ([f'{name}_prob_{c}' for name in names for c in ('win', 'draw', 'loss')]
                   + [past_mover_col, past_opponent_col, has_history_mover_col, has_history_opponent_col])
    df = df[df[has_history_mover_col].astype(bool) & df[has_history_opponent_col].astype(bool)]
    df = restrict_to_common_rows(df, cols_needed)
    df = df.copy()
    diff = df[past_mover_col] - df[past_opponent_col]
    df['_past_diff_bin'] = ((diff // bin_width) * bin_width + bin_width / 2).round(4)

    bin_values = sorted(df['_past_diff_bin'].dropna().unique())
    display_names, colors, linestyles = _resolve_plot_style(names, display_names, color_overrides, linestyle_overrides)
    fig, ax = plt.subplots(figsize=figsize or BASE_FIGSIZE)
    _binned_line_plot(ax, df, names, None, df['_past_diff_bin'], bin_values, bin_values,
                       colors, linestyles, title, display_names, metric='confidence')
    ax.set_xlabel('Past-performance difference, mover minus opponent')

    plt.tight_layout()
    plt.show()


# (n) ACCURACY BY HOURS SINCE LAST GAME

def plot_accuracy_by_hours_since_log_ratio(df: pd.DataFrame, names: list[str],
                                            hours_since_mover_col: str = 'hours_since_mover',
                                            hours_since_opponent_col: str = 'hours_since_opponent',
                                            mover_result_col: str = 'mover_result',
                                            step_multiplier: float = 2.0,
                                            n_bins_per_side: int = 3,
                                            ylim: tuple[float, float] = (0, 80),
                                            title: str = 'Result-prediction accuracy by hours-since-last-game ratio',
                                            show_hist: bool = False,
                                            show_brier: bool = False,
                                            figsize: tuple[float, float] | None = None,
                                            display_names: dict[str, str] | None = None,
                                            elo_bins: tuple[int, int] | None = None,
                                            elo_bin_cfg: EloBinConfig = ELO_BINS,
                                            color_overrides: list[str | None] | None = None,
                                            linestyle_overrides: list[str | None] | None = None) -> None:
    """Plots per-model accuracy (and optionally Brier score) against true mover_result, binned by log(hours_since_opponent / hours_since_mover), as one row or two elo-bin-restricted rows. Rows with no prior game for either player are excluded."""
    n_before = len(df)
    df = df[df[hours_since_mover_col].notna() & df[hours_since_opponent_col].notna()]
    n_excluded = n_before - len(df)
    if n_excluded:
        print(f'plot_accuracy_by_hours_since_log_ratio: excluded {n_excluded:,} of {n_before:,} rows '
              f'({n_excluded / n_before * 100:.1f}%) -- no prior game for mover and/or opponent.')

    cols_needed = ([f'{name}_predicted_class' for name in names]
                   + [mover_result_col, hours_since_mover_col, hours_since_opponent_col])
    if show_brier:
        cols_needed += [f'{name}_prob_{c}' for name in names for c in ('win', 'draw', 'loss')]
    if elo_bins is not None:
        cols_needed += ['mover_elo', 'opponent_elo']
    df = restrict_to_common_rows(df, cols_needed)
    df = df.copy()

    with np.errstate(divide='ignore', invalid='ignore'):
        ratio = df[hours_since_opponent_col] / df[hours_since_mover_col]
        log_ratio = np.log(ratio) / np.log(step_multiplier)

    edges = _log_ratio_bin_edges(n_bins_per_side)
    labels = _log_ratio_bin_labels(edges, step_multiplier)
    df['_hours_ratio_bin'] = pd.cut(log_ratio, bins=edges, labels=False, right=True, include_lowest=True)
    bin_values = list(range(len(labels)))

    display_names, colors, linestyles = _resolve_plot_style(names, display_names, color_overrides, linestyle_overrides)
    panel_width, panel_height = figsize or BASE_FIGSIZE
    subsets, row_labels = ([df], [None]) if elo_bins is None else _split_by_elo_bin(df, elo_bins, elo_bin_cfg)
    n_rows, n_cols = len(subsets), (2 if show_brier else 1)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(panel_width * n_cols, panel_height * n_rows), squeeze=False)

    for row_idx, (sub_df, row_label) in enumerate(zip(subsets, row_labels)):
        row_title = title if row_label is None else f'{title} -- {row_label}'

        ax_acc = axes[row_idx][0]
        _binned_line_plot(ax_acc, sub_df, names, mover_result_col, sub_df['_hours_ratio_bin'], bin_values, labels,
                           colors, linestyles, row_title, display_names, metric='accuracy', ylim=ylim)
        ax_acc.set_xlabel('Hours since last game, opponent / mover (log ratio)')
        ax_acc.tick_params(axis='x', rotation=45)
        if show_hist:
            _add_hist_twin(ax_acc, sub_df['_hours_ratio_bin'], bin_values, labels, bar_width=0.9)

        if show_brier:
            ax_brier = axes[row_idx][1]
            _binned_line_plot(ax_brier, sub_df, names, mover_result_col, sub_df['_hours_ratio_bin'], bin_values, labels,
                               colors, linestyles, f'{row_title} (Brier)', display_names, metric='brier')
            ax_brier.set_xlabel('Hours since last game, opponent / mover (log ratio)')
            ax_brier.tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.show()


def plot_confidence_by_hours_since_log_ratio(df: pd.DataFrame, names: list[str],
                                              hours_since_mover_col: str = 'hours_since_mover',
                                              hours_since_opponent_col: str = 'hours_since_opponent',
                                              step_multiplier: float = 2.0,
                                              n_bins_per_side: int = 3,
                                              title: str = 'Mean predicted confidence by hours-since-last-game ratio',
                                              figsize: tuple[float, float] | None = None,
                                              display_names: dict[str, str] | None = None,
                                              color_overrides: list[str | None] | None = None,
                                              linestyle_overrides: list[str | None] | None = None) -> None:
    """Plots each model's mean max predicted probability, binned by log(hours_since_opponent / hours_since_mover)."""
    df = df[df[hours_since_mover_col].notna() & df[hours_since_opponent_col].notna()]
    cols_needed = [f'{name}_prob_{c}' for name in names for c in ('win', 'draw', 'loss')] + [hours_since_mover_col, hours_since_opponent_col]
    df = restrict_to_common_rows(df, cols_needed)
    df = df.copy()

    with np.errstate(divide='ignore', invalid='ignore'):
        ratio = df[hours_since_opponent_col] / df[hours_since_mover_col]
        log_ratio = np.log(ratio) / np.log(step_multiplier)

    edges = _log_ratio_bin_edges(n_bins_per_side)
    labels = _log_ratio_bin_labels(edges, step_multiplier)
    df['_hours_ratio_bin'] = pd.cut(log_ratio, bins=edges, labels=False, right=True, include_lowest=True)
    bin_values = list(range(len(labels)))

    display_names, colors, linestyles = _resolve_plot_style(names, display_names, color_overrides, linestyle_overrides)
    fig, ax = plt.subplots(figsize=figsize or BASE_FIGSIZE)
    _binned_line_plot(ax, df, names, None, df['_hours_ratio_bin'], bin_values, labels,
                       colors, linestyles, title, display_names, metric='confidence')
    ax.set_xlabel('Hours since last game, opponent / mover (log ratio)')
    ax.tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.show()


def plot_accuracy_by_hours_since_freshest(df: pd.DataFrame, names: list[str],
                                           hours_since_mover_col: str = 'hours_since_mover',
                                           hours_since_opponent_col: str = 'hours_since_opponent',
                                           mover_result_col: str = 'mover_result',
                                           bin_width: float = 24.0,
                                           ylim: tuple[float, float] = (0, 80),
                                           title: str = 'Result-prediction accuracy by hours since either player last played',
                                           show_hist: bool = False,
                                           show_brier: bool = False,
                                           figsize: tuple[float, float] | None = None,
                                           display_names: dict[str, str] | None = None,
                                           elo_bins: tuple[int, int] | None = None,
                                           elo_bin_cfg: EloBinConfig = ELO_BINS,
                                           color_overrides: list[str | None] | None = None,
                                           linestyle_overrides: list[str | None] | None = None) -> None:
    """Plots per-model accuracy (and optionally Brier score) against true mover_result, binned by min(hours_since_mover, hours_since_opponent), as one row or two elo-bin-restricted rows. Rows with no prior game for either player are excluded."""
    n_before = len(df)
    df = df[df[hours_since_mover_col].notna() & df[hours_since_opponent_col].notna()]
    n_excluded = n_before - len(df)
    if n_excluded:
        print(f'plot_accuracy_by_hours_since_freshest: excluded {n_excluded:,} of {n_before:,} rows '
              f'({n_excluded / n_before * 100:.1f}%) -- no prior game for mover and/or opponent.')

    cols_needed = ([f'{name}_predicted_class' for name in names]
                   + [mover_result_col, hours_since_mover_col, hours_since_opponent_col])
    if show_brier:
        cols_needed += [f'{name}_prob_{c}' for name in names for c in ('win', 'draw', 'loss')]
    if elo_bins is not None:
        cols_needed += ['mover_elo', 'opponent_elo']
    df = restrict_to_common_rows(df, cols_needed)
    df = df.copy()

    hours_freshest = df[[hours_since_mover_col, hours_since_opponent_col]].min(axis=1)
    df['_hours_freshest_bin'] = (hours_freshest // bin_width) * bin_width + bin_width / 2

    display_names, colors, linestyles = _resolve_plot_style(names, display_names, color_overrides, linestyle_overrides)
    panel_width, panel_height = figsize or BASE_FIGSIZE
    subsets, row_labels = ([df], [None]) if elo_bins is None else _split_by_elo_bin(df, elo_bins, elo_bin_cfg)
    n_rows, n_cols = len(subsets), (2 if show_brier else 1)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(panel_width * n_cols, panel_height * n_rows), squeeze=False)

    for row_idx, (sub_df, row_label) in enumerate(zip(subsets, row_labels)):
        bin_series = sub_df['_hours_freshest_bin']
        bin_values = sorted(bin_series.dropna().unique())
        row_title = title if row_label is None else f'{title} -- {row_label}'

        ax_acc = axes[row_idx][0]
        _binned_line_plot(ax_acc, sub_df, names, mover_result_col, bin_series, bin_values, bin_values,
                           colors, linestyles, row_title, display_names, metric='accuracy', ylim=ylim)
        ax_acc.set_xlabel('Hours since the more recent of the two players last played')
        if show_hist:
            _add_hist_twin(ax_acc, bin_series, bin_values, bin_values, bar_width=bin_width * 0.9)

        if show_brier:
            ax_brier = axes[row_idx][1]
            _binned_line_plot(ax_brier, sub_df, names, mover_result_col, bin_series, bin_values, bin_values,
                               colors, linestyles, f'{row_title} (Brier)', display_names, metric='brier')
            ax_brier.set_xlabel('Hours since the more recent of the two players last played')

    plt.tight_layout()
    plt.show()


def plot_confidence_by_hours_since_freshest(df: pd.DataFrame, names: list[str],
                                             hours_since_mover_col: str = 'hours_since_mover',
                                             hours_since_opponent_col: str = 'hours_since_opponent',
                                             bin_width: float = 24.0,
                                             title: str = 'Mean predicted confidence by hours since either player last played',
                                             figsize: tuple[float, float] | None = None,
                                             display_names: dict[str, str] | None = None,
                                             color_overrides: list[str | None] | None = None,
                                             linestyle_overrides: list[str | None] | None = None) -> None:
    """Plots each model's mean max predicted probability, binned by min(hours_since_mover, hours_since_opponent)."""
    df = df[df[hours_since_mover_col].notna() & df[hours_since_opponent_col].notna()]
    cols_needed = [f'{name}_prob_{c}' for name in names for c in ('win', 'draw', 'loss')] + [hours_since_mover_col, hours_since_opponent_col]
    df = restrict_to_common_rows(df, cols_needed)
    df = df.copy()

    hours_freshest = df[[hours_since_mover_col, hours_since_opponent_col]].min(axis=1)
    df['_hours_freshest_bin'] = (hours_freshest // bin_width) * bin_width + bin_width / 2
    bin_values = sorted(df['_hours_freshest_bin'].dropna().unique())
    display_names, colors, linestyles = _resolve_plot_style(names, display_names, color_overrides, linestyle_overrides)

    fig, ax = plt.subplots(figsize=figsize or BASE_FIGSIZE)
    _binned_line_plot(ax, df, names, None, df['_hours_freshest_bin'], bin_values, bin_values,
                       colors, linestyles, title, display_names, metric='confidence')
    ax.set_xlabel('Hours since the more recent of the two players last played')

    plt.tight_layout()
    plt.show()


# (o) TITLES

def _title_mismatch_mask(df: pd.DataFrame, mover_title_col: str, opponent_title_col: str,
                          mover_elo_col: str, opponent_elo_col: str,
                          include_womens: bool, include_no_title: bool = False) -> tuple[pd.Series, pd.Series]:
    """Returns (in_scope, is_mismatch). in_scope marks games where both players hold a title in the same
    track (open, or women's if include_womens), or -- if include_no_title -- one player is titled (either
    track allowed) and the other is untitled ('no_title'). Untitled-vs-untitled is never in scope.
    is_mismatch marks the lower-elo player holding the stronger title (untitled treated as strength 0,
    the lowest of all)."""
    tracks_allowed = {'open'} | ({'womens'} if include_womens else set())

    mover_track = df[mover_title_col].map(title_track)
    opponent_track = df[opponent_title_col].map(title_track)

    mover_titled = mover_track.isin(tracks_allowed)
    opponent_titled = opponent_track.isin(tracks_allowed)
    same_track_titled = (mover_track == opponent_track) & mover_titled

    mover_strength = pd.to_numeric(df[mover_title_col].map(title_strength), errors='coerce')
    opponent_strength = pd.to_numeric(df[opponent_title_col].map(title_strength), errors='coerce')

    if include_no_title:
        mover_untitled = df[mover_title_col] == 'no_title'
        opponent_untitled = df[opponent_title_col] == 'no_title'
        cross_titled_untitled = (mover_titled & opponent_untitled) | (opponent_titled & mover_untitled)
        in_scope = same_track_titled | cross_titled_untitled
        mover_strength = mover_strength.where(~mover_untitled, 0)
        opponent_strength = opponent_strength.where(~opponent_untitled, 0)
    else:
        in_scope = same_track_titled

    mover_is_lower_elo = df[mover_elo_col] < df[opponent_elo_col]
    opponent_is_lower_elo = df[opponent_elo_col] < df[mover_elo_col]

    is_mismatch = in_scope & (
        (mover_is_lower_elo & (mover_strength > opponent_strength))
        | (opponent_is_lower_elo & (opponent_strength > mover_strength))
    )

    return in_scope, is_mismatch.fillna(False)


def plot_accuracy_by_title_mismatch(df: pd.DataFrame, names: list[str],
                                     mover_title_col: str = 'mover_title',
                                     opponent_title_col: str = 'opponent_title',
                                     mover_elo_col: str = 'mover_elo',
                                     opponent_elo_col: str = 'opponent_elo',
                                     mover_result_col: str = 'mover_result',
                                     include_womens: bool = False,
                                     include_no_title: bool = False,
                                     ylim: tuple[float, float] = (0, 100),
                                     title: str | None = None,
                                     figsize: tuple[float, float] | None = None,
                                     display_names: dict[str, str] | None = None) -> None:
    """Plots each model's accuracy against true mover_result, split into title/elo-mismatched (a lower-elo, stronger-titled player is present) vs aligned bars. In scope: both players titled in the same track (open, or women's if include_womens); also titled-vs-untitled pairs if include_no_title (untitled treated as the lowest possible title strength). Untitled-vs-untitled is never in scope."""
    cols_needed = ([f'{name}_predicted_class' for name in names]
                   + [mover_title_col, opponent_title_col, mover_elo_col, opponent_elo_col, mover_result_col])
    df = restrict_to_common_rows(df, cols_needed)
    display_names = DEFAULT_DISPLAY_NAMES if display_names is None else display_names

    in_scope, is_mismatch = _title_mismatch_mask(df, mover_title_col, opponent_title_col,
                                                  mover_elo_col, opponent_elo_col, include_womens, include_no_title)
    n_before = len(df)
    df = df[in_scope].copy()
    is_mismatch = is_mismatch[in_scope]
    print(f'plot_accuracy_by_title_mismatch: {len(df):,} of {n_before:,} rows in scope; '
          f'{int(is_mismatch.sum()):,} flagged as mismatch.')

    true_class = _true_class_series(df, mover_result_col)

    mismatch_acc, aligned_acc = [], []
    for name in names:
        correct = df[f'{name}_predicted_class'] == true_class
        mismatch_acc.append(correct[is_mismatch].mean() * 100 if is_mismatch.sum() else float('nan'))
        aligned_acc.append(correct[~is_mismatch].mean() * 100 if (~is_mismatch).sum() else float('nan'))

    bar_labels = [resolve_display_name(name, display_names) for name in names]
    x = np.arange(len(names))
    width = 0.35

    fig, ax = plt.subplots(figsize=figsize or BASE_FIGSIZE)
    ax.bar(x - width / 2, aligned_acc, width, color='tab:blue', label='Title/elo aligned')
    ax.bar(x + width / 2, mismatch_acc, width, color='tab:orange', label='Underrated title-holder present')

    ax.set_xticks(x)
    ax.set_xticklabels(bar_labels)
    ax.set_ylabel('Accuracy (%)')
    ax.set_title(title or 'Accuracy by title/elo mismatch')
    ax.legend()
    ax.set_ylim(*ylim)

    plt.tight_layout()
    plt.show()


def plot_result_rate_by_title_mismatch(df: pd.DataFrame, *,
                                        mover_title_col: str = 'mover_title',
                                        opponent_title_col: str = 'opponent_title',
                                        mover_elo_col: str = 'mover_elo',
                                        opponent_elo_col: str = 'opponent_elo',
                                        mover_result_col: str = 'mover_result',
                                        include_womens: bool = False,
                                        include_no_title: bool = False,
                                        ylim: tuple[float, float] = (0, 100),
                                        title: str = "Result rate (higher-elo player's perspective) by title/elo mismatch",
                                        figsize: tuple[float, float] | None = None) -> None:
    """Plots the true win/draw/loss rate, from the higher-elo player's perspective, comparing title/elo-mismatched vs aligned games. In scope: both players titled in the same track (open, or women's if include_womens); also titled-vs-untitled pairs if include_no_title. Untitled-vs-untitled is never in scope."""
    cols_needed = [mover_title_col, opponent_title_col, mover_elo_col, opponent_elo_col, mover_result_col]
    df = restrict_to_common_rows(df, cols_needed)

    in_scope, is_mismatch = _title_mismatch_mask(df, mover_title_col, opponent_title_col,
                                                  mover_elo_col, opponent_elo_col, include_womens, include_no_title)
    n_before = len(df)
    df = df[in_scope].copy()
    is_mismatch = is_mismatch[in_scope]
    print(f'plot_result_rate_by_title_mismatch: {len(df):,} of {n_before:,} rows in scope; '
          f'{int(is_mismatch.sum()):,} flagged as mismatch.')

    df['_res_favorite'] = np.where(df[mover_elo_col] >= df[opponent_elo_col],
                                    df[mover_result_col], 1 - df[mover_result_col])
    favorite_class = _true_class_series(df, '_res_favorite')

    groups = {'Aligned': ~is_mismatch, 'Mismatch': is_mismatch}
    rates = {
        group_name: [(favorite_class[mask] == cls).mean() * 100 if mask.sum() else float('nan')
                     for cls in RESULT_CLASS_NAMES]
        for group_name, mask in groups.items()
    }

    x = np.arange(len(RESULT_CLASS_NAMES))
    width = 0.35
    fig, ax = plt.subplots(figsize=figsize or BASE_FIGSIZE)
    ax.bar(x - width / 2, rates['Aligned'], width, color='tab:blue', label='Aligned')
    ax.bar(x + width / 2, rates['Mismatch'], width, color='tab:orange', label='Mismatch')

    ax.set_xticks(x)
    ax.set_xticklabels([c.capitalize() for c in RESULT_CLASS_NAMES])
    ax.set_ylabel('Rate (%)')
    ax.set_title(title)
    ax.legend()
    ax.set_ylim(*ylim)

    plt.tight_layout()
    plt.show()

def table_classification_report(df: pd.DataFrame, names: list[str],
                                  mover_result_col: str = 'mover_result',
                                  display_names: dict[str, str] | None = None) -> pd.DataFrame:
    """Returns a long-format DataFrame of precision/recall/f1-score/support per (model, class)."""
    cols_needed = [f'{name}_predicted_class' for name in names] + [mover_result_col]
    df = restrict_to_common_rows(df, cols_needed)
    true_class = _true_class_series(df, mover_result_col)
    classes_present = [c for c in RESULT_CLASS_NAMES if (true_class == c).any()]

    display_names = DEFAULT_DISPLAY_NAMES if display_names is None else display_names
    rows = []
    for name in names:
        pred_class = df[f'{name}_predicted_class']
        report = classification_report(true_class, pred_class, labels=classes_present,
                                         output_dict=True, zero_division=0)
        for cls in classes_present:
            rows.append({
                'Model': resolve_display_name(name, display_names),
                'Class': cls.capitalize(),
                'Precision (%)': round(report[cls]['precision'] * 100, 1),
                'Recall (%)': round(report[cls]['recall'] * 100, 1),
                'F1 (%)': round(report[cls]['f1-score'] * 100, 1),
                'Support': int(report[cls]['support']),
            })
    return pd.DataFrame(rows)


def plot_confusion_matrix(df: pd.DataFrame, names: list[str],
                           mover_result_col: str = 'mover_result',
                           title: str = 'Confusion matrix by true class',
                           figsize: tuple[float, float] | None = None,
                           display_names: dict[str, str] | None = None) -> None:
    """Plots one true-class-normalized confusion matrix panel per model. Diagonal cells match
    the recall values from plot_accuracy_by_class; off-diagonal cells show where the rest of
    each true class's mass ends up."""
    cols_needed = [f'{name}_predicted_class' for name in names] + [mover_result_col]
    df = restrict_to_common_rows(df, cols_needed)
    true_class = _true_class_series(df, mover_result_col)
    classes_present = [c for c in RESULT_CLASS_NAMES if (true_class == c).any()]

    display_names = DEFAULT_DISPLAY_NAMES if display_names is None else display_names
    panel_width, panel_height = figsize or BASE_FIGSIZE
    fig, axes = plt.subplots(1, len(names), figsize=(panel_width * len(names), panel_height), squeeze=False)

    for i, name in enumerate(names):
        pred_class = df[f'{name}_predicted_class']
        cm = sk_confusion_matrix(true_class, pred_class, labels=classes_present, normalize='true')
        ax = axes[0][i]
        ax.imshow(cm, cmap='Blues', vmin=0, vmax=1)
        ax.set_xticks(range(len(classes_present)))
        ax.set_yticks(range(len(classes_present)))
        ax.set_xticklabels([c.capitalize() for c in classes_present])
        ax.set_yticklabels([c.capitalize() for c in classes_present])
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')
        ax.set_title(resolve_display_name(name, display_names))
        for r in range(cm.shape[0]):
            for c in range(cm.shape[1]):
                ax.text(c, r, f'{cm[r, c] * 100:.1f}%', ha='center', va='center',
                        color='white' if cm[r, c] > 0.5 else 'black', fontsize=9)

    fig.suptitle(title)
    plt.tight_layout()
    plt.show()