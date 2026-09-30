"""
utils_eval.py
Model-independent evaluation helpers: baselines, per-row metrics, binned accuracy, calibration,
display-name mapping.

Latest changes: 27/09/26:
- Docstring tightening
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from src.utils.utils_chess import EloBinConfig, ELO_BINS, elo_bin_by_mover, elo_bin_labels, phase_label

####################
# CONSTANTS
####################

# Optimisation direction per metric name
METRIC_MODES = {'loss': 'min', 'log_loss': 'min', 'accuracy': 'max', 'macro_f1': 'max'}

####################
# FUNCTIONS
####################

# (a) NAIVE/BASELINE PREDICTORS

def naive_binary_pred(win_prob: pd.Series) -> pd.Series:
    """Thresholds win_prob at 0.5, raises on NaN."""
    if win_prob.isna().any():
        raise ValueError(f'{int(win_prob.isna().sum())} NaN values in win_prob, naive_binary_pred requires none.')
    return pd.Series(np.where(win_prob > 0.5, 1.0, 0.0), index=win_prob.index)


def _class_counts(class_labels: torch.Tensor, n_classes: int) -> np.ndarray:
    """Returns per-class counts."""
    return torch.bincount(class_labels, minlength=n_classes).numpy()


def majority_baseline(train_class_labels: torch.Tensor, n_classes: int) -> int:
    """Returns most common class index."""
    counts = _class_counts(train_class_labels, n_classes)
    return int(counts.argmax())


def null_log_loss(y_bin: pd.Series) -> float:
    """Returns log-loss of predicting class base rates."""
    freqs = pd.Series(y_bin).value_counts(normalize=True)
    return -np.sum(freqs * np.log(freqs))

# (b) PER-ROW METRICS AND AGGREGATORS

def _per_row_correct(y_val: np.ndarray, p_val: np.ndarray, classes: np.ndarray) -> np.ndarray:
    """Returns 1.0/0.0 per row for argmax(p_val) == y_val."""
    pred = classes[np.argmax(p_val, axis=1)]
    return (pred == y_val).astype(float)


def _per_row_logloss(y_val: np.ndarray, p_val: np.ndarray, classes: np.ndarray) -> np.ndarray:
    """Returns clipped -log(p[true_class]) per row."""
    class_to_idx = {c: i for i, c in enumerate(classes)}
    y_idx = np.array([class_to_idx[y] for y in y_val])
    p_clipped = np.clip(p_val, 1e-15, 1 - 1e-15)
    return -np.log(p_clipped[np.arange(len(y_idx)), y_idx])


def _per_row_brier(y_val: np.ndarray, p_val: np.ndarray, classes: np.ndarray) -> np.ndarray:
    """Returns per-row squared error between one-hot y_val and p_val."""
    y_onehot = pd.get_dummies(y_val).reindex(columns=classes, fill_value=0).to_numpy(dtype=float)
    return np.sum((y_onehot - p_val) ** 2, axis=1)


def accuracy(y_val: np.ndarray, p_val: np.ndarray, classes: np.ndarray) -> float:
    """Returns fraction of rows where argmax(p_val) == y_val."""
    return float(_per_row_correct(y_val, p_val, classes).mean())


def log_loss(y_val: np.ndarray, p_val: np.ndarray, classes: np.ndarray) -> float:
    """Returns mean NLL of true class."""
    return float(_per_row_logloss(y_val, p_val, classes).mean())


def multiclass_brier(y_val: np.ndarray, p_val: np.ndarray, classes: np.ndarray) -> float:
    """Returns mean multiclass Brier score."""
    return float(_per_row_brier(y_val, p_val, classes).mean())


def mean_ci(per_row_values: np.ndarray, z: float = 1.96,
            game_ids: pd.Series | np.ndarray | None = None) -> tuple[float, float, float]:
    """Returns (mean, ci_low, ci_high), averaged per game first if game_ids given."""
    if game_ids is not None:
        values = pd.Series(np.asarray(per_row_values)).groupby(np.asarray(game_ids)).mean().to_numpy()
    else:
        values = np.asarray(per_row_values)

    xbar = values.mean()
    se = values.std(ddof=1) / np.sqrt(len(values))
    return xbar, xbar - z * se, xbar + z * se

# (c) BINNED ACCURACY REPORTING

def accuracy_by_elo_bin(df: pd.DataFrame, actual_col: str, preds: dict[str, str],
                        cfg: EloBinConfig = ELO_BINS, method: str = 'mean') -> pd.DataFrame:
    """Returns per-model accuracy by elo bin, preds maps label to pred_col."""
    binned, edges = elo_bin_by_mover(df, cfg, method)
    labels = elo_bin_labels(edges)

    rows = []
    for model_label, pred_col in preds.items():
        valid = binned[actual_col].notna() & binned[pred_col].notna()
        for i in range(len(edges) - 1):
            mask = valid & (binned['elo_bin'] == i)
            if mask.sum() == 0:
                continue
            correct = binned.loc[mask, pred_col].to_numpy() == binned.loc[mask, actual_col].to_numpy()
            rows.append({'elo_bin': i, 'bin_label': labels[i], 'model': model_label,
                         'accuracy': correct.mean(), 'n': int(mask.sum())})

    return pd.DataFrame(rows)


def accuracy_by_phase(df: pd.DataFrame, actual_col: str, preds: dict[str, str],
                      phase_col: str = 'phase') -> pd.DataFrame:
    """Returns per-model accuracy by game phase, preds maps label to pred_col."""
    phase_bin = df[phase_col].map(phase_label)

    rows = []
    for model_label, pred_col in preds.items():
        valid = df[actual_col].notna() & df[pred_col].notna() & phase_bin.notna()
        for bin_label in ('opening', 'middlegame', 'endgame'):
            mask = valid & (phase_bin == bin_label)
            if mask.sum() == 0:
                continue
            correct = df.loc[mask, pred_col].to_numpy() == df.loc[mask, actual_col].to_numpy()
            rows.append({'phase': bin_label, 'model': model_label,
                         'accuracy': correct.mean(), 'n': int(mask.sum())})

    return pd.DataFrame(rows)

# (d) METRIC COMPARISON

def metric_mode(metric: str) -> str:
    """Returns 'min' or 'max' for metric."""
    if metric not in METRIC_MODES:
        raise KeyError(f'No mode registered for metric {metric!r}. Registered: {list(METRIC_MODES)}')
    return METRIC_MODES[metric]


def is_better(new: float, best: float, mode: str) -> bool:
    """Returns whether new beats best under mode."""
    return new > best if mode == 'max' else new < best

# (e) CALIBRATION

def fit_temperature(logits: torch.Tensor, targets: torch.Tensor, max_iter: int = 100) -> float:
    """Fits single temperature on logits by minimising NLL."""
    log_t = torch.zeros(1, requires_grad=True)
    optimizer = torch.optim.LBFGS([log_t], lr=0.1, max_iter=max_iter)
    nll = nn.CrossEntropyLoss()

    def closure() -> torch.Tensor:
        optimizer.zero_grad()
        loss = nll(logits / log_t.exp(), targets)
        loss.backward()
        return loss

    optimizer.step(closure)
    return float(log_t.exp().item())


def apply_temperature(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    """Returns softmax probs of temperature-scaled logits."""
    return torch.softmax(logits / temperature, dim=1)


def fit_binary_temperature(logit: torch.Tensor, targets: torch.Tensor, max_iter: int = 100) -> float:
    """Fits single temperature for binary logit."""
    two_class_logits = torch.stack([torch.zeros_like(logit), logit], dim=1)
    return fit_temperature(two_class_logits, targets, max_iter=max_iter)


def apply_binary_temperature(logit: torch.Tensor, temperature: float) -> torch.Tensor:
    """Returns temperature-scaled P(win) for binary logit."""
    two_class_logits = torch.stack([torch.zeros_like(logit), logit], dim=1)
    return apply_temperature(two_class_logits, temperature)[:, 1]


def _binary_probit_probs(preds: torch.Tensor, c: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
    """Returns (win, loss) probs for scalar preds under cutpoint c and scale sigma."""
    z = (c - preds) / sigma
    cdf = 0.5 * (1.0 + torch.erf(z / np.sqrt(2.0)))
    p_loss = cdf.clamp(min=1e-8)
    p_win = (1.0 - cdf).clamp(min=1e-8)
    return torch.stack([p_win, p_loss], dim=1)


def fit_binary_probit(preds: np.ndarray, targets: np.ndarray, max_iter: int = 200) -> tuple[float, float]:
    """Fits probit (c, sigma) on scalar preds vs decisive targets (loss=0, win=1)."""
    preds_t = torch.as_tensor(preds, dtype=torch.float32)
    targets_t = torch.as_tensor(targets, dtype=torch.long)
    target_idx = 1 - targets_t

    c = torch.tensor(float(preds_t.mean()), requires_grad=True)
    sigma_init = float(preds_t.std())
    if not np.isfinite(sigma_init) or sigma_init <= 0:
        sigma_init = 1.0
    log_sigma = torch.tensor(float(np.log(sigma_init)), requires_grad=True)
    optimizer = torch.optim.LBFGS([c, log_sigma], lr=0.1, max_iter=max_iter)

    def closure() -> torch.Tensor:
        optimizer.zero_grad()
        probs = _binary_probit_probs(preds_t, c, log_sigma.exp())
        loss = -torch.log(probs[torch.arange(len(target_idx)), target_idx]).mean()
        loss.backward()
        return loss

    optimizer.step(closure)
    return float(c.item()), float(log_sigma.exp().item())


def apply_binary_probit(preds: np.ndarray, params: tuple[float, float]) -> np.ndarray:
    """Returns (win, loss) probs for scalar preds under fitted probit."""
    c, sigma = params
    probs = _binary_probit_probs(torch.as_tensor(preds, dtype=torch.float32),
                                 torch.tensor(c), torch.tensor(sigma))
    return probs.detach().numpy()


def collapse_to_expected_score(probs: np.ndarray | torch.Tensor) -> np.ndarray:
    """Collapses (loss, draw, win) probs to expected score."""
    if isinstance(probs, torch.Tensor):
        probs = probs.detach().numpy()
    return probs[:, 2] + 0.5 * probs[:, 1]


def binary_probit_prediction_cols(raw_scores: np.ndarray, probit_params: tuple[float, float]) -> dict:
    """Returns prob_win, prob_draw (always 0), prob_loss, predicted_class under fitted probit."""
    probs = apply_binary_probit(raw_scores, probit_params)
    prob_win = probs[:, 0]
    prob_loss = probs[:, 1]
    prob_draw = np.zeros_like(prob_win)
    predicted_class = np.where(prob_win >= 0.5, 'win', 'loss')
    return {'prob_win': prob_win, 'prob_draw': prob_draw, 'prob_loss': prob_loss,
            'predicted_class': predicted_class}

# (f) DISPLAY NAMES

def resolve_display_name(name: str, display_names: dict[str, str] | None = None) -> str:
    """Returns display label for name, or name if absent."""
    return (display_names or {}).get(name, name)


def resolve_display_names(names: list[str], display_names: dict[str, str] | None = None) -> dict[str, str]:
    """Maps each name to its display label."""
    return {name: resolve_display_name(name, display_names) for name in names}