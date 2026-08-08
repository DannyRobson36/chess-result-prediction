"""
utils_eval.py
Model-independent evaluation helpers: baselines, per-row metrics, binned accuracy, calibration.

Latest changes: 08/08/26:
- Initial commit
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from scripts.utils.utils_chess import EloBinConfig, ELO_BINS, elo_bin_by_mover, elo_bin_labels, phase_label

####################
# CONSTANTS
####################

METRIC_MODES = {'loss': 'min', 'log_loss': 'min', 'accuracy': 'max', 'macro_f1': 'max'}

####################
# FUNCTIONS
####################

# (a) NAIVE/BASELINE PREDICTORS

def naive_binary_pred(win_prob: pd.Series) -> pd.Series:
    """Thresholds win_prob at 0.5 into a binary prediction; raises if any values are NaN."""
    if win_prob.isna().any():
        raise ValueError(f'{int(win_prob.isna().sum())} NaN values in win_prob; naive_binary_pred requires none.')
    return pd.Series(np.where(win_prob > 0.5, 1.0, 0.0), index=win_prob.index)


def _class_counts(class_labels: torch.Tensor, n_classes: int) -> np.ndarray:
    """Returns per-class counts from class_labels."""
    return torch.bincount(class_labels, minlength=n_classes).numpy()


def majority_baseline(train_class_labels: torch.Tensor, n_classes: int) -> int:
    """Returns the most common class index in train_class_labels."""
    counts = _class_counts(train_class_labels, n_classes)
    return int(counts.argmax())


def null_log_loss(y_bin: pd.Series) -> float:
    """Computes the log-loss of a baseline predicting each class's own base rate."""
    freqs = pd.Series(y_bin).value_counts(normalize=True)
    return -np.sum(freqs * np.log(freqs))


# (b) PER-ROW METRICS AND AGGREGATORS

def _per_row_correct(y_val: np.ndarray, p_val: np.ndarray, classes: np.ndarray) -> np.ndarray:
    """Returns 1.0/0.0 per row for whether argmax(p_val) matches y_val."""
    pred = classes[np.argmax(p_val, axis=1)]
    return (pred == y_val).astype(float)


def _per_row_logloss(y_val: np.ndarray, p_val: np.ndarray, classes: np.ndarray) -> np.ndarray:
    """Returns -log(p[true_class]) per row, clipped for stability."""
    class_to_idx = {c: i for i, c in enumerate(classes)}
    y_idx = np.array([class_to_idx[y] for y in y_val])
    p_clipped = np.clip(p_val, 1e-15, 1 - 1e-15)
    return -np.log(p_clipped[np.arange(len(y_idx)), y_idx])


def _per_row_brier(y_val: np.ndarray, p_val: np.ndarray, classes: np.ndarray) -> np.ndarray:
    """Returns per-row squared error between y_val and predicted probs p_val."""
    y_onehot = pd.get_dummies(y_val).reindex(columns=classes, fill_value=0).to_numpy()
    return np.sum((y_onehot - p_val) ** 2, axis=1)


def accuracy(y_val: np.ndarray, p_val: np.ndarray, classes: np.ndarray) -> float:
    """Mean of _per_row_correct: fraction of rows where argmax(p_val) = y_val."""
    return float(_per_row_correct(y_val, p_val, classes).mean())


def log_loss(y_val: np.ndarray, p_val: np.ndarray, classes: np.ndarray) -> float:
    """Mean of _per_row_logloss: mean negative log likelihood of the true class."""
    return float(_per_row_logloss(y_val, p_val, classes).mean())


def multiclass_brier(y_val: np.ndarray, p_val: np.ndarray, classes: np.ndarray) -> float:
    """Mean of _per_row_brier: mean squared error between one-hot labels and predicted probabilities."""
    return float(_per_row_brier(y_val, p_val, classes).mean())


def mean_ci(per_row_values: np.ndarray, z: float = 1.96,
            game_ids: pd.Series | np.ndarray | None = None) -> tuple[float, float, float]:
    """Returns (mean, ci_low, ci_high) via normal approx; averages to per-game values first if game_ids given."""
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
    """Computes per-model accuracy by elo bin for each {label: pred_col} in preds."""
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
    """Computes per-model accuracy grouped by phase_label (opening/middlegame/endgame)."""
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
    """Returns 'min' or 'max' for a metric name."""
    if metric not in METRIC_MODES:
        raise KeyError(f"No mode registered for metric '{metric}'. Registered: {list(METRIC_MODES)}")
    return METRIC_MODES[metric]


def is_better(new: float, best: float, mode: str) -> bool:
    """Returns whether new beats best under mode."""
    return new > best if mode == 'max' else new < best


# (e) CALIBRATION

def fit_temperature(logits: torch.Tensor, targets: torch.Tensor, max_iter: int = 100) -> float:
    """Fits a single temperature on logits by minimising NLL."""
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
    """Returns class probabilities from logits scaled by temperature."""
    return torch.softmax(logits / temperature, dim=1)


def _ordered_probit_probs(preds: torch.Tensor, c1: torch.Tensor, c2: torch.Tensor,
                           sigma: torch.Tensor) -> torch.Tensor:
    """Returns W/D/L probabilities for scalar preds under cutpoints c1<c2 and scale sigma."""
    z1 = (c1 - preds) / sigma
    z2 = (c2 - preds) / sigma
    cdf1 = 0.5 * (1.0 + torch.erf(z1 / np.sqrt(2.0)))
    cdf2 = 0.5 * (1.0 + torch.erf(z2 / np.sqrt(2.0)))
    p_loss = cdf1
    p_draw = (cdf2 - cdf1).clamp(min=1e-8)
    p_win = (1.0 - cdf2).clamp(min=1e-8)
    return torch.stack([p_win, p_draw, p_loss.clamp(min=1e-8)], dim=1)


def fit_ordered_probit(preds: np.ndarray, targets: np.ndarray, max_iter: int = 200) -> tuple[float, float, float]:
    """Fits two cutpoints to convert scalar predictions to W/D/L probabilities."""
    preds_t = torch.as_tensor(preds, dtype=torch.float32)
    targets_t = torch.as_tensor(targets, dtype=torch.long)
    target_idx = 2 - targets_t  # RESULT_TO_CLASS is loss=0/draw=1/win=2; probs are win-first

    c1 = torch.tensor(0.25, requires_grad=True)
    log_gap = torch.tensor(float(np.log(0.5)), requires_grad=True)
    log_sigma = torch.tensor(float(np.log(0.25)), requires_grad=True)
    optimizer = torch.optim.LBFGS([c1, log_gap, log_sigma], lr=0.1, max_iter=max_iter)

    def closure() -> torch.Tensor:
        optimizer.zero_grad()
        probs = _ordered_probit_probs(preds_t, c1, c1 + log_gap.exp(), log_sigma.exp())
        loss = -torch.log(probs[torch.arange(len(target_idx)), target_idx]).mean()
        loss.backward()
        return loss

    optimizer.step(closure)
    return float(c1.item()), float((c1 + log_gap.exp()).item()), float(log_sigma.exp().item())


def apply_ordered_probit(preds: np.ndarray, params: tuple[float, float, float]) -> np.ndarray:
    """Returns W/D/L probabilities for scalar preds under fitted ordered probit params."""
    c1, c2, sigma = params
    probs = _ordered_probit_probs(torch.as_tensor(preds, dtype=torch.float32),
                                   torch.tensor(c1), torch.tensor(c2), torch.tensor(sigma))
    return probs.detach().numpy()


def _binary_probit_probs(preds: torch.Tensor, c: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
    """Returns W/L probabilities for scalar preds under a single cutpoint c and scale sigma."""
    z = (c - preds) / sigma
    cdf = 0.5 * (1.0 + torch.erf(z / np.sqrt(2.0)))
    p_loss = cdf.clamp(min=1e-8)
    p_win = (1.0 - cdf).clamp(min=1e-8)
    return torch.stack([p_win, p_loss], dim=1)


def fit_binary_probit(preds: np.ndarray, targets: np.ndarray, max_iter: int = 200) -> tuple[float, float]:
    """Fits a single cutpoint to convert scalar predictions to W/L probabilities."""
    preds_t = torch.as_tensor(preds, dtype=torch.float32)
    targets_t = torch.as_tensor(targets, dtype=torch.long)
    target_idx = 1 - targets_t  # TWO_WAY_CLASS_NAMES is loss=0/win=1; probs are win-first

    c = torch.tensor(0.5, requires_grad=True)
    log_sigma = torch.tensor(float(np.log(0.25)), requires_grad=True)
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
    """Returns W/L probabilities for scalar preds under fitted binary probit params."""
    c, sigma = params
    probs = _binary_probit_probs(torch.as_tensor(preds, dtype=torch.float32),
                                  torch.tensor(c), torch.tensor(sigma))
    return probs.detach().numpy()