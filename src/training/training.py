"""
training.py
Batches SplitData, runs training loop with early stopping, saves checkpoint for inference.

Latest changes: 27/09/26:
- Docstring tightening, reordering, elo-weight alpha default fixed
"""

import os
import time
from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, classification_report
from torch.optim.lr_scheduler import LRScheduler, ReduceLROnPlateau

from src.features.features import SplitData, PrepConfig
from src.models.model_arch import get_output_type, get_predict_fn
from src.models.model_config import BaseModelConfig
from src.utils.utils_chess import RESULT_CLASS_NAMES
from src.utils.utils_eval import (
    metric_mode, is_better, fit_temperature, fit_binary_probit, collapse_to_expected_score,
)

####################
# CONSTANTS
####################

LOSS_REGISTRY: dict[str, type] = {
    'cross_entropy': nn.CrossEntropyLoss,
    'bce_logits': nn.BCEWithLogitsLoss,
    'mse': nn.MSELoss,
}

DEFAULT_LOSS_BY_OUTPUT_TYPE = {
    'classification': 'cross_entropy',
    'regression': 'bce_logits',
}

SCHEDULER_REGISTRY: dict[str, type] = {
    'step': torch.optim.lr_scheduler.StepLR,
    'cosine': torch.optim.lr_scheduler.CosineAnnealingLR,
    'plateau': torch.optim.lr_scheduler.ReduceLROnPlateau,
}

ELO_WEIGHT_DEFAULT_ALPHA = 0.5
ELO_WEIGHT_MAX_RATIO = 5.0

# Rows in per-epoch train-loss probe
TRAIN_PROBE_SIZE = 100_000

ADAM_BETAS = (0.9, 0.999)
ADAM_EPS = 1e-8

####################
# FUNCTIONS
####################

# (a) DEVICE

def get_device() -> torch.device:
    """Returns cuda if available, else cpu."""
    if torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')


def _sync_device(device: torch.device) -> None:
    """Blocks until queued GPU work finishes."""
    if device.type == 'cuda':
        torch.cuda.synchronize()

# (b) BATCHING

def iterate_batches(idx: np.ndarray, batch_size: int, shuffle: bool = True,
                    seed: int | None = None, drop_last: bool = False) -> Iterator[np.ndarray]:
    """Yields batch index arrays over idx."""
    idx = np.asarray(idx)
    if shuffle:
        rng = np.random.default_rng(seed)
        idx = idx[rng.permutation(len(idx))]

    n = len(idx)
    end = n - (n % batch_size) if drop_last else n
    for start in range(0, end, batch_size):
        yield idx[start:start + batch_size]


def make_batch(split: SplitData, idx: np.ndarray, device: torch.device) -> dict:
    """Slices split at idx, moves to device, casts to model dtypes."""
    batch = {
        'elo_mean_bin': split.elo_mean_bin[idx].to(device).long(),
        'elo_self_bin': split.elo_self_bin[idx].to(device).long(),
        'elo_oppo_bin': split.elo_oppo_bin[idx].to(device).long(),
    }
    if split.boards is not None:
        batch['boards'] = split.boards[idx].to(device).float()
    if split.board_token_ids is not None:
        batch['board_token_ids'] = split.board_token_ids[idx].to(device).long()
    if split.legal_dest is not None:
        batch['legal_dest'] = split.legal_dest[idx].to(device).float()
    if split.attacked_mover is not None:
        batch['attacked_mover'] = split.attacked_mover[idx].to(device).float()
    if split.attacked_opponent is not None:
        batch['attacked_opponent'] = split.attacked_opponent[idx].to(device).float()
    for name, tensor in split.features.items():
        batch[name] = tensor[idx].to(device)
    return batch


def probe_idx(split: SplitData, n: int, seed: int = 0) -> np.ndarray:
    """Returns random subsample of up to n row positions."""
    rng = np.random.default_rng(seed)
    n = min(n, len(split))
    return rng.choice(len(split), size=n, replace=False)


def get_targets(split: SplitData, batch_idx: np.ndarray, output_type: str, device: torch.device) -> torch.Tensor:
    """Returns targets for output_type at batch_idx."""
    if output_type == 'classification':
        return split.result_class[batch_idx].to(device).long()
    return split.result_cont[batch_idx].to(device)


def get_aux_targets(split: SplitData, batch_idx: np.ndarray, device: torch.device) -> torch.Tensor:
    """Stacks aux targets at batch_idx into (b, 3, 64) tensor."""
    return torch.stack([
        split.legal_dest[batch_idx],
        split.attacked_mover[batch_idx],
        split.attacked_opponent[batch_idx],
    ], dim=1).to(device).float()

# (c) LOSS & SCHEDULER

def build_loss(loss_name: str, **kwargs) -> nn.Module:
    """Builds per-sample loss by name."""
    if loss_name not in LOSS_REGISTRY:
        raise ValueError(f'Unknown loss {loss_name!r}, choose from {list(LOSS_REGISTRY)}')
    kwargs.setdefault('reduction', 'none')
    return LOSS_REGISTRY[loss_name](**kwargs)


def resolve_loss_name(output_type: str, loss_name: str | None) -> str:
    """Returns loss_name, or default for output_type."""
    return loss_name if loss_name is not None else DEFAULT_LOSS_BY_OUTPUT_TYPE[output_type]


def build_scheduler(optimizer: torch.optim.Optimizer, schedule_cfg: dict | None, n_epochs: int,
                    mode: str) -> LRScheduler | ReduceLROnPlateau | None:
    """Builds LR scheduler from schedule_cfg, or None."""
    if schedule_cfg is None:
        return None
    schedule_cfg = dict(schedule_cfg)
    kind = schedule_cfg.pop('type')
    if kind not in SCHEDULER_REGISTRY:
        raise ValueError(f'Unknown scheduler {kind!r}, choose from {list(SCHEDULER_REGISTRY)}')
    if kind == 'cosine':
        schedule_cfg.setdefault('T_max', n_epochs)
    if kind == 'plateau':
        schedule_cfg.setdefault('mode', mode)
    return SCHEDULER_REGISTRY[kind](optimizer, **schedule_cfg)


def _resolve_aux_loss_weight(model: nn.Module, arch_name: str) -> float | None:
    """Validates aux head config, returns loss weight or None."""
    if not hasattr(model, 'aux_head'):
        aux_head_cfg = getattr(model.cfg, 'aux_head', None)
        if aux_head_cfg is not None and aux_head_cfg.enabled:
            raise ValueError(f'cfg.aux_head.enabled is True but {arch_name!r} has no aux head support.')
        return None

    aux_head_cfg = model.cfg.aux_head
    if model.aux_head is not None:
        if aux_head_cfg.loss_weight is None:
            raise ValueError('cfg.aux_head.enabled is True but cfg.aux_head.loss_weight is None.')
        return aux_head_cfg.loss_weight

    if aux_head_cfg.loss_weight is not None:
        raise ValueError('cfg.aux_head.loss_weight was set but cfg.aux_head.enabled is False.')
    return None

# (d) ELO-BIN LOSS WEIGHTING

def compute_elo_bin_weights(split: SplitData, alpha: float = ELO_WEIGHT_DEFAULT_ALPHA,
                            max_ratio: float | None = ELO_WEIGHT_MAX_RATIO,
                            device: torch.device | None = None) -> torch.Tensor:
    """Returns mean-normalised (1/count)**alpha elo-bin weights, capped at max_ratio x min."""
    counts = torch.bincount(split.elo_mean_bin.long(), minlength=split.n_elo_bins).float().clamp(min=1)
    weights = (1.0 / counts) ** alpha
    weights = weights / weights.mean()
    if max_ratio is not None:
        weights = weights.clamp(max=weights.min() * max_ratio)
    return weights.to(device) if device is not None else weights


def _resolve_bin_weights(train: SplitData, cfg: 'TrainConfig', device: torch.device) -> torch.Tensor | None:
    """Builds elo-bin weights from cfg, or None."""
    if not cfg.elo_weighted_loss:
        if cfg.elo_weight_alpha is not None:
            raise ValueError('cfg.elo_weight_alpha was set but cfg.elo_weighted_loss is False.')
        return None
    alpha = cfg.elo_weight_alpha if cfg.elo_weight_alpha is not None else ELO_WEIGHT_DEFAULT_ALPHA
    return compute_elo_bin_weights(train, alpha=alpha, device=device)

# (e) LR WARMUP

def apply_warmup_lr(optimizer: torch.optim.Optimizer, base_lr: float, step: int, warmup_steps: int) -> None:
    """Linearly ramps lr up to base_lr over warmup_steps."""
    scale = min(1.0, (step + 1) / warmup_steps)
    for group in optimizer.param_groups:
        group['lr'] = base_lr * scale


def _resolve_warmup_steps(warmup_prop: float | None, n_train: int, batch_size: int, n_epochs: int) -> int | None:
    """Converts warmup_prop to step count, capped at one epoch."""
    if warmup_prop is None:
        return None
    batches_per_epoch = n_train // batch_size
    total_steps = batches_per_epoch * n_epochs
    requested_steps = max(1, round(warmup_prop * total_steps))
    return min(requested_steps, batches_per_epoch)

# (f) TRAIN & EVALUATE ONE EPOCH

def train_one_epoch(model: nn.Module, arch_name: str, loss_fn: nn.Module, optimizer: torch.optim.Optimizer,
                    split: SplitData, idx: np.ndarray, output_type: str, batch_size: int = 64,
                    seed: int | None = None, device: torch.device | None = None, drop_last: bool = True,
                    bin_weights: torch.Tensor | None = None, grad_clip_norm: float | None = None,
                    warmup_steps: int | None = None, base_lr: float | None = None,
                    step_counter: list[int] | None = None,
                    aux_loss_fn: nn.Module | None = None, aux_loss_weight: float | None = None) -> None:
    """Runs one training epoch, with optional elo-bin weighting and aux loss."""
    device = device or get_device()
    predict_fn = get_predict_fn(arch_name)
    model.train()
    for batch_idx in iterate_batches(idx, batch_size, shuffle=True, seed=seed, drop_last=drop_last):
        if warmup_steps is not None and step_counter[0] < warmup_steps:
            apply_warmup_lr(optimizer, base_lr, step_counter[0], warmup_steps)

        batch = make_batch(split, batch_idx, device)
        targets = get_targets(split, batch_idx, output_type, device)

        optimizer.zero_grad()
        preds, aux_preds = predict_fn(model, batch)
        per_sample = loss_fn(preds, targets)
        if bin_weights is not None:
            loss = (per_sample * bin_weights[batch['elo_mean_bin']]).mean()
        else:
            loss = per_sample.mean()

        if aux_loss_weight is not None:
            aux_targets = get_aux_targets(split, batch_idx, device)
            aux_loss = aux_loss_fn(aux_preds, aux_targets).mean()
            loss = loss + aux_loss_weight * aux_loss

        loss.backward()
        if grad_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
        optimizer.step()

        if step_counter is not None:
            step_counter[0] += 1


def evaluate_with_loss(model: nn.Module, arch_name: str, split: SplitData, idx: np.ndarray, loss_fn: nn.Module,
                       output_type: str, class_names: list[str] | None = None,
                       batch_size: int = 64, device: torch.device | None = None,
                       aux_loss_fn: nn.Module | None = None, aux_loss_weight: float | None = None) -> dict:
    """Returns loss, plus accuracy/macro_f1 if classification, plus aux_loss if aux head on."""
    device = device or get_device()
    predict_fn = get_predict_fn(arch_name)
    model.eval()
    total_loss, total_n = 0.0, 0
    total_aux_loss, total_aux_n = 0.0, 0
    all_preds, all_targets = [], []
    with torch.no_grad():
        for batch_idx in iterate_batches(idx, batch_size, shuffle=False, drop_last=False):
            batch = make_batch(split, batch_idx, device)
            targets = get_targets(split, batch_idx, output_type, device)
            preds, aux_preds = predict_fn(model, batch)
            per_sample = loss_fn(preds, targets)

            total_loss += per_sample.sum().item()
            total_n += len(batch_idx)

            if aux_loss_weight is not None:
                aux_targets = get_aux_targets(split, batch_idx, device)
                aux_per_element = aux_loss_fn(aux_preds, aux_targets)
                total_aux_loss += aux_per_element.sum().item()
                total_aux_n += aux_per_element.numel()

            if output_type == 'classification':
                all_preds.append(preds.argmax(dim=1).cpu())
                all_targets.append(targets.cpu())

    metrics = {'loss': total_loss / total_n}
    if output_type == 'classification':
        if class_names is None:
            class_names = RESULT_CLASS_NAMES
        preds_arr = torch.cat(all_preds).numpy()
        targs_arr = torch.cat(all_targets).numpy()
        report = classification_report(targs_arr, preds_arr, labels=list(range(len(class_names))),
                                       target_names=class_names, output_dict=True, zero_division=0)
        metrics['accuracy'] = accuracy_score(targs_arr, preds_arr)
        metrics['macro_f1'] = report['macro avg']['f1-score']
    if aux_loss_weight is not None:
        metrics['aux_loss'] = total_aux_loss / total_aux_n
    return metrics

# (g) FULL TRAINING LOOP

def run_training(model: nn.Module, arch_name: str, train: SplitData, val: SplitData, cfg: 'TrainConfig',
                 train_probe_idx: np.ndarray | None = None,
                 val_probe_idx: np.ndarray | None = None, device: torch.device | None = None,
                 profile_time: bool = False) -> dict:
    """Trains with early stopping, returns history plus best-epoch weights, metrics and lr."""
    device = device or get_device()
    model = model.to(device)
    output_type = get_output_type(arch_name)
    if output_type == 'regression' and cfg.primary_metric != 'loss':
        raise ValueError(f'cfg.primary_metric={cfg.primary_metric!r} unavailable for regression, use loss.')
    mode = metric_mode(cfg.primary_metric)

    loss_name = resolve_loss_name(output_type, cfg.loss_name)
    loss_fn = build_loss(loss_name)

    aux_loss_weight = _resolve_aux_loss_weight(model, arch_name)
    aux_loss_fn = build_loss('bce_logits') if aux_loss_weight is not None else None
    if aux_loss_weight is not None and (train.legal_dest is None or val.legal_dest is None):
        raise ValueError(f'{arch_name} aux_head enabled (loss_weight={aux_loss_weight}) but train/val '
                         f'SplitData built without aux_targets=True.')

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, betas=ADAM_BETAS, eps=ADAM_EPS,
                                 weight_decay=cfg.weight_decay)
    scheduler = build_scheduler(optimizer, cfg.schedule_cfg, cfg.n_epochs, mode)
    bin_weights = _resolve_bin_weights(train, cfg, device)
    warmup_steps = _resolve_warmup_steps(cfg.warmup_prop, len(train), cfg.batch_size, cfg.n_epochs)

    keys = ['loss', 'accuracy', 'macro_f1'] if output_type == 'classification' else ['loss']
    if aux_loss_weight is not None:
        keys = keys + ['aux_loss']
    history = {f'{prefix}_{k}': [] for prefix in ('train', 'val') for k in keys}
    history['epoch_time'] = []
    history['lr'] = []

    if train_probe_idx is None:
        train_probe_idx = probe_idx(train, n=TRAIN_PROBE_SIZE, seed=cfg.seed)
    val_idx = val_probe_idx if val_probe_idx is not None else np.arange(len(val))
    train_idx = np.arange(len(train))

    best_score, best_epoch, best_state_dict = None, None, None
    best_train_metrics, best_val_metrics, best_lr = None, None, None
    epochs_without_improvement = 0
    step_counter = [0]

    for epoch in range(cfg.n_epochs):
        _sync_device(device)
        epoch_start = time.time()

        train_one_epoch(model, arch_name, loss_fn, optimizer, train, train_idx, output_type,
                        batch_size=cfg.batch_size, seed=cfg.seed + epoch, device=device,
                        drop_last=True, bin_weights=bin_weights, grad_clip_norm=cfg.grad_clip_norm,
                        warmup_steps=warmup_steps, base_lr=cfg.lr, step_counter=step_counter,
                        aux_loss_fn=aux_loss_fn, aux_loss_weight=aux_loss_weight)
        if profile_time:
            _sync_device(device)
            train_elapsed = time.time() - epoch_start

        train_metrics = evaluate_with_loss(model, arch_name, train, train_probe_idx, loss_fn, output_type,
                                           batch_size=cfg.eval_batch_size, device=device,
                                           aux_loss_fn=aux_loss_fn, aux_loss_weight=aux_loss_weight)
        if profile_time:
            _sync_device(device)
            train_eval_elapsed = time.time() - epoch_start - train_elapsed

        val_metrics = evaluate_with_loss(model, arch_name, val, val_idx, loss_fn, output_type,
                                         batch_size=cfg.eval_batch_size, device=device,
                                         aux_loss_fn=aux_loss_fn, aux_loss_weight=aux_loss_weight)

        _sync_device(device)
        epoch_time = time.time() - epoch_start
        if profile_time:
            val_eval_elapsed = epoch_time - train_elapsed - train_eval_elapsed

        for key in keys:
            history[f'train_{key}'].append(train_metrics[key])
            history[f'val_{key}'].append(val_metrics[key])
        history['epoch_time'].append(epoch_time)
        history['lr'].append(optimizer.param_groups[0]['lr'])

        if cfg.verbose:
            parts = [f'epoch {epoch + 1}/{cfg.n_epochs}']
            for key in keys:
                parts.append(f'train_{key}={train_metrics[key]:.4f}')
                parts.append(f'val_{key}={val_metrics[key]:.4f}')
            parts.append(f'time={epoch_time:.2f}s')
            print('  '.join(parts))
            if profile_time:
                other_elapsed = time.time() - epoch_start - train_elapsed - train_eval_elapsed - val_eval_elapsed
                print(f'  time breakdown: train={train_elapsed:.2f}s  train_eval={train_eval_elapsed:.2f}s  '
                      f'val_eval={val_eval_elapsed:.2f}s  other={other_elapsed:.2f}s')

        score = val_metrics[cfg.primary_metric]
        if best_score is None or is_better(score, best_score, mode):
            best_score, best_epoch = score, epoch + 1
            best_state_dict = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_train_metrics = dict(train_metrics)
            best_val_metrics = dict(val_metrics)
            best_lr = history['lr'][-1]
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if cfg.patience is not None and epochs_without_improvement >= cfg.patience:
                if cfg.verbose:
                    print(f'Early stopping at epoch {epoch + 1} '
                          f'(no {cfg.primary_metric} improvement in {cfg.patience} epochs)')
                break

        if scheduler is not None:
            if isinstance(scheduler, ReduceLROnPlateau):
                scheduler.step(score)
            else:
                scheduler.step()

    return {'history': history, 'best_epoch': best_epoch, 'best_score': best_score,
            'best_state_dict': best_state_dict, 'best_train_metrics': best_train_metrics,
            'best_val_metrics': best_val_metrics, 'best_lr': best_lr}

# (h) CHECKPOINTING

def _run_raw_preds(model: nn.Module, arch_name: str, split: SplitData, device: torch.device,
                   batch_size: int) -> torch.Tensor:
    """Returns raw main-head output over all of split."""
    predict_fn = get_predict_fn(arch_name)
    model.eval()
    idx = np.arange(len(split))
    all_preds = []
    with torch.no_grad():
        for batch_idx in iterate_batches(idx, batch_size, shuffle=False, drop_last=False):
            batch = make_batch(split, batch_idx, device)
            preds, _aux_preds = predict_fn(model, batch)
            all_preds.append(preds.cpu())
    return torch.cat(all_preds)


def _fit_checkpoint_calibration(model: nn.Module, arch_name: str, val: SplitData, val_wl: SplitData,
                                device: torch.device, batch_size: int) -> tuple[float | None, tuple[float, float]]:
    """If classification, fits temperature on val and probit on val_wl. If regression, probit only."""
    output_type = get_output_type(arch_name)
    raw_val_wl = _run_raw_preds(model, arch_name, val_wl, device, batch_size)

    if output_type == 'classification':
        raw_val = _run_raw_preds(model, arch_name, val, device, batch_size)
        temperature = fit_temperature(raw_val, val.result_class.long())

        expected_score = collapse_to_expected_score(torch.softmax(raw_val_wl, dim=1))
        probit_params = fit_binary_probit(expected_score, val_wl.result_cont.long().numpy())
        return temperature, probit_params

    probit_params = fit_binary_probit(raw_val_wl.numpy(), val_wl.result_cont.long().numpy())
    return None, probit_params


def _history_to_dataframe(history: dict) -> pd.DataFrame:
    """Converts history to one-row-per-epoch DataFrame."""
    n_epochs = len(history['epoch_time'])
    return pd.DataFrame({'epoch': range(1, n_epochs + 1), **history})


def save_checkpoint(model: nn.Module, arch_name: str, model_cfg: BaseModelConfig, train_cfg: 'TrainConfig',
                    run_result: dict, val: SplitData, val_wl: SplitData, prep_cfg: PrepConfig, path: str,
                    n_elo_bins: int | None = None, device: torch.device | None = None) -> None:
    """Loads best weights, fits calibration, saves TrainedModel to path."""
    device = device or get_device()
    model = model.to(device)
    model.load_state_dict(run_result['best_state_dict'])

    temperature, probit_params = _fit_checkpoint_calibration(model, arch_name, val, val_wl, device,
                                                             train_cfg.eval_batch_size)
    epoch_history = _history_to_dataframe(run_result['history'])

    checkpoint = TrainedModel(
        arch_name=arch_name,
        model_cfg=model_cfg,
        train_cfg=train_cfg,
        state_dict=run_result['best_state_dict'],
        n_elo_bins=n_elo_bins,
        prep_cfg=prep_cfg,
        best_epoch=run_result['best_epoch'],
        best_score=run_result['best_score'],
        best_train_metrics=run_result['best_train_metrics'],
        best_val_metrics=run_result['best_val_metrics'],
        best_lr=run_result['best_lr'],
        epoch_history=epoch_history,
        temperature=temperature,
        probit_params=probit_params,
    )

    dirname = os.path.dirname(path)
    if dirname:
        os.makedirs(dirname, exist_ok=True)
    torch.save(checkpoint, path)
    temp_str = f'T={temperature:.3f}' if temperature is not None else 'T=n/a (regression)'
    print(f'Saved checkpoint ({arch_name}, {temp_str}, probit={probit_params}) to {path}')

####################
# CLASSES
####################

@dataclass
class TrainConfig:
    """Training run settings."""
    n_epochs: int = 20
    batch_size: int = 64
    eval_batch_size: int = 16384
    lr: float = 1e-3
    weight_decay: float = 0.0
    loss_name: str | None = None

    primary_metric: str = 'loss'
    patience: int | None = 5

    grad_clip_norm: float | None = None
    warmup_prop: float | None = None
    schedule_cfg: dict | None = None

    elo_weighted_loss: bool = False
    elo_weight_alpha: float | None = None

    seed: int = 0
    verbose: bool = True


@dataclass
class TrainedModel:
    """Trained model bundle: weights, configs, calibration, best-epoch metrics, history."""
    arch_name: str
    model_cfg: BaseModelConfig
    train_cfg: TrainConfig
    state_dict: dict
    n_elo_bins: int | None
    prep_cfg: PrepConfig
    best_epoch: int
    best_score: float
    best_train_metrics: dict
    best_val_metrics: dict
    best_lr: float
    epoch_history: pd.DataFrame
    temperature: float | None
    probit_params: tuple[float, float]