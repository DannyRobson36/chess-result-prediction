"""
training.py
Batches a SplitData, runs one training loop with early stopping, saves a checkpoint for inference.

Latest changes: 12/08/26:
- Regression now allows for accuracy predictions via sigmoid(logits)
"""

import os
import random
import time
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, classification_report

from scripts.features.features import SplitData, PrepConfig, class_names_for_mode, remap_targets
from scripts.models.model_arch import get_output_type, get_predict_fn
from scripts.models.model_config import BaseModelConfig
from scripts.utils.utils_eval import metric_mode, is_better, fit_temperature, fit_binary_temperature

####################
# CONSTANTS
####################

LOSS_REGISTRY: dict[str, type] = {
    "cross_entropy": nn.CrossEntropyLoss,
    "bce_logits": nn.BCEWithLogitsLoss,
    "mse": nn.MSELoss,
}

DEFAULT_LOSS_BY_OUTPUT_TYPE = {
    "classification": "cross_entropy",
    "regression": "bce_logits",
}

SCHEDULER_REGISTRY: dict[str, type] = {
    "step": torch.optim.lr_scheduler.StepLR,
    "cosine": torch.optim.lr_scheduler.CosineAnnealingLR,
    "plateau": torch.optim.lr_scheduler.ReduceLROnPlateau,
}

ELO_WEIGHT_DEFAULT_ALPHA = 0.5
ELO_WEIGHT_MAX_RATIO = 5.0

####################
# FUNCTIONS
####################

# (a) REPRODUCIBILITY & DEVICE

def set_seed(seed: int) -> None:
    """Seeds python, numpy, and torch (cpu + cuda)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_device() -> torch.device:
    """Returns cuda if available, else cpu."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

def _sync_device(device: torch.device) -> None:
    """Blocks until queued GPU work on device finishes."""
    if device.type == "cuda":
        torch.cuda.synchronize()

# (b) BATCHING

def iterate_batches(idx: np.ndarray, batch_size: int, shuffle: bool = True,
                     seed: int | None = None, drop_last: bool = False):
    """Yields index arrays covering each idx once."""
    idx = np.asarray(idx)
    if shuffle:
        rng = np.random.default_rng(seed)
        idx = idx[rng.permutation(len(idx))]

    n = len(idx)
    end = n - (n % batch_size) if drop_last else n
    for start in range(0, end, batch_size):
        yield idx[start:start + batch_size]

def make_batch(split: SplitData, idx: np.ndarray, device: torch.device) -> dict:
    """Slices split at idx, moves to device, and casts compressed storage dtypes to what PyTorch ops expect."""
    batch = {
        "boards": split.boards[idx].to(device).float(),
        "board_token_ids": split.board_token_ids[idx].to(device).long(),
        "elo_mean_bin": split.elo_mean_bin[idx].to(device).long(),
        "elo_self_bin": split.elo_self_bin[idx].to(device).long(),
        "elo_oppo_bin": split.elo_oppo_bin[idx].to(device).long(),
    }
    for name, tensor in split.features.items():
        batch[name] = tensor[idx].to(device)
    return batch

def probe_idx(split: SplitData, n: int, seed: int = 0) -> np.ndarray:
    """Returns a random subsample of row positions from split."""
    rng = np.random.default_rng(seed)
    n = min(n, len(split))
    return rng.choice(len(split), size=n, replace=False)

def _get_targets(split: SplitData, batch_idx: np.ndarray, output_type: str,
                  two_way: bool, device: torch.device) -> torch.Tensor:
    """Returns the target tensor matching output_type, sliced at batch_idx and moved to device."""
    if output_type == "classification":
        return remap_targets(split.result_class[batch_idx], two_way).to(device).long()
    return split.result_cont[batch_idx].to(device)

# (c) LOSS & SCHEDULER

def build_loss(loss_name: str, **kwargs) -> nn.Module:
    """Builds a per-sample (reduction='none') loss module by name from LOSS_REGISTRY."""
    if loss_name not in LOSS_REGISTRY:
        raise ValueError(f"Unknown loss '{loss_name}', choose from {list(LOSS_REGISTRY)}")
    kwargs.setdefault("reduction", "none")
    return LOSS_REGISTRY[loss_name](**kwargs)

def _resolve_loss_name(output_type: str, loss_name: str | None) -> str:
    """Returns loss_name if given, else the standard default loss for output_type."""
    return loss_name if loss_name is not None else DEFAULT_LOSS_BY_OUTPUT_TYPE[output_type]

def build_scheduler(optimizer, schedule_cfg: dict | None, n_epochs: int, mode: str):
    """Builds an LR scheduler by name from schedule_cfg, or None for a fixed lr."""
    if schedule_cfg is None:
        return None
    schedule_cfg = dict(schedule_cfg)
    kind = schedule_cfg.pop("type")
    if kind not in SCHEDULER_REGISTRY:
        raise ValueError(f"Unknown scheduler '{kind}', choose from {list(SCHEDULER_REGISTRY)}")
    if kind == "cosine":
        schedule_cfg.setdefault("T_max", n_epochs)
    if kind == "plateau":
        schedule_cfg.setdefault("mode", mode)
    return SCHEDULER_REGISTRY[kind](optimizer, **schedule_cfg)

# (d) ELO-BIN LOSS WEIGHTING

def compute_elo_bin_weights(split: SplitData, alpha: float = 1.0, max_ratio: float | None = ELO_WEIGHT_MAX_RATIO,
                             device: torch.device | None = None) -> torch.Tensor:
    """Returns mean-normalised (1/count)**alpha weights per elo bin, capped to max_ratio times the smallest weight."""
    counts = torch.bincount(split.elo_mean_bin.long(), minlength=split.n_elo_bins).float().clamp(min=1)
    weights = (1.0 / counts) ** alpha
    weights = weights / weights.mean()
    if max_ratio is not None:
        weights = weights.clamp(max=weights.min() * max_ratio)
    return weights.to(device) if device is not None else weights

def _resolve_bin_weights(train: SplitData, cfg: "TrainConfig", device: torch.device) -> torch.Tensor | None:
    """Builds elo-bin loss weights from cfg, or None if elo_weighted_loss is off."""
    if not cfg.elo_weighted_loss:
        if cfg.elo_weight_alpha is not None:
            raise ValueError("cfg.elo_weight_alpha was set but cfg.elo_weighted_loss is False.")
        return None
    alpha = cfg.elo_weight_alpha if cfg.elo_weight_alpha is not None else ELO_WEIGHT_DEFAULT_ALPHA
    return compute_elo_bin_weights(train, alpha=alpha, device=device)

# (e) LR WARMUP

def apply_warmup_lr(optimizer, base_lr: float, step: int, warmup_steps: int) -> None:
    """Linearly scales optimizer's lr from 0 to base_lr over warmup_steps, in-place."""
    scale = min(1.0, (step + 1) / warmup_steps)
    for group in optimizer.param_groups:
        group["lr"] = base_lr * scale

def _resolve_warmup_steps(warmup_prop: float | None, n_train: int, batch_size: int, n_epochs: int) -> int | None:
    """Converts warmup_prop (fraction of total training batches) into an absolute warmup_steps count."""
    if warmup_prop is None:
        return None
    batches_per_epoch = n_train // batch_size
    total_steps = batches_per_epoch * n_epochs
    return max(1, round(warmup_prop * total_steps))

# (f) TRAIN & EVALUATE ONE EPOCH

def train_one_epoch(model: nn.Module, arch_name: str, loss_fn: nn.Module, optimizer, split: SplitData,
                     idx: np.ndarray, output_type: str, two_way: bool = False, batch_size: int = 64,
                     seed: int | None = None, device: torch.device | None = None, drop_last: bool = True,
                     bin_weights: torch.Tensor | None = None, grad_clip_norm: float | None = None,
                     warmup_steps: int | None = None, base_lr: float | None = None,
                     step_counter: list | None = None) -> None:
    """Runs one training epoch: forward, per-sample loss (optionally elo-bin weighted), backward, step."""
    device = device or get_device()
    predict_fn = get_predict_fn(arch_name)
    model.train()
    for batch_idx in iterate_batches(idx, batch_size, shuffle=True, seed=seed, drop_last=drop_last):
        if warmup_steps is not None and step_counter[0] < warmup_steps:
            apply_warmup_lr(optimizer, base_lr, step_counter[0], warmup_steps)

        batch = make_batch(split, batch_idx, device)
        targets = _get_targets(split, batch_idx, output_type, two_way, device)

        optimizer.zero_grad()
        preds = predict_fn(model, batch)
        per_sample = loss_fn(preds, targets)
        if bin_weights is not None:
            loss = (per_sample * bin_weights[batch["elo_mean_bin"]]).mean()
        else:
            loss = per_sample.mean()
        loss.backward()
        if grad_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
        optimizer.step()

        if step_counter is not None:
            step_counter[0] += 1

def evaluate_with_loss(model: nn.Module, arch_name: str, split: SplitData, idx: np.ndarray, loss_fn: nn.Module,
                        output_type: str, two_way: bool = False, class_names: list[str] | None = None,
                        batch_size: int = 64, device: torch.device | None = None) -> dict:
    """Evaluates a model on idx: loss, accuracy, and macro_f1."""
    if output_type == "regression" and not two_way:
        raise ValueError("Regression-output architectures are only valid for two-way (no-draw) data; "
                          "got two_way=False, so split.result_cont is not guaranteed to be pure {0.0, 1.0}.")

    device = device or get_device()
    predict_fn = get_predict_fn(arch_name)
    model.eval()
    total_loss, total_n = 0.0, 0
    all_preds, all_targets = [], []
    with torch.no_grad():
        for batch_idx in iterate_batches(idx, batch_size, shuffle=False, drop_last=False):
            batch = make_batch(split, batch_idx, device)
            targets = _get_targets(split, batch_idx, output_type, two_way, device)
            preds = predict_fn(model, batch)
            per_sample = loss_fn(preds, targets)

            total_loss += per_sample.sum().item()
            total_n += len(batch_idx)

            if output_type == "classification":
                all_preds.append(preds.argmax(dim=1).cpu())
                all_targets.append(targets.cpu())
            else:
                all_preds.append((torch.sigmoid(preds) >= 0.5).long().cpu())
                all_targets.append(targets.long().cpu())

    metrics = {"loss": total_loss / total_n}
    if class_names is None:
        class_names = class_names_for_mode(two_way)
    preds_arr = torch.cat(all_preds).numpy()
    targs_arr = torch.cat(all_targets).numpy()
    report = classification_report(targs_arr, preds_arr, target_names=class_names,
                                    output_dict=True, zero_division=0)
    metrics["accuracy"] = accuracy_score(targs_arr, preds_arr)
    metrics["macro_f1"] = report["macro avg"]["f1-score"]
    return metrics

# (g) FULL TRAINING LOOP

def run_training(model: nn.Module, arch_name: str, train: SplitData, val: SplitData, cfg: "TrainConfig",
                  two_way: bool = False, train_probe_idx: np.ndarray | None = None,
                  val_probe_idx: np.ndarray | None = None, device: torch.device | None = None) -> dict:
    """Trains with early stopping on cfg.primary_metric; returns history and best epoch (1-indexed) weights."""
    device = device or get_device()
    model = model.to(device)
    output_type = get_output_type(arch_name)
    mode = metric_mode(cfg.primary_metric)

    loss_name = _resolve_loss_name(output_type, cfg.loss_name)
    loss_fn = build_loss(loss_name)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = build_scheduler(optimizer, cfg.schedule_cfg, cfg.n_epochs, mode)
    bin_weights = _resolve_bin_weights(train, cfg, device)
    warmup_steps = _resolve_warmup_steps(cfg.warmup_prop, len(train), cfg.batch_size, cfg.n_epochs)

    keys = ["loss", "accuracy", "macro_f1"]
    history = {f"{prefix}_{k}": [] for prefix in ("train", "val") for k in keys}
    history["epoch_time"] = []
    history["lr"] = []

    if train_probe_idx is None:
        train_probe_idx = probe_idx(train, n=min(5000, len(train)), seed=cfg.seed)
    val_idx = val_probe_idx if val_probe_idx is not None else np.arange(len(val))
    train_idx = np.arange(len(train))

    best_score, best_epoch, best_state_dict = None, None, None
    epochs_without_improvement = 0
    step_counter = [0]

    for epoch in range(cfg.n_epochs):
        _sync_device(device)
        epoch_start = time.time()

        train_one_epoch(model, arch_name, loss_fn, optimizer, train, train_idx, output_type,
                         two_way=two_way, batch_size=cfg.batch_size, seed=cfg.seed + epoch, device=device,
                         drop_last=True, bin_weights=bin_weights, grad_clip_norm=cfg.grad_clip_norm,
                         warmup_steps=warmup_steps, base_lr=cfg.lr, step_counter=step_counter)

        train_metrics = evaluate_with_loss(model, arch_name, train, train_probe_idx, loss_fn, output_type,
                                            two_way=two_way, batch_size=cfg.batch_size, device=device)
        val_metrics = evaluate_with_loss(model, arch_name, val, val_idx, loss_fn, output_type,
                                          two_way=two_way, batch_size=cfg.batch_size, device=device)

        _sync_device(device)
        epoch_time = time.time() - epoch_start

        for key in keys:
            history[f"train_{key}"].append(train_metrics[key])
            history[f"val_{key}"].append(val_metrics[key])
        history["epoch_time"].append(epoch_time)
        history["lr"].append(optimizer.param_groups[0]["lr"])

        if cfg.verbose:
            parts = [f"epoch {epoch+1}/{cfg.n_epochs}"]
            for key in keys:
                parts.append(f"train_{key}={train_metrics[key]:.4f}")
                parts.append(f"val_{key}={val_metrics[key]:.4f}")
            parts.append(f"time={epoch_time:.2f}s")
            print("  ".join(parts))

        score = val_metrics[cfg.primary_metric]
        if best_score is None or is_better(score, best_score, mode):
            best_score, best_epoch = score, epoch + 1
            best_state_dict = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if cfg.patience is not None and epochs_without_improvement >= cfg.patience:
                if cfg.verbose:
                    print(f"Early stopping at epoch {epoch+1} "
                          f"(no improvement in {cfg.primary_metric} for {cfg.patience} epochs)")
                break

        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(score)
            else:
                scheduler.step()

    return {"history": history, "best_epoch": best_epoch, "best_score": best_score,
            "best_state_dict": best_state_dict}

# (h) CHECKPOINTING

def _fit_checkpoint_temperature(model: nn.Module, arch_name: str, val: SplitData, two_way: bool,
                                 device: torch.device, batch_size: int = 256) -> float:
    """Runs the (already best-weights-loaded) model on val and fits a temperature for its output_type."""
    output_type = get_output_type(arch_name)
    predict_fn = get_predict_fn(arch_name)
    model.eval()
    idx = np.arange(len(val))
    all_preds = []
    with torch.no_grad():
        for batch_idx in iterate_batches(idx, batch_size, shuffle=False, drop_last=False):
            batch = make_batch(val, batch_idx, device)
            all_preds.append(predict_fn(model, batch).cpu())
    preds = torch.cat(all_preds)

    if output_type == "classification":
        targets = remap_targets(val.result_class, two_way).long()
        return fit_temperature(preds, targets)
    if not two_way:
        raise ValueError("Regression-output architectures are only valid for two-way (no-draw) data; "
                          "got two_way=False, so val.result_cont is not guaranteed to be pure {0.0, 1.0}.")
    targets = val.result_cont.long()
    return fit_binary_temperature(preds, targets)

def save_checkpoint(model: nn.Module, arch_name: str, model_cfg: BaseModelConfig, run_result: dict,
                     val: SplitData, prep_cfg: PrepConfig, two_way: bool, path: str,
                     n_elo_bins: int | None = None, device: torch.device | None = None) -> None:
    """Loads best_state_dict, fits a temperature on val, and saves everything needed for inference to path."""
    device = device or get_device()
    model = model.to(device)
    model.load_state_dict(run_result["best_state_dict"])

    temperature = _fit_checkpoint_temperature(model, arch_name, val, two_way, device)

    checkpoint = TrainedModel(
        arch_name=arch_name,
        model_cfg=model_cfg,
        state_dict=run_result["best_state_dict"],
        n_elo_bins=n_elo_bins,
        two_way=two_way,
        prep_cfg=prep_cfg,
        best_epoch=run_result["best_epoch"],
        best_score=run_result["best_score"],
        temperature=temperature,
    )

    dirname = os.path.dirname(path)
    if dirname:
        os.makedirs(dirname, exist_ok=True)
    torch.save(checkpoint, path)
    print(f"Saved checkpoint ({arch_name}, temperature={temperature:.3f}) to {path}")

####################
# CLASSES
####################

@dataclass
class TrainConfig:
    """Training run settings: optimisation, schedule, early stopping, elo-bin reweighting."""
    n_epochs: int = 20
    batch_size: int = 64
    lr: float = 1e-3
    weight_decay: float = 0.0
    loss_name: str | None = None

    primary_metric: str = "loss"
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
    """Everything needed to reconstruct a trained model and run inference on new data."""
    arch_name: str
    model_cfg: BaseModelConfig
    state_dict: dict
    n_elo_bins: int | None
    two_way: bool
    prep_cfg: PrepConfig
    best_epoch: int
    best_score: float
    temperature: float