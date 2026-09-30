"""
training_diagnostics.py
Quick post-training diagnostics at notebook level - history plots and per-elo-bin metrics.

Latest changes: 20/08/26:
- evaluate_by_elo_bin now unpacks predict_fn's (preds, aux_preds) return
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report

from src.features.features import SplitData
from src.models.model_arch import get_output_type, get_predict_fn
from src.training.training import (
    iterate_batches, make_batch, get_targets, get_device, build_loss, resolve_loss_name,
)
from src.utils.utils_chess import RESULT_CLASS_NAMES

####################
# FUNCTIONS
####################

# (a) TRAINING HISTORY

def plot_training_history(out: dict, primary_metric: str = "loss") -> None:
    """Plots train/val history for every metric present, marking the best epoch."""
    keys = [k[len("train_"):] for k in out["history"] if k.startswith("train_")]
    fig, axes = plt.subplots(1, len(keys), figsize=(5 * len(keys), 4), squeeze=False)
    axes = axes[0]
    for ax, key in zip(axes, keys):
        ax.plot(out["history"][f"train_{key}"], label="train")
        ax.plot(out["history"][f"val_{key}"], label="val")
        ax.axvline(out["best_epoch"] - 1, color="grey", linestyle="--")
        ax.set_title(key)
        ax.legend()
    plt.tight_layout()
    plt.show()

    print(f"best epoch: {out['best_epoch']}  best {primary_metric}: {out['best_score']:.4f}")


# (b) PER-ELO-BIN METRICS

def evaluate_by_elo_bin(model: nn.Module, arch_name: str, out: dict, split: SplitData,
                         loss_name: str | None = None, batch_size: int = 64,
                         device: torch.device | None = None) -> pd.DataFrame:
    """Restores best weights from out, then computes loss per elo bin on split (plus accuracy/
    macro_f1 for classification only)."""
    device = device or get_device()
    model = model.to(device)
    model.load_state_dict(out["best_state_dict"])

    output_type = get_output_type(arch_name)
    loss_fn = build_loss(resolve_loss_name(output_type, loss_name))
    predict_fn = get_predict_fn(arch_name)

    model.eval()
    idx = np.arange(len(split))
    all_preds, all_targets, all_losses, all_bins = [], [], [], []
    with torch.no_grad():
        for batch_idx in iterate_batches(idx, batch_size, shuffle=False, drop_last=False):
            batch = make_batch(split, batch_idx, device)
            targets = get_targets(split, batch_idx, output_type, device)
            preds, _aux_preds = predict_fn(model, batch)
            per_sample = loss_fn(preds, targets)

            all_losses.append(per_sample.cpu())
            all_bins.append(split.elo_mean_bin[batch_idx])
            if output_type == "classification":
                all_preds.append(preds.argmax(dim=1).cpu())
                all_targets.append(targets.cpu())

    losses_arr = torch.cat(all_losses).numpy()
    bins_arr = torch.cat(all_bins).numpy()
    if output_type == "classification":
        preds_arr = torch.cat(all_preds).numpy()
        targs_arr = torch.cat(all_targets).numpy()

    rows = []
    for i in range(split.n_elo_bins):
        mask = bins_arr == i
        if mask.sum() == 0:
            continue
        row = {"elo_bin": i, "bin_label": split.bin_labels[i],
               "loss": float(losses_arr[mask].mean()), "n": int(mask.sum())}
        if output_type == "classification":
            report = classification_report(targs_arr[mask], preds_arr[mask],
                                            labels=list(range(len(RESULT_CLASS_NAMES))),
                                            target_names=RESULT_CLASS_NAMES, output_dict=True, zero_division=0)
            row["accuracy"] = accuracy_score(targs_arr[mask], preds_arr[mask])
            row["macro_f1"] = report["macro avg"]["f1-score"]
        rows.append(row)

    return pd.DataFrame(rows)