"""
training_diagnostics.py
Quick post-training diagnostics at notebook level - history plots and per-elo-bin metrics.

Latest changes: 13/08/26:
- Initial commit
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report

from scripts.features.features import SplitData, class_names_for_mode
from scripts.models.model_arch import get_output_type, get_predict_fn
from scripts.training.training import (
    iterate_batches, make_batch, get_targets, get_device, build_loss, resolve_loss_name,
)

####################
# FUNCTIONS
####################

# (a) TRAINING HISTORY

def plot_training_history(out: dict, primary_metric: str = "loss") -> None:
    """Plots train/val loss, accuracy, and macro_f1 side by side, marking the best epoch."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, key in zip(axes, ["loss", "accuracy", "macro_f1"]):
        ax.plot(out["history"][f"train_{key}"], label="train")
        ax.plot(out["history"][f"val_{key}"], label="val")
        ax.axvline(out["best_epoch"] - 1, color="grey", linestyle="--")
        ax.set_title(key)
        ax.legend()
    plt.tight_layout()
    plt.show()

    print(f"best epoch: {out['best_epoch']}  best {primary_metric}: {out['best_score']:.4f}")


# (b) PER-ELO-BIN METRICS

def evaluate_by_elo_bin(model: nn.Module, arch_name: str, out: dict, split: SplitData, two_way: bool,
                         loss_name: str | None = None, batch_size: int = 64,
                         device: torch.device | None = None) -> pd.DataFrame:
    """Restores best weights from out, then computes loss/accuracy/macro_f1 per elo bin on split."""
    device = device or get_device()
    model = model.to(device)
    model.load_state_dict(out["best_state_dict"])

    output_type = get_output_type(arch_name)
    if output_type == "regression" and not two_way:
        raise ValueError("Regression-output architectures are only valid for two-way (no-draw) data; "
                          "got two_way=False, so split.result_cont is not guaranteed to be pure {0.0, 1.0}.")

    loss_fn = build_loss(resolve_loss_name(output_type, loss_name))
    predict_fn = get_predict_fn(arch_name)
    class_names = class_names_for_mode(two_way)

    model.eval()
    idx = np.arange(len(split))
    all_preds, all_targets, all_losses, all_bins = [], [], [], []
    with torch.no_grad():
        for batch_idx in iterate_batches(idx, batch_size, shuffle=False, drop_last=False):
            batch = make_batch(split, batch_idx, device)
            targets = get_targets(split, batch_idx, output_type, two_way, device)
            preds = predict_fn(model, batch)
            per_sample = loss_fn(preds, targets)

            if output_type == "classification":
                pred_class = preds.argmax(dim=1)
            else:
                pred_class = (torch.sigmoid(preds) >= 0.5).long()
                targets = targets.long()

            all_preds.append(pred_class.cpu())
            all_targets.append(targets.cpu())
            all_losses.append(per_sample.cpu())
            all_bins.append(split.elo_mean_bin[batch_idx])

    preds_arr = torch.cat(all_preds).numpy()
    targs_arr = torch.cat(all_targets).numpy()
    losses_arr = torch.cat(all_losses).numpy()
    bins_arr = torch.cat(all_bins).numpy()

    rows = []
    for i in range(split.n_elo_bins):
        mask = bins_arr == i
        if mask.sum() == 0:
            continue
        report = classification_report(targs_arr[mask], preds_arr[mask], labels=list(range(len(class_names))),
                                        target_names=class_names, output_dict=True, zero_division=0)
        rows.append({
            "elo_bin": i,
            "bin_label": split.bin_labels[i],
            "loss": float(losses_arr[mask].mean()),
            "accuracy": accuracy_score(targs_arr[mask], preds_arr[mask]),
            "macro_f1": report["macro avg"]["f1-score"],
            "n": int(mask.sum()),
        })

    return pd.DataFrame(rows)