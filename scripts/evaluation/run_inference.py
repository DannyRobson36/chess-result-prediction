"""
run_inference.py
Runs a trained checkpoint on a val/test positions CSV, writing predictions in
compare_predictions-ready form: game_id, fen, prob_win, prob_draw, prob_loss, predicted_class.

CLI (all required, no defaults):
    --checkpoint-path   Full path to the .pt checkpoint.
    --input-path        Path to the val/test positions CSV to evaluate.
    --output-path       Full path to write the predictions CSV to.
    --no-calibration    Optional. Use raw (uncalibrated) softmax/sigmoid instead of
                         the fitted temperature stored in the checkpoint.

Run:
    !python run_inference.py --checkpoint-path /content/drive/.../maia2_value_feature_run1.pt 
        --input-path /content/drive/.../val_900k_res_bal_wl.csv 
        --output-path /content/drive/.../val_predictions_maia2_value_feature_run1.csv

Latest changes: 14/08/26:
- Initial commit
"""

import argparse
import os
import sys
import tempfile

import numpy as np
import pandas as pd
import torch

from scripts.features.features import apply_prepared_splits, class_names_for_mode
from scripts.models.model_arch import build_model, get_output_type, get_predict_fn
from scripts.training.training import TrainedModel, iterate_batches, make_batch, get_device
from scripts.utils.utils_eval import apply_temperature, apply_binary_temperature

####################
# CONSTANTS
####################

REQUIRED_OUTPUT_COLS = ['game_id', 'fen', 'prob_win', 'prob_draw', 'prob_loss', 'predicted_class']

# batch size for inference; hand-edit here rather than via CLI
BATCH_SIZE = 1024

####################
# FUNCTIONS
####################

# (a) CLI

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parses the --checkpoint-path, --input-path, --output-path, --no-calibration CLI arguments."""
    parser = argparse.ArgumentParser(
        description='Runs a trained checkpoint on a val/test positions CSV, writing predictions '
                     'in compare_predictions-ready form.'
    )
    parser.add_argument('--checkpoint-path', required=True,
                         help='Full path to the .pt checkpoint (from save_checkpoint).')
    parser.add_argument('--input-path', required=True,
                         help='Path to the val/test positions CSV to evaluate. Must contain mover_result.')
    parser.add_argument('--output-path', required=True,
                         help='Full path (including filename) to write the predictions CSV to.')
    parser.add_argument('--no-calibration', action='store_true',
                         help='Use raw (uncalibrated) softmax/sigmoid instead of the fitted '
                              'temperature stored in the checkpoint.')
    return parser.parse_args(argv)

# (b) INFERENCE

def run_inference_batches(model: torch.nn.Module, arch_name: str, split, batch_size: int,
                           device: torch.device) -> torch.Tensor:
    """Runs model over every row of split, returning concatenated raw model output (logits)."""
    predict_fn = get_predict_fn(arch_name)
    model.eval()
    idx = np.arange(len(split))
    all_preds = []
    with torch.no_grad():
        for batch_idx in iterate_batches(idx, batch_size, shuffle=False, drop_last=False):
            batch = make_batch(split, batch_idx, device)
            all_preds.append(predict_fn(model, batch).cpu())
    return torch.cat(all_preds)


def compute_probs(raw_preds: torch.Tensor, output_type: str, two_way: bool,
                   temperature: float | None) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Converts raw model output into (prob_win, prob_draw, prob_loss, predicted_class) arrays."""
    n = raw_preds.shape[0]

    if output_type == 'classification':
        probs = apply_temperature(raw_preds, temperature) if temperature is not None else torch.softmax(raw_preds, dim=1)
        class_names = class_names_for_mode(two_way)
        predicted_class = np.array(class_names)[probs.argmax(dim=1).numpy()]

        if two_way:
            prob_loss = probs[:, 0].numpy()
            prob_win = probs[:, 1].numpy()
            prob_draw = np.zeros(n, dtype='float64')
        else:
            prob_loss = probs[:, 0].numpy()
            prob_draw = probs[:, 1].numpy()
            prob_win = probs[:, 2].numpy()
    else:
        prob_win_t = apply_binary_temperature(raw_preds, temperature) if temperature is not None else torch.sigmoid(raw_preds)
        prob_win = prob_win_t.numpy()
        prob_loss = 1.0 - prob_win
        prob_draw = np.zeros(n, dtype='float64')
        predicted_class = np.where(prob_win >= 0.5, 'win', 'loss')

    return prob_win, prob_draw, prob_loss, predicted_class

# (c) MAIN

def main(argv: list[str] | None = None) -> None:
    """Loads a checkpoint, applies it to a positions CSV, writes a predictions CSV."""
    args = parse_args(argv)

    if not os.path.exists(args.checkpoint_path):
        sys.exit(f'No checkpoint found at {args.checkpoint_path}.')
    if not os.path.exists(args.input_path):
        sys.exit(f'No input CSV found at {args.input_path}.')

    device = get_device()
    checkpoint: TrainedModel = torch.load(args.checkpoint_path, map_location=device, weights_only=False)
    print(f'Loaded checkpoint: arch_name={checkpoint.arch_name}, two_way={checkpoint.two_way}, '
          f'temperature={checkpoint.temperature:.3f}')

    model = build_model(checkpoint.arch_name, checkpoint.model_cfg,
                         n_elo_bins=checkpoint.n_elo_bins, two_way=checkpoint.two_way)
    model.load_state_dict(checkpoint.state_dict)
    model = model.to(device)

    print(f'Loading positions from: {args.input_path}')
    df = pd.read_csv(args.input_path)
    print(f'Loaded {len(df):,} position(s).')

    with tempfile.TemporaryDirectory() as tmp_dir:
        split = apply_prepared_splits(df, checkpoint.prep_cfg, out_dir=tmp_dir)

        print(f'Running inference on {len(split):,} position(s)...')
        output_type = get_output_type(checkpoint.arch_name)
        raw_preds = run_inference_batches(model, checkpoint.arch_name, split, BATCH_SIZE, device)

        temperature = None if args.no_calibration else checkpoint.temperature
        prob_win, prob_draw, prob_loss, predicted_class = compute_probs(
            raw_preds, output_type, checkpoint.two_way, temperature)

        df_out = pd.DataFrame({
            'game_id': split.game_id,
            'fen': split.fen,
            'prob_win': prob_win,
            'prob_draw': prob_draw,
            'prob_loss': prob_loss,
            'predicted_class': predicted_class,
        })

    assert list(df_out.columns) == REQUIRED_OUTPUT_COLS, 'output columns do not match REQUIRED_OUTPUT_COLS.'

    calib_str = f'calibrated (T={checkpoint.temperature:.3f})' if temperature is not None else 'uncalibrated (raw)'
    print(f'Predictions computed, {calib_str}.')

    out_dir = os.path.dirname(args.output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    df_out.to_csv(args.output_path, index=False)
    print(f'Saved {len(df_out):,} rows to {args.output_path}')

####################
# ENTRY POINT
####################
if __name__ == '__main__':
    main()