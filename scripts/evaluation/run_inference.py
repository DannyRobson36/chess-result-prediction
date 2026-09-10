"""
run_inference.py
Runs a trained checkpoint on a val/test positions CSV, writing predictions in
compare_predictions-ready form: game_id, fen, prob_win, prob_draw, prob_loss, predicted_class.
Auto-detects whether the input contains draws: if so, writes native temperature-calibrated
3-way probabilities; if not, collapses to a single win-probability scalar via --collapse-method,
then either probit-calibrates it or writes it directly, per --probit.

CLI (all required):
    --checkpoint-path   Full path to the .pt checkpoint.
    --input-path        Path to the val/test positions CSV to evaluate.
    --output-path       Full path to write the predictions CSV to.
    --probit            True or False. Only affects decisive-only (no-draw) classification input.
    --collapse-method   'expected_score' or 'renorm_wl'. Only affects decisive-only classification input.

Run:
    !python run_inference.py --checkpoint-path /content/drive/.../maia2_value_feature_run1.pt 
        --input-path /content/drive/.../val_900k_res_bal_wl.csv 
        --output-path /content/drive/.../val_predictions_maia2_value_feature_run1.csv 
        --probit False --collapse-method renorm_wl

Latest changes: 10/09/26:
- Added --collapse-method CLI flag
"""

import argparse
import os
import sys
import tempfile

import numpy as np
import pandas as pd
import torch

from scripts.features.features import apply_prepared_splits
from scripts.models.model_arch import build_model, get_output_type, get_predict_fn
from scripts.training.training import TrainedModel, iterate_batches, make_batch, get_device
from scripts.utils.utils_chess import RESULT_CLASS_NAMES
from scripts.utils.utils_eval import apply_temperature, collapse_to_expected_score, binary_probit_prediction_cols

####################
# CONSTANTS
####################

REQUIRED_OUTPUT_COLS = ['game_id', 'fen', 'prob_win', 'prob_draw', 'prob_loss', 'predicted_class']

# batch size for inference; hand-edit here rather than via CLI
BATCH_SIZE = 1024

COLLAPSE_METHODS = ['expected_score', 'renorm_wl']

####################
# FUNCTIONS
####################

# (a) CLI

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parses the --checkpoint-path, --input-path, --output-path, --probit, --collapse-method CLI arguments."""
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
    parser.add_argument('--probit', required=True, choices=['True', 'False'],
                         help='True applies the fitted binary probit to decisive-only classification '
                              'input; False writes the collapsed win probability directly. No effect on '
                              'has-draws input or on regression architectures.')
    parser.add_argument('--collapse-method', required=True, choices=COLLAPSE_METHODS,
                         help="expected_score: win + 0.5*draw. renorm_wl: win / (win + loss), draw mass "
                              "ignored. Only affects decisive-only classification input; renorm_wl requires "
                              "--probit False.")
    return parser.parse_args(argv)

# (b) INFERENCE

def run_inference_batches(model: torch.nn.Module, arch_name: str, split, batch_size: int,
                           device: torch.device) -> torch.Tensor:
    """Runs model over every row of split, returning concatenated raw main-head output."""
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


def _collapse_renorm_wl(probs: np.ndarray | torch.Tensor) -> np.ndarray:
    """Collapses (loss, draw, win)-ordered 3-way probabilities into win / (win + loss)."""
    if isinstance(probs, torch.Tensor):
        probs = probs.detach().numpy()
    return probs[:, 2] / (probs[:, 2] + probs[:, 0])


def _collapsed_score_prediction_cols(win_prob: np.ndarray) -> dict:
    """Returns prob_win/prob_draw/prob_loss/predicted_class directly from a win probability, uncalibrated."""
    prob_win = win_prob
    prob_loss = 1.0 - win_prob
    prob_draw = np.zeros_like(prob_win)
    predicted_class = np.where(prob_win >= 0.5, 'win', 'loss')
    return {'prob_win': prob_win, 'prob_draw': prob_draw, 'prob_loss': prob_loss,
            'predicted_class': predicted_class}


def compute_probs(raw_preds: torch.Tensor, output_type: str, has_draws: bool, use_probit: bool,
                   collapse_method: str, temperature: float | None,
                   probit_params: tuple[float, float]) -> dict:
    """Converts raw model output into prob_win/prob_draw/prob_loss/predicted_class arrays."""
    if output_type == 'classification':
        if has_draws:
            calibrated = apply_temperature(raw_preds, temperature)
            predicted_class = np.array(RESULT_CLASS_NAMES)[calibrated.argmax(dim=1).numpy()]
            return {
                'prob_win': calibrated[:, 2].numpy(),
                'prob_draw': calibrated[:, 1].numpy(),
                'prob_loss': calibrated[:, 0].numpy(),
                'predicted_class': predicted_class,
            }
        probs_3way = torch.softmax(raw_preds, dim=1)
        if collapse_method == 'expected_score':
            win_prob = collapse_to_expected_score(probs_3way)
        else:
            win_prob = _collapse_renorm_wl(probs_3way)
        if use_probit:
            return binary_probit_prediction_cols(win_prob, probit_params)
        return _collapsed_score_prediction_cols(win_prob)

    return binary_probit_prediction_cols(raw_preds.numpy(), probit_params)

# (c) MAIN

def main(argv: list[str] | None = None) -> None:
    """Loads a checkpoint, applies it to a positions CSV, writes a predictions CSV."""
    args = parse_args(argv)
    use_probit = args.probit == 'True'

    if args.collapse_method == 'renorm_wl' and use_probit:
        sys.exit("--collapse-method renorm_wl requires --probit False "
                  "(the fitted probit is calibrated for the expected_score scale).")

    if not os.path.exists(args.checkpoint_path):
        sys.exit(f'No checkpoint found at {args.checkpoint_path}.')
    if not os.path.exists(args.input_path):
        sys.exit(f'No input CSV found at {args.input_path}.')

    device = get_device()
    checkpoint: TrainedModel = torch.load(args.checkpoint_path, map_location=device, weights_only=False)
    temp_str = f'{checkpoint.temperature:.3f}' if checkpoint.temperature is not None else 'n/a (regression)'
    print(f'Loaded checkpoint: arch_name={checkpoint.arch_name}, temperature={temp_str}, '
          f'probit_params={checkpoint.probit_params}')

    model = build_model(checkpoint.arch_name, checkpoint.model_cfg, n_elo_bins=checkpoint.n_elo_bins)
    model.load_state_dict(checkpoint.state_dict)
    model = model.to(device)

    output_type = get_output_type(checkpoint.arch_name)
    if output_type == 'regression' and (not use_probit or args.collapse_method != 'expected_score'):
        print('Note: --probit and --collapse-method have no effect on regression architectures; '
              'probit is always applied directly to the raw output.')

    print(f'Loading positions from: {args.input_path}')
    df = pd.read_csv(args.input_path)
    print(f'Loaded {len(df):,} position(s).')

    with tempfile.TemporaryDirectory() as tmp_dir:
        split = apply_prepared_splits(df, checkpoint.prep_cfg, out_dir=tmp_dir)

        has_draws = bool((split.result_class == 1).any())
        if has_draws:
            mode_str = 'native 3-way (temperature-calibrated)'
        else:
            collapse_str = ('expected score (win + 0.5*draw)' if args.collapse_method == 'expected_score'
                             else 'win / (win + loss), draw mass ignored')
            calib_str = 'probit-calibrated' if use_probit else 'uncalibrated'
            mode_str = f'collapsed via {collapse_str}, {calib_str}, prob_draw=0'
        print(f'Input contains draws: {has_draws} -- writing {mode_str}.')

        print(f'Running inference on {len(split):,} position(s)...')
        raw_preds = run_inference_batches(model, checkpoint.arch_name, split, BATCH_SIZE, device)

        probs = compute_probs(raw_preds, output_type, has_draws, use_probit, args.collapse_method,
                               checkpoint.temperature, checkpoint.probit_params)

        df_out = pd.DataFrame({
            'game_id': split.game_id,
            'fen': split.fen,
            **probs,
        })

    assert list(df_out.columns) == REQUIRED_OUTPUT_COLS, 'output columns do not match REQUIRED_OUTPUT_COLS.'

    print('Predictions computed.')

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