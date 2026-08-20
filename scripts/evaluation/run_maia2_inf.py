"""
run_maia2_inf.py
Maia2 evaluation of one split's positions, writing both raw (Maia2's own mover-perspective
win probability) and probit-calibrated predictions in compare_predictions-ready form, plus
a JSON record of the fitted calibration.

Run:
    !python run_maia2_inf.py --input-path /path/to/val_900k_res_bal.csv --calib-path /path/to/val_900k_res_bal_wl.csv
        --split val
    !python run_maia2_inf.py --input-path /path/to/val_900k_res_bal.csv --calib-path /path/to/val_900k_res_bal_wl.csv
        --split val --device gpu

CLI:
    --input-path   Path to the split's positions CSV to score. Must contain game_id, fen,
                   next_move, mover_elo, opponent_elo.
    --calib-path   Path to val_wl's positions CSV, decisive-only. Same columns plus mover_result.
                   Used only to fit probit calibration.
    --split        Which split --input-path is: val or test. Used to name output files.
    --output-dir   Folder to write predictions into (calibration JSON goes into a calib/
                   subfolder of this). Default: PREDICTIONS_DIR (config.py).
    --device       "cpu" or "gpu". Default: "cpu".

Latest changes: 19/08/26:
- Calibration JSON now written to output-dir/calib/
"""

import os
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from scripts.config import PREDICTIONS_DIR
from scripts.utils.utils_eval import fit_binary_probit, binary_probit_prediction_cols

import argparse
import importlib.util
import json
import subprocess
from datetime import datetime

import numpy as np
import pandas as pd
import chess

####################
# CONSTANTS
####################

# Maia2 time-control variant to load. Not exposed as a CLI arg since only device
# was asked to be adjustable here.
MODEL_TYPE = 'rapid'

BATCH_SIZE = 1024
NUM_WORKERS = os.cpu_count() or 1

REQUIRED_SCORE_COLUMNS = ['game_id', 'fen', 'next_move', 'mover_elo', 'opponent_elo']
REQUIRED_CALIB_COLUMNS = REQUIRED_SCORE_COLUMNS + ['mover_result']

####################
# FUNCTIONS
####################

# (a) VALIDATION

def _validate_columns(columns, required: list[str], path: str) -> None:
    """Exits with a clear message if any required column is missing from columns."""
    missing = [c for c in required if c not in columns]
    if missing:
        sys.exit(f'{path} is missing required column(s): {missing}. Required: {required}.')

# (b) DEPENDENCIES -- maia2 needs pyzstd first, then itself with --no-deps

def ensure_maia2_installed() -> None:
    """Installs pyzstd then maia2 (--no-deps) if maia2 isn't already importable."""
    if importlib.util.find_spec('maia2') is not None:
        return
    print('maia2 not found -- installing (pyzstd, then maia2 --no-deps)...')
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'pyzstd', '-q'])
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '--no-deps', 'maia2', '-q'])
    print('maia2 installed successfully.')

# (c) POSITION LOADING

def load_positions(input_path: str, required: list[str]) -> pd.DataFrame:
    """Loads input_path, keeping only the required columns."""
    header = pd.read_csv(input_path, nrows=0).columns
    _validate_columns(header, required, input_path)
    return pd.read_csv(input_path, usecols=required)

# (d) MOVER-PERSPECTIVE HELPERS

def mover_is_white(fens: pd.Series) -> np.ndarray:
    """Returns a bool array: True where the side to move in fen is white."""
    return np.array([chess.Board(f).turn == chess.WHITE for f in fens])

def count_terminal(fens: pd.Series) -> int:
    """Returns how many fens are already game-over positions."""
    return sum(chess.Board(f).is_game_over() for f in fens)

# (e) MAIA2 INFERENCE

def run_maia2_predictions(df: pd.DataFrame, model, maia2_inference, batch_size: int,
                           num_workers: int) -> np.ndarray:
    """Runs Maia2 inference on df, returning mover-perspective win probabilities. Maia2's own
    win_probs output is white-perspective; flipped here to mover-perspective."""
    maia2_input = df.rename(columns={
        'fen': 'board', 'next_move': 'move', 'mover_elo': 'active_elo',
    })[['board', 'move', 'active_elo', 'opponent_elo']]

    data, _acc = maia2_inference.inference_batch(
        maia2_input, model, verbose=1, batch_size=batch_size, num_workers=num_workers,
    )
    # _acc is Maia2's move-prediction accuracy -- unused, this script is result-prediction only.

    assert len(data) == len(df), (
        f'Row count mismatch: Maia2 returned {len(data)} rows for {len(df)} input rows.'
    )
    assert (data['board'].to_numpy() == df['fen'].to_numpy()).all(), (
        'board/fen mismatch -- Maia2 inference did not preserve row order.'
    )

    white_win_prob = data['win_probs'].to_numpy(dtype='float64')
    is_white = mover_is_white(df['fen'])
    return np.where(is_white, white_win_prob, 1.0 - white_win_prob)

# (f) EVAL -> PREDICTION COLUMNS

def raw_prediction_cols(mover_win_prob: np.ndarray) -> dict:
    """Returns prob_win/prob_draw/prob_loss/predicted_class directly from Maia2's own
    (unfitted) mover-perspective win probability."""
    prob_loss = 1.0 - mover_win_prob
    prob_draw = np.zeros_like(mover_win_prob)
    predicted_class = np.where(mover_win_prob >= 0.5, 'win', 'loss')
    return {'prob_win': mover_win_prob, 'prob_draw': prob_draw, 'prob_loss': prob_loss,
            'predicted_class': predicted_class}

# (g) CLI

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parses --input-path, --calib-path, --split, --output-dir, --device CLI arguments."""
    parser = argparse.ArgumentParser(
        description='Maia2 evaluation of one split, writing both raw (Maia2\'s own win '
                     'probability) and probit-calibrated predictions in compare_predictions-ready form.'
    )
    parser.add_argument('--input-path', required=True,
                         help='Path to the split\'s positions CSV to score. Must contain game_id, '
                              'fen, next_move, mover_elo, opponent_elo.')
    parser.add_argument('--calib-path', required=True,
                         help='Path to val_wl\'s positions CSV (decisive-only), used to fit probit '
                              'calibration. Same columns plus mover_result.')
    parser.add_argument('--split', required=True, choices=['val', 'test'],
                         help='Which split --input-path is, used to name the output files.')
    parser.add_argument('--output-dir', default=PREDICTIONS_DIR, help=f'Default: {PREDICTIONS_DIR}')
    parser.add_argument('--device', default='cpu', choices=['cpu', 'gpu'],
                         help='Device to run Maia2 on. Default: cpu.')
    return parser.parse_args(argv)

def main(argv: list[str] | None = None) -> None:
    """Fits probit calibration on --calib-path, evaluates --input-path (reusing the calibration
    pass if the two paths are the same file), writes raw and calibrated predictions plus a
    calibration-record JSON."""
    args = parse_args(argv)

    if not os.path.exists(args.input_path):
        sys.exit(f'No input CSV found at {args.input_path}.')
    if not os.path.exists(args.calib_path):
        sys.exit(f'No calibration CSV found at {args.calib_path}.')

    ensure_maia2_installed()
    from maia2 import model as maia2_model_module
    from maia2 import inference as maia2_inference

    print(f'Loading Maia2 ({MODEL_TYPE}) on device={args.device}...')
    maia2_model = maia2_model_module.from_pretrained(type=MODEL_TYPE, device=args.device)

    print(f'Loading calibration positions from: {args.calib_path}')
    df_calib = load_positions(args.calib_path, REQUIRED_CALIB_COLUMNS)
    if (df_calib['mover_result'] == 0.5).any():
        sys.exit(f'{args.calib_path} contains drawn (mover_result=0.5) rows -- calibration must '
                  f'be fit on decisive-only (val_wl) data.')
    print(f'Loaded {len(df_calib):,} calibration position(s).')
    print(f'There were {count_terminal(df_calib["fen"])} terminal calibration position(s) (expected 0).')

    print(f'Analysing {len(df_calib):,} calibration positions with Maia2 '
          f'(batch_size={BATCH_SIZE}, num_workers={NUM_WORKERS})...')
    calib_mover_win_prob = run_maia2_predictions(df_calib, maia2_model, maia2_inference, BATCH_SIZE, NUM_WORKERS)

    calib_targets = df_calib['mover_result'].to_numpy(dtype='int64')
    probit_params = fit_binary_probit(calib_mover_win_prob, calib_targets)
    print(f'Fitted probit calibration: c={probit_params[0]:.4f}, sigma={probit_params[1]:.4f}')

    same_file = os.path.realpath(args.input_path) == os.path.realpath(args.calib_path)
    if same_file:
        print('--input-path and --calib-path are the same file -- reusing calibration '
              'evaluations, skipping a second pass.')
        df_score = df_calib
        score_mover_win_prob = calib_mover_win_prob
    else:
        print(f'Loading scored positions from: {args.input_path}')
        df_score = load_positions(args.input_path, REQUIRED_SCORE_COLUMNS)
        print(f'Loaded {len(df_score):,} scored position(s).')
        print(f'There were {count_terminal(df_score["fen"])} terminal scored position(s) (expected 0).')

        print(f'Analysing {len(df_score):,} scored positions with Maia2 '
              f'(batch_size={BATCH_SIZE}, num_workers={NUM_WORKERS})...')
        score_mover_win_prob = run_maia2_predictions(df_score, maia2_model, maia2_inference, BATCH_SIZE, NUM_WORKERS)

    base = {'game_id': df_score['game_id'].to_numpy(), 'fen': df_score['fen'].to_numpy()}
    output_cols = ['game_id', 'fen', 'prob_win', 'prob_draw', 'prob_loss', 'predicted_class']

    df_raw = pd.DataFrame({**base, **raw_prediction_cols(score_mover_win_prob)})
    df_calibrated = pd.DataFrame({**base, **binary_probit_prediction_cols(score_mover_win_prob, probit_params)})

    assert list(df_raw.columns) == output_cols, 'raw output columns do not match output_cols.'
    assert list(df_calibrated.columns) == output_cols, 'calibrated output columns do not match output_cols.'

    os.makedirs(args.output_dir, exist_ok=True)
    calib_dir = os.path.join(args.output_dir, 'calib')
    os.makedirs(calib_dir, exist_ok=True)

    raw_path = os.path.join(args.output_dir, f'{args.split}_predictions_maia2_raw.csv')
    calibrated_path = os.path.join(args.output_dir, f'{args.split}_predictions_maia2_calibrated.csv')
    calib_json_path = os.path.join(calib_dir, f'{args.split}_predictions_maia2_calibration.json')

    df_raw.to_csv(raw_path, index=False)
    df_calibrated.to_csv(calibrated_path, index=False)
    with open(calib_json_path, 'w') as f:
        json.dump({
            'c': probit_params[0], 'sigma': probit_params[1],
            'model_type': MODEL_TYPE,
            'calib_source_path': args.calib_path,
            'n_calib_positions': len(df_calib),
            'fitted_at': datetime.now().isoformat(timespec='seconds'),
        }, f, indent=2)

    print(f'Saved {len(df_raw):,} rows to {raw_path}')
    print(f'Saved {len(df_calibrated):,} rows to {calibrated_path}')
    print(f'Saved calibration record to {calib_json_path}')

####################
# ENTRY POINT
####################
if __name__ == '__main__':
    main()