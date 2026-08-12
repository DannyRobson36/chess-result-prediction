"""
maia2_reader.py
Maia2 win/loss-only evaluation of positions in an input CSV, output directly in
model-eval-ready form: game_id, fen, prob_win, prob_draw, prob_loss, predicted_class.

CLI (input/output paths required):
    --input-path    Path to the input CSV to evaluate. Must contain game_id, fen,
                    next_move, mover_elo, opponent_elo. 
    --output-path   Full path (including filename) to write the output CSV to.
    --device        'cpu' or 'gpu'. Default: 'cpu'.

Run:
    !python maia2_reader.py --input-path /path/to/val_pos.csv --output-path /path/to/val_predictions_maia2.csv
    !python maia2_reader.py --input-path /path/to/val_pos.csv --output-path /path/to/val_predictions_maia2.csv --device gpu

Latest changes: 12/08/26:
- Initial commit
"""

import importlib.util
import subprocess
import sys

import argparse
import os

import numpy as np
import pandas as pd
import chess

####################
# CONSTANTS
####################

MODEL_TYPE = 'rapid'
BATCH_SIZE = 1024
NUM_WORKERS = os.cpu_count() or 1
REQUIRED_COLUMNS = ['game_id', 'fen', 'next_move', 'mover_elo', 'opponent_elo']

####################
# FUNCTIONS
####################

# (a) VALIDATION & SETUP

def _validate_input_columns(columns: pd.Index | list[str], input_path: str) -> None:
    """Exits if any REQUIRED_COLUMNS are missing from the input CSV's header."""
    missing = [c for c in REQUIRED_COLUMNS if c not in columns]
    if missing:
        sys.exit(f'{input_path} is missing required column(s): {missing}. '
                 f'Required: {REQUIRED_COLUMNS}')

def ensure_maia2_installed() -> None:
    """Installs maia2 (via pyzstd, then maia2 --no-deps) if not already present."""
    if importlib.util.find_spec('maia2') is not None:
        return
    print('maia2 not found, installing (pyzstd, then maia2 --no-deps)...')
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'pyzstd', '-q'])
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '--no-deps', 'maia2', '-q'])
    print('maia2 installed.')

# (b) LOADING

def load_positions(input_path: str) -> pd.DataFrame:
    """Loads every row of the input CSV, in order, keeping only REQUIRED_COLUMNS."""
    header = pd.read_csv(input_path, nrows=0).columns
    _validate_input_columns(header, input_path)
    return pd.read_csv(input_path, usecols=REQUIRED_COLUMNS)

# (c) MOVER PERSPECTIVE

def mover_is_white(fens: pd.Series) -> np.ndarray:
    """Returns a bool array: True where the side to move in fen is white."""
    return np.array([chess.Board(f).turn == chess.WHITE for f in fens])

def count_terminal(fens: pd.Series) -> int:
    """Returns how many fens are already game-over positions (checkmate/stalemate/etc)."""
    return sum(chess.Board(f).is_game_over() for f in fens)

# (d) PROBABILITY CONVERSION

def to_mover_win_prob(white_win_prob: np.ndarray, is_white: np.ndarray) -> np.ndarray:
    """Flips a white-perspective win probability to mover-perspective."""
    return np.where(is_white, white_win_prob, 1.0 - white_win_prob)

# (e) CLI

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parses the --input-path, --output-path, --device CLI arguments."""
    parser = argparse.ArgumentParser(
        description='Maia2 win/loss evaluation of positions in an input CSV, '
                     'output directly in model-eval-ready form.'
    )
    parser.add_argument('--input-path', required=True,
                         help='Path to the input CSV to evaluate. Must contain '
                              'game_id, fen, next_move, mover_elo, opponent_elo.')
    parser.add_argument('--output-path', required=True,
                         help='Full path (including filename) to write the output CSV to.')
    parser.add_argument('--device', default='cpu', choices=['cpu', 'gpu'],
                         help='Device to run Maia2 on. Default: cpu.')
    return parser.parse_args(argv)

# (f) MAIN

def main(argv: list[str] | None = None) -> None:
    """Runs the full pipeline: load positions, evaluate with Maia2, write predictions CSV."""
    args = parse_args(argv)
    input_path = args.input_path
    output_path = args.output_path

    if not os.path.exists(input_path):
        sys.exit(f'No input CSV found at {input_path}.')

    ensure_maia2_installed()
    from maia2 import model as maia2_model_module
    from maia2 import inference as maia2_inference

    print(f'Loading positions from: {input_path}')
    df_in = load_positions(input_path)
    print(f'Loaded {len(df_in):,} position(s).')

    n_terminal = count_terminal(df_in['fen'])
    print(f'{n_terminal} terminal position(s) (expected 0).')

    maia2_input = df_in.rename(columns={
        'fen': 'board',
        'next_move': 'move',
        'mover_elo': 'active_elo',
    })[['board', 'move', 'active_elo', 'opponent_elo']]

    print(f'Loading Maia2 ({MODEL_TYPE}) on device={args.device}...')
    maia2_model = maia2_model_module.from_pretrained(type=MODEL_TYPE, device=args.device)

    print(f'Analysing {len(maia2_input):,} positions with Maia2 '
          f'(batch_size={BATCH_SIZE}, num_workers={NUM_WORKERS})...')
    data, _acc = maia2_inference.inference_batch(
        maia2_input, maia2_model, verbose=1, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS,
    )

    assert len(data) == len(df_in), (
        f'Row count mismatch: Maia2 returned {len(data)} rows for {len(df_in)} input rows.'
    )
    assert (data['board'].to_numpy() == df_in['fen'].to_numpy()).all(), (
        'board/fen mismatch -- Maia2 inference did not preserve row order.'
    )

    white_win_prob = data['win_probs'].to_numpy(dtype='float64')
    is_white = mover_is_white(df_in['fen'])
    prob_win = to_mover_win_prob(white_win_prob, is_white)
    prob_loss = 1.0 - prob_win
    prob_draw = np.zeros(len(df_in), dtype='float64')
    predicted_class = np.where(prob_win >= 0.5, 'win', 'loss')

    df_out = pd.DataFrame({
        'game_id': df_in['game_id'].to_numpy(),
        'fen': df_in['fen'].to_numpy(),
        'prob_win': prob_win,
        'prob_draw': prob_draw,
        'prob_loss': prob_loss,
        'predicted_class': predicted_class,
    })

    assert len(df_out) == len(df_in), 'Output row count does not match input row count.'

    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    df_out.to_csv(output_path, index=False)
    print(f'Saved {len(df_out):,} rows to {output_path}')

####################
# ENTRY POINT
####################
if __name__ == '__main__':
    main()