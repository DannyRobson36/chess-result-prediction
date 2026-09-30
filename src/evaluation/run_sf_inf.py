"""
run_sf_inf.py
Stockfish evaluation of one split's positions, writing win-probability predictions via the
fixed Lichess win% curve, in compare_predictions-ready form.

Run:
    !python run_sf_inf.py --input-path /path/to/val_900k_res_bal.csv --split val --depth 7

CLI:
    --input-path   Path to the split's positions CSV to score. Must contain game_id, fen.
    --split        Which split --input-path is: val or test. Used to name output files.
    --output-dir   Folder to write predictions into. Default: PREDICTIONS_DIR (config.py).
    --depth        Single Stockfish search depth to evaluate at.

Latest changes: 01/09/26:
- Switched output filename convention
"""

import os
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from src.config import PREDICTIONS_DIR

import argparse
import csv
import multiprocessing as mp
import signal
import subprocess

import chess
import chess.engine
import numpy as np
import pandas as pd
from tqdm import tqdm

####################
# CONSTANTS
####################

# Sentinel mover-perspective eval for a forced mate.
MATE_SCORE_MOVER = 5_000

# Centipawn -> win-probability sigmoid constant (the standard Lichess win% curve).
WIN_PCT_CONST = 0.00368208

# Per-engine hash size in MB.
HASH_MB_PER_ENGINE = 16

# Timeout for one position at one depth before stepping the search depth down.
ATTEMPT_TIMEOUT_SECONDS = 120

ENGINE_PATH = '/usr/games/stockfish'

REQUIRED_SCORE_COLUMNS = ['game_id', 'fen']

NUM_WORKERS = os.cpu_count() or 1

####################
# FUNCTIONS
####################

# (a) VALIDATION / SETUP

def _validate_columns(fieldnames: list[str] | None, required: list[str], path: str) -> None:
    """Exits with a clear message if any required column is missing from fieldnames."""
    missing = [c for c in required if c not in (fieldnames or [])]
    if missing:
        sys.exit(f'{path} is missing required column(s): {missing}. Required: {required}.')

def ensure_stockfish_installed(engine_path: str) -> None:
    """Installs stockfish via apt-get if engine_path doesn't already exist, exiting if that fails."""
    if os.path.exists(engine_path):
        return
    print('Stockfish not found -- installing via apt-get...')
    install = subprocess.run(['apt-get', 'install', '-y', 'stockfish'], capture_output=True, text=True)
    if install.returncode != 0 or not os.path.exists(engine_path):
        sys.exit(f'Stockfish install failed or binary not found at {engine_path}.\n{install.stderr}')
    print('Stockfish installed successfully.')

# (b) POSITION LOADING

def load_positions(input_path: str, required: list[str]) -> list[dict]:
    """Loads every row of input_path, keeping only the required columns."""
    with open(input_path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        _validate_columns(reader.fieldnames, required, input_path)
        rows = [{col: row[col] for col in required} for row in reader]
    return rows

# (c) ENGINE POOL / EVALUATION

def _start_engine() -> chess.engine.SimpleEngine:
    """Starts one stockfish engine process for this worker."""
    engine = chess.engine.SimpleEngine.popen_uci(ENGINE_PATH)
    engine.configure({'Threads': 1, 'Hash': HASH_MB_PER_ENGINE})
    return engine

def _init_worker() -> None:
    """Starts this worker's persistent engine, stored in the module-level _engine global."""
    global _engine
    _engine = _start_engine()

def _restart_engine() -> None:
    """Closes and restarts this worker's engine after a timeout or error."""
    global _engine
    try:
        _engine.close()
    except Exception:
        try:
            _engine.transport.kill()
        except Exception:
            pass
    _engine = _start_engine()

def _alarm_handler(signum, frame) -> None:
    """SIGALRM handler that converts a timeout into a _TaskTimeout."""
    raise _TaskTimeout()

def _analyse_clean(board: chess.Board, depth: int) -> dict:
    """Clears this worker's hash table, then analyses board at depth."""
    _engine.configure({'Clear Hash': None})
    return _engine.analyse(board, chess.engine.Limit(depth=depth))

def _terminal_eval(board: chess.Board) -> float:
    """Mover-perspective eval for a terminal position: -MATE_SCORE_MOVER for checkmate, 0 otherwise."""
    if board.is_checkmate():
        return -MATE_SCORE_MOVER
    return 0

def _process_one_inner(fen: str, depth: int) -> tuple[float, bool]:
    """Evaluates one position at one depth, skipping the engine for terminal positions.
    Returns (eval_val, is_terminal)."""
    board = chess.Board(fen)
    mover_is_white = board.turn

    if board.is_game_over():
        return _terminal_eval(board), True

    info = _analyse_clean(board, depth)
    white_score = info['score'].white()

    if white_score.is_mate():
        mate_for_white = white_score.mate() > 0
        mate_for_mover = (mate_for_white == mover_is_white)
        eval_val = MATE_SCORE_MOVER if mate_for_mover else -MATE_SCORE_MOVER
    else:
        raw_cp = white_score.score()
        eval_val = raw_cp if mover_is_white else -raw_cp

    return eval_val, False

def _process_one(args: tuple[str, int]) -> dict:
    """Evaluates one position, stepping search depth down by 1 (no same-depth retry) on each
    timeout/error, down to depth 1. Raises if depth 1 also fails."""
    fen, target_depth = args
    for depth in range(target_depth, 0, -1):
        signal.signal(signal.SIGALRM, _alarm_handler)
        signal.alarm(ATTEMPT_TIMEOUT_SECONDS)
        try:
            eval_val, is_terminal = _process_one_inner(fen, depth)
            signal.alarm(0)
            fallback = (not is_terminal) and (depth != target_depth)
            return {'eval': eval_val, 'is_terminal': is_terminal, 'fallback': fallback}
        except _TaskTimeout:
            signal.alarm(0)
            _restart_engine()
        except Exception:
            signal.alarm(0)
            _restart_engine()
    raise RuntimeError(
        f'Stockfish failed to evaluate this position even at depth 1 '
        f'(every depth from {target_depth} down to 1 timed out or errored): {fen}'
    )

def evaluate_positions(rows: list[dict], depth: int, desc: str) -> tuple[np.ndarray, int, int]:
    """Evaluates every row's fen at depth via a multiprocessing pool of persistent engines.
    Returns (evals, n_terminal, n_fallback)."""
    tasks = [(row['fen'], depth) for row in rows]
    results = []
    with mp.Pool(processes=NUM_WORKERS, initializer=_init_worker) as pool:
        for res in tqdm(pool.imap(_process_one, tasks), total=len(tasks), desc=desc, unit='pos'):
            results.append(res)

    assert len(results) == len(rows), (
        f'Internal error: got {len(results)} results for {len(rows)} input rows.'
    )

    evals = np.array([r['eval'] for r in results], dtype='float64')
    n_terminal = sum(r['is_terminal'] for r in results)
    n_fallback = sum(r['fallback'] for r in results)
    return evals, n_terminal, n_fallback

# (d) EVAL -> PREDICTION COLUMNS

def eval_to_win_prob(cp: np.ndarray) -> np.ndarray:
    """Maps mover-perspective centipawns to a win probability via the fixed Lichess sigmoid."""
    return 1.0 / (1.0 + np.exp(-WIN_PCT_CONST * np.asarray(cp, dtype='float64')))

def raw_prediction_cols(evals: np.ndarray) -> dict:
    """Returns prob_win/prob_draw/prob_loss/predicted_class from raw cp evals via the fixed
    Lichess win% curve, unfitted."""
    prob_win = eval_to_win_prob(evals)
    prob_loss = 1.0 - prob_win
    prob_draw = np.zeros_like(prob_win)
    predicted_class = np.where(prob_win >= 0.5, 'win', 'loss')
    return {'prob_win': prob_win, 'prob_draw': prob_draw, 'prob_loss': prob_loss, 'predicted_class': predicted_class}

# (e) CLI

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parses --input-path, --split, --output-dir, --depth CLI arguments."""
    parser = argparse.ArgumentParser(
        description='Stockfish evaluation of one split, writing win-probability predictions '
                     'via the fixed Lichess win% curve.'
    )
    parser.add_argument('--input-path', required=True,
                         help='Path to the split\'s positions CSV to score. Must contain game_id, fen.')
    parser.add_argument('--split', required=True, choices=['val', 'test'],
                         help='Which split --input-path is, used to name the output file.')
    parser.add_argument('--output-dir', default=PREDICTIONS_DIR, help=f'Default: {PREDICTIONS_DIR}')
    parser.add_argument('--depth', type=int, required=True,
                         help='Single Stockfish search depth to evaluate at.')
    args = parser.parse_args(argv)
    if args.depth <= 0:
        parser.error('--depth must be > 0.')
    return args

def main(argv: list[str] | None = None) -> None:
    """Evaluates --input-path's positions at --depth and writes raw Lichess-curve win-probability
    predictions to a single CSV."""
    args = parse_args(argv)

    if not os.path.exists(args.input_path):
        sys.exit(f'No input CSV found at {args.input_path}.')

    ensure_stockfish_installed(ENGINE_PATH)
    print(f'Detected {NUM_WORKERS} CPU core(s) -- using all of them as workers.')

    score_rows = load_positions(args.input_path, REQUIRED_SCORE_COLUMNS)
    print(f'Evaluating {len(score_rows):,} position(s) at depth {args.depth}...')
    score_evals, score_n_terminal, score_n_fallback = evaluate_positions(score_rows, args.depth, 'scoring')
    print(f'{score_n_terminal} terminal, {score_n_fallback} depth-fallback position(s).')

    eval_col = f'eval_{args.depth}'
    base = {
        'game_id': [r['game_id'] for r in score_rows],
        'fen': [r['fen'] for r in score_rows],
        eval_col: score_evals,
    }
    output_cols = ['game_id', 'fen', eval_col, 'prob_win', 'prob_draw', 'prob_loss', 'predicted_class']

    df_raw = pd.DataFrame({**base, **raw_prediction_cols(score_evals)})
    assert list(df_raw.columns) == output_cols, 'output columns do not match output_cols.'

    os.makedirs(args.output_dir, exist_ok=True)

    name_tag = f'stockfish_d{args.depth}'
    output_path = os.path.join(args.output_dir, f'{args.split}_pred_{name_tag}.csv')

    df_raw.to_csv(output_path, index=False)
    print(f'Saved {len(df_raw):,} rows to {output_path}')

####################
# CLASSES
####################

class _TaskTimeout(Exception):
    """Raised when a single position's Stockfish analysis exceeds ATTEMPT_TIMEOUT_SECONDS."""

####################
# ENTRY POINT
####################
if __name__ == '__main__':
    main()